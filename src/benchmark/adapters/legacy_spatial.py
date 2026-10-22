"""Bounded adapters that call the unmodified SEDR, GraphST and SpaGCN authors' code.

These methods fit on the supplied unlabeled spatial cohort. They do not offer
FreqNiche's frozen cross-donor encoder; results must be marked transductive.
"""

from __future__ import annotations

import importlib.util
import importlib
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[3]
SOURCES = {
    "sedr": ROOT / "baselines/SEDR",
    "graphst": ROOT / "baselines/GraphST",
    "spagcn": ROOT / "baselines/SpaGCN/SpaGCN_package",
}


def _source(method: str) -> None:
    source = SOURCES[method]
    if not source.is_dir():
        raise FileNotFoundError(f"Official {method} source is missing: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _unlabeled_copy(data: ad.AnnData) -> ad.AnnData:
    result = data.copy()
    result.obs = result.obs[[name for name in ("donor_id", "sample_id") if name in result.obs]].copy()
    for key in list(result.obsm):
        if key != "spatial":
            del result.obsm[key]
    if "spatial" not in result.obsm or np.asarray(result.obsm["spatial"]).shape != (result.n_obs, 2):
        raise ValueError("Official spatial baseline input needs a two-column obsm['spatial']")
    return result


def _balanced_edge_mask(adj_label: torch.Tensor, seed: int) -> torch.Tensor:
    """Supply SEDR's documented 1:1 positive/negative edge mask.

    The vendored mask_generator uses an invalid float dtype in torch.randperm
    under current PyTorch. Passing a precomputed mask keeps the author's model
    and loss untouched and avoids that broken helper.
    """
    positive = adj_label.coalesce().indices().cpu().numpy()
    n = adj_label.shape[0]
    positive_pairs = set(zip(positive[0].tolist(), positive[1].tolist()))
    if len(positive_pairs) > n * n - len(positive_pairs):
        raise ValueError("SEDR graph does not have enough distinct negative pairs")
    rng = np.random.default_rng(seed)
    needed = len(positive_pairs)
    negative: set[tuple[int, int]] = set()
    while len(negative) < needed:
        row, col = rng.integers(0, n, size=2).tolist()
        pair = (row, col)
        if pair not in positive_pairs:
            negative.add(pair)
    neg = np.asarray(sorted(negative), dtype=np.int64).T
    indices = np.concatenate([positive, neg], axis=1)
    values = np.concatenate([np.ones(positive.shape[1]), np.zeros(neg.shape[1])]).astype(np.float32)
    return torch.sparse_coo_tensor(
        torch.from_numpy(indices), torch.from_numpy(values), adj_label.shape,
    ).coalesce()


def _sample_isolated_knn(data: ad.AnnData, n_neighbors: int) -> sp.csr_matrix:
    coords = np.asarray(data.obsm["spatial"], dtype=np.float32)
    samples = (
        data.obs["sample_id"].astype(str).to_numpy()
        if "sample_id" in data.obs else np.repeat("one_sample", data.n_obs)
    )
    rows, cols = [], []
    for sample in np.unique(samples):
        indices = np.flatnonzero(samples == sample)
        if len(indices) < 2:
            raise ValueError(f"Spatial sample {sample!r} has fewer than two cells")
        k = min(n_neighbors + 1, len(indices))
        neighbors = NearestNeighbors(n_neighbors=k).fit(coords[indices]).kneighbors(
            coords[indices], return_distance=False
        )
        for local_row, local_neighbors in enumerate(neighbors):
            valid = local_neighbors[local_neighbors != local_row][:n_neighbors]
            rows.extend([int(indices[local_row])] * len(valid))
            cols.extend(indices[valid].tolist())
    directed = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(data.n_obs, data.n_obs)
    )
    directed.setdiag(0)
    directed.eliminate_zeros()
    return directed


def fit_official_embedding(
    method: str,
    data: ad.AnnData,
    *,
    seed: int,
    device: str,
    smoke: bool = False,
    max_cells: int = 10000,
    artifact_dir: Path | None = None,
    resume: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit once on a bounded unlabeled cohort and return its native embedding."""
    if method not in SOURCES:
        raise ValueError(f"Unknown additional baseline: {method}")
    if artifact_dir is not None:
        from .spatial_artifacts import artifact_code_binding, code_binding_matches, data_fingerprint, verify_artifact
        code_binding = artifact_code_binding(method)
        artifact_dir = Path(artifact_dir)
        if artifact_dir.exists():
            if not resume:
                raise FileExistsError(f"Refusing to overwrite fitted assets: {artifact_dir}")
            saved = verify_artifact(
                artifact_dir, expected_method=method, expected_seed=seed,
                expected_fingerprint=data_fingerprint(data),
            )
            if saved["smoke"] != smoke:
                raise RuntimeError("Smoke and formal assets cannot be mixed")
            if not code_binding_matches(saved.get("code_binding"), code_binding):
                raise RuntimeError("Source code differs from saved fitted assets; use a new run directory")
            return np.load(artifact_dir / "embedding.npy", allow_pickle=False), saved["protocol"]
    if data.n_obs < 32 or data.n_obs > max_cells:
        raise ValueError(f"{method} accepts 32..{max_cells} cells in this bounded protocol; got {data.n_obs}")
    estimated_dense_gib = data.n_obs * data.n_obs * 4 / (1024**3)
    if estimated_dense_gib > 0.4:
        raise MemoryError(f"One dense N×N matrix would use {estimated_dense_gib:.2f} GiB; refuse full-cohort run")
    _source(method)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    work = _unlabeled_copy(data)
    if sp.issparse(work.X):
        work.X = work.X.tocsr()
    run_device = device if device == "cpu" or torch.cuda.is_available() else "cpu"

    if method == "sedr":
        import SEDR
        from SEDR.graph_func import preprocess_graph

        count = min(200, work.n_obs - 1, work.n_vars - 1)
        if count < 2:
            raise ValueError("SEDR requires at least three genes")
        sc.pp.normalize_total(work, target_sum=1e6)
        sc.pp.log1p(work)
        sc.pp.scale(work)
        matrix = work.X.toarray() if sp.issparse(work.X) else np.asarray(work.X)
        fitted_pca = PCA(n_components=count, random_state=seed)
        x_pca = fitted_pca.fit_transform(matrix).astype(np.float32)
        directed = _sample_isolated_knn(work, n_neighbors=12)
        spatial_graph = ((directed + directed.T) > 0).astype(np.float32).tocsr()
        adj_label = (spatial_graph + sp.eye(work.n_obs, format="csr")).tocoo()
        positive = torch.sparse_coo_tensor(
            np.stack([adj_label.row, adj_label.col]), adj_label.data.astype(np.float32),
            adj_label.shape,
        ).coalesce()
        graph_dict = {
            "adj_norm": preprocess_graph(spatial_graph).coalesce(),
            "adj_label": positive,
            "norm_value": work.n_obs**2 / float((work.n_obs**2 - adj_label.sum()) * 2),
        }
        graph_dict["mask"] = _balanced_edge_mask(graph_dict["adj_label"], seed)
        trainer = SEDR.Sedr(x_pca, graph_dict, mode="clustering", device=run_device)
        if smoke:
            trainer.train_without_dec(epochs=2, lr=0.01, decay=0.01)
        else:
            trainer.train_with_dec(epochs=200, N=1)
        recorded_mask = {}
        original_mask = trainer.model.encoding_mask_noise
        def capture_mask(adj, x, rate):
            masked = original_mask(adj, x, rate)
            recorded_mask["nodes"] = masked[2][0].detach().cpu().clone()
            return masked
        context = patch.object(trainer.model, "encoding_mask_noise", capture_mask) if artifact_dir is not None else nullcontext()
        with context:
            embedding = trainer.process()[0]
        protocol = {"official_core": "SEDR.Sedr", "graph_neighbors": 12,
                    "pca_components": count, "training": "2-epoch smoke" if smoke else "200+200 epoch DEC",
                    "negative_edges": "balanced precomputed mask; official mask helper incompatible with current PyTorch",
                    "graph_scope": "sample_id_isolated"}

    elif method == "graphst":
        from GraphST.GraphST import GraphST
        from GraphST.preprocess import add_contrastive_label, get_feature

        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)
        work.var["highly_variable"] = True
        directed = _sample_isolated_knn(work, n_neighbors=3)
        work.obsm["graph_neigh"] = directed.toarray().astype(np.float32)
        work.obsm["adj"] = ((directed + directed.T) > 0).toarray().astype(np.float32)
        add_contrastive_label(work)
        get_feature(work)
        trainer = GraphST(work, device=torch.device(run_device), random_seed=seed,
                          datatype="Stereo", epochs=2 if smoke else 600)
        trained = trainer.train()
        # GraphST's own clustering helper applies PCA(20) to its returned
        # reconstruction before mclust/Leiden; preserve that readout feature.
        fitted_pca = PCA(n_components=min(20, work.n_obs - 1, work.n_vars - 1), random_state=seed)
        embedding = fitted_pca.fit_transform(trained.obsm["emb"])
        protocol = {"official_core": "GraphST.GraphST", "datatype": "Stereo/sparse graph",
                    "epochs": 2 if smoke else 600, "graph_neighbors": 3,
                    "official_preclustering_pca": embedding.shape[1],
                    "graph_scope": "sample_id_isolated"}

    else:
        import SpaGCN as spg

        if not smoke and importlib.util.find_spec("louvain") is None:
            raise ModuleNotFoundError(
                "Official SpaGCN Louvain initialization needs the 'louvain' package; "
                "install it in an isolated baseline environment, not the FreqNiche environment"
            )
        spg.prefilter_genes(work, min_cells=3)
        if work.n_vars < 3:
            raise ValueError("SpaGCN preprocessing retained fewer than three genes")
        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)
        if sp.issparse(work.X):
            work.X = work.X.toarray().astype(np.float32)
        coordinates = np.asarray(work.obsm["spatial"], dtype=np.float32)
        adj = spg.calculate_adj_matrix(
            x=coordinates[:, 0].tolist(), y=coordinates[:, 1].tolist(), histology=False,
        )
        if "sample_id" in work.obs:
            sample_ids = work.obs["sample_id"].astype(str).to_numpy()
            adj[sample_ids[:, None] != sample_ids[None, :]] = np.inf
        l_value = spg.search_l(0.5, adj, start=0.01, end=1000, tol=0.01, max_run=100)
        model = spg.SpaGCN()
        model.set_l(l_value)
        # Capture the actual PCA fitted inside the official class without refitting
        # it or consuming additional RNG draws.
        pca_instances = []
        owner = importlib.import_module(type(model).__module__)
        original_pca = owner.PCA
        def capture_pca(*args, **kwargs):
            value = original_pca(*args, **kwargs)
            pca_instances.append(value)
            return value
        context = patch.object(owner, "PCA", capture_pca) if artifact_dir is not None else nullcontext()
        with context:
            model.train(
                work, adj, num_pcs=min(50, work.n_vars - 1), init_spa=True,
                init="kmeans" if smoke else "louvain", n_clusters=10 if smoke else None,
                res=0.4, tol=5e-3, lr=0.05, max_epochs=2 if smoke else 200,
            )
        if artifact_dir is not None:
            fitted_pca = pca_instances[0]
        embedding = model.model.predict(model.embed, model.adj_exp)[0].detach().cpu().numpy()
        protocol = {"official_core": "SpaGCN.SpaGCN", "histology": False,
                    "init": "kmeans smoke only" if smoke else "official Louvain",
                    "epochs": 2 if smoke else 200, "l": float(l_value),
                    "graph_scope": "sample_id_isolated"}

    result = np.asarray(embedding, dtype=np.float32)
    if result.shape[0] != data.n_obs or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid {method} embedding: {result.shape}")
    if artifact_dir is not None:
        from .spatial_artifacts import cpu_tensor, save_artifact
        graph = None
        native = result
        if method == "sedr":
            learned_model = trainer.model
            init_kwargs = {
                "input_dim": int(trainer.input_dim),
                "feat_hidden1": learned_model.feat_hidden1,
                "feat_hidden2": learned_model.feat_hidden2,
                "gcn_hidden1": learned_model.gcn_hidden1,
                "gcn_hidden2": learned_model.gcn_hidden2,
                "p_drop": learned_model.p_drop, "alpha": learned_model.alpha,
                "dec_clsuter_n": learned_model.dec_cluster_n,
            }
            inputs = {"X": cpu_tensor(trainer.X), "adj_norm": cpu_tensor(trainer.adj_norm),
                      "mask_nodes": recorded_mask["nodes"]}
        elif method == "graphst":
            learned_model = trainer.model
            init_kwargs = {"in_features": trainer.dim_input, "out_features": trainer.dim_output}
            inputs = {
                "features": cpu_tensor(trainer.features), "features_a": cpu_tensor(trainer.features_a),
                "adj": cpu_tensor(trainer.adj), "graph_neigh": cpu_tensor(trainer.graph_neigh, sparse=True),
            }
            native = np.asarray(trained.obsm["emb"], dtype=np.float32)
        else:
            learned_model = model.model
            init_kwargs = {"nfeat": model.embed.shape[1], "nhid": learned_model.nhid,
                           "alpha": learned_model.alpha}
            inputs = {"embed": cpu_tensor(model.embed)}
            graph = model.adj_exp
        save_artifact(
            artifact_dir, method=method, seed=seed, source=data, processed=work,
            model=learned_model, init_kwargs=init_kwargs, model_inputs=inputs,
            embedding=result, native_embedding=native, pca=fitted_pca,
            protocol=protocol, code_binding=code_binding, graph=graph, smoke=smoke,
        )
    return result, protocol
