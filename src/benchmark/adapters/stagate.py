from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import torch

from .base import CanonicalAdapterResult, CanonicalSpatialAdapter
from .common import (
    ROOT, canonical_mclust_R, full_resolution_visium_coordinates, git_commit, prepare_hvg_adata,
    refine_knn_majority, require_spatial, source_tree_sha256,
)


STAGATE_REPOSITORY = ROOT / "baselines/STAGATE_pyG"


class OfficialSTAGATEAdapter(CanonicalSpatialAdapter):
    """Official STAGATE-pyG training and true-K mclust clustering."""

    name = "STAGATE_official"

    def __init__(
        self,
        *,
        protocol: str = "canonical",
        track: str = "canonical",
        sample_key: str = "sample_id",
        n_top_genes: int = 3000,
        radius_cutoff: float = 50.0,
        epochs: int = 500,
    ) -> None:
        if protocol not in {"canonical", "controlled"}:
            raise ValueError(f"Unknown STAGATE protocol: {protocol}")
        self.protocol = protocol
        self.track = track
        self.sample_key = sample_key
        self.n_top_genes = n_top_genes
        self.radius_cutoff = radius_cutoff
        self.epochs = epochs

    def fit_transform(
        self,
        adata: ad.AnnData,
        *,
        n_clusters: int | None = None,
        seed: int = 0,
        device: torch.device | None = None,
        embedding_only: bool = False,
        checkpoint_dir: Path | str | None = None,
    ) -> CanonicalAdapterResult:
        if not embedding_only and n_clusters is None:
            raise ValueError("n_clusters must be provided when embedding_only=False")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if checkpoint_dir is not None:
            import warnings
            warnings.warn(
                "STAGATE official train_STAGATE API does not expose the internal PyG model object; "
                "checkpoint saving is unsupported. Val/Test embeddings, result JSONs, configs, and "
                "environment hashes serve as official reproducible assets.",
                UserWarning,
                stacklevel=2,
            )

        if not STAGATE_REPOSITORY.is_dir():
            raise FileNotFoundError(f"Official STAGATE source missing: {STAGATE_REPOSITORY}")
        sys.path.insert(0, str(STAGATE_REPOSITORY))
        import STAGATE_pyG

        if self.protocol == "controlled":
            data = adata.copy()
        else:
            data = prepare_hvg_adata(adata, n_top_genes=self.n_top_genes)
        require_spatial(data)
        fullres_coordinates, coordinate_source = full_resolution_visium_coordinates(data)
        data.obsm["spatial"] = fullres_coordinates

        # Build sample-isolated spatial graph
        from .common import build_sample_isolated_spatial_graph
        import pandas as pd
        import hashlib

        adj = build_sample_isolated_spatial_graph(
            data,
            sample_key=self.sample_key,
            radius=self.radius_cutoff,
            allow_single_sample=True,
        )

        # Assert zero cross-sample leakage
        if self.sample_key in data.obs.columns:
            groups = data.obs[self.sample_key].to_numpy()
            rows_chk, cols_chk = adj.nonzero()
            cross_edges = int((groups[rows_chk] != groups[cols_chk]).sum())
            if cross_edges > 0:
                raise RuntimeError(f"FATAL: Detected {cross_edges} illegal cross-sample edges in STAGATE graph!")
        else:
            cross_edges = 0

        csr_chk = adj.tocsr()
        h = hashlib.sha256()
        h.update(csr_chk.data.tobytes())
        h.update(csr_chk.indices.tobytes())
        h.update(csr_chk.indptr.tobytes())
        csr_hash = h.hexdigest()

        # Populate STAGATE Spatial_Net
        rows, cols = adj.nonzero()
        mask = rows != cols
        df_net = pd.DataFrame({
            "Cell1": data.obs_names[rows[mask]],
            "Cell2": data.obs_names[cols[mask]],
            "Distance": np.ones(mask.sum(), dtype=np.float32),
        })
        data.uns["Spatial_Net"] = df_net
        trained = STAGATE_pyG.train_STAGATE(
            data,
            hidden_dims=[512, 30],
            n_epochs=self.epochs,
            lr=1e-4,
            random_seed=seed,
            save_loss=True,
            save_reconstrction=False,
            verbose=False,
            device=device,
        )
        embedding = np.asarray(trained.obsm["STAGATE"], dtype=np.float32)
        if embedding_only:
            labels = None
            refined = None
        elif self.protocol == "controlled":
            from .common import controlled_gmm_labels
            labels = controlled_gmm_labels(embedding, n_clusters=n_clusters, seed=seed)
            refined = refine_knn_majority(
                labels, fullres_coordinates, n_neighbors=50
            )
        else:
            labels = canonical_mclust_R(
                embedding,
                n_clusters=n_clusters,
                seed=seed,
                model_names="EEE",
            )
            refined = refine_knn_majority(
                labels, fullres_coordinates, n_neighbors=50
            )
        result = CanonicalAdapterResult(
            method=self.name,
            embedding=embedding,
            canonical_labels=labels,
            refined_labels=refined,
            metadata={
                "implementation": "official STAGATE_pyG",
                "repository": "https://github.com/zhanglabtools/STAGATE_pyG",
                "commit": git_commit(STAGATE_REPOSITORY),
                "source_tree_sha256": source_tree_sha256(STAGATE_REPOSITORY),
                "n_top_genes": self.n_top_genes,
                "benchmark_protocol": self.protocol,
                "track": self.track,
                "graph_builder": "stagate_sample_isolated_radius",
                "sample_key": self.sample_key,
                "csr_hash": csr_hash,
                "cross_sample_edges": cross_edges,
                "graph_mode": "Radius",
                "radius_cutoff": self.radius_cutoff,
                "coordinate_source": coordinate_source,
                "epochs": self.epochs,
                "learning_rate": 1e-4,
                "hidden_dims": [512, 30],
                "canonical_clusterer": "mclust_EEE_trueK",
                "canonical_refinement": "SpaGCN_style_50NN_majority",
                "controlled_clusterer": "GaussianMixture_full_trueK",
            },
        )
        result.validate(adata.n_obs)
        return result
