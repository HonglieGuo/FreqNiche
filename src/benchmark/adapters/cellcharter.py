from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from .base import CanonicalAdapterResult, CanonicalSpatialAdapter
from .common import (
    ROOT,
    build_sample_isolated_spatial_graph,
    git_commit,
    require_spatial,
    source_tree_sha256,
)

CELLCHARTER_REPOSITORY = ROOT / "baselines/cellcharter"
CELLCHARTER_SOURCE = CELLCHARTER_REPOSITORY / "src"


def _import_cellcharter():
    if not CELLCHARTER_SOURCE.is_dir():
        raise FileNotFoundError(f"CellCharter source missing: {CELLCHARTER_SOURCE}")
    sys.path.insert(0, str(CELLCHARTER_SOURCE))

    import importlib.metadata
    _orig_version = importlib.metadata.version

    def _mock_version(pkg_name: str) -> str:
        if pkg_name in ("cellcharter", "nichepca"):
            return "0.3.7"
        return _orig_version(pkg_name)

    try:
        importlib.metadata.version = _mock_version
        import cellcharter as cc
    finally:
        importlib.metadata.version = _orig_version
    return cc


def _compute_csr_hash(adj: sp.spmatrix) -> str:
    csr = adj.tocsr()
    h = hashlib.sha256()
    h.update(csr.data.tobytes())
    h.update(csr.indices.tobytes())
    h.update(csr.indptr.tobytes())
    return h.hexdigest()


class CanonicalCellCharterAdapter(CanonicalSpatialAdapter):
    """Canonical CellCharter Pipeline: scVI raw counts -> Delaunay/sample isolation -> aggregation -> cc.tl.Cluster."""

    name = "CellCharter_canonical"

    def __init__(
        self,
        *,
        track: str = "canonical",
        sample_key: str = "sample_id",
        n_latent: int = 20,
        n_layers: int = 3,
        n_neighbors: int = 6,
        radius: float | None = 50.0,
        distance_percentile: float = 99.0,
        max_epochs: int = 200,
        batch_size: int = 128,
    ) -> None:
        self.track = track
        self.sample_key = sample_key
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.distance_percentile = distance_percentile
        self.max_epochs = max_epochs
        self.batch_size = batch_size

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
        require_spatial(adata)
        cc = _import_cellcharter()
        import squidpy as sq
        import scvi

        # Set random seeds for reproducibility
        scvi.settings.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        n_spots = adata.n_obs
        ad_work = adata.copy()

        # Step 0: Raw counts non-destructive validation without densification
        counts_mat = ad_work.layers["counts"] if "counts" in ad_work.layers else ad_work.X
        values = counts_mat.data if sp.issparse(counts_mat) else np.asarray(counts_mat)

        if values.size:
            if np.min(values) < 0:
                raise ValueError("Expression matrix contains negative values; raw counts must be non-negative!")
            if not np.all(np.isfinite(values)):
                raise ValueError("Expression matrix contains non-finite values (NaN/Inf)!")
            if not np.allclose(values, np.rint(values)):
                raise ValueError("Expression matrix contains non-integer count values!")

        if "counts" not in ad_work.layers:
            ad_work.layers["counts"] = ad_work.X.copy()

        # Step 1: scVI training using raw counts and sample_key batch
        scvi.model.SCVI.setup_anndata(
            ad_work,
            layer="counts",
            batch_key=self.sample_key,
        )
        model = scvi.model.SCVI(
            ad_work,
            n_latent=self.n_latent,
            gene_likelihood="nb",
        )

        # Device mapping to PyTorch Lightning trainer params
        if device is not None:
            accelerator = "gpu" if device.type == "cuda" else "cpu"
            devices = [device.index] if (device.index is not None and accelerator == "gpu") else 1
        else:
            accelerator = "gpu" if torch.cuda.is_available() else "cpu"
            devices = 1 if accelerator == "gpu" else "auto"

        model.train(
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            early_stopping=True,
            accelerator=accelerator,
            devices=devices,
            plan_kwargs={"lr": 1e-3},
        )

        # Internal Fail-Safe Model Checkpoint Persistence
        if checkpoint_dir is not None:
            save_dir = Path(checkpoint_dir) / f"scvi_model_seed_{seed}"
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(save_dir), overwrite=True)
            print(f"[OK] Persisted scVI model directly inside adapter to: {save_dir}")

        ad_work.obsm["X_scVI"] = model.get_latent_representation()

        # Step 2: Spatial graph construction with physical slice isolation
        if self.track == "canonical":
            sq.gr.spatial_neighbors(
                ad_work,
                library_key=self.sample_key,
                coord_type="generic",
                delaunay=True,
            )
            cc.gr.remove_long_links(ad_work, distance_percentile=self.distance_percentile)
            graph_builder = "cellcharter_delaunay_long_links"
        else:
            build_sample_isolated_spatial_graph(
                ad_work,
                sample_key=self.sample_key,
                radius=self.radius,
                n_neighbors=self.n_neighbors if self.radius is None else None,
            )
            graph_builder = "sample_isolated_radius_50"

        # Assert zero cross-sample leakage
        adj = ad_work.obsp["spatial_connectivities"]
        groups = ad_work.obs[self.sample_key].to_numpy()
        rows, cols = adj.nonzero()
        cross_edges = int((groups[rows] != groups[cols]).sum())
        if cross_edges > 0:
            raise RuntimeError(f"FATAL: Detected {cross_edges} illegal cross-sample edges in CellCharter graph!")

        csr_hash = _compute_csr_hash(adj)

        # Step 3: Neighborhood aggregation
        cc.gr.aggregate_neighbors(
            ad_work,
            use_rep="X_scVI",
            n_layers=self.n_layers,
            out_key="X_cellcharter",
        )
        emb = np.asarray(ad_work.obsm["X_cellcharter"], dtype=np.float32)

        # Step 4: Spatial domain clustering
        canonical_labels = None
        readout = "embedding_only"
        if not embedding_only and n_clusters is not None:
            if self.track == "canonical":
                cluster_model = cc.tl.Cluster(n_clusters=n_clusters, random_state=seed)
                cluster_model.fit(ad_work, use_rep="X_cellcharter")
                preds = cluster_model.predict(ad_work, use_rep="X_cellcharter")
                canonical_labels = np.asarray(preds.to_numpy() if hasattr(preds, "to_numpy") else preds, dtype=int)
                readout = "cellcharter_cluster"
            else:
                gmm = GaussianMixture(n_components=n_clusters, random_state=seed)
                canonical_labels = gmm.fit_predict(emb)
                readout = "controlled_sklearn_gmm"

        result = CanonicalAdapterResult(
            method=f"cellcharter_{self.track}",
            embedding=emb,
            canonical_labels=canonical_labels,
            metadata={
                "adapter": self.name,
                "track": self.track,
                "graph_builder": graph_builder,
                "sample_key": self.sample_key,
                "radius_or_k": self.radius if self.track == "controlled" else f"delaunay_p{self.distance_percentile}",
                "csr_hash": csr_hash,
                "cross_sample_edges": cross_edges,
                "canonical_scvi_pipeline": True,
                "clustering_readout": readout,
                "accelerator": accelerator,
                "n_latent": self.n_latent,
                "n_layers": self.n_layers,
                "seed": seed,
                "upstream_commit": git_commit(CELLCHARTER_REPOSITORY),
                "source_tree_sha256": source_tree_sha256(CELLCHARTER_SOURCE),
            },
        )
        result.validate(n_spots)
        return result


class ControlledCellCharterAdapter(CanonicalSpatialAdapter):
    """Controlled CellCharter aggregation on PCA features (explicitly non-scVI baseline)."""

    name = "CellCharter_controlled_pca"

    def __init__(
        self,
        *,
        protocol: str = "controlled",
        sample_key: str = "sample_id",
        pca_dim: int = 50,
        n_layers: int = 3,
        n_neighbors: int = 6,
        radius: float | None = 50.0,
    ) -> None:
        self.protocol = protocol
        self.sample_key = sample_key
        self.pca_dim = pca_dim
        self.n_layers = n_layers
        self.n_neighbors = n_neighbors
        self.radius = radius

    def fit_transform(
        self,
        adata: ad.AnnData,
        *,
        n_clusters: int | None = None,
        seed: int = 0,
        device: torch.device | None = None,
        embedding_only: bool = False,
    ) -> CanonicalAdapterResult:
        del device
        require_spatial(adata)
        cc = _import_cellcharter()

        n_spots = adata.shape[0]
        ad_input = adata.copy()

        # Controlled PCA
        pca = PCA(n_components=self.pca_dim, random_state=seed)
        expr_pca = pca.fit_transform(ad_input.X.toarray() if sp.issparse(ad_input.X) else ad_input.X)
        ad_input.obsm["X_pca"] = expr_pca

        # Sample-isolated graph
        build_sample_isolated_spatial_graph(
            ad_input,
            sample_key=self.sample_key,
            radius=self.radius,
            n_neighbors=self.n_neighbors if self.radius is None else None,
        )
        adj = ad_input.obsp["spatial_connectivities"]
        groups = ad_input.obs[self.sample_key].to_numpy()
        rows, cols = adj.nonzero()
        cross_edges = int((groups[rows] != groups[cols]).sum())
        assert cross_edges == 0, f"Detected {cross_edges} illegal cross-sample edges!"
        csr_hash = _compute_csr_hash(adj)

        # Aggregate features
        cc.gr.aggregate_neighbors(ad_input, use_rep="X_pca", n_layers=self.n_layers, out_key="X_cellcharter")
        emb = np.asarray(ad_input.obsm["X_cellcharter"], dtype=np.float32)

        canonical_labels = None
        if not embedding_only and n_clusters is not None:
            gmm = GaussianMixture(n_components=n_clusters, random_state=seed)
            canonical_labels = gmm.fit_predict(emb)

        result = CanonicalAdapterResult(
            method="cellcharter_pca_aggregate",
            embedding=emb,
            canonical_labels=canonical_labels,
            metadata={
                "adapter": self.name,
                "track": "controlled",
                "graph_builder": "sample_isolated_radius_50",
                "sample_key": self.sample_key,
                "radius_or_k": self.radius if self.radius is not None else self.n_neighbors,
                "csr_hash": csr_hash,
                "cross_sample_edges": cross_edges,
                "pca_dim": self.pca_dim,
                "n_layers": self.n_layers,
                "canonical_scvi_pipeline": False,
                "seed": seed,
                "upstream_commit": git_commit(CELLCHARTER_REPOSITORY),
                "source_tree_sha256": source_tree_sha256(CELLCHARTER_SOURCE),
            },
        )
        result.validate(n_spots)
        return result


# Backward-compatible alias for existing runners
OfficialCellCharterAdapter = ControlledCellCharterAdapter


class PrecomputedSCVICellCharterAdapter(CanonicalSpatialAdapter):
    """CellCharter neighborhood aggregation using precomputed scVI latent (Smoke/Exploratory only)."""

    name = "CellCharter_precomputed_scvi"

    def __init__(
        self,
        *,
        scvi_rep_key: str = "X_source_scvi",
        n_layers: int = 3,
        n_neighbors: int = 6,
        sample_key: str = "sample_id",
    ) -> None:
        self.scvi_rep_key = scvi_rep_key
        self.n_layers = n_layers
        self.n_neighbors = n_neighbors
        self.sample_key = sample_key

    def fit_transform(
        self,
        adata: ad.AnnData,
        *,
        n_clusters: int | None = None,
        seed: int = 0,
        device: torch.device | None = None,
        embedding_only: bool = True,
    ) -> CanonicalAdapterResult:
        del device
        require_spatial(adata)
        if self.scvi_rep_key not in adata.obsm:
            raise KeyError(f"adata.obsm misses precomputed latent '{self.scvi_rep_key}'")

        cc = _import_cellcharter()
        n_spots = adata.shape[0]
        ad_input = adata.copy()

        # Build sample-isolated spatial graph
        build_sample_isolated_spatial_graph(
            ad_input,
            sample_key=self.sample_key,
            n_neighbors=self.n_neighbors,
        )
        adj = ad_input.obsp["spatial_connectivities"]
        groups = ad_input.obs[self.sample_key].to_numpy()
        rows, cols = adj.nonzero()
        cross_edges = int((groups[rows] != groups[cols]).sum())
        assert cross_edges == 0, f"Detected {cross_edges} illegal cross-sample edges!"

        cc.gr.aggregate_neighbors(
            ad_input,
            use_rep=self.scvi_rep_key,
            n_layers=self.n_layers,
            out_key="X_cellcharter_canonical",
        )
        emb = np.asarray(ad_input.obsm["X_cellcharter_canonical"], dtype=np.float32)

        canonical_labels = None
        if not embedding_only and n_clusters is not None:
            gmm = GaussianMixture(n_components=n_clusters, random_state=seed)
            canonical_labels = gmm.fit_predict(emb)

        result = CanonicalAdapterResult(
            method=self.name,
            embedding=emb,
            canonical_labels=canonical_labels,
            metadata={
                "adapter": self.name,
                "track": "smoke_precomputed",
                "scvi_source": "dataset_precomputed_zenodo_scvi",
                "canonical_scvi_pipeline": False,
                "n_layers": self.n_layers,
                "n_neighbors": self.n_neighbors,
                "representation_backend": f"Precomputed_{self.scvi_rep_key}",
                "seed": seed,
                "upstream_commit": git_commit(CELLCHARTER_REPOSITORY),
                "source_tree_sha256": source_tree_sha256(CELLCHARTER_SOURCE),
            },
        )
        result.validate(n_spots)
        return result
