from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import scipy.sparse as sps
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from .base import CanonicalAdapterResult, CanonicalSpatialAdapter
from .common import ROOT, git_commit, require_spatial, source_tree_sha256


class OfficialCellNicheAdapter(CanonicalSpatialAdapter):
    """Official CellNiche Adapter for spatial microenvironment representation.

    Strict protocol adherence:
    - Input mode: 'expr' (uses adata.X only, never touches obs['cell_type'] or obs['niche'])
    - Unsupervised embedding: row-normalized contrastive node embeddings
    - If embedding_only=True: canonical_labels is None, zero True-K leakage
    """

    name = "CellNiche_official"

    def __init__(
        self,
        *,
        protocol: str = "controlled",
        track: str | None = None,
        sample_key: str = "sample_id",
        radius: float | None = 50.0,
        hidden_channels: list[int] | None = None,
        strategy: str = "freq",
        n_neighbors: int = 6,
        epochs: int = 5,
        batch_size: int = 512,
        lr: float = 0.001,
        wt: int = 20,
        wl: int = 5,
    ) -> None:
        if track is None:
            track = protocol
        elif track != protocol:
            raise ValueError(f"Conflicting parameters in CellNiche adapter: protocol='{protocol}' vs track='{track}'")
        self.protocol = protocol
        self.track = track
        self.sample_key = sample_key
        self.radius = radius
        self.hidden_channels = hidden_channels or [256, 128]
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.wt = wt
        self.wl = wl

    def fit_transform(
        self,
        adata: ad.AnnData,
        *,
        n_clusters: int | None = None,
        seed: int = 0,
        device: torch.device | None = None,
        embedding_only: bool = True,
        checkpoint_dir: Path | str | None = None,
    ) -> CanonicalAdapterResult:
        require_spatial(adata)

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        # Set reproducibility seeds
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        import cellniche
        from cellniche.model import Encoder, Model
        from cellniche.sampler import NeighborSampler
        from cellniche.utils import get_positivePairs
        from torch_sparse import SparseTensor
        from .common import build_sample_isolated_spatial_graph

        ad_work = adata.copy()
        n_spots = ad_work.n_obs

        # Ensure spatial connectivities exist with sample isolation
        if "spatial_connectivities" not in ad_work.obsp:
            if self.track == "canonical":
                build_sample_isolated_spatial_graph(
                    ad_work,
                    sample_key=self.sample_key,
                    n_neighbors=self.n_neighbors,
                )
                graph_builder = "cellniche_isolated_knn"
            else:
                build_sample_isolated_spatial_graph(
                    ad_work,
                    sample_key=self.sample_key,
                    radius=self.radius,
                    n_neighbors=self.n_neighbors if self.radius is None else None,
                )
                graph_builder = "sample_isolated_radius_50"
        else:
            graph_builder = "precomputed_spatial_connectivities"

        # Assert zero cross-sample leakage
        adj_raw = ad_work.obsp["spatial_connectivities"]
        if self.sample_key in ad_work.obs.columns:
            groups = ad_work.obs[self.sample_key].to_numpy()
            rows_chk, cols_chk = adj_raw.nonzero()
            cross_edges = int((groups[rows_chk] != groups[cols_chk]).sum())
            if cross_edges > 0:
                raise RuntimeError(f"FATAL: Detected {cross_edges} illegal cross-sample edges in CellNiche graph!")
        else:
            cross_edges = 0

        import hashlib
        csr_chk = adj_raw.tocsr()
        h = hashlib.sha256()
        h.update(csr_chk.data.tobytes())
        h.update(csr_chk.indices.tobytes())
        h.update(csr_chk.indptr.tobytes())
        csr_hash = h.hexdigest()

        # Extract features (strictly expr, never pheno labels)
        X_mat = ad_work.X.toarray() if hasattr(ad_work.X, "toarray") else ad_work.X
        x = torch.as_tensor(X_mat, dtype=torch.float)

        # Extract adjacency & build edge_index without self-loops
        adj_sp = adj_raw.astype(bool)
        if not isinstance(adj_sp, sps.csr_matrix):
            adj_sp = adj_sp.tocsr()
        adj_sp = adj_sp.tolil()
        adj_sp.setdiag(0)
        adj_sp = adj_sp.tocsr()
        adj_sp.eliminate_zeros()
        rows, cols = adj_sp.nonzero()
        edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)

        # Sparse adjacency tensor for neighbor sampling
        adj_t = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=torch.ones(edge_index.shape[1]),
            sparse_sizes=(n_spots, n_spots),
        )

        effective_batch_size = min(self.batch_size, n_spots)
        train_loader = NeighborSampler(
            edge_index,
            adj_t,
            sizes=[10, 10],
            is_train=True,
            wt=self.wt,
            wl=self.wl,
            batch_size=effective_batch_size,
            shuffle=True,
            drop_last=False,
        )

        test_loader = NeighborSampler(
            edge_index,
            adj_t,
            sizes=[10, 10],
            is_train=False,
            wt=self.wt,
            wl=self.wl,
            batch_size=effective_batch_size * 2,
            shuffle=False,
            drop_last=False,
        )

        # Instantiate CellNiche Encoder & Model
        encoder = Encoder(
            in_channels=x.shape[1],
            hidden_channels=self.hidden_channels,
            dropout=0.0,
            negative_slope=0.5,
        ).to(device)

        model = Model(
            encoder=encoder,
            decoder_hidden=None,
            project_hidden=None,
            use_weight=False,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.0)

        # Training loop
        x_dev = x.to(device)
        model.train()
        for _ in range(self.epochs):
            for (bs, n_id, adjs), adj_batch, *rest in train_loader:
                adjs = [adj.to(device) for adj in adjs]
                adj_batch = adj_batch.to(device)
                pos_mask = get_positivePairs(adj_batch, x_dev[n_id], strategy=self.strategy)

                optimizer.zero_grad()
                z, proj_z, rec_expr, target_expr = model(x_dev[n_id], adjs=adjs)
                z = F.normalize(z, p=2, dim=1)
                proj_z = F.normalize(proj_z, p=2, dim=1)
                contrast_loss, _ = model.compute_loss(
                    z, pos_mask, proj_z, rec_expr, target=target_expr
                )
                contrast_loss.backward()
                optimizer.step()

        # Persist model checkpoint if requested
        if checkpoint_dir is not None:
            ckpt_path = Path(checkpoint_dir)
            ckpt_path.mkdir(parents=True, exist_ok=True)
            save_file = ckpt_path / f"cellniche_model_seed_{seed}.pt"
            torch.save({
                "encoder_state_dict": encoder.state_dict(),
                "model_state_dict": model.state_dict(),
                "seed": seed,
            }, save_file)
            print(f"[OK] Persisted CellNiche checkpoint to: {save_file}")

        # Inference
        model.eval()
        z_all = torch.zeros((n_spots, self.hidden_channels[-1]), device="cpu")
        with torch.no_grad():
            for (bs, n_id, adjs), _, batch in test_loader:
                adjs = [adj.to(device) for adj in adjs]
                z, _, _, _ = model(x[n_id].to(device), adjs=adjs)
                z_all[batch] = z.detach().cpu().float()

        z_all = F.normalize(z_all, p=2, dim=1)
        emb = z_all.numpy().astype(np.float32)

        # Labels
        labels = None
        if not embedding_only and n_clusters is not None:
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
            labels = kmeans.fit_predict(emb)

        result = CanonicalAdapterResult(
            method="cellniche",
            embedding=emb,
            canonical_labels=labels,
            metadata={
                "adapter": self.name,
                "protocol": self.protocol,
                "embedding_type": "expr",
                "hidden_channels": self.hidden_channels,
                "strategy": self.strategy,
                "epochs": self.epochs,
                "n_neighbors": self.n_neighbors,
                "batch_size": self.batch_size,
                "seed": seed,
                "device": str(device),
                "cellniche_version": getattr(cellniche, "__version__", "0.1.8"),
                "canonical_labels_assigned": labels is not None,
                "embedding_only": embedding_only,
            },
        )
        result.validate(n_spots)
        return result
