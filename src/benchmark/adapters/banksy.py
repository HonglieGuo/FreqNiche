from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from .base import CanonicalAdapterResult, CanonicalSpatialAdapter
from .common import (
    ROOT, canonical_mclust_R, git_commit, prepare_banksy_dlpfc_adata,
    prepare_hvg_adata, require_spatial, source_tree_sha256,
)


BANKSY_REPOSITORY = ROOT / "baselines/Banksy_py"
BANKSY_SOURCE = BANKSY_REPOSITORY / "src"


class OfficialBANKSYAdapter(CanonicalSpatialAdapter):
    """Official pyBANKSY feature construction with true-K mclust."""

    name = "BANKSY_official"

    def __init__(
        self,
        *,
        protocol: str = "canonical",
        n_top_genes: int | None = None,
        n_neighbors: int = 18,
        lambda_param: float = 0.2,
        max_m: int = 1,
        pca_dim: int = 20,
        return_diagnostics: bool = False,
    ) -> None:
        if protocol not in {"canonical", "controlled"}:
            raise ValueError(f"Unknown BANKSY protocol: {protocol}")
        self.protocol = protocol
        self.n_top_genes = (
            int(n_top_genes) if n_top_genes is not None
            else (2000 if protocol == "canonical" else 3000)
        )
        self.n_neighbors = n_neighbors
        self.lambda_param = lambda_param
        self.max_m = max_m
        self.pca_dim = pca_dim
        self.sample_key = "sample_id"
        self.return_diagnostics = return_diagnostics

    def fit_transform(
        self,
        adata: ad.AnnData,
        *,
        n_clusters: int | None = None,
        seed: int = 0,
        device: torch.device | None = None,
        embedding_only: bool = False,
        return_diagnostics: bool | None = None,
    ) -> CanonicalAdapterResult:
        del device  # Official BANKSY feature construction is CPU based.
        if not embedding_only and n_clusters is None:
            raise ValueError("n_clusters must be provided when embedding_only=False")
        if return_diagnostics is None:
            return_diagnostics = self.return_diagnostics
        np.random.seed(seed)
        if not BANKSY_SOURCE.is_dir():
            raise FileNotFoundError(f"Official BANKSY source missing: {BANKSY_SOURCE}")
        sys.path.insert(0, str(BANKSY_SOURCE))
        import scipy.sparse as sp
        from banksy.cluster_methods import mclust_R, run_Leiden_partition
        from banksy.embed_banksy import generate_banksy_matrix
        from banksy.initialize_banksy import initialize_banksy
        from banksy_utils.umap_pca import pca_umap
        data = (
            prepare_banksy_dlpfc_adata(adata, n_top_genes=self.n_top_genes)
            if self.protocol == "canonical"
            else adata.copy()
        )
        require_spatial(data)
        data.obs["_canonical_x"] = np.asarray(data.obsm["spatial"])[:, 0]
        data.obs["_canonical_y"] = np.asarray(data.obsm["spatial"])[:, 1]
        coordinate_keys = ("_canonical_x", "_canonical_y", "spatial")

        n_obs = data.n_obs
        sample_col = self.sample_key if self.sample_key in data.obs.columns else None
        unique_samples = data.obs[sample_col].unique() if sample_col else [None]
        decay_type = "scaled_gaussian"
        extras: dict[str, Any] = {}

        if sample_col is not None and len(unique_samples) > 1:
            # Multi-sample isolated graph construction:
            # Build spatial neighbor weights independently per sample to ensure 0 cross-sample edges
            all_rows = {m: [] for m in range(self.max_m + 1)}
            all_cols = {m: [] for m in range(self.max_m + 1)}
            all_data = {m: [] for m in range(self.max_m + 1)}

            for s in unique_samples:
                s_mask = (data.obs[sample_col] == s).values
                s_idx = np.where(s_mask)[0]
                s_adata = data[s_idx].copy()

                max_k_allowed = max(1, (s_adata.n_obs - 1) // (self.max_m + 1))
                s_k = min(self.n_neighbors, max_k_allowed)
                s_dict = initialize_banksy(
                    s_adata,
                    coord_keys=coordinate_keys,
                    num_neighbours=s_k,
                    nbr_weight_decay=decay_type,
                    max_m=self.max_m,
                    plt_edge_hist=False,
                    plt_nbr_weights=False,
                    plt_agf_angles=False,
                    plt_theta=False,
                )
                for m in range(self.max_m + 1):
                    sub_w = s_dict[decay_type]["weights"][m].tocoo()
                    all_rows[m].append(s_idx[sub_w.row])
                    all_cols[m].append(s_idx[sub_w.col])
                    all_data[m].append(sub_w.data)

            # Assemble global block-diagonal sparse CSR matrices preserving native dtype per m
            global_weights_graphs = {}
            for m in range(self.max_m + 1):
                cat_rows = np.concatenate(all_rows[m]) if all_rows[m] else np.array([], dtype=int)
                cat_cols = np.concatenate(all_cols[m]) if all_cols[m] else np.array([], dtype=int)
                if all_data[m]:
                    weight_dtype = np.result_type(*[arr.dtype for arr in all_data[m]])
                    cat_data = np.concatenate(all_data[m]).astype(weight_dtype, copy=False)
                else:
                    weight_dtype = np.float32
                    cat_data = np.array([], dtype=weight_dtype)
                global_w = sp.csr_matrix((cat_data, (cat_rows, cat_cols)), shape=(n_obs, n_obs), dtype=weight_dtype)
                global_weights_graphs[m] = global_w

            if self.max_m >= 1 and 1 in global_weights_graphs and global_weights_graphs[1].nnz > 0:
                assert np.iscomplexobj(global_weights_graphs[1].data), (
                    f"BANKSY m=1 AGF spatial weights must be complex-valued, got {global_weights_graphs[1].dtype}"
                )

            banksy_dictionary = {decay_type: {"weights": global_weights_graphs}}
            cross_edges = 0
            for m in range(self.max_m + 1):
                w = global_weights_graphs[m]
                r, c = w.nonzero()
                groups = data.obs[sample_col].to_numpy()
                n_cross = int((groups[r] != groups[c]).sum())
                if n_cross > 0:
                    raise RuntimeError(f"FATAL: Detected {n_cross} cross-sample edges in BANKSY weights for m={m}")
            track_name = "BANKSY-Controlled-Multisample"
        else:
            k_eff = min(self.n_neighbors, max(1, (data.n_obs - 1) // (self.max_m + 1)))
            banksy_dictionary = initialize_banksy(
                data,
                coord_keys=coordinate_keys,
                num_neighbours=k_eff,
                nbr_weight_decay=decay_type,
                max_m=self.max_m,
                plt_edge_hist=False,
                plt_nbr_weights=False,
                plt_agf_angles=False,
                plt_theta=False,
            )
            cross_edges = 0
            track_name = self.protocol

        banksy_dictionary, banksy_matrix = generate_banksy_matrix(
            data,
            banksy_dictionary,
            lambda_list=[self.lambda_param],
            max_m=self.max_m,
            plot_std=False,
            save_matrix=False,
            variance_balance=False,
            verbose=False,
        )

        # Gate diagnostic copies behind return_diagnostics to avoid inflating production scale RAM
        if return_diagnostics:
            raw_local_features = np.asarray(
                banksy_dictionary[decay_type]["norm_counts_concatenated"], dtype=np.float32
            ).copy()
            extras["raw_local_features"] = raw_local_features
            extras["pre_pca_matrix"] = raw_local_features
            extras["zscored_banksy_matrix"] = np.asarray(banksy_matrix.X, dtype=np.float32).copy()

        pca_umap(
            banksy_dictionary,
            pca_dims=[self.pca_dim],
            plt_remaining_var=False,
            add_umap=False,
        )
        embedding = np.asarray(
            banksy_matrix.obsm[f"reduced_pc_{self.pca_dim}"], dtype=np.float32
        )
        if embedding_only:
            labels = None
            canonical_clusterer = "none_embedding_only"
        elif self.protocol == "canonical":
            assert n_clusters is not None
            resolutions = np.round(np.linspace(0.05, 1.6, 32), 3).tolist()
            results, _ = run_Leiden_partition(
                banksy_dict=banksy_dictionary,
                resolutions=resolutions,
                num_nn=50,
                num_iterations=-1,
                partition_seed=seed,
                match_labels=False,
                annotations=None,
                max_labels=None,
                verbose=False,
            )
            candidates = results[results["num_labels"].astype(int) == n_clusters].copy()
            if candidates.empty:
                # Two-stage fine resolution search around closest range
                diffs = (results["num_labels"].astype(int) - n_clusters).abs()
                best_res = float(results.loc[diffs.idxmin(), "resolution"])
                fine_resolutions = np.round(
                    np.linspace(max(0.01, best_res - 0.15), best_res + 0.15, 60), 4
                ).tolist()
                fine_results, _ = run_Leiden_partition(
                    banksy_dict=banksy_dictionary,
                    resolutions=fine_resolutions,
                    num_nn=50,
                    num_iterations=-1,
                    partition_seed=seed,
                    match_labels=False,
                    annotations=None,
                    max_labels=None,
                    verbose=False,
                )
                candidates = fine_results[fine_results["num_labels"].astype(int) == n_clusters].copy()

            if candidates.empty:
                raise RuntimeError(
                    f"BANKSY strict true-K failure: no Leiden resolution produced exactly {n_clusters} clusters"
                )

            candidates = candidates.sort_values("resolution")
            candidate_labels = np.stack([
                np.asarray(value.dense if hasattr(value, "dense") else value)
                for value in candidates["labels"]
            ])
            labels = candidate_labels[len(candidate_labels) // 2]
            extras["canonical_candidate_labels"] = candidate_labels
            extras["canonical_candidate_resolutions"] = candidates[
                "resolution"
            ].to_numpy(dtype=np.float32)
            canonical_clusterer = "Leiden_resolution_sweep_matched_K"
        else:
            assert n_clusters is not None
            from .common import controlled_gmm_labels

            labels = controlled_gmm_labels(
                embedding,
                n_clusters=n_clusters,
                seed=seed,
            )
            canonical_clusterer = "GaussianMixture_full_trueK"

        result = CanonicalAdapterResult(
            method=self.name,
            embedding=embedding,
            canonical_labels=labels,
            metadata={
                "implementation": "official pyBANKSY",
                "repository": "https://github.com/prabhakarlab/Banksy_py",
                "commit": git_commit(BANKSY_REPOSITORY),
                "source_tree_sha256": source_tree_sha256(BANKSY_REPOSITORY),
                "protocol": self.protocol,
                "track": track_name,
                "sample_isolation": "block_diagonal" if sample_col and len(unique_samples) > 1 else "single_sample",
                "cross_sample_edges": cross_edges,
                "n_top_genes": self.n_top_genes,
                "controlled_input_preprocessing": "preprocessed_common_3000_gene_matrix_no_additional_normalization",
                "lambda_param": self.lambda_param,
                "preprocessing": (
                    "median_library_normalize_log1p_seurat_2000hvg"
                    if self.protocol == "canonical"
                    else "preprocessed_common_3000_gene_matrix_no_additional_normalization"
                ),
                "n_neighbors": self.n_neighbors,
                "lambda": self.lambda_param,
                "max_m": self.max_m,
                "pca_dim": self.pca_dim,
                "canonical_clusterer": canonical_clusterer,
                "controlled_clusterer": "GaussianMixture_full_trueK",
            },
            extras=extras,
        )
        result.validate(adata.n_obs)
        return result
