"""Evaluation Engine for W2.4 Competitive Benchmark (Production v3.2).

Strictly adheres to:
1. Primary Ranking Metric: Test Donor-Macro ARI (equal-weighted average across D028, D082, D145, D151).
2. Common Leiden Readout (Primary):
   - Shared deterministic 10-fold 80% bootstrap manifest across all candidate resolutions, models, and seeds.
   - Ground-truth labels are 100% EXCLUDED during resolution selection.
   - Validation barcodes order SHA-256 is explicitly verified to prevent row misalignment.
   - Frozen r* is applied to Test 8,000 centers to compute un-biased clustering embeddings quality.
   - Feature post-processing policy is strictly locked to "raw embedding / no post-scaling".
3. Diagnostic & Secondary Metrics:
   - Donor-Macro NMI
   - Worst-Donor ARI (robustness lower-bound)
   - Donor ARI Std (cross-donor variance)
   - Median Donor ARI
   - ASW (Average Silhouette Width)
4. Official End-to-End Readout (Secondary):
   - Evaluates official algorithm end-to-end clustering outputs with identical donor-macro metric implementations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from src.freqniche_st.runtime_paths import resolve_run_output

ROOT = Path(__file__).resolve().parents[2]
VAL_BARCODES_DEFAULT_PATH = ROOT / "outputs/contracts/human_skin/validation_barcodes_8k.csv"
DEFAULT_MANIFEST_PATH = (
    resolve_run_output("baselines/Baselines_outputs/runs/human_skin/2026-09-18/20260918_human_skin_benchmark_final") / "bootstrap_manifest_val8k_10fold.json"
)


def compute_barcodes_sha256(barcodes: list[str]) -> str:
    """Computes SHA-256 hash of ordered cell barcodes list."""
    h = hashlib.sha256()
    for bc in barcodes:
        h.update(bc.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def generate_or_load_val_bootstrap_manifest(
    manifest_path: Path | str | None = None,
    val_barcodes_path: Path | str | None = None,
    n_cells: int = 8000,
    n_bootstraps: int = 10,
    subsample_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Generates or loads the deterministic 10-fold 80% bootstrap manifest.

    Ensures identical cell subsets across all resolutions, methods, and seeds,
    preventing Monte Carlo sampling variance in unsupervised stability selection.
    Includes validation barcodes SHA-256 hash to ensure row-order consistency.
    """
    manifest_file = Path(manifest_path) if manifest_path is not None else DEFAULT_MANIFEST_PATH
    val_bc_file = Path(val_barcodes_path) if val_barcodes_path is not None else VAL_BARCODES_DEFAULT_PATH

    if val_bc_file.exists():
        df_bc = pd.read_csv(val_bc_file)
        val_barcodes = list(df_bc["cell_barcode"])
        expected_bc_sha = compute_barcodes_sha256(val_barcodes)
    else:
        val_barcodes = None
        expected_bc_sha = None

    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["n_cells"] == n_cells, f"Manifest n_cells mismatch: {data['n_cells']} vs {n_cells}"
        assert data["n_bootstraps"] == n_bootstraps, "Manifest n_bootstraps mismatch"
        assert abs(data["subsample_ratio"] - subsample_ratio) < 1e-5, "Manifest subsample_ratio mismatch"

        if expected_bc_sha is not None and "validation_barcodes_sha256" in data:
            assert data["validation_barcodes_sha256"] == expected_bc_sha, (
                f"Validation barcodes row-order hash mismatch! Expected {expected_bc_sha}, "
                f"found {data['validation_barcodes_sha256']}"
            )

        indices = [np.array(idx_list, dtype=np.int64) for idx_list in data["bootstrap_indices"]]
        return indices, data

    # Generate deterministically
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    subsample_size = int(n_cells * subsample_ratio)

    bootstrap_indices = []
    for _ in range(n_bootstraps):
        sub_idx = rng.choice(n_cells, size=subsample_size, replace=False)
        sub_idx.sort()
        bootstrap_indices.append(sub_idx)

    manifest_data = {
        "manifest_version": "1.0",
        "description": "Deterministic 10-fold 80% bootstrap sub-sampling manifest for W2.4 Validation 8,000 centers.",
        "n_cells": n_cells,
        "n_bootstraps": n_bootstraps,
        "subsample_ratio": subsample_ratio,
        "subsample_size": subsample_size,
        "seed": seed,
        "validation_barcodes_sha256": expected_bc_sha,
        "bootstrap_indices": [idx.tolist() for idx in bootstrap_indices],
    }

    # Compute manifest payload SHA-256
    payload_str = json.dumps(manifest_data, sort_keys=True)
    manifest_data["manifest_sha256"] = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return bootstrap_indices, manifest_data


def select_unsupervised_resolution_r_star(
    val_embeddings: np.ndarray,
    resolutions: list[float] | None = None,
    manifest_path: Path | str | None = None,
    val_barcodes_path: Path | str | None = None,
    k_min: int = 5,
    k_max: int = 25,
    embedding_postprocessing: str = "raw",
    seed: int = 42,
) -> tuple[float, dict[str, Any]]:
    """Selects optimal Leiden resolution r* strictly via unsupervised subsampling stability on Validation embeddings.

    Labels are 100% excluded.
    Strictly uses shared deterministic bootstrap manifest to eliminate sampling differences.
    Embedding post-processing is strictly locked to 'raw'.
    """
    assert embedding_postprocessing == "raw", (
        f"Unsupported embedding_postprocessing: '{embedding_postprocessing}'. W2.4 protocol locks to 'raw'."
    )

    if resolutions is None:
        resolutions = [0.2, 0.4, 0.6, 0.8, 1.0]

    val_embeddings = np.asarray(val_embeddings, dtype=np.float32)
    n_cells = val_embeddings.shape[0]

    # Load shared deterministic bootstrap indices
    bootstrap_indices, manifest_meta = generate_or_load_val_bootstrap_manifest(
        manifest_path=manifest_path,
        val_barcodes_path=val_barcodes_path,
        n_cells=n_cells,
        n_bootstraps=10,
        subsample_ratio=0.8,
        seed=seed,
    )
    n_bootstraps = len(bootstrap_indices)

    scores_per_res: dict[float, float] = {}
    mean_k_per_res: dict[float, float] = {}

    for res in resolutions:
        clusterings: list[np.ndarray] = []

        for b in range(n_bootstraps):
            sub_idx = bootstrap_indices[b]
            z_sub = val_embeddings[sub_idx]

            adata_sub = ad.AnnData(X=z_sub)
            sc.pp.neighbors(adata_sub, n_neighbors=15, use_rep="X")
            sc.tl.leiden(adata_sub, resolution=res, key_added="leiden", random_state=seed + b)

            clusterings.append(adata_sub.obs["leiden"].to_numpy())

        pair_aris: list[float] = []
        n_clusters_list = [len(np.unique(c)) for c in clusterings]
        mean_k = float(np.mean(n_clusters_list))
        mean_k_per_res[res] = mean_k

        for i in range(n_bootstraps):
            for j in range(i + 1, n_bootstraps):
                idx_i = bootstrap_indices[i]
                idx_j = bootstrap_indices[j]
                common_cells, pos_i, pos_j = np.intersect1d(idx_i, idx_j, return_indices=True)
                if len(common_cells) > 10:
                    c_i = clusterings[i][pos_i]
                    c_j = clusterings[j][pos_j]
                    pair_aris.append(float(adjusted_rand_score(c_i, c_j)))

        scores_per_res[res] = float(np.mean(pair_aris)) if pair_aris else 0.0

    # Rule: First restrict to resolutions where mean cluster count is in [k_min, k_max]
    valid_res = [r for r in resolutions if k_min <= mean_k_per_res[r] <= k_max]

    if valid_res:
        # Highest stability score; tie-breaker: smaller r (more conservative)
        best_res = max(valid_res, key=lambda r: (scores_per_res[r], -r))
    else:
        # Fallback: minimize distance to [k_min, k_max]
        def dist_to_interval(r: float) -> float:
            k = mean_k_per_res[r]
            if k < k_min:
                return k_min - k
            if k > k_max:
                return k - k_max
            return 0.0

        best_res = min(resolutions, key=lambda r: (dist_to_interval(r), -scores_per_res[r]))

    report = {
        "optimal_r_star": float(best_res),
        "stability_scores": {str(k): float(v) for k, v in scores_per_res.items()},
        "mean_clusters": {str(k): float(v) for k, v in mean_k_per_res.items()},
        "best_stability_score": float(scores_per_res[best_res]),
        "manifest_sha256": manifest_meta.get("manifest_sha256"),
        "validation_barcodes_sha256": manifest_meta.get("validation_barcodes_sha256"),
        "embedding_postprocessing": embedding_postprocessing,
    }

    return float(best_res), report


def evaluate_common_leiden_readout(
    test_embeddings: np.ndarray,
    test_metadata: pd.DataFrame,
    frozen_r_star: float,
    embedding_postprocessing: str = "raw",
    seed: int = 42,
    compute_asw: bool = True,
) -> dict[str, Any]:
    """Applies frozen r* to Test centers embeddings, and computes un-biased Test Donor-Macro metrics.

    Metrics calculated:
    - Test Donor-Macro ARI (primary)
    - Donor-Macro NMI
    - Worst-Donor ARI
    - Donor ARI Std
    - Median Donor ARI
    - ASW
    """
    assert embedding_postprocessing == "raw", (
        f"Unsupported embedding_postprocessing: '{embedding_postprocessing}'. W2.4 protocol locks to 'raw'."
    )
    test_embeddings = np.asarray(test_embeddings, dtype=np.float32)
    assert len(test_embeddings) == len(test_metadata), (
        f"Mismatch: embeddings {test_embeddings.shape[0]} vs metadata {len(test_metadata)}"
    )

    # 1. Cluster all Test centers on latent kNN graph using frozen r*
    adata = ad.AnnData(X=test_embeddings)
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X")
    sc.tl.leiden(adata, resolution=frozen_r_star, key_added="pred_cluster", random_state=seed)
    y_pred_all = adata.obs["pred_cluster"].to_numpy()

    # 2. Extract ground-truth niche and donor_id
    donor_col = "donor_id"
    niche_col = "niche"
    assert donor_col in test_metadata.columns, f"Missing '{donor_col}' in metadata"
    assert niche_col in test_metadata.columns, f"Missing '{niche_col}' in metadata"

    donors = sorted(test_metadata[donor_col].unique())
    per_donor_metrics: dict[str, dict[str, float]] = {}
    donor_aris: list[float] = []
    donor_nmis: list[float] = []

    for d in donors:
        mask_d = (test_metadata[donor_col] == d).to_numpy()
        y_true_d = test_metadata.loc[mask_d, niche_col].to_numpy()
        y_pred_d = y_pred_all[mask_d]

        # Valid mask (non-null labels)
        valid_mask = ~pd.isna(y_true_d)
        y_true_valid = y_true_d[valid_mask]
        y_pred_valid = y_pred_d[valid_mask]

        if len(np.unique(y_true_valid)) >= 2:
            ari_d = float(adjusted_rand_score(y_true_valid, y_pred_valid))
            nmi_d = float(normalized_mutual_info_score(y_true_valid, y_pred_valid))
        else:
            ari_d, nmi_d = 0.0, 0.0

        per_donor_metrics[d] = {
            "n_cells": int(mask_d.sum()),
            "ari": ari_d,
            "nmi": nmi_d,
            "n_true_niches": int(len(np.unique(y_true_valid))),
            "n_pred_clusters": int(len(np.unique(y_pred_valid))),
        }
        donor_aris.append(ari_d)
        donor_nmis.append(nmi_d)

    # 3. Macro metrics (equal weighting across donors)
    macro_ari = float(np.mean(donor_aris))
    worst_ari = float(np.min(donor_aris))
    std_ari = float(np.std(donor_aris))
    median_ari = float(np.median(donor_aris))
    macro_nmi = float(np.mean(donor_nmis))

    # 4. Overall Silhouette width (ASW)
    asw = float("nan")
    if compute_asw and len(np.unique(y_pred_all)) > 1:
        try:
            # Subsample up to 4,000 cells for fast silhouette computation if large
            if len(test_embeddings) > 4000:
                sub_idx = np.random.RandomState(seed).choice(len(test_embeddings), 4000, replace=False)
                asw = float(silhouette_score(test_embeddings[sub_idx], y_pred_all[sub_idx]))
            else:
                asw = float(silhouette_score(test_embeddings, y_pred_all))
        except Exception:
            asw = float("nan")

    return {
        "readout_type": "common_leiden",
        "frozen_r_star": float(frozen_r_star),
        "embedding_postprocessing": embedding_postprocessing,
        "test_donor_macro_ari": macro_ari,
        "worst_donor_ari": worst_ari,
        "donor_ari_std": std_ari,
        "median_donor_ari": median_ari,
        "test_donor_macro_nmi": macro_nmi,
        "asw_silhouette": asw,
        "total_pred_clusters": int(len(np.unique(y_pred_all))),
        "per_donor_metrics": per_donor_metrics,
    }


def evaluate_official_labels_readout(
    pred_labels: np.ndarray,
    test_metadata: pd.DataFrame,
    label_source: str = "official_clustering",
) -> dict[str, Any]:
    """Evaluates official algorithm end-to-end clustering outputs with identical donor-macro rules."""
    pred_labels = np.asarray(pred_labels)
    assert len(pred_labels) == len(test_metadata), (
        f"Mismatch: labels {len(pred_labels)} vs metadata {len(test_metadata)}"
    )

    donor_col = "donor_id"
    niche_col = "niche"
    donors = sorted(test_metadata[donor_col].unique())
    per_donor_metrics: dict[str, dict[str, float]] = {}
    donor_aris: list[float] = []
    donor_nmis: list[float] = []

    for d in donors:
        mask_d = (test_metadata[donor_col] == d).to_numpy()
        y_true_d = test_metadata.loc[mask_d, niche_col].to_numpy()
        y_pred_d = pred_labels[mask_d]

        valid_mask = ~pd.isna(y_true_d)
        y_true_valid = y_true_d[valid_mask]
        y_pred_valid = y_pred_d[valid_mask]

        if len(np.unique(y_true_valid)) >= 2:
            ari_d = float(adjusted_rand_score(y_true_valid, y_pred_valid))
            nmi_d = float(normalized_mutual_info_score(y_true_valid, y_pred_valid))
        else:
            ari_d, nmi_d = 0.0, 0.0

        per_donor_metrics[d] = {
            "n_cells": int(mask_d.sum()),
            "ari": ari_d,
            "nmi": nmi_d,
            "n_true_niches": int(len(np.unique(y_true_valid))),
            "n_pred_clusters": int(len(np.unique(y_pred_valid))),
        }
        donor_aris.append(ari_d)
        donor_nmis.append(nmi_d)

    return {
        "readout_type": "official_end_to_end",
        "label_source": label_source,
        "test_donor_macro_ari": float(np.mean(donor_aris)),
        "worst_donor_ari": float(np.min(donor_aris)),
        "donor_ari_std": float(np.std(donor_aris)),
        "median_donor_ari": float(np.median(donor_aris)),
        "test_donor_macro_nmi": float(np.mean(donor_nmis)),
        "total_pred_clusters": int(len(np.unique(pred_labels))),
        "per_donor_metrics": per_donor_metrics,
    }
