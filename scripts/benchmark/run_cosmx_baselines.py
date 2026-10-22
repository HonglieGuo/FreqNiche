"""
Unified Baseline Benchmark Runner for CosMx Human NSCLC (W2.6).

Supported Methods (Controlled Benchmark Track aligned with SEA-AD):
1. raw_pca: Standard 50-d PCA on normalized expression (Inductive lower bound, fit on Train+Val).
2. px_resdiff: 50-d DualFreq PCA on spatial smoothed PX and high-pass residual H (Inductive, fit on Train+Val).
3. banksy: BANKSY-Controlled spatial feature augmentation (Transductive, lambda=0.2, pca_dim=20).
4. cellcharter: CellCharter-PCA-Controlled neighborhood aggregation (Transductive, pca_dim=50, n_layers=3, R=50.0um).
5. cellniche: CellNiche-Controlled microenvironment contrastive representation (Transductive, R=50.0um).
6. stagate: STAGATE-Controlled spatial graph autoencoder (Transductive, R=50.0um, optional).

Strict Protocol Alignment:
- Physical coordinates: strictly reads adata.obsm['spatial_um'] (or 'spatial', which equals spatial_um), in microns;
  NO secondary scaling by 0.120280945!
- Physical Label Removal: All adapters strictly receive adata_fit with all niche/cluster labels physically dropped.
- Preflight Guard: Mandates H5AD SHA-256 == eb755843..., coordinate invariants, and file SHA match with audit manifest.
- Fail-Closed Execution: Refuses to silently overwrite or reuse existing assets without explicit --overwrite or verified --resume.
- Lightweight Genuine Smoke: Generates an isolated small adata_smoke across 2 physical samples (1,000 cells total).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.freqniche_st.graph import build_global_spatial_graph
from src.freqniche_st.runtime_paths import resolve_run_output, resolve_baseline_output

DATASET_PATH = WORKSPACE_ROOT / "datasets" / "h5ad" / "cosmx_human_nsclc" / "cosmx_human_nsclc.h5ad"
EXPECTED_H5AD_SHA256 = "eb755843022db83dbbcc4c3da426dad0426cd706a4e88b1bac68cad046c751f4"

BASE_DIR = resolve_run_output("outputs/runs/freqniche/cosmx_nsclc/2026-09-17/20260917_cosmx_nsclc_final")
BASELINES_DIR = resolve_baseline_output("baselines/Baselines_outputs/runs/cosmx_nsclc/2026-09-17/20260917_cosmx_nsclc_baselines")
EMB_DIR = BASELINES_DIR / "embeddings"
ASSIGN_DIR = BASELINES_DIR / "assignments"
RES_DIR = BASELINES_DIR / "results"

VAL_CENTERS_PATH = BASE_DIR / "val_centers.json"
VAL_FOLDS_PATH = BASE_DIR / "val_bootstrap_folds.json"
TEST_CENTERS_PATH = BASE_DIR / "test_centers.json"
AUDIT_MANIFEST_PATH = BASE_DIR / "cosmx_data_audit_manifest.json"

FROZEN_RESOLUTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
K_MIN = 5
K_MAX = 25
DEFAULT_SEEDS = [42, 123, 2024]

METHOD_METADATA = {
    "raw_pca": {
        "variant": "Raw PCA",
        "paradigm": "Inductive",
        "fit_data_scope": "train_val_unsupervised",
        "description": "50-d PCA on normalized expression (fit on Train+Val, transform Test)"
    },
    "px_resdiff": {
        "variant": "DualFreq (px_resdiff)",
        "paradigm": "Inductive",
        "fit_data_scope": "train_val_unsupervised",
        "description": "50-d PCA on [PX, H] multi-frequency features (fit on Train+Val, transform Test)"
    },
    "banksy": {
        "variant": "BANKSY-Controlled",
        "paradigm": "Transductive",
        "fit_data_scope": "full_cohort_unsupervised",
        "description": "Controlled BANKSY spatial feature augmentation (lambda=0.2, pca_dim=20)"
    },
    "cellcharter": {
        "variant": "CellCharter-PCA-Controlled",
        "paradigm": "Transductive",
        "fit_data_scope": "full_cohort_unsupervised",
        "description": "Controlled CellCharter neighborhood aggregation (pca_dim=50, n_layers=3, R=50um)"
    },
    "cellniche": {
        "variant": "CellNiche-Controlled",
        "paradigm": "Transductive",
        "fit_data_scope": "full_cohort_unsupervised",
        "description": "Controlled CellNiche microenvironment contrastive representation (R=50um, 5 epochs)"
    },
    "stagate": {
        "variant": "STAGATE-Controlled",
        "paradigm": "Transductive",
        "fit_data_scope": "full_cohort_unsupervised",
        "description": "Controlled STAGATE spatial graph autoencoder (R=50um)"
    }
}


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# ==============================================================================
# PREFLIGHT INTEGRITY GATE
# ==============================================================================

def verify_cosmx_baseline_preflight(
    adata: ad.AnnData,
    val_cohort: Dict[str, Any],
    test_cohort: Dict[str, Any],
    val_folds: Dict[str, Any]
) -> str:
    """
    Strict Cryptographic & Physical Invariant Gate. Fail Closed on any discrepancy.
    """
    print("\n" + "=" * 80)
    print("RUNNING STRICT COSMX BASELINE PREFLIGHT INTEGRITY GATE")
    print("=" * 80)

    # 1. Verify Master Audit Manifest Existence & H5AD SHA-256
    print(" -> [1/6] Checking Master Data Audit Manifest & H5AD cryptographic hash ...")
    if not AUDIT_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing master audit manifest at {AUDIT_MANIFEST_PATH}")
    with open(AUDIT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        audit_manifest = json.load(f)

    h5ad_sha = compute_file_sha256(DATASET_PATH)
    if h5ad_sha != EXPECTED_H5AD_SHA256:
        raise ValueError(
            f"CRITICAL H5AD SHA MISMATCH! Expected {EXPECTED_H5AD_SHA256}, but got {h5ad_sha}. "
            "Remote server is using uncorrected coordinates! Fail Closed."
        )
    if audit_manifest.get("h5ad_sha256") != EXPECTED_H5AD_SHA256:
        raise ValueError(f"Audit manifest h5ad_sha256 mismatch! Got {audit_manifest.get('h5ad_sha256')}")
    print(f"    Verified H5AD SHA-256: {h5ad_sha[:16]}...")

    # 2. Verify Cohort Files SHA against Master Audit Manifest
    print(" -> [2/6] Verifying Cohort Manifests SHA against Audit Manifest ...")
    val_centers_sha = compute_file_sha256(VAL_CENTERS_PATH)
    expected_val_sha = audit_manifest["cohorts"]["validation"]["file_sha256"]
    if val_centers_sha != expected_val_sha:
        raise ValueError(f"val_centers.json SHA mismatch! Got {val_centers_sha}, expected {expected_val_sha}")

    test_centers_sha = compute_file_sha256(TEST_CENTERS_PATH)
    expected_test_sha = audit_manifest["cohorts"]["test"]["file_sha256"]
    if test_centers_sha != expected_test_sha:
        raise ValueError(f"test_centers.json SHA mismatch! Got {test_centers_sha}, expected {expected_test_sha}")

    folds_sha = compute_file_sha256(VAL_FOLDS_PATH)
    expected_folds_sha = audit_manifest["cohorts"]["val_bootstrap_folds"]["file_sha256"]
    if folds_sha != expected_folds_sha:
        raise ValueError(f"val_bootstrap_folds.json SHA mismatch! Got {folds_sha}, expected {expected_folds_sha}")
    print("    Verified val_centers, test_centers, and folds SHA matches master manifest.")

    # 3. Verify Coordinate Invariants
    print(" -> [3/6] Checking coordinate contracts ...")
    assert "spatial_um" in adata.obsm, "Missing 'spatial_um' in adata.obsm"
    assert "spatial" in adata.obsm, "Missing 'spatial' in adata.obsm"
    max_coord_diff = float(np.max(np.abs(adata.obsm["spatial"] - adata.obsm["spatial_um"])))
    assert max_coord_diff == 0.0, f"spatial and spatial_um differ by {max_coord_diff}"
    assert (adata.obs["spatial_unit"].astype(str) == "micron").all(), "spatial_unit must be all 'micron'"
    print("    Verified physical micron coordinate contracts (diff=0.0).")

    # 4. Verify Validation Cohort Row Mapping
    print(" -> [4/6] Checking Validation cohort alignment ...")
    val_idx = np.array(val_cohort["global_indices"], dtype=np.int64)
    val_barcodes = np.array(val_cohort["cell_barcodes"], dtype=str)
    assert len(val_idx) == 10000, f"Expected 10000 val centers, got {len(val_idx)}"
    assert len(np.unique(val_idx)) == 10000, "Val centers must be unique"
    assert val_idx.min() >= 0 and val_idx.max() < adata.n_obs, "Val indices out of bounds"
    assert np.array_equal(adata.obs_names[val_idx].astype(str), val_barcodes), "Val barcode mismatch"
    assert (adata.obs["split"].iloc[val_idx] == "validation").all(), "Val cells must have split == 'validation'"
    assert adata.obs["donor_id"].iloc[val_idx].nunique() == 1, "Val cells must be from Lung5 only"
    print("    Verified 10,000 Validation centers row-mapping invariants.")

    # 5. Verify Test Cohort Row Mapping
    print(" -> [5/6] Checking Test cohort alignment ...")
    test_idx = np.array(test_cohort["global_indices"], dtype=np.int64)
    test_barcodes = np.array(test_cohort["cell_barcodes"], dtype=str)
    assert len(test_idx) == 10000, f"Expected 10000 test centers, got {len(test_idx)}"
    assert len(np.unique(test_idx)) == 10000, "Test centers must be unique"
    assert test_idx.min() >= 0 and test_idx.max() < adata.n_obs, "Test indices out of bounds"
    assert np.array_equal(adata.obs_names[test_idx].astype(str), test_barcodes), "Test barcode mismatch"
    assert (adata.obs["split"].iloc[test_idx] == "test").all(), "Test cells must have split == 'test'"
    assert adata.obs["donor_id"].iloc[test_idx].nunique() == 1, "Test cells must be from Lung9 only"
    print("    Verified 10,000 Test centers row-mapping invariants.")

    # 6. Verify Bootstrap Folds
    print(" -> [6/6] Checking 10-Fold Bootstrap Partition ...")
    assert len(val_folds) == 10, f"Expected 10 folds, got {len(val_folds)}"
    for fn, fd in val_folds.items():
        f_idx = np.array(fd["indices_in_val_centers"], dtype=np.int64)
        assert len(f_idx) == 8000, f"Fold {fn} expected 8000 indices, got {len(f_idx)}"
        assert len(np.unique(f_idx)) == 8000, f"Fold {fn} indices must be unique"
        assert f_idx.min() >= 0 and f_idx.max() < 10000, f"Fold {fn} index out of [0, 9999]"
    print("    Verified 10 Validation folds (8,000 unique indices in [0, 9999]).")

    print("=" * 80)
    print("PREFLIGHT PASSED: ALL 6 CRYPTOGRAPHIC & ROW-MAPPING CONTRACTS SATISFIED!")
    print("=" * 80 + "\n")
    return h5ad_sha


def prepare_unsupervised_adata(adata: ad.AnnData) -> ad.AnnData:
    """
    Physically drops all niche and cluster labels from AnnData so that adapters
    have ZERO access to reference ground-truth labels during fitting.
    """
    adata_fit = adata.copy()
    label_cols = ["niche", "cluster_20", "cluster_3", "cluster_8", "reference_niche", "cell_type"]
    cols_to_drop = [c for c in label_cols if c in adata_fit.obs.columns]
    if cols_to_drop:
        adata_fit.obs.drop(columns=cols_to_drop, inplace=True)
    return adata_fit


# ==============================================================================
# RESOLUTION SELECTION & EVALUATION
# ==============================================================================

def select_unsupervised_resolution(
    val_emb: np.ndarray,
    bootstrap_folds: Dict[str, Any],
    resolutions: List[float] = FROZEN_RESOLUTIONS,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    seed: int = 42
) -> Tuple[float, Dict[str, Any]]:
    """Selects optimal Leiden resolution r* for a single seed via unsupervised stability on 10 folds."""
    fold_names = [f"fold_{i}" for i in range(len(bootstrap_folds))]
    fold_indices_list = [np.array(bootstrap_folds[fname]["indices_in_val_centers"], dtype=np.int64) for fname in fold_names]
    n_bootstraps = len(fold_indices_list)

    scores_per_res: Dict[float, float] = {}
    mean_k_per_res: Dict[float, float] = {}

    for res in resolutions:
        clusterings: List[np.ndarray] = []
        for b in range(n_bootstraps):
            sub_idx = fold_indices_list[b]
            z_sub = val_emb[sub_idx]

            adata_sub = ad.AnnData(X=z_sub.copy())
            sc.pp.neighbors(adata_sub, n_neighbors=15, use_rep="X")
            sc.tl.leiden(adata_sub, resolution=res, key_added="leiden", random_state=seed + b)
            clusterings.append(adata_sub.obs["leiden"].to_numpy().astype(int))

        n_clusters_list = [len(np.unique(c)) for c in clusterings]
        mean_k = float(np.mean(n_clusters_list))
        mean_k_per_res[res] = mean_k

        pair_aris: List[float] = []
        for i in range(n_bootstraps):
            for j in range(i + 1, n_bootstraps):
                idx_i = fold_indices_list[i]
                idx_j = fold_indices_list[j]
                common_cells, pos_i, pos_j = np.intersect1d(idx_i, idx_j, return_indices=True)
                if len(common_cells) > 10:
                    c_i = clusterings[i][pos_i]
                    c_j = clusterings[j][pos_j]
                    pair_aris.append(float(adjusted_rand_score(c_i, c_j)))

        scores_per_res[res] = float(np.mean(pair_aris)) if pair_aris else 0.0

    valid_res = [r for r in resolutions if k_min <= mean_k_per_res[r] <= k_max]
    if valid_res:
        best_res = max(valid_res, key=lambda r: (scores_per_res[r], -r))
    else:
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
        "valid_resolutions": valid_res
    }
    return float(best_res), report


def evaluate_test_clustering(
    test_emb: np.ndarray,
    frozen_r_star: float,
    test_cohort: Dict[str, Any],
    adata_orig: ad.AnnData,
    method_name: str,
    seed: int
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluates Test embedding at frozen r*, exports assignments and donor metrics."""
    meta = METHOD_METADATA[method_name]
    test_global_indices = np.array(test_cohort["global_indices"], dtype=np.int64)
    test_barcodes = [test_cohort["cell_barcodes"][i] for i in range(len(test_global_indices))]
    test_donors = adata_orig.obs["donor_id"].iloc[test_global_indices].to_numpy()
    test_sample_ids = adata_orig.obs["sample_id"].iloc[test_global_indices].to_numpy()
    test_true_labels = adata_orig.obs["niche"].iloc[test_global_indices].to_numpy()

    adata_t = ad.AnnData(X=test_emb.copy())
    sc.pp.neighbors(adata_t, n_neighbors=15, use_rep="X", random_state=seed)
    sc.tl.leiden(adata_t, resolution=frozen_r_star, key_added="pred", random_state=seed)
    pred_test = adata_t.obs["pred"].to_numpy().astype(int)

    # 1. Assignments DataFrame (Standardized 11 columns)
    df_assignments = pd.DataFrame({
        "cell_barcode": test_barcodes,
        "donor_id": test_donors,
        "sample_id": test_sample_ids,
        "reference_niche": test_true_labels,
        "predicted_cluster": pred_test,
        "method_name": method_name,
        "method_variant": meta["variant"],
        "paradigm": meta["paradigm"],
        "fit_data_scope": meta["fit_data_scope"],
        "seed": seed,
        "r_star": frozen_r_star
    })

    # 2. Metrics calculation
    ari = float(adjusted_rand_score(test_true_labels, pred_test))
    nmi = float(normalized_mutual_info_score(test_true_labels, pred_test))
    n_clusters = int(len(np.unique(pred_test)))

    # Compute Reference-Niche ASW (subsampling to 5000 for efficiency if large)
    try:
        if len(test_emb) > 5000:
            sub_idx = np.random.RandomState(seed).choice(len(test_emb), size=5000, replace=False)
            asw = float(silhouette_score(test_emb[sub_idx], test_true_labels[sub_idx]))
        else:
            asw = float(silhouette_score(test_emb, test_true_labels))
    except Exception as e:
        print(f"Warning: ASW computation failed ({e}), setting to 0.0")
        asw = 0.0

    metrics = {
        "method_name": method_name,
        "method_variant": meta["variant"],
        "paradigm": meta["paradigm"],
        "fit_data_scope": meta["fit_data_scope"],
        "seed": seed,
        "r_star": frozen_r_star,
        "n_clusters": n_clusters,
        "ari": ari,
        "nmi": nmi,
        "reference_niche_asw": asw
    }
    return metrics, df_assignments


# ==============================================================================
# BASELINE FEATURE EXTRACTORS (ADAPTERS RECEIVE UNLABELED adata_fit ONLY)
# ==============================================================================

def run_raw_pca(
    adata_fit: ad.AnnData,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    n_components: int = 50,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Method 1: Raw PCA on standardized expression (fit on Train+Val, transform Test)."""
    print(f"[{time.strftime('%H:%M:%S')}] Running Raw PCA (n_components={n_components}, seed={seed}) ...")
    ad_pca = adata_fit.copy()
    sc.pp.normalize_total(ad_pca, target_sum=1e4)
    sc.pp.log1p(ad_pca)

    X = ad_pca.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    pca = PCA(n_components=n_components, random_state=seed)
    train_val_mask = (ad_pca.obs["split"] != "test").to_numpy()
    pca.fit(X[train_val_mask])

    val_emb = pca.transform(X[val_indices]).astype(np.float32)
    test_emb = pca.transform(X[test_indices]).astype(np.float32)
    return val_emb, test_emb


_CACHED_MULTI_FREQ = None

def run_px_resdiff(
    adata_fit: ad.AnnData,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    n_components: int = 50,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Method 2: DualFreq baseline (Spatial convolution PX + high-pass residual H, fit on Train+Val)."""
    global _CACHED_MULTI_FREQ
    print(f"[{time.strftime('%H:%M:%S')}] Running DualFreq / px_resdiff (R=50um spatial smoothing, seed={seed}) ...")

    if _CACHED_MULTI_FREQ is None:
        print(f"[{time.strftime('%H:%M:%S')}] Computing spatial diffusion PX and high-pass residual H on full cohort ...")
        ad_work = adata_fit.copy()
        sc.pp.normalize_total(ad_work, target_sum=1e4)
        sc.pp.log1p(ad_work)

        coords = ad_work.obsm["spatial_um"].astype(np.float32)
        sample_ids = ad_work.obs["sample_id"].to_numpy()

        # Build sample-isolated spatial graph with micron coordinates
        P, A = build_global_spatial_graph(coords=coords, sample_ids=sample_ids, radius=50.0, sigma=16.6667, max_neighbors=None)

        X = ad_work.X
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        PX = P.dot(X).astype(np.float32)
        H = X - PX
        _CACHED_MULTI_FREQ = np.concatenate([PX, H], axis=1)
        print(f"[{time.strftime('%H:%M:%S')}] Cached multi-frequency features: shape={_CACHED_MULTI_FREQ.shape}")

    multi_freq_features = _CACHED_MULTI_FREQ
    train_val_mask = (adata_fit.obs["split"] != "test").to_numpy()
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(multi_freq_features[train_val_mask])

    val_emb = pca.transform(multi_freq_features[val_indices]).astype(np.float32)
    test_emb = pca.transform(multi_freq_features[test_indices]).astype(np.float32)
    return val_emb, test_emb


def run_banksy(
    adata_fit: ad.AnnData,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Method 3: Official BANKSY-Controlled adapter (Transductive, R=50um)."""
    print(f"[{time.strftime('%H:%M:%S')}] Running BANKSY adapter (seed={seed}) ...")
    from src.benchmark.adapters.banksy import OfficialBANKSYAdapter

    adapter = OfficialBANKSYAdapter(
        protocol="controlled",
        lambda_param=0.2,
        pca_dim=20,
        return_diagnostics=False
    )
    res = adapter.fit_transform(adata_fit, seed=seed, embedding_only=True)
    full_emb = np.asarray(res.embedding, dtype=np.float32)
    val_emb = full_emb[val_indices]
    test_emb = full_emb[test_indices]
    return val_emb, test_emb


def run_cellcharter(
    adata_fit: ad.AnnData,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Method 4: Official CellCharter-PCA-Controlled adapter (Transductive, R=50um)."""
    print(f"[{time.strftime('%H:%M:%S')}] Running CellCharter-PCA-Controlled adapter (seed={seed}) ...")
    from src.benchmark.adapters.cellcharter import ControlledCellCharterAdapter

    adapter = ControlledCellCharterAdapter(
        protocol="controlled",
        sample_key="sample_id",
        pca_dim=50,
        n_layers=3,
        radius=50.0
    )
    res = adapter.fit_transform(adata_fit, seed=seed, embedding_only=True)
    full_emb = np.asarray(res.embedding, dtype=np.float32)
    val_emb = full_emb[val_indices]
    test_emb = full_emb[test_indices]
    return val_emb, test_emb


def run_cellniche(
    adata_fit: ad.AnnData,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Method 5: Official CellNiche adapter (Transductive, R=50um)."""
    print(f"[{time.strftime('%H:%M:%S')}] Running CellNiche adapter (seed={seed}) ...")
    from src.benchmark.adapters.cellniche import OfficialCellNicheAdapter

    adapter = OfficialCellNicheAdapter(
        protocol="controlled",
        sample_key="sample_id",
        radius=50.0,
        epochs=5,
        batch_size=512
    )
    res = adapter.fit_transform(adata_fit, seed=seed, embedding_only=True)
    full_emb = np.asarray(res.embedding, dtype=np.float32)
    val_emb = full_emb[val_indices]
    test_emb = full_emb[test_indices]
    return val_emb, test_emb


def run_stagate(
    adata_fit: ad.AnnData,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Method 6: Official STAGATE adapter (Transductive, R=50um)."""
    print(f"[{time.strftime('%H:%M:%S')}] Running STAGATE adapter (seed={seed}) ...")
    from src.benchmark.adapters.stagate import OfficialSTAGATEAdapter

    adapter = OfficialSTAGATEAdapter(
        protocol="controlled",
        sample_key="sample_id",
        radius=50.0
    )
    res = adapter.fit_transform(adata_fit, seed=seed, embedding_only=True)
    full_emb = np.asarray(res.embedding, dtype=np.float32)
    val_emb = full_emb[val_indices]
    test_emb = full_emb[test_indices]
    return val_emb, test_emb


METHOD_RUNNERS = {
    "raw_pca": run_raw_pca,
    "px_resdiff": run_px_resdiff,
    "banksy": run_banksy,
    "cellcharter": run_cellcharter,
    "cellniche": run_cellniche,
    "stagate": run_stagate
}


# ==============================================================================
# TRULY LIGHTWEIGHT SMOKE TEST (ISOLATED 1000-CELL SUBSET)
# ==============================================================================

def create_smoke_dataset(adata_fit: ad.AnnData) -> Tuple[ad.AnnData, np.ndarray, np.ndarray]:
    """
    Constructs a truly lightweight AnnData across 2 physical samples (500 cells each, 1,000 total).
    Returns (adata_smoke, local_val_indices, local_test_indices).
    """
    print("[SMOKE SETUP] Sampling 500 cells from a Val sample and 500 cells from a Test sample ...")
    val_samples = adata_fit.obs[adata_fit.obs["split"] == "validation"]["sample_id"].unique()
    test_samples = adata_fit.obs[adata_fit.obs["split"] == "test"]["sample_id"].unique()

    s_val = val_samples[0]
    s_test = test_samples[0]

    val_subset_idx = np.where(adata_fit.obs["sample_id"] == s_val)[0][:500]
    test_subset_idx = np.where(adata_fit.obs["sample_id"] == s_test)[0][:500]

    smoke_indices = np.concatenate([val_subset_idx, test_subset_idx])
    adata_smoke = adata_fit[smoke_indices].copy()

    # Re-index locally inside adata_smoke
    local_val_idx = np.arange(0, 500, dtype=np.int64)[:100]  # Take 100 as val centers
    local_test_idx = np.arange(500, 1000, dtype=np.int64)[:100]  # Take 100 as test centers

    print(f"[SMOKE SETUP] Created adata_smoke: shape={adata_smoke.shape}, samples={[s_val, s_test]}")
    return adata_smoke, local_val_idx, local_test_idx


def run_smoke_test(adata_smoke: ad.AnnData, local_val: np.ndarray, local_test: np.ndarray, method_name: str) -> bool:
    """Executes a fast smoke test on adata_smoke."""
    print(f"\n[SMOKE] Testing method: {method_name} ...")
    runner = METHOD_RUNNERS[method_name]

    t0 = time.time()
    try:
        val_emb, test_emb = runner(adata_smoke, local_val, local_test, seed=42)
        assert val_emb.shape[0] == len(local_val), f"Expected {len(local_val)} val samples, got {val_emb.shape[0]}"
        assert test_emb.shape[0] == len(local_test), f"Expected {len(local_test)} test samples, got {test_emb.shape[0]}"
        assert np.isfinite(val_emb).all(), "val_emb contains NaN or Inf!"
        assert np.isfinite(test_emb).all(), "test_emb contains NaN or Inf!"
        print(f"[SMOKE PASS] {method_name} passed smoke test in {time.time() - t0:.2f}s! (emb_dim={val_emb.shape[1]})")
        return True
    except Exception as e:
        print(f"[SMOKE FAILED] {method_name} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==============================================================================
# MAIN ORCHESTRATOR (FAIL CLOSED)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="W2.6 CosMx NSCLC Baseline Benchmark Execution")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["raw_pca", "px_resdiff", "banksy", "cellcharter", "cellniche"],
        help="List of methods to run: raw_pca, px_resdiff, banksy, cellcharter, cellniche, stagate"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Random seeds")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test for specified methods on 1,000-cell subset")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly overwrite existing benchmark outputs")
    parser.add_argument("--resume", action="store_true", help="Safely resume and reuse verified existing outputs")
    args = parser.parse_args()

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    ASSIGN_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("W2.6 COSMX NSCLC BASELINE BENCHMARK EXECUTION")
    print(f"Methods: {args.methods}")
    print(f"Seeds: {args.seeds}")
    print(f"Smoke Mode: {args.smoke}")
    print(f"Overwrite: {args.overwrite} | Resume: {args.resume}")
    print(f"Dataset: {DATASET_PATH}")
    print("=" * 80)

    # 1. Load Datasets and Cohort Manifests
    print(f"[{time.strftime('%H:%M:%S')}] Loading AnnData ...")
    adata_orig = ad.read_h5ad(DATASET_PATH)
    print(f"Loaded AnnData: shape={adata_orig.shape}, genes={adata_orig.n_vars}")

    with open(VAL_CENTERS_PATH, "r", encoding="utf-8") as f:
        val_cohort = json.load(f)
    with open(VAL_FOLDS_PATH, "r", encoding="utf-8") as f:
        val_folds = json.load(f)
    with open(TEST_CENTERS_PATH, "r", encoding="utf-8") as f:
        test_cohort = json.load(f)

    # 2. Strict Preflight Integrity Check (Mandatory!)
    h5ad_sha = verify_cosmx_baseline_preflight(adata_orig, val_cohort, test_cohort, val_folds)

    # 3. Physically strip labels for unsupervised adapters
    print(f"[{time.strftime('%H:%M:%S')}] Physically stripping niche/cluster labels for adapter safety ...")
    adata_fit = prepare_unsupervised_adata(adata_orig)
    print(f"Remaining columns in adata_fit.obs: {list(adata_fit.obs.columns)}")

    # 4. Handle Smoke Mode (Uses small isolated adata_smoke)
    if args.smoke:
        adata_smoke, local_val, local_test = create_smoke_dataset(adata_fit)
        all_passed = True
        for m in args.methods:
            if m not in METHOD_RUNNERS:
                print(f"Unknown method: {m}")
                continue
            passed = run_smoke_test(adata_smoke, local_val, local_test, m)
            if not passed:
                all_passed = False
        print("\n" + "=" * 80)
        print(f"SMOKE TEST SUMMARY: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        print("=" * 80)
        sys.exit(0 if all_passed else 1)

    val_global_indices = np.array(val_cohort["global_indices"], dtype=np.int64)
    test_global_indices = np.array(test_cohort["global_indices"], dtype=np.int64)
    script_sha = compute_file_sha256(Path(__file__).resolve())

    # 5. Run Methods with Strict Fail-Closed Guards
    for method_name in args.methods:
        if method_name not in METHOD_RUNNERS:
            print(f"Skipping unknown method: {method_name}")
            continue

        runner = METHOD_RUNNERS[method_name]
        meta = METHOD_METADATA[method_name]
        summary_path = RES_DIR / f"{method_name}_benchmark_summary.json"

        # Check for any lingering output files (summary, embeddings, assignments)
        existing_artifacts = []
        if summary_path.exists():
            existing_artifacts.append(str(summary_path))
        for s in args.seeds:
            v_p = EMB_DIR / f"{method_name}_val_seed_{s}.npy"
            t_p = EMB_DIR / f"{method_name}_test_seed_{s}.npy"
            a_p = ASSIGN_DIR / f"{method_name}_test_assignments_seed_{s}.csv"
            if v_p.exists(): existing_artifacts.append(str(v_p))
            if t_p.exists(): existing_artifacts.append(str(t_p))
            if a_p.exists(): existing_artifacts.append(str(a_p))

        if existing_artifacts and not args.overwrite and not args.resume:
            raise FileExistsError(
                f"Fail Closed: Existing artifact(s) found for '{method_name}': {existing_artifacts[:3]}... "
                "Execution refused by default. Specify --overwrite to recalculate or --resume to reuse verified outputs."
            )

        print("\n" + "#" * 80)
        print(f"STARTING METHOD: {method_name.upper()} ({meta['variant']})")
        print(f"Paradigm: {meta['paradigm']} | Scope: {meta['fit_data_scope']}")
        print("#" * 80)

        # Load existing summary if resuming
        existing_summary = None
        if args.resume and summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    existing_summary = json.load(f)
                assert existing_summary.get("h5ad_sha256") == EXPECTED_H5AD_SHA256, "Manifest H5AD SHA mismatch in existing summary"
            except Exception as e:
                print(f"[RESUME WARNING] Existing summary at {summary_path} invalid ({e}). Will re-run invalid seeds.")
                existing_summary = None

        seed_results = {}
        aris = []
        nmis = []
        asws = []
        ks = []
        locked_r_stars = {}

        for seed in args.seeds:
            print(f"\n--- Method: {method_name} | Seed: {seed} ---")
            val_emb_path = EMB_DIR / f"{method_name}_val_seed_{seed}.npy"
            test_emb_path = EMB_DIR / f"{method_name}_test_seed_{seed}.npy"
            assign_path = ASSIGN_DIR / f"{method_name}_test_assignments_seed_{seed}.csv"

            can_reuse = False
            if args.resume and existing_summary and str(seed) in existing_summary.get("seed_results", {}):
                s_info = existing_summary["seed_results"][str(seed)]
                if val_emb_path.exists() and test_emb_path.exists() and assign_path.exists():
                    val_sha = compute_file_sha256(val_emb_path)
                    test_sha = compute_file_sha256(test_emb_path)
                    if (val_sha == s_info.get("val_embedding_sha256") and
                            test_sha == s_info.get("test_embedding_sha256")):
                        val_emb = np.load(val_emb_path)
                        test_emb = np.load(test_emb_path)
                        if (val_emb.shape == (10000, val_emb.shape[1]) and
                                test_emb.shape == (10000, test_emb.shape[1]) and
                                np.isfinite(val_emb).all() and np.isfinite(test_emb).all()):
                            print(f"[RESUME VERIFIED] All cryptographic and shape assertions passed for Seed {seed}.")
                            test_metrics = s_info["test_metrics"]
                            sel_report = s_info["selection_report"]
                            r_star = s_info["r_star"]
                            can_reuse = True

            if not can_reuse:
                t0 = time.time()
                val_emb, test_emb = runner(adata_fit, val_global_indices, test_global_indices, seed=seed)
                print(f"Extracted embeddings in {time.time() - t0:.1f}s | Val={val_emb.shape}, Test={test_emb.shape}")
                assert np.isfinite(val_emb).all(), "val_emb contains NaN or Inf!"
                assert np.isfinite(test_emb).all(), "test_emb contains NaN or Inf!"

                # Save embeddings atomically
                tmp_val = val_emb_path.with_suffix(".tmp.npy")
                tmp_test = test_emb_path.with_suffix(".tmp.npy")
                np.save(tmp_val, val_emb)
                np.save(tmp_test, test_emb)
                os.replace(tmp_val, val_emb_path)
                os.replace(tmp_test, test_emb_path)

                # Phase 1: Unsupervised Resolution Selection
                print(f"Selecting optimal r* on 10 Validation folds (Seed {seed}) ...")
                r_star, sel_report = select_unsupervised_resolution(
                    val_emb=val_emb,
                    bootstrap_folds=val_folds,
                    seed=seed
                )
                print(f"Selected r*={r_star:.1f} (Stability={sel_report['best_stability_score']:.4f})")

                # Phase 2: Single-pass Test Blind Evaluation (Reads ground-truth from adata_orig)
                print(f"Evaluating Test clustering at frozen r*={r_star:.1f} ...")
                test_metrics, df_assign = evaluate_test_clustering(
                    test_emb=test_emb,
                    frozen_r_star=r_star,
                    test_cohort=test_cohort,
                    adata_orig=adata_orig,
                    method_name=method_name,
                    seed=seed
                )
                print(f"Seed {seed} Test Results: ARI={test_metrics['ari']:.4f} | NMI={test_metrics['nmi']:.4f} | "
                      f"Ref-Niche ASW={test_metrics['reference_niche_asw']:.4f} | K={test_metrics['n_clusters']}")

                # Save assignments CSV atomically
                tmp_assign = assign_path.with_suffix(".tmp.csv")
                df_assign.to_csv(tmp_assign, index=False)
                os.replace(tmp_assign, assign_path)

            locked_r_stars[str(seed)] = r_star
            seed_results[str(seed)] = {
                "r_star": r_star,
                "selection_report": sel_report,
                "test_metrics": test_metrics,
                "val_embedding_sha256": compute_file_sha256(val_emb_path),
                "test_embedding_sha256": compute_file_sha256(test_emb_path),
                "test_assignment_sha256": compute_file_sha256(assign_path)
            }
            aris.append(test_metrics["ari"])
            nmis.append(test_metrics["nmi"])
            asws.append(test_metrics["reference_niche_asw"])
            ks.append(test_metrics["n_clusters"])

        # Compute 3-seed summary statistics (ddof=1)
        summary = {
            "method_name": method_name,
            "method_variant": meta["variant"],
            "paradigm": meta["paradigm"],
            "fit_data_scope": meta["fit_data_scope"],
            "description": meta["description"],
            "dataset": "cosmx_human_nsclc",
            "h5ad_sha256": h5ad_sha,
            "script_sha256": script_sha,
            "mean_ari": float(np.mean(aris)),
            "std_ari": float(np.std(aris, ddof=1)) if len(aris) > 1 else 0.0,
            "mean_nmi": float(np.mean(nmis)),
            "std_nmi": float(np.std(nmis, ddof=1)) if len(nmis) > 1 else 0.0,
            "mean_reference_niche_asw": float(np.mean(asws)),
            "std_reference_niche_asw": float(np.std(asws, ddof=1)) if len(asws) > 1 else 0.0,
            "mean_clusters": float(np.mean(ks)),
            "std_clusters": float(np.std(ks, ddof=1)) if len(ks) > 1 else 0.0,
            "locked_r_stars": locked_r_stars,
            "seed_results": seed_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save summary JSON atomically
        tmp_summary = summary_path.with_suffix(".tmp.json")
        with open(tmp_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        os.replace(tmp_summary, summary_path)

        print("\n" + "=" * 80)
        print(f"SUMMARY FOR {meta['variant']} ({meta['paradigm']}):")
        print(f"ARI: {summary['mean_ari']:.4f} ± {summary['std_ari']:.4f}")
        print(f"NMI: {summary['mean_nmi']:.4f} ± {summary['std_nmi']:.4f}")
        print(f"Reference-Niche ASW: {summary['mean_reference_niche_asw']:.4f} ± {summary['std_reference_niche_asw']:.4f}")
        print(f"Clusters (K): {summary['mean_clusters']:.1f} ± {summary['std_clusters']:.1f}")
        print(f"Saved summary to {summary_path}")
        print("=" * 80)


if __name__ == "__main__":
    main()
