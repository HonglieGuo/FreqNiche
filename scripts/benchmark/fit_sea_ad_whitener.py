"""
Fit and Freeze Train-Only PCA Whitener for SEA-AD (Stage 2).

Strictly enforces:
1. Input high-frequency residuals H = X - PX:
   - Evaluated on Train context (1.21M cells) using frozen R=50um P operator
   - Extracted ONLY for the 68,940 frozen Train centers
2. Fitting specification:
   - TrainOnlyWhitening(k_components=32, lambda_min=1e-4)
   - Fitted on H_train_centers (Shape: 68940 x 180)
   - Zero raw X direct whitening (strictly residual target whitening)
3. Output persistence:
   - Save whitener parameters and state dict
   - Compute and log SHA-256 signatures
"""

import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.freqniche_st.whitening import TrainOnlyWhitening, TrainOnlyPCA
from src.freqniche_st.runtime_paths import resolve_dataset_path, resolve_run_output


DATASET_PATH = resolve_dataset_path("FREQNICHE_DATASET_PATH", "datasets/h5ad/sea_ad/sea_ad_mtg_merfish.h5ad")
OUTPUT_DIR = resolve_run_output("outputs/runs/freqniche/sea_ad/2026-09-15/20260915_sea_ad_final")
CACHE_DIR = OUTPUT_DIR / "cache"
MANIFEST_PATH = OUTPUT_DIR / "cohort_manifest.json"
TRAIN_CENTERS_PATH = OUTPUT_DIR / "train_centers.json"
TRAIN_P_PATH = CACHE_DIR / "train_graph_P.npz"


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    print("=" * 80)
    print("STAGE 2: SEA-AD TRAIN-ONLY RESIDUAL WHITENER FITTING")
    print("=" * 80)

    # 1. Assert pre-requisites
    assert MANIFEST_PATH.exists(), f"Missing cohort manifest: {MANIFEST_PATH}"
    assert TRAIN_CENTERS_PATH.exists(), f"Missing train centers: {TRAIN_CENTERS_PATH}"
    assert TRAIN_P_PATH.exists(), f"Missing cached train P operator: {TRAIN_P_PATH}"

    with open(TRAIN_CENTERS_PATH, "r", encoding="utf-8") as f:
        train_cohort = json.load(f)

    train_barcodes = train_cohort["cell_barcodes"]
    train_global_indices = train_cohort["global_indices"]
    n_centers = len(train_barcodes)
    assert n_centers == 68940, f"Expected 68,940 train centers, got {n_centers}"
    print(f"Loaded train centers: {n_centers:,} centers.")

    # 2. Load Train context expression
    print(f"Loading AnnData from {DATASET_PATH} ...")
    adata = sc.read_h5ad(DATASET_PATH)
    train_mask = (adata.obs["split"] == "train").to_numpy()
    train_indices = np.where(train_mask)[0]
    n_train_cells = len(train_indices)
    assert n_train_cells == 1210773, f"Expected 1,210,773 train cells, got {n_train_cells}"

    print("Slicing Train context AnnData and applying normalize_total(1e4) + log1p() ...")
    adata_train = adata[train_indices].copy()
    sc.pp.normalize_total(adata_train, target_sum=1e4)
    sc.pp.log1p(adata_train)

    X_train_full = adata_train.X
    if sp.issparse(X_train_full):
        X_train_full = X_train_full.toarray()
    X_train_full = np.asarray(X_train_full, dtype=np.float32)

    # 3. Load Cached P operator & Compute Residuals H = X - PX
    print(f"Loading cached Train diffusion operator P from {TRAIN_P_PATH} ...")
    P_train = sp.load_npz(TRAIN_P_PATH)
    assert P_train.shape == (n_train_cells, n_train_cells), f"Unexpected P shape: {P_train.shape}"

    # Map global center indices to train-local indices
    global_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(train_indices)}
    train_center_local_indices = np.array([global_to_local[g] for g in train_global_indices], dtype=np.int64)

    print("Computing smoothed expression PX for the 68,940 centers ...")
    # To be memory efficient: P_sub = P_train[train_center_local_indices]
    P_centers = P_train[train_center_local_indices]
    PX_centers = (P_centers @ X_train_full).astype(np.float32)

    X_centers = X_train_full[train_center_local_indices]
    H_centers = (X_centers - PX_centers).astype(np.float32)
    assert H_centers.shape == (68940, 180), f"Unexpected H_centers shape: {H_centers.shape}"
    print(f"Calculated high-frequency residuals H = X - PX. Shape: {H_centers.shape}")
    print(f"Residual statistics: mean={H_centers.mean():.4f}, std={H_centers.std():.4f}, min={H_centers.min():.4f}, max={H_centers.max():.4f}")

    # 4. Fit TrainOnlyWhitening on H_centers (Strict Contract: H = X - PX)
    print("\nFitting TrainOnlyWhitening(k_components=32, lambda_min=1e-4) on H_centers ...")
    whitener = TrainOnlyWhitening(k_components=32, lambda_min=1e-4)
    whitener.fit(H_centers)

    assert whitener.is_fitted, "Whitener fitting failed!"
    assert whitener.whitening_matrix_.shape == (180, 32), f"Unexpected whitening matrix shape: {whitener.whitening_matrix_.shape}"
    assert whitener.mean_.shape == (1, 180), f"Unexpected mean shape: {whitener.mean_.shape}"
    print("TrainOnlyWhitening successfully fitted.")

    # Also fit TrainOnlyPCA for cell identity branch
    print("Fitting TrainOnlyPCA(k_components=32) on X_centers ...")
    pca_id = TrainOnlyPCA(k_components=32)
    pca_id.fit(X_centers)
    assert pca_id.is_fitted, "TrainOnlyPCA fitting failed!"
    print("TrainOnlyPCA successfully fitted.")

    # 5. Save Whitener Parameters and State Dict
    whitener_save_path = OUTPUT_DIR / "whitener.pt"
    whitener_state = {
        "dataset_id": "sea_ad_mtg_merfish",
        "whitener_class": "TrainOnlyWhitening",
        "fitted_on": "high_frequency_residual_H_equals_X_minus_PX",
        "n_samples_fitted": 68940,
        "n_features": 180,
        "k_components": 32,
        "lambda_min": 1e-4,
        "mean_": whitener.mean_,
        "components_": whitener.components_,
        "singular_values_": whitener.singular_values_,
        "whitening_matrix_": whitener.whitening_matrix_,
        "n_components_fitted_": whitener.n_components_fitted_,
        "pca_id_mean_": pca_id.mean_,
        "pca_id_components_": pca_id.components_,
        "train_centers_sha256": train_cohort["cohort_sha256"]
    }
    torch.save(whitener_state, whitener_save_path)
    whitener_sha256 = compute_file_sha256(whitener_save_path)
    print(f"\nWhitener saved to: {whitener_save_path}")
    print(f"Whitener SHA-256: {whitener_sha256}")

    # 6. Update Master Cohort Manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest["whitener"] = {
        "file": str(whitener_save_path.relative_to(WORKSPACE_ROOT)),
        "sha256": whitener_sha256,
        "k_components": 32,
        "target_type": "high_frequency_residuals",
        "formula": "H = X - PX",
        "fit_samples": 68940
    }
    manifest["status"] = "STAGE2_WHITENER_FITTED_READY_FOR_SMOKE"

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Updated master manifest at: {MANIFEST_PATH}")
    print("=" * 80)
    print("STAGE 2 COMPLETE: Whitener fitted and verified strictly on residual targets.")
    print("=" * 80)


if __name__ == "__main__":
    main()
