"""
Fit and Freeze Train-Only PCA Whitener for CosMx NSCLC (W2.6 Stage 2).

Strictly enforces:
1. Input high-frequency residuals H = X - PX:
   - Evaluated on Train context (242,515 cells) using frozen R=50um P operator
   - Extracted strictly for the 60,000 frozen Train centers
2. Fitting specification:
   - TrainOnlyWhitening(k_components=32, lambda_min=1e-4)
   - Fitted on H_train_centers (Shape: 60,000 x 960)
   - Whitener projection matrix: (960, 32)
3. Output persistence:
   - Save whitener parameters and state dict to outputs/runs/freqniche/cosmx_nsclc/2026-09-17/20260917_cosmx_nsclc_final/whitener.pt
   - Compute and log SHA-256 signatures in whitener_manifest.json
"""

from __future__ import annotations

import os
import hashlib
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.freqniche_st.whitening import TrainOnlyWhitening
from src.freqniche_st.runtime_paths import resolve_dataset_path, resolve_run_output

DATASET_PATH = resolve_dataset_path("FREQNICHE_DATASET_PATH", "datasets/h5ad/cosmx_human_nsclc/cosmx_human_nsclc.h5ad")
OUTPUT_DIR = resolve_run_output("outputs/runs/freqniche/cosmx_nsclc/2026-09-17/20260917_cosmx_nsclc_final")
CACHE_DIR = OUTPUT_DIR / "cache"
TRAIN_CENTERS_PATH = OUTPUT_DIR / "train_centers.json"
TRAIN_P_PATH = CACHE_DIR / "cosmx_train_graph_P.npz"
WHITENER_OUT_PATH = OUTPUT_DIR / "whitener.pt"
WHITENER_MANIFEST_PATH = OUTPUT_DIR / "whitener_manifest.json"


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    print("=" * 85)
    print("W2.6 STAGE 2: COSMX NSCLC TRAIN-ONLY RESIDUAL WHITENER FITTING (k=32)")
    print("=" * 85)

    assert TRAIN_CENTERS_PATH.exists(), f"Missing train centers: {TRAIN_CENTERS_PATH}"
    assert TRAIN_P_PATH.exists(), f"Missing cached train P operator: {TRAIN_P_PATH}"

    # 1. Load Train centers
    with open(TRAIN_CENTERS_PATH, "r", encoding="utf-8") as f:
        train_cohort = json.load(f)

    train_global_indices = train_cohort["global_indices"]
    n_centers = len(train_global_indices)
    assert n_centers == 60000, f"Expected 60,000 train centers, got {n_centers}"
    print(f"Loaded Train centers: {n_centers:,} centers.")

    # 2. Load Train context expression
    print(f"\nLoading AnnData from {DATASET_PATH} ...")
    adata = ad.read_h5ad(DATASET_PATH)
    train_mask = (adata.obs["split"] == "train").to_numpy()
    train_indices = np.where(train_mask)[0]
    n_train_cells = len(train_indices)
    assert n_train_cells == 242515, f"Expected 242,515 train cells, got {n_train_cells}"

    print("Slicing Train context AnnData and applying normalize_total(1e4) + log1p() ...")
    adata_train = adata[train_indices].copy()
    sc.pp.normalize_total(adata_train, target_sum=1e4)
    sc.pp.log1p(adata_train)

    X_train_full = adata_train.X
    if sp.issparse(X_train_full):
        X_train_full = X_train_full.toarray()
    X_train_full = np.asarray(X_train_full, dtype=np.float32)
    assert np.isfinite(X_train_full).all(), "Non-finite values found in Train expression!"

    # 3. Load Cached P operator & Compute Residuals H = X - PX
    print(f"\nLoading cached Train diffusion operator P from {TRAIN_P_PATH} ...")
    P_train = sp.load_npz(TRAIN_P_PATH)
    assert P_train.shape == (n_train_cells, n_train_cells), f"Unexpected P shape: {P_train.shape}"

    # Map global center indices to train-local indices
    global_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(train_indices)}
    train_center_local_indices = np.array([global_to_local[g] for g in train_global_indices], dtype=np.int64)

    print("Computing smoothed expression PX for the 60,000 Train centers ...")
    P_centers = P_train[train_center_local_indices]
    PX_centers = (P_centers @ X_train_full).astype(np.float32)

    X_centers = X_train_full[train_center_local_indices]
    H_centers = X_centers - PX_centers
    assert H_centers.shape == (60000, 960), f"Unexpected H_centers shape: {H_centers.shape}"
    assert np.isfinite(H_centers).all(), "Non-finite values found in Train center residuals H!"

    print(f"H_centers computed: shape={H_centers.shape}, min={H_centers.min():.4f}, max={H_centers.max():.4f}, mean={H_centers.mean():.4f}")

    # 4. Fit Train-only Whitener (k_components=32)
    print("\nFitting TrainOnlyWhitening(k_components=32, lambda_min=1e-4) ...")
    whitener = TrainOnlyWhitening(k_components=32, lambda_min=1e-4)
    H_torch = torch.from_numpy(H_centers)
    whitener.fit(H_torch)

    # Verification
    H_white = whitener.transform(H_torch)
    assert H_white.shape == (60000, 32), f"Unexpected whitened shape: {H_white.shape}"
    cov_white = np.cov(H_white.T)
    diag_cov = np.diag(cov_white)
    print(f"Whitened target checks: shape={H_white.shape}")
    print(f" - Mean diagonal variance: {diag_cov.mean():.4f} (target ~1.0)")
    print(f" - Min variance: {diag_cov.min():.4f}, Max variance: {diag_cov.max():.4f}")

    # 5. Persist Whitener Checkpoint
    whitener_state = {
        "dataset_id": "cosmx_human_nsclc",
        "whitener_class": "TrainOnlyWhitening",
        "fitted_on": "high_frequency_residual_H_equals_X_minus_PX",
        "n_samples_fitted": 60000,
        "n_features": 960,
        "k_components": 32,
        "lambda_min": 1e-4,
        "mean_": whitener.mean_,
        "components_": whitener.components_,
        "singular_values_": whitener.singular_values_,
        "whitening_matrix_": whitener.whitening_matrix_,
        "n_components_fitted_": whitener.n_components_fitted_,
        "in_dim": 960,
        "out_dim": 32,
    }
    torch.save(whitener_state, WHITENER_OUT_PATH)
    sha256_hash = compute_file_sha256(WHITENER_OUT_PATH)
    print(f"[OK] Persisted CosMx Whitener to: {WHITENER_OUT_PATH}")
    print(f" - SHA-256: {sha256_hash}")

    h_input_sha256 = hashlib.sha256(np.ascontiguousarray(H_centers).tobytes()).hexdigest()

    manifest_payload = {
        "whitener_file": str(WHITENER_OUT_PATH.name),
        "sha256": sha256_hash,
        "in_features": 960,
        "k_components": 32,
        "lambda_min": 1e-4,
        "n_train_centers_fitted": 60000,
        "mean_var": float(diag_cov.mean()),
        "h_input_sha256": h_input_sha256,
    }
    with open(WHITENER_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)
    print(f"[OK] Whitener manifest written to: {WHITENER_MANIFEST_PATH}")

    # Also update master cosmx_data_audit_manifest.json atomically
    master_manifest_path = OUTPUT_DIR / "cosmx_data_audit_manifest.json"
    if master_manifest_path.exists():
        with open(master_manifest_path, "r", encoding="utf-8") as f:
            master = json.load(f)
        master["whitener"] = {
            "file": "whitener.pt",
            "in_features": 960,
            "k_target": 32,
            "lambda_min": 1e-4,
            "sha256": sha256_hash,
            "mean_var": float(diag_cov.mean()),
            "h_input_sha256": h_input_sha256,
        }
        temp_master = master_manifest_path.with_suffix(".tmp.json")
        with open(temp_master, "w", encoding="utf-8") as f:
            json.dump(master, f, indent=2)
        os.replace(temp_master, master_manifest_path)
        print(f"[OK] Master audit manifest updated atomically with Whitener contract at: {master_manifest_path}")

    print("=" * 85)


if __name__ == "__main__":
    main()
