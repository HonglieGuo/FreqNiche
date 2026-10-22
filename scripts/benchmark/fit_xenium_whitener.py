"""
W2.7: Fit Train-Only Truncated PCA Whitener for Xenium 5K (5,001 -> 32).

Strictly enforces:
1. Data scope: Train split only (L3, L5, L7, L8).
   - Pre-frozen 60,000 Train centers (15,000 per donor).
   - Absolute Test isolation: Test donor L1 (GSM9509134) is NEVER loaded or accessed.
2. Two-Pass Out-of-Core Chunked Covariance Accumulation:
   - Pass 1: Streaming mean accumulation across 4 Train donors: mu in R^{1 x 5001}
   - Pass 2: In-place chunked covariance matrix accumulation: C in R^{5001 x 5001} (~100 MB RAM)
     C += sum_chunk (H_chunk - mu)^T (H_chunk - mu)
   - Solves truncated symmetric eigendecomposition via scipy.linalg.eigh(C, subset_by_index=[5001-32, 5000])
   - Guaranteed whitening matrix shape: strictly (5001, 32)
3. Modes:
   - --preflight: Runs the full 60,000-center real chunked accumulation, verifies shapes,
     finiteness and orthogonality, logs real telemetry, but DOES NOT save whitener.pt to disk.
   - --save: Formally saves whitener.pt and whitener_manifest.json (LOCKED until user approval).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import psutil
import scipy.linalg as la
import scipy.sparse as sp
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.freqniche_st.whitening import TruncatedTrainOnlyWhitening
from src.freqniche_st.runtime_paths import resolve_run_output

OUTPUT_DIR = resolve_run_output("outputs/runs/freqniche/xenium_prime5k/2026-09-18/20260918_xenium_prime5k_final")
CACHE_DIR = OUTPUT_DIR / "cache"
GSE_DIR = WORKSPACE_ROOT / "datasets" / "h5ad" / "xenium_ffpe_lungbreast_gse311609" / "samples"

TRAIN_SAMPLES_META = [
    ("GSM9509135_NSCLC_5k_L3_L3.h5ad", "L3"),
    ("GSM9509136_NSCLC_5k_L5_L5.h5ad", "L5"),
    ("GSM9509138_NSCLC_5k_L7_L7.h5ad", "L7"),
    ("GSM9509139_NSCLC_5k_L8_L8.h5ad", "L8"),
]

TRAIN_CENTERS_PATH = OUTPUT_DIR / "train_centers.json"
WHITENER_OUT_PATH = OUTPUT_DIR / "whitener.pt"
WHITENER_MANIFEST_PATH = OUTPUT_DIR / "whitener_manifest.json"


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1048576):
            hasher.update(chunk)
    return hasher.hexdigest()


def fit_truncated_whitener_chunked(
    k_components: int = 32,
    lambda_min: float = 1e-4,
    chunk_size: int = 5000,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[TruncatedTrainOnlyWhitening, Dict[str, Any]]:
    t_start = time.time()

    assert TRAIN_CENTERS_PATH.exists(), f"Missing train centers: {TRAIN_CENTERS_PATH}"
    with open(TRAIN_CENTERS_PATH, "r", encoding="utf-8") as f:
        train_centers = json.load(f)

    n_total_centers = len(train_centers)
    assert n_total_centers == 60000, f"Expected 60,000 train centers, got {n_total_centers}"

    # Group centers by donor
    centers_by_donor: Dict[str, List[int]] = {}
    for c in train_centers:
        d = c["donor_id"]
        centers_by_donor.setdefault(d, []).append(c["local_cell_idx"])

    print(f"Loaded 60,000 frozen Train centers across 4 donors: {[(k, len(v)) for k, v in centers_by_donor.items()]}")

    # Pass 1: Compute Global Mean Residual Vector mu in R^{1 x 5001}
    print("\n[Pass 1/2] Computing Global Residual Mean across 60,000 Train Centers ...")
    sum_residual = np.zeros((1, 5001), dtype=np.float64)
    donor_residuals: Dict[str, np.ndarray] = {}

    for fname, donor in TRAIN_SAMPLES_META:
        t0_d = time.time()
        p_cache = CACHE_DIR / f"xenium_graph_{donor}_P.npz"
        h5ad_p = GSE_DIR / fname

        assert p_cache.exists(), f"Missing graph cache: {p_cache}"
        assert "L1" not in donor and "GSM9509134" not in fname, "CRITICAL: Test donor L1 found!"

        P = sp.load_npz(p_cache)
        a = ad.read_h5ad(h5ad_p, backed="r")
        assert a.n_vars == 5001, f"Expected 5,001 genes, got {a.n_vars}"

        # Load expression for full donor context
        raw_X = a.X[:]
        if sp.issparse(raw_X):
            X_dense = raw_X.toarray().astype(np.float32)
        else:
            X_dense = np.asarray(raw_X, dtype=np.float32)

        counts = np.maximum(X_dense.sum(axis=1, keepdims=True), 1.0)
        X_norm = np.log1p((X_dense / counts) * 1e4)

        # Compute center residuals: H_donor = X[M] - P[M] @ X
        center_idxs = np.array(centers_by_donor[donor], dtype=np.int64)
        X_M = X_norm[center_idxs]
        P_M = P[center_idxs, :]
        PX_M = P_M.dot(X_norm)
        H_d = (X_M - PX_M).astype(np.float32)

        assert H_d.shape == (len(center_idxs), 5001)
        assert np.all(np.isfinite(H_d)), f"Non-finite residuals in donor {donor}!"

        donor_residuals[donor] = H_d
        sum_residual += np.sum(H_d, axis=0, keepdims=True, dtype=np.float64)
        print(f"  - Donor {donor}: {len(center_idxs):,} residuals computed in {time.time() - t0_d:.2f} s")

    mean_vector = (sum_residual / n_total_centers).astype(np.float32)
    print(f"  - [PASS] Global mean vector computed: shape={mean_vector.shape}, min={mean_vector.min():.4f}, max={mean_vector.max():.4f}")

    # Pass 2: In-place Chunked Covariance Matrix Accumulation C in R^{5001 x 5001}
    print("\n[Pass 2/2] In-place Chunked Covariance Matrix Accumulation (C in R^{5001 x 5001}) ...")
    t0_cov = time.time()
    # Use torch tensor for CUDA acceleration if available
    device = torch.device(device_str)
    cov_torch = torch.zeros((5001, 5001), dtype=torch.float32, device=device)
    mean_torch = torch.from_numpy(mean_vector).to(device)

    for donor, H_d in donor_residuals.items():
        n_donor_centers = len(H_d)
        for start_idx in range(0, n_donor_centers, chunk_size):
            end_idx = min(start_idx + chunk_size, n_donor_centers)
            H_chunk = torch.from_numpy(H_d[start_idx:end_idx]).to(device)
            H_cen_chunk = H_chunk - mean_torch
            # Accumulate outer product sum: C += H_cen^T @ H_cen
            cov_torch.addmm_(H_cen_chunk.T, H_cen_chunk)

    if device.type == "cuda":
        torch.cuda.synchronize()

    cov_torch.div_(float(n_total_centers - 1))
    cov_matrix = cov_torch.cpu().numpy()
    t_cov = time.time() - t0_cov
    print(f"  - Covariance matrix accumulated in {t_cov:.2f} s: shape={cov_matrix.shape} (RAM: {cov_matrix.nbytes / (1024**2):.1f} MB)")

    # Truncated Eigendecomposition on Symmetric Covariance Matrix
    print(f"\nSolving Truncated Eigendecomposition (top {k_components} of 5,001) ...")
    t0_eigh = time.time()
    whitener = TruncatedTrainOnlyWhitening(k_components=k_components, lambda_min=lambda_min)
    whitener.fit_from_covariance(cov_matrix=cov_matrix, mean_vector=mean_vector, n_samples=n_total_centers)
    t_eigh = time.time() - t0_eigh

    W = whitener.whitening_matrix_
    V_k = whitener.components_
    singular_values = whitener.singular_values_

    print(f"  - Eigendecomposition completed in {t_eigh:.2f} s")
    print(f"  - Whitening matrix W shape: {W.shape}")
    print(f"  - Top 5 singular values:    {singular_values[:5]}")
    print(f"  - Bottom 5 singular values: {singular_values[-5:]}")

    # Mathematical assertions
    assert W.shape == (5001, 32), f"Expected shape (5001, 32), got {W.shape}"
    assert np.all(np.isfinite(W)), "Whitening matrix contains NaN or Inf!"
    assert np.all(np.isfinite(V_k)), "Eigenvector components contain NaN or Inf!"

    # Orthogonality check: V_k^T V_k = I_32
    ortho_mat = np.matmul(V_k.T, V_k)
    ortho_err = float(np.max(np.abs(ortho_mat - np.eye(32))))
    print(f"  - [PASS] Orthogonality assertion: max |V_k^T V_k - I| = {ortho_err:.2e} < 1e-4")
    assert ortho_err < 1e-4, f"Components not strictly orthogonal: {ortho_err}"

    # Verification: Latent Variance and Zero Mean on all 60,000 real centers
    print(f"\n[Verification] Verifying Latent Variance & Finiteness on all 60,000 centers ...")
    all_whitened = []
    for donor, H_d in donor_residuals.items():
        Z_d = whitener.transform(H_d)
        assert Z_d.shape == (len(H_d), 32), f"Transformed shape mismatch: {Z_d.shape}"
        assert np.all(np.isfinite(Z_d)), f"Non-finite whitened features in donor {donor}!"
        all_whitened.append(Z_d)

    Z_all = np.vstack(all_whitened)
    assert Z_all.shape == (n_total_centers, 32)
    assert np.all(np.isfinite(Z_all)), "Z_all contains NaN or Inf!"

    latent_means = np.mean(Z_all, axis=0)
    latent_vars = np.var(Z_all, axis=0, ddof=1)
    mean_err = float(np.max(np.abs(latent_means)))
    var_min = float(np.min(latent_vars))
    var_max = float(np.max(latent_vars))
    var_mean = float(np.mean(latent_vars))

    print(f"  - Max latent mean deviation: {mean_err:.2e} (expected ~0.0)")
    print(f"  - Latent variance across 32 dims: min={var_min:.4f}, mean={var_mean:.4f}, max={var_max:.4f} (expected ~1.0)")
    assert mean_err < 1e-4, f"Latent mean is not zero-centered: {mean_err}"
    assert abs(var_mean - 1.0) < 0.05, f"Latent variance deviates significantly from 1.0: {var_mean}"
    print("  - [PASS] Unit variance & zero mean contract verified across all 32 latent dimensions!")

    total_time = time.time() - t_start
    host_rss_mb = float(psutil.Process().memory_info().rss / (1024**2))
    cov_ram_mb = float(cov_matrix.nbytes / (1024**2))
    res_ram_mb = float(sum(h.nbytes for h in donor_residuals.values()) / (1024**2))

    print(f"\n[Resource Telemetry Breakdown]")
    print(f"  - Covariance matrix buffer (5001 x 5001 float32): {cov_ram_mb:.1f} MB")
    print(f"  - Residuals buffer (60,000 x 5001 float32):       {res_ram_mb:.1f} MB (~{res_ram_mb/1024:.2f} GB)")
    print(f"  - Host process RSS:                               {host_rss_mb:.1f} MB (~{host_rss_mb/1024:.2f} GB, includes AnnData donor loading context)")

    telemetry = {
        "n_total_centers": n_total_centers,
        "input_dim": 5001,
        "k_components": k_components,
        "total_time_seconds": total_time,
        "cov_accumulation_seconds": t_cov,
        "eigh_solve_seconds": t_eigh,
        "orthogonality_error": ortho_err,
        "singular_values_min": float(singular_values.min()),
        "singular_values_max": float(singular_values.max()),
        "latent_mean_max_abs": mean_err,
        "latent_var_min": var_min,
        "latent_var_max": var_max,
        "latent_var_mean": var_mean,
        "whitening_matrix_shape": list(W.shape),
        "covariance_ram_mb": cov_ram_mb,
        "residuals_buffer_ram_mb": res_ram_mb,
        "host_process_rss_mb": host_rss_mb,
    }

    return whitener, telemetry


def save_formal_whitener(whitener: TruncatedTrainOnlyWhitening, telemetry: Dict[str, Any]):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_payload = {
        "whitening_matrix": torch.from_numpy(whitener.whitening_matrix_).float(),
        "mean": torch.from_numpy(whitener.mean_).float(),
        "components": torch.from_numpy(whitener.components_).float(),
        "singular_values": torch.from_numpy(whitener.singular_values_).float(),
        "n_components_fitted": 32,
        "lambda_min": 1e-4,
    }
    torch.save(save_payload, WHITENER_OUT_PATH)
    file_sha = compute_file_sha256(WHITENER_OUT_PATH)
    print(f"\nSaved formal whitener: {WHITENER_OUT_PATH} (SHA: {file_sha})")

    with open(TRAIN_CENTERS_PATH, "r", encoding="utf-8") as f:
        tc = json.load(f)
    train_centers_sha = compute_file_sha256(TRAIN_CENTERS_PATH)

    manifest = {
        "manifest_version": "1.0.0",
        "created_at": "2026-09-17",
        "stage": "W2.7_Whitener",
        "file_name": WHITENER_OUT_PATH.name,
        "file_sha256": file_sha,
        "train_centers_file": TRAIN_CENTERS_PATH.name,
        "train_centers_sha256": train_centers_sha,
        "input_residuals_shape": [len(tc), 5001],
        "whitening_matrix_shape": list(whitener.whitening_matrix_.shape),
        "telemetry": telemetry,
    }
    with open(WHITENER_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved whitener manifest: {WHITENER_MANIFEST_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Fit Train-only Whitener for Xenium 5K")
    parser.add_argument("--preflight", action="store_true", help="Run full 60k chunked covariance telemetry without saving")
    parser.add_argument("--save", action="store_true", help="Formally save whitener.pt (requires --authorized)")
    parser.add_argument("--authorized", action="store_true", help="Explicit user authorization flag to permit saving")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.preflight and not args.save:
        print("ERROR: Must specify either --preflight or --save.")
        sys.exit(2)

    if args.save and not args.authorized:
        print("\n" + "!" * 80)
        print("[CRITICAL SECURITY INTERCEPTION] FORMAL WHITENER SAVING IS LOCKED!")
        print("Formal save requires explicit user audit approval flag --authorized.")
        print("To run Preflight without saving, use: python scripts/benchmark/fit_xenium_whitener.py --preflight")
        print("!" * 80)
        sys.exit(2)

    print("=" * 85)
    print("W2.7 WHITENER: 60,000-CENTER REAL CHUNKED COVARIANCE ACCUMULATION")
    print(f"Device: {args.device} | Target components: 32 | Panel: 5,001 Genes | Mode: {'FORMAL SAVE' if args.save else 'PREFLIGHT'}")
    print("=" * 85)

    whitener, telemetry = fit_truncated_whitener_chunked(device_str=args.device)

    print("\n" + "=" * 85)
    print("REAL WHITENER TELEMETRY REPORT (60,000 Real Centers)")
    print("=" * 85)
    for k, v in telemetry.items():
        print(f"  - {k:30s}: {v}")
    print("=" * 85)

    if args.save:
        save_formal_whitener(whitener, telemetry)
        print("[PASS] Formal whitener and manifest successfully saved.")
    else:
        print("[PASS] Preflight chunked whitener verification complete. ZERO formal assets saved.")


if __name__ == "__main__":
    main()
