"""
W2.7: Build and Cache Physical Spatial Graphs for GSE311609 NSCLC 5K Cohort.

Constructs and cryptographically locks:
1. Train Graphs (4 donors):
   - L3: GSM9509135 (219,228 cells)
   - L5: GSM9509136 (64,883 cells)
   - L7: GSM9509138 (36,827 cells)
   - L8: GSM9509139 (78,734 cells)
2. Validation Graph (1 donor):
   - L6: GSM9509137 (64,819 cells)
3. Hardcoded Test Isolation Guard:
   - Test donor L1 (GSM9509134) is NEVER processed or touched.
4. Physical Coordinates & Topology Contract:
   - Coordinates from adata.obsm["spatial"] (native microns).
   - R = 50.0 um, sigma = 16.666667 um.
   - Strict assertions: symmetry A = A^T, self-loops = 1.0, row-stochasticity sum_j P_ij = 1.0.
5. Saves NPZ files to outputs/runs/freqniche/xenium_prime5k/2026-09-18/20260918_xenium_prime5k_final/cache/ and records SHA-256 manifest.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import anndata as ad
import numpy as np
import scipy.sparse as sp

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.freqniche_st.graph import build_global_spatial_graph
from src.freqniche_st.runtime_paths import resolve_run_output

OUTPUT_DIR = resolve_run_output("outputs/runs/freqniche/xenium_prime5k/2026-09-18/20260918_xenium_prime5k_final")
CACHE_DIR = OUTPUT_DIR / "cache"
GSE_DIR = WORKSPACE_ROOT / "datasets" / "h5ad" / "xenium_ffpe_lungbreast_gse311609" / "samples"

DONOR_SAMPLES = [
    ("GSM9509135_NSCLC_5k_L3_L3.h5ad", "L3", "train"),
    ("GSM9509136_NSCLC_5k_L5_L5.h5ad", "L5", "train"),
    ("GSM9509138_NSCLC_5k_L7_L7.h5ad", "L7", "train"),
    ("GSM9509139_NSCLC_5k_L8_L8.h5ad", "L8", "train"),
    ("GSM9509137_NSCLC_5k_L6_L6.h5ad", "L6", "validation"),
]

RADIUS_UM = 50.0
SIGMA_UM = 16.666667


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1048576):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    print("=" * 85)
    print("W2.7: BUILDING & CACHING PHYSICAL SPATIAL GRAPHS (R=50um, sigma=16.67um)")
    print("=" * 85)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Test Isolation Guard
    for fname, donor, split in DONOR_SAMPLES:
        assert "L1" not in donor and "GSM9509134" not in fname, "CRITICAL: Test donor L1 found!"

    graphs_manifest: Dict[str, Any] = {
        "manifest_version": "1.0.0",
        "created_at": "2026-09-17",
        "stage": "W2.7_Preparation",
        "radius_um": RADIUS_UM,
        "sigma_um": SIGMA_UM,
        "graphs": {},
    }

    t0_all = time.time()
    for fname, donor, split in DONOR_SAMPLES:
        h5ad_p = GSE_DIR / fname
        assert h5ad_p.exists(), f"Missing H5AD: {h5ad_p}"

        p_cache_path = CACHE_DIR / f"xenium_graph_{donor}_P.npz"
        a_cache_path = CACHE_DIR / f"xenium_graph_{donor}_A.npz"

        print(f"\nProcessing {donor} ({split}): {fname} ...")
        t0 = time.time()
        a = ad.read_h5ad(h5ad_p, backed="r")
        coords = np.asarray(a.obsm["spatial"], dtype=np.float32)
        n_cells = len(coords)
        sample_ids = np.array([f"{donor}_{fname}"] * n_cells)

        print(f"  - Loaded {n_cells:,} coordinates in {time.time() - t0:.2f} s")

        # Build graph
        t0_build = time.time()
        P, A = build_global_spatial_graph(
            coords=coords,
            sample_ids=sample_ids,
            radius=RADIUS_UM,
            sigma=SIGMA_UM,
            max_neighbors=None,
        )
        t_build = time.time() - t0_build
        print(f"  - Built graph in {t_build:.2f} s: {A.nnz:,} edges (avg degree: {A.nnz/n_cells - 1:.1f})")

        # Mathematical assertions
        # 1. Self-loops
        diag = A.diagonal()
        assert np.allclose(diag, 1.0, atol=1e-5), "Self loops must be 1.0"
        # 2. Symmetry
        diff_sym = (A - A.T).tocoo()
        max_sym_err = float(np.max(np.abs(diff_sym.data))) if len(diff_sym.data) > 0 else 0.0
        assert max_sym_err < 1e-5, f"Adjacency not symmetric: max err {max_sym_err}"
        # 3. Row stochasticity
        p_row_sums = np.asarray(P.sum(axis=1)).ravel()
        max_p_err = float(np.max(np.abs(p_row_sums - 1.0)))
        assert max_p_err < 1e-5, f"P row sums exceed tolerance: max err {max_p_err}"
        print(f"  - [PASS] Topology verified: symmetry max_err={max_sym_err:.2e}, P row-sum max_err={max_p_err:.2e}")

        # Save NPZ
        sp.save_npz(p_cache_path, P)
        sp.save_npz(a_cache_path, A)

        p_sha = compute_file_sha256(p_cache_path)
        a_sha = compute_file_sha256(a_cache_path)

        p_size_mb = p_cache_path.stat().st_size / (1024 * 1024)
        a_size_mb = a_cache_path.stat().st_size / (1024 * 1024)

        print(f"  - Saved P ({p_size_mb:.1f} MB, SHA: {p_sha[:16]}...)")
        print(f"  - Saved A ({a_size_mb:.1f} MB, SHA: {a_sha[:16]}...)")

        graphs_manifest["graphs"][donor] = {
            "sample_id": fname.replace(".h5ad", ""),
            "split": split,
            "n_cells": n_cells,
            "n_edges": int(A.nnz),
            "p_file": p_cache_path.name,
            "p_sha256": p_sha,
            "p_size_bytes": int(p_cache_path.stat().st_size),
            "a_file": a_cache_path.name,
            "a_sha256": a_sha,
            "a_size_bytes": int(a_cache_path.stat().st_size),
        }

    manifest_out = OUTPUT_DIR / "xenium_graphs_manifest.json"
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(graphs_manifest, f, indent=2)

    total_time = time.time() - t0_all
    print("\n" + "=" * 85)
    print(f"[SUCCESS] ALL 5 GRAPHS BUILT & CACHED IN {total_time:.1f} s")
    print(f"Manifest written to: {manifest_out}")
    print("=" * 85)


if __name__ == "__main__":
    main()
