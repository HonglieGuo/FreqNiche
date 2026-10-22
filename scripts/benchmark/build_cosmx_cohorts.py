"""
Build and Freeze CosMx NSCLC Cohorts, Spatial Graph Caches, and Center Manifests (W2.6 Stage 1 - v4.0 Approved).

Strictly enforces:
1. Dataset path: datasets/h5ad/cosmx_human_nsclc/cosmx_human_nsclc.h5ad (765,771 cells, 960 genes)
2. Rigorous physical coordinate read contract:
   - Direct read from adata.obsm["spatial_um"] (already in physical micrometers, NO secondary scaling)
   - Radius R = 50.0 um, Gaussian kernel sigma = 50.0 / 3.0 = 16.666667 um
3. Unified spatial graph contract:
   - Built via shared build_global_spatial_graph() from src.freqniche_st.graph
   - A_train is Gaussian-distance weighted symmetric adjacency (NOT binary)
   - P_train is row-stochastic diffusion operator (sum_j P_ij = 1.0)
   - Strict sample isolation (0 cross-sample edges)
4. Center quotas & sampling:
   - Train (3 donors: Lung12, Lung13, Lung6): strictly 20,000 non-isolated centers each -> 60,000 centers
   - Validation (1 donor: Lung5): strictly 10,000 non-isolated centers
   - Test (1 donor: Lung9): strictly 10,000 non-isolated centers (strictly isolated until Phase 2)
   - 10-Fold bootstrap folds on Validation centers (8,000 centers per fold)
5. Cryptographic triple-hash signatures:
   - file_sha256: physical file hash
   - sorted_barcode_sha256: set invariance hash
   - ordered_barcode_sha256: execution sequence deterministic hash
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.freqniche_st.graph import build_global_spatial_graph
from src.freqniche_st.runtime_paths import resolve_dataset_path, resolve_run_output

DATASET_PATH = resolve_dataset_path("FREQNICHE_DATASET_PATH", "datasets/h5ad/cosmx_human_nsclc/cosmx_human_nsclc.h5ad")
OUTPUT_DIR = resolve_run_output("outputs/runs/freqniche/cosmx_nsclc/2026-09-17/20260917_cosmx_nsclc_final")
CACHE_DIR = OUTPUT_DIR / "cache"

RADIUS_UM = 50.0
SIGMA_UM = RADIUS_UM / 3.0  # 16.666667 um
SCALE_FACTOR_UM_PER_PIXEL = 0.120280945
RANDOM_SEED = 42


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_sorted_barcode_sha256(items: List[str]) -> str:
    serialized = json.dumps(sorted(items), separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_ordered_barcode_sha256(items: List[str]) -> str:
    serialized = json.dumps(list(items), separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def main():
    print("=" * 85)
    print("W2.6 STAGE 1: COSMX NSCLC DATA AUDIT, SPATIAL GRAPH BUILDING & COHORT FREEZING")
    print("=" * 85)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Dataset
    print(f"\n[1/6] Loading AnnData from {DATASET_PATH} ...")
    assert DATASET_PATH.exists(), f"Missing dataset: {DATASET_PATH}"
    adata = ad.read_h5ad(DATASET_PATH)
    n_cells, n_genes = adata.shape
    print(f"Loaded AnnData: {n_cells:,} cells x {n_genes} genes.")
    assert n_cells == 765771, f"Expected 765,771 cells, got {n_cells}"
    assert n_genes == 960, f"Expected 960 genes, got {n_genes}"

    # 2. Verify and read physical micron coordinates directly (NO secondary scaling)
    print("\n[2/6] Verifying physical micron coordinate contract (obsm['spatial_um']) ...")
    assert "spatial_um" in adata.obsm, "Missing obsm['spatial_um'] in dataset!"
    assert "spatial_pixel" in adata.obsm, "Missing obsm['spatial_pixel'] in dataset!"
    coords_um = np.asarray(adata.obsm["spatial_um"], dtype=np.float32)
    assert coords_um.shape == (n_cells, 2), f"Coordinates shape mismatch: {coords_um.shape}"
    assert np.isfinite(coords_um).all(), "Coordinates contain non-finite values!"
    print(f"Physical micron coordinates verified directly: min={coords_um.min(axis=0)}, max={coords_um.max(axis=0)}")

    # 3. Build sample-isolated spatial graph on Train split using shared build_global_spatial_graph()
    print(f"\n[3/6] Building Train spatial graph via shared build_global_spatial_graph() (R={RADIUS_UM} um, sigma={SIGMA_UM:.4f} um) ...")
    train_mask = (adata.obs["split"] == "train").to_numpy()
    train_indices = np.where(train_mask)[0]
    n_train_cells = len(train_indices)
    assert n_train_cells == 242515, f"Expected 242,515 train cells, got {n_train_cells}"

    train_coords_um = coords_um[train_indices]
    train_sample_ids = adata.obs["sample_id"].iloc[train_indices].to_numpy()

    # Call shared function
    P_train, A_train = build_global_spatial_graph(
        coords=train_coords_um,
        sample_ids=train_sample_ids,
        radius=RADIUS_UM,
        sigma=SIGMA_UM,
        max_neighbors=None,
    )
    print(f"Built Train graph: A={A_train.shape}, nnz={A_train.nnz:,} | P={P_train.shape}, nnz={P_train.nnz:,}")

    # 4. Strict Acceptance Assertions on Train Graphs
    print("\n[4/6] Executing topological contract assertions on Train graphs ...")
    # 4.1 Symmetry of A
    diff_sym = (A_train - A_train.T).tocoo()
    max_sym_err = float(np.max(np.abs(diff_sym.data))) if len(diff_sym.data) > 0 else 0.0
    assert max_sym_err < 1e-5, f"A is not strictly symmetric! Max error = {max_sym_err}"
    print(f" [PASS] Symmetry verified: max |A - A^T| = {max_sym_err:.2e} < 1e-5.")

    # 4.2 Self-loops
    diag_a = A_train.diagonal()
    assert np.allclose(diag_a, 1.0, atol=1e-5), "Self-loops are not all 1.0!"
    assert len(diag_a) == n_train_cells, "Diagonal length mismatch!"
    print(f" [PASS] Self-loops verified: exactly {n_train_cells:,} diagonal elements equal 1.0.")

    # 4.3 Non-diagonal weights are not all 1.0 (continuous Gaussian weighting)
    # Check that there are diverse weights between 0 and 1
    coo_a = A_train.tocoo()
    off_diag_mask = (coo_a.row != coo_a.col)
    off_diag_weights = coo_a.data[off_diag_mask]
    assert len(off_diag_weights) > 0, "No off-diagonal edges found!"
    assert not np.allclose(off_diag_weights, 1.0), "Adjacency is binary! Expected continuous Gaussian weights."
    min_w, max_w, mean_w = float(off_diag_weights.min()), float(off_diag_weights.max()), float(off_diag_weights.mean())
    print(f" [PASS] Continuous Gaussian weighting verified: off-diagonal weights min={min_w:.4f}, mean={mean_w:.4f}, max={max_w:.4f}.")

    # 4.4 Row sums of P
    p_row_sums = np.asarray(P_train.sum(axis=1)).ravel()
    max_row_sum_err = float(np.max(np.abs(p_row_sums - 1.0)))
    assert max_row_sum_err < 1e-5, f"P row sums exceed tolerance: max error = {max_row_sum_err}"
    print(f" [PASS] P row-stochasticity verified: max |P1 - 1| = {max_row_sum_err:.2e} < 1e-5.")

    # 4.5 Check cross-sample isolation
    # For every edge in A_train, both endpoints must belong to the same sample_id
    rows_sample = train_sample_ids[coo_a.row]
    cols_sample = train_sample_ids[coo_a.col]
    cross_sample_edges = int(np.sum(rows_sample != cols_sample))
    assert cross_sample_edges == 0, f"Detected {cross_sample_edges} cross-sample edges in Train graph!"
    print(" [PASS] Sample isolation verified: exactly 0 cross-sample edges.")

    # 4.6 Degrees & Isolated cells (isolated cell = degree == 1 since self-loop=1)
    train_degrees = np.asarray((A_train > 0).sum(axis=1)).ravel()
    isolated_train_local = (train_degrees <= 1)
    print(f"Train Graph Degree Stats (R={RADIUS_UM} um):")
    print(f" - Mean neighbor count (including self): {train_degrees.mean():.2f}")
    print(f" - Median neighbor count: {np.median(train_degrees):.1f}")
    print(f" - Isolated cells (no neighbors except self): {isolated_train_local.sum():,} ({isolated_train_local.mean() * 100:.3f}%)")

    # Save Train graph caches atomically
    train_p_path = CACHE_DIR / "cosmx_train_graph_P.npz"
    train_a_path = CACHE_DIR / "cosmx_train_graph_A.npz"
    sp.save_npz(train_p_path, P_train)
    sp.save_npz(train_a_path, A_train)
    print(f"Saved Train P ({train_p_path.stat().st_size / (1024**2):.2f} MiB) and A ({train_a_path.stat().st_size / (1024**2):.2f} MiB).")

    # 5. Build global topology to identify isolated cells across whole dataset for clean sampling
    print("\n[5/6] Sampling Train (60k), Validation (10k), and Test (10k) center cohorts ...")
    rng = np.random.RandomState(RANDOM_SEED)

    # 5.1 Sample Train Centers (3 donors: Lung12, Lung13, Lung6, strictly 20,000 non-isolated centers each)
    train_donors = sorted(adata.obs.loc[train_mask, "donor_id"].unique().tolist())
    print(f"Train donors: {train_donors}")
    train_selected_global_indices = []

    # Map local train index to global index
    train_local_to_global = {l: g for l, g in enumerate(train_indices)}
    global_to_train_local = {g: l for l, g in enumerate(train_indices)}

    for d in train_donors:
        donor_cells_local = [l for l in range(n_train_cells) if adata.obs["donor_id"].iloc[train_local_to_global[l]] == d and not isolated_train_local[l]]
        assert len(donor_cells_local) >= 20000, f"Donor {d} only has {len(donor_cells_local)} candidates, need 20,000"
        chosen_local = rng.choice(donor_cells_local, size=20000, replace=False)
        chosen_global = [train_local_to_global[l] for l in chosen_local]
        train_selected_global_indices.extend(chosen_global)

    train_selected_global_indices = sorted(train_selected_global_indices)
    assert len(train_selected_global_indices) == 60000
    print(f"Selected Train centers: strictly {len(train_selected_global_indices):,} centers (20,000 per donor).")

    # 5.2 Build Validation Graph and sample Validation Centers (Lung5, strictly 10,000 non-isolated centers)
    val_mask = (adata.obs["split"] == "validation").to_numpy()
    val_indices = np.where(val_mask)[0]
    val_coords_um = coords_um[val_indices]
    val_sample_ids = adata.obs["sample_id"].iloc[val_indices].to_numpy()
    P_val, A_val = build_global_spatial_graph(coords=val_coords_um, sample_ids=val_sample_ids, radius=RADIUS_UM, sigma=SIGMA_UM, max_neighbors=None)
    val_degrees = np.asarray((A_val > 0).sum(axis=1)).ravel()
    isolated_val_local = (val_degrees <= 1)
    val_cand_local = np.where(~isolated_val_local)[0]
    val_chosen_local = rng.choice(val_cand_local, size=10000, replace=False)
    val_chosen_global = sorted([val_indices[l] for l in val_chosen_local])
    assert len(val_chosen_global) == 10000
    print(f"Selected Validation centers: strictly {len(val_chosen_global):,} centers (from Lung5).")

    # 5.3 Build Test Graph and sample Test Centers (Lung9, strictly 10,000 non-isolated centers)
    test_mask = (adata.obs["split"] == "test").to_numpy()
    test_indices = np.where(test_mask)[0]
    test_coords_um = coords_um[test_indices]
    test_sample_ids = adata.obs["sample_id"].iloc[test_indices].to_numpy()
    P_test, A_test = build_global_spatial_graph(coords=test_coords_um, sample_ids=test_sample_ids, radius=RADIUS_UM, sigma=SIGMA_UM, max_neighbors=None)
    test_degrees = np.asarray((A_test > 0).sum(axis=1)).ravel()
    isolated_test_local = (test_degrees <= 1)
    test_cand_local = np.where(~isolated_test_local)[0]
    test_chosen_local = rng.choice(test_cand_local, size=10000, replace=False)
    test_chosen_global = sorted([test_indices[l] for l in test_chosen_local])
    assert len(test_chosen_global) == 10000
    print(f"Selected Test centers: strictly {len(test_chosen_global):,} centers (from Lung9).")

    # Ensure pure python ints for robust json serialization
    train_selected_global_indices = [int(x) for x in train_selected_global_indices]
    val_chosen_global = [int(x) for x in val_chosen_global]
    test_chosen_global = [int(x) for x in test_chosen_global]

    # Export Cohort JSONs with Triple Hashes
    train_barcodes = adata.obs_names[train_selected_global_indices].tolist()
    val_barcodes = adata.obs_names[val_chosen_global].tolist()
    test_barcodes = adata.obs_names[test_chosen_global].tolist()

    train_centers_path = OUTPUT_DIR / "train_centers.json"
    val_centers_path = OUTPUT_DIR / "val_centers.json"
    test_centers_path = OUTPUT_DIR / "test_centers.json"

    train_payload = {
        "cohort": "train_3_donors_60k_centers",
        "n_centers": len(train_selected_global_indices),
        "donors": {d: 20000 for d in train_donors},
        "global_indices": train_selected_global_indices,
        "cell_barcodes": train_barcodes,
        "sorted_barcode_sha256": compute_sorted_barcode_sha256(train_barcodes),
        "ordered_barcode_sha256": compute_ordered_barcode_sha256(train_barcodes),
    }
    with open(train_centers_path, "w", encoding="utf-8") as f:
        json.dump(train_payload, f, indent=2)

    val_payload = {
        "cohort": "val_1_donor_10k_centers",
        "n_centers": len(val_chosen_global),
        "donor": "Lung5",
        "global_indices": val_chosen_global,
        "cell_barcodes": val_barcodes,
        "sorted_barcode_sha256": compute_sorted_barcode_sha256(val_barcodes),
        "ordered_barcode_sha256": compute_ordered_barcode_sha256(val_barcodes),
    }
    with open(val_centers_path, "w", encoding="utf-8") as f:
        json.dump(val_payload, f, indent=2)

    test_payload = {
        "cohort": "test_1_donor_10k_centers",
        "n_centers": len(test_chosen_global),
        "donor": "Lung9",
        "global_indices": test_chosen_global,
        "cell_barcodes": test_barcodes,
        "sorted_barcode_sha256": compute_sorted_barcode_sha256(test_barcodes),
        "ordered_barcode_sha256": compute_ordered_barcode_sha256(test_barcodes),
    }
    with open(test_centers_path, "w", encoding="utf-8") as f:
        json.dump(test_payload, f, indent=2)

    # 6. Generate 10-Fold Bootstrap on Validation Centers (8,000 centers each)
    print("\n[6/6] Generating 10-Fold Bootstrap sub-samples on Validation centers ...")
    bootstrap_folds = {}
    for fold_id in range(10):
        sub_rng = np.random.RandomState(RANDOM_SEED + 100 + fold_id)
        sub_indices_in_val = sub_rng.choice(10000, size=8000, replace=False)
        sub_global_indices = [int(val_chosen_global[i]) for i in sub_indices_in_val]
        sub_indices_in_val_py = [int(i) for i in sub_indices_in_val]
        sub_barcodes = [val_barcodes[i] for i in sub_indices_in_val]
        bootstrap_folds[f"fold_{fold_id}"] = {
            "indices_in_val_centers": sub_indices_in_val_py,
            "global_indices": sub_global_indices,
            "sample_size": 8000,
            "sorted_barcode_sha256": compute_sorted_barcode_sha256(sub_barcodes),
            "ordered_barcode_sha256": compute_ordered_barcode_sha256(sub_barcodes),
        }

    folds_path = OUTPUT_DIR / "val_bootstrap_folds.json"
    with open(folds_path, "w", encoding="utf-8") as f:
        json.dump(bootstrap_folds, f, indent=2)
    print("Saved 10-Fold Bootstrap definitions to val_bootstrap_folds.json.")

    # Calculate file hashes
    p_sha = compute_file_sha256(train_p_path)
    a_sha = compute_file_sha256(train_a_path)
    tc_file_sha = compute_file_sha256(train_centers_path)
    vc_file_sha = compute_file_sha256(val_centers_path)
    testc_file_sha = compute_file_sha256(test_centers_path)
    folds_file_sha = compute_file_sha256(folds_path)

    # Manifest
    manifest = {
        "dataset": "cosmx_human_nsclc",
        "h5ad_file": "datasets/h5ad/cosmx_human_nsclc/cosmx_human_nsclc.h5ad",
        "h5ad_sha256": compute_file_sha256(DATASET_PATH),
        "scale_conversion": {
            "factor_um_per_pixel": SCALE_FACTOR_UM_PER_PIXEL,
            "source_provenance": "CosMx official SMI documentation indicates ~0.12 um/px; repo conversion implementation adopts exact value 0.120280945",
            "radius_um": RADIUS_UM,
            "sigma_um": SIGMA_UM,
            "coordinate_source": "adata.obsm['spatial_um']",
        },
        "quotas": {
            "train_centers": 60000,
            "val_centers": 10000,
            "test_centers": 10000,
        },
        "train_graph_cache": {
            "P_file": "cosmx_train_graph_P.npz",
            "P_sha256": p_sha,
            "A_file": "cosmx_train_graph_A.npz",
            "A_sha256": a_sha,
            "nnz": int(A_train.nnz),
            "mean_degree": float(train_degrees.mean()),
            "max_symmetry_error": max_sym_err,
            "max_row_sum_error": max_row_sum_err,
            "cross_sample_edges": 0,
        },
        "cohorts": {
            "train": {
                "file": "train_centers.json",
                "count": 60000,
                "file_sha256": tc_file_sha,
                "sorted_barcode_sha256": train_payload["sorted_barcode_sha256"],
                "ordered_barcode_sha256": train_payload["ordered_barcode_sha256"],
            },
            "validation": {
                "file": "val_centers.json",
                "count": 10000,
                "file_sha256": vc_file_sha,
                "sorted_barcode_sha256": val_payload["sorted_barcode_sha256"],
                "ordered_barcode_sha256": val_payload["ordered_barcode_sha256"],
            },
            "test": {
                "file": "test_centers.json",
                "count": 10000,
                "file_sha256": testc_file_sha,
                "sorted_barcode_sha256": test_payload["sorted_barcode_sha256"],
                "ordered_barcode_sha256": test_payload["ordered_barcode_sha256"],
            },
            "val_bootstrap_folds": {
                "file": "val_bootstrap_folds.json",
                "folds": 10,
                "file_sha256": folds_file_sha,
            },
        },
        "acceptance_assertions": {
            "A_is_symmetric": True,
            "A_diagonal_self_loops_all_one": True,
            "A_off_diagonal_gaussian_weighted_not_binary": True,
            "P_row_sums_equal_one": True,
            "cross_sample_edges_strictly_zero": True,
            "train_val_graphs_built_with_identical_shared_function": True,
        },
    }

    manifest_path = OUTPUT_DIR / "cosmx_data_audit_manifest.json"
    temp_manifest = manifest_path.with_suffix(".tmp.json")
    with open(temp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(temp_manifest, manifest_path)

    print("\n" + "=" * 85)
    print("W2.6 STAGE 1 COHORT & CACHE BUILDING COMPLETE!")
    print(f"Manifest exported atomically to: {manifest_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()
