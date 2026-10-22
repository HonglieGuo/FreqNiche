"""
W2.7: Freeze Xenium Train and Validation Center Cohorts.

Pre-freezes:
1. Train Centers (60,000 total):
   - Exactly 15,000 non-isolated centers per Train donor:
     * L3: GSM9509135 (219k cells total)
     * L5: GSM9509136 (65k cells total)
     * L7: GSM9509138 (37k cells total)
     * L8: GSM9509139 (79k cells total)
   - Filter condition: degree >= 1 in physical R=50.0 um spatial graph (non-isolated).
   - Fixed RNG seed: 42.
2. Validation Centers (10,000 total):
   - Sampled from Validation donor L6: GSM9509137 (65k cells total).
   - Non-isolated (degree >= 1 in R=50.0 um graph).
   - Fixed RNG seed: 42.
3. Cryptographic Integrity:
   - Output train_centers.json, val_centers.json.
   - Computes Triple-Hash (file SHA-256, sorted barcode SHA-256, ordered barcode SHA-256).
   - Records xenium_cohorts_manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from src.freqniche_st.runtime_paths import resolve_run_output
from typing import Dict, List, Any

import anndata as ad
import numpy as np
from scipy.spatial import cKDTree

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = resolve_run_output("outputs/runs/freqniche/xenium_prime5k/2026-09-18/20260918_xenium_prime5k_final")
GSE_DIR = WORKSPACE_ROOT / "datasets" / "h5ad" / "xenium_ffpe_lungbreast_gse311609" / "samples"

TRAIN_SAMPLES = [
    ("GSM9509135_NSCLC_5k_L3_L3.h5ad", "L3", 15000),
    ("GSM9509136_NSCLC_5k_L5_L5.h5ad", "L5", 15000),
    ("GSM9509138_NSCLC_5k_L7_L7.h5ad", "L7", 15000),
    ("GSM9509139_NSCLC_5k_L8_L8.h5ad", "L8", 15000),
]

VAL_SAMPLE = ("GSM9509137_NSCLC_5k_L6_L6.h5ad", "L6", 10000)

RANDOM_SEED = 42
RADIUS_UM = 50.0


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1048576):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_sorted_barcode_sha256(items: List[str]) -> str:
    serialized = json.dumps(sorted(items), separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_ordered_barcode_sha256(items: List[str]) -> str:
    serialized = json.dumps(list(items), separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sample_non_isolated_centers(
    h5ad_path: Path, donor_id: str, n_samples: int, seed: int
) -> List[Dict[str, Any]]:
    print(f"  Sampling {n_samples:,} non-isolated centers from {donor_id} ({h5ad_path.name}) ...")
    a = ad.read_h5ad(h5ad_path, backed="r")
    coords = np.asarray(a.obsm["spatial"], dtype=np.float32)
    barcodes = list(a.obs_names)
    n_cells = len(coords)

    # Build cKDTree to find non-isolated cells (at least 1 neighbor within RADIUS_UM besides self)
    tree = cKDTree(coords)
    # query radius returns list of neighbor indices for each cell
    neighbor_counts = tree.query_ball_point(coords, r=RADIUS_UM, return_length=True)
    non_isolated_mask = (neighbor_counts > 1)
    valid_indices = np.where(non_isolated_mask)[0]
    n_valid = len(valid_indices)

    print(f"    - Total cells: {n_cells:,}, non-isolated (degree >= 1): {n_valid:,} ({n_valid/n_cells*100:.2f}%)")
    assert n_valid >= n_samples, f"Donor {donor_id} only has {n_valid} non-isolated cells, requested {n_samples}"

    rng = np.random.RandomState(seed)
    chosen_local = rng.choice(valid_indices, size=n_samples, replace=False)
    chosen_local = np.sort(chosen_local)  # keep deterministic sorted order per donor

    centers = []
    for idx in chosen_local:
        centers.append({
            "donor_id": donor_id,
            "sample_id": h5ad_path.name.replace(".h5ad", ""),
            "local_cell_idx": int(idx),
            "cell_barcode": str(barcodes[idx]),
            "x_um": float(coords[idx, 0]),
            "y_um": float(coords[idx, 1]),
            "degree_50um": int(neighbor_counts[idx] - 1),
        })

    return centers


def main():
    print("=" * 85)
    print("W2.7 STEP 3 PREPARATION: FREEZING XENIUM TRAIN & VAL COHORTS")
    print(f"Random seed: {RANDOM_SEED} | Radius: {RADIUS_UM} um")
    print("=" * 85)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Hardcoded safety guard: Test donor L1 MUST NOT be in the sampling list
    assert all("L1" not in s[1] for s in TRAIN_SAMPLES), "CRITICAL: Test donor L1 found in train samples!"
    assert "L1" not in VAL_SAMPLE[1], "CRITICAL: Test donor L1 found in val sample!"

    # 1. Sample Train Centers
    print("\n[1/3] Sampling Train Centers (15k x 4 donors = 60,000 centers) ...")
    train_centers = []
    donor_train_subcounts = {}
    for fname, donor_id, n_req in TRAIN_SAMPLES:
        p = GSE_DIR / fname
        assert p.exists(), f"Sample not found: {p}"
        sub_centers = sample_non_isolated_centers(p, donor_id, n_req, seed=RANDOM_SEED)
        train_centers.extend(sub_centers)
        donor_train_subcounts[donor_id] = len(sub_centers)

    assert len(train_centers) == 60000, f"Expected 60,000 train centers, got {len(train_centers)}"
    print(f"[OK] Successfully sampled {len(train_centers):,} Train centers: {donor_train_subcounts}")

    # 2. Sample Validation Centers
    print("\n[2/3] Sampling Validation Centers (10,000 centers from L6) ...")
    val_fname, val_donor_id, val_req = VAL_SAMPLE
    p_val = GSE_DIR / val_fname
    assert p_val.exists(), f"Val sample not found: {p_val}"
    val_centers = sample_non_isolated_centers(p_val, val_donor_id, val_req, seed=RANDOM_SEED)
    assert len(val_centers) == 10000, f"Expected 10,000 val centers, got {len(val_centers)}"
    print(f"[OK] Successfully sampled {len(val_centers):,} Validation centers from {val_donor_id}")

    # 3. Save JSON Files & Compute Cryptographic Hashes
    print("\n[3/3] Saving center cohorts and computing cryptographic manifests ...")
    train_centers_path = OUTPUT_DIR / "train_centers.json"
    val_centers_path = OUTPUT_DIR / "val_centers.json"
    manifest_path = OUTPUT_DIR / "xenium_cohorts_manifest.json"

    with open(train_centers_path, "w", encoding="utf-8") as f:
        json.dump(train_centers, f, indent=2)

    with open(val_centers_path, "w", encoding="utf-8") as f:
        json.dump(val_centers, f, indent=2)

    train_barcodes = [c["cell_barcode"] for c in train_centers]
    val_barcodes = [c["cell_barcode"] for c in val_centers]

    manifest = {
        "manifest_version": "1.0.0",
        "created_at": "2026-09-17",
        "stage": "W2.7_Preparation",
        "description": "Pre-frozen Train (60k) and Validation (10k) center cohorts for Xenium 5K",
        "random_seed": RANDOM_SEED,
        "radius_um": RADIUS_UM,
        "train_cohort": {
            "total_centers": len(train_centers),
            "donors": donor_train_subcounts,
            "file_sha256": compute_file_sha256(train_centers_path),
            "sorted_barcode_sha256": compute_sorted_barcode_sha256(train_barcodes),
            "ordered_barcode_sha256": compute_ordered_barcode_sha256(train_barcodes),
        },
        "validation_cohort": {
            "total_centers": len(val_centers),
            "donor": val_donor_id,
            "file_sha256": compute_file_sha256(val_centers_path),
            "sorted_barcode_sha256": compute_sorted_barcode_sha256(val_barcodes),
            "ordered_barcode_sha256": compute_ordered_barcode_sha256(val_barcodes),
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"  - Wrote train centers: {train_centers_path} (SHA: {manifest['train_cohort']['file_sha256'][:16]}...)")
    print(f"  - Wrote val centers:   {val_centers_path} (SHA: {manifest['validation_cohort']['file_sha256'][:16]}...)")
    print(f"  - Wrote manifest:      {manifest_path}")
    print("\n" + "=" * 85)
    print("[SUCCESS] XENIUM TRAIN & VAL COHORTS PRE-FROZEN WITH CRYPTOGRAPHIC LOCK")
    print("=" * 85)


if __name__ == "__main__":
    main()
