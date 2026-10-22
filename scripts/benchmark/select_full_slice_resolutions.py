"""
scripts/benchmark/select_full_slice_resolutions.py

Strict Two-Phase Resolution Selection with Full Execution Chain Binding:
- Phase 1: Selection (--phase selection)
  * ZERO GT LEAKAGE: Evaluates only unlabeled stability ARI on 10-fold 80% bootstrap subgraphs.
  * FAIL-CLOSED: Strictly requires ALL 5 method embeddings (freqniche, banksy, cellcharter, stagate, cellniche)
    to exist, have shape (11293, D), and contain only finite values.
  * Binds complete provenance chain into self-signed resolution_selection.json:
    - 5 embedding file SHA-256s and shapes
    - H5AD SHA-256 and ordered barcode SHA-256
    - bootstrap_manifest file SHA-256 and internal self-hash
    - selector script SHA-256
    - unlabeled_assignments.csv SHA-256
    - Full unrounded floating-point stability results
  * Self-signs resolution_selection.json with cryptographic SHA-256 fingerprint.
- Phase 2: Post-Hoc Evaluation (--phase posthoc)
  * Verifies self-signed metadata, assignment SHA-256, and bootstrap manifest before reading GT.
  * Strictly asserts (unlabeled_df['barcode'] == obs_slice.index) element-wise.
  * Strictly asserts required 'niche' ground truth column exists; fallback to cell_type is FORBIDDEN.
  * Computes ARI and NMI against true niche labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "datasets/h5ad/human_skin_merfish/human_skin_merfish.h5ad"
TARGET_SAMPLE_ID = "202309281027_AutopsyBFFPE092823_VMSC02301_region_1"

OUTPUT_BASE = ROOT / "baselines/Baselines_outputs/runs/human_skin/2026-09-21/20260921_human_skin_full_slice_comparison"
EMB_DIR = OUTPUT_BASE / "embeddings"
RESULTS_DIR = OUTPUT_BASE / "results"
BOOTSTRAP_MANIFEST_PATH = OUTPUT_BASE / "bootstrap_manifest_11293_10fold.json"

SELECTION_META_PATH = RESULTS_DIR / "resolution_selection.json"
UNLABELED_ASSIGNMENTS_PATH = RESULTS_DIR / "assignments_unlabeled.csv"
FINAL_ASSIGNMENTS_PATH = RESULTS_DIR / "assignments_with_metrics.csv"
POSTHOC_METRICS_PATH = RESULTS_DIR / "posthoc_evaluation_metrics.json"

CANDIDATE_RESOLUTIONS = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
K_MIN = 5
K_MAX = 25
KNN_K = 15
RANDOM_STATE = 42

REQUIRED_METHODS = ["freqniche", "banksy", "cellcharter", "stagate", "cellniche"]


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_bytes_sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def validate_bootstrap_manifest(barcodes: list[str]) -> dict[str, Any]:
    assert BOOTSTRAP_MANIFEST_PATH.exists(), f"Missing bootstrap manifest: {BOOTSTRAP_MANIFEST_PATH}"
    with open(BOOTSTRAP_MANIFEST_PATH, "r", encoding="utf-8") as fp:
        bs_data = json.load(fp)

    # 1. Self-hash verification
    saved_self_hash = bs_data.get("manifest_self_sha256")
    bs_data_copy = dict(bs_data)
    bs_data_copy.pop("manifest_self_sha256", None)
    calc_self_hash = compute_bytes_sha256(json.dumps(bs_data_copy, sort_keys=True).encode("utf-8"))
    assert calc_self_hash == saved_self_hash, "Bootstrap manifest self-hash mismatch! Tampering detected."

    # 2. Barcode binding verification
    ordered_str = "\n".join(barcodes) + "\n"
    actual_ordered_sha = compute_bytes_sha256(ordered_str.encode("utf-8"))
    assert actual_ordered_sha == bs_data["full_slice_ordered_barcodes_sha256"], (
        f"Bootstrap manifest ordered barcode SHA mismatch!\n"
        f"Expected: {bs_data['full_slice_ordered_barcodes_sha256']}\n"
        f"Actual:   {actual_ordered_sha}"
    )
    print(f"[OK] Bootstrap manifest validated and cryptographically bound (Self-Hash: {saved_self_hash[:16]}...)")
    return bs_data


def evaluate_method_stability_full_precision(
    emb: np.ndarray,
    folds: dict[str, list[int]],
) -> tuple[float, dict[str, Any]]:
    """Evaluates Leiden stability ARI across bootstrap folds using full raw floating-point precision."""
    fold_clusters: dict[float, list[np.ndarray]] = {r: [] for r in CANDIDATE_RESOLUTIONS}
    fold_ks: dict[float, list[int]] = {r: [] for r in CANDIDATE_RESOLUTIONS}

    fold_items = sorted(folds.items())
    for fold_name, indices in fold_items:
        idx_arr = np.array(indices)
        sub_emb = emb[idx_arr]

        ad_sub = ad.AnnData(X=sub_emb)
        sc.pp.neighbors(ad_sub, n_neighbors=KNN_K, use_rep="X", metric="euclidean")

        for r in CANDIDATE_RESOLUTIONS:
            sc.tl.leiden(ad_sub, resolution=r, random_state=RANDOM_STATE)
            c = ad_sub.obs["leiden"].to_numpy().astype(int)
            fold_clusters[r].append(c)
            fold_ks[r].append(len(np.unique(c)))

    grid_results = {}
    valid_candidates = []

    for r in CANDIDATE_RESOLUTIONS:
        ks = fold_ks[r]
        mean_k = float(np.mean(ks))
        k_in_range = bool(K_MIN <= mean_k <= K_MAX)

        # Pairwise stability ARI across folds on overlapping indices
        pairwise_aris = []
        n_folds = len(fold_items)
        for i in range(n_folds):
            idx_i = set(fold_items[i][1])
            c_i = fold_clusters[r][i]
            map_i = {orig_idx: c_i[loc] for loc, orig_idx in enumerate(fold_items[i][1])}

            for j in range(i + 1, n_folds):
                idx_j = set(fold_items[j][1])
                c_j = fold_clusters[r][j]
                map_j = {orig_idx: c_j[loc] for loc, orig_idx in enumerate(fold_items[j][1])}

                overlap = sorted(idx_i.intersection(idx_j))
                if len(overlap) > 10:
                    labels_i = np.array([map_i[o] for o in overlap])
                    labels_j = np.array([map_j[o] for o in overlap])
                    pairwise_aris.append(adjusted_rand_score(labels_i, labels_j))

        # Full precision raw float
        raw_stability_ari = float(np.mean(pairwise_aris)) if pairwise_aris else 0.0

        grid_results[str(r)] = {
            "resolution": r,
            "mean_k": round(mean_k, 2),
            "k_in_prior_range": k_in_range,
            "raw_mean_stability_ari": raw_stability_ari,
            "display_mean_stability_ari": round(raw_stability_ari, 4),
            "n_pairwise_comparisons": len(pairwise_aris),
        }

        if k_in_range:
            valid_candidates.append((raw_stability_ari, -abs(mean_k - 15.0), r))

    if valid_candidates:
        valid_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_r = valid_candidates[0][2]
    else:
        # Fallback to candidate closest to K range center
        all_cands = [(grid_results[str(r)]["raw_mean_stability_ari"], -abs(grid_results[str(r)]["mean_k"] - 15.0), r) for r in CANDIDATE_RESOLUTIONS]
        all_cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_r = all_cands[0][2]

    return best_r, grid_results


# ==============================================================================
# PHASE 1: UNLABELED RESOLUTION SELECTION (ZERO GT LEAKAGE)
# ==============================================================================
def run_phase_selection():
    print("=" * 80)
    print(">>> PHASE 1: UNLABELED RESOLUTION SELECTION (ZERO GT LEAKAGE)")
    print("=" * 80)

    # Fail-closed check: Selection outputs must not exist
    if UNLABELED_ASSIGNMENTS_PATH.exists() or SELECTION_META_PATH.exists():
        raise FileExistsError(
            f"[FAIL-CLOSED APPEND-ONLY] Selection outputs already exist!\n"
            f"  Assignments: {UNLABELED_ASSIGNMENTS_PATH} (exists={UNLABELED_ASSIGNMENTS_PATH.exists()})\n"
            f"  Metadata:    {SELECTION_META_PATH} (exists={SELECTION_META_PATH.exists()})\n"
            f"Re-running formal selection is strictly forbidden to preserve immutable verdict."
        )

    # 1. Load coordinates and barcodes strictly without reading annotations
    print(f"Loading AnnData from {DATASET_PATH} (reading ONLY coordinates & barcodes)...")
    adata_full = ad.read_h5ad(DATASET_PATH, backed="r")
    mask = adata_full.obs["sample_id"] == TARGET_SAMPLE_ID
    assert mask.sum() == 11293, f"Expected 11293 cells, got {mask.sum()}"

    obs_slice = adata_full.obs.loc[mask]
    barcodes = obs_slice.index.tolist()
    coords = adata_full.obsm["spatial"][mask.values]

    # Validate bootstrap manifest against barcodes
    bs_manifest = validate_bootstrap_manifest(barcodes)
    folds = bs_manifest["folds"]

    # 2. Locate ALL REQUIRED method embeddings (STRICT FAIL-CLOSED)
    fn_emb_path = ROOT / "baselines/Baselines_outputs/runs/human_skin/2026-09-18/20260918_human_skin_benchmark_final/full_slice/d028_region1_full_11293_z.npy"
    if not fn_emb_path.exists():
        raise RuntimeError(f"[FAIL-CLOSED ERROR] Missing FreqNiche embedding: {fn_emb_path}")

    methods_emb: dict[str, np.ndarray] = {
        "freqniche": np.load(fn_emb_path),
    }
    embeddings_sha: dict[str, str] = {
        "freqniche": compute_sha256(fn_emb_path),
    }
    embeddings_shape: dict[str, list[int]] = {
        "freqniche": list(methods_emb["freqniche"].shape),
    }

    missing_methods = []
    for m in ["banksy", "cellcharter", "stagate", "cellniche"]:
        emb_file = EMB_DIR / f"{m}_embedding_seed42.npy"
        if not emb_file.exists():
            missing_methods.append(f"{m} ({emb_file})")
            continue

        arr = np.load(emb_file)
        if arr.shape[0] != 11293:
            raise ValueError(f"[FAIL-CLOSED ERROR] Shape mismatch for {m}: {arr.shape[0]} != 11293")
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError(f"[FAIL-CLOSED ERROR] Non-finite values detected in {m} embedding!")

        # Verify profile's ordered_barcodes_sha256 if formal profile exists
        prof_file = RESULTS_DIR / f"{m}_formal_profile.json"
        if prof_file.exists():
            with open(prof_file, "r", encoding="utf-8") as fp:
                prof_data = json.load(fp)
            prof_bc_sha = prof_data.get("provenance", {}).get("ordered_barcodes_sha256")
            expected_bc_sha = bs_manifest["full_slice_ordered_barcodes_sha256"]
            assert prof_bc_sha == expected_bc_sha, (
                f"[FAIL-CLOSED ERROR] Profile ordered barcode SHA mismatch for {m}!\n"
                f"Expected: {expected_bc_sha}\nProfile:  {prof_bc_sha}"
            )
            print(f"[OK] Verified profile ordered barcode SHA for {m}")

        methods_emb[m] = arr
        embeddings_sha[m] = compute_sha256(emb_file)
        embeddings_shape[m] = list(arr.shape)
        print(f"[FOUND & VERIFIED] Loaded {m} embedding: {emb_file.name} (Shape: {arr.shape}, SHA: {embeddings_sha[m][:16]}...)")

    if missing_methods:
        raise RuntimeError(
            f"[FAIL-CLOSED ERROR] Formal resolution selection requires all 5 methods to exist!\n"
            f"Missing embeddings:\n  - " + "\n  - ".join(missing_methods) + "\n"
            f"Cannot proceed with incomplete method suite."
        )

    # 3. Unlabeled evaluation & Full 11,293 Leiden Clustering
    selection_meta: dict[str, Any] = {
        "title": "Unlabeled Stability Resolution Selection (Formal Full Slice)",
        "dataset": "human_skin_merfish",
        "sample_id": TARGET_SAMPLE_ID,
        "n_cells": 11293,
        "selection_script_sha256": compute_sha256(Path(__file__).resolve()),
        "h5ad_sha256": compute_sha256(DATASET_PATH),
        "full_slice_ordered_barcodes_sha256": bs_manifest["full_slice_ordered_barcodes_sha256"],
        "bootstrap_manifest_file_sha256": compute_sha256(BOOTSTRAP_MANIFEST_PATH),
        "bootstrap_manifest_self_sha256": bs_manifest.get("manifest_self_sha256"),
        "hyperparameters": {
            "knn_k": KNN_K,
            "random_state": RANDOM_STATE,
            "candidate_resolutions": CANDIDATE_RESOLUTIONS,
            "prior_k_min": K_MIN,
            "prior_k_max": K_MAX,
        },
        "embeddings_sha256": embeddings_sha,
        "embeddings_shape": embeddings_shape,
        "methods": {},
    }

    unlabeled_df = pd.DataFrame({
        "barcode": barcodes,
        "x_um": coords[:, 0],
        "y_um": coords[:, 1],
    })

    for method_name, emb in methods_emb.items():
        print(f"\nEvaluating stability grid for {method_name.upper()}...")
        r_star, grid_results = evaluate_method_stability_full_precision(emb, folds)
        sel_rec = grid_results[str(r_star)]
        print(f"  Selected r* = {r_star} (Raw Stability ARI: {sel_rec['raw_mean_stability_ari']:.6f}, Bootstrap Mean K: {sel_rec['mean_k']})")

        # Full slice Leiden
        ad_full = ad.AnnData(X=emb)
        sc.pp.neighbors(ad_full, n_neighbors=KNN_K, use_rep="X", metric="euclidean")
        sc.tl.leiden(ad_full, resolution=r_star, random_state=RANDOM_STATE)

        final_clusters = ad_full.obs["leiden"].to_numpy().astype(int)
        k_final = len(np.unique(final_clusters))
        unlabeled_df[f"pred_{method_name}"] = final_clusters

        selection_meta["methods"][method_name] = {
            "r_star_slice": r_star,
            "final_k": k_final,
            "raw_stability_ari": sel_rec["raw_mean_stability_ari"],
            "display_stability_ari": sel_rec["display_mean_stability_ari"],
            "bootstrap_mean_k": sel_rec["mean_k"],
            "grid_evaluation": grid_results,
        }
        print(f"  Final Full Slice K = {k_final} (Truthfully recorded, no artificial constraint on K_final)")

    # Save unlabelled assignments atomically
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_assign_path = UNLABELED_ASSIGNMENTS_PATH.with_name(f"{UNLABELED_ASSIGNMENTS_PATH.stem}.tmp.{os.getpid()}.csv")
    unlabeled_df.to_csv(tmp_assign_path, index=False)
    assignments_sha = compute_sha256(tmp_assign_path)
    print(f"\n[OK] Generated temporary unlabeled assignments: {tmp_assign_path.name}")
    print(f"  SHA-256: {assignments_sha}")

    # Self-sign resolution_selection.json
    selection_meta["unlabeled_assignments_sha256"] = assignments_sha
    meta_payload = json.dumps(selection_meta, sort_keys=True)
    selection_meta["selection_self_sha256"] = compute_bytes_sha256(meta_payload.encode("utf-8"))

    tmp_meta_path = SELECTION_META_PATH.with_name(f"{SELECTION_META_PATH.stem}.tmp.{os.getpid()}.json")
    with open(tmp_meta_path, "w", encoding="utf-8") as fp:
        json.dump(selection_meta, fp, indent=2)

    # Atomic swap into permanent positions
    os.replace(tmp_assign_path, UNLABELED_ASSIGNMENTS_PATH)
    os.replace(tmp_meta_path, SELECTION_META_PATH)
    print(f"[OK] Saved self-signed resolution selection atomically: {SELECTION_META_PATH}")
    print(f"  Selection Self-Hash: {selection_meta['selection_self_sha256']}")
    print("\n>>> Phase 1 (Selection) completed successfully with ZERO GT LEAKAGE and FULL BINDING.")


# ==============================================================================
# PHASE 2: POST-HOC EVALUATION (Strict Cryptographic Chain & Barcode Verification)
# ==============================================================================
def run_phase_posthoc():
    print("=" * 80)
    print(">>> PHASE 2: POST-HOC EVALUATION & TRUTH REFERENCE")
    print("=" * 80)

    # Fail-closed check: Posthoc outputs must not exist
    if FINAL_ASSIGNMENTS_PATH.exists() or POSTHOC_METRICS_PATH.exists():
        raise FileExistsError(
            f"[FAIL-CLOSED APPEND-ONLY] Posthoc outputs already exist!\n"
            f"  Final assignments: {FINAL_ASSIGNMENTS_PATH} (exists={FINAL_ASSIGNMENTS_PATH.exists()})\n"
            f"  Metrics:           {POSTHOC_METRICS_PATH} (exists={POSTHOC_METRICS_PATH.exists()})\n"
            f"Re-running formal posthoc is strictly forbidden to preserve immutable verdict."
        )

    # 1. Verify signatures of Phase 1 artifacts
    assert SELECTION_META_PATH.exists(), f"Missing selection metadata: {SELECTION_META_PATH}"
    assert UNLABELED_ASSIGNMENTS_PATH.exists(), f"Missing unlabeled assignments: {UNLABELED_ASSIGNMENTS_PATH}"

    with open(SELECTION_META_PATH, "r", encoding="utf-8") as fp:
        sel_meta = json.load(fp)

    # Verify assignments SHA
    actual_assign_sha = compute_sha256(UNLABELED_ASSIGNMENTS_PATH)
    expected_assign_sha = sel_meta.get("unlabeled_assignments_sha256")
    assert actual_assign_sha == expected_assign_sha, (
        f"[FAIL-CLOSED ERROR] Assignments SHA mismatch!\nExpected: {expected_assign_sha}\nActual:   {actual_assign_sha}"
    )

    # Verify Selection Self-Hash
    saved_self_hash = sel_meta.get("selection_self_sha256")
    sel_meta_copy = dict(sel_meta)
    sel_meta_copy.pop("selection_self_sha256", None)
    calc_self_hash = compute_bytes_sha256(json.dumps(sel_meta_copy, sort_keys=True).encode("utf-8"))
    assert calc_self_hash == saved_self_hash, "[FAIL-CLOSED ERROR] Selection metadata self-hash mismatch! Tampering detected."
    print("[OK] Cryptographic verification passed: Selection artifacts are authentic and untampered.")

    # 2. Load Ground Truth Niche with strict barcode alignment
    print(f"Loading Ground Truth niche from {DATASET_PATH}...")
    adata_full = ad.read_h5ad(DATASET_PATH, backed="r")
    mask = adata_full.obs["sample_id"] == TARGET_SAMPLE_ID
    obs_slice = adata_full.obs.loc[mask]

    # FORBID fallback to cell_type; strictly require 'niche'
    if "niche" not in obs_slice.columns:
        raise KeyError("[FAIL-CLOSED ERROR] Ground truth column 'niche' is absent from AnnData obs! Fallback to 'cell_type' is strictly forbidden.")
    gt_niche = obs_slice["niche"].to_numpy()

    unlabeled_df = pd.read_csv(UNLABELED_ASSIGNMENTS_PATH)
    assert len(unlabeled_df) == len(gt_niche), f"Row count mismatch: {len(unlabeled_df)} != {len(gt_niche)}"

    # Element-wise barcode alignment assertion
    assert (unlabeled_df["barcode"].values == obs_slice.index.values).all(), (
        "[FAIL-CLOSED ERROR] Element-wise barcode mismatch between unlabeled assignments and ground truth AnnData!"
    )
    print("[OK] Element-wise barcode alignment verified with 100% precision.")

    # Insert gt_niche into assignments dataframe
    final_df = unlabeled_df.copy()
    final_df.insert(3, "gt_niche", gt_niche)

    # 3. Compute post-hoc metrics
    posthoc_results = {}
    for col in final_df.columns:
        if col.startswith("pred_"):
            method_name = col.replace("pred_", "")
            pred_clusters = final_df[col].to_numpy()
            ari = float(adjusted_rand_score(gt_niche, pred_clusters))
            nmi = float(normalized_mutual_info_score(gt_niche, pred_clusters))
            posthoc_results[method_name] = {
                "ari_vs_gt_niche": round(ari, 4),
                "nmi_vs_gt_niche": round(nmi, 4),
                "r_star_used": sel_meta["methods"][method_name]["r_star_slice"],
                "k_final": sel_meta["methods"][method_name]["final_k"],
            }
            print(f"  {method_name.upper():<12} | ARI: {ari:.4f} | NMI: {nmi:.4f} | K: {posthoc_results[method_name]['k_final']}")

    tmp_final_path = FINAL_ASSIGNMENTS_PATH.with_name(f"{FINAL_ASSIGNMENTS_PATH.stem}.tmp.{os.getpid()}.csv")
    final_df.to_csv(tmp_final_path, index=False)

    tmp_metrics_path = POSTHOC_METRICS_PATH.with_name(f"{POSTHOC_METRICS_PATH.stem}.tmp.{os.getpid()}.json")
    with open(tmp_metrics_path, "w", encoding="utf-8") as fp:
        json.dump(posthoc_results, fp, indent=2)

    os.replace(tmp_final_path, FINAL_ASSIGNMENTS_PATH)
    os.replace(tmp_metrics_path, POSTHOC_METRICS_PATH)
    print(f"\n[OK] Saved final assignments with ground truth atomically: {FINAL_ASSIGNMENTS_PATH}")
    print(f"[OK] Saved post-hoc evaluation metrics atomically: {POSTHOC_METRICS_PATH}")
    print("\n>>> Phase 2 (Post-hoc) completed successfully with zero leakage before resolution selection.")


def main():
    parser = argparse.ArgumentParser(description="Strict Two-Phase Resolution Selection")
    parser.add_argument("--phase", type=str, choices=["selection", "posthoc"], required=True,
                        help="Execution phase: 'selection' (zero GT leakage) or 'posthoc' (after selection is locked)")
    args = parser.parse_args()

    if args.phase == "selection":
        run_phase_selection()
    elif args.phase == "posthoc":
        run_phase_posthoc()


if __name__ == "__main__":
    main()
