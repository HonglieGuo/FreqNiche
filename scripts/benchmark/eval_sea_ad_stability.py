"""
Stage 4 Evaluation for FreqNiche-ST v1.0 on SEA-AD MTG MERFISH.

Strict Protocol Alignment:
1. Two-Phase Physical Isolation:
   - Phase 'validation':
     * Verifies 10 bootstrap folds in val_bootstrap_folds.json: exactly 8,000 unique indices in [0, 9999] with SHA-256 match;
     * Loads Validation context only (10,000 centers, niche_valid==True);
     * Extracts Validation embeddings for 3 seeds using frozen best_val_model.pt (whitener=None, is_train=False);
     * Evaluates intra-seed subsampling stability for each seed independently across resolutions [0.2, 0.4, 0.6, 0.8, 1.0];
     * Enforces cluster count constraint [k_min=5, k_max=25];
     * Selects independent r* for each seed;
     * Computes cross-seed pairwise ARI purely as an auxiliary readout;
     * Binds cryptographic SHA-256 hashes of checkpoints, val centers, val folds, val embeddings, evaluation script,
       and all frozen protocol hyper-parameters into resolution_selection.json;
     * Signs resolution_selection.json with payload SHA-256;
     * ZERO Test data or labels are accessed or imported in this phase.
   - Phase 'test':
     * Unlocks only after resolution_selection.json is verified;
     * Enforces non-silent overwrite protection (refuses to silently reuse/overwrite test assets without --overwrite);
     * Re-verifies Checkpoints SHA-256 and evaluation script SHA-256 against resolution_selection.json;
     * Loads Test context (10,000 centers across 5 test donors, niche_valid==True);
     * Builds Test spatial graph (R=50.0 um, max_neighbors=None);
     * Extracts Test embeddings for 3 seeds using frozen best_val_model.pt;
     * Applies each seed's frozen r* to cluster Test centers;
     * Exports per-seed downstream cell assignments (test_assignments_seed_{seed}.csv) containing:
       cell_barcode, donor_id, sample_id, niche, predicted_cluster, seed, r_star;
     * Computes Macro ARI, Macro NMI, Worst-Donor ARI, and ASW against cortical layers {'L1', 'L2/3', 'L4', 'L5', 'L6'};
     * Performs 1,000-iteration Donor Bootstrap for 95% Confidence Intervals;
     * Computes 3-seed summary statistics with ddof=1 (sample standard deviation);
     * Exports final CSV and JSON reports.
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import torch
from torch.utils.data import DataLoader

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.freqniche_st.graph import build_global_spatial_graph
from src.freqniche_st.models.freqniche_model import FreqNicheDiTModel
from src.freqniche_st.dataset import SubgraphDataset, collate_subgraphs
from src.freqniche_st.runtime_paths import resolve_dataset_path, resolve_run_output
from src.freqniche_st.model_config import load_model_config, build_freqniche_model, inference_batch_size, verify_training_config


DATASET_PATH = resolve_dataset_path("FREQNICHE_DATASET_PATH", "datasets/h5ad/sea_ad/sea_ad_mtg_merfish.h5ad")
OUTPUT_DIR = resolve_run_output("outputs/runs/freqniche/sea_ad/2026-09-15/20260915_sea_ad_final")
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
MANIFEST_PATH = OUTPUT_DIR / "cohort_manifest.json"
VAL_CENTERS_PATH = OUTPUT_DIR / "val_centers.json"
VAL_FOLDS_PATH = OUTPUT_DIR / "val_bootstrap_folds.json"
TEST_CENTERS_PATH = OUTPUT_DIR / "test_centers.json"
RESOLUTION_SELECTION_PATH = OUTPUT_DIR / "resolution_selection.json"
MODEL_CONFIG = load_model_config(output_dir=OUTPUT_DIR)

SEEDS = [42, 123, 2024]
FROZEN_RESOLUTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
K_MIN = 5
K_MAX = 25


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_sha256_list(items: List[str]) -> str:
    """Computes SHA-256 of a list of strings matching build_sea_ad_cohorts.py convention."""
    hasher = hashlib.sha256()
    for item in items:
        hasher.update(item.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def extract_embeddings(
    model: FreqNicheDiTModel,
    dataset: SubgraphDataset,
    device: torch.device,
    batch_size: int = 128
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_subgraphs,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    all_z = []
    with torch.no_grad():
        for batch in loader:
            z = model.encoder(
                X_vis=batch["X_vis"].to(device),
                delta_s_vis=batch["delta_s_vis"].to(device),
                vis_pad_mask=batch["vis_pad_mask"].to(device),
                L_vis_cond=batch["L_vis_cond"].to(device),
                H_vis_cond=batch["H_vis_cond"].to(device)
            )
            all_z.append(z.cpu().numpy())
    emb = np.concatenate(all_z, axis=0)
    assert np.all(np.isfinite(emb)), "FATAL: Non-finite embeddings detected!"
    return emb


def select_unsupervised_resolution_for_seed(
    val_emb: np.ndarray,
    bootstrap_folds: Dict[str, Any],
    resolutions: List[float] = FROZEN_RESOLUTIONS,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    seed: int = 42
) -> Tuple[float, Dict[str, Any]]:
    """
    Selects optimal Leiden resolution r* for a single seed via unsupervised
    subsampling stability on the 10 frozen bootstrap folds.
    
    Strictly excludes ground-truth labels.
    Uses [k_min, k_max] cluster constraint with deterministic tie-breaking.
    """
    fold_names = [f"fold_{i}" for i in range(10)]
    fold_indices_list = [np.array(bootstrap_folds[fname]["indices_in_val_centers"], dtype=np.int64) for fname in fold_names]
    n_bootstraps = len(fold_indices_list)

    scores_per_res: Dict[float, float] = {}
    mean_k_per_res: Dict[float, float] = {}
    all_pair_aris_per_res: Dict[float, List[float]] = {}

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
        all_pair_aris_per_res[res] = pair_aris

    # Restrict to resolutions within [k_min, k_max]
    valid_res = [r for r in resolutions if k_min <= mean_k_per_res[r] <= k_max]
    if valid_res:
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
        "valid_resolutions_in_k_bounds": valid_res
    }
    return float(best_res), report


def evaluate_donor_level_metrics(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    donor_ids: np.ndarray
) -> Dict[str, Any]:
    """Computes Macro ARI, Macro NMI, and Worst-donor ARI across unique test donors."""
    unique_donors = np.unique(donor_ids)
    donor_aris = []
    donor_nmis = []

    for d in unique_donors:
        mask = (donor_ids == d)
        y_true_d = true_labels[mask]
        y_pred_d = pred_labels[mask]

        ari_d = float(adjusted_rand_score(y_true_d, y_pred_d))
        nmi_d = float(normalized_mutual_info_score(y_true_d, y_pred_d))
        donor_aris.append(ari_d)
        donor_nmis.append(nmi_d)

    return {
        "macro_ari": float(np.mean(donor_aris)),
        "macro_nmi": float(np.mean(donor_nmis)),
        "worst_donor_ari": float(np.min(donor_aris)),
        "donor_aris": {str(d): float(a) for d, a in zip(unique_donors, donor_aris)},
        "donor_nmis": {str(d): float(n) for d, n in zip(unique_donors, donor_nmis)}
    }


def run_validation_phase(device: torch.device, overwrite: bool = False):
    """
    Phase 1: Validation Embedding Extraction & Intra-Seed Stability Selection.
    Strictly isolated: No test data or test labels loaded.
    """
    print("=" * 80)
    print("PHASE 1: VALIDATION STABILITY SELECTION & r* FREEZING")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 80)

    if RESOLUTION_SELECTION_PATH.exists() and not overwrite:
        raise RuntimeError(
            f"Resolution selection file already exists at {RESOLUTION_SELECTION_PATH}!\n"
            "Pass --overwrite to explicitly re-run Phase 1 Validation."
        )

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Audit Prerequisites and 10-Fold Invariants
    print("\n[1/5] Auditing Checkpoints, Cohort Manifest and 10-Fold Bootstrap Partition ...")
    assert MANIFEST_PATH.exists(), f"Missing cohort manifest: {MANIFEST_PATH}"
    assert VAL_CENTERS_PATH.exists(), f"Missing validation centers: {VAL_CENTERS_PATH}"
    assert VAL_FOLDS_PATH.exists(), f"Missing validation folds: {VAL_FOLDS_PATH}"

    with open(VAL_CENTERS_PATH, "r", encoding="utf-8") as f:
        val_cohort = json.load(f)
    with open(VAL_FOLDS_PATH, "r", encoding="utf-8") as f:
        val_folds = json.load(f)

    n_val = len(val_cohort["cell_barcodes"])
    assert n_val == 10000, f"Expected 10,000 val centers, got {n_val}"
    assert len(val_folds) == 10, f"Expected 10 bootstrap folds, got {len(val_folds)}"

    # Strict fold verification
    val_barcodes_arr = np.array(val_cohort["cell_barcodes"])
    for fname in [f"fold_{i}" for i in range(10)]:
        assert fname in val_folds, f"Missing {fname} in val_bootstrap_folds.json"
        f_indices = np.array(val_folds[fname]["indices_in_val_centers"], dtype=np.int64)
        assert len(f_indices) == 8000, f"{fname} has {len(f_indices)} cells, expected 8,000"
        assert len(np.unique(f_indices)) == 8000, f"{fname} has duplicate indices"
        assert np.min(f_indices) >= 0 and np.max(f_indices) <= 9999, f"{fname} index out of [0, 9999] bounds"

        # Check barcode SHA-256 hash match
        f_barcodes = val_barcodes_arr[f_indices].tolist()
        f_sha = compute_sha256_list(f_barcodes)
        assert f_sha == val_folds[fname]["sha256"], f"{fname} SHA-256 mismatch! Computed {f_sha}, expected {val_folds[fname]['sha256']}"

    print(f" -> Validation centers verified: {n_val:,} cells.")
    print(" -> All 10 bootstrap folds verified: exactly 8,000 unique indices in [0, 9999] with matching SHA-256.")

    ckpt_paths = {}
    ckpt_shas = {}
    for s in SEEDS:
        p = CHECKPOINTS_DIR / f"seed_{s}" / "best_val_model.pt"
        hist_p = CHECKPOINTS_DIR / f"seed_{s}" / "training_history.json"
        assert p.exists() and hist_p.exists(), f"Missing checkpoint for seed {s}: {p}"
        with open(hist_p, "r", encoding="utf-8") as f:
            hist = json.load(f)
        verify_training_config(hist, MODEL_CONFIG, asset=str(hist_p))
        real_sha = compute_file_sha256(p)
        assert real_sha == hist["checkpoint_sha256"], f"SHA mismatch for seed {s} checkpoint!"
        ckpt_paths[s] = p
        ckpt_shas[s] = real_sha
        print(f" -> Seed {s} checkpoint verified (Best Epoch {hist['best_epoch']}, Val Loss: {hist['best_val_loss']:.4f}, SHA: {real_sha[:16]}...)")

    # 2. Load AnnData for Validation Context ONLY
    print(f"\n[2/5] Loading Validation Context from {DATASET_PATH} ...")
    adata = sc.read_h5ad(DATASET_PATH)
    val_mask = (adata.obs["split"] == "validation").to_numpy()
    val_ctx_indices = np.where(val_mask)[0]
    print(f" -> Extracted Validation context: {len(val_ctx_indices):,} cells.")

    print(" -> Applying frozen preprocessing: normalize_total(1e4) + log1p() ...")
    adata_val = adata[val_ctx_indices].copy()
    sc.pp.normalize_total(adata_val, target_sum=1e4)
    sc.pp.log1p(adata_val)

    X_val_full = adata_val.X
    if sp.issparse(X_val_full):
        X_val_full = X_val_full.toarray()
    X_val_full = np.asarray(X_val_full, dtype=np.float32)
    coords_val_full = adata.obsm["spatial"][val_ctx_indices].astype(np.float32)
    val_sample_ids = adata.obs["sample_id"].iloc[val_ctx_indices].to_numpy()

    print(" -> Building global spatial graph for Validation context (R=50.0 um, max_neighbors=None) ...")
    P_val, A_val = build_global_spatial_graph(coords=coords_val_full, sample_ids=val_sample_ids, radius=50.0, max_neighbors=None)
    row_idx, col_idx = A_val.nonzero()
    assert int((val_sample_ids[row_idx] != val_sample_ids[col_idx]).sum()) == 0, "Cross-sample edges detected in Val graph!"

    # Map global indices to local context indices
    val_global_to_local = {g: l for l, g in enumerate(val_ctx_indices)}
    val_center_local_indices = np.array([val_global_to_local[g] for g in val_cohort["global_indices"]], dtype=np.int64)

    val_dataset = SubgraphDataset(
        center_indices=val_center_local_indices,
        A_global=A_val,
        P_global=P_val,
        X_global=X_val_full,
        coords_global=coords_val_full,
        whitener=None,
        k_max=MODEL_CONFIG["k_max"],
        m_max=MODEL_CONFIG["m_max"],
        is_train=False
    )

    # 3. Extract Validation Embeddings
    print("\n[3/5] Extracting Validation Embeddings (10,000 centers) across 3 Seeds ...")
    val_embeddings = {}
    val_embedding_shas = {}
    for s in SEEDS:
        emb_file = EMBEDDINGS_DIR / f"val_embeddings_seed_{s}.npy"
        if emb_file.exists() and not overwrite:
            print(f" -> Loading existing Validation embeddings for Seed {s} from {emb_file.name} ...")
            z_val = np.load(emb_file)
            assert z_val.shape == (10000, MODEL_CONFIG["d_model"]) and np.all(np.isfinite(z_val))
        else:
            print(f" -> Inferring Validation embeddings for Seed {s} ...")
            model = build_freqniche_model(180, MODEL_CONFIG).to(device)
            state_dict = torch.load(ckpt_paths[s], map_location=device)
            model.load_state_dict(state_dict)
            z_val = extract_embeddings(model, val_dataset, device=device,
                                       batch_size=inference_batch_size(OUTPUT_DIR, 128))
            np.save(emb_file, z_val)
        val_embeddings[s] = z_val
        val_embedding_shas[s] = compute_file_sha256(emb_file)

    # 4. Independent Unsupervised Stability Selection per Seed
    print(f"\n[4/5] Running Independent Unsupervised Stability Selection across {FROZEN_RESOLUTIONS} ...")
    print(f"Cluster count constraints: [{K_MIN}, {K_MAX}] clusters.")
    seed_selection_reports = {}
    frozen_r_stars = {}

    for s in SEEDS:
        print(f"\n--- Seed {s} Stability Sweep on 10 Frozen Folds ---")
        best_r, report = select_unsupervised_resolution_for_seed(
            val_emb=val_embeddings[s],
            bootstrap_folds=val_folds,
            resolutions=FROZEN_RESOLUTIONS,
            k_min=K_MIN,
            k_max=K_MAX,
            seed=s
        )
        frozen_r_stars[s] = best_r
        seed_selection_reports[s] = report

        print(f"{'Resolution (r)':<16} | {'Stability Score (Mean Pairwise ARI)':<36} | {'Mean Clusters (K)':<18} | {'Valid in [5, 25]'}")
        print("-" * 85)
        for r in FROZEN_RESOLUTIONS:
            score = report["stability_scores"][str(r)]
            mean_k = report["mean_clusters"][str(r)]
            is_valid = r in report["valid_resolutions_in_k_bounds"]
            marker = "(* SELECTED r*)" if r == best_r else ""
            print(f"{r:<16.2f} | {score:<36.4f} | {mean_k:<18.1f} | {str(is_valid):<14} {marker}")
        print(f" -> Seed {s} Optimal Resolution Locked: r* = {best_r} (Stability Score: {report['best_stability_score']:.4f})")

    # 5. Auxiliary Cross-Seed Pairwise ARI Readout (for transparency)
    print("\n[5/5] Computing Auxiliary Cross-Seed Pairwise ARI on full Validation set ...")
    auxiliary_cross_seed_aris = {}
    val_full_clusterings = {s: {} for s in SEEDS}
    for s in SEEDS:
        adata_full = ad.AnnData(X=val_embeddings[s].copy())
        sc.pp.neighbors(adata_full, n_neighbors=15, use_rep="X", random_state=42)
        for r in FROZEN_RESOLUTIONS:
            key = f"leiden_{r}"
            sc.tl.leiden(adata_full, resolution=r, key_added=key, random_state=42)
            val_full_clusterings[s][r] = adata_full.obs[key].to_numpy().astype(int)

    for r in FROZEN_RESOLUTIONS:
        c_42 = val_full_clusterings[42][r]
        c_123 = val_full_clusterings[123][r]
        c_2024 = val_full_clusterings[2024][r]
        pair_aris = [
            float(adjusted_rand_score(c_42, c_123)),
            float(adjusted_rand_score(c_42, c_2024)),
            float(adjusted_rand_score(c_123, c_2024))
        ]
        auxiliary_cross_seed_aris[str(r)] = {
            "mean_pairwise_ari": float(np.mean(pair_aris)),
            "std_pairwise_ari": float(np.std(pair_aris, ddof=1)),
            "pairs": {
                "42_vs_123": pair_aris[0],
                "42_vs_2024": pair_aris[1],
                "123_vs_2024": pair_aris[2]
            }
        }
        print(f"  r = {r:.1f} -> Cross-seed Pairwise ARI: {np.mean(pair_aris):.4f} ± {np.std(pair_aris, ddof=1):.4f}")

    # Build and Sign resolution_selection.json with cryptographic asset bindings
    script_path = Path(__file__).resolve()
    script_sha = compute_file_sha256(script_path)

    selection_manifest = {
        "manifest_version": "2.5",
        "description": "Deterministic Unsupervised Resolution r* Selection on Validation Cohort",
        "dataset": "sea_ad_mtg_merfish",
        "bound_input_assets": {
            "evaluation_script": {
                "path": str(script_path.relative_to(WORKSPACE_ROOT)),
                "sha256": script_sha
            },
            "val_centers_manifest": {
                "path": str(VAL_CENTERS_PATH.relative_to(WORKSPACE_ROOT)),
                "sha256": compute_file_sha256(VAL_CENTERS_PATH)
            },
            "val_bootstrap_folds": {
                "path": str(VAL_FOLDS_PATH.relative_to(WORKSPACE_ROOT)),
                "sha256": compute_file_sha256(VAL_FOLDS_PATH)
            },
            "checkpoints": {
                str(s): {
                    "path": str((CHECKPOINTS_DIR / f"seed_{s}" / "best_val_model.pt").relative_to(WORKSPACE_ROOT)),
                    "sha256": ckpt_shas[s]
                }
                for s in SEEDS
            },
            "val_embeddings": {
                str(s): {
                    "path": str((EMBEDDINGS_DIR / f"val_embeddings_seed_{s}.npy").relative_to(WORKSPACE_ROOT)),
                    "sha256": val_embedding_shas[s]
                }
                for s in SEEDS
            }
        },
        "protocol_contracts": {
            "preprocessing": "sc.pp.normalize_total(target_sum=1e4) + sc.pp.log1p()",
            "spatial_radius_um": 50.0,
            "k_max_subgraph": MODEL_CONFIG["k_max"],
            "embedding_postprocessing": "raw",
            "k_min_clusters": K_MIN,
            "k_max_clusters": K_MAX,
            "resolutions_grid": FROZEN_RESOLUTIONS,
            "whitener": "None (pure inference)",
            "masking": "zero_masking (pure inference)"
        },
        "frozen_r_star_per_seed": {str(s): frozen_r_stars[s] for s in SEEDS},
        "per_seed_reports": {str(s): seed_selection_reports[s] for s in SEEDS},
        "auxiliary_cross_seed_readout": auxiliary_cross_seed_aris,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    payload_str = json.dumps(selection_manifest, sort_keys=True)
    manifest_sha = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
    selection_manifest["selection_manifest_sha256"] = manifest_sha

    with open(RESOLUTION_SELECTION_PATH, "w", encoding="utf-8") as f:
        json.dump(selection_manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE: RESOLUTION SELECTION LOCKED & SIGNED")
    print(f"Output File: {RESOLUTION_SELECTION_PATH}")
    print(f"Payload SHA-256: {manifest_sha}")
    print(f"Locked r* per seed: {frozen_r_stars}")
    print("Zero Test data accessed. Phase 1 concluded successfully.")
    print("=" * 80)


def run_test_phase(device: torch.device, overwrite: bool = False):
    """
    Phase 2: Test Evaluation at Locked r* per Seed.
    Unlocks Test context ONLY after resolution_selection.json is verified.
    """
    print("=" * 80)
    print("PHASE 2: TEST BLIND EVALUATION AT LOCKED r*")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 80)

    # 1. Non-Silent Overwrite Protection
    table4_path = OUTPUT_DIR / "table4_sea_ad_freqniche_results.csv"
    test_json_path = OUTPUT_DIR / "test_evaluation_results.json"
    assignment_paths = [OUTPUT_DIR / f"test_assignments_seed_{s}.csv" for s in SEEDS]
    existing_outputs = [p for p in [table4_path, test_json_path] + assignment_paths if p.exists()]

    if existing_outputs and not overwrite:
        raise RuntimeError(
            f"Existing Test evaluation outputs found: {[p.name for p in existing_outputs]}!\n"
            "Refusing to overwrite without explicit permission. Pass --overwrite to re-evaluate Test."
        )

    # 2. Audit Frozen Resolution Selection and Cryptographic Bindings
    assert RESOLUTION_SELECTION_PATH.exists(), (
        f"Missing resolution selection manifest: {RESOLUTION_SELECTION_PATH}. "
        "Run '--phase validation' first to freeze r* before testing!"
    )
    with open(RESOLUTION_SELECTION_PATH, "r", encoding="utf-8") as f:
        selection_data = json.load(f)

    # Verify SHA-256 integrity of resolution selection manifest
    recorded_sha = selection_data.pop("selection_manifest_sha256")
    calc_sha = hashlib.sha256(json.dumps(selection_data, sort_keys=True).encode("utf-8")).hexdigest()
    assert recorded_sha == calc_sha, f"Resolution selection SHA mismatch: {recorded_sha} vs {calc_sha}"
    selection_data["selection_manifest_sha256"] = recorded_sha

    frozen_r_stars = {int(s): float(r) for s, r in selection_data["frozen_r_star_per_seed"].items()}
    print(f"Verified frozen r* from manifest: {frozen_r_stars}")

    # Re-verify bound checkpoints and script SHA-256
    print("Verifying cryptographic asset bindings against resolution manifest ...")
    script_sha = compute_file_sha256(Path(__file__).resolve())
    expected_script_sha = selection_data["bound_input_assets"]["evaluation_script"]["sha256"]
    assert script_sha == expected_script_sha, (
        f"Evaluation script SHA-256 mismatch since resolution selection! Current: {script_sha}, Expected: {expected_script_sha}"
    )

    ckpt_paths = {}
    for s in SEEDS:
        p = CHECKPOINTS_DIR / f"seed_{s}" / "best_val_model.pt"
        assert p.exists(), f"Missing checkpoint: {p}"
        current_sha = compute_file_sha256(p)
        expected_sha = selection_data["bound_input_assets"]["checkpoints"][str(s)]["sha256"]
        assert current_sha == expected_sha, f"Seed {s} checkpoint SHA-256 mismatch with resolution manifest!"
        ckpt_paths[s] = p

    print(" -> All Checkpoint and script bindings verified 100%.")

    # 3. Load Test Cohort and Prerequisites
    assert TEST_CENTERS_PATH.exists(), f"Missing test centers: {TEST_CENTERS_PATH}"
    with open(TEST_CENTERS_PATH, "r", encoding="utf-8") as f:
        test_cohort = json.load(f)

    n_test = len(test_cohort["cell_barcodes"])
    assert n_test == 10000, f"Expected 10,000 test centers, got {n_test}"

    # 4. Load AnnData and Slice Test Context
    print(f"\n[1/5] Loading Test Context from {DATASET_PATH} ...")
    adata = sc.read_h5ad(DATASET_PATH)
    test_mask = (adata.obs["split"] == "test").to_numpy()
    test_ctx_indices = np.where(test_mask)[0]
    print(f" -> Extracted Test context: {len(test_ctx_indices):,} cells.")

    print(" -> Applying frozen preprocessing: normalize_total(1e4) + log1p() ...")
    adata_test = adata[test_ctx_indices].copy()
    sc.pp.normalize_total(adata_test, target_sum=1e4)
    sc.pp.log1p(adata_test)

    X_test_full = adata_test.X
    if sp.issparse(X_test_full):
        X_test_full = X_test_full.toarray()
    X_test_full = np.asarray(X_test_full, dtype=np.float32)
    coords_test_full = adata.obsm["spatial"][test_ctx_indices].astype(np.float32)
    test_sample_ids_ctx = adata.obs["sample_id"].iloc[test_ctx_indices].to_numpy()

    print(" -> Building global spatial graph for Test context (R=50.0 um, max_neighbors=None) ...")
    P_test, A_test = build_global_spatial_graph(coords=coords_test_full, sample_ids=test_sample_ids_ctx, radius=50.0, max_neighbors=None)
    row_idx, col_idx = A_test.nonzero()
    assert int((test_sample_ids_ctx[row_idx] != test_sample_ids_ctx[col_idx]).sum()) == 0, "Cross-sample edges detected in Test graph!"

    # Map global indices to local context indices
    test_global_to_local = {g: l for l, g in enumerate(test_ctx_indices)}
    test_center_local_indices = np.array([test_global_to_local[g] for g in test_cohort["global_indices"]], dtype=np.int64)

    test_dataset = SubgraphDataset(
        center_indices=test_center_local_indices,
        A_global=A_test,
        P_global=P_test,
        X_global=X_test_full,
        coords_global=coords_test_full,
        whitener=None,
        k_max=MODEL_CONFIG["k_max"],
        m_max=MODEL_CONFIG["m_max"],
        is_train=False
    )

    # 5. Extract Test Embeddings
    print("\n[2/5] Extracting Test Embeddings (10,000 centers) across 3 Seeds ...")
    test_embeddings = {}
    for s in SEEDS:
        emb_file = EMBEDDINGS_DIR / f"test_embeddings_seed_{s}.npy"
        if emb_file.exists() and not overwrite:
            raise RuntimeError(
                f"Existing test embedding file found: {emb_file.name}!\n"
                "Refusing silent reuse. Pass --overwrite to explicitly re-extract or re-evaluate."
            )
        print(f" -> Inferring Test embeddings for Seed {s} ...")
        model = build_freqniche_model(180, MODEL_CONFIG).to(device)
        state_dict = torch.load(ckpt_paths[s], map_location=device)
        model.load_state_dict(state_dict)
        z_test = extract_embeddings(model, test_dataset, device=device,
                                    batch_size=inference_batch_size(OUTPUT_DIR, 128))
        np.save(emb_file, z_test)
        test_embeddings[s] = z_test

    # 6. Evaluate Test at Frozen r* per Seed & Export Downstream Assignments
    print("\n[3/5] Evaluating Test Embeddings at Frozen r* per Seed ...")
    test_global_indices = test_cohort["global_indices"]
    test_barcodes = test_cohort["cell_barcodes"]
    test_sample_ids = adata.obs["sample_id"].iloc[test_global_indices].to_numpy()
    test_donors = adata.obs["donor_id"].iloc[test_global_indices].to_numpy()
    test_true_labels = adata.obs["niche"].iloc[test_global_indices].to_numpy()

    unique_test_donors = np.unique(test_donors)
    assert len(unique_test_donors) == 5, f"Expected 5 test donors, got {len(unique_test_donors)}"

    seed_test_metrics = []
    test_clusterings_at_r_star = {}

    for s in SEEDS:
        r_star = frozen_r_stars[s]
        print(f" -> Evaluating Seed {s} at locked r*={r_star:.1f} ...")
        adata_t = ad.AnnData(X=test_embeddings[s].copy())
        sc.pp.neighbors(adata_t, n_neighbors=15, use_rep="X", random_state=42)
        sc.tl.leiden(adata_t, resolution=r_star, key_added=f"leiden_{r_star}", random_state=42)
        pred_test = adata_t.obs[f"leiden_{r_star}"].to_numpy().astype(int)
        test_clusterings_at_r_star[s] = pred_test

        # Export downstream cell assignments CSV
        df_assignment = pd.DataFrame({
            "cell_barcode": test_barcodes,
            "donor_id": test_donors,
            "sample_id": test_sample_ids,
            "niche": test_true_labels,
            "predicted_cluster": pred_test,
            "seed": s,
            "r_star": r_star
        })
        assignment_out = OUTPUT_DIR / f"test_assignments_seed_{s}.csv"
        df_assignment.to_csv(assignment_out, index=False)
        print(f"    Saved downstream cell assignments to {assignment_out.name}")

        metrics = evaluate_donor_level_metrics(
            pred_labels=pred_test,
            true_labels=test_true_labels,
            donor_ids=test_donors
        )

        # ASW (Bio-conservation against cortical layer ground truth)
        asw_val = float(silhouette_score(test_embeddings[s], test_true_labels, metric="euclidean"))
        metrics["asw"] = asw_val
        metrics["seed"] = s
        metrics["r_star"] = r_star
        metrics["n_clusters"] = int(len(np.unique(pred_test)))
        seed_test_metrics.append(metrics)

        print(f"    Seed {s} (r*={r_star:.1f}) -> Macro ARI: {metrics['macro_ari']:.4f}, Macro NMI: {metrics['macro_nmi']:.4f}, Worst-Donor ARI: {metrics['worst_donor_ari']:.4f}, ASW: {metrics['asw']:.4f}, K: {metrics['n_clusters']}")

    # 7. 1,000-Iteration Donor Bootstrap for 95% Confidence Intervals
    print("\n[4/5] Computing 1,000-Iteration Donor Bootstrap 95% Confidence Intervals ...")
    bootstrap_macro_aris = []
    bootstrap_macro_nmis = []
    bootstrap_worst_aris = []

    rng = np.random.RandomState(42)
    for b in range(1000):
        boot_donors = rng.choice(unique_test_donors, size=len(unique_test_donors), replace=True)
        seed_b_aris = []
        seed_b_nmis = []
        seed_b_worsts = []
        for s in SEEDS:
            d_aris = [adjusted_rand_score(test_true_labels[(test_donors == d)], test_clusterings_at_r_star[s][(test_donors == d)]) for d in boot_donors]
            d_nmis = [normalized_mutual_info_score(test_true_labels[(test_donors == d)], test_clusterings_at_r_star[s][(test_donors == d)]) for d in boot_donors]
            seed_b_aris.append(np.mean(d_aris))
            seed_b_nmis.append(np.mean(d_nmis))
            seed_b_worsts.append(np.min(d_aris))

        bootstrap_macro_aris.append(np.mean(seed_b_aris))
        bootstrap_macro_nmis.append(np.mean(seed_b_nmis))
        bootstrap_worst_aris.append(np.mean(seed_b_worsts))

    ci_macro_ari = (float(np.percentile(bootstrap_macro_aris, 2.5)), float(np.percentile(bootstrap_macro_aris, 97.5)))
    ci_macro_nmi = (float(np.percentile(bootstrap_macro_nmis, 2.5)), float(np.percentile(bootstrap_macro_nmis, 97.5)))
    ci_worst_ari = (float(np.percentile(bootstrap_worst_aris, 2.5)), float(np.percentile(bootstrap_worst_aris, 97.5)))

    macro_aris = [m["macro_ari"] for m in seed_test_metrics]
    macro_nmis = [m["macro_nmi"] for m in seed_test_metrics]
    worst_aris = [m["worst_donor_ari"] for m in seed_test_metrics]
    asws = [m["asw"] for m in seed_test_metrics]
    k_clusters = [m["n_clusters"] for m in seed_test_metrics]

    # Summary with sample standard deviation (ddof=1)
    summary = {
        "dataset": "sea_ad_mtg_merfish",
        "cohort": "test_5_donors_10k_centers",
        "frozen_r_star_per_seed": frozen_r_stars,
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "macro_ari": {
            "mean": float(np.mean(macro_aris)),
            "std": float(np.std(macro_aris, ddof=1)),
            "ci_95_bootstrap_1000": ci_macro_ari,
            "values": macro_aris
        },
        "macro_nmi": {
            "mean": float(np.mean(macro_nmis)),
            "std": float(np.std(macro_nmis, ddof=1)),
            "ci_95_bootstrap_1000": ci_macro_nmi,
            "values": macro_nmis
        },
        "worst_donor_ari": {
            "mean": float(np.mean(worst_aris)),
            "std": float(np.std(worst_aris, ddof=1)),
            "ci_95_bootstrap_1000": ci_worst_ari,
            "values": worst_aris
        },
        "asw": {
            "mean": float(np.mean(asws)),
            "std": float(np.std(asws, ddof=1)),
            "values": asws
        },
        "n_clusters": {
            "mean": float(np.mean(k_clusters)),
            "std": float(np.std(k_clusters, ddof=1)),
            "values": k_clusters
        },
        "per_seed_results": seed_test_metrics
    }

    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 8. Formatted Table 4 Export
    table4_rows = []
    for m in seed_test_metrics:
        table4_rows.append({
            "Method": "FreqNiche-ST v1.0",
            "Seed": m["seed"],
            "r*": m["r_star"],
            "Clusters (K)": m["n_clusters"],
            "Macro ARI": f"{m['macro_ari']:.4f}",
            "Macro NMI": f"{m['macro_nmi']:.4f}",
            "Worst-Donor ARI": f"{m['worst_donor_ari']:.4f}",
            "ASW": f"{m['asw']:.4f}"
        })
    table4_rows.append({
        "Method": "FreqNiche-ST 3-Seed Mean",
        "Seed": "Mean ± Std (ddof=1)",
        "r*": "/".join([str(frozen_r_stars[s]) for s in SEEDS]),
        "Clusters (K)": f"{np.mean(k_clusters):.1f} ± {np.std(k_clusters, ddof=1):.1f}",
        "Macro ARI": f"{np.mean(macro_aris):.4f} ± {np.std(macro_aris, ddof=1):.4f}",
        "Macro NMI": f"{np.mean(macro_nmis):.4f} ± {np.std(macro_nmis, ddof=1):.4f}",
        "Worst-Donor ARI": f"{np.mean(worst_aris):.4f} ± {np.std(worst_aris, ddof=1):.4f}",
        "ASW": f"{np.mean(asws):.4f} ± {np.std(asws, ddof=1):.4f}"
    })
    df_table4 = pd.DataFrame(table4_rows)
    df_table4.to_csv(table4_path, index=False)

    print("\n[5/5] Final Blind Evaluation Table (SEA-AD 5 Test Donors):")
    print("=" * 85)
    print(df_table4.to_string(index=False))
    print("=" * 85)
    print(f"95% Bootstrap CIs (1,000 iterations):")
    print(f"  Macro ARI: [{ci_macro_ari[0]:.4f}, {ci_macro_ari[1]:.4f}]")
    print(f"  Macro NMI: [{ci_macro_nmi[0]:.4f}, {ci_macro_nmi[1]:.4f}]")
    print(f"  Worst-Donor ARI: [{ci_worst_ari[0]:.4f}, {ci_worst_ari[1]:.4f}]")
    print(f"Report saved to: {test_json_path}")
    print(f"Table 4 saved to: {table4_path}")


def main():
    parser = argparse.ArgumentParser(description="W2.5 SEA-AD Stage 4 Evaluation Protocol")
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["validation", "test"],
        help="Evaluation phase: 'validation' to freeze r*, or 'test' to evaluate Test cohort."
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwrite of existing evaluation outputs/embeddings.")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.phase == "validation":
        run_validation_phase(device=device, overwrite=args.overwrite)
    elif args.phase == "test":
        run_test_phase(device=device, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
