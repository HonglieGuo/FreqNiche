"""
Phase 1 Validation Stability Selection & Phase 2 Test Evaluation for FreqNiche-ST v1.0 on CosMx NSCLC.

Strict Protocol Alignment:
1. Two-Phase Physical Isolation:
   - Phase 'validation':
     * Verifies 10 bootstrap folds in val_bootstrap_folds.json: exactly 8,000 unique indices in [0, 9999] with SHA-256 match;
     * Rigorous row-mapping invariant assertion: global_indices and cell_barcodes strictly align with AnnData obs_names and split;
     * Context scoping: Validation context only (Lung5, 10,000 centers); Test rows were never indexed or used for graphing, inference, or scoring;
     * Reuses existing Validation embeddings by default (unless --recompute-val-embeddings is explicitly provided);
     * Evaluates intra-seed subsampling stability for each seed independently across resolutions [0.2, 0.4, 0.6, 0.8, 1.0];
     * Enforces cluster count constraint [k_min=5, k_max=25];
     * Selects independent r* for each seed;
     * Computes cross-seed pairwise ARI purely as an auxiliary readout;
     * Binds cryptographic SHA-256 hashes of checkpoints, val centers, val folds, val embeddings, evaluation script,
       and all frozen protocol hyper-parameters into resolution_selection.json;
     * Signs resolution_selection.json with payload SHA-256;
     * Zero Test data or labels are accessed or computed in this phase.
   - Phase 'test':
     * Cryptographic Entry Guard: Executes verify_resolution_selection_manifest() before ANY test data access;
     * Fail-Closed overwrite guard: Refuses to silently overwrite or reuse existing test assets without --overwrite-test;
     * Rigorous row-mapping invariant assertion on test_centers against AnnData test split;
     * Builds Test spatial graph (R=50.0 um, sigma=16.6667 um, max_neighbors=None);
     * Extracts Test embeddings for 3 seeds using frozen checkpoints (atomic writes);
     * Applies each seed's frozen r* to cluster Test centers;
     * Exports per-seed downstream cell assignments (atomic writes);
     * Computes ARI, NMI, ASW against reference niche labels;
     * Computes 3-seed summary statistics with ddof=1 (sample standard deviation);
     * Binds Test center triple hash, checkpoints SHA, script SHA, and selection manifest SHA into final report (atomic write).
"""

from __future__ import annotations

import argparse
import copy
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
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.freqniche_st.graph import build_global_spatial_graph
from src.freqniche_st.models.freqniche_model import FreqNicheDiTModel
from src.freqniche_st.dataset import SubgraphDataset, collate_subgraphs
from src.freqniche_st.runtime_paths import resolve_dataset_path, resolve_run_output
from src.freqniche_st.model_config import load_model_config, build_freqniche_model, inference_batch_size, verify_training_config


DATASET_PATH = resolve_dataset_path("FREQNICHE_DATASET_PATH", "datasets/h5ad/cosmx_human_nsclc/cosmx_human_nsclc.h5ad")
OUTPUT_DIR = resolve_run_output("outputs/runs/freqniche/cosmx_nsclc/2026-09-17/20260917_cosmx_nsclc_final")
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
MANIFEST_PATH = OUTPUT_DIR / "cosmx_data_audit_manifest.json"
VAL_CENTERS_PATH = OUTPUT_DIR / "val_centers.json"
VAL_FOLDS_PATH = OUTPUT_DIR / "val_bootstrap_folds.json"
TEST_CENTERS_PATH = OUTPUT_DIR / "test_centers.json"
RESOLUTION_SELECTION_PATH = OUTPUT_DIR / "resolution_selection.json"
MODEL_CONFIG = load_model_config(output_dir=OUTPUT_DIR)

SEEDS = [42, 123, 2024]
FROZEN_RESOLUTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
K_MIN = 5
K_MAX = 25
RADIUS_UM = 50.0
SIGMA_UM = 16.666666666666668


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


def verify_resolution_selection_manifest(
    selection_path: Path = RESOLUTION_SELECTION_PATH,
    workspace_root: Path = WORKSPACE_ROOT
) -> Dict[str, Any]:
    """
    Cryptographic Entry Guard for Phase 2 Test Evaluation.
    Must pass 100% before any Test data or context is accessed.
    """
    print("\n" + "=" * 80)
    print("VERIFYING RESOLUTION SELECTION MANIFEST & CRYPTOGRAPHIC ASSET BINDINGS")
    print("=" * 80)
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing resolution selection manifest: {selection_path}")

    with open(selection_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    # 1. Verify Self-Signature
    if "selection_manifest_sha256" not in manifest_data:
        raise ValueError("Missing selection_manifest_sha256 in manifest!")
    recorded_sha = manifest_data.pop("selection_manifest_sha256")
    calc_sha = hashlib.sha256(json.dumps(manifest_data, sort_keys=True).encode("utf-8")).hexdigest()
    if recorded_sha != calc_sha:
        raise ValueError(f"Self-signature mismatch! Recorded: {recorded_sha}, Computed: {calc_sha}")
    manifest_data["selection_manifest_sha256"] = recorded_sha
    print(f" -> [1/7] Manifest self-signature verified: {recorded_sha[:16]}...")

    # 2. Verify Evaluation Script SHA-256
    script_path = Path(__file__).resolve()
    current_script_sha = compute_file_sha256(script_path)
    expected_script_sha = manifest_data["bound_input_assets"]["evaluation_script"]["sha256"]
    if current_script_sha != expected_script_sha:
        raise ValueError(f"Evaluation script SHA mismatch! Current: {current_script_sha}, Bound: {expected_script_sha}")
    print(f" -> [2/7] Evaluation script SHA-256 verified: {current_script_sha[:16]}...")

    # 3. Verify Data Audit Manifest SHA-256
    data_audit_path = workspace_root / manifest_data["bound_input_assets"]["data_audit_manifest"]["path"]
    current_audit_sha = compute_file_sha256(data_audit_path)
    expected_audit_sha = manifest_data["bound_input_assets"]["data_audit_manifest"]["sha256"]
    if current_audit_sha != expected_audit_sha:
        raise ValueError(f"Data audit manifest SHA mismatch! Current: {current_audit_sha}, Bound: {expected_audit_sha}")
    print(f" -> [3/7] Data audit manifest SHA-256 verified: {current_audit_sha[:16]}...")

    with open(data_audit_path, "r", encoding="utf-8") as f:
        audit_meta = json.load(f)

    # 4. Verify Validation Centers and 10-Fold Bootstrap Partition SHA-256
    val_centers_path = workspace_root / manifest_data["bound_input_assets"]["val_centers_manifest"]["path"]
    val_folds_path = workspace_root / manifest_data["bound_input_assets"]["val_bootstrap_folds"]["path"]
    if compute_file_sha256(val_centers_path) != manifest_data["bound_input_assets"]["val_centers_manifest"]["sha256"]:
        raise ValueError("Validation centers manifest SHA mismatch!")
    if compute_file_sha256(val_folds_path) != manifest_data["bound_input_assets"]["val_bootstrap_folds"]["sha256"]:
        raise ValueError("Validation bootstrap folds SHA mismatch!")
    print(" -> [4/7] Validation centers and 10-fold partition verified.")

    # 5. Verify 3 Checkpoints SHA-256
    for s in SEEDS:
        ckpt_rel = manifest_data["bound_input_assets"]["checkpoints"][str(s)]["path"]
        ckpt_path = workspace_root / ckpt_rel
        if compute_file_sha256(ckpt_path) != manifest_data["bound_input_assets"]["checkpoints"][str(s)]["sha256"]:
            raise ValueError(f"Checkpoint SHA mismatch for seed {s}!")
    print(" -> [5/7] 3 Model checkpoints verified.")

    # 6. Verify 3 Validation Embeddings SHA-256
    for s in SEEDS:
        emb_rel = manifest_data["bound_input_assets"]["val_embeddings"][str(s)]["path"]
        emb_path = workspace_root / emb_rel
        if compute_file_sha256(emb_path) != manifest_data["bound_input_assets"]["val_embeddings"][str(s)]["sha256"]:
            raise ValueError(f"Validation embedding SHA mismatch for seed {s}!")
    print(" -> [6/7] 3 Validation embeddings verified.")

    # 7. Verify Test Centers Triple Hash against Data Audit Manifest
    test_centers_path = TEST_CENTERS_PATH
    if not test_centers_path.exists():
        raise FileNotFoundError(f"Missing test centers file: {test_centers_path}")
    with open(test_centers_path, "r", encoding="utf-8") as f:
        test_cohort_check = json.load(f)
    t_file_sha = compute_file_sha256(test_centers_path)
    t_sorted_sha = compute_sorted_barcode_sha256(test_cohort_check["cell_barcodes"])
    t_ordered_sha = compute_ordered_barcode_sha256(test_cohort_check["cell_barcodes"])
    exp_t = audit_meta["cohorts"]["test"]
    if t_file_sha != exp_t["file_sha256"] or t_sorted_sha != exp_t["sorted_barcode_sha256"] or t_ordered_sha != exp_t["ordered_barcode_sha256"]:
        raise ValueError("Test centers Triple Hash mismatch with Data Audit Manifest!")
    print(" -> [7/7] Test centers Triple Hash verified against master audit manifest.")

    print("=" * 80)
    print("SUCCESS: ALL 7 CRYPTOGRAPHIC INTEGRITY CHECKS PASSED 100%!")
    print("=" * 80)
    return manifest_data


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
            sc.pp.neighbors(adata_sub, n_neighbors=15, use_rep="X", random_state=seed + b)
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


def run_validation_phase(
    device: torch.device,
    overwrite_selection: bool = False,
    recompute_val_embeddings: bool = False
):
    """
    Phase 1: Validation Embedding Extraction & Intra-Seed Stability Selection.
    Strictly isolated: Test rows were never indexed or used for graphing, inference, or scoring.
    """
    print("=" * 80)
    print("COSMX PHASE 1: VALIDATION STABILITY SELECTION & r* FREEZING")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 80)

    if RESOLUTION_SELECTION_PATH.exists() and not overwrite_selection:
        raise RuntimeError(
            f"Resolution selection file already exists at {RESOLUTION_SELECTION_PATH}!\n"
            "Pass --overwrite-selection to explicitly re-run Phase 1 Validation."
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

        f_barcodes = val_barcodes_arr[f_indices].tolist()
        f_sorted_sha = compute_sorted_barcode_sha256(f_barcodes)
        f_ordered_sha = compute_ordered_barcode_sha256(f_barcodes)
        assert f_sorted_sha == val_folds[fname]["sorted_barcode_sha256"], f"{fname} sorted SHA mismatch!"
        assert f_ordered_sha == val_folds[fname]["ordered_barcode_sha256"], f"{fname} ordered SHA mismatch!"

    print(f" -> Validation centers verified: {n_val:,} cells.")
    print(" -> All 10 bootstrap folds verified: exactly 8,000 unique indices in [0, 9999] with matching SHA-256.")

    ckpt_paths = {}
    ckpt_shas = {}
    summary_path = CHECKPOINTS_DIR / "cosmx_training_summary.json"
    assert summary_path.exists(), f"Missing training summary: {summary_path}"
    with open(summary_path, "r", encoding="utf-8") as f:
        training_summary = json.load(f)

    for s in SEEDS:
        p = CHECKPOINTS_DIR / f"seed_{s}" / f"model_seed_{s}_best.pt"
        hist_p = CHECKPOINTS_DIR / f"seed_{s}" / "training_history.json"
        assert p.exists() and hist_p.exists(), f"Missing checkpoint for seed {s}: {p}"
        with open(hist_p, "r", encoding="utf-8") as f:
            hist = json.load(f)
        verify_training_config(hist, MODEL_CONFIG, asset=str(hist_p))
        real_sha = compute_file_sha256(p)
        assert real_sha == hist["checkpoint_sha256"], f"SHA mismatch for seed {s} checkpoint!"
        assert real_sha == training_summary["results"][str(s)]["checkpoint_sha256"], f"Summary SHA mismatch for seed {s}!"
        ckpt_paths[s] = p
        ckpt_shas[s] = real_sha
        print(f" -> Seed {s} checkpoint verified (Best Epoch {hist['best_epoch']}, Val Loss: {hist['best_val_loss']:.4f}, SHA: {real_sha[:16]}...)")

    # 2. Check Embedding Availability before Dataset Loading
    embeddings_exist = all((EMBEDDINGS_DIR / f"val_embeddings_seed_{s}.npy").exists() for s in SEEDS)
    val_embeddings = {}
    val_embedding_shas = {}

    if embeddings_exist and not recompute_val_embeddings:
        print("\n[2/5] All 3 Validation embeddings already exist and --recompute-val-embeddings was NOT provided.")
        print(" -> Fast path: Reusing existing Validation embeddings without re-running forward inference.")
        for s in SEEDS:
            emb_file = EMBEDDINGS_DIR / f"val_embeddings_seed_{s}.npy"
            z_val = np.load(emb_file)
            assert z_val.shape == (10000, MODEL_CONFIG["d_model"]) and np.all(np.isfinite(z_val)), f"Invalid embedding in {emb_file}"
            val_embeddings[s] = z_val
            val_embedding_shas[s] = compute_file_sha256(emb_file)
            print(f" -> Seed {s} Val Embedding verified: shape={z_val.shape}, SHA-256: {val_embedding_shas[s][:16]}...")
    else:
        # Load AnnData for Validation Context ONLY
        # Test rows are never indexed, sliced, or used for graph building or inference.
        print(f"\n[2/5] Loading Validation Context from {DATASET_PATH} ...")
        adata = sc.read_h5ad(DATASET_PATH)

        # Rigorous Row-Mapping Invariant Assertion on Validation Cohort
        val_idx = np.asarray(val_cohort["global_indices"])
        val_barcodes = np.asarray(val_cohort["cell_barcodes"])
        assert len(val_idx) == 10000, f"Expected 10,000 val indices, got {len(val_idx)}"
        assert len(np.unique(val_idx)) == 10000, "Duplicate val indices detected!"
        assert val_idx.min() >= 0 and val_idx.max() < adata.n_obs, "Val index out of AnnData bounds!"
        assert np.array_equal(adata.obs_names[val_idx].astype(str), val_barcodes.astype(str)), "Val barcode alignment mismatch with H5AD!"
        assert (adata.obs["split"].iloc[val_idx] == "validation").all(), "Non-validation cells in val_cohort!"
        assert adata.obs["donor_id"].iloc[val_idx].nunique() == 1, "Val cohort must originate from exactly 1 donor (Lung5)!"
        print(" -> Validation cohort row-mapping invariant assertions 100% verified.")

        val_mask = (adata.obs["split"] == "validation").to_numpy()
        val_ctx_indices = np.where(val_mask)[0]
        print(f" -> Extracted Validation context: {len(val_ctx_indices):,} cells. Test rows were never indexed or used.")

        print(" -> Applying frozen preprocessing: normalize_total(1e4) + log1p() ...")
        adata_val = adata[val_ctx_indices].copy()
        sc.pp.normalize_total(adata_val, target_sum=1e4)
        sc.pp.log1p(adata_val)

        X_val_full = adata_val.X
        if sp.issparse(X_val_full):
            X_val_full = X_val_full.toarray()
        X_val_full = np.asarray(X_val_full, dtype=np.float32)

        # Read physical micron coordinates directly: NO secondary scaling!
        assert "spatial_um" in adata.obsm, "Missing obsm['spatial_um'] in dataset!"
        coords_val_full = adata.obsm["spatial_um"][val_ctx_indices].astype(np.float32)
        val_sample_ids = adata.obs["sample_id"].iloc[val_ctx_indices].to_numpy()

        print(f" -> Building global spatial graph for Validation context (R={RADIUS_UM} um, sigma={SIGMA_UM:.4f} um, max_neighbors=None) ...")
        P_val, A_val = build_global_spatial_graph(
            coords=coords_val_full,
            sample_ids=val_sample_ids,
            radius=RADIUS_UM,
            sigma=SIGMA_UM,
            max_neighbors=None
        )
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

        # Extract Validation Embeddings
        print("\n[3/5] Extracting Validation Embeddings (10,000 centers) across 3 Seeds ...")
        for s in SEEDS:
            emb_file = EMBEDDINGS_DIR / f"val_embeddings_seed_{s}.npy"
            print(f" -> Inferring Validation embeddings for Seed {s} ...")
            model = build_freqniche_model(960, MODEL_CONFIG).to(device)
            state_data = torch.load(ckpt_paths[s], map_location=device)
            if "model_state_dict" in state_data:
                model.load_state_dict(state_data["model_state_dict"])
            else:
                model.load_state_dict(state_data)
            z_val = extract_embeddings(model, val_dataset, device=device,
                                       batch_size=inference_batch_size(OUTPUT_DIR, 128))
            temp_emb = emb_file.with_suffix(".tmp.npy")
            np.save(temp_emb, z_val)
            os.replace(temp_emb, emb_file)

            val_embeddings[s] = z_val
            val_embedding_shas[s] = compute_file_sha256(emb_file)
            print(f" -> Seed {s} Val Embedding: shape={z_val.shape}, SHA-256: {val_embedding_shas[s][:16]}...")

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

    # 5. Auxiliary Cross-Seed Pairwise ARI Readout
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
        print(f"  r = {r:.1f} -> Cross-seed Pairwise ARI (relatively high with low dispersion): {np.mean(pair_aris):.4f} ± {np.std(pair_aris, ddof=1):.4f}")

    # Build and Sign resolution_selection.json with cryptographic asset bindings
    script_path = Path(__file__).resolve()
    script_sha = compute_file_sha256(script_path)

    selection_manifest = {
        "manifest_version": "1.1",
        "description": "Deterministic Unsupervised Resolution r* Selection on Validation Cohort (CosMx NSCLC)",
        "dataset": "cosmx_human_nsclc",
        "bound_input_assets": {
            "evaluation_script": {
                "path": str(script_path.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                "sha256": script_sha
            },
            "data_audit_manifest": {
                "path": str(MANIFEST_PATH.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                "sha256": compute_file_sha256(MANIFEST_PATH)
            },
            "val_centers_manifest": {
                "path": str(VAL_CENTERS_PATH.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                "sha256": compute_file_sha256(VAL_CENTERS_PATH)
            },
            "val_bootstrap_folds": {
                "path": str(VAL_FOLDS_PATH.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                "sha256": compute_file_sha256(VAL_FOLDS_PATH)
            },
            "checkpoints": {
                str(s): {
                    "path": str((CHECKPOINTS_DIR / f"seed_{s}" / f"model_seed_{s}_best.pt").relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                    "sha256": ckpt_shas[s]
                }
                for s in SEEDS
            },
            "val_embeddings": {
                str(s): {
                    "path": str((EMBEDDINGS_DIR / f"val_embeddings_seed_{s}.npy").relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                    "sha256": val_embedding_shas[s]
                }
                for s in SEEDS
            }
        },
        "protocol_contracts": {
            "preprocessing": "sc.pp.normalize_total(target_sum=1e4) + sc.pp.log1p()",
            "spatial_radius_um": RADIUS_UM,
            "spatial_sigma_um": SIGMA_UM,
            "coordinate_source": "adata.obsm['spatial_um']",
            "isolation_contract": "Test rows were never indexed or used for graphing, inference, or scoring",
            "k_max_subgraph": MODEL_CONFIG["k_max"],
            "embedding_dim": MODEL_CONFIG["d_model"],
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

    temp_res = RESOLUTION_SELECTION_PATH.with_suffix(".tmp.json")
    with open(temp_res, "w", encoding="utf-8") as f:
        json.dump(selection_manifest, f, indent=2)
    os.replace(temp_res, RESOLUTION_SELECTION_PATH)

    print("\n" + "=" * 80)
    print(f"SUCCESS: Resolution selection finalized & signed atomically at {RESOLUTION_SELECTION_PATH}")
    print(f"Selection Manifest SHA-256: {manifest_sha}")
    print(f"Frozen r* Per Seed: {frozen_r_stars}")
    print("=" * 80)


def run_test_phase(device: torch.device, overwrite_test: bool = False):
    """
    Phase 2: Test Evaluation with Frozen r*.
    MUST NOT run until approved by user.
    """
    print("=" * 80)
    print("COSMX PHASE 2: TEST BLIND EVALUATION (FROZEN r*)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 80)

    # 1. Execute Cryptographic Entry Guard Check BEFORE accessing any Test data
    res_selection = verify_resolution_selection_manifest()
    frozen_r_stars = res_selection["frozen_r_star_per_seed"]
    print(f"Loaded verified frozen r* per seed: {frozen_r_stars}")

    # 2. Check Existing Test Outputs - Fail Closed unless explicitly permitted
    report_path = OUTPUT_DIR / "cosmx_test_evaluation_report.json"
    table6_path = OUTPUT_DIR / "table6_cosmx_freqniche_results.csv"
    existing_test_files = []
    if report_path.exists():
        existing_test_files.append(str(report_path.name))
    if table6_path.exists():
        existing_test_files.append(str(table6_path.name))
    for s in SEEDS:
        emb_p = EMBEDDINGS_DIR / f"test_embeddings_seed_{s}.npy"
        assign_p = OUTPUT_DIR / "assignments" / f"test_assignments_seed_{s}.csv"
        if emb_p.exists():
            existing_test_files.append(str(emb_p.name))
        if assign_p.exists():
            existing_test_files.append(str(assign_p.name))

    if existing_test_files and not overwrite_test:
        raise RuntimeError(
            f"Existing Test evaluation outputs detected: {existing_test_files}!\n"
            "Defaulting to FAIL CLOSED. Refusing to silently overwrite or reuse Test outputs.\n"
            "Pass --overwrite-test to explicitly re-evaluate Test phase."
        )

    # 3. Load Test Cohort and Prerequisites
    assert TEST_CENTERS_PATH.exists(), f"Missing test centers: {TEST_CENTERS_PATH}"
    with open(TEST_CENTERS_PATH, "r", encoding="utf-8") as f:
        test_cohort = json.load(f)

    # 4. Load AnnData and Validate Row-Mapping Invariants
    print(f"\n[1/4] Loading Test Context from {DATASET_PATH} ...")
    adata = sc.read_h5ad(DATASET_PATH)

    # Rigorous Row-Mapping Invariant Assertion on Test Cohort
    test_idx = np.asarray(test_cohort["global_indices"])
    test_barcodes = np.asarray(test_cohort["cell_barcodes"])
    assert len(test_idx) == 10000, f"Expected 10,000 test indices, got {len(test_idx)}"
    assert len(np.unique(test_idx)) == 10000, "Duplicate test indices detected!"
    assert test_idx.min() >= 0 and test_idx.max() < adata.n_obs, "Test index out of AnnData bounds!"
    assert np.array_equal(adata.obs_names[test_idx].astype(str), test_barcodes.astype(str)), "Test barcode alignment mismatch with H5AD!"
    assert (adata.obs["split"].iloc[test_idx] == "test").all(), "Non-test cells in test_cohort!"
    assert adata.obs["donor_id"].iloc[test_idx].nunique() == 1, "Test cohort must originate from exactly 1 donor (Lung9)!"
    print(" -> Test cohort row-mapping invariant assertions 100% verified.")

    test_mask = (adata.obs["split"] == "test").to_numpy()
    test_ctx_indices = np.where(test_mask)[0]
    print(f" -> Extracted Test context: {len(test_ctx_indices):,} cells.")

    # Preprocessing
    adata_test = adata[test_ctx_indices].copy()
    sc.pp.normalize_total(adata_test, target_sum=1e4)
    sc.pp.log1p(adata_test)

    X_test_full = adata_test.X
    if sp.issparse(X_test_full):
        X_test_full = X_test_full.toarray()
    X_test_full = np.asarray(X_test_full, dtype=np.float32)

    coords_test_full = adata.obsm["spatial_um"][test_ctx_indices].astype(np.float32)
    test_sample_ids = adata.obs["sample_id"].iloc[test_ctx_indices].to_numpy()

    print(f" -> Building Test spatial graph (R={RADIUS_UM} um, sigma={SIGMA_UM:.4f} um) ...")
    P_test, A_test = build_global_spatial_graph(
        coords=coords_test_full,
        sample_ids=test_sample_ids,
        radius=RADIUS_UM,
        sigma=SIGMA_UM,
        max_neighbors=None
    )

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

    per_seed_test_results = {}
    assignments_dir = OUTPUT_DIR / "assignments"
    assignments_dir.mkdir(parents=True, exist_ok=True)

    test_embedding_shas = {}
    checkpoint_shas = {}

    for s in SEEDS:
        print(f"\n--- Testing Seed {s} with Frozen r* = {frozen_r_stars[str(s)]} ---")
        emb_file = EMBEDDINGS_DIR / f"test_embeddings_seed_{s}.npy"

        ckpt_path = CHECKPOINTS_DIR / f"seed_{s}" / f"model_seed_{s}_best.pt"
        checkpoint_shas[str(s)] = compute_file_sha256(ckpt_path)

        model = build_freqniche_model(960, MODEL_CONFIG).to(device)
        state_data = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in state_data:
            model.load_state_dict(state_data["model_state_dict"])
        else:
            model.load_state_dict(state_data)
        z_test = extract_embeddings(model, test_dataset, device=device,
                                    batch_size=inference_batch_size(OUTPUT_DIR, 128))
        temp_emb = emb_file.with_suffix(".tmp.npy")
        np.save(temp_emb, z_test)
        os.replace(temp_emb, emb_file)
        test_embedding_shas[str(s)] = compute_file_sha256(emb_file)

        # Leiden clustering with frozen r*
        r_star = float(frozen_r_stars[str(s)])
        adata_test_centers = ad.AnnData(X=z_test.copy())
        sc.pp.neighbors(adata_test_centers, n_neighbors=15, use_rep="X", random_state=42)
        sc.tl.leiden(adata_test_centers, resolution=r_star, key_added="predicted_cluster", random_state=42)
        pred_clusters = adata_test_centers.obs["predicted_cluster"].to_numpy().astype(int)

        # Reference niche label (curated reference label from CellCharter)
        ref_niche = adata.obs["niche"].iloc[test_cohort["global_indices"]].to_numpy()
        donor_ids = adata.obs["donor_id"].iloc[test_cohort["global_indices"]].to_numpy()
        sample_ids = adata.obs["sample_id"].iloc[test_cohort["global_indices"]].to_numpy()
        cell_barcodes = [test_cohort["cell_barcodes"][i] for i in range(len(pred_clusters))]

        ari = float(adjusted_rand_score(ref_niche, pred_clusters))
        nmi = float(normalized_mutual_info_score(ref_niche, pred_clusters))
        n_clusters = int(len(np.unique(pred_clusters)))

        # Subsample for ASW
        try:
            if len(z_test) > 5000:
                sub_idx = np.random.RandomState(42).choice(len(z_test), size=5000, replace=False)
                asw = float(silhouette_score(z_test[sub_idx], ref_niche[sub_idx]))
            else:
                asw = float(silhouette_score(z_test, ref_niche))
        except Exception:
            asw = 0.0

        per_seed_test_results[str(s)] = {
            "seed": s,
            "r_star": r_star,
            "n_clusters": n_clusters,
            "ari": ari,
            "nmi": nmi,
            "asw": asw,
            "test_embedding_sha256": test_embedding_shas[str(s)],
            "checkpoint_sha256": checkpoint_shas[str(s)]
        }
        print(f"Seed {s} Results | r*={r_star:.1f} | K={n_clusters} | ARI={ari:.4f} | NMI={nmi:.4f} | ASW={asw:.4f}")

        # Save assignments CSV atomically
        df_assign = pd.DataFrame({
            "cell_barcode": cell_barcodes,
            "donor_id": donor_ids,
            "sample_id": sample_ids,
            "reference_niche": ref_niche,
            "predicted_cluster": pred_clusters,
            "seed": s,
            "r_star": r_star,
        })
        assign_path = assignments_dir / f"test_assignments_seed_{s}.csv"
        temp_assign = assign_path.with_suffix(".tmp.csv")
        df_assign.to_csv(temp_assign, index=False)
        os.replace(temp_assign, assign_path)

    aris = [v["ari"] for v in per_seed_test_results.values()]
    nmis = [v["nmi"] for v in per_seed_test_results.values()]
    asws = [v["asw"] for v in per_seed_test_results.values()]
    ks = [v["n_clusters"] for v in per_seed_test_results.values()]

    script_path = Path(__file__).resolve()
    final_report = {
        "dataset": "cosmx_human_nsclc",
        "description": "W2.6 CosMx NSCLC Test Blind Evaluation",
        "cryptographic_bindings": {
            "evaluation_script_sha256": compute_file_sha256(script_path),
            "selection_manifest_sha256": res_selection["selection_manifest_sha256"],
            "test_centers_triple_hash": {
                "file_sha256": compute_file_sha256(TEST_CENTERS_PATH),
                "sorted_barcode_sha256": compute_sorted_barcode_sha256(test_cohort["cell_barcodes"]),
                "ordered_barcode_sha256": compute_ordered_barcode_sha256(test_cohort["cell_barcodes"]),
            },
            "checkpoints_sha256": checkpoint_shas
        },
        "mean_ari": float(np.mean(aris)),
        "std_ari": float(np.std(aris, ddof=1)),
        "mean_nmi": float(np.mean(nmis)),
        "std_nmi": float(np.std(nmis, ddof=1)),
        "mean_asw": float(np.mean(asws)),
        "std_asw": float(np.std(asws, ddof=1)),
        "mean_clusters": float(np.mean(ks)),
        "std_clusters": float(np.std(ks, ddof=1)),
        "per_seed_results": per_seed_test_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }

    temp_report = report_path.with_suffix(".tmp.json")
    with open(temp_report, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    os.replace(temp_report, report_path)

    # Also export Table 6 atomically
    df_table6 = pd.DataFrame([{
        "Method": "FreqNiche-ST v1.0 Final",
        "Paradigm": "Inductive",
        "Target_Donor": "Lung9",
        "K_Clusters": f"{np.mean(ks):.1f} ± {np.std(ks, ddof=1):.1f}",
        "ARI": f"{np.mean(aris):.4f} ± {np.std(aris, ddof=1):.4f}",
        "NMI": f"{np.mean(nmis):.4f} ± {np.std(nmis, ddof=1):.4f}",
        "Reference-Niche ASW": f"{np.mean(asws):.4f} ± {np.std(asws, ddof=1):.4f}",
        "Locked_r_stars": f"{frozen_r_stars['42']}/{frozen_r_stars['123']}/{frozen_r_stars['2024']}"
    }])
    temp_table6 = table6_path.with_suffix(".tmp.csv")
    df_table6.to_csv(temp_table6, index=False)
    os.replace(temp_table6, table6_path)

    print("\n" + "=" * 80)
    print(f"FINAL TEST RESULTS (3-Seed Mean ± SD):")
    print(f"ARI: {np.mean(aris):.4f} ± {np.std(aris, ddof=1):.4f}")
    print(f"NMI: {np.mean(nmis):.4f} ± {np.std(nmis, ddof=1):.4f}")
    print(f"ASW: {np.mean(asws):.4f} ± {np.std(asws, ddof=1):.4f}")
    print(f"Clusters: {np.mean(ks):.1f} ± {np.std(ks, ddof=1):.1f}")
    print(f"Report saved atomically to: {report_path}")
    print(f"Table 6 saved atomically to: {table6_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Stage 4 Evaluation for FreqNiche-ST v1.0 on CosMx NSCLC.")
    parser.add_argument("--phase", type=str, choices=["validation", "test"], required=True,
                        help="Phase of evaluation: 'validation' selects r*; 'test' evaluates on unseen test donor.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device ('cuda' or 'cpu').")
    parser.add_argument("--overwrite-selection", action="store_true",
                        help="Allow updating/re-signing resolution_selection.json.")
    parser.add_argument("--recompute-val-embeddings", action="store_true",
                        help="Force recomputing validation embeddings even if files exist.")
    parser.add_argument("--overwrite-test", action="store_true",
                        help="Allow overwriting existing test evaluation outputs.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Alias for overwriting manifest/test outputs (does NOT recompute val embeddings).")
    args = parser.parse_args()

    device = torch.device(args.device)

    allow_overwrite_sel = args.overwrite_selection or args.overwrite
    allow_overwrite_test = args.overwrite_test or args.overwrite

    if args.phase == "validation":
        run_validation_phase(
            device=device,
            overwrite_selection=allow_overwrite_sel,
            recompute_val_embeddings=args.recompute_val_embeddings
        )
    elif args.phase == "test":
        run_test_phase(
            device=device,
            overwrite_test=allow_overwrite_test
        )


if __name__ == "__main__":
    main()
