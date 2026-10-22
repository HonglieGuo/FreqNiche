"""
scripts/benchmark/eval_xenium_test.py

W2.7 Xenium Human NSCLC 5K: Phase 2 Test Evaluation on Unseen Donor L1 (GSM9509134).
================================================================================
Strict Reproducibility & Cryptographic Integrity Protocol:
1. Entry Guard & Immutability Protection:
   - Verifies Phase 1 manifest (resolution_selection.json) self-signature (987659f6...)
     and physical file SHA-256 (fbc29711...).
   - Verifies that Phase 1 script (eval_xenium_stability.py) has NOT been modified
     (matches frozen SHA-256: bb445814ad2df6401fd7930c6d1fd671b948a11a7beadf5580ec681507c2601a).
   - Verifies all bound Phase 1 upstream input assets:
     * Data audit manifest (060e9cd2...)
     * Cohorts manifest (5420afaa...)
     * Validation centers (23492085...)
     * Validation 10-fold bootstrap partitions (a6177cc2...)
     * Whitener checkpoint (9e2a0c6d...)
     * 3 Model checkpoints (Seed 42, 123, 2024)
     * 3 Validation embeddings (Seed 42, 123, 2024)
2. Fail-Closed Execution Guard:
   - Requires explicit --authorized CLI flag to execute test evaluation.
   - Halts immediately with exit code 2 if --authorized is not provided.
   - Comprehensive overwrite guard: scans for any existing test graphs, embeddings,
     assignments, or reports. Fails closed unless --overwrite is explicitly passed.
     Zero silent partial reuse of embeddings or graph caches.
3. Test Data & Spatial Context:
   - Unseen Test donor L1: GSM9509134_NSCLC_5k_L1_L1.h5ad (67,705 cells).
   - Dynamically loads and verifies L1 H5AD SHA-256 directly from verified audit manifest
     (expected: 656cd5fe255d71e5afa98151c3d23212683093b57dcbd419c96cf5e9d79c2312).
   - Freezes all 67,705 cells in original H5AD row order as inference centers.
   - Preserves full physical micron spatial continuity (adata.obsm['spatial']).
   - Rigorous mathematical validation of L1 spatial graph caches (symmetry, self-loops=1.0,
     row-stochasticity P_sum=1.0, shape=(67705, 67705)).
4. Strictly Frozen Hyperparameters:
   - Uses frozen optimal resolutions from Phase 1 manifest:
     * Seed 42   -> r* = 0.4
     * Seed 123  -> r* = 0.2
     * Seed 2024 -> r* = 0.6
   - Zero test-driven hyperparameter tuning or post-hoc resolution searching.
5. Strict Biological Ground Truth Purity:
   - No reference niche or cell-type ground truth is read, indexed, or outputted
     (niche_valid_count == 0 on Xenium L1).
   - NO Reference Niche ASW.
6. Comprehensive Multi-Dimensional Unsupervised Metrics:
   - Predicted-cluster internal ASW (Silhouette score in 128-dim space).
   - Embedding geometry: Effective Rank (Roy & Vetterli, 2007) and variance explained spectrum.
   - Clustering structure balance: Cluster count K, Cluster Entropy, max cluster ratio (p_max),
     min cluster ratio (p_min).
   - Local Neighborhood Purity: Mean & standard deviation of intra-cluster fraction in physical 50 um graph.
   - Spatial Autocorrelation Moran's I with Feature Permutation Null on Fixed Spatial Graph:
     * Evaluates Moran's I_real on Top-5 embedding PCs.
     * Evaluates Moran's I_real on cluster one-hot indicator fields.
     * Rigorous Feature Permutation Negative Control (N_perm = 100 permutations on fixed spatial graph),
       reporting null mean, null std, delta I = (I_real - I_null), z-score, and empirical p-value.
7. Atomic Persistence & Cryptographic Binding:
   - Atomic writes for test embeddings: outputs/runs/freqniche/xenium_prime5k/2026-09-18/20260918_xenium_prime5k_final/embeddings/test_embeddings_seed_{seed}.npy
   - Atomic writes for cell assignments: outputs/runs/freqniche/xenium_prime5k/2026-09-18/20260918_xenium_prime5k_final/assignments/xenium_test_assignments_seed_{seed}.csv
   - Atomic write for final evaluation report: outputs/runs/freqniche/xenium_prime5k/2026-09-18/20260918_xenium_prime5k_final/xenium_test_evaluation_report.json
   - Binds SHA-256 hashes of test runner script, Phase 1 manifest, checkpoints, L1 H5AD, L1 graph cache,
     and all generated test assets into the self-signed report.
================================================================================
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
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, silhouette_score
import torch
from torch.utils.data import DataLoader

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.freqniche_st.graph import build_global_spatial_graph
from src.freqniche_st.dataset import SubgraphDataset, collate_subgraphs
from src.freqniche_st.models.freqniche_model import FreqNicheDiTModel
from src.freqniche_st.runtime_paths import resolve_run_output
from src.freqniche_st.model_config import load_model_config, build_freqniche_model, inference_batch_size

OUTPUT_DIR = resolve_run_output("outputs/runs/freqniche/xenium_prime5k/2026-09-18/20260918_xenium_prime5k_final")
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
ASSIGNMENTS_DIR = OUTPUT_DIR / "assignments"
CACHE_DIR = OUTPUT_DIR / "cache"
GSE_DIR = WORKSPACE_ROOT / "datasets" / "h5ad" / "xenium_ffpe_lungbreast_gse311609" / "samples"

AUDIT_MANIFEST_PATH = OUTPUT_DIR / "xenium_data_audit_manifest.json"
COHORTS_MANIFEST_PATH = OUTPUT_DIR / "xenium_cohorts_manifest.json"
VAL_CENTERS_PATH = OUTPUT_DIR / "val_centers.json"
VAL_FOLDS_PATH = OUTPUT_DIR / "val_bootstrap_folds.json"
WHITENER_PATH = OUTPUT_DIR / "whitener.pt"
TRAINING_SUMMARY_PATH = CHECKPOINTS_DIR / "xenium_training_summary.json"
RESOLUTION_SELECTION_PATH = OUTPUT_DIR / "resolution_selection.json"
FINAL_REPORT_PATH = OUTPUT_DIR / "xenium_test_evaluation_report.json"
MODEL_CONFIG = load_model_config(output_dir=OUTPUT_DIR)

PHASE1_SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "benchmark" / "eval_xenium_stability.py"
EXPECTED_PHASE1_SCRIPT_SHA = "bb445814ad2df6401fd7930c6d1fd671b948a11a7beadf5580ec681507c2601a"
EXPECTED_PHASE1_SELF_SIGNATURE = "987659f63b1d3a5c6d7c66ecc0ecb7d9d2eaa9d6d1fc30ff9c675ca846c56c07"
EXPECTED_PHASE1_FILE_SHA = "fbc297117bfaf3f05fdf3b95f4b81bf8ac2ea73ea00ecc327ba9f0f5a6dcf0f7"

TEST_SAMPLE_FNAME = "GSM9509134_NSCLC_5k_L1_L1.h5ad"
TEST_DONOR_ID = "L1"

SEEDS = [42, 123, 2024]
RADIUS_UM = 50.0
SIGMA_UM = 16.666666666666668
N_PERMUTATIONS = 100


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


def check_overwrite_guards(allow_overwrite: bool):
    """
    Scans for any existing L1 Test assets (graph npz, embeddings, assignments, reports).
    Fails closed immediately if any asset is detected and allow_overwrite is False.
    Zero silent partial reuse.
    """
    existing_assets = []

    # Check report
    if FINAL_REPORT_PATH.exists():
        existing_assets.append(str(FINAL_REPORT_PATH))

    # Check test graphs
    l1_p = CACHE_DIR / f"xenium_graph_{TEST_DONOR_ID}_P.npz"
    l1_a = CACHE_DIR / f"xenium_graph_{TEST_DONOR_ID}_A.npz"
    if l1_p.exists():
        existing_assets.append(str(l1_p))
    if l1_a.exists():
        existing_assets.append(str(l1_a))

    # Check embeddings and assignments
    for s in SEEDS:
        emb_p = EMBEDDINGS_DIR / f"test_embeddings_seed_{s}.npy"
        assign_p = ASSIGNMENTS_DIR / f"xenium_test_assignments_seed_{s}.csv"
        if emb_p.exists():
            existing_assets.append(str(emb_p))
        if assign_p.exists():
            existing_assets.append(str(assign_p))

    if existing_assets and not allow_overwrite:
        msg = (
            f"FAIL-CLOSED OVERWRITE GUARD: Existing Test assets detected on disk without --overwrite!\n"
            f"Found {len(existing_assets)} asset(s):\n"
            + "\n".join(f"  - {p}" for p in existing_assets[:8])
        )
        if len(existing_assets) > 8:
            msg += f"\n  ... and {len(existing_assets) - 8} more."
        msg += "\nPass --overwrite explicitly if you intend to re-run test evaluation."
        raise RuntimeError(msg)


def verify_phase1_entry_guard() -> Dict[str, Any]:
    """
    Cryptographic Entry Guard:
    Verifies that Phase 1 resolution_selection.json exists, self-signature is authentic,
    file SHA matches recorded audit, Phase 1 evaluation script bb445814... was NOT modified,
    and all 7 bound input assets are 100% bitwise verified.
    """
    print("\n" + "=" * 85)
    print("STEP 1: CRYPTOGRAPHIC ENTRY GUARD & PHASE 1 RESOLUTION MANIFEST AUDIT")
    print("=" * 85)

    if not RESOLUTION_SELECTION_PATH.exists():
        raise FileNotFoundError(f"Missing Phase 1 manifest: {RESOLUTION_SELECTION_PATH}")

    # Historical direct invocations retain the exact frozen W2.7 hash lock.
    # A newly orchestrated run instead verifies its own signed Phase-1 asset.
    fresh_run = bool(os.environ.get("FREQNICHE_RUN_DIR"))

    # 1. Physical file SHA check
    actual_file_sha = compute_file_sha256(RESOLUTION_SELECTION_PATH)
    if not fresh_run and actual_file_sha != EXPECTED_PHASE1_FILE_SHA:
        raise ValueError(
            f"Phase 1 manifest file SHA-256 mismatch!\n"
            f"  Expected: {EXPECTED_PHASE1_FILE_SHA}\n"
            f"  Actual:   {actual_file_sha}"
        )
    if fresh_run:
        print(f" -> [1/8] Phase 1 manifest physical SHA computed for this run: {actual_file_sha[:16]}...")
    else:
        print(f" -> [1/8] Phase 1 manifest legacy physical SHA verified: {actual_file_sha[:16]}...")

    # 2. Self-signature verification
    with open(RESOLUTION_SELECTION_PATH, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    if "selection_manifest_sha256" not in manifest_data:
        raise ValueError("Missing selection_manifest_sha256 in manifest!")
    recorded_sig = manifest_data.pop("selection_manifest_sha256")
    calc_sig = hashlib.sha256(json.dumps(manifest_data, sort_keys=True).encode("utf-8")).hexdigest()

    if recorded_sig != calc_sig or (not fresh_run and recorded_sig != EXPECTED_PHASE1_SELF_SIGNATURE):
        raise ValueError(
            f"Phase 1 manifest self-signature mismatch!\n"
            f"  Recorded: {recorded_sig}\n"
            f"  Expected: {EXPECTED_PHASE1_SELF_SIGNATURE}\n"
            f"  Computed: {calc_sig}"
        )
    manifest_data["selection_manifest_sha256"] = recorded_sig
    print(f" -> [2/8] Phase 1 manifest self-signature verified: {recorded_sig[:16]}...")

    # 3. Phase 1 evaluation script SHA immutability check
    if not PHASE1_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Missing Phase 1 script: {PHASE1_SCRIPT_PATH}")
    current_p1_script_sha = compute_file_sha256(PHASE1_SCRIPT_PATH)
    expected_p1_sha = manifest_data["bound_input_assets"]["evaluation_script"]["sha256"]
    if current_p1_script_sha != expected_p1_sha or (not fresh_run and current_p1_script_sha != EXPECTED_PHASE1_SCRIPT_SHA):
        raise ValueError(
            f"Phase 1 evaluation script has been tampered with or modified!\n"
            f"  Expected: {expected_p1_sha if fresh_run else EXPECTED_PHASE1_SCRIPT_SHA}\n"
            f"  Current:  {current_p1_script_sha}"
        )
    print(f" -> [3/8] Phase 1 script immutability verified (SHA: {current_p1_script_sha[:16]}...).")

    # 4. Data audit manifest & cohorts manifest SHA check
    audit_sha = compute_file_sha256(AUDIT_MANIFEST_PATH)
    cohorts_sha = compute_file_sha256(COHORTS_MANIFEST_PATH)
    assert audit_sha == manifest_data["bound_input_assets"]["data_audit_manifest"]["sha256"]
    assert cohorts_sha == manifest_data["bound_input_assets"]["cohorts_manifest"]["sha256"]
    print(" -> [4/8] Master data audit manifest and cohorts manifest verified.")

    # 5. Validation centers & 10-fold bootstrap partitions check
    vc_sha = compute_file_sha256(VAL_CENTERS_PATH)
    vf_sha = compute_file_sha256(VAL_FOLDS_PATH)
    assert vc_sha == manifest_data["bound_input_assets"]["val_centers_manifest"]["sha256"]
    assert vf_sha == manifest_data["bound_input_assets"]["val_bootstrap_folds"]["sha256"]
    print(" -> [5/8] Validation centers and 10-fold bootstrap partitions verified.")

    # 6. Whitener checkpoint SHA check
    wh_sha = compute_file_sha256(WHITENER_PATH)
    assert wh_sha == manifest_data["bound_input_assets"]["whitener"]["sha256"]
    print(f" -> [6/8] Formal Whitener checkpoint verified (SHA: {wh_sha[:16]}...).")

    # 7. Checkpoints SHA check (3 seeds)
    for s in SEEDS:
        ckpt_p = CHECKPOINTS_DIR / f"xenium_freqniche_trial24_seed_{s}.pt"
        real_ckpt_sha = compute_file_sha256(ckpt_p)
        expected_ckpt_sha = manifest_data["bound_input_assets"]["checkpoints"][str(s)]["sha256"]
        assert real_ckpt_sha == expected_ckpt_sha, f"Seed {s} checkpoint SHA mismatch!"
    print(" -> [7/8] All 3 Trial 24 model checkpoints verified.")

    # 8. Validation embeddings SHA check (3 seeds)
    for s in SEEDS:
        val_emb_p = EMBEDDINGS_DIR / f"val_embeddings_seed_{s}.npy"
        real_emb_sha = compute_file_sha256(val_emb_p)
        expected_emb_sha = manifest_data["bound_input_assets"]["val_embeddings"][str(s)]["sha256"]
        assert real_emb_sha == expected_emb_sha, f"Seed {s} Val embedding SHA mismatch!"
    print(" -> [8/8] All 3 Validation embeddings verified.")

    frozen_r_stars = {int(k): float(v) for k, v in manifest_data["frozen_r_star_per_seed"].items()}
    print(f"\n[PASS] ALL 8 PRE-REQUISITE CRYPTOGRAPHIC INTEGRITY CHECKS 100% PASSED!")
    print(f"Frozen r* per seed: Seed 42 -> r*={frozen_r_stars[42]}, Seed 123 -> r*={frozen_r_stars[123]}, Seed 2024 -> r*={frozen_r_stars[2024]}")
    return manifest_data


def validate_spatial_graph_invariants(A_csr: sp.csr_matrix, P_csr: sp.csr_matrix, n_expected_cells: int):
    """
    Rigorously verifies spatial graph mathematical invariants:
    1. Shapes match (n_expected_cells, n_expected_cells)
    2. Symmetry: A = A^T
    3. Self-loops: diagonal elements of A are strictly 1.0
    4. Row-stochasticity: every row sum of P equals 1.0
    """
    assert A_csr.shape == (n_expected_cells, n_expected_cells), f"A shape mismatch: {A_csr.shape}"
    assert P_csr.shape == (n_expected_cells, n_expected_cells), f"P shape mismatch: {P_csr.shape}"

    diag_A = A_csr.diagonal()
    assert np.allclose(diag_A, 1.0), "A_csr diagonal self-loops must all equal 1.0"

    diff_sym = (A_csr - A_csr.T).tocsr()
    max_asym = np.abs(diff_sym.data).max() if diff_sym.nnz > 0 else 0.0
    assert max_asym < 1e-6, f"A_csr is not symmetric! Max asymmetry: {max_asym}"

    row_sums = np.array(P_csr.sum(axis=1)).flatten()
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"P_csr rows must sum to 1.0! Max diff: {np.abs(row_sums - 1.0).max()}"


def load_or_build_test_context(test_h5ad_path: Path) -> Tuple[np.ndarray, np.ndarray, sp.csr_matrix, sp.csr_matrix, List[str], str, str, str]:
    """
    Loads test donor L1 H5AD and verifies its SHA-256 dynamically against audit manifest.
    Builds or loads physical spatial graphs (P and A) for L1, with full invariant assertions.
    Returns:
      coords: (67705, 2)
      X_norm: (67705, 5001)
      A_csr, P_csr
      barcodes: List[str]
      h5ad_sha, a_graph_sha, p_graph_sha
    """
    print("\n" + "=" * 85)
    print(f"STEP 2: LOADING TEST DONOR {TEST_DONOR_ID} ({test_h5ad_path.name})")
    print("=" * 85)

    if not test_h5ad_path.exists():
        raise FileNotFoundError(f"Missing test sample: {test_h5ad_path}")

    # Dynamically read expected L1 SHA-256 and cell count from verified audit manifest
    with open(AUDIT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        audit_manifest = json.load(f)

    sample_meta = audit_manifest["primary_cohort"]["samples"].get(TEST_SAMPLE_FNAME)
    if not sample_meta:
        raise KeyError(f"Sample {TEST_SAMPLE_FNAME} not found in {AUDIT_MANIFEST_PATH}")

    expected_h5ad_sha = sample_meta["file_sha256"]
    expected_cell_count = sample_meta["n_cells"]

    print(f" -> Verifying test sample {TEST_SAMPLE_FNAME} SHA-256 against audit manifest ...")
    h5ad_sha = compute_file_sha256(test_h5ad_path)
    if h5ad_sha != expected_h5ad_sha:
        raise ValueError(
            f"Test sample H5AD SHA-256 mismatch with data audit manifest!\n"
            f"  Expected: {expected_h5ad_sha}\n"
            f"  Actual:   {h5ad_sha}"
        )
    print(f" -> Test sample H5AD SHA verified: {h5ad_sha[:16]}...")

    print(" -> Loading AnnData and physical coordinates ...")
    a = ad.read_h5ad(test_h5ad_path, backed="r")
    assert "spatial" in a.obsm, "Missing obsm['spatial'] in test H5AD!"
    coords = np.asarray(a.obsm["spatial"], dtype=np.float32)
    barcodes = list(a.obs_names)
    n_cells = len(coords)
    assert n_cells == expected_cell_count, f"Expected {expected_cell_count} cells, got {n_cells}"
    print(f" -> Verified cell count: {n_cells:,} cells from donor {TEST_DONOR_ID}.")

    # Load normalized expression
    print(" -> Applying frozen preprocessing: normalize_total(1e4) + log1p() ...")
    raw_X = a.X[:]
    if sp.issparse(raw_X):
        X_dense = raw_X.toarray().astype(np.float32)
    else:
        X_dense = np.asarray(raw_X, dtype=np.float32)
    counts = np.maximum(X_dense.sum(axis=1, keepdims=True), 1.0)
    X_norm = np.log1p((X_dense / counts) * 1e4)
    assert X_norm.shape == (n_cells, 5001), f"Expected shape ({n_cells}, 5001), got {X_norm.shape}"

    # Load or build physical spatial graph
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p_cache_path = CACHE_DIR / f"xenium_graph_{TEST_DONOR_ID}_P.npz"
    a_cache_path = CACHE_DIR / f"xenium_graph_{TEST_DONOR_ID}_A.npz"

    if p_cache_path.exists() and a_cache_path.exists():
        print(f" -> Loading cached L1 physical spatial graphs from {CACHE_DIR.name} ...")
        P_csr = sp.load_npz(p_cache_path)
        A_csr = sp.load_npz(a_cache_path)
        print(" -> Verifying loaded graph mathematical invariants ...")
        validate_spatial_graph_invariants(A_csr, P_csr, n_cells)
    else:
        print(f" -> Building L1 physical spatial graph (R={RADIUS_UM} um, sigma={SIGMA_UM:.4f} um) ...")
        sample_ids = np.array([f"{TEST_DONOR_ID}_{TEST_SAMPLE_FNAME}"] * n_cells)
        P_csr, A_csr = build_global_spatial_graph(
            coords=coords,
            sample_ids=sample_ids,
            radius=RADIUS_UM,
            sigma=SIGMA_UM,
            max_neighbors=None
        )
        print(" -> Validating newly constructed graph mathematical invariants ...")
        validate_spatial_graph_invariants(A_csr, P_csr, n_cells)

        print(" -> Atomically caching L1 spatial graphs ...")
        temp_p = p_cache_path.with_suffix(".tmp.npz")
        temp_a = a_cache_path.with_suffix(".tmp.npz")
        sp.save_npz(temp_p, P_csr)
        sp.save_npz(temp_a, A_csr)
        os.replace(temp_p, p_cache_path)
        os.replace(temp_a, a_cache_path)

    p_graph_sha = compute_file_sha256(p_cache_path)
    a_graph_sha = compute_file_sha256(a_cache_path)
    print(f" -> L1 spatial graph verified: {A_csr.nnz:,} directed edges (A SHA: {a_graph_sha[:16]}..., P SHA: {p_graph_sha[:16]}...).")

    return coords, X_norm, A_csr, P_csr, barcodes, h5ad_sha, a_graph_sha, p_graph_sha


def extract_test_embeddings_for_seed(
    seed: int,
    model_ckpt_path: Path,
    dataset: SubgraphDataset,
    device: torch.device,
    batch_size: int = 128
) -> np.ndarray:
    """Extract embeddings with the frozen training architecture."""
    model = build_freqniche_model(5001, MODEL_CONFIG).to(device)

    state_data = torch.load(model_ckpt_path, map_location=device, weights_only=True)
    if "model_state_dict" in state_data:
        model.load_state_dict(state_data["model_state_dict"])
    else:
        model.load_state_dict(state_data)

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
    assert emb.shape == (len(dataset), MODEL_CONFIG["d_model"]), f"Shape mismatch: {emb.shape}"
    assert np.all(np.isfinite(emb)), f"FATAL: Non-finite embeddings detected for seed {seed}!"
    return emb


def compute_effective_rank(emb: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Effective Rank and variance spectrum via centered SVD."""
    emb_centered = emb - np.mean(emb, axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(emb_centered, full_matrices=False)
    s_sum = np.sum(s)
    if s_sum <= 0:
        return 1.0, {}
    p = s / s_sum
    p = p[p > 1e-12]
    spectral_entropy = -float(np.sum(p * np.log(p)))
    erank = float(np.exp(spectral_entropy))

    eigenvals = (s ** 2) / max(len(emb) - 1, 1)
    tot_var = np.sum(eigenvals)
    var_ratio = eigenvals / max(tot_var, 1e-12)
    var_spectrum = {
        "top_1_ratio": float(var_ratio[0]),
        "top_5_ratio": float(np.sum(var_ratio[:5])),
        "top_10_ratio": float(np.sum(var_ratio[:10])),
        "top_30_ratio": float(np.sum(var_ratio[:min(30, len(var_ratio))])),
    }
    return erank, var_spectrum


def compute_local_neighborhood_purity(labels: np.ndarray, A_csr: sp.csr_matrix) -> Tuple[float, float]:
    """
    Computes local neighborhood purity for each node in graph A_csr.
    Purity_i = (number of neighbors with label == label_i) / (total neighbors of i).
    Excludes self-loops in computation.
    """
    A_off = A_csr.copy().tolil()
    A_off.setdiag(0)
    A_off = A_off.tocsr()

    purities = []
    for i in range(len(labels)):
        row_indices = A_off.indices[A_off.indptr[i]:A_off.indptr[i+1]]
        if len(row_indices) == 0:
            continue
        c_i = labels[i]
        c_neighbors = labels[row_indices]
        purity_i = float(np.mean(c_neighbors == c_i))
        purities.append(purity_i)

    if not purities:
        return 0.0, 0.0
    return float(np.mean(purities)), float(np.std(purities, ddof=1))


def compute_morans_i_raw(values: np.ndarray, W_csr: sp.csr_matrix) -> float:
    """Computes Moran's I on a 1D field given spatial weight matrix W_csr."""
    n = len(values)
    v_centered = values - np.mean(values)
    denom = np.sum(v_centered ** 2)
    if denom <= 1e-12:
        return 0.0
    W_sum = W_csr.sum()
    if W_sum <= 1e-12:
        return 0.0
    num = v_centered.dot(W_csr.dot(v_centered))
    return float((n / W_sum) * (num / denom))


def evaluate_morans_i_with_feature_permutation_null(
    feature_matrix: np.ndarray,
    A_csr: sp.csr_matrix,
    feature_names: Optional[List[str]] = None,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes Moran's I for features across fixed spatial graph with rigorous
    Feature Permutation Negative Control (n_permutations runs on fixed spatial topology).
    Returns real Moran's I, null mean, null std, delta I, z-score, empirical p-value.
    """
    # Build row-normalized spatial weights (off-diagonal only)
    A_off = A_csr.copy().tolil()
    A_off.setdiag(0)
    A_off = A_off.tocsr().astype(np.float32)

    deg = np.array(A_off.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    inv_deg = sp.diags(1.0 / deg)
    W_csr = inv_deg.dot(A_off).tocsr()

    n_features = feature_matrix.shape[1]
    feature_results = []
    rng = np.random.RandomState(seed)

    for f_idx in range(n_features):
        fname = feature_names[f_idx] if feature_names else f"feature_{f_idx}"
        vals = feature_matrix[:, f_idx]
        i_real = compute_morans_i_raw(vals, W_csr)

        # Feature Permutation negative control on fixed spatial graph
        null_dist = np.zeros(n_permutations, dtype=np.float32)
        for perm in range(n_permutations):
            perm_indices = rng.permutation(len(vals))
            null_dist[perm] = compute_morans_i_raw(vals[perm_indices], W_csr)

        null_mean = float(np.mean(null_dist))
        null_std = float(np.std(null_dist, ddof=1)) if n_permutations > 1 else 1e-6
        delta_i = float(i_real - null_mean)
        z_score = float(delta_i / max(null_std, 1e-8))
        emp_p = float((np.sum(null_dist >= i_real) + 1.0) / (n_permutations + 1.0))

        feature_results.append({
            "feature_name": fname,
            "feature_index": f_idx,
            "moran_i_real": i_real,
            "null_mean": null_mean,
            "null_std": null_std,
            "delta_i": delta_i,
            "z_score": z_score,
            "empirical_p_value": emp_p
        })

    mean_i_real = float(np.mean([r["moran_i_real"] for r in feature_results]))
    mean_delta_i = float(np.mean([r["delta_i"] for r in feature_results]))
    mean_z_score = float(np.mean([r["z_score"] for r in feature_results]))

    return {
        "mean_moran_i_real": mean_i_real,
        "mean_delta_i": mean_delta_i,
        "mean_z_score": mean_z_score,
        "n_permutations": n_permutations,
        "null_type": "feature_permutation_on_fixed_spatial_graph",
        "per_feature_details": feature_results
    }


def run_test_evaluation(
    device: torch.device,
    authorized: bool = False,
    recompute_embeddings: bool = False,
    overwrite: bool = False
):
    """
    Formal Phase 2 Test Execution on unseen donor L1 (GSM9509134).
    """
    print("=" * 85)
    print("W2.7 XENIUM PHASE 2: TEST EVALUATION ON UNSEEN DONOR L1 (GSM9509134)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 85)

    # 1. Top-Level Fail-Closed Authorization Guard
    if not authorized:
        print("\n" + "!" * 85)
        print("FAIL-CLOSED GUARD: Formal test evaluation requires explicit --authorized permission.")
        print("Test runner is halting with exit code 2. Zero test files modified.")
        print("!" * 85)
        sys.exit(2)

    # 2. Comprehensive Overwrite Guard
    check_overwrite_guards(allow_overwrite=overwrite)

    # 3. Cryptographic Entry Guard
    phase1_manifest = verify_phase1_entry_guard()
    frozen_r_stars = {int(k): float(v) for k, v in phase1_manifest["frozen_r_star_per_seed"].items()}

    # 4. Define and load test context
    test_h5ad_path = GSE_DIR / TEST_SAMPLE_FNAME
    coords, X_norm, A_csr, P_csr, barcodes, h5ad_sha, a_graph_sha, p_graph_sha = load_or_build_test_context(test_h5ad_path)
    n_cells = len(coords)

    # Setup directories
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    ASSIGNMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # 5. Create Test Dataset across all 67,705 centers in original row order
    print("\n" + "=" * 85)
    print(f"STEP 3: CONSTRUCTING SUBGRAPH DATASET FOR ALL {n_cells:,} INFERENCE CENTERS")
    print("=" * 85)
    center_indices = np.arange(n_cells, dtype=np.int64)

    test_dataset = SubgraphDataset(
        center_indices=center_indices,
        A_global=A_csr,
        P_global=P_csr,
        X_global=X_norm,
        coords_global=coords,
        whitener=None,
        k_max=MODEL_CONFIG["k_max"],
        m_max=MODEL_CONFIG["m_max"],
        is_train=False,
        is_val_objective=False
    )

    test_embeddings = {}
    test_embedding_shas = {}
    test_assignment_shas = {}
    per_seed_results = {}

    # 6. Infer embeddings and evaluate per seed
    print("\n" + "=" * 85)
    print("STEP 4: INFERENCE, FROZEN-r* LEIDEN CLUSTERING & MULTI-DIMENSIONAL EVALUATION")
    print("=" * 85)

    for s in SEEDS:
        print(f"\n" + "-" * 90)
        print(f"EVALUATING SEED {s} (Locked r* = {frozen_r_stars[s]}) ON UNSEEN DONOR L1")
        print("-" * 90)

        emb_file = EMBEDDINGS_DIR / f"test_embeddings_seed_{s}.npy"
        assign_file = ASSIGNMENTS_DIR / f"xenium_test_assignments_seed_{s}.csv"
        ckpt_path = CHECKPOINTS_DIR / f"xenium_freqniche_trial24_seed_{s}.pt"

        if emb_file.exists() and not recompute_embeddings:
            print(f" -> Loading existing test embeddings for Seed {s} from {emb_file.name} ...")
            z_test = np.load(emb_file)
            assert z_test.shape == (n_cells, MODEL_CONFIG["d_model"]) and np.all(np.isfinite(z_test))
        else:
            print(f" -> Extracting test embeddings for Seed {s} ({n_cells:,} centers) ...")
            t0_infer = time.perf_counter()
            z_test = extract_test_embeddings_for_seed(
                seed=s,
                model_ckpt_path=ckpt_path,
                dataset=test_dataset,
                device=device,
                batch_size=inference_batch_size(OUTPUT_DIR, 128)
            )
            dur_infer = time.perf_counter() - t0_infer
            print(f" -> Extracted in {dur_infer:.2f} s ({n_cells / dur_infer:.1f} cells/s). Atomically saving ...")
            temp_emb = emb_file.with_suffix(".tmp.npy")
            np.save(temp_emb, z_test)
            os.replace(temp_emb, emb_file)

        test_embeddings[s] = z_test
        emb_sha = compute_file_sha256(emb_file)
        test_embedding_shas[s] = emb_sha
        print(f" -> Test embedding SHA-256: {emb_sha[:16]}...")

        # Feature Geometry: Effective Rank & Variance Spectrum
        erank, var_spectrum = compute_effective_rank(z_test)
        print(f" -> Effective Rank: {erank:.2f} / {MODEL_CONFIG['d_model']} (Top-1: {var_spectrum['top_1_ratio']*100:.1f}%, Top-10: {var_spectrum['top_10_ratio']*100:.1f}%)")

        # Frozen r* Clustering
        r_star = frozen_r_stars[s]
        print(f" -> Running Leiden clustering at strictly locked r* = {r_star} ...")
        adata_seed = ad.AnnData(X=z_test.copy())
        sc.pp.neighbors(adata_seed, n_neighbors=15, use_rep="X", random_state=s)
        sc.tl.leiden(adata_seed, resolution=r_star, key_added="predicted_niche", random_state=s)
        labels = adata_seed.obs["predicted_niche"].to_numpy().astype(int)
        k_clusters = len(np.unique(labels))

        # Cluster Structure Balance & Entropy
        counts = pd.Series(labels).value_counts().values
        probs = counts / float(len(labels))
        p_max = float(np.max(probs))
        p_min = float(np.min(probs))
        entropy = -float(np.sum(probs * np.log(probs)))

        print(f" -> Clusters K: {k_clusters}, Max cluster ratio (p_max): {p_max:.4f}, Min (p_min): {p_min:.4f}, Entropy: {entropy:.4f}")

        # Predicted-Cluster Internal ASW (Silhouette in 128-dim space on fixed subsample for efficiency & stability)
        print(" -> Computing predicted-cluster internal ASW (Euclidean Silhouette score) ...")
        if k_clusters > 1:
            # Subsample 10,000 cells with deterministic seed for robust and reproducible silhouette computation
            asw_val = float(silhouette_score(z_test, labels, metric="euclidean", sample_size=10000, random_state=s))
        else:
            asw_val = 0.0
        print(f" -> Predicted-cluster internal ASW: {asw_val:.4f}")

        # Local Neighborhood Purity in Physical Graph
        print(" -> Computing Local Neighborhood Purity in physical 50 um spatial graph ...")
        purity_mean, purity_std = compute_local_neighborhood_purity(labels, A_csr)
        print(f" -> Local Neighborhood Purity: {purity_mean:.4f} ± {purity_std:.4f}")

        # Spatial Autocorrelation: Moran's I on Top-5 PCs with Feature Permutation Null on Fixed Graph
        print(f" -> Computing Moran's I on Top-5 embedding PCs ({N_PERMUTATIONS} feature permutations) ...")
        z_centered = z_test - np.mean(z_test, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(z_centered, full_matrices=False)
        top_5_pcs = z_centered.dot(vh[:5].T)

        moran_pcs_results = evaluate_morans_i_with_feature_permutation_null(
            feature_matrix=top_5_pcs,
            A_csr=A_csr,
            feature_names=[f"PC_{i+1}" for i in range(5)],
            n_permutations=N_PERMUTATIONS,
            seed=s
        )
        print(f" -> Moran's I (Top-5 PCs): Real = {moran_pcs_results['mean_moran_i_real']:.4f}, "
              f"Delta I = {moran_pcs_results['mean_delta_i']:.4f}, z-score = {moran_pcs_results['mean_z_score']:.2f}")

        # Spatial Autocorrelation: Moran's I on Cluster One-Hot Indicator Fields
        print(f" -> Computing Moran's I on Cluster One-Hot indicators ({N_PERMUTATIONS} label permutations) ...")
        labels_onehot = np.eye(k_clusters, dtype=np.float32)[labels]
        moran_clusters_results = evaluate_morans_i_with_feature_permutation_null(
            feature_matrix=labels_onehot,
            A_csr=A_csr,
            feature_names=[f"Cluster_{k}" for k in range(k_clusters)],
            n_permutations=N_PERMUTATIONS,
            seed=s + 50000
        )
        print(f" -> Moran's I (Cluster Indicators): Real = {moran_clusters_results['mean_moran_i_real']:.4f}, "
              f"Delta I = {moran_clusters_results['mean_delta_i']:.4f}, z-score = {moran_clusters_results['mean_z_score']:.2f}")

        # Atomically save assignment CSV
        print(f" -> Atomically saving cell assignments to {assign_file.name} ...")
        df_assign = pd.DataFrame({
            "cell_barcode": barcodes,
            "local_cell_idx": np.arange(n_cells, dtype=np.int64),
            "x_um": coords[:, 0],
            "y_um": coords[:, 1],
            "predicted_niche_id": labels
        })
        temp_assign = assign_file.with_suffix(".tmp.csv")
        df_assign.to_csv(temp_assign, index=False)
        os.replace(temp_assign, assign_file)
        assign_sha = compute_file_sha256(assign_file)
        test_assignment_shas[s] = assign_sha
        print(f" -> Assignment CSV SHA-256: {assign_sha[:16]}...")

        per_seed_results[str(s)] = {
            "locked_r_star": r_star,
            "k_clusters": int(k_clusters),
            "internal_asw": float(asw_val),
            "effective_rank": float(erank),
            "variance_spectrum": var_spectrum,
            "max_cluster_ratio_p_max": float(p_max),
            "min_cluster_ratio_p_min": float(p_min),
            "cluster_entropy": float(entropy),
            "local_neighborhood_purity_mean": float(purity_mean),
            "local_neighborhood_purity_std": float(purity_std),
            "morans_i_top5_pcs": moran_pcs_results,
            "morans_i_cluster_onehot": moran_clusters_results,
            "embedding_file_sha256": emb_sha,
            "assignment_file_sha256": assign_sha
        }

    # 7. Cross-Seed Pairwise ARI on Test Set
    print("\n" + "=" * 85)
    print("STEP 5: COMPUTING CROSS-SEED PAIRWISE ARI ON TEST SET")
    print("=" * 85)
    test_labels = {
        s: pd.read_csv(ASSIGNMENTS_DIR / f"xenium_test_assignments_seed_{s}.csv")["predicted_niche_id"].to_numpy()
        for s in SEEDS
    }
    pair_aris = [
        float(adjusted_rand_score(test_labels[42], test_labels[123])),
        float(adjusted_rand_score(test_labels[42], test_labels[2024])),
        float(adjusted_rand_score(test_labels[123], test_labels[2024]))
    ]
    cross_seed_ari_mean = float(np.mean(pair_aris))
    cross_seed_ari_std = float(np.std(pair_aris, ddof=1))
    print(f"Cross-Seed Pairwise ARI at locked r* (0.4/0.2/0.6): {cross_seed_ari_mean:.4f} ± {cross_seed_ari_std:.4f}")
    print(f"  42 vs 123:  {pair_aris[0]:.4f}")
    print(f"  42 vs 2024: {pair_aris[1]:.4f}")
    print(f"  123 vs 2024:{pair_aris[2]:.4f}")

    # 8. Compute 3-Seed Aggregated Statistics
    all_ks = [per_seed_results[str(s)]["k_clusters"] for s in SEEDS]
    all_asws = [per_seed_results[str(s)]["internal_asw"] for s in SEEDS]
    all_eranks = [per_seed_results[str(s)]["effective_rank"] for s in SEEDS]
    all_p_max = [per_seed_results[str(s)]["max_cluster_ratio_p_max"] for s in SEEDS]
    all_entropy = [per_seed_results[str(s)]["cluster_entropy"] for s in SEEDS]
    all_purity = [per_seed_results[str(s)]["local_neighborhood_purity_mean"] for s in SEEDS]
    all_moran_pcs_delta = [per_seed_results[str(s)]["morans_i_top5_pcs"]["mean_delta_i"] for s in SEEDS]
    all_moran_pcs_z = [per_seed_results[str(s)]["morans_i_top5_pcs"]["mean_z_score"] for s in SEEDS]
    all_moran_cls_delta = [per_seed_results[str(s)]["morans_i_cluster_onehot"]["mean_delta_i"] for s in SEEDS]
    all_moran_cls_z = [per_seed_results[str(s)]["morans_i_cluster_onehot"]["mean_z_score"] for s in SEEDS]

    summary_stats = {
        "k_clusters": {"mean": float(np.mean(all_ks)), "std": float(np.std(all_ks, ddof=1))},
        "internal_asw": {"mean": float(np.mean(all_asws)), "std": float(np.std(all_asws, ddof=1))},
        "effective_rank": {"mean": float(np.mean(all_eranks)), "std": float(np.std(all_eranks, ddof=1))},
        "max_cluster_ratio_p_max": {"mean": float(np.mean(all_p_max)), "std": float(np.std(all_p_max, ddof=1))},
        "cluster_entropy": {"mean": float(np.mean(all_entropy)), "std": float(np.std(all_entropy, ddof=1))},
        "local_neighborhood_purity": {"mean": float(np.mean(all_purity)), "std": float(np.std(all_purity, ddof=1))},
        "moran_i_top5_pcs_delta": {"mean": float(np.mean(all_moran_pcs_delta)), "std": float(np.std(all_moran_pcs_delta, ddof=1))},
        "moran_i_top5_pcs_z_score": {"mean": float(np.mean(all_moran_pcs_z)), "std": float(np.std(all_moran_pcs_z, ddof=1))},
        "moran_i_cluster_onehot_delta": {"mean": float(np.mean(all_moran_cls_delta)), "std": float(np.std(all_moran_cls_delta, ddof=1))},
        "moran_i_cluster_onehot_z_score": {"mean": float(np.mean(all_moran_cls_z)), "std": float(np.std(all_moran_cls_z, ddof=1))},
        "cross_seed_pairwise_ari": {"mean": cross_seed_ari_mean, "std": cross_seed_ari_std}
    }

    # 9. Build and Sign Final Evaluation Report
    this_script_path = Path(__file__).resolve()
    this_script_sha = compute_file_sha256(this_script_path)

    report_payload = {
        "report_version": "1.0.0",
        "stage": "W2.7_Phase2_Test_Evaluation",
        "dataset": "xenium_ffpe_lungbreast_gse311609",
        "test_donor": TEST_DONOR_ID,
        "test_sample": TEST_SAMPLE_FNAME,
        "n_test_cells": n_cells,
        "bound_input_assets": {
            "test_runner_script": {
                "path": str(this_script_path.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                "sha256": this_script_sha
            },
            "phase1_selection_manifest": {
                "path": str(RESOLUTION_SELECTION_PATH.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                "physical_file_sha256": compute_file_sha256(RESOLUTION_SELECTION_PATH),
                "self_signature_sha256": phase1_manifest["selection_manifest_sha256"]
            },
            "phase1_evaluation_script": {
                "path": str(PHASE1_SCRIPT_PATH.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                "sha256": compute_file_sha256(PHASE1_SCRIPT_PATH)
            },
            "test_h5ad": {
                "path": str(test_h5ad_path.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                "sha256": h5ad_sha
            },
            "test_graph_a": {
                "path": str((CACHE_DIR / f"xenium_graph_{TEST_DONOR_ID}_A.npz").relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                "sha256": a_graph_sha
            },
            "test_graph_p": {
                "path": str((CACHE_DIR / f"xenium_graph_{TEST_DONOR_ID}_P.npz").relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                "sha256": p_graph_sha
            },
            "checkpoints": {
                str(s): {
                    "path": str((CHECKPOINTS_DIR / f"xenium_freqniche_trial24_seed_{s}.pt").relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                    "sha256": compute_file_sha256(CHECKPOINTS_DIR / f"xenium_freqniche_trial24_seed_{s}.pt")
                }
                for s in SEEDS
            },
            "test_embeddings": {
                str(s): {
                    "path": str((EMBEDDINGS_DIR / f"test_embeddings_seed_{s}.npy").relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                    "sha256": test_embedding_shas[s]
                }
                for s in SEEDS
            },
            "test_assignments": {
                str(s): {
                    "path": str((ASSIGNMENTS_DIR / f"xenium_test_assignments_seed_{s}.csv").relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                    "sha256": test_assignment_shas[s]
                }
                for s in SEEDS
            }
        },
        "protocol_contracts": {
            "spatial_radius_um": RADIUS_UM,
            "spatial_sigma_um": SIGMA_UM,
            "preprocessing": "sc.pp.normalize_total(target_sum=1e4) + sc.pp.log1p()",
            "inference_mode": "inductive_full_subgraph_deterministic_zero_masking",
            "k_max_subgraph": MODEL_CONFIG["k_max"],
            "embedding_dim": MODEL_CONFIG["d_model"],
            "locked_r_stars": frozen_r_stars,
            "spatial_negative_control": "feature_permutation_on_fixed_spatial_graph",
            "spatial_permutation_repeats": N_PERMUTATIONS,
            "ground_truth_isolation": "Strictly pure unsupervised evaluation. Zero reference niche labels used."
        },
        "per_seed_results": per_seed_results,
        "three_seed_summary_stats": summary_stats,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }

    payload_str = json.dumps(report_payload, sort_keys=True)
    report_sha = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
    report_payload["report_sha256"] = report_sha

    temp_report = FINAL_REPORT_PATH.with_suffix(".tmp.json")
    with open(temp_report, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    os.replace(temp_report, FINAL_REPORT_PATH)

    print("\n" + "=" * 85)
    print("SUCCESS: PHASE 2 TEST EVALUATION COMPLETE & CRYPTOGRAPHICALLY LOCKED")
    print(f"Report saved to: {FINAL_REPORT_PATH}")
    print(f"Report self-signature SHA-256: {report_sha}")
    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description="W2.7 Xenium 5K Phase 2 Test Evaluation Runner.")
    parser.add_argument("--authorized", action="store_true",
                        help="Must be explicitly passed to unlock test evaluation. Fail-closed without this flag.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device ('cuda' or 'cpu').")
    parser.add_argument("--recompute-embeddings", action="store_true",
                        help="Force recomputing test embeddings even if files exist.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing test evaluation report and outputs.")
    args = parser.parse_args()

    device = torch.device(args.device)

    run_test_evaluation(
        device=device,
        authorized=args.authorized,
        recompute_embeddings=args.recompute_embeddings,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
