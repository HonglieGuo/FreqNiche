"""
scripts/benchmark/train_xenium.py

W2.7 Xenium Human NSCLC 5K Formal Training & Real-Scale Preflight Runner
================================================================================
Strict Reproducibility & Cryptographic Verification Protocol:
  - 3 Random Seeds: 42, 123, 2024
  - Pre-requisites Cryptographic Audit:
      * Xenium audit manifest SHA-256 verified
      * 5,001 gene panel and order verified
      * Center cohorts verified with Triple Hashes (train_centers.json, val_centers.json)
      * Physical spatial graph caches (L3, L5, L7, L8, L6) verified
      * Native micron coordinates (obsm['spatial']) verified
      * Whitener contract: 5,001 -> 32 Train-only Truncated Whitening
  - Architecture: Frozen Trial 24 Configuration
      * in_features = 5001 (Xenium Prime 5K panel)
      * k_target = 32 (TruncatedTrainOnlyWhitening)
      * d_model = 128
      * n_enc_layers = 3 (Transformer Encoder 3 layers), n_heads = 8, mlp_ratio = 2.0
      * n_dit_layers = 2 (DiT 2 layers), n_heads = 8
      * timesteps = 50, lambda_cov = 0.04, lambda_var = 1.0, mode = 'full_freq'
  - Multi-Donor Context Architecture:
      * Train Split (399,672 cells): ConcatDataset across 4 Train donors (L3, L5, L7, L8)
      * Train Centers: exactly 60,000 centers (15,000 per donor, balanced)
      * Validation Split (64,819 cells): donor L6 with 10,000 frozen validation centers
  - Optimization:
      * AdamW(lr=1e-3, weight_decay=1e-3), clip_grad_norm=1.0, batch_size=64
      * Early stopping: patience=7, min_delta=1e-4 on Validation Denoising Loss
      * max_epochs = 50
  - Red Lines & Safeguards:
      * Absolute Test isolation: Test donor L1 (GSM9509134) is NEVER loaded or touched
      * Overwrite Protection: Fail-closed if checkpoint already exists without --overwrite
      * Real-Scale Preflight (--preflight): Tests real multi-donor pipeline and real validation
        across multiple batches without saving formal checkpoints
      * Unauthorized formal execution: strictly exits with non-zero exit code (exit code 2)
================================================================================
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import psutil
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.freqniche_st.dataset import SubgraphDataset, collate_subgraphs
from src.freqniche_st.models.freqniche_model import FreqNicheDiTModel
from src.freqniche_st.whitening import TruncatedTrainOnlyWhitening
from src.freqniche_st.runtime_paths import resolve_run_output
from src.freqniche_st.model_config import load_model_config, build_freqniche_model
from src.freqniche_st.checkpointing import atomic_torch_save, capture_rng_state, restore_rng_state
from src.freqniche_st.orchestration import atomic_json

OUTPUT_DIR = resolve_run_output("outputs/runs/freqniche/xenium_prime5k/2026-09-18/20260918_xenium_prime5k_final")
CKPT_DIR = OUTPUT_DIR / "checkpoints"
CACHE_DIR = OUTPUT_DIR / "cache"
GSE_DIR = REPO_ROOT / "datasets" / "h5ad" / "xenium_ffpe_lungbreast_gse311609" / "samples"

TRAIN_SAMPLES_META = [
    ("GSM9509135_NSCLC_5k_L3_L3.h5ad", "L3"),
    ("GSM9509136_NSCLC_5k_L5_L5.h5ad", "L5"),
    ("GSM9509138_NSCLC_5k_L7_L7.h5ad", "L7"),
    ("GSM9509139_NSCLC_5k_L8_L8.h5ad", "L8"),
]
VAL_SAMPLE_META = ("GSM9509137_NSCLC_5k_L6_L6.h5ad", "L6")

TRAIN_CENTERS_PATH = OUTPUT_DIR / "train_centers.json"
VAL_CENTERS_PATH = OUTPUT_DIR / "val_centers.json"
WHITENER_PATH = OUTPUT_DIR / "whitener.pt"
WHITENER_MANIFEST_PATH = OUTPUT_DIR / "whitener_manifest.json"
MANIFEST_PATH = OUTPUT_DIR / "xenium_data_audit_manifest.json"
COHORTS_MANIFEST_PATH = OUTPUT_DIR / "xenium_cohorts_manifest.json"
GRAPHS_MANIFEST_PATH = OUTPUT_DIR / "xenium_graphs_manifest.json"


def compute_file_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1048576):
            sha.update(chunk)
    return sha.hexdigest()


def completed_checkpoint_result(path: Path, seed: int, config: dict, budget: dict) -> dict:
    """Verify and reuse a complete seed, including legacy checkpoints from this run."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["seed"] != seed or payload["model_config"] != config:
        raise ValueError(f"Completed checkpoint seed/model mismatch: {path}")
    recorded_budget = payload.get("training_budget")
    if recorded_budget is None:
        manifest = json.loads((OUTPUT_DIR / "run_manifest.json").read_text(encoding="utf-8"))
        recorded_budget = {key: manifest["request"][key] for key in budget}
    if recorded_budget != budget:
        raise ValueError(f"Completed checkpoint training budget mismatch: {path}")
    actual = payload["total_epochs_trained"]
    if (actual != len(payload["train_loss_history"]) or actual != len(payload["val_loss_history"])
            or not 1 <= payload["best_epoch"] <= actual <= budget["max_epochs"]):
        raise ValueError(f"Incomplete checkpoint history: {path}")
    if payload["best_val_loss"] != payload["val_loss_history"][payload["best_epoch"] - 1]:
        raise ValueError(f"Best checkpoint loss/history mismatch: {path}")
    return {"seed": seed, "best_epoch": payload["best_epoch"], "best_val_loss": payload["best_val_loss"],
            "total_epochs": actual, "total_time_seconds": payload["total_time_seconds"],
            "cuda_operator_peak_allocated_mib": payload.get("cuda_operator_peak_allocated_mib", 0.0),
            "cuda_peak_reserved_mib": payload.get("cuda_peak_reserved_mib", 0.0),
            "host_rss_mib": payload.get("host_rss_mib", 0.0), "checkpoint_file": path.name,
            "checkpoint_sha256": compute_file_sha256(path), "model_config": config,
            "single_thread_backward": payload.get("single_thread_backward", False)}


def verify_prerequisites(require_whitener: bool = False):
    print("\n" + "=" * 80)
    print("STEP 1: CRYPTOGRAPHIC PREREQUISITES & AUDIT CONTRACT GATEKEEPING")
    print("=" * 80)

    assert MANIFEST_PATH.exists(), f"Audit manifest missing: {MANIFEST_PATH}"
    assert COHORTS_MANIFEST_PATH.exists(), f"Cohorts manifest missing: {COHORTS_MANIFEST_PATH}"
    assert GRAPHS_MANIFEST_PATH.exists(), f"Graphs manifest missing: {GRAPHS_MANIFEST_PATH}"
    assert TRAIN_CENTERS_PATH.exists(), f"Train centers missing: {TRAIN_CENTERS_PATH}"
    assert VAL_CENTERS_PATH.exists(), f"Val centers missing: {VAL_CENTERS_PATH}"

    # 1. Verify audit manifest and cohorts manifest integrity
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        audit_manifest = json.load(f)
    with open(COHORTS_MANIFEST_PATH, "r", encoding="utf-8") as f:
        cm = json.load(f)
    with open(GRAPHS_MANIFEST_PATH, "r", encoding="utf-8") as f:
        gm = json.load(f)

    train_file_sha = compute_file_sha256(TRAIN_CENTERS_PATH)
    assert train_file_sha == cm["train_cohort"]["file_sha256"], "Train centers SHA-256 mismatch!"
    val_file_sha = compute_file_sha256(VAL_CENTERS_PATH)
    assert val_file_sha == cm["validation_cohort"]["file_sha256"], "Val centers SHA-256 mismatch!"

    print(f" [PASS] Manifests verified: {MANIFEST_PATH.name}, {COHORTS_MANIFEST_PATH.name}, {GRAPHS_MANIFEST_PATH.name}")
    print(f" [PASS] Train centers verified: {cm['train_cohort']['total_centers']:,} centers (SHA: {train_file_sha[:16]}...)")
    print(f" [PASS] Val centers verified:   {cm['validation_cohort']['total_centers']:,} centers (SHA: {val_file_sha[:16]}...)")

    # 2. Hardcoded Test Isolation Guard
    for fname, donor in TRAIN_SAMPLES_META:
        assert "L1" not in donor and "GSM9509134" not in fname, "CRITICAL: Test donor L1 found in Train samples!"
    assert "L1" not in VAL_SAMPLE_META[1] and "GSM9509134" not in VAL_SAMPLE_META[0], "CRITICAL: Test donor L1 found in Val sample!"
    print(" [PASS] Absolute Test Isolation verified: Test donor L1 (GSM9509134) strictly excluded.")

    # 3. Verify Source H5AD SHA-256, 5,001 Genes Order, and Center Row Mapping
    expected_genes_sha = audit_manifest["gene_panel"]["ordered_genes_sha256"]
    with open(TRAIN_CENTERS_PATH, "r", encoding="utf-8") as f:
        train_centers = json.load(f)
    with open(VAL_CENTERS_PATH, "r", encoding="utf-8") as f:
        val_centers = json.load(f)

    centers_by_donor: Dict[str, List[Dict[str, Any]]] = {}
    for c in train_centers:
        centers_by_donor.setdefault(c["donor_id"], []).append(c)
    for c in val_centers:
        centers_by_donor.setdefault(c["donor_id"], []).append(c)

    all_samples = list(TRAIN_SAMPLES_META) + [VAL_SAMPLE_META]
    for fname, donor in all_samples:
        h5ad_path = GSE_DIR / fname
        assert h5ad_path.exists(), f"Source H5AD missing: {h5ad_path}"

        # Verify H5AD SHA-256
        actual_h5ad_sha = compute_file_sha256(h5ad_path)
        expected_h5ad_sha = audit_manifest["primary_cohort"]["samples"][fname]["file_sha256"]
        assert actual_h5ad_sha == expected_h5ad_sha, f"H5AD SHA-256 mismatch for {fname}: got {actual_h5ad_sha}, expected {expected_h5ad_sha}"

        # Verify Gene Panel and Exact Order
        a = ad.read_h5ad(h5ad_path, backed="r")
        assert a.n_vars == 5001, f"Expected 5,001 genes in {fname}, got {a.n_vars}"
        genes_list = list(a.var_names)
        actual_genes_sha = hashlib.sha256(("\n".join(genes_list)).encode("utf-8")).hexdigest()
        assert actual_genes_sha == expected_genes_sha, f"Gene order SHA-256 mismatch in {fname}!"

        # Verify Graph Cache SHAs
        p_file = CACHE_DIR / gm["graphs"][donor]["p_file"]
        a_file = CACHE_DIR / gm["graphs"][donor]["a_file"]
        assert p_file.exists() and a_file.exists(), f"Graph cache files missing for {donor}"
        assert compute_file_sha256(p_file) == gm["graphs"][donor]["p_sha256"], f"P graph SHA mismatch for {donor}"
        assert compute_file_sha256(a_file) == gm["graphs"][donor]["a_sha256"], f"A graph SHA mismatch for {donor}"

        # Vectorized verification of Center Row Index to Barcode and Spatial Coordinates
        donor_c_list = centers_by_donor[donor]
        c_indices = np.array([c["local_cell_idx"] for c in donor_c_list], dtype=np.int64)
        assert (c_indices >= 0).all() and (c_indices < a.n_obs).all(), f"Center index out of bounds in {donor}"

        expected_barcodes = np.array([c["cell_barcode"] for c in donor_c_list])
        actual_barcodes = np.asarray(a.obs_names)[c_indices]
        assert (actual_barcodes == expected_barcodes).all(), f"Barcode mapping mismatch in {donor}!"

        expected_coords = np.array([[c["x_um"], c["y_um"]] for c in donor_c_list], dtype=np.float32)
        actual_coords = np.asarray(a.obsm["spatial"])[c_indices].astype(np.float32)
        assert np.allclose(actual_coords, expected_coords, atol=1e-3), f"Spatial coordinates mismatch in {donor}!"

        print(f" [PASS] Donor {donor} ({fname}): H5AD SHA, 5001-gene order, Graph caches, and {len(donor_c_list):,} center barcodes/coords verified.")

    # 4. Whitener verification (when formal training is requested)
    if require_whitener:
        assert WHITENER_PATH.exists(), f"Whitener file missing for formal training: {WHITENER_PATH}"
        assert WHITENER_MANIFEST_PATH.exists(), f"Whitener manifest missing: {WHITENER_MANIFEST_PATH}"
        with open(WHITENER_MANIFEST_PATH, "r", encoding="utf-8") as f:
            wm = json.load(f)
        actual_wh_sha = compute_file_sha256(WHITENER_PATH)
        assert actual_wh_sha == wm["file_sha256"], f"Whitener SHA-256 mismatch! File: {actual_wh_sha}, Manifest: {wm['file_sha256']}"
        assert train_file_sha == wm["train_centers_sha256"], "Whitener manifest train centers SHA does not match current frozen centers!"
        assert wm["whitening_matrix_shape"] == [5001, 32], f"Unexpected whitening shape in manifest: {wm['whitening_matrix_shape']}"
        print(f" [PASS] Formal Whitener verified: {WHITENER_PATH.name} (SHA: {actual_wh_sha[:16]}...) linked to Manifest.")


def load_all_donor_contexts() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Preloads Train donors (L3, L5, L7, L8) and Validation donor (L6)
    into host memory once so multi-seed training does not repeatedly reload H5ADs.
    """
    print("\n" + "=" * 80)
    print("STEP 2: PRELOADING IMMUTABLE DONOR CONTEXTS INTO HOST RAM")
    print("=" * 80)
    t_start = time.time()
    with open(TRAIN_CENTERS_PATH, "r", encoding="utf-8") as f:
        train_centers = json.load(f)
    with open(VAL_CENTERS_PATH, "r", encoding="utf-8") as f:
        val_centers = json.load(f)

    centers_by_donor: Dict[str, List[int]] = {}
    for c in train_centers:
        centers_by_donor.setdefault(c["donor_id"], []).append(c["local_cell_idx"])

    train_contexts: Dict[str, Dict[str, Any]] = {}
    for fname, donor in TRAIN_SAMPLES_META:
        t0 = time.time()
        p_cache = CACHE_DIR / f"xenium_graph_{donor}_P.npz"
        a_cache = CACHE_DIR / f"xenium_graph_{donor}_A.npz"
        h5ad_p = GSE_DIR / fname

        P_csr = sp.load_npz(p_cache)
        A_csr = sp.load_npz(a_cache)
        a = ad.read_h5ad(h5ad_p, backed="r")
        coords = np.asarray(a.obsm["spatial"], dtype=np.float32)

        raw_X = a.X[:]
        if sp.issparse(raw_X):
            X_dense = raw_X.toarray().astype(np.float32)
        else:
            X_dense = np.asarray(raw_X, dtype=np.float32)
        counts = np.maximum(X_dense.sum(axis=1, keepdims=True), 1.0)
        X_norm = np.log1p((X_dense / counts) * 1e4)

        center_indices = np.array(centers_by_donor[donor], dtype=np.int64)
        train_contexts[donor] = {
            "center_indices": center_indices,
            "A_global": A_csr,
            "P_global": P_csr,
            "X_global": X_norm,
            "coords_global": coords,
        }
        print(f"  - Preloaded Train donor {donor}: {len(center_indices):,} centers on {len(coords):,} cell context ({time.time() - t0:.2f} s)")

    # Preload Validation Donor (L6)
    t0_val = time.time()
    val_fname, val_donor = VAL_SAMPLE_META
    val_p_cache = CACHE_DIR / f"xenium_graph_{val_donor}_P.npz"
    val_a_cache = CACHE_DIR / f"xenium_graph_{val_donor}_A.npz"
    val_h5ad_p = GSE_DIR / val_fname

    val_P = sp.load_npz(val_p_cache)
    val_A = sp.load_npz(val_a_cache)
    a_val = ad.read_h5ad(val_h5ad_p, backed="r")
    val_coords = np.asarray(a_val.obsm["spatial"], dtype=np.float32)
    raw_val_X = a_val.X[:]
    if sp.issparse(raw_val_X):
        val_X_dense = raw_val_X.toarray().astype(np.float32)
    else:
        val_X_dense = np.asarray(raw_val_X, dtype=np.float32)
    val_counts = np.maximum(val_X_dense.sum(axis=1, keepdims=True), 1.0)
    val_X_norm = np.log1p((val_X_dense / val_counts) * 1e4)

    val_center_indices = np.array([c["local_cell_idx"] for c in val_centers], dtype=np.int64)
    val_context = {
        "center_indices": val_center_indices,
        "A_global": val_A,
        "P_global": val_P,
        "X_global": val_X_norm,
        "coords_global": val_coords,
    }
    print(f"  - Preloaded Validation donor {val_donor}: {len(val_center_indices):,} centers on {len(val_coords):,} cell context ({time.time() - t0_val:.2f} s)")

    host_rss_mb = float(psutil.Process().memory_info().rss / (1024**2))
    print(f"  - Total preloading time: {time.time() - t_start:.2f} s | Host Process RSS: {host_rss_mb:.1f} MB (~{host_rss_mb/1024:.2f} GB)")
    return train_contexts, val_context


def build_datasets_for_seed(
    train_contexts: Dict[str, Dict[str, Any]],
    val_context: Dict[str, Any],
    whitener: Optional[TruncatedTrainOnlyWhitening],
    seed: int,
    k_max: int = 64,
    m_max: int = 32,
    rho: float = 0.5,
    p_center_mask: float = 0.3,
) -> Tuple[ConcatDataset, SubgraphDataset]:
    """
    Constructs fresh SubgraphDataset instances with explicit seed=seed for strict random stream isolation.
    """
    train_datasets = []
    for fname, donor in TRAIN_SAMPLES_META:
        ctx = train_contexts[donor]
        ds_d = SubgraphDataset(
            center_indices=ctx["center_indices"],
            A_global=ctx["A_global"],
            P_global=ctx["P_global"],
            X_global=ctx["X_global"],
            coords_global=ctx["coords_global"],
            whitener=whitener,
            k_max=k_max,
            m_max=m_max,
            rho=rho,
            p_center_mask=p_center_mask,
            is_train=True,
            seed=seed,
        )
        train_datasets.append(ds_d)

    train_dataset = ConcatDataset(train_datasets)
    assert len(train_dataset) == 60000, f"Expected 60,000 train centers, got {len(train_dataset)}"

    val_dataset = SubgraphDataset(
        center_indices=val_context["center_indices"],
        A_global=val_context["A_global"],
        P_global=val_context["P_global"],
        X_global=val_context["X_global"],
        coords_global=val_context["coords_global"],
        whitener=whitener,
        k_max=k_max,
        m_max=m_max,
        rho=rho,
        p_center_mask=p_center_mask,
        is_train=False,
        is_val_objective=True,
        seed=seed,
    )
    assert len(val_dataset) == 10000, f"Expected 10,000 val centers, got {len(val_dataset)}"

    return train_dataset, val_dataset


def run_real_scale_preflight(
    device: torch.device,
    batch_size: int = 64,
    n_train_batches: int = 10,
    n_val_batches: int = 5,
    model_config: Dict[str, Any] | None = None,
):
    """
    Executes real-scale preflight across full 4-donor Train context and full Val donor context.
    Evaluates real multi-batch DataLoader latency, real forward+backward steps, and real validation.
    DOES NOT SAVE CHECKPOINTS.
    """
    print("\n" + "=" * 85)
    print("W2.7 REAL-SCALE PREFLIGHT RUNNER (Zero Formal Checkpoints Saved)")
    print(f"Device: {device} | Batch Size: {batch_size} | Test Batches: Train={n_train_batches}, Val={n_val_batches}")
    print("=" * 85)
    cfg = model_config or load_model_config()

    # 1. Fit in-memory Whitener via fit_truncated_whitener_chunked
    if str(REPO_ROOT / "scripts" / "benchmark") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))
    from fit_xenium_whitener import fit_truncated_whitener_chunked
    print("\n[Preflight Step 1/4] Fitting in-memory chunked whitener across 60,000 real centers ...")
    whitener, wh_telem = fit_truncated_whitener_chunked(device_str=str(device))
    assert whitener.whitening_matrix_.shape == (5001, 32)

    # 2. Preload donor contexts and verify seed isolation
    print("\n[Preflight Step 2/4] Preloading contexts and validating Seed Random Stream Isolation ...")
    train_contexts, val_context = load_all_donor_contexts()

    dataset_settings = {key: cfg[key] for key in ("k_max", "m_max", "rho", "p_center_mask")}
    ds_s42, val_s42 = build_datasets_for_seed(train_contexts, val_context, whitener, seed=42, **dataset_settings)
    ds_s123, val_s123 = build_datasets_for_seed(train_contexts, val_context, whitener, seed=123, **dataset_settings)

    item_42 = ds_s42[0]
    item_123 = ds_s123[0]
    assert item_42["center_idx"] == item_123["center_idx"], "Sanity check: center index must match for same index."
    # Because seed=42 and seed=123 have independent RNGs, their sampled subgraph targets and visible nodes differ
    diff_targets = not torch.equal(item_42["delta_s_target"], item_123["delta_s_target"])
    diff_vis = not torch.equal(item_42["delta_s_vis"], item_123["delta_s_vis"])
    diff_stream = diff_targets or diff_vis
    print(f"  - [PASS] Seed Isolation Contract verified: SubgraphDataset sampled nodes differ between seeds (diff_targets={diff_targets}, diff_vis={diff_vis})")
    assert diff_stream, "CRITICAL: SubgraphDataset produced identical sampled nodes for different seeds!"

    # 3. Create preflight DataLoader for seed 42
    train_gen = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        ds_s42,
        batch_size=batch_size,
        shuffle=True,
        generator=train_gen,
        collate_fn=collate_subgraphs,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_s42,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_subgraphs,
        num_workers=0,
    )

    # 4. Instantiate Trial 24 Model
    print(f"\n[Preflight Step 3/4] Instantiating model (5,001 genes, Enc={cfg['n_enc']}, DiT={cfg['n_dit']}, Heads={cfg['n_heads']}) ...")
    model = build_freqniche_model(5001, cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    # 5. Run real training steps
    print(f"\n[Preflight Step 4/4] Executing {n_train_batches} train forward/backward steps + {n_val_batches} val steps ...")
    train_iter = iter(train_loader)
    train_step_durations = []
    train_losses = []

    model.train()
    for step_i in range(n_train_batches):
        t0_s = time.perf_counter()
        batch = next(train_iter)
        t_data = time.perf_counter() - t0_s

        t0_comp = time.perf_counter()
        optimizer.zero_grad()
        out = model(
            H_0_target=batch["H_0_target"].to(device),
            X_vis=batch["X_vis"].to(device),
            delta_s_vis=batch["delta_s_vis"].to(device),
            vis_pad_mask=batch["vis_pad_mask"].to(device),
            delta_s_target=batch["delta_s_target"].to(device),
            target_mask=batch["target_mask"].to(device),
            L_vis_cond=batch["L_vis_cond"].to(device),
            H_vis_cond=batch["H_vis_cond"].to(device),
        )
        loss = out["loss"]
        assert torch.isfinite(loss).all(), f"Non-finite loss at step {step_i}: {loss}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_comp = time.perf_counter() - t0_comp
        t_step = time.perf_counter() - t0_s

        train_step_durations.append(t_step)
        train_losses.append(loss.item())
        print(f"  - Train Step {step_i+1:02d}/{n_train_batches:02d}: Loss={loss.item():.4f} | Total={t_step:.3f}s (Data={t_data:.3f}s, Comp={t_comp:.3f}s)")

    # Run real validation steps
    model.eval()
    val_iter = iter(val_loader)
    val_step_durations = []
    val_losses = []
    val_gen = torch.Generator(device=device).manual_seed(42 + 99999)

    with torch.no_grad():
        for val_i in range(n_val_batches):
            t0_v = time.perf_counter()
            b_val = next(val_iter)
            H_0 = b_val["H_0_target"].to(device)
            B_val = H_0.shape[0]
            t_val = torch.randint(1, model.timesteps + 1, size=(B_val,), device=device, generator=val_gen)
            noise_val = torch.randn(H_0.shape, device=device, generator=val_gen)
            H_t_val = model.schedule.q_sample(x_0=H_0, t=t_val, noise=noise_val)

            z_val_niche = model.encoder(
                X_vis=b_val["X_vis"].to(device),
                delta_s_vis=b_val["delta_s_vis"].to(device),
                vis_pad_mask=b_val["vis_pad_mask"].to(device),
                L_vis_cond=b_val["L_vis_cond"].to(device),
                H_vis_cond=b_val["H_vis_cond"].to(device),
            )
            pred_noise_val = model.denoiser(
                H_t=H_t_val,
                t=t_val,
                z_v_niche=z_val_niche,
                delta_s_target=b_val["delta_s_target"].to(device),
                target_mask=b_val["target_mask"].to(device),
            )
            diff_sq = (pred_noise_val - noise_val) ** 2
            token_l = diff_sq.sum(dim=-1)
            m_flt = b_val["target_mask"].to(device).float()
            val_loss_batch = (token_l * m_flt).sum() / max(m_flt.sum(), 1.0)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t_val_step = time.perf_counter() - t0_v

            val_step_durations.append(t_val_step)
            val_losses.append(val_loss_batch.item())
            print(f"  - Val Step {val_i+1:02d}/{n_val_batches:02d}: Denoise Loss={val_loss_batch.item():.4f} | Time={t_val_step:.3f}s")

    peak_vram_alloc = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    peak_vram_reserved = torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else 0.0
    host_rss_mb = float(psutil.Process().memory_info().rss / (1024**2))

    avg_step_sec = float(np.mean(train_step_durations))
    steps_per_epoch = int(np.ceil(60000 / batch_size))
    est_train_epoch_sec = steps_per_epoch * avg_step_sec
    val_steps_total = int(np.ceil(10000 / batch_size))
    est_val_epoch_sec = val_steps_total * float(np.mean(val_step_durations))
    est_total_epoch_sec = est_train_epoch_sec + est_val_epoch_sec
    est_total_epoch_min = est_total_epoch_sec / 60.0

    print("\n" + "=" * 85)
    print("REAL-SCALE PREFLIGHT TELEMETRY REPORT (60k Train Context + 10k Val Context)")
    print("=" * 85)
    print(f"  - Multi-Donor Context:                   399,672 Train cells + 64,819 Val cells (5,001 Genes)")
    print(f"  - Model Parameters:                      {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  - PyTorch CUDA Operator Peak Allocated:  {peak_vram_alloc:.2f} MiB")
    print(f"  - PyTorch CUDA Reserved Caching Memory:  {peak_vram_reserved:.2f} MiB")
    print(f"  - Host Process RSS Memory:               {host_rss_mb:.1f} MB (~{host_rss_mb/1024:.2f} GB)")
    print(f"  - Mean Training Step Latency:            {avg_step_sec:.3f} s ({batch_size / avg_step_sec:.1f} centers/s)")
    print(f"  - Batch-Extrapolated Epoch Duration:     ~{est_total_epoch_min:.1f} min/epoch (Extrapolation based on")
    print(f"                                            {n_train_batches} train batches; actual throughput varies with system")
    print(f"                                            load and disk I/O; does NOT constitute a hard runtime or zero-OOM guarantee)")
    print(f"  - Loss Trajectory (Finite):              Train {train_losses[0]:.4f} -> {train_losses[-1]:.4f} | Val mean {np.mean(val_losses):.4f}")
    print("=" * 85)
    print("[PASS] Real-Scale Preflight verified with complete random stream isolation. ZERO formal assets saved.")


def train_single_seed(
    seed: int,
    device: torch.device,
    train_contexts: Dict[str, Dict[str, Any]],
    val_context: Dict[str, Any],
    whitener: TruncatedTrainOnlyWhitening,
    batch_size: int = 64,
    max_epochs: int = 50,
    patience: int = 7,
    min_delta: float = 1e-4,
    overwrite: bool = False,
    model_config: Dict[str, Any] | None = None,
    resume: bool = False,
    single_thread_backward: bool = False,
) -> Dict[str, Any]:
    """
    Formal full training for a single seed with strict random isolation.
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"xenium_freqniche_trial24_seed_{seed}.pt"
    resume_path = CKPT_DIR / f"xenium_resume_seed_{seed}.pt"
    cfg = model_config or load_model_config()
    budget = {"max_epochs": max_epochs, "patience": patience, "batch_size": batch_size, "min_delta": min_delta}

    if ckpt_path.exists() and resume:
        result = completed_checkpoint_result(ckpt_path, seed, cfg, budget)
        print(f"[REUSE COMPLETE] Seed {seed}: {ckpt_path}", flush=True)
        return result

    if ckpt_path.exists() and not overwrite:
        raise RuntimeError(f"FAIL-CLOSED: Checkpoint {ckpt_path.name} already exists. Pass --overwrite to replace.")
    if resume_path.exists() and not resume and not overwrite:
        raise RuntimeError(f"Resume state exists: {resume_path}; pass --resume instead of silently restarting")

    print("\n" + "=" * 80)
    print(f"STARTING FORMAL TRAINING FOR SEED {seed}")
    print("=" * 80)

    # 1. Strict RNG seeding per seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # 2. Build seed-isolated datasets
    train_dataset, val_dataset = build_datasets_for_seed(
        train_contexts=train_contexts,
        val_context=val_context,
        whitener=whitener,
        seed=seed,
        k_max=cfg["k_max"], m_max=cfg["m_max"],
        rho=cfg["rho"], p_center_mask=cfg["p_center_mask"],
    )

    train_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_gen,
        collate_fn=collate_subgraphs,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_subgraphs,
        num_workers=0,
    )

    model = build_freqniche_model(5001, cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None
    patience_counter = 0
    train_loss_history = []
    val_loss_history = []

    contract = {"seed": seed, "model_config": cfg, "training_budget": budget,
                "single_thread_backward": single_thread_backward, "device": str(device),
                "runner_sha256": compute_file_sha256(Path(__file__)),
                "checkpointing_sha256": compute_file_sha256(REPO_ROOT / "src/freqniche_st/checkpointing.py"),
                "torch_version": str(torch.__version__), "torch_cuda_version": torch.version.cuda,
                "assets": {path.name: compute_file_sha256(path) for path in (
                    MANIFEST_PATH, COHORTS_MANIFEST_PATH, GRAPHS_MANIFEST_PATH,
                    TRAIN_CENTERS_PATH, VAL_CENTERS_PATH, WHITENER_PATH, WHITENER_MANIFEST_PATH)}}
    start_epoch, previous_elapsed = 1, 0.0
    if resume and resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        if saved["contract"] != contract:
            raise ValueError(f"Resume contract changed: {resume_path}")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        best_val_loss, best_epoch = saved["best_val_loss"], saved["best_epoch"]
        best_state_dict, patience_counter = saved["best_state_dict"], saved["patience_counter"]
        train_loss_history, val_loss_history = saved["train_loss_history"], saved["val_loss_history"]
        start_epoch, previous_elapsed = saved["epoch"] + 1, saved["elapsed_seconds"]
        if len(train_loss_history) != saved["epoch"] or len(val_loss_history) != saved["epoch"]:
            raise ValueError("Resume epoch/history mismatch")
        restore_rng_state(saved["rng"], train_gen, train_dataset, val_dataset, device)
        print(f"[RESUME] Seed {seed} from epoch {start_epoch}; best={best_val_loss:.6f} @ {best_epoch}", flush=True)
    elif resume:
        print(f"[START MISSING SEED] Seed {seed}; no saved epoch state, starting at epoch 1", flush=True)
    print(f"[RUNTIME] single_thread_backward={single_thread_backward}; epoch checkpoints={resume_path}", flush=True)
    t_start = time.perf_counter()

    epochs = range(start_epoch, max_epochs + 1) if patience_counter < patience else ()
    for epoch in epochs:
        t_ep_start = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for b in train_loader:
            optimizer.zero_grad()
            out = model(
                H_0_target=b["H_0_target"].to(device),
                X_vis=b["X_vis"].to(device),
                delta_s_vis=b["delta_s_vis"].to(device),
                vis_pad_mask=b["vis_pad_mask"].to(device),
                delta_s_target=b["delta_s_target"].to(device),
                target_mask=b["target_mask"].to(device),
                L_vis_cond=b["L_vis_cond"].to(device),
                H_vis_cond=b["H_vis_cond"].to(device),
            )
            loss = out["loss"]
            assert torch.isfinite(loss).all(), f"Non-finite loss in epoch {epoch}: {loss}"
            with torch.autograd.set_multithreading_enabled(not single_thread_backward):
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_loss_history.append(avg_train_loss)

        # Validation evaluation
        model.eval()
        val_losses = []
        val_gen = torch.Generator(device=device).manual_seed(seed + 99999)

        with torch.no_grad():
            for b_val in val_loader:
                H_0 = b_val["H_0_target"].to(device)
                B_val = H_0.shape[0]
                t_val = torch.randint(1, model.timesteps + 1, size=(B_val,), device=device, generator=val_gen)
                noise_val = torch.randn(H_0.shape, device=device, generator=val_gen)
                H_t_val = model.schedule.q_sample(x_0=H_0, t=t_val, noise=noise_val)

                z_val_niche = model.encoder(
                    X_vis=b_val["X_vis"].to(device),
                    delta_s_vis=b_val["delta_s_vis"].to(device),
                    vis_pad_mask=b_val["vis_pad_mask"].to(device),
                    L_vis_cond=b_val["L_vis_cond"].to(device),
                    H_vis_cond=b_val["H_vis_cond"].to(device),
                )
                pred_noise_val = model.denoiser(
                    H_t=H_t_val,
                    t=t_val,
                    z_v_niche=z_val_niche,
                    delta_s_target=b_val["delta_s_target"].to(device),
                    target_mask=b_val["target_mask"].to(device),
                )
                diff_sq = (pred_noise_val - noise_val) ** 2
                token_l = diff_sq.sum(dim=-1)
                m_flt = b_val["target_mask"].to(device).float()
                val_loss_batch = (token_l * m_flt).sum() / max(m_flt.sum(), 1.0)
                val_losses.append(val_loss_batch.item())

        mean_val_loss = float(np.mean(val_losses))
        if not np.isfinite(mean_val_loss):
            raise RuntimeError(f"Non-finite Validation loss at epoch {epoch}")
        val_loss_history.append(mean_val_loss)
        ep_duration = time.perf_counter() - t_ep_start

        is_best = False
        if mean_val_loss < (best_val_loss - min_delta):
            best_val_loss = mean_val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
            is_best = True
        else:
            patience_counter += 1

        atomic_torch_save({"contract": contract, "epoch": epoch,
                           "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                           "best_state_dict": best_state_dict, "best_val_loss": best_val_loss,
                           "best_epoch": best_epoch, "patience_counter": patience_counter,
                           "train_loss_history": train_loss_history, "val_loss_history": val_loss_history,
                           "elapsed_seconds": previous_elapsed + time.perf_counter() - t_start,
                           "rng": capture_rng_state(train_gen, train_dataset, val_dataset, device)}, resume_path)

        best_flag = " [* BEST]" if is_best else f" [Patience {patience_counter}/{patience}]"
        print(
            f"Seed {seed:4d} | Epoch {epoch:02d}/{max_epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Denoise Loss: {mean_val_loss:.4f} | "
            f"Duration: {ep_duration:.1f}s{best_flag}"
        )

        if patience_counter >= patience:
            print(f"\n[Early Stopping] No improvement for {patience} epochs. Stopped at epoch {epoch}.")
            break

    total_time = previous_elapsed + time.perf_counter() - t_start
    cuda_operator_peak_allocated_mib = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    cuda_peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else 0.0
    host_rss_mib = float(psutil.Process().memory_info().rss / (1024**2))

    # Save checkpoint atomically
    save_payload = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_epochs_trained": len(train_loss_history),
        "total_time_seconds": total_time,
        "cuda_operator_peak_allocated_mib": cuda_operator_peak_allocated_mib,
        "cuda_peak_reserved_mib": cuda_peak_reserved_mib,
        "host_rss_mib": host_rss_mib,
        "hyperparameters": {
            "in_features": 5001,
            "k_target": 32,
            "d_model": cfg["d_model"],
            "n_enc_layers": cfg["n_enc"],
            "n_dit_layers": cfg["n_dit"],
            "n_heads": cfg["n_heads"],
            "timesteps": cfg["timesteps"],
            "lambda_cov": cfg["lambda_cov"],
            "lambda_var": 1.0,
            "radius_um": 50.0,
            "sigma_um": 16.666667,
        },
        "model_config": cfg,
        "training_budget": budget,
        "single_thread_backward": single_thread_backward,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "model_state_dict": best_state_dict,
    }

    atomic_torch_save(save_payload, ckpt_path)
    ckpt_sha = compute_file_sha256(ckpt_path)
    print(f"Saved Checkpoint: {ckpt_path} (SHA: {ckpt_sha[:16]}...)")

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_epochs": len(train_loss_history),
        "total_time_seconds": total_time,
        "cuda_operator_peak_allocated_mib": cuda_operator_peak_allocated_mib,
        "cuda_peak_reserved_mib": cuda_peak_reserved_mib,
        "host_rss_mib": host_rss_mib,
        "checkpoint_file": ckpt_path.name,
        "checkpoint_sha256": ckpt_sha,
        "model_config": cfg,
        "single_thread_backward": single_thread_backward,
    }


def main():
    parser = argparse.ArgumentParser(description="W2.7 Xenium Training & Preflight Runner")
    parser.add_argument("--preflight", action="store_true", help="Run real-scale multi-donor preflight without saving checkpoints")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds to train formally")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=64)
    parser.add_argument("--max_epochs", "--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min_delta", "--min-delta", type=float, default=1e-4)
    parser.add_argument("--model-config-path", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Allow overwrite of existing checkpoints")
    parser.add_argument("--resume", action="store_true", help="Reuse completed seeds and resume full epoch checkpoints")
    parser.add_argument("--single-thread-backward", action="store_true", help="Run autograd backward on the calling thread")
    parser.add_argument("--authorized", action="store_true", help="Explicit authorization flag for formal training")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be combined")
    model_config = load_model_config(args.model_config_path, output_dir=OUTPUT_DIR if args.model_config_path else None)
    if args.max_epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.min_delta < 0:
        parser.error("max-epochs, patience and batch-size must be positive; min-delta must be nonnegative")

    device = torch.device(args.device)

    # 1. Real-scale preflight mode
    if args.preflight:
        verify_prerequisites(require_whitener=False)
        run_real_scale_preflight(device=device, batch_size=args.batch_size, model_config=model_config)
        return

    # 2. Formal training mode: STRICT FAIL-CLOSED PROTECTION
    if not args.authorized:
        print("\n" + "!" * 80)
        print("[CRITICAL SECURITY INTERCEPTION] FORMAL TRAINING IS STRICTLY LOCKED!")
        print("Formal 3-Seed training cannot proceed without explicit user audit authorization.")
        print("To run Preflight telemetry, use: python scripts/benchmark/train_xenium.py --preflight")
        print("!" * 80)
        sys.exit(2)  # Fail-closed with non-zero exit code

    # If explicitly authorized by user:
    verify_prerequisites(require_whitener=True)
    # Strictly load with weights_only=True. Fails immediately if payload contains untrusted types.
    wh_state = torch.load(WHITENER_PATH, map_location="cpu", weights_only=True)

    def _to_numpy(val):
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy()
        return np.asarray(val)

    whitener = TruncatedTrainOnlyWhitening(k_components=32)
    whitener.whitening_matrix_ = _to_numpy(wh_state["whitening_matrix"])
    whitener.mean_ = _to_numpy(wh_state["mean"])
    whitener.components_ = _to_numpy(wh_state["components"])
    whitener.singular_values_ = _to_numpy(wh_state["singular_values"])
    whitener.n_components_fitted_ = 32
    whitener.is_fitted = True

    # Preload donor contexts once
    train_contexts, val_context = load_all_donor_contexts()

    results = []
    for seed in args.seeds:
        res = train_single_seed(
            seed=seed,
            device=device,
            train_contexts=train_contexts,
            val_context=val_context,
            whitener=whitener,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            overwrite=args.overwrite,
            model_config=model_config,
            resume=args.resume,
            single_thread_backward=args.single_thread_backward,
        )
        results.append(res)

        # Publish per-seed completion metadata immediately, even if a later seed fails.
        atomic_json(CKPT_DIR / "xenium_training_summary.json", results)

    summary_path = CKPT_DIR / "xenium_training_summary.json"
    atomic_json(summary_path, results)
    print(f"\n[ALL SEEDS FINISHED] Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
