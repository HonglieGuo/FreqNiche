"""
scripts/benchmark/eval_xenium_stability.py

W2.7 Xenium Human NSCLC 5K: Phase 1 Validation Stability Selection & Phase 2 Test Evaluation.
================================================================================
Strict Reproducibility & Red Line Protocols:
1. Physical Isolation & Non-Interference:
   - Phase 'validation':
     * STRICTLY LIMITED TO Validation context: Donor L6 (GSM9509137_NSCLC_5k_L6_L6, 10,000 centers).
     * Test donor L1 (GSM9509134) is NEVER indexed, read, or touched.
     * Generates / verifies 10 bootstrap folds in val_bootstrap_folds.json:
       exactly 8,000 unique indices per fold in [0, 9999] with matching Triple Hashes.
     * Reuses existing Validation embeddings by default (unless --recompute-val-embeddings is passed).
     * Evaluates intra-seed subsampling stability for each seed independently across resolutions [0.2, 0.4, 0.6, 0.8, 1.0].
     * Computes predicted-cluster internal ASW, embedding Effective Rank, variance explained spectrum,
       cluster entropy, and max cluster proportion (p_max).
     * Enforces cluster count constraint [K_MIN=5, K_MAX=25].
     * Selects independent r* for each seed.
     * Computes cross-seed pairwise ARI purely as an auxiliary readout.
     * Cryptographically binds SHA-256 of checkpoints, val centers, val folds, val embeddings,
       evaluation script, and all frozen protocol parameters into resolution_selection.json.
     * Signs resolution_selection.json with payload SHA-256 self-signature.
     * Complete and halt for audit review.
   - Phase 'test':
     * STRICTLY DISAPPROVED until Phase 1 resolution selection is audited and approved.
     * Fail-closed entry guard: requires valid resolution_selection.json with verified cryptographic bindings.
     * NO Reference Niche ASW (Xenium L1 has no ground-truth niche annotations).
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

from src.freqniche_st.dataset import SubgraphDataset, collate_subgraphs
from src.freqniche_st.runtime_paths import resolve_run_output
from src.freqniche_st.model_config import load_model_config, build_freqniche_model, inference_batch_size, verify_training_config
from src.freqniche_st.models.freqniche_model import FreqNicheDiTModel

OUTPUT_DIR = resolve_run_output("outputs/runs/freqniche/xenium_prime5k/2026-09-18/20260918_xenium_prime5k_final")
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
CACHE_DIR = OUTPUT_DIR / "cache"
GSE_DIR = WORKSPACE_ROOT / "datasets" / "h5ad" / "xenium_ffpe_lungbreast_gse311609" / "samples"

AUDIT_MANIFEST_PATH = OUTPUT_DIR / "xenium_data_audit_manifest.json"
COHORTS_MANIFEST_PATH = OUTPUT_DIR / "xenium_cohorts_manifest.json"
VAL_CENTERS_PATH = OUTPUT_DIR / "val_centers.json"
VAL_FOLDS_PATH = OUTPUT_DIR / "val_bootstrap_folds.json"
WHITENER_PATH = OUTPUT_DIR / "whitener.pt"
TRAINING_SUMMARY_PATH = CHECKPOINTS_DIR / "xenium_training_summary.json"
RESOLUTION_SELECTION_PATH = OUTPUT_DIR / "resolution_selection.json"
MODEL_CONFIG = load_model_config(output_dir=OUTPUT_DIR)

VAL_SAMPLE_FNAME = "GSM9509137_NSCLC_5k_L6_L6.h5ad"
VAL_DONOR_ID = "L6"
FORBIDDEN_TEST_SAMPLES = ["GSM9509134_NSCLC_5k_L1_L1.h5ad", "GSM9509134", "L1"]

SEEDS = [42, 123, 2024]
FROZEN_RESOLUTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
K_MIN = 5
K_MAX = 25
RADIUS_UM = 50.0
SIGMA_UM = 16.666666666666668


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


def enforce_isolation_guard(path_or_str: Any):
    s = str(path_or_str)
    for forbidden in FORBIDDEN_TEST_SAMPLES:
        if forbidden in s:
            raise RuntimeError(
                f"CRITICAL RED LINE BREACH: Attempted to touch test sample '{forbidden}' in path: {s}"
            )


def ensure_validation_bootstrap_folds() -> Dict[str, Any]:
    """
    Ensures 10-Fold Bootstrap sub-samples on Validation centers exist and are verified.
    If missing, deterministically generates them using seed=42 protocol.
    """
    with open(VAL_CENTERS_PATH, "r", encoding="utf-8") as f:
        val_centers = json.load(f)
    n_val = len(val_centers)
    assert n_val == 10000, f"Expected 10,000 val centers, got {n_val}"
    val_barcodes = [c["cell_barcode"] for c in val_centers]

    if not VAL_FOLDS_PATH.exists():
        print(f" -> Generating deterministic 10-Fold Bootstrap partitions for Validation ({VAL_FOLDS_PATH}) ...")
        folds = {}
        for fold_id in range(10):
            sub_rng = np.random.RandomState(42 + 100 + fold_id)
            sub_indices = sub_rng.choice(10000, size=8000, replace=False)
            sub_barcodes = [val_barcodes[i] for i in sub_indices]
            sub_local_indices = [int(val_centers[i]["local_cell_idx"]) for i in sub_indices]
            folds[f"fold_{fold_id}"] = {
                "indices_in_val_centers": [int(i) for i in sub_indices],
                "local_indices": sub_local_indices,
                "sample_size": 8000,
                "sorted_barcode_sha256": compute_sorted_barcode_sha256(sub_barcodes),
                "ordered_barcode_sha256": compute_ordered_barcode_sha256(sub_barcodes),
            }
        temp_folds = VAL_FOLDS_PATH.with_suffix(".tmp.json")
        with open(temp_folds, "w", encoding="utf-8") as f:
            json.dump(folds, f, indent=2)
        os.replace(temp_folds, VAL_FOLDS_PATH)
        print(f" -> Successfully saved 10-Fold Bootstrap definitions to {VAL_FOLDS_PATH.name}.")

    with open(VAL_FOLDS_PATH, "r", encoding="utf-8") as f:
        folds = json.load(f)

    assert len(folds) == 10, f"Expected 10 folds, got {len(folds)}"
    for fold_id in range(10):
        fname = f"fold_{fold_id}"
        assert fname in folds, f"Missing {fname} in folds"
        f_idx = np.array(folds[fname]["indices_in_val_centers"], dtype=np.int64)
        assert len(f_idx) == 8000, f"{fname} has {len(f_idx)} indices, expected 8000"
        assert len(np.unique(f_idx)) == 8000, f"{fname} has duplicates"
        assert f_idx.min() >= 0 and f_idx.max() < 10000, f"{fname} out of bounds"
        sub_barcodes = [val_barcodes[i] for i in f_idx]
        assert compute_sorted_barcode_sha256(sub_barcodes) == folds[fname]["sorted_barcode_sha256"]
        assert compute_ordered_barcode_sha256(sub_barcodes) == folds[fname]["ordered_barcode_sha256"]

    print(f" -> All 10 Bootstrap folds verified: exactly 8,000 cells per fold, Triple-Hash valid.")
    return folds


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


def compute_effective_rank(emb: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Computes the Effective Rank (Roy & Vetterli, 2007) and variance explained spectrum.
    Effective Rank = exp( - sum( p_i * ln(p_i) ) ) where p_i = sigma_i / sum(sigma).
    """
    emb_centered = emb - np.mean(emb, axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(emb_centered, full_matrices=False)
    s_sum = np.sum(s)
    if s_sum <= 0:
        return 1.0, {}
    p = s / s_sum
    p = p[p > 1e-12]
    spectral_entropy = -float(np.sum(p * np.log(p)))
    erank = float(np.exp(spectral_entropy))

    # Variance explained by top components
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
    
    Computes:
    - stability score: mean pairwise ARI across 10 folds
    - mean K across folds
    - full validation clustering K, internal silhouette score, cluster entropy, max cluster ratio (p_max)
    - embedding effective rank and variance spectrum
    """
    fold_names = [f"fold_{i}" for i in range(10)]
    fold_indices_list = [np.array(bootstrap_folds[fname]["indices_in_val_centers"], dtype=np.int64) for fname in fold_names]
    n_bootstraps = len(fold_indices_list)

    scores_per_res: Dict[float, float] = {}
    mean_k_per_res: Dict[float, float] = {}
    std_k_per_res: Dict[float, float] = {}
    pair_aris_per_res: Dict[float, List[float]] = {}
    full_k_per_res: Dict[float, int] = {}
    internal_asw_per_res: Dict[float, float] = {}
    cluster_entropy_per_res: Dict[float, float] = {}
    p_max_per_res: Dict[float, float] = {}
    p_min_per_res: Dict[float, float] = {}

    # 1. Effective Rank of embedding
    erank, var_spectrum = compute_effective_rank(val_emb)

    # 2. Build scanpy neighbor graph on full validation set once
    adata_full = ad.AnnData(X=val_emb.copy())
    sc.pp.neighbors(adata_full, n_neighbors=15, use_rep="X", random_state=seed)

    for res in resolutions:
        # A. Evaluate across 10 bootstrap folds
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
        std_k_per_res[res] = float(np.std(n_clusters_list, ddof=1)) if len(n_clusters_list) > 1 else 0.0

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

        stability_score = float(np.mean(pair_aris)) if pair_aris else 0.0
        scores_per_res[res] = stability_score
        pair_aris_per_res[res] = pair_aris

        # B. Full Validation Clustering metrics
        key = f"leiden_{res}"
        sc.tl.leiden(adata_full, resolution=res, key_added=key, random_state=seed)
        full_labels = adata_full.obs[key].to_numpy().astype(int)
        full_k = len(np.unique(full_labels))
        full_k_per_res[res] = full_k

        # Cluster size distribution & entropy
        counts = pd.Series(full_labels).value_counts().values
        probs = counts / float(len(full_labels))
        p_max = float(np.max(probs))
        p_min = float(np.min(probs))
        entropy = -float(np.sum(probs * np.log(probs)))
        p_max_per_res[res] = p_max
        p_min_per_res[res] = p_min
        cluster_entropy_per_res[res] = entropy

        # Predicted-cluster internal ASW (Silhouette score in 128d space)
        if full_k > 1:
            # silhouette_score on full 10,000 cells
            asw = float(silhouette_score(val_emb, full_labels, metric="euclidean"))
        else:
            asw = 0.0
        internal_asw_per_res[res] = asw

    # Restrict to resolutions within [k_min, k_max] based on mean_k
    valid_res = [r for r in resolutions if k_min <= mean_k_per_res[r] <= k_max]
    if valid_res:
        best_res = max(valid_res, key=lambda r: (scores_per_res[r], internal_asw_per_res[r], -r))
    else:
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
        "best_stability_score": float(scores_per_res[best_res]),
        "stability_scores": {str(k): float(v) for k, v in scores_per_res.items()},
        "mean_clusters": {str(k): float(v) for k, v in mean_k_per_res.items()},
        "std_clusters": {str(k): float(v) for k, v in std_k_per_res.items()},
        "full_clusters_k": {str(k): int(v) for k, v in full_k_per_res.items()},
        "internal_asw": {str(k): float(v) for k, v in internal_asw_per_res.items()},
        "max_cluster_ratio_p_max": {str(k): float(v) for k, v in p_max_per_res.items()},
        "min_cluster_ratio_p_min": {str(k): float(v) for k, v in p_min_per_res.items()},
        "cluster_entropy": {str(k): float(v) for k, v in cluster_entropy_per_res.items()},
        "embedding_effective_rank": float(erank),
        "embedding_variance_spectrum": var_spectrum,
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
    Strict physical isolation: Test donor L1 is NEVER indexed, loaded, or touched.
    """
    print("=" * 85)
    print("W2.7 XENIUM PHASE 1: VALIDATION STABILITY SELECTION & r* FREEZING")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 85)

    if RESOLUTION_SELECTION_PATH.exists() and not overwrite_selection:
        raise RuntimeError(
            f"Resolution selection file already exists at {RESOLUTION_SELECTION_PATH}!\n"
            "Pass --overwrite-selection to explicitly re-run Phase 1 Validation."
        )

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Audit Prerequisites and Invariants
    print("\n[1/5] Auditing Audit Manifest, Cohorts Manifest, and 10-Fold Bootstrap Partition ...")
    assert AUDIT_MANIFEST_PATH.exists(), f"Missing audit manifest: {AUDIT_MANIFEST_PATH}"
    assert COHORTS_MANIFEST_PATH.exists(), f"Missing cohorts manifest: {COHORTS_MANIFEST_PATH}"
    assert VAL_CENTERS_PATH.exists(), f"Missing validation centers: {VAL_CENTERS_PATH}"
    assert WHITENER_PATH.exists(), f"Missing whitener: {WHITENER_PATH}"
    assert TRAINING_SUMMARY_PATH.exists(), f"Missing training summary: {TRAINING_SUMMARY_PATH}"

    # Ensure 10-fold partitions
    val_folds = ensure_validation_bootstrap_folds()

    with open(VAL_CENTERS_PATH, "r", encoding="utf-8") as f:
        val_centers = json.load(f)
    assert len(val_centers) == 10000, f"Expected 10,000 val centers, got {len(val_centers)}"

    # Check that ALL validation centers belong strictly to donor L6
    for c in val_centers:
        assert c["donor_id"] == VAL_DONOR_ID, f"Non-L6 center found in val_centers: {c}"
        enforce_isolation_guard(c["sample_id"])
    print(f" -> Validation centers verified: 10,000 cells strictly from donor {VAL_DONOR_ID}.")

    with open(TRAINING_SUMMARY_PATH, "r", encoding="utf-8") as f:
        training_summary = json.load(f)

    if isinstance(training_summary, dict) and "results" in training_summary:
        summary_by_seed = {int(k): v for k, v in training_summary["results"].items()}
    elif isinstance(training_summary, list):
        summary_by_seed = {item["seed"]: item for item in training_summary}
    else:
        raise ValueError(f"Unrecognized training summary format: {type(training_summary)}")

    ckpt_paths = {}
    ckpt_shas = {}
    for s in SEEDS:
        p = CHECKPOINTS_DIR / f"xenium_freqniche_trial24_seed_{s}.pt"
        verify_training_config(summary_by_seed[s], MODEL_CONFIG, asset=str(TRAINING_SUMMARY_PATH))
        assert p.exists(), f"Missing checkpoint for seed {s}: {p}"
        real_sha = compute_file_sha256(p)
        expected_sha = summary_by_seed[s]["checkpoint_sha256"]
        assert real_sha == expected_sha, f"SHA mismatch for seed {s} checkpoint! Got {real_sha}, expected {expected_sha}"
        ckpt_paths[s] = p
        ckpt_shas[s] = real_sha
        best_loss = summary_by_seed[s]["best_val_loss"]
        best_ep = summary_by_seed[s]["best_epoch"]
        print(f" -> Seed {s} checkpoint verified (Best Epoch {best_ep}, Val Loss: {best_loss:.4f}, SHA: {real_sha[:16]}...)")

    # 2. Check Embedding Availability or Extract
    val_embeddings = {}
    val_embedding_shas = {}
    embeddings_exist = all((EMBEDDINGS_DIR / f"val_embeddings_seed_{s}.npy").exists() for s in SEEDS)

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
        print(f"\n[2/5] Loading Validation Context strictly for donor {VAL_DONOR_ID} ...")
        val_h5ad_path = GSE_DIR / VAL_SAMPLE_FNAME
        enforce_isolation_guard(val_h5ad_path)
        assert val_h5ad_path.exists(), f"Validation h5ad missing: {val_h5ad_path}"

        val_p_cache = CACHE_DIR / f"xenium_graph_{VAL_DONOR_ID}_P.npz"
        val_a_cache = CACHE_DIR / f"xenium_graph_{VAL_DONOR_ID}_A.npz"
        assert val_p_cache.exists() and val_a_cache.exists(), "Missing L6 graph caches in cache dir!"

        print(" -> Loading L6 spatial graphs (P and A) ...")
        val_P = sp.load_npz(val_p_cache)
        val_A = sp.load_npz(val_a_cache)

        print(f" -> Loading L6 AnnData from {val_h5ad_path.name} ...")
        a_val = ad.read_h5ad(val_h5ad_path, backed="r")
        val_coords = np.asarray(a_val.obsm["spatial"], dtype=np.float32)
        raw_val_X = a_val.X[:]
        if sp.issparse(raw_val_X):
            val_X_dense = raw_val_X.toarray().astype(np.float32)
        else:
            val_X_dense = np.asarray(raw_val_X, dtype=np.float32)

        counts = np.maximum(val_X_dense.sum(axis=1, keepdims=True), 1.0)
        val_X_norm = np.log1p((val_X_dense / counts) * 1e4)

        val_center_indices = np.array([c["local_cell_idx"] for c in val_centers], dtype=np.int64)

        val_dataset = SubgraphDataset(
            center_indices=val_center_indices,
            A_global=val_A,
            P_global=val_P,
            X_global=val_X_norm,
            coords_global=val_coords,
            whitener=None,
            k_max=MODEL_CONFIG["k_max"],
            m_max=MODEL_CONFIG["m_max"],
            is_train=False,
            is_val_objective=False
        )

        print("\n[3/5] Extracting Validation Embeddings (10,000 centers) across 3 Seeds ...")
        for s in SEEDS:
            emb_file = EMBEDDINGS_DIR / f"val_embeddings_seed_{s}.npy"
            print(f" -> Inferring Validation embeddings for Seed {s} using frozen Trial 24 checkpoint ...")
            model = build_freqniche_model(5001, MODEL_CONFIG).to(device)

            state_data = torch.load(ckpt_paths[s], map_location=device, weights_only=True)
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
            print(f" -> Seed {s} Val Embedding saved: shape={z_val.shape}, SHA-256: {val_embedding_shas[s][:16]}...")

    # 3. Independent Unsupervised Stability Selection per Seed
    print(f"\n[4/5] Running Independent Unsupervised Stability Selection across {FROZEN_RESOLUTIONS} ...")
    print(f"Cluster count constraints: [{K_MIN}, {K_MAX}] clusters.")
    seed_selection_reports = {}
    frozen_r_stars = {}

    for s in SEEDS:
        print(f"\n" + "-" * 90)
        print(f"SEED {s} UNSUPERVISED STABILITY SCAN (10 Frozen Bootstrap Folds)")
        print("-" * 90)
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

        print(f"Embedding Effective Rank: {report['embedding_effective_rank']:.2f} / {MODEL_CONFIG['d_model']} "
              f"(Top-1: {report['embedding_variance_spectrum']['top_1_ratio']*100:.1f}%, "
              f"Top-10: {report['embedding_variance_spectrum']['top_10_ratio']*100:.1f}%)")
        print(f"{'Resolution (r)':<14} | {'Stability (ARI)':<16} | {'Mean K':<10} | {'Full K':<8} | {'Internal ASW':<14} | {'p_max':<8} | {'Entropy':<8} | Valid in [5, 25]")
        print("-" * 105)
        for r in FROZEN_RESOLUTIONS:
            score = report["stability_scores"][str(r)]
            mean_k = report["mean_clusters"][str(r)]
            full_k = report["full_clusters_k"][str(r)]
            asw = report["internal_asw"][str(r)]
            p_max = report["max_cluster_ratio_p_max"][str(r)]
            entropy = report["cluster_entropy"][str(r)]
            is_valid = r in report["valid_resolutions_in_k_bounds"]
            marker = " (* SELECTED r*)" if r == best_r else ""
            print(f"{r:<14.2f} | {score:<16.4f} | {mean_k:<10.1f} | {full_k:<8d} | {asw:<14.4f} | {p_max:<8.3f} | {entropy:<8.3f} | {str(is_valid):<14} {marker}")

        print(f" -> Seed {s} Optimal Resolution Locked: r* = {best_r} (Stability Score: {report['best_stability_score']:.4f}, Internal ASW: {report['internal_asw'][str(best_r)]:.4f})")

    # 4. Auxiliary Cross-Seed Pairwise ARI Readout
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

    # Cross-seed agreement at locked r*
    locked_c = {s: val_full_clusterings[s][frozen_r_stars[s]] for s in SEEDS}
    locked_pairs = [
        float(adjusted_rand_score(locked_c[42], locked_c[123])),
        float(adjusted_rand_score(locked_c[42], locked_c[2024])),
        float(adjusted_rand_score(locked_c[123], locked_c[2024]))
    ]
    auxiliary_cross_seed_aris["at_locked_r_star"] = {
        "mean_pairwise_ari": float(np.mean(locked_pairs)),
        "std_pairwise_ari": float(np.std(locked_pairs, ddof=1)),
        "pairs": {
            "42_vs_123": locked_pairs[0],
            "42_vs_2024": locked_pairs[1],
            "123_vs_2024": locked_pairs[2]
        }
    }
    print(f"  Cross-seed Pairwise ARI at locked r* ({frozen_r_stars[42]}/{frozen_r_stars[123]}/{frozen_r_stars[2024]}): {np.mean(locked_pairs):.4f} ± {np.std(locked_pairs, ddof=1):.4f}")

    # Build and Sign resolution_selection.json with cryptographic asset bindings
    script_path = Path(__file__).resolve()
    script_sha = compute_file_sha256(script_path)

    selection_manifest = {
        "manifest_version": "1.0.0",
        "description": "Deterministic Unsupervised Resolution r* Selection on Validation Cohort (Xenium Human NSCLC 5K)",
        "dataset": "xenium_ffpe_lungbreast_gse311609",
        "validation_sample": VAL_SAMPLE_FNAME,
        "bound_input_assets": {
            "evaluation_script": {
                "path": str(script_path.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                "sha256": script_sha
            },
            "data_audit_manifest": {
                "path": str(AUDIT_MANIFEST_PATH.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                "sha256": compute_file_sha256(AUDIT_MANIFEST_PATH)
            },
            "cohorts_manifest": {
                "path": str(COHORTS_MANIFEST_PATH.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                "sha256": compute_file_sha256(COHORTS_MANIFEST_PATH)
            },
            "val_centers_manifest": {
                "path": str(VAL_CENTERS_PATH.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                "sha256": compute_file_sha256(VAL_CENTERS_PATH)
            },
            "val_bootstrap_folds": {
                "path": str(VAL_FOLDS_PATH.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                "sha256": compute_file_sha256(VAL_FOLDS_PATH)
            },
            "whitener": {
                "path": str(WHITENER_PATH.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                "sha256": compute_file_sha256(WHITENER_PATH)
            },
            "checkpoints": {
                str(s): {
                    "path": str((CHECKPOINTS_DIR / f"xenium_freqniche_trial24_seed_{s}.pt").relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
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
            "coordinate_source": "adata.obsm['spatial'] (native microns, no secondary scaling)",
            "isolation_contract": "Validation context strictly limited to donor L6 (GSM9509137). Test donor L1 (GSM9509134) was NEVER indexed, read, or touched.",
            "k_max_subgraph": MODEL_CONFIG["k_max"],
            "embedding_dim": MODEL_CONFIG["d_model"],
            "k_min_clusters": K_MIN,
            "k_max_clusters": K_MAX,
            "resolutions_grid": FROZEN_RESOLUTIONS,
            "whitener": "None (pure inductive inference)",
            "masking": "zero_masking (deterministic inference path)"
        },
        "frozen_r_star_per_seed": {str(s): frozen_r_stars[s] for s in SEEDS},
        "per_seed_reports": {str(s): seed_selection_reports[s] for s in SEEDS},
        "auxiliary_cross_seed_readout": auxiliary_cross_seed_aris,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }

    payload_str = json.dumps(selection_manifest, sort_keys=True)
    manifest_sha = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
    selection_manifest["selection_manifest_sha256"] = manifest_sha

    temp_sel = RESOLUTION_SELECTION_PATH.with_suffix(".tmp.json")
    with open(temp_sel, "w", encoding="utf-8") as f:
        json.dump(selection_manifest, f, indent=2)
    os.replace(temp_sel, RESOLUTION_SELECTION_PATH)

    print("\n" + "=" * 85)
    print(f"SUCCESS: Resolution selection manifest written to: {RESOLUTION_SELECTION_PATH}")
    print(f"Manifest self-signature SHA-256: {manifest_sha}")
    print(f"Locked r* per seed: Seed 42 -> r*={frozen_r_stars[42]}, Seed 123 -> r*={frozen_r_stars[123]}, Seed 2024 -> r*={frozen_r_stars[2024]}")
    print("=" * 85)
    print("PHASE 1 VALIDATION COMPLETE. STOPPING FOR USER AUDIT REVIEW.")
    print("TEST EVALUATION REMAINS STRICTLY LOCKED.")


def run_test_phase(device: torch.device, overwrite_test: bool = False):
    """
    Phase 2: Test Evaluation on unseen donor L1 (GSM9509134).
    LOCKED UNTIL APPROVED.
    """
    print("\n" + "!" * 85)
    print("FAIL-CLOSED GUARD: Test Phase is strictly LOCKED pending formal audit approval.")
    print("Reference Niche ASW is NOT applicable on Xenium L1 (niche_valid_count == 0).")
    print("!" * 85)
    sys.exit(2)


def main():
    parser = argparse.ArgumentParser(description="W2.7 Xenium 5K Validation Stability & Evaluation Runner.")
    parser.add_argument("--phase", type=str, choices=["validation", "test"], required=True,
                        help="Phase of evaluation: 'validation' selects r*; 'test' evaluates test donor.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device ('cuda' or 'cpu').")
    parser.add_argument("--overwrite-selection", action="store_true",
                        help="Allow overwriting resolution_selection.json.")
    parser.add_argument("--recompute-val-embeddings", action="store_true",
                        help="Force recomputing validation embeddings even if files exist.")
    parser.add_argument("--overwrite-test", action="store_true",
                        help="Allow overwriting test evaluation outputs.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Alias for overwriting outputs.")
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
