"""Canonical FreqNiche Human Skin training backend.

The default/final route trains the frozen Trial-24 architecture on the signed
68,940-center Train contract and evaluates model selection on the signed 8,000
Validation centers. Test donors are excluded from the graph, whitener, training
and Validation tensors. Existing checkpoints are verified before reuse; silent
overwrite and silent cold restart are prohibited.

Legacy Top-3 verification helpers remain in this module only so historical
manifests can still be audited. The public orchestrator always invokes
``--final-model`` and writes to a new immutable dated run directory.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import sys
import time
import json
import random
import hashlib
import argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from freqniche_st.graph import build_global_spatial_graph
from freqniche_st.dataset import SubgraphDataset, collate_subgraphs
from freqniche_st.data_cache import GraphDataCache
from freqniche_st.data_loading import stage_batch, iter_batches
from freqniche_st.models import FreqNicheDiTModel
from freqniche_st.model_config import load_model_config
from freqniche_st.whitening import TrainOnlyWhitening
from freqniche_st.validation_worker import isolated_validation
from sklearn.metrics import adjusted_rand_score


# -------------------------------------------------------------------------
# Cryptographic and Provenance Constants
# -------------------------------------------------------------------------
EXPECTED_TOP3_SELF_SIG = "07954463ea0a688ef6742b8cc6a349ff64c3da7f3f2e2a899b45180458d398cd"
EXPECTED_MANIFEST_SELF_SIG = "0696475d548e6a4fad1118ad1096e9e79cd476215d989109f46536dc150f2d77"
EXPECTED_TRAIN_ORDERED_HASH = "a117f218539d88e84e5e8e64ba1b916a105c1fd99aa237d3c8e6bd6fd23cf31a"
W23_VALID_TRIALS = [10, 24, 20]
W23_VALID_SEEDS = [42, 123, 2024]
TEST_DONORS = {"D028", "D082", "D145", "D151"}
EXPECTED_VAL_DONORS = {"D004", "D077", "D083", "D107"}

DEFAULT_ADATA_PATH = "datasets/h5ad/human_skin_merfish/human_skin_merfish.h5ad"
DEFAULT_SPLITS_PATH = "datasets/h5ad/human_skin_merfish/donor_splits.csv"
DEFAULT_MANIFEST_PATH = "outputs/contracts/human_skin/train_centers_68940.json"
DEFAULT_VAL_BARCODES_PATH = "outputs/contracts/human_skin/validation_barcodes_8k.csv"
DEFAULT_TOP3_PATH = "outputs/experiments/human_skin/model_selection/top3_selection.json"
FINAL_TRIAL_24_CONFIG = {
    "trial_id": 24,
    "radius": 50.0,
    "k_max": 64,
    "rho": 0.5,
    "p_center_mask": 0.3,
    "d_model": 128,
    "n_enc": 3,
    "n_dit": 2,
    "n_heads": 8,
    "lr": 0.001,
    "wd": 0.001,
    "timesteps": 50,
    "lambda_cov": 0.04,
    "desc": "Deep-Enc + Strong-Reg + T=50",
    "m_max": 32,
}

EXPECTED_INPUT_HASHES = {
    DEFAULT_SPLITS_PATH: "5fa1e567b8482ae83336e3ce3dd403f08a8af01f8a9588c9e730bc8ee8da0803",
    DEFAULT_MANIFEST_PATH: "2c5e44c3aeab60df7643b40df05b34c7c08509d384574c085cdaf6982eb7609d",
    DEFAULT_VAL_BARCODES_PATH: "b37d6c6a4e2240eb5082be739ca2347f2d55fa6afbe7d36284b03d73ace051e9",
    DEFAULT_TOP3_PATH: "f7775e1995d55d55ea28b7563a67d6da9c14c7ebf0010d70a439596b2308b86b"
}


def log(msg: str):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t_str}] {msg}", flush=True)


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_expression_matrix_hash(X: np.ndarray, chunk_size: int = 50000) -> str:
    h = hashlib.sha256()
    n_rows = X.shape[0]
    for i in range(0, n_rows, chunk_size):
        chunk = np.ascontiguousarray(X[i:i + chunk_size], dtype=np.float32)
        h.update(chunk.tobytes())
    return h.hexdigest()[:24]


def get_graph_fingerprint(
    coords: np.ndarray,
    sample_ids: np.ndarray,
    gene_names: List[str],
    radius: float,
    sigma: float
) -> str:
    graph_py_path = os.path.join(os.path.dirname(__file__), "graph.py")
    graph_code_hash = compute_file_sha256(graph_py_path)

    h = hashlib.sha256()
    h.update(b"graph_contract_v1.2_P_A_order")
    h.update(np.ascontiguousarray(coords, dtype=np.float32).tobytes())
    h.update(",".join(map(str, sample_ids)).encode("utf-8"))
    h.update(",".join(gene_names).encode("utf-8"))
    h.update(f"r_{radius:.4f}_s_{sigma:.4f}_selfloop1.0".encode("utf-8"))
    h.update(graph_code_hash.encode("utf-8"))
    return h.hexdigest()[:24]


def get_whitener_fingerprint(
    graph_fingerprint: str,
    expression_data_hash: str,
    train_manifest_hash: str,
    k_target: int = 32
) -> str:
    whitening_py_path = os.path.join(os.path.dirname(__file__), "whitening.py")
    whitening_code_hash = compute_file_sha256(whitening_py_path)

    h = hashlib.sha256()
    h.update(b"whitener_contract_v1.2")
    h.update(graph_fingerprint.encode("utf-8"))
    h.update(expression_data_hash.encode("utf-8"))
    h.update(train_manifest_hash.encode("utf-8"))
    h.update(f"norm1e4_log1p_ktarget_{k_target}".encode("utf-8"))
    h.update(whitening_code_hash.encode("utf-8"))
    return h.hexdigest()[:24]


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_rng_states(dl_generator: torch.Generator, dataset_rng: random.Random) -> Dict[str, Any]:
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "dataloader_generator_state": dl_generator.get_state(),
        "dataset_rng_state": dataset_rng.get_state()
    }


def restore_rng_states(state_dict: Dict[str, Any], dl_generator: torch.Generator, dataset_rng: Optional[random.Random] = None):
    random.setstate(state_dict["python_rng_state"])
    np.random.set_state(state_dict["numpy_rng_state"])

    cpu_state = state_dict["torch_cpu_rng_state"]
    if isinstance(cpu_state, torch.Tensor):
        cpu_state = cpu_state.cpu().to(torch.uint8)
    torch.set_rng_state(cpu_state)

    if torch.cuda.is_available() and state_dict.get("torch_cuda_rng_state") is not None:
        cuda_states = state_dict["torch_cuda_rng_state"]
        if isinstance(cuda_states, list):
            cuda_states = [s.cpu().to(torch.uint8) if isinstance(s, torch.Tensor) else s for s in cuda_states]
        elif isinstance(cuda_states, torch.Tensor):
            cuda_states = cuda_states.cpu().to(torch.uint8)
        torch.cuda.set_rng_state_all(cuda_states)

    dl_gen_state = state_dict["dataloader_generator_state"]
    if isinstance(dl_gen_state, torch.Tensor):
        dl_gen_state = dl_gen_state.cpu().to(torch.uint8)
    dl_generator.set_state(dl_gen_state)

    if dataset_rng is not None and "dataset_rng_state" in state_dict:
        dataset_rng.set_state(state_dict["dataset_rng_state"])


# -------------------------------------------------------------------------
# Manifest & Top-3 Verification Functions
# -------------------------------------------------------------------------
def verify_top3_selection_manifest(top3_path: str) -> Dict[int, Dict[str, Any]]:
    assert os.path.exists(top3_path), f"Missing Top-3 selection file: {top3_path}"
    with open(top3_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    # 1. Self-signature verification
    d_copy = dict(d)
    sig_in_file = d_copy.pop("sha256", None)
    calc_sig = hashlib.sha256(json.dumps(d_copy, indent=2).encode("utf-8")).hexdigest()
    assert sig_in_file == EXPECTED_TOP3_SELF_SIG, f"Top-3 signature in file {sig_in_file} != expected {EXPECTED_TOP3_SELF_SIG}"
    assert calc_sig == EXPECTED_TOP3_SELF_SIG, f"Calculated Top-3 signature {calc_sig} != expected {EXPECTED_TOP3_SELF_SIG}"

    # 2. Automated re-verification of Top-8 source selection file on disk
    top8_source_path = "outputs/experiments/human_skin/model_selection/top8_selection.json"
    assert os.path.exists(top8_source_path), f"Missing Top-8 selection source file: {top8_source_path}"
    actual_top8_hash = compute_file_sha256(top8_source_path)
    expected_top8_hash = d.get("top8_source_selection_hash")
    assert actual_top8_hash == expected_top8_hash, (
        f"Top-8 source selection hash mismatch: {actual_top8_hash} != {expected_top8_hash}"
    )

    # 3. Automated re-verification of 8 Stage-2 eval files on disk
    trial_eval_hashes = d.get("trial_eval_hashes", {})
    assert len(trial_eval_hashes) == 8, f"Expected 8 trial eval hashes in Top-3 manifest, found {len(trial_eval_hashes)}"
    for tid_str, expected_eval_hash in trial_eval_hashes.items():
        tid_int = int(tid_str)
        eval_path = f"outputs/experiments/human_skin/model_selection/trial_{tid_int:02d}/eval_result_ep30.json"
        assert os.path.exists(eval_path), f"Missing Stage 2 eval file for trial {tid_int}: {eval_path}"
        actual_eval_hash = compute_file_sha256(eval_path)
        assert actual_eval_hash == expected_eval_hash, (
            f"Stage 2 trial {tid_int} eval hash mismatch: {actual_eval_hash} != {expected_eval_hash}"
        )

    assert d["top3_trial_ids"] == [10, 24, 20], f"Unexpected Top-3 trial IDs: {d['top3_trial_ids']}"

    candidates = {}
    for item in d["top3_rankings"]:
        tid = item["trial_id"]
        cfg = item["config"]
        assert cfg["radius"] == 50.0, f"Trial {tid} radius must be 50.0"
        assert cfg["k_max"] == 64, f"Trial {tid} k_max must be 64"
        assert cfg["rho"] == 0.5, f"Trial {tid} rho must be 0.5"
        assert cfg["p_center_mask"] == 0.3, f"Trial {tid} p_center_mask must be 0.3"
        assert cfg["lambda_cov"] == 0.04, f"Trial {tid} lambda_cov must be 0.04"
        assert cfg.get("m_max", 32) == 32, f"Trial {tid} m_max must be 32"
        candidates[tid] = cfg

    log(f"[Verification PASS] Top-3 selection verified (self-sig: {EXPECTED_TOP3_SELF_SIG[:16]}, Top-8 source & 8 eval files verified on disk)")
    return candidates


def verify_train_manifest(
    manifest_path: str,
    adata_all: ad.AnnData,
    splits_df: Optional[pd.DataFrame] = None,
    verify_legacy_source: bool = True,
    require_legacy_signature: bool = True,
) -> Tuple[List[str], str]:
    assert os.path.exists(manifest_path), f"Missing 68k manifest: {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    # 1. Self-signature verification
    d_copy = dict(d)
    sig_in_file = d_copy.pop("sha256", None)
    calc_sig = hashlib.sha256(json.dumps(d_copy, indent=2).encode("utf-8")).hexdigest()
    assert sig_in_file == calc_sig, f"Manifest self-signature mismatch: {sig_in_file} != {calc_sig}"
    if require_legacy_signature:
        assert sig_in_file == EXPECTED_MANIFEST_SELF_SIG, (
            f"Legacy manifest sig {sig_in_file} != expected {EXPECTED_MANIFEST_SELF_SIG}"
        )

    # 2. Verify source CSV exists and matches hash
    source_csv = d.get("source_csv")
    if verify_legacy_source:
        assert source_csv is not None and os.path.exists(source_csv), f"Manifest source_csv missing: {source_csv}"
        actual_csv_hash = compute_file_sha256(source_csv)
        assert actual_csv_hash == d.get("source_csv_sha256"), (
            f"Manifest source_csv SHA mismatch: {actual_csv_hash} != {d.get('source_csv_sha256')}"
        )

    barcodes = d["barcodes"]
    assert len(barcodes) == 68940, f"Expected 68,940 barcodes, found {len(barcodes)}"
    assert len(set(barcodes)) == 68940, "Barcodes contain duplicates!"
    if not require_legacy_signature:
        barcode_hash = hashlib.sha256("".join(f"{barcode}\n" for barcode in barcodes).encode("utf-8")).hexdigest()
        assert barcode_hash == EXPECTED_TRAIN_ORDERED_HASH, "Train barcode order differs from frozen contract"
        if "ordered_barcodes_sha256" in d:
            assert d["ordered_barcodes_sha256"] == barcode_hash, "Manifest ordered barcode hash mismatch"

    manifest_donor_counts = d.get("donor_counts", {})
    assert sum(manifest_donor_counts.values()) == 68940, (
        f"Manifest donor_counts sum {sum(manifest_donor_counts.values())} != 68940"
    )

    # 3. If splits provided, verify exactly 14 train donors
    all_obs = adata_all.obs
    donor_col = "donor_id" if "donor_id" in all_obs.columns else "donor"
    if splits_df is not None:
        split_donor_col = "donor_id" if "donor_id" in splits_df.columns else "donor"
        train_donors_expected = set(splits_df[splits_df["split"] == "train"][split_donor_col])
        assert set(manifest_donor_counts.keys()) == train_donors_expected, (
            f"Manifest donors {set(manifest_donor_counts.keys())} != expected 14 train donors {train_donors_expected}"
        )

    # 4. Verify donor counts match AnnData counts exactly
    obs_donor_counts = {k: v for k, v in all_obs.loc[barcodes, donor_col].value_counts().to_dict().items() if v > 0}
    for d_id, expected_cnt in manifest_donor_counts.items():
        actual_cnt = obs_donor_counts.get(d_id, 0)
        assert actual_cnt == expected_cnt, (
            f"Donor {d_id} count mismatch: AnnData has {actual_cnt}, manifest records {expected_cnt}"
        )

    # 5. Ensure none are test donors
    manifest_donors = set(obs_donor_counts.keys())
    leaked_test = manifest_donors.intersection(TEST_DONORS)
    assert len(leaked_test) == 0, f"CRITICAL LEAK: Test donors found in train manifest: {leaked_test}"

    log(f"[Verification PASS] 68,940 train manifest verified (14 train donors, 0 test leakage, self-sig: {sig_in_file[:16]})")
    return barcodes, sig_in_file


# -------------------------------------------------------------------------
# Graph & Whitener Build / Cache Functions
# -------------------------------------------------------------------------
def get_or_build_graph(
    coords: np.ndarray,
    sample_ids: np.ndarray,
    gene_names: List[str],
    radius: float,
    cache_dir: str = "outputs/cache/graphs"
) -> Tuple[sp.csr_matrix, sp.csr_matrix, str]:
    os.makedirs(cache_dir, exist_ok=True)
    sigma = radius / 3.0
    fp = get_graph_fingerprint(coords, sample_ids, gene_names, radius, sigma)
    cache_path = os.path.join(cache_dir, f"graph_{fp}.npz")

    if os.path.exists(cache_path):
        log(f"  [Graph Cache] Loading verified graph for R={radius}um (FP: {fp})...")
        loader = np.load(cache_path)
        A = sp.csr_matrix((loader["A_data"], loader["A_indices"], loader["A_indptr"]), shape=loader["A_shape"])
        P = sp.csr_matrix((loader["P_data"], loader["P_indices"], loader["P_indptr"]), shape=loader["P_shape"])

        assert (A != A.T).nnz == 0 or np.max(np.abs(A - A.T)) == 0, "Cached A is not symmetric!"
        assert np.allclose(np.asarray(P.sum(axis=1)).ravel(), 1.0, atol=1e-4), "Cached P is not row-stochastic!"
        assert np.all(A.diagonal() == 1.0), "Cached A diagonal must be 1.0!"
        return P, A, fp

    log(f"  [Graph Cache] Building graph for R={radius}um (FP: {fp})...")
    t0 = time.time()
    P, A = build_global_spatial_graph(coords=coords, sample_ids=sample_ids, radius=radius, sigma=sigma)

    log("  [Graph Cache] Verifying mathematical properties of (P, A)...")
    symm_diff = np.max(np.abs(A - A.T))
    assert symm_diff == 0.0 or (A != A.T).nnz == 0, f"A is not symmetric! max diff: {symm_diff}"
    row_sum_diff = np.max(np.abs(np.asarray(P.sum(axis=1)).ravel() - 1.0))
    assert row_sum_diff < 1e-4, f"P is not row-stochastic! max diff: {row_sum_diff}"
    assert np.all(A.diagonal() == 1.0), "A diagonal must be 1.0 (self-loops)!"
    log(f"  [Graph Cache] Verified: A symmetric, P row-stochastic, self-loops=1.0 (nnz={A.nnz}) in {time.time()-t0:.2f}s.")

    tmp_path = os.path.join(cache_dir, f"tmp_{os.getpid()}_{int(time.time()*1000)}_{fp}.npz")
    np.savez_compressed(
        tmp_path,
        A_data=A.data, A_indices=A.indices, A_indptr=A.indptr, A_shape=A.shape,
        P_data=P.data, P_indices=P.indices, P_indptr=P.indptr, P_shape=P.shape
    )
    os.replace(tmp_path, cache_path)
    log(f"  [Graph Cache] Atomically persisted to {cache_path}.")
    return P, A, fp


def get_or_build_whitener(
    X_global: np.ndarray,
    P_global: sp.csr_matrix,
    train_center_indices: np.ndarray,
    graph_fingerprint: str,
    expression_data_hash: str,
    train_manifest_hash: str,
    k_target: int = 32,
    cache_dir: str = "outputs/cache/whiteners"
) -> Tuple[TrainOnlyWhitening, str]:
    os.makedirs(cache_dir, exist_ok=True)
    fp = get_whitener_fingerprint(graph_fingerprint, expression_data_hash, train_manifest_hash, k_target)
    cache_path = os.path.join(cache_dir, f"whitener_{fp}.pt")

    whitener = TrainOnlyWhitening(k_components=k_target)
    if os.path.exists(cache_path):
        log(f"  [Whitener Cache] Loading verified whitener (FP: {fp})...")
        state = torch.load(cache_path, map_location="cpu", weights_only=False)
        whitener.mean_ = state["mean_"]
        whitener.components_ = state["components_"]
        whitener.singular_values_ = state["singular_values_"]
        whitener.whitening_matrix_ = state["whitening_matrix_"]
        whitener.n_components_fitted_ = state["n_components_fitted_"]
        whitener.is_fitted = True
        assert whitener.n_components_fitted_ == k_target, f"Fitted components {whitener.n_components_fitted_} != {k_target}"
        return whitener, fp

    log(f"  [Whitener Cache] Fitting residual whitener on {len(train_center_indices)} train centers (FP: {fp})...")
    t0 = time.time()
    PX_train = P_global[train_center_indices] @ X_global
    H_train = X_global[train_center_indices] - PX_train
    whitener.fit(H_train)

    tmp_path = os.path.join(cache_dir, f"tmp_{os.getpid()}_{int(time.time()*1000)}_{fp}.pt")
    torch.save({
        "mean_": whitener.mean_,
        "components_": whitener.components_,
        "singular_values_": whitener.singular_values_,
        "whitening_matrix_": whitener.whitening_matrix_,
        "n_components_fitted_": whitener.n_components_fitted_
    }, tmp_path)
    os.replace(tmp_path, cache_path)
    log(f"  [Whitener Cache] Fit in {time.time()-t0:.2f}s and atomically persisted to {cache_path}.")
    return whitener, fp


# -------------------------------------------------------------------------
# Core W2.3 Single Run Training Engine
# -------------------------------------------------------------------------
def train_single_run(
    trial_id: int,
    seed: int,
    trial_cfg: Dict[str, Any],
    train_center_indices: np.ndarray,
    val_center_indices: np.ndarray,
    val_donor_labels: np.ndarray,
    adata_train_val: ad.AnnData,
    P_global: sp.csr_matrix,
    A_global: sp.csr_matrix,
    X_global: np.ndarray,
    coords_global: np.ndarray,
    whitener: TrainOnlyWhitening,
    graph_fp: str,
    whitener_fp: str,
    manifest_hash: str,
    output_base: str,
    device: torch.device,
    max_epochs: int = 50,
    patience: int = 7,
    min_delta: float = 1e-4,
    batch_size: int = 64,
    restart_run: bool = False,
    mock_val_losses: Optional[List[float]] = None,
    pause_after_epoch: Optional[int] = None,
    selection_hash: Optional[str] = EXPECTED_TOP3_SELF_SIG,
) -> Dict[str, Any]:
    run_dir = os.path.join(output_base, f"trial_{trial_id}_seed_{seed}")
    result_json_path = os.path.join(run_dir, "eval_result.json")
    best_val_model_path = os.path.join(run_dir, "best_val_model.pt")
    resume_state_path = os.path.join(run_dir, "resume_state_last.pt")
    emb_path = os.path.join(run_dir, "val_embeddings.npy")

    # 1. Restart Handling: Archive existing corrupted/prior directory safely
    if restart_run and os.path.exists(run_dir):
        archive_dir = os.path.join(output_base, f"trial_{trial_id}_seed_{seed}_archived_{int(time.time())}")
        log(f"[RESTART RUN] Archiving existing run directory from {run_dir} to {archive_dir}...")
        os.rename(run_dir, archive_dir)

    os.makedirs(run_dir, exist_ok=True)

    # 2. Strict Verification of Completed Results (Skip Logic)
    if os.path.exists(result_json_path):
        log(f"Found completed result file at {result_json_path}, verifying compliance against frozen contracts...")
        try:
            with open(result_json_path, "r", encoding="utf-8") as f:
                prev_res = json.load(f)

            assert prev_res.get("trial_id") == trial_id, f"trial_id mismatch: {prev_res.get('trial_id')} != {trial_id}"
            assert prev_res.get("seed") == seed, f"seed mismatch: {prev_res.get('seed')} != {seed}"
            assert prev_res.get("config") == trial_cfg, "config mismatch against frozen Top-3 config"
            assert prev_res.get("train_manifest_hash") == manifest_hash, "manifest_hash mismatch"
            assert prev_res.get("top3_selection_hash") == selection_hash, "top3_selection_hash mismatch"
            assert prev_res.get("graph_fingerprint") == graph_fp, "graph_fingerprint mismatch"
            assert prev_res.get("whitener_fingerprint") == whitener_fp, "whitener_fingerprint mismatch"
            best_ep = prev_res.get("best_val_epoch", -1)
            stop_ep = prev_res.get("stopped_epoch", -1)
            assert 1 <= best_ep <= stop_ep <= max_epochs, f"Invalid epoch range: best={best_ep}, stopped={stop_ep}"

            donor_aris = prev_res.get("donor_aris", {})
            assert set(donor_aris.keys()) == EXPECTED_VAL_DONORS, f"Invalid donor_aris donors: {set(donor_aris.keys())}"
            for d, score in donor_aris.items():
                assert np.isfinite(score), f"Non-finite ARI for donor {d}: {score}"
            assert np.isfinite(prev_res.get("donor_macro_ari", np.nan)), "Non-finite donor_macro_ari"
            assert np.isfinite(prev_res.get("worst_donor_ari", np.nan)), "Non-finite worst_donor_ari"
            assert np.isfinite(prev_res.get("stability_score", np.nan)), "Non-finite stability_score"

            # Check model and embeddings files
            assert os.path.exists(best_val_model_path) and os.path.getsize(best_val_model_path) > 0, "best_val_model.pt missing or empty"
            assert os.path.exists(emb_path) and os.path.getsize(emb_path) > 0, "val_embeddings.npy missing or empty"
            z_val_loaded = np.load(emb_path)
            assert z_val_loaded.shape == (len(val_center_indices), trial_cfg["d_model"]), f"Embedding shape mismatch: {z_val_loaded.shape}"
            assert np.all(np.isfinite(z_val_loaded)), "Embedding contains non-finite values"

            # Check SHA-256 bindings for best_val_model, embeddings, and runner code
            assert "best_val_model_sha256" in prev_res, "best_val_model_sha256 missing in eval_result.json"
            actual_best_model_sha = compute_file_sha256(best_val_model_path)
            assert actual_best_model_sha == prev_res["best_val_model_sha256"], (
                f"best_val_model.pt SHA mismatch: {actual_best_model_sha} != {prev_res['best_val_model_sha256']}"
            )

            assert "val_embeddings_sha256" in prev_res, "val_embeddings_sha256 missing in eval_result.json"
            actual_emb_sha = compute_file_sha256(emb_path)
            assert actual_emb_sha == prev_res["val_embeddings_sha256"], (
                f"val_embeddings.npy SHA mismatch: {actual_emb_sha} != {prev_res['val_embeddings_sha256']}"
            )

            assert "runner_code_sha256" in prev_res, "runner_code_sha256 missing in eval_result.json"
            actual_runner_sha = compute_file_sha256(os.path.abspath(__file__))
            assert actual_runner_sha == prev_res["runner_code_sha256"], (
                f"runner_code_sha256 mismatch: {actual_runner_sha} != {prev_res['runner_code_sha256']}"
            )

            log(f"[SKIP PASS] Run trial_{trial_id}_seed_{seed} verified 100% compliant with all frozen contracts and artifact hashes. Skipping execution.")
            return prev_res
        except Exception as e:
            raise RuntimeError(
                f"Corrupted or invalid completed result found at {result_json_path}: {e}.\n"
                f"Silent overwrite or auto-restart is strictly forbidden to prevent contamination!\n"
                f"If you intend to discard and restart this run, specify --restart-run."
            )

    log(f"\n{'='*75}\n Starting W2.3 Run: Trial {trial_id} | Seed {seed} | {trial_cfg.get('desc')}\n{'='*75}")

    # Set exact seeds for cold-start
    set_all_seeds(seed)
    dl_generator = torch.Generator()
    dl_generator.manual_seed(seed)

    graph_cache = GraphDataCache(A=A_global, X=X_global, P=P_global, max_entries=30000)

    # 3. Prepare Datasets & Loaders
    train_dataset = SubgraphDataset(
        center_indices=train_center_indices,
        A_global=A_global,
        P_global=P_global,
        X_global=X_global,
        coords_global=coords_global,
        whitener=whitener,
        k_max=trial_cfg["k_max"],
        rho=trial_cfg["rho"],
        p_center_mask=trial_cfg["p_center_mask"],
        is_train=True,
        is_val_objective=False,
        m_max=trial_cfg.get("m_max", 32),
        seed=seed,
        data_cache=graph_cache
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_subgraphs,
        generator=dl_generator,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    val_dataset = SubgraphDataset(
        center_indices=val_center_indices,
        A_global=A_global,
        P_global=P_global,
        X_global=X_global,
        coords_global=coords_global,
        whitener=whitener,
        k_max=trial_cfg["k_max"],
        rho=trial_cfg["rho"],
        p_center_mask=trial_cfg["p_center_mask"],
        is_train=False,
        is_val_objective=True,
        m_max=trial_cfg.get("m_max", 32),
        seed=seed,
        data_cache=graph_cache
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_subgraphs,
        pin_memory=(device.type == "cuda")
    )

    # 4. Cold-Start Model & Optimizer
    in_features = X_global.shape[1]
    model = FreqNicheDiTModel(
        in_features=in_features,
        k_target=whitener.n_components_fitted_,
        d_model=trial_cfg["d_model"],
        n_enc_layers=trial_cfg["n_enc"],
        n_dit_layers=trial_cfg["n_dit"],
        n_heads=trial_cfg["n_heads"],
        timesteps=trial_cfg["timesteps"],
        mode="full_freq",
        lambda_var=1.0,
        lambda_cov=trial_cfg["lambda_cov"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trial_cfg["lr"],
        weight_decay=trial_cfg.get("weight_decay", trial_cfg.get("wd", 1e-4))
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # Isolated deterministic validation generator: seed + 99999
    val_generator = torch.Generator(device=device)

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_epoch = -1
    patience_counter = 0
    epoch_times = []
    actual_mask_ratios = []
    stopped_early = False
    final_epoch = 0

    # 5. Rigorous Resume Checkpoint Verification
    if os.path.exists(resume_state_path):
        log(f"Found existing resume state at {resume_state_path}, checking full integrity...")
        try:
            ckpt = torch.load(resume_state_path, map_location=device, weights_only=False)
            assert ckpt["trial_id"] == trial_id, f"Trial ID mismatch: {ckpt['trial_id']} != {trial_id}"
            assert ckpt["seed"] == seed, f"Seed mismatch: {ckpt['seed']} != {seed}"
            assert ckpt["trial_config"] == trial_cfg, "trial_config mismatch against frozen config"
            assert ckpt["graph_fingerprint"] == graph_fp, "graph_fingerprint mismatch in checkpoint"
            assert ckpt["whitener_fingerprint"] == whitener_fp, "whitener_fingerprint mismatch in checkpoint"
            assert ckpt["train_manifest_hash"] == manifest_hash, "train_manifest_hash mismatch in checkpoint"
            assert ckpt.get("top3_selection_hash") == selection_hash, "top3_selection_hash mismatch in checkpoint"
            assert ckpt.get("max_epochs") == max_epochs, f"max_epochs mismatch: {ckpt.get('max_epochs')} != {max_epochs}"
            assert ckpt.get("patience") == patience, f"patience mismatch: {ckpt.get('patience')} != {patience}"
            assert abs(ckpt.get("min_delta", 0.0) - min_delta) < 1e-9, f"min_delta mismatch: {ckpt.get('min_delta')} != {min_delta}"
            assert ckpt.get("batch_size") == batch_size, f"batch_size mismatch: {ckpt.get('batch_size')} != {batch_size}"
            assert ckpt.get("optimizer_type") == "AdamW", f"optimizer_type mismatch: {ckpt.get('optimizer_type')} != AdamW"
            assert abs(ckpt.get("gradient_clip_norm", 0.0) - 1.0) < 1e-9, f"gradient_clip_norm mismatch: {ckpt.get('gradient_clip_norm')} != 1.0"

            if "runner_code_sha256" in ckpt:
                actual_runner_sha = compute_file_sha256(os.path.abspath(__file__))
                assert ckpt["runner_code_sha256"] == actual_runner_sha, (
                    f"Runner code version mismatch on resume: {ckpt['runner_code_sha256']} != {actual_runner_sha}"
                )

            # Check best_val_model exists and matches recorded hash
            assert os.path.exists(best_val_model_path), f"best_val_model.pt missing for checkpoint at {best_val_model_path}"
            expected_best_sha = ckpt.get("best_val_model_sha256")
            if expected_best_sha is not None:
                actual_best_sha = compute_file_sha256(best_val_model_path)
                assert actual_best_sha == expected_best_sha, f"best_val_model.pt SHA mismatch: {actual_best_sha} != {expected_best_sha}"

            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt["best_val_loss"]
            best_val_epoch = ckpt["best_val_epoch"]
            patience_counter = ckpt["patience_counter"]
            epoch_times = ckpt.get("epoch_times", [])
            actual_mask_ratios = ckpt.get("actual_mask_ratios", [])
            restore_rng_states(ckpt, dl_generator, train_dataset.rng)
            log(f"Resume checkpoint successfully verified. Epoch: {ckpt['epoch']}, Best val loss: {best_val_loss:.4f} @ Ep {best_val_epoch}, Patience: {patience_counter}/{patience}")

            # Check if early stopping condition was already met
            if patience_counter >= patience:
                log(f"[RESUME EARLY STOPPING] Checkpoint at epoch {ckpt['epoch']} already met early stopping patience ({patience_counter}/{patience}). Directly proceeding to best model rollback!")
                stopped_early = True
                final_epoch = ckpt["epoch"]
        except Exception as e:
            raise RuntimeError(
                f"Corrupted or invalid resume checkpoint at {resume_state_path}: {e}.\n"
                f"Silent cold-start or auto-restart is strictly forbidden!\n"
                f"If you wish to discard this run and restart, re-run with --restart-run."
            )

    # 6. Training Loop with Early Stopping
    if not stopped_early:
        for epoch in range(start_epoch, max_epochs + 1):
            final_epoch = epoch
            t_ep_start = time.time()
            model.train()
            total_loss = 0.0
            total_diff = 0.0
            n_batches = 0
            batch_mask_ratios = []

            for batch in iter_batches(train_loader, prefetch=(device.type == "cuda")):
                batch = stage_batch(batch, device)
                for k in batch:
                    if isinstance(batch[k], torch.Tensor) and batch[k].device != device:
                        batch[k] = batch[k].to(device)

                optimizer.zero_grad()
                out = model(
                    X_vis=batch["X_vis"],
                    delta_s_vis=batch["delta_s_vis"],
                    vis_pad_mask=batch["vis_pad_mask"],
                    H_0_target=batch["H_0_target"],
                    delta_s_target=batch["delta_s_target"],
                    target_mask=batch["target_mask"],
                    L_vis_cond=batch["L_vis_cond"],
                    H_vis_cond=batch["H_vis_cond"]
                )

                loss = out["loss"]
                loss.backward()

                # Finite check
                is_finite = True
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        is_finite = False
                        break
                if not is_finite:
                    dump_path = os.path.join(run_dir, f"nonfinite_grad_ep{epoch}_b{n_batches}.pt")
                    torch.save({"batch": batch, "out": out, "model": model.state_dict()}, dump_path)
                    raise RuntimeError(f"Non-finite gradient in Trial {trial_id} Seed {seed} Ep {epoch}! Dumped to {dump_path}")

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_diff += out["loss_diff"].item()
                n_batches += 1

                target_counts = batch["target_mask"].sum(dim=1).float()
                vis_counts = batch["vis_pad_mask"].sum(dim=1).float()
                total_sub_counts = torch.clamp(target_counts + vis_counts, min=1.0)
                actual_ratio = (target_counts / total_sub_counts).mean().item()
                batch_mask_ratios.append(actual_ratio)

            ep_time = time.time() - t_ep_start
            epoch_times.append(ep_time)
            mean_actual_mask = float(np.mean(batch_mask_ratios))
            actual_mask_ratios.append(mean_actual_mask)

            # Validation Loss Evaluation
            if mock_val_losses is not None and len(mock_val_losses) >= epoch:
                mean_val_loss = float(mock_val_losses[epoch - 1])
            else:
                model.eval()
                val_loss = 0.0
                val_batches = 0
                val_generator.manual_seed(seed + 99999)  # Strict paired deterministic noise

                with torch.no_grad():
                    for batch in iter_batches(val_loader, prefetch=(device.type == "cuda")):
                        batch = stage_batch(batch, device)
                        for k in batch:
                            if isinstance(batch[k], torch.Tensor) and batch[k].device != device:
                                batch[k] = batch[k].to(device)
                        out = model(
                            X_vis=batch["X_vis"],
                            delta_s_vis=batch["delta_s_vis"],
                            vis_pad_mask=batch["vis_pad_mask"],
                            H_0_target=batch["H_0_target"],
                            delta_s_target=batch["delta_s_target"],
                            target_mask=batch["target_mask"],
                            L_vis_cond=batch["L_vis_cond"],
                            H_vis_cond=batch["H_vis_cond"],
                            generator=val_generator
                        )
                        val_loss += out["loss"].item()
                        val_batches += 1
                mean_val_loss = val_loss / max(1, val_batches)

            # Check for valid improvement
            is_best = False
            if mean_val_loss < (best_val_loss - min_delta):
                best_val_loss = mean_val_loss
                best_val_epoch = epoch
                patience_counter = 0
                is_best = True

                # Save best checkpoint atomically
                tmp_best = best_val_model_path + f".tmp_{os.getpid()}"
                torch.save(model.state_dict(), tmp_best)
                os.replace(tmp_best, best_val_model_path)
            else:
                patience_counter += 1

            best_marker = " [*BEST*]" if is_best else ""
            log(f"Ep {epoch:02d}/{max_epochs:02d} | Train Loss: {total_loss/n_batches:.4f} | "
                f"Val Loss: {mean_val_loss:.4f} (Best: {best_val_loss:.4f} @ Ep {best_val_epoch:02d}){best_marker} | "
                f"Patience: {patience_counter}/{patience} | MaskRatio: {mean_actual_mask:.3f} | Time: {ep_time:.1f}s")

            # Save resume state atomically with complete metadata
            rng_states = save_rng_states(dl_generator, train_dataset.rng)
            best_sha = compute_file_sha256(best_val_model_path) if os.path.exists(best_val_model_path) else None
            tmp_resume = resume_state_path + f".tmp_{os.getpid()}"
            torch.save({
                "epoch": epoch,
                "trial_id": trial_id,
                "seed": seed,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_epoch": best_val_epoch,
                "patience_counter": patience_counter,
                "epoch_times": epoch_times,
                "actual_mask_ratios": actual_mask_ratios,
                "trial_config": trial_cfg,
                "graph_fingerprint": graph_fp,
                "whitener_fingerprint": whitener_fp,
                "train_manifest_hash": manifest_hash,
                "top3_selection_hash": selection_hash,
                "max_epochs": max_epochs,
                "patience": patience,
                "min_delta": min_delta,
                "batch_size": batch_size,
                "optimizer_type": "AdamW",
                "gradient_clip_norm": 1.0,
                "best_val_model_sha256": best_sha,
                "runner_code_sha256": compute_file_sha256(os.path.abspath(__file__)),
                **rng_states
            }, tmp_resume)
            os.replace(tmp_resume, resume_state_path)

            # Simulated crash/pause for resume test
            if pause_after_epoch is not None and epoch >= pause_after_epoch:
                log(f"[SIMULATED INTERRUPTION] Pausing execution after epoch {epoch} to test resume capability.")
                return {"paused": True, "epoch": epoch, "best_val_epoch": best_val_epoch, "best_val_loss": best_val_loss}

            # Early stopping check
            if patience_counter >= patience:
                log(f"\n[EARLY STOPPING TRIGGERED] No improvement of >= {min_delta} for {patience} consecutive epochs. Stopping at epoch {epoch}.")
                stopped_early = True
                break

    peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

    # 7. Mandatory Best Model Rollback for Final Inference
    assert os.path.exists(best_val_model_path), f"CRITICAL: best_val_model.pt missing at {best_val_model_path}"
    log(f"\nRolling back to BEST VALIDATION MODEL from Epoch {best_val_epoch} (Loss: {best_val_loss:.4f}) for final evaluation...")
    model.load_state_dict(torch.load(best_val_model_path, map_location=device, weights_only=False))
    model.eval()

    # 8. Extract Validation Embeddings
    log(f"Extracting validation embeddings for {len(val_center_indices)} centers using best_val_model...")
    val_inf_dataset = SubgraphDataset(
        center_indices=val_center_indices,
        A_global=A_global,
        P_global=P_global,
        X_global=X_global,
        coords_global=coords_global,
        whitener=whitener,
        k_max=trial_cfg["k_max"],
        m_max=trial_cfg.get("m_max", 32),
        is_train=False,
        is_val_objective=False,
        data_cache=graph_cache
    )
    val_inf_loader = DataLoader(
        val_inf_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_subgraphs,
        pin_memory=(device.type == "cuda")
    )

    z_list = []
    with torch.no_grad():
        for batch in iter_batches(val_inf_loader, prefetch=(device.type == "cuda")):
            batch = stage_batch(batch, device)
            for k in batch:
                if isinstance(batch[k], torch.Tensor) and batch[k].device != device:
                    batch[k] = batch[k].to(device)
            z = model.encoder(
                X_vis=batch["X_vis"],
                delta_s_vis=batch["delta_s_vis"],
                vis_pad_mask=batch["vis_pad_mask"],
                L_vis_cond=batch["L_vis_cond"],
                H_vis_cond=batch["H_vis_cond"]
            )
            z_list.append(z.cpu().numpy())
    z_val = np.concatenate(z_list, axis=0)

    np.save(emb_path, z_val)

    # 9. Unsupervised Resolution Stability Selection (r*)
    log("Running 10x 80% bootstrap unsupervised stability search across r in [0.2, 0.4, 0.6, 0.8, 1.0]...")
    readout_path = Path(run_dir) / "validation_readout.json"
    readout = isolated_validation(Path(emb_path), seed, readout_path)
    best_r, stab_info = readout["r_star"], readout["stability"]
    stability_score = float(stab_info["stability_scores"][str(best_r)])
    log(f"Unsupervised selected r* = {best_r} (Stability Score: {stability_score:.4f})")

    # 10. Final Clustering & Multi-Donor ARI Assessment
    pred_clusters = np.asarray(readout["pred_clusters"])

    donor_series = pd.Series(val_donor_labels)
    donor_aris = []
    donor_ari_dict = {}

    for donor in sorted(donor_series.unique()):
        mask = (donor_series.values == donor)
        y_true = adata_train_val.obs.iloc[val_center_indices[mask]]["niche"].values
        y_pred = pred_clusters[mask]
        ari = float(adjusted_rand_score(y_true, y_pred))
        donor_aris.append(ari)
        donor_ari_dict[donor] = ari

    donor_macro_ari = float(np.mean(donor_aris))
    worst_donor_ari = float(np.min(donor_aris))
    donor_ari_std = float(np.std(donor_aris, ddof=1)) if len(donor_aris) > 1 else 0.0

    eval_result = {
        "trial_id": trial_id,
        "seed": seed,
        "desc": trial_cfg["desc"],
        "best_val_epoch": best_val_epoch,
        "stopped_epoch": final_epoch,
        "stopped_early": stopped_early,
        "best_val_loss": best_val_loss,
        "r_star": best_r,
        "stability_score": stability_score,
        "donor_macro_ari": donor_macro_ari,
        "worst_donor_ari": worst_donor_ari,
        "donor_ari_std_within_donors": donor_ari_std,
        "donor_aris": donor_ari_dict,
        "mean_epoch_time_sec": float(np.mean(epoch_times)) if epoch_times else 0.0,
        "total_train_time_sec": float(np.sum(epoch_times)) if epoch_times else 0.0,
        "peak_gpu_mem_mb": peak_gpu_mb,
        "actual_mask_ratio_mean": float(np.mean(actual_mask_ratios)) if actual_mask_ratios else 0.0,
        "expected_mask_ratio": trial_cfg["rho"],
        "config": trial_cfg,
        "graph_fingerprint": graph_fp,
        "whitener_fingerprint": whitener_fp,
        "train_manifest_hash": manifest_hash,
        "top3_selection_hash": selection_hash,
        "best_val_model_sha256": compute_file_sha256(best_val_model_path),
        "val_embeddings_sha256": compute_file_sha256(emb_path),
        "runner_code_sha256": compute_file_sha256(os.path.abspath(__file__)),
        "validation_readout_sha256": compute_file_sha256(str(readout_path)),
    }

    tmp_json = result_json_path + f".tmp_{os.getpid()}"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2)
    os.replace(tmp_json, result_json_path)

    log(f"=== Run Completed: Trial {trial_id:02d} Seed {seed:4d} | Macro ARI: {donor_macro_ari:.4f}, Worst: {worst_donor_ari:.4f}, BestValLoss: {best_val_loss:.4f} (Ep {best_val_epoch:02d}) ===\n")
    return eval_result


# -------------------------------------------------------------------------
# Formal Smoke & Contract Rigor Test
# -------------------------------------------------------------------------
def run_formal_smoke_test(
    adata_train_val: ad.AnnData,
    splits_df: pd.DataFrame,
    device: torch.device
):
    log("\n" + "=" * 75)
    log(" Running W2.3 Smoke Test: Top-3 Signatures, 68k Manifest, Cold Start, Early Stopping & Rollback")
    log("=" * 75)

    smoke_base = "outputs/runs/freqniche/human_skin/2026-09-08/20260908_human_skin_training_final/smoke_scratch"
    os.makedirs(smoke_base, exist_ok=True)

    try:
        # 1. Verify Top-3, Manifest & Validation Barcodes Distribution
        log("\n[Smoke 1/5] Verifying Top-3 Cryptographic Signature, Manifest & Validation Distribution...")
        top3_path = "outputs/experiments/human_skin/model_selection/top3_selection.json"
        candidates = verify_top3_selection_manifest(top3_path)
        manifest_path = "outputs/runs/freqniche/human_skin/2026-09-08/20260908_human_skin_training_final/manifest_train_68k.json"
        train_barcodes, manifest_hash = verify_train_manifest(manifest_path, adata_train_val, splits_df)

        val_barcodes_path = "outputs/freqniche_wave_a_diagnostics/original_val_barcodes.csv"
        val_df = pd.read_csv(val_barcodes_path)
        val_bcs = val_df["cell_barcode"].tolist() if "cell_barcode" in val_df.columns else val_df.iloc[:, 0].tolist()
        assert len(val_bcs) == 8000 and len(set(val_bcs)) == 8000, "Val barcodes must be 8,000 unique!"
        donor_col = "donor_id" if "donor_id" in adata_train_val.obs.columns else "donor"
        val_cnts = {k: v for k, v in adata_train_val.obs.loc[val_bcs, donor_col].value_counts().to_dict().items() if v > 0}
        assert set(val_cnts.keys()) == EXPECTED_VAL_DONORS
        for d in EXPECTED_VAL_DONORS:
            assert val_cnts[d] == 2000, f"Donor {d} count {val_cnts[d]} != 2,000"
        log("  Verified: 8,000 validation barcodes unique, exactly 2,000 per donor (D004, D077, D083, D107).")

        # 2. Cold Start Random Initialization Parity / Divergence
        log("\n[Smoke 2/5] Verifying Cold Start Random Initialization Parity / Divergence...")
        cfg = candidates[10]
        set_all_seeds(42)
        m1 = FreqNicheDiTModel(
            in_features=50, k_target=32, d_model=cfg["d_model"],
            n_enc_layers=cfg["n_enc"], n_dit_layers=cfg["n_dit"],
            n_heads=cfg["n_heads"], timesteps=cfg["timesteps"], mode="full_freq"
        )
        set_all_seeds(123)
        m2 = FreqNicheDiTModel(
            in_features=50, k_target=32, d_model=cfg["d_model"],
            n_enc_layers=cfg["n_enc"], n_dit_layers=cfg["n_dit"],
            n_heads=cfg["n_heads"], timesteps=cfg["timesteps"], mode="full_freq"
        )
        p1 = next(m1.parameters()).detach().cpu().numpy()
        p2 = next(m2.parameters()).detach().cpu().numpy()
        assert not np.allclose(p1, p2), "Cold-start models with seeds 42 and 123 must have distinct initial weights!"
        log("  Verified: Models initialized with different seeds have distinct weights (Cold-start valid).")

        # 3. Controlled Early Stopping & Best Checkpoint Rollback Verification
        log("\n[Smoke 3/5] Testing Injected Loss Sequence Early Stopping & Historical Best Rollback...")
        mini_centers = np.arange(128)
        val_mini_centers = np.arange(128, 192)
        val_mini_donors = np.array(["D004"] * 16 + ["D077"] * 16 + ["D083"] * 16 + ["D107"] * 16)
        coords_mini = adata_train_val.obsm["spatial"][:256]
        sample_mini = adata_train_val.obs["sample_id"].values[:256]
        X_mini = np.ascontiguousarray(adata_train_val.X[:256].toarray() if sp.issparse(adata_train_val.X) else adata_train_val.X[:256], dtype=np.float32)

        P_mini, A_mini = build_global_spatial_graph(coords=coords_mini, sample_ids=sample_mini, radius=50.0, sigma=16.6667)
        PX_mini = P_mini[mini_centers] @ X_mini
        H_mini = X_mini[mini_centers] - PX_mini
        whitener_mini = TrainOnlyWhitening(k_components=32)
        whitener_mini.fit(H_mini)

        # Injected losses: Ep 1=10.0, Ep 2=9.0 (BEST), Ep 3=9.2, Ep 4=9.3 -> Early stops at Ep 4 (patience=2)
        injected_losses = [10.0, 9.0, 9.2, 9.3]
        dir_cont = os.path.join(smoke_base, "continuous_run")
        os.makedirs(dir_cont, exist_ok=True)

        res_cont = train_single_run(
            trial_id=10,
            seed=42,
            trial_cfg=cfg,
            train_center_indices=mini_centers,
            val_center_indices=val_mini_centers,
            val_donor_labels=val_mini_donors,
            adata_train_val=adata_train_val,
            P_global=P_mini,
            A_global=A_mini,
            X_global=X_mini,
            coords_global=coords_mini,
            whitener=whitener_mini,
            graph_fp="smoke_graph_fp",
            whitener_fp="smoke_whitener_fp",
            manifest_hash=manifest_hash,
            output_base=dir_cont,
            device=device,
            max_epochs=10,
            patience=2,
            min_delta=1e-4,
            batch_size=32,
            mock_val_losses=injected_losses
        )

        assert res_cont["stopped_early"] is True, "Run should have triggered early stopping!"
        assert res_cont["stopped_epoch"] == 4, f"Expected early stop at epoch 4, got {res_cont['stopped_epoch']}"
        assert res_cont["best_val_epoch"] == 2, f"Expected best val epoch 2, got {res_cont['best_val_epoch']}"
        assert abs(res_cont["best_val_loss"] - 9.0) < 1e-6, f"Expected best val loss 9.0, got {res_cont['best_val_loss']}"
        log(f"  Verified: Early stopping triggered at Ep {res_cont['stopped_epoch']} with Best Ep={res_cont['best_val_epoch']} (Loss: {res_cont['best_val_loss']}).")

        # 4. Resume Continuity & Bitwise Equivalence Assertion
        log("\n[Smoke 4/5] Testing Interrupted Resume Equivalence & Patience Boundary...")
        dir_res = os.path.join(smoke_base, "resumed_run")
        os.makedirs(dir_res, exist_ok=True)

        # Path 1: Run up to epoch 3 (patience=1), then pause
        train_single_run(
            trial_id=10,
            seed=42,
            trial_cfg=cfg,
            train_center_indices=mini_centers,
            val_center_indices=val_mini_centers,
            val_donor_labels=val_mini_donors,
            adata_train_val=adata_train_val,
            P_global=P_mini,
            A_global=A_mini,
            X_global=X_mini,
            coords_global=coords_mini,
            whitener=whitener_mini,
            graph_fp="smoke_graph_fp",
            whitener_fp="smoke_whitener_fp",
            manifest_hash=manifest_hash,
            output_base=dir_res,
            device=device,
            max_epochs=10,  # Exact matching max_epochs
            patience=2,
            min_delta=1e-4,
            batch_size=32,
            mock_val_losses=injected_losses,
            pause_after_epoch=3
        )

        # Path 2: Resume from epoch 3 checkpoint with max_epochs=10, should stop at epoch 4!
        res_resumed = train_single_run(
            trial_id=10,
            seed=42,
            trial_cfg=cfg,
            train_center_indices=mini_centers,
            val_center_indices=val_mini_centers,
            val_donor_labels=val_mini_donors,
            adata_train_val=adata_train_val,
            P_global=P_mini,
            A_global=A_mini,
            X_global=X_mini,
            coords_global=coords_mini,
            whitener=whitener_mini,
            graph_fp="smoke_graph_fp",
            whitener_fp="smoke_whitener_fp",
            manifest_hash=manifest_hash,
            output_base=dir_res,
            device=device,
            max_epochs=10,
            patience=2,
            min_delta=1e-4,
            batch_size=32,
            mock_val_losses=injected_losses
        )

        assert res_resumed["stopped_epoch"] == 4, f"Resumed run should stop at epoch 4, got {res_resumed['stopped_epoch']}"
        assert res_resumed["best_val_epoch"] == 2, f"Resumed run best epoch should be 2, got {res_resumed['best_val_epoch']}"

        # Bitwise parameter equality check on best_val_model.pt
        best_cont = torch.load(os.path.join(dir_cont, "trial_10_seed_42", "best_val_model.pt"), map_location="cpu", weights_only=False)
        best_res = torch.load(os.path.join(dir_res, "trial_10_seed_42", "best_val_model.pt"), map_location="cpu", weights_only=False)
        max_diff = max(torch.max(torch.abs(best_cont[k] - best_res[k])).item() for k in best_cont)
        log(f"  Max parameter difference between continuous and resumed best_val_model: {max_diff:.2e}")
        assert max_diff < 1e-6, f"Best model discrepancy between continuous and resumed run! max_diff={max_diff}"

        # Embedding equality check
        emb_cont = np.load(os.path.join(dir_cont, "trial_10_seed_42", "val_embeddings.npy"))
        emb_res = np.load(os.path.join(dir_res, "trial_10_seed_42", "val_embeddings.npy"))
        emb_diff = np.max(np.abs(emb_cont - emb_res))
        log(f"  Max embedding difference between continuous and resumed run: {emb_diff:.2e}")
        assert emb_diff < 1e-6, f"Embedding discrepancy between continuous and resumed run! max_diff={emb_diff}"
        assert abs(res_cont["donor_macro_ari"] - res_resumed["donor_macro_ari"]) < 1e-6, "Macro ARI must be bitwise identical"

        log("  Verified: Continuous and resumed runs achieve exact bitwise equivalence on early stopping and rollback.")

        # 5. Checkpoint Hard-Failure Defense Assertion
        log("\n[Smoke 5/5] Asserting Checkpoint Corruption Hard-Failure Defense...")
        corrupt_dir = os.path.join(smoke_base, "corrupt_test")
        os.makedirs(os.path.join(corrupt_dir, "trial_10_seed_42"), exist_ok=True)
        # Write corrupted checkpoint
        with open(os.path.join(corrupt_dir, "trial_10_seed_42", "resume_state_last.pt"), "w") as f:
            f.write("corrupted garbage")

        hard_fail_caught = False
        try:
            train_single_run(
                trial_id=10, seed=42, trial_cfg=cfg,
                train_center_indices=mini_centers, val_center_indices=val_mini_centers,
                val_donor_labels=val_mini_donors, adata_train_val=adata_train_val,
                P_global=P_mini, A_global=A_mini, X_global=X_mini, coords_global=coords_mini,
                whitener=whitener_mini, graph_fp="smoke_graph_fp", whitener_fp="smoke_whitener_fp",
                manifest_hash=manifest_hash, output_base=corrupt_dir, device=device,
                max_epochs=2, patience=2, restart_run=False
            )
        except RuntimeError as e:
            hard_fail_caught = True
            log(f"  Verified: Corrupted checkpoint successfully triggered hard failure without silent overwrite ({e.args[0][:60]}...)")

        assert hard_fail_caught, "Corrupted checkpoint did not trigger expected hard failure!"
        log("\nAll W2.3 Smoke and Contract Rigor Assertions PASSED Successfully!")

    finally:
        # Clean up temporary smoke scratch files
        if os.path.exists(smoke_base):
            import shutil
            shutil.rmtree(smoke_base, ignore_errors=True)
            log(f"[Cleanup] Cleaned up temporary smoke directory: {smoke_base}")


# -------------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Canonical Human Skin FreqNiche Training Backend")
    parser.add_argument("--adata-path", default="datasets/h5ad/human_skin_merfish/human_skin_merfish.h5ad")
    parser.add_argument("--splits-path", default="datasets/h5ad/human_skin_merfish/donor_splits.csv")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--val-barcodes-path", default=DEFAULT_VAL_BARCODES_PATH)
    parser.add_argument("--top3-path", default=DEFAULT_TOP3_PATH)
    parser.add_argument("--output-base", default="outputs/runs/freqniche/human_skin/2026-09-08/20260908_human_skin_training_final")
    parser.add_argument("--trial-id", type=int, default=None, help="Run specific trial ID (10, 24, 20)")
    parser.add_argument("--seed", type=int, default=None, help="Run specific seed (42, 123, 2024)")
    parser.add_argument("--smoke-test", action="store_true", help="Run formal end-to-end smoke & contract tests only")
    parser.add_argument("--restart-run", action="store_true", help="Explicitly archive existing run directory and restart cold")
    parser.add_argument("--diagnostic-mode", action="store_true", help="Allow custom parameters for non-production debugging")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--model-config-path", default=None,
                        help="JSON model hyperparameters recorded by the training/reproduction launcher")
    parser.add_argument(
        "--final-model",
        action="store_true",
        help="Run only the frozen Trial-24 architecture in a canonical timestamped output directory.",
    )

    args = parser.parse_args()
    fresh_run = bool(os.environ.get("FREQNICHE_RUN_DIR"))
    if args.model_config_path is not None and (not args.final_model or not args.diagnostic_mode):
        raise ValueError("Model tuning requires --final-model --diagnostic-mode")

    # CLI Production Lockdown Assertions
    if not args.diagnostic_mode and not args.smoke_test:
        if args.trial_id is not None:
            assert args.trial_id in W23_VALID_TRIALS, f"Protocol Violation: Trial ID {args.trial_id} not in Top-3 {W23_VALID_TRIALS}! Use --diagnostic-mode for debugging."
        if args.seed is not None:
            assert args.seed in W23_VALID_SEEDS, f"Protocol Violation: Seed {args.seed} not in frozen seeds {W23_VALID_SEEDS}! Use --diagnostic-mode for debugging."
        assert args.max_epochs == 50, f"Protocol Violation: max_epochs must be 50, got {args.max_epochs}!"
        assert args.patience == 7, f"Protocol Violation: patience must be 7, got {args.patience}!"
        assert abs(args.min_delta - 1e-4) < 1e-9, f"Protocol Violation: min_delta must be 1e-4, got {args.min_delta}!"
        assert args.batch_size == 64, f"Protocol Violation: batch_size must be 64, got {args.batch_size}!"
        if not args.final_model:
            assert os.path.normpath(args.output_base) == os.path.normpath("outputs/runs/freqniche/human_skin/2026-09-08/20260908_human_skin_training_final"), (
                f"Protocol Violation: output_base must be 'outputs/runs/freqniche/human_skin/2026-09-08/20260908_human_skin_training_final' in legacy production mode! Got: {args.output_base}"
            )
        # Lock all input file paths in production mode
        assert os.path.normpath(args.adata_path) == os.path.normpath(DEFAULT_ADATA_PATH), (
            f"Protocol Violation: adata_path must be '{DEFAULT_ADATA_PATH}' in production mode! Got: {args.adata_path}. Use --diagnostic-mode to override."
        )
        assert os.path.normpath(args.splits_path) == os.path.normpath(DEFAULT_SPLITS_PATH), (
            f"Protocol Violation: splits_path must be '{DEFAULT_SPLITS_PATH}' in production mode! Got: {args.splits_path}. Use --diagnostic-mode to override."
        )
        assert os.path.normpath(args.manifest_path) == os.path.normpath(DEFAULT_MANIFEST_PATH), (
            f"Protocol Violation: manifest_path must be '{DEFAULT_MANIFEST_PATH}' in production mode! Got: {args.manifest_path}. Use --diagnostic-mode to override."
        )
        assert os.path.normpath(args.val_barcodes_path) == os.path.normpath(DEFAULT_VAL_BARCODES_PATH), (
            f"Protocol Violation: val_barcodes_path must be '{DEFAULT_VAL_BARCODES_PATH}' in production mode! Got: {args.val_barcodes_path}. Use --diagnostic-mode to override."
        )
        if not args.final_model:
            assert os.path.normpath(args.top3_path) == os.path.normpath(DEFAULT_TOP3_PATH), (
                f"Protocol Violation: top3_path must be '{DEFAULT_TOP3_PATH}' in production mode! Got: {args.top3_path}. Use --diagnostic-mode to override."
            )
        # Verify SHA-256 of locked input files
        locked_inputs = dict(EXPECTED_INPUT_HASHES)
        if args.final_model:
            locked_inputs.pop(DEFAULT_TOP3_PATH, None)
        for fpath, expected_h in locked_inputs.items():
            actual_h = compute_file_sha256(fpath)
            assert actual_h == expected_h, f"Integrity Violation: {fpath} hash mismatch: {actual_h} != {expected_h}!"
    elif args.diagnostic_mode:
        assert os.path.normpath(args.output_base) != os.path.normpath("outputs/runs/freqniche/human_skin/2026-09-08/20260908_human_skin_training_final"), (
            "Safety Violation: Diagnostic mode output_base MUST NOT be the official production directory!"
        )
        log(f"[DIAGNOSTIC MODE ENABLED] Running with custom parameters in isolated directory: {args.output_base}")

    # Canonical final-model runs may use custom training-budget values in an
    # isolated run directory, but their scientific inputs remain immutable.
    if args.final_model:
        for fpath, expected_h in EXPECTED_INPUT_HASHES.items():
            if fpath == DEFAULT_TOP3_PATH or (fresh_run and fpath == DEFAULT_MANIFEST_PATH):
                continue
            actual_h = compute_file_sha256(fpath)
            assert actual_h == expected_h, (
                f"Integrity Violation: {fpath} hash mismatch: {actual_h} != {expected_h}!"
            )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu explicitly")
    device = torch.device(args.device)
    log(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Data (Train & Val Donors ONLY; Test Expression/Coords/Labels strictly quarantined)
    splits_df = pd.read_csv(args.splits_path)
    donor_col_split = "donor_id" if "donor_id" in splits_df.columns else "donor"
    train_donors = set(splits_df[splits_df["split"] == "train"][donor_col_split])
    val_donors = set(splits_df[splits_df["split"].isin(["val", "validation"])][donor_col_split])
    test_donors = set(splits_df[splits_df["split"] == "test"][donor_col_split])
    train_val_donors = train_donors | val_donors

    # STRICT TEST DATA QUARANTINE
    assert test_donors == TEST_DONORS, f"Test donor split mismatch! Expected: {TEST_DONORS}, Found: {test_donors}"

    log(f"Loading Human Skin MERFISH (backed='r') from {args.adata_path}...")
    adata_full = sc.read_h5ad(args.adata_path, backed="r")
    donor_col_obs = "donor_id" if "donor_id" in adata_full.obs.columns else "donor"
    donor_mask = adata_full.obs[donor_col_obs].isin(train_val_donors)
    adata_train_val = adata_full[donor_mask].to_memory()
    log(f"Quarantined {len(adata_full) - len(adata_train_val)} test cells. Retained {len(adata_train_val)} train+val cells.")

    # Preprocessing contract
    log("Applying normalize_total(target_sum=1e4) + log1p...")
    sc.pp.normalize_total(adata_train_val, target_sum=1e4)
    sc.pp.log1p(adata_train_val)

    if sp.issparse(adata_train_val.X):
        adata_train_val.X = adata_train_val.X.toarray().astype(np.float32)
    else:
        adata_train_val.X = np.asarray(adata_train_val.X, dtype=np.float32)

    if args.smoke_test:
        run_formal_smoke_test(adata_train_val, splits_df, device)
        return

    # 2. Cryptographic Integrity Checks
    if args.final_model:
        top3_candidates = {24: dict(FINAL_TRIAL_24_CONFIG)}
        if args.model_config_path is not None:
            model_config = load_model_config(
                args.model_config_path, output_dir=Path(args.output_base)
            )
            changed = any(top3_candidates[24][key] != value for key, value in model_config.items())
            top3_candidates[24].update(model_config)
            if changed:
                top3_candidates[24]["desc"] = "Isolated Validation hyperparameter candidate"
        train_barcodes, manifest_hash = verify_train_manifest(
            args.manifest_path,
            adata_train_val,
            splits_df,
            verify_legacy_source=False,
            require_legacy_signature=not fresh_run,
        )
    else:
        top3_candidates = verify_top3_selection_manifest(args.top3_path)
        train_barcodes, manifest_hash = verify_train_manifest(args.manifest_path, adata_train_val, splits_df)

    # 3. Load Validation Centers & Assert Non-duplicate and Donor Balance
    assert os.path.exists(args.val_barcodes_path), f"Missing val barcodes file: {args.val_barcodes_path}"
    val_df = pd.read_csv(args.val_barcodes_path)
    val_barcodes = val_df["cell_barcode"].tolist() if "cell_barcode" in val_df.columns else (
        val_df["barcode"].tolist() if "barcode" in val_df.columns else val_df.iloc[:, 0].tolist()
    )
    assert len(val_barcodes) == 8000, f"Expected 8,000 validation barcodes, got {len(val_barcodes)}"
    assert len(set(val_barcodes)) == 8000, "Validation barcodes contain duplicates!"

    val_donor_counts = {k: v for k, v in adata_train_val.obs.loc[val_barcodes, donor_col_obs].value_counts().to_dict().items() if v > 0}
    assert set(val_donor_counts.keys()) == EXPECTED_VAL_DONORS, f"Validation donors in file {set(val_donor_counts.keys())} != {EXPECTED_VAL_DONORS}"
    for d_id in EXPECTED_VAL_DONORS:
        assert val_donor_counts.get(d_id, 0) == 2000, f"Validation donor {d_id} count is {val_donor_counts.get(d_id, 0)}, expected exactly 2,000!"

    # Map barcodes to indices
    obs_index_map = {bc: i for i, bc in enumerate(adata_train_val.obs_names)}
    train_center_indices = np.array([obs_index_map[bc] for bc in train_barcodes], dtype=np.int64)
    val_center_indices = np.array([obs_index_map[bc] for bc in val_barcodes], dtype=np.int64)
    val_donor_labels = adata_train_val.obs.iloc[val_center_indices][donor_col_obs].values

    # 4. Prepare Shared Matrices & Graph
    coords_global = adata_train_val.obsm["spatial"]
    sample_ids_global = adata_train_val.obs["sample_id"].values
    gene_names = list(adata_train_val.var_names)
    X_global = adata_train_val.X
    X_global = np.ascontiguousarray(X_global, dtype=np.float32)

    log("Computing expression matrix SHA-256 chunk-hash...")
    expr_hash = compute_expression_matrix_hash(X_global)

    # Build or Load R=50um Spatial Graph
    P_global, A_global, graph_fp = get_or_build_graph(
        coords=coords_global,
        sample_ids=sample_ids_global,
        gene_names=gene_names,
        radius=50.0
    )

    # Verify R=50um non-diagonal degree for all 68,940 train centers
    off_diag_deg = np.diff(A_global.indptr) - 1
    train_off_diag = off_diag_deg[train_center_indices]
    isolated_on_50 = np.sum(train_off_diag == 0)
    assert isolated_on_50 == 0, f"CRITICAL: Found {isolated_on_50} isolated train centers on R=50um graph!"
    assert np.min(train_off_diag) >= 1, f"Minimum off-diagonal degree on R=50um is {np.min(train_off_diag)} (< 1)!"
    log(f"[Topology Verified] All 68,940 train centers have off-diagonal degree >= 1 on R=50um graph (min: {np.min(train_off_diag)}, median: {np.median(train_off_diag):.1f}).")

    # Fit or Load Whitener
    whitener, whitener_fp = get_or_build_whitener(
        X_global=X_global,
        P_global=P_global,
        train_center_indices=train_center_indices,
        graph_fingerprint=graph_fp,
        expression_data_hash=expr_hash,
        train_manifest_hash=manifest_hash,
        k_target=32
    )

    # 5. Determine Execution Plan
    target_trials = [24] if args.final_model else ([args.trial_id] if args.trial_id is not None else W23_VALID_TRIALS)
    target_seeds = [args.seed] if args.seed is not None else W23_VALID_SEEDS

    log(f"\nExecution Matrix: Trials {target_trials} x Seeds {target_seeds} (Total {len(target_trials) * len(target_seeds)} runs)")

    for tid in target_trials:
        cfg = top3_candidates[tid]
        for s in target_seeds:
            train_single_run(
                trial_id=tid,
                seed=s,
                trial_cfg=cfg,
                train_center_indices=train_center_indices,
                val_center_indices=val_center_indices,
                val_donor_labels=val_donor_labels,
                adata_train_val=adata_train_val,
                P_global=P_global,
                A_global=A_global,
                X_global=X_global,
                coords_global=coords_global,
                whitener=whitener,
                graph_fp=graph_fp,
                whitener_fp=whitener_fp,
                manifest_hash=manifest_hash,
                output_base=args.output_base,
                device=device,
                max_epochs=args.max_epochs,
                patience=args.patience,
                min_delta=args.min_delta,
                batch_size=args.batch_size,
                restart_run=args.restart_run,
                selection_hash=None if fresh_run else EXPECTED_TOP3_SELF_SIG,
            )

    log("\nAll requested W2.3 refinement runs finished successfully!")


if __name__ == "__main__":
    main()
