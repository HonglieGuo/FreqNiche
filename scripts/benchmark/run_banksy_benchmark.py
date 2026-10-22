"""Run BANKSY-Controlled-Multisample Benchmark on Human Skin MERFISH (Production v3.2).

Strictly adheres to:
1. Multi-sample block-diagonal spatial graph without cross-sample leakage.
2. Preserves complex AGF directional weights (complex128).
3. Simultaneously extracts and saves both Validation (8,000) and Test (8,000) embeddings.
4. Performs Common Leiden Readout r* selection strictly using the shared deterministic
   10-fold 80% bootstrap manifest (bootstrap_manifest_val8k_10fold.json).
5. Evaluates Test centers across 3 seeds (42, 123, 2024) under raw embedding policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.benchmark.adapters.banksy import OfficialBANKSYAdapter
from src.benchmark.evaluation_w24 import (
    evaluate_common_leiden_readout,
    select_unsupervised_resolution_r_star,
)
from src.freqniche_st.runtime_paths import resolve_run_output

DATASET_PATH = ROOT / "datasets/h5ad/human_skin_merfish/human_skin_merfish.h5ad"
VAL_BARCODES_PATH = ROOT / "outputs/contracts/human_skin/validation_barcodes_8k.csv"
TEST_CENTERS_PATH = ROOT / "outputs/contracts/human_skin/test_centers_8k.csv"

FINAL_DIR = resolve_run_output("baselines/Baselines_outputs/runs/human_skin/2026-09-18/20260918_human_skin_benchmark_final")
RES_DIR = FINAL_DIR / "results"
EMB_DIR = FINAL_DIR / "embeddings"


def compute_file_sha256(filepath: Path | str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="BANKSY Human Skin benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    args = parser.parse_args()
    print("=" * 80)
    print(">>> BANKSY-Controlled-Multisample W2.4 Benchmark Runner (Production v3.2)")
    print("=" * 80)

    RES_DIR.mkdir(parents=True, exist_ok=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Validation and Test Mapping
    val_df = pd.read_csv(VAL_BARCODES_PATH)
    test_df = pd.read_csv(TEST_CENTERS_PATH)
    assert len(val_df) == 8000
    assert len(test_df) == 8000
    val_barcodes = set(val_df["cell_barcode"])
    test_barcodes = list(test_df["cell_barcode"])

    # 2. Load AnnData
    print(f"Loading AnnData from {DATASET_PATH} ...")
    adata = ad.read_h5ad(DATASET_PATH)
    print(f"AnnData loaded: {adata.shape}")

    obs_names = list(adata.obs_names)
    bc_to_idx = {bc: idx for idx, bc in enumerate(obs_names)}
    val_indices = np.array([bc_to_idx[bc] for bc in val_df["cell_barcode"]], dtype=np.int64)
    test_indices = np.array([bc_to_idx[bc] for bc in test_barcodes], dtype=np.int64)

    # 3. Instantiate Official BANKSY Adapter
    print("Running OfficialBANKSYAdapter.fit_transform() across all samples ...")
    adapter = OfficialBANKSYAdapter(
        protocol="controlled",
        lambda_param=0.2,
        pca_dim=20,
        return_diagnostics=False,
    )

    t_feat_start = time.time()
    res = adapter.fit_transform(
        adata,
        seed=42,
        embedding_only=True,
    )
    feat_time = time.time() - t_feat_start
    print(f"[OK] BANKSY embedding constructed in {feat_time:.2f}s, shape: {res.embedding.shape}")

    full_embeddings = res.embedding
    val_emb = full_embeddings[val_indices]
    test_emb = full_embeddings[test_indices]
    assert val_emb.shape == (8000, 20)
    assert test_emb.shape == (8000, 20)

    # 4. Save both Validation and Test embeddings
    val_emb_path = EMB_DIR / "banksy_controlled_val_z.npy"
    test_emb_path = EMB_DIR / "banksy_controlled_test_z.npy"
    np.save(val_emb_path, val_emb)
    np.save(test_emb_path, test_emb)
    val_emb_sha = compute_file_sha256(val_emb_path)
    test_emb_sha = compute_file_sha256(test_emb_path)
    print(f"[OK] Saved BANKSY Validation embeddings to {val_emb_path} (SHA: {val_emb_sha[:16]})")
    print(f"[OK] Saved BANKSY Test embeddings to {test_emb_path} (SHA: {test_emb_sha[:16]})")

    # 5. Common Leiden Readout: Select r* on Validation 8,000 centers via shared bootstrap manifest
    print("\n[Common Readout] Selecting optimal r* on Validation 8,000 centers via shared bootstrap manifest ...")
    best_r, stability_report = select_unsupervised_resolution_r_star(
        val_embeddings=val_emb,
        resolutions=[0.2, 0.4, 0.6, 0.8, 1.0],
        embedding_postprocessing="raw",
        k_min=5,
        k_max=25,
        seed=42,
    )
    print(f" -> Selected optimal r*: {best_r} (Stability score: {stability_report['best_stability_score']:.4f})")

    # Evaluate on Test across 3 clustering seeds (42, 123, 2024)
    seeds = args.seeds
    for seed in seeds:
        print(f"\nEvaluating Common Leiden Readout on Test centers with clustering seed {seed} ...")
        eval_res = evaluate_common_leiden_readout(
            test_embeddings=test_emb,
            test_metadata=test_df,
            frozen_r_star=best_r,
            embedding_postprocessing="raw",
            seed=seed,
            compute_asw=True,
        )

        total_rss = float(psutil.Process().memory_info().rss / (1024 * 1024))
        out_data = {
            "benchmark_type": "primary",
            "method": "BANKSY-Controlled-Multisample",
            "seed": seed,
            "paradigm": "Transductive",
            "status": "PASS",
            "elapsed_sec": feat_time,
            "method_gpu_allocated_mb": 0.0,
            "system_gpu_used_peak_mb": None,
            "peak_rss_mb": total_rss,
            "common_readout": {
                "stability_selection": stability_report,
                "test_evaluation": eval_res,
            },
            "embedding_file": str(test_emb_path.relative_to(ROOT)),
            "embedding_sha256": test_emb_sha,
            "val_embedding_file": str(val_emb_path.relative_to(ROOT)),
            "val_embedding_sha256": val_emb_sha,
        }

        res_path = RES_DIR / f"banksy_controlled_seed_{seed}.json"
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)

        print(f"[OK] Seed {seed} Test Donor-Macro ARI : {eval_res['test_donor_macro_ari']:.4f}")
        print(f" -> Worst-Donor ARI : {eval_res['worst_donor_ari']:.4f}")
        print(f" -> Donor-Macro NMI : {eval_res['test_donor_macro_nmi']:.4f}")
        print(f" -> ASW Silhouette  : {eval_res['asw_silhouette']:.4f}")
        print(f" -> Wrote result to {res_path}")

    print("\n" + "=" * 80)
    print(">>> BANKSY-Controlled-Multisample Benchmark Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
