"""Run CellCharter-Canonical Benchmark on Human Skin MERFISH (Production v3.2).

Executes official CellCharter pipeline with fail-safe checkpoint persistence:
1. scVI latent representation (n_latent=20, n_layers=3).
   - scVI model is persisted directly inside adapter upon training completion.
2. Spatial neighborhood aggregation with physical sample isolation.
3. Intermediate embeddings (Val 8k, Test 8k) and official labels are persisted immediately
   before any clustering/evaluation steps.
4. Common Leiden Readout on Test 8,000 centers with r* selected on Validation 8,000 centers
   using shared deterministic 10-fold bootstrap manifest.
5. Official cc.tl.Cluster GMM clustering for secondary readout.
Runs 3 Seeds (42, 123, 2024).
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
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.benchmark.adapters.cellcharter import CanonicalCellCharterAdapter
from src.benchmark.evaluation_w24 import (
    evaluate_common_leiden_readout,
    evaluate_official_labels_readout,
    select_unsupervised_resolution_r_star,
)
from src.freqniche_st.runtime_paths import resolve_run_output

DATASET_PATH = ROOT / "datasets/h5ad/human_skin_merfish/human_skin_merfish.h5ad"
VAL_BARCODES_PATH = ROOT / "outputs/contracts/human_skin/validation_barcodes_8k.csv"
TEST_CENTERS_PATH = ROOT / "outputs/contracts/human_skin/test_centers_8k.csv"

FINAL_DIR = resolve_run_output("baselines/Baselines_outputs/runs/human_skin/2026-09-18/20260918_human_skin_benchmark_final")
RES_DIR = FINAL_DIR / "results"
EMB_DIR = FINAL_DIR / "embeddings"
CKPT_DIR = FINAL_DIR / "checkpoints/cellcharter"


def compute_file_sha256(filepath: Path | str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="CellCharter-Canonical W2.4 Benchmark Runner")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for scVI training")
    parser.add_argument("--max-epochs", type=int, default=200, help="Max epochs for scVI training")
    parser.add_argument("--seed", type=int, default=None, help="Run single seed if specified")
    args = parser.parse_args()

    print("=" * 80)
    print(">>> CellCharter-Canonical W2.4 Benchmark Runner (Production v3.2)")
    print(f"Config: batch_size={args.batch_size}, max_epochs={args.max_epochs}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    RES_DIR.mkdir(parents=True, exist_ok=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    val_df = pd.read_csv(VAL_BARCODES_PATH)
    test_df = pd.read_csv(TEST_CENTERS_PATH)
    assert len(val_df) == 8000
    assert len(test_df) == 8000
    test_barcodes = list(test_df["cell_barcode"])

    print(f"Loading AnnData from {DATASET_PATH} ...")
    adata = ad.read_h5ad(DATASET_PATH)
    print(f"AnnData shape: {adata.shape}")

    obs_names = list(adata.obs_names)
    bc_to_idx = {bc: idx for idx, bc in enumerate(obs_names)}
    val_indices = np.array([bc_to_idx[bc] for bc in val_df["cell_barcode"]], dtype=np.int64)
    test_indices = np.array([bc_to_idx[bc] for bc in test_barcodes], dtype=np.int64)

    seeds = [args.seed] if args.seed is not None else [42, 123, 2024]
    for seed in seeds:
        print(f"\n" + "=" * 60)
        print(f"--- Running CellCharter-Canonical Seed {seed} ---")
        print("=" * 60)
        t_start = time.time()

        val_emb_path = EMB_DIR / f"cellcharter_canonical_seed_{seed}_val_z.npy"
        test_emb_path = EMB_DIR / f"cellcharter_canonical_seed_{seed}_test_z.npy"
        labels_path = EMB_DIR / f"cellcharter_canonical_seed_{seed}_official_labels.npy"

        # Checkpoint healing: if embeddings already extracted, skip training
        if val_emb_path.exists() and test_emb_path.exists() and labels_path.exists():
            print(f"[RESUME] Found pre-persisted embeddings and labels for Seed {seed}. Loading from disk ...")
            val_emb = np.load(val_emb_path)
            test_emb = np.load(test_emb_path)
            test_official_labels = np.load(labels_path)
            elapsed = 0.0
            meth_gpu = 0.0
            peak_rss = float(psutil.Process().memory_info().rss / (1024 * 1024))
        else:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            adapter = CanonicalCellCharterAdapter(
                track="canonical",
                n_latent=20,
                n_layers=3,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
            )

            # Run official fit_transform with internal model checkpointing
            res = adapter.fit_transform(
                adata,
                n_clusters=10,  # For official GMM clustering readout
                seed=seed,
                device=device,
                embedding_only=False,
                checkpoint_dir=CKPT_DIR,
            )
            elapsed = time.time() - t_start
            meth_gpu = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
            peak_rss = float(psutil.Process().memory_info().rss / (1024 * 1024))

            full_emb = res.embedding
            val_emb = full_emb[val_indices]
            test_emb = full_emb[test_indices]
            assert val_emb.shape[0] == 8000
            assert test_emb.shape[0] == 8000

            # Pre-emptive Fail-Safe Persistence: Save embeddings & labels BEFORE clustering
            np.save(val_emb_path, val_emb)
            np.save(test_emb_path, test_emb)
            if res.canonical_labels is not None:
                test_official_labels = res.canonical_labels[test_indices]
                np.save(labels_path, test_official_labels)
            else:
                test_official_labels = None
            print(f"[OK] Pre-emptively persisted Val/Test embeddings and labels for Seed {seed}")

        test_emb_sha = compute_file_sha256(test_emb_path)

        # 1. Common Leiden Readout: Select r* on Val via shared bootstrap manifest
        print("\nSelecting optimal r* on Validation 8,000 centers via shared bootstrap manifest ...")
        best_r, stability_report = select_unsupervised_resolution_r_star(
            val_embeddings=val_emb,
            resolutions=[0.2, 0.4, 0.6, 0.8, 1.0],
            embedding_postprocessing="raw",
            k_min=5,
            k_max=25,
            seed=42,
        )
        print(f" -> Selected optimal r*: {best_r} (Stability score: {stability_report['best_stability_score']:.4f})")

        # Evaluate on Test centers
        common_eval = evaluate_common_leiden_readout(
            test_embeddings=test_emb,
            test_metadata=test_df,
            frozen_r_star=best_r,
            embedding_postprocessing="raw",
            seed=seed,
            compute_asw=True,
        )

        # 2. Official End-to-End Readout
        official_eval = None
        if test_official_labels is not None:
            official_eval = evaluate_official_labels_readout(
                pred_labels=test_official_labels,
                test_metadata=test_df,
                label_source="cellcharter_cc_tl_cluster",
            )

        out_data = {
            "benchmark_type": "primary",
            "method": "CellCharter-Canonical",
            "seed": seed,
            "paradigm": "Transductive",
            "status": "PASS",
            "elapsed_sec": elapsed,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "method_gpu_allocated_mb": meth_gpu,
            "system_gpu_used_peak_mb": None,
            "peak_rss_mb": peak_rss,
            "common_readout": {
                "stability_selection": stability_report,
                "test_evaluation": common_eval,
            },
            "official_readout": official_eval,
            "embedding_file": str(test_emb_path.relative_to(ROOT)),
            "embedding_sha256": test_emb_sha,
        }

        res_path = RES_DIR / f"cellcharter_canonical_seed_{seed}.json"
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)

        print(f"[OK] Seed {seed} Test Donor-Macro ARI : {common_eval['test_donor_macro_ari']:.4f}")
        print(f" -> Worst-Donor ARI : {common_eval['worst_donor_ari']:.4f}")
        print(f" -> Donor-Macro NMI : {common_eval['test_donor_macro_nmi']:.4f}")
        print(f" -> ASW Silhouette  : {common_eval['asw_silhouette']:.4f}")
        if official_eval is not None:
            print(f" -> Official GMM Test ARI : {official_eval['test_donor_macro_ari']:.4f}")
        print(f" -> Saved to {res_path}")

    print("\n" + "=" * 80)
    print(">>> CellCharter-Canonical Benchmark Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
