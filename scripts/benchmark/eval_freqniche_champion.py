"""Evaluate FreqNiche-ST v1.0 Final (Trial 24) on Internal Fixed Human Skin Test Cohort.

Loads frozen checkpoints for Seeds 42, 123, 2024 from W2.3,
extracts embeddings on exactly the 8,000 fixed Test centers,
and evaluates with frozen r*=0.6 using Common Leiden Readout.
"""

from __future__ import annotations

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
import scanpy as sc
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.benchmark.evaluation_w24 import evaluate_common_leiden_readout
from src.freqniche_st.dataset import SubgraphDataset, collate_subgraphs
from src.freqniche_st.graph import build_global_spatial_graph
from src.freqniche_st.models import FreqNicheDiTModel
from src.freqniche_st.whitening import TrainOnlyWhitening
from src.freqniche_st.runtime_paths import resolve_run_output, project_path
from src.freqniche_st.model_config import load_model_config, inference_batch_size

DATASET_PATH = ROOT / "datasets/h5ad/human_skin_merfish/human_skin_merfish.h5ad"
SPLITS_PATH = ROOT / "datasets/h5ad/human_skin_merfish/donor_splits.csv"
TEST_CENTERS_PATH = ROOT / "outputs/contracts/human_skin/test_centers_8k.csv"
FROZEN_MANIFEST_PATH = ROOT / "outputs/runs/freqniche/human_skin/2026-09-08/20260908_human_skin_training_final/trial_24_frozen_manifest.json"

OUT_DIR = resolve_run_output("outputs/runs/freqniche/human_skin/2026-09-18/20260918_human_skin_evaluation_final")
TRAINING_DIR = Path(os.environ.get("FREQNICHE_TRAIN_DIR", str(OUT_DIR)))
if not TRAINING_DIR.is_absolute():
    TRAINING_DIR = ROOT / TRAINING_DIR
RES_DIR = OUT_DIR / "results"
EMB_DIR = OUT_DIR / "embeddings"


def compute_file_sha256(filepath: Path | str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print(">>> FreqNiche-ST v1.0 Final (Trial 24) Test Cohort Re-evaluation")
    print("=" * 80)

    if os.environ.get("FREQNICHE_RUN_DIR"):
        run_request = json.loads((TRAINING_DIR / "run_manifest.json").read_text(encoding="utf-8"))["request"]
        device = torch.device(run_request["device"])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if os.environ.get("FREQNICHE_RUN_DIR"):
        frozen_ckpts = {}
        trial_cfg = None
        run_manifest = json.loads((TRAINING_DIR / "run_manifest.json").read_text(encoding="utf-8"))
        seeds = [int(seed) for seed in run_manifest["request"]["seeds"]]
        for seed in seeds:
            seed_dir = TRAINING_DIR / f"trial_24_seed_{seed}"
            result_path = seed_dir / "eval_result.json"
            model_path = seed_dir / "best_val_model.pt"
            if not result_path.exists() or not model_path.exists():
                raise FileNotFoundError(f"Canonical training assets missing for seed {seed}: {seed_dir}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if trial_cfg is None:
                trial_cfg = result["config"]
            elif trial_cfg != result["config"]:
                raise RuntimeError("Checkpoint configurations differ across seeds")
            frozen_ckpts[str(seed)] = {
                "model_path": str(model_path),
                "model_sha256": compute_file_sha256(model_path),
                "frozen_r_star": result["r_star"],
                "stability_score": result["stability_score"],
                "whitener_fingerprint": result["whitener_fingerprint"],
            }
        assert trial_cfg is not None
        run_model_config = load_model_config(output_dir=OUT_DIR)
        for key, value in run_model_config.items():
            if trial_cfg.get(key) != value:
                raise RuntimeError(f"Test model config differs from frozen training config: {key}")
    else:
        assert FROZEN_MANIFEST_PATH.exists(), f"Missing frozen manifest: {FROZEN_MANIFEST_PATH}"
        with open(FROZEN_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        trial_cfg = manifest["config"]
        frozen_ckpts = manifest["frozen_checkpoints"]
        seeds = [42, 123, 2024]
        for seed in seeds:
            historical_result = ROOT / f"outputs/runs/freqniche/human_skin/2026-09-08/20260908_human_skin_training_final/trial_24_seed_{seed}/eval_result.json"
            if not historical_result.is_file():
                raise FileNotFoundError(f"Historical training result missing: {historical_result}")
            frozen_ckpts[str(seed)]["whitener_fingerprint"] = json.loads(
                historical_result.read_text(encoding="utf-8")
            )["whitener_fingerprint"]
    print(f"Trial Config: {trial_cfg}")

    # Load Test Metadata
    test_meta_df = pd.read_csv(TEST_CENTERS_PATH)
    assert len(test_meta_df) == 8000, f"Expected 8000 test centers, got {len(test_meta_df)}"
    test_barcodes = list(test_meta_df["cell_barcode"])

    # Load AnnData
    print(f"\nLoading AnnData from {DATASET_PATH} ...")
    adata = ad.read_h5ad(DATASET_PATH)
    print(f"AnnData shape: {adata.shape}")

    # Standard preprocessing
    print("Applying normalize_total(1e4) and log1p() ...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    X_global = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    coords_global = adata.obsm["spatial"]

    # Match test indices
    obs_names = list(adata.obs_names)
    bc_to_idx = {bc: idx for idx, bc in enumerate(obs_names)}
    test_center_indices = np.array([bc_to_idx[bc] for bc in test_barcodes], dtype=np.int64)

    # Build spatial graph with R=50.0 um
    print("Building global spatial graph with R=50.0 um ...")
    sample_col = "sample_id" if "sample_id" in adata.obs else "sample"
    sample_ids = adata.obs[sample_col].to_numpy()
    P_global, A_global = build_global_spatial_graph(
        coords=coords_global,
        sample_ids=sample_ids,
        radius=50.0,
    )

    # Reuse the exact residual whitener fitted by the training backend.
    # Refitting on raw expression would change the model inputs at Test time.
    whitener_fps = {info.get("whitener_fingerprint") for info in frozen_ckpts.values()}
    if len(whitener_fps) != 1 or None in whitener_fps:
        raise RuntimeError("Missing or inconsistent training whitener fingerprints")
    whitener_path = ROOT / "outputs/cache/whiteners" / f"whitener_{next(iter(whitener_fps))}.pt"
    if not whitener_path.is_file():
        raise FileNotFoundError(f"Frozen residual whitener missing: {whitener_path}")
    state = torch.load(whitener_path, map_location="cpu", weights_only=False)
    whitener = TrainOnlyWhitening(k_components=32)
    for key in ("mean_", "components_", "singular_values_", "whitening_matrix_", "n_components_fitted_"):
        setattr(whitener, key, state[key])
    whitener.is_fitted = True
    if whitener.n_components_fitted_ != 32:
        raise RuntimeError("Frozen whitener does not have the expected 32 components")

    RES_DIR.mkdir(parents=True, exist_ok=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    test_dataset = SubgraphDataset(
        center_indices=test_center_indices,
        A_global=A_global,
        P_global=P_global,
        X_global=X_global,
        coords_global=coords_global,
        whitener=whitener,
        k_max=trial_cfg["k_max"],
        m_max=trial_cfg.get("m_max", 32),
        is_train=False,
        is_val_objective=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=inference_batch_size(OUT_DIR, 64),
        shuffle=False,
        collate_fn=collate_subgraphs,
        pin_memory=(device.type == "cuda"),
    )

    for seed in seeds:
        print(f"\n--- Evaluating Seed {seed} ---")
        t0 = time.time()
        ckpt_info = frozen_ckpts[str(seed)]
        model_path = project_path(ckpt_info["model_path"])
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        assert model_path.exists(), f"Missing model checkpoint: {model_path}"

        # Verify model hash
        actual_sha = compute_file_sha256(model_path)
        assert actual_sha == ckpt_info["model_sha256"], (
            f"Model SHA mismatch for seed {seed}: {actual_sha} != {ckpt_info['model_sha256']}"
        )

        model = FreqNicheDiTModel(
            in_features=X_global.shape[1],
            d_model=trial_cfg["d_model"],
            n_enc_layers=trial_cfg["n_enc"],
            n_dit_layers=trial_cfg["n_dit"],
            n_heads=trial_cfg["n_heads"],
            timesteps=trial_cfg["timesteps"],
            lambda_cov=trial_cfg["lambda_cov"],
        ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.eval()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        z_list = []
        with torch.no_grad():
            for batch in test_loader:
                X_vis = batch["X_vis"].to(device)
                delta_s_vis = batch["delta_s_vis"].to(device)
                vis_pad_mask = batch["vis_pad_mask"].to(device)
                L_vis_cond = batch["L_vis_cond"].to(device)
                H_vis_cond = batch["H_vis_cond"].to(device)

                z = model.encode_niche(
                    X_vis=X_vis,
                    delta_s_vis=delta_s_vis,
                    vis_pad_mask=vis_pad_mask,
                    L_vis_cond=L_vis_cond,
                    H_vis_cond=H_vis_cond,
                )
                z_list.append(z.cpu().numpy())

        test_embeddings = np.concatenate(z_list, axis=0).astype(np.float32)
        assert test_embeddings.shape == (8000, trial_cfg["d_model"])
        elapsed = time.time() - t0

        meth_gpu = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0

        # Save embeddings
        emb_file = EMB_DIR / f"freqniche_st_v1_trial24_seed_{seed}_test_z.npy"
        np.save(emb_file, test_embeddings)
        emb_sha = compute_file_sha256(emb_file)

        # Common Leiden Readout using frozen r*=0.6
        frozen_r = float(ckpt_info["frozen_r_star"])
        eval_res = evaluate_common_leiden_readout(
            test_embeddings=test_embeddings,
            test_metadata=test_meta_df,
            frozen_r_star=frozen_r,
            seed=seed,
            compute_asw=True,
        )

        out_data = {
            "benchmark_type": "primary",
            "method": "FreqNiche-ST v1.0 Final",
            "seed": seed,
            "paradigm": "Inductive",
            "status": "PASS",
            "elapsed_sec": elapsed,
            "method_gpu_allocated_mb": meth_gpu,
            "system_gpu_used_peak_mb": None,
            "peak_rss_mb": float(psutil.Process().memory_info().rss / (1024 * 1024)),
            "common_readout": {
                "stability_selection": {
                    "optimal_r_star": frozen_r,
                    "validation_stability_score": ckpt_info["stability_score"],
                },
                "test_evaluation": eval_res,
            },
            "embedding_file": str(emb_file.relative_to(ROOT)),
            "embedding_sha256": emb_sha,
            "checkpoint_sha256": actual_sha,
        }

        res_path = RES_DIR / f"freqniche_st_v1_trial24_seed_{seed}.json"
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)

        print(f"[OK] Seed {seed} Evaluated:")
        print(f" -> Test Donor-Macro ARI : {eval_res['test_donor_macro_ari']:.4f}")
        print(f" -> Worst-Donor ARI      : {eval_res['worst_donor_ari']:.4f}")
        print(f" -> Donor-Macro NMI      : {eval_res['test_donor_macro_nmi']:.4f}")
        print(f" -> ASW Silhouette       : {eval_res['asw_silhouette']:.4f}")
        print(f" -> Wrote result to {res_path}")

    print("\n" + "=" * 80)
    print(">>> FreqNiche-ST v1.0 Final Test Evaluation Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
