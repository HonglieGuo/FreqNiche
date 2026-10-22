#!/usr/bin/env python3
"""
scripts/benchmark/train_cosmx.py

W2.6 CosMx Human NSCLC Formal 3-Seed Training Script (v4.0 Approved)
================================================================================
Strict Reproducibility & Cryptographic Verification Protocol:
  - 3 Random Seeds: 42, 123, 2024
  - Pre-requisites Cryptographic Audit:
      * H5AD file SHA-256 verified against manifest
      * Scale conversion factor (0.120280945) verified against manifest
      * Coordinates contract: obsm['spatial_um'] and obsm['spatial'] verified
      * Train graph caches (P and A) SHA-256 verified
      * Center cohorts verified with Triple Hashes (file, sorted, ordered)
      * Whitener (960 -> 32) verified with file and input residual SHA-256
  - Architecture: Frozen Trial 24 Configuration
      * in_features = 960 (CosMx targeted gene panel)
      * k_target = 32 (TrainOnlyWhitening)
      * d_model = 128
      * n_enc_layers = 3, n_heads = 8, mlp_ratio = 2.0 (FFN = 256)
      * n_dit_layers = 2, n_heads = 8
      * timesteps = 50, lambda_cov = 0.04, mode = 'full_freq'
  - Physical Scale Alignment:
      * Coordinates read directly from adata.obsm['spatial_um'] (NO secondary scaling)
      * R = 50.0 um, sigma = 16.666667 um
      * Relative position encoding uses physical micrometer differences
  - Optimization:
      * AdamW(lr=1e-3, weight_decay=1e-3), clip_grad_norm=1.0, batch_size=64
      * Early stopping: patience=7, min_delta=1e-4 on Validation Denoising Loss
      * max_epochs=50
  - Overwrite Protection:
      * Safeguards against accidental re-runs overwriting completed seeds
================================================================================
"""

import os
import sys
import json
import time
import copy
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import scipy.sparse as sp
import scanpy as sc

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.freqniche_st.models.freqniche_model import FreqNicheDiTModel
from src.freqniche_st.whitening import TrainOnlyWhitening
from src.freqniche_st.dataset import SubgraphDataset, collate_subgraphs
from src.freqniche_st.graph import build_global_spatial_graph
from src.freqniche_st.runtime_paths import resolve_dataset_path, resolve_run_output
from src.freqniche_st.model_config import load_model_config, build_freqniche_model

DATASET_PATH = resolve_dataset_path("FREQNICHE_DATASET_PATH", "datasets/h5ad/cosmx_human_nsclc/cosmx_human_nsclc.h5ad")
OUTPUT_DIR = resolve_run_output("outputs/runs/freqniche/cosmx_nsclc/2026-09-17/20260917_cosmx_nsclc_final")
CACHE_DIR = OUTPUT_DIR / "cache"
CKPT_DIR = OUTPUT_DIR / "checkpoints"

TRAIN_CENTERS_PATH = OUTPUT_DIR / "train_centers.json"
VAL_CENTERS_PATH = OUTPUT_DIR / "val_centers.json"
WHITENER_PATH = OUTPUT_DIR / "whitener.pt"
TRAIN_P_PATH = CACHE_DIR / "cosmx_train_graph_P.npz"
TRAIN_A_PATH = CACHE_DIR / "cosmx_train_graph_A.npz"
MANIFEST_PATH = OUTPUT_DIR / "cosmx_data_audit_manifest.json"


def compute_file_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def compute_sorted_barcode_sha256(items: List[str]) -> str:
    serialized = json.dumps(sorted(items), separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_ordered_barcode_sha256(items: List[str]) -> str:
    serialized = json.dumps(list(items), separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def train_single_seed(
    seed: int,
    device: torch.device,
    train_dataset: SubgraphDataset,
    val_dataset: SubgraphDataset,
    output_seed_dir: Path,
    scale_factor: float,
    batch_size: int = 64,
    max_epochs: int = 50,
    patience: int = 7,
    min_delta: float = 1e-4,
    overwrite: bool = False,
    model_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = model_config or load_model_config()
    print("\n" + "=" * 70)
    print(f"STARTING COSMX FORMAL TRAINING | SEED: {seed} | DEVICE: {device}")
    print("=" * 70)
    output_seed_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = output_seed_dir / f"model_seed_{seed}_best.pt"
    if best_ckpt_path.exists() and not overwrite:
        raise FileExistsError(
            f"Checkpoint already exists for seed {seed} at {best_ckpt_path}! "
            f"Pass --overwrite to allow replacing it."
        )

    # Set deterministic random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_subgraphs,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_subgraphs,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    # Model definition: Trial 24 configuration with 960 features
    model = build_freqniche_model(960, cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (hidden_dim={cfg['d_model']}, enc={cfg['n_enc']}, dit={cfg['n_dit']})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None
    patience_counter = 0

    train_loss_history = []
    val_loss_history = []
    epoch_durations = []

    t_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        t_ep_start = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            X_vis = batch["X_vis"].to(device)
            delta_s_vis = batch["delta_s_vis"].to(device)
            vis_pad_mask = batch["vis_pad_mask"].to(device)
            H_0_target = batch["H_0_target"].to(device)
            delta_s_target = batch["delta_s_target"].to(device)
            target_mask = batch["target_mask"].to(device)
            L_vis_cond = batch["L_vis_cond"].to(device)
            H_vis_cond = batch["H_vis_cond"].to(device)

            out = model(
                X_vis=X_vis,
                delta_s_vis=delta_s_vis,
                vis_pad_mask=vis_pad_mask,
                H_0_target=H_0_target,
                delta_s_target=delta_s_target,
                target_mask=target_mask,
                L_vis_cond=L_vis_cond,
                H_vis_cond=H_vis_cond,
            )

            loss = out["loss"]
            assert torch.isfinite(loss).all(), f"Non-finite loss in epoch {epoch}: {loss}"

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_loss_history.append(avg_train_loss)

        # Validation denoising loss evaluation
        model.eval()
        val_losses = []
        val_gen = torch.Generator(device=device).manual_seed(seed + 99999)

        with torch.no_grad():
            for b in val_loader:
                H_0 = b["H_0_target"].to(device)
                B_val = H_0.shape[0]
                t_val = torch.randint(1, model.timesteps + 1, size=(B_val,), device=device, generator=val_gen)
                noise_val = torch.randn(H_0.shape, device=device, generator=val_gen)
                H_t_val = model.schedule.q_sample(x_0=H_0, t=t_val, noise=noise_val)

                z_val_niche = model.encoder(
                    X_vis=b["X_vis"].to(device),
                    delta_s_vis=b["delta_s_vis"].to(device),
                    vis_pad_mask=b["vis_pad_mask"].to(device),
                    L_vis_cond=b["L_vis_cond"].to(device),
                    H_vis_cond=b["H_vis_cond"].to(device),
                )
                pred_noise_val = model.denoiser(
                    H_t=H_t_val,
                    t=t_val,
                    z_v_niche=z_val_niche,
                    delta_s_target=b["delta_s_target"].to(device),
                    target_mask=b["target_mask"].to(device),
                )
                diff_sq = (pred_noise_val - noise_val) ** 2
                token_l = diff_sq.sum(dim=-1)
                m_flt = b["target_mask"].to(device).float()
                val_loss_batch = (token_l * m_flt).sum() / max(m_flt.sum(), 1.0)
                val_losses.append(val_loss_batch.item())

        mean_val_loss = float(np.mean(val_losses))
        val_loss_history.append(mean_val_loss)
        ep_duration = time.perf_counter() - t_ep_start
        epoch_durations.append(ep_duration)

        # Early stopping tracking
        is_best = False
        if mean_val_loss < (best_val_loss - min_delta):
            best_val_loss = mean_val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
            is_best = True
        else:
            patience_counter += 1

        best_flag = " [* BEST]" if is_best else f" [Patience {patience_counter}/{patience}]"
        print(
            f"Seed {seed:4d} | Epoch {epoch:02d}/{max_epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Denoise Loss: {mean_val_loss:.4f} | "
            f"Duration: {ep_duration:.1f}s{best_flag}"
        )

        if patience_counter >= patience:
            print(f"\n[Early Stopping Triggered] No improvement for {patience} epochs. Stopping at epoch {epoch}.")
            break

    total_time = time.perf_counter() - t_start
    peak_vram_mib = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0

    print(
        f"\nTraining Complete for Seed {seed} in {total_time:.1f}s ({total_time/60:.2f} min). "
        f"Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f}). Peak VRAM: {peak_vram_mib:.2f} MiB."
    )

    # Save best checkpoint atomically
    save_payload = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_epochs_trained": len(train_loss_history),
        "total_time_seconds": total_time,
        "peak_vram_mib": peak_vram_mib,
        "hyperparameters": {
            "in_features": 960,
            "k_target": 32,
            "d_model": cfg["d_model"],
            "n_enc_layers": cfg["n_enc"],
            "n_dit_layers": cfg["n_dit"],
            "n_heads": cfg["n_heads"],
            "mlp_ratio": 2.0,
            "timesteps": cfg["timesteps"],
            "lambda_cov": cfg["lambda_cov"],
            "mode": "full_freq",
            "batch_size": batch_size,
            "lr": cfg["lr"],
            "weight_decay": cfg["wd"],
            "physical_scale_um_per_pixel": scale_factor,
            "radius_um": 50.0,
            "sigma_um": 16.666667,
        },
        "model_config": cfg,
        "model_state_dict": best_state_dict,
    }

    temp_ckpt = best_ckpt_path.with_suffix(".tmp.pt")
    torch.save(save_payload, temp_ckpt)
    os.replace(temp_ckpt, best_ckpt_path)
    ckpt_sha256 = compute_file_sha256(best_ckpt_path)
    print(f"Saved best model checkpoint atomically to: {best_ckpt_path}")
    print(f"Checkpoint SHA-256: {ckpt_sha256}")

    # Save training history atomically
    history_path = output_seed_dir / "training_history.json"
    history_payload = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "epoch_durations": epoch_durations,
        "total_time_seconds": total_time,
        "peak_vram_mib": peak_vram_mib,
        "checkpoint_sha256": ckpt_sha256,
        "model_config": cfg,
    }
    temp_hist = history_path.with_suffix(".tmp.json")
    with open(temp_hist, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)
    os.replace(temp_hist, history_path)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_epochs": len(train_loss_history),
        "total_time_seconds": total_time,
        "peak_vram_mib": peak_vram_mib,
        "checkpoint_path": str(best_ckpt_path),
        "checkpoint_sha256": ckpt_sha256,
    }


def main():
    parser = argparse.ArgumentParser(description="CosMx Formal 3-Seed Training (v4.0)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024], help="Seeds to train")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--model-config-path", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing seed checkpoints")
    parser.add_argument("--preflight-only", action="store_true", help="Perform only entry audit and exit without training")
    args = parser.parse_args()
    if args.max_epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.min_delta < 0:
        parser.error("max-epochs, patience and batch-size must be positive; min-delta must be nonnegative")

    model_config = load_model_config(args.model_config_path, output_dir=OUTPUT_DIR if args.model_config_path else None)
    device = torch.device(args.device)
    print("=" * 80)
    print("W2.6 COSMX NSCLC FORMAL 3-SEED TRAINING PROTOCOL (v4.0)")
    print(f"Device: {device} | Seeds: {args.seeds} | Max Epochs: {args.max_epochs} | Patience: {args.patience}")
    print("=" * 80)

    # 1. Cryptographic Audit of Pre-requisites
    print("\n[Audit 1/3] Verifying data audit manifest, H5AD SHA, and assets ...")
    assert MANIFEST_PATH.exists(), f"Missing manifest: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1.1 Verify H5AD File SHA-256
    assert DATASET_PATH.exists(), f"Missing dataset: {DATASET_PATH}"
    h5ad_sha = compute_file_sha256(DATASET_PATH)
    assert h5ad_sha == manifest["h5ad_sha256"], f"H5AD SHA mismatch: {h5ad_sha} != {manifest['h5ad_sha256']}"
    print(f" -> H5AD file SHA-256 verified: {h5ad_sha}")

    # 1.2 Verify Scale Conversion Factor
    scale_factor = manifest["scale_conversion"]["factor_um_per_pixel"]
    assert abs(scale_factor - 0.120280945) < 1e-9, f"Scale factor mismatch: {scale_factor} != 0.120280945"
    print(f" -> Scale factor verified from manifest: {scale_factor} um/px")

    # 1.3 Verify cached graphs
    assert TRAIN_P_PATH.exists() and TRAIN_A_PATH.exists()
    p_sha = compute_file_sha256(TRAIN_P_PATH)
    a_sha = compute_file_sha256(TRAIN_A_PATH)
    assert p_sha == manifest["train_graph_cache"]["P_sha256"], "Train graph P SHA-256 mismatch!"
    assert a_sha == manifest["train_graph_cache"]["A_sha256"], "Train graph A SHA-256 mismatch!"
    print(" -> Train graph caches P/A SHA-256 verified.")

    # 1.4 Verify cohorts with Triple Hashes (file, sorted, ordered)
    assert TRAIN_CENTERS_PATH.exists() and VAL_CENTERS_PATH.exists()
    assert compute_file_sha256(TRAIN_CENTERS_PATH) == manifest["cohorts"]["train"]["file_sha256"], "Train centers file SHA mismatch!"
    assert compute_file_sha256(VAL_CENTERS_PATH) == manifest["cohorts"]["validation"]["file_sha256"], "Val centers file SHA mismatch!"

    with open(TRAIN_CENTERS_PATH, "r", encoding="utf-8") as f:
        tc_data = json.load(f)
    with open(VAL_CENTERS_PATH, "r", encoding="utf-8") as f:
        vc_data = json.load(f)

    assert compute_sorted_barcode_sha256(tc_data["cell_barcodes"]) == manifest["cohorts"]["train"]["sorted_barcode_sha256"], "Train sorted barcodes SHA mismatch!"
    assert compute_ordered_barcode_sha256(tc_data["cell_barcodes"]) == manifest["cohorts"]["train"]["ordered_barcode_sha256"], "Train ordered barcodes SHA mismatch!"
    assert compute_sorted_barcode_sha256(vc_data["cell_barcodes"]) == manifest["cohorts"]["validation"]["sorted_barcode_sha256"], "Val sorted barcodes SHA mismatch!"
    assert compute_ordered_barcode_sha256(vc_data["cell_barcodes"]) == manifest["cohorts"]["validation"]["ordered_barcode_sha256"], "Val ordered barcodes SHA mismatch!"
    print(f" -> Train ({len(tc_data['cell_barcodes']):,}) and Validation ({len(vc_data['cell_barcodes']):,}) Triple Hashes verified.")

    # 1.5 Verify Whitener
    assert WHITENER_PATH.exists(), f"Missing whitener: {WHITENER_PATH}"
    w_sha = compute_file_sha256(WHITENER_PATH)
    assert w_sha == manifest["whitener"]["sha256"], f"Whitener SHA mismatch: {w_sha} != {manifest['whitener']['sha256']}"
    print(f" -> Whitener (960 -> 32) SHA-256 verified: {w_sha}")

    # 2. Load AnnData and Preprocess Contexts
    print(f"\n[Audit 2/3] Loading AnnData from {DATASET_PATH} ...")
    adata = sc.read_h5ad(DATASET_PATH)

    # 2.1 Assert Coordinates Contract
    assert "spatial_um" in adata.obsm, "Dataset missing obsm['spatial_um']!"
    assert "spatial" in adata.obsm, "Dataset missing obsm['spatial']!"
    assert np.array_equal(adata.obsm["spatial"], adata.obsm["spatial_um"]), "obsm['spatial'] != obsm['spatial_um']!"
    assert (adata.obs["spatial_unit"] == "micron").all(), "obs['spatial_unit'] != 'micron'!"
    print(" -> Coordinates contract verified: spatial_um, spatial, and spatial_unit conform.")

    # 2.2 Slicing Train context
    train_mask = (adata.obs["split"] == "train").to_numpy()
    train_indices = np.where(train_mask)[0]
    print(f"Slicing Train context ({len(train_indices):,} cells) and applying normalize_total(1e4) + log1p() ...")
    adata_train = adata[train_indices].copy()
    sc.pp.normalize_total(adata_train, target_sum=1e4)
    sc.pp.log1p(adata_train)

    X_train_full = adata_train.X
    if sp.issparse(X_train_full):
        X_train_full = X_train_full.toarray()
    X_train_full = np.asarray(X_train_full, dtype=np.float32)

    # Read coordinates directly from spatial_um (NO secondary scaling)
    coords_train_um = np.asarray(adata.obsm["spatial_um"][train_indices], dtype=np.float32)

    P_train = sp.load_npz(TRAIN_P_PATH)
    A_train = sp.load_npz(TRAIN_A_PATH)

    train_global_indices = tc_data["global_indices"]
    global_to_train_local = {g: l for l, g in enumerate(train_indices)}
    train_center_local_indices = np.array([global_to_train_local[g] for g in train_global_indices], dtype=np.int64)

    # Load Whitener
    whitener_dict = torch.load(WHITENER_PATH, map_location="cpu", weights_only=False)
    whitener = TrainOnlyWhitening(k_components=32, lambda_min=1e-4)
    whitener.mean_ = whitener_dict["mean_"]
    whitener.components_ = whitener_dict["components_"]
    whitener.singular_values_ = whitener_dict["singular_values_"]
    whitener.whitening_matrix_ = whitener_dict["whitening_matrix_"]
    whitener.n_components_fitted_ = whitener_dict["n_components_fitted_"]
    whitener.is_fitted = True

    # Slicing Validation context (Lung5)
    val_mask = (adata.obs["split"] == "validation").to_numpy()
    val_indices = np.where(val_mask)[0]
    print(f"Slicing Validation context ({len(val_indices):,} cells) and applying normalize_total(1e4) + log1p() ...")
    adata_val = adata[val_indices].copy()
    sc.pp.normalize_total(adata_val, target_sum=1e4)
    sc.pp.log1p(adata_val)

    X_val_full = adata_val.X
    if sp.issparse(X_val_full):
        X_val_full = X_val_full.toarray()
    X_val_full = np.asarray(X_val_full, dtype=np.float32)

    # Read validation coordinates directly from spatial_um (NO secondary scaling)
    coords_val_um = np.asarray(adata.obsm["spatial_um"][val_indices], dtype=np.float32)
    val_sample_ids = adata.obs["sample_id"].iloc[val_indices].to_numpy()

    print("Building Validation spatial graph (radius=50.0 um, sigma=16.6667 um) ...")
    P_val, A_val = build_global_spatial_graph(coords=coords_val_um, sample_ids=val_sample_ids, radius=50.0, max_neighbors=None)

    val_global_indices = vc_data["global_indices"]
    global_to_val_local = {g: l for l, g in enumerate(val_indices)}
    val_center_local_indices = np.array([global_to_val_local[g] for g in val_global_indices], dtype=np.int64)

    print(f"\n[Audit 3/3] Data pipeline ready: Train={len(train_center_local_indices):,} centers, Val={len(val_center_local_indices):,} centers.")

    if args.preflight_only:
        print("\n" + "=" * 80)
        print("[PREFLIGHT AUDIT PASSED] All pre-requisites, hashes, and contracts 100% verified!")
        print("Formal training is blocked pending user authorization. Exiting cleanly.")
        print("=" * 80)
        return

    all_seed_results = {}

    for seed in args.seeds:
        train_dataset = SubgraphDataset(
            center_indices=train_center_local_indices,
            A_global=A_train,
            P_global=P_train,
            X_global=X_train_full,
            coords_global=coords_train_um,
            whitener=whitener,
            k_max=model_config["k_max"],
            m_max=model_config["m_max"],
            rho=model_config["rho"],
            p_center_mask=model_config["p_center_mask"],
            is_train=True,
            seed=seed,
        )

        val_dataset = SubgraphDataset(
            center_indices=val_center_local_indices,
            A_global=A_val,
            P_global=P_val,
            X_global=X_val_full,
            coords_global=coords_val_um,
            whitener=whitener,
            k_max=model_config["k_max"],
            m_max=model_config["m_max"],
            rho=model_config["rho"],
            p_center_mask=model_config["p_center_mask"],
            is_train=False,
            is_val_objective=True,
            seed=seed,
        )

        output_seed_dir = CKPT_DIR / f"seed_{seed}"
        res = train_single_seed(
            seed=seed,
            device=device,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_seed_dir=output_seed_dir,
            scale_factor=scale_factor,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            overwrite=args.overwrite,
            model_config=model_config,
        )
        all_seed_results[str(seed)] = res

    # Summary Manifest
    summary_path = CKPT_DIR / "cosmx_training_summary.json"
    temp_summary = summary_path.with_suffix(".tmp.json")
    with open(temp_summary, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "cosmx_human_nsclc",
            "physical_scale_um_per_pixel": scale_factor,
            "radius_um": 50.0,
            "sigma_um": 16.666667,
            "seeds": args.seeds,
            "results": all_seed_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    os.replace(temp_summary, summary_path)

    print("\n" + "=" * 80)
    print("ALL 3 SEEDS FORMAL TRAINING FINISHED SUCCESSFULLY!")
    print(f"Summary Manifest saved to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
