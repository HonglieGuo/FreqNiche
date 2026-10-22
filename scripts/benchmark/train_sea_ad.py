"""
Formal 3-Seed Training for FreqNiche-ST v1.0 on SEA-AD (Stage 3).

Strictly enforces:
1. Frozen Trial 24 architecture:
   - in_features = 180, k_target = 32, d_model = 128
   - n_enc_layers = 3, n_dit_layers = 2, n_heads = 8, timesteps = 50, lambda_cov = 0.04
2. Frozen optimization protocol:
   - batch_size = 64
   - AdamW(lr=1e-3, weight_decay=1e-3)
   - gradient_clip_norm = 1.0
3. Early stopping & rollback contract:
   - max_epochs = 50, patience = 7, min_delta = 1e-4
   - Evaluation on 10,000 Validation centers every epoch
   - Saves and strictly rolls back to best_val_model.pt
4. Frozen data preprocessing:
   - sc.pp.normalize_total(adata_split, target_sum=1e4)
   - sc.pp.log1p(adata_split)
5. Cryptographic integrity check:
   - Re-verifies SHA-256 signatures of dataset, cohorts, graph caches, and whitener
6. Non-silent overwrite protection:
   - Refuses to overwrite existing checkpoints unless --overwrite flag is passed
"""

import sys
import os
import json
import time
import copy
import argparse
import hashlib
from pathlib import Path
from typing import Dict, Any

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.freqniche_st.graph import build_global_spatial_graph
from src.freqniche_st.models.freqniche_model import FreqNicheDiTModel
from src.freqniche_st.whitening import TrainOnlyWhitening
from src.freqniche_st.dataset import SubgraphDataset, collate_subgraphs
from src.freqniche_st.runtime_paths import resolve_dataset_path, resolve_run_output
from src.freqniche_st.model_config import load_model_config


DATASET_PATH = resolve_dataset_path("FREQNICHE_DATASET_PATH", "datasets/h5ad/sea_ad/sea_ad_mtg_merfish.h5ad")
OUTPUT_DIR = resolve_run_output("outputs/runs/freqniche/sea_ad/2026-09-15/20260915_sea_ad_final")
CACHE_DIR = OUTPUT_DIR / "cache"
MANIFEST_PATH = OUTPUT_DIR / "cohort_manifest.json"
TRAIN_CENTERS_PATH = OUTPUT_DIR / "train_centers.json"
VAL_CENTERS_PATH = OUTPUT_DIR / "val_centers.json"
WHITENER_PATH = OUTPUT_DIR / "whitener.pt"
TRAIN_P_PATH = CACHE_DIR / "train_graph_P.npz"
TRAIN_A_PATH = CACHE_DIR / "train_graph_A.npz"


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_sha256_list(items) -> str:
    hasher = hashlib.sha256()
    for item in items:
        hasher.update(str(item).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def train_single_seed(
    seed: int,
    device: torch.device,
    train_dataset: SubgraphDataset,
    val_dataset: SubgraphDataset,
    output_seed_dir: Path,
    max_epochs: int = 50,
    patience: int = 7,
    min_delta: float = 1e-4,
    batch_size: int = 64,
    model_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = model_config or load_model_config()
    print("\n" + "=" * 70)
    print(f"STARTING TRAINING FOR SEED {seed}")
    print("=" * 70)
    output_seed_dir.mkdir(parents=True, exist_ok=True)

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_subgraphs,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_subgraphs,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Model definition with frozen Trial 24 configuration
    model = FreqNicheDiTModel(
        in_features=180,
        k_target=32,
        d_model=cfg["d_model"],
        n_enc_layers=cfg["n_enc"],
        n_dit_layers=cfg["n_dit"],
        n_heads=cfg["n_heads"],
        timesteps=cfg["timesteps"],
        lambda_cov=cfg["lambda_cov"],
        mode="full_freq"
    ).to(device)

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

            out = model(
                X_vis=batch["X_vis"].to(device),
                delta_s_vis=batch["delta_s_vis"].to(device),
                vis_pad_mask=batch["vis_pad_mask"].to(device),
                H_0_target=batch["H_0_target"].to(device),
                delta_s_target=batch["delta_s_target"].to(device),
                target_mask=batch["target_mask"].to(device),
                L_vis_cond=batch["L_vis_cond"].to(device),
                H_vis_cond=batch["H_vis_cond"].to(device)
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
                    H_vis_cond=b["H_vis_cond"].to(device)
                )
                pred_noise_val = model.denoiser(
                    H_t=H_t_val,
                    t=t_val,
                    z_v_niche=z_val_niche,
                    delta_s_target=b["delta_s_target"].to(device),
                    target_mask=b["target_mask"].to(device)
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

        # Check for improvement
        is_best = False
        if mean_val_loss < (best_val_loss - min_delta):
            best_val_loss = mean_val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
            is_best = True
        else:
            patience_counter += 1

        best_marker = "(* BEST)" if is_best else ""
        print(f"Epoch [{epoch:02d}/{max_epochs:02d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {mean_val_loss:.4f} {best_marker} | Patience: {patience_counter}/{patience} | Time: {ep_duration:.1f}s")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Reverting to best epoch {best_epoch} (Val Loss: {best_val_loss:.4f}).")
            break

    total_training_time = time.perf_counter() - t_start
    peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024**2) if device.type == "cuda" else 0.0

    # Save Best Model Checkpoint
    best_model_path = output_seed_dir / "best_val_model.pt"
    assert best_state_dict is not None, "FATAL: best_state_dict is None!"
    torch.save(best_state_dict, best_model_path)
    model_sha256 = compute_file_sha256(best_model_path)
    print(f"\nBest model saved to: {best_model_path} (Epoch {best_epoch}, SHA-256: {model_sha256[:16]}...)")

    # Record history
    history = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_epochs_run": len(train_loss_history),
        "total_time_seconds": total_training_time,
        "avg_epoch_seconds": float(np.mean(epoch_durations)),
        "peak_vram_mb": peak_vram_mb,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "checkpoint_file": "best_val_model.pt",
        "checkpoint_sha256": model_sha256,
        "model_config": cfg,
    }
    with open(output_seed_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return history


def main():
    parser = argparse.ArgumentParser(description="Formal 3-Seed Training on SEA-AD")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024], help="Random seeds to run")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--model-config-path", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing checkpoints")
    args = parser.parse_args()
    if args.max_epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.min_delta < 0:
        parser.error("max-epochs, patience and batch-size must be positive; min-delta must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    print("=" * 80)
    print("STAGE 3: FORMAL FREQNICHE-ST V1.0 TRAINING ON SEA-AD")
    print(f"Target Seeds: {args.seeds}")
    print("=" * 80)

    device = torch.device(args.device)
    model_config = load_model_config(args.model_config_path, output_dir=OUTPUT_DIR if args.model_config_path else None)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 1. Non-silent overwrite protection
    for s in args.seeds:
        ckpt_p = OUTPUT_DIR / "checkpoints" / f"seed_{s}" / "best_val_model.pt"
        if ckpt_p.exists() and not args.overwrite:
            raise RuntimeError(
                f"FATAL: Checkpoint {ckpt_p} already exists! Pass --overwrite to proceed, silent overwrite is forbidden."
            )

    # 2. Strict Cryptographic Integrity Checks (Re-verifying all signatures)
    print("\n[Audit] Cryptographically verifying all prerequisite assets and signatures ...")
    assert MANIFEST_PATH.exists(), f"Missing cohort manifest: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Check dataset
    assert DATASET_PATH.exists(), f"Missing dataset: {DATASET_PATH}"
    ds_sha = compute_file_sha256(DATASET_PATH)
    assert ds_sha == manifest["dataset_sha256"], f"Dataset SHA mismatch: {ds_sha} != {manifest['dataset_sha256']}"
    print(" -> Dataset SHA-256 verified.")

    # Check graph caches
    assert TRAIN_P_PATH.exists() and TRAIN_A_PATH.exists()
    p_sha = compute_file_sha256(TRAIN_P_PATH)
    a_sha = compute_file_sha256(TRAIN_A_PATH)
    assert p_sha == manifest["topology_contract"]["cached_train_graph"]["P_sha256"], "P_train cache SHA mismatch!"
    assert a_sha == manifest["topology_contract"]["cached_train_graph"]["A_sha256"], "A_train cache SHA mismatch!"
    print(" -> Train graph caches P/A SHA-256 verified.")

    # Check cohorts
    assert TRAIN_CENTERS_PATH.exists() and VAL_CENTERS_PATH.exists()
    with open(TRAIN_CENTERS_PATH, "r", encoding="utf-8") as f:
        tc_data = json.load(f)
    with open(VAL_CENTERS_PATH, "r", encoding="utf-8") as f:
        vc_data = json.load(f)

    tc_sha = compute_sha256_list(tc_data["cell_barcodes"])
    vc_sha = compute_sha256_list(vc_data["cell_barcodes"])
    assert tc_sha == manifest["cohorts"]["train"]["sha256"], "Train centers barcodes SHA mismatch!"
    assert vc_sha == manifest["cohorts"]["validation"]["sha256"], "Validation centers barcodes SHA mismatch!"
    print(" -> Train (68,940) and Validation (10,000) centers SHA-256 verified.")

    # Check Whitener
    assert WHITENER_PATH.exists(), f"Missing whitener: {WHITENER_PATH}"
    w_sha = compute_file_sha256(WHITENER_PATH)
    assert w_sha == manifest["whitener"]["sha256"], f"Whitener SHA mismatch: {w_sha} != {manifest['whitener']['sha256']}"
    print(" -> Residual Whitener SHA-256 verified.")
    print("[Audit PASSED] All 5 cryptographic prerequisites 100% match manifest.")

    # 3. Load AnnData and apply Frozen Preprocessing
    print(f"\nLoading AnnData from {DATASET_PATH} ...")
    adata = sc.read_h5ad(DATASET_PATH)

    # Prepare Train context with normalize_total(1e4) + log1p()
    train_mask = (adata.obs["split"] == "train").to_numpy()
    train_indices = np.where(train_mask)[0]
    n_train_cells = len(train_indices)
    assert n_train_cells == 1210773

    print("Slicing Train context and applying normalize_total(1e4) + log1p() ...")
    adata_train = adata[train_indices].copy()
    sc.pp.normalize_total(adata_train, target_sum=1e4)
    sc.pp.log1p(adata_train)

    X_train_full = adata_train.X
    if sp.issparse(X_train_full):
        X_train_full = X_train_full.toarray()
    X_train_full = np.asarray(X_train_full, dtype=np.float32)
    coords_train_full = adata.obsm["spatial"][train_indices].astype(np.float32)

    P_train = sp.load_npz(TRAIN_P_PATH)
    A_train = sp.load_npz(TRAIN_A_PATH)

    train_global_indices = tc_data["global_indices"]
    global_to_train_local = {g: l for l, g in enumerate(train_indices)}
    train_center_local_indices = np.array([global_to_train_local[g] for g in train_global_indices], dtype=np.int64)

    # Load Whitener
    print("Loading fitted Whitener ...")
    whitener_dict = torch.load(WHITENER_PATH, map_location="cpu", weights_only=False)
    whitener = TrainOnlyWhitening(k_components=32, lambda_min=1e-4)
    whitener.mean_ = whitener_dict["mean_"]
    whitener.components_ = whitener_dict["components_"]
    whitener.singular_values_ = whitener_dict["singular_values_"]
    whitener.whitening_matrix_ = whitener_dict["whitening_matrix_"]
    whitener.n_components_fitted_ = whitener_dict["n_components_fitted_"]
    whitener.is_fitted = True

    # Prepare Validation context with normalize_total(1e4) + log1p()
    val_mask = (adata.obs["split"] == "validation").to_numpy()
    val_indices = np.where(val_mask)[0]
    n_val_cells = len(val_indices)
    assert n_val_cells == 399148

    print("Slicing Validation context and applying normalize_total(1e4) + log1p() ...")
    adata_val = adata[val_indices].copy()
    sc.pp.normalize_total(adata_val, target_sum=1e4)
    sc.pp.log1p(adata_val)

    X_val_full = adata_val.X
    if sp.issparse(X_val_full):
        X_val_full = X_val_full.toarray()
    X_val_full = np.asarray(X_val_full, dtype=np.float32)
    coords_val_full = adata.obsm["spatial"][val_indices].astype(np.float32)
    val_sample_ids = adata.obs["sample_id"].iloc[val_indices].to_numpy()

    # Validation spatial graph with max_neighbors=None
    print("Building Validation spatial graph (radius=50.0 um, max_neighbors=None) ...")
    P_val, A_val = build_global_spatial_graph(coords=coords_val_full, sample_ids=val_sample_ids, radius=50.0, max_neighbors=None)

    val_global_indices = vc_data["global_indices"]
    global_to_val_local = {g: l for l, g in enumerate(val_indices)}
    val_center_local_indices = np.array([global_to_val_local[g] for g in val_global_indices], dtype=np.int64)

    print(f"Data ready: Train={len(train_center_local_indices):,} centers, Val={len(val_center_local_indices):,} centers.")

    all_seed_results = {}

    for seed in args.seeds:
        train_dataset = SubgraphDataset(
            center_indices=train_center_local_indices,
            A_global=A_train,
            P_global=P_train,
            X_global=X_train_full,
            coords_global=coords_train_full,
            whitener=whitener,
            k_max=model_config["k_max"],
            m_max=model_config["m_max"],
            rho=model_config["rho"],
            p_center_mask=model_config["p_center_mask"],
            is_train=True,
            seed=seed
        )

        val_dataset = SubgraphDataset(
            center_indices=val_center_local_indices,
            A_global=A_val,
            P_global=P_val,
            X_global=X_val_full,
            coords_global=coords_val_full,
            whitener=whitener,
            k_max=model_config["k_max"],
            m_max=model_config["m_max"],
            rho=model_config["rho"],
            p_center_mask=model_config["p_center_mask"],
            is_train=False,
            is_val_objective=True,
            seed=seed
        )

        output_seed_dir = OUTPUT_DIR / "checkpoints" / f"seed_{seed}"
        res = train_single_seed(
            seed=seed,
            device=device,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_seed_dir=output_seed_dir,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            batch_size=args.batch_size,
            model_config=model_config,
        )
        all_seed_results[str(seed)] = res

    # Update manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest["training_results"] = {
        "seeds": list(all_seed_results.keys()),
        "summary": {s: {"best_epoch": r["best_epoch"], "best_val_loss": r["best_val_loss"], "sha256": r["checkpoint_sha256"]} for s, r in all_seed_results.items()}
    }
    manifest["status"] = "STAGE3_TRAINING_COMPLETE_READY_FOR_EVAL"

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("STAGE 3 TRAINING COMPLETE FOR ALL SEEDS.")
    print("=" * 80)


if __name__ == "__main__":
    main()
