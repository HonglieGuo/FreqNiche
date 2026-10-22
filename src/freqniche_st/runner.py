"""
Training and Dual-Track Evaluation Runner for FreqNiche-ST Wave A Experiments.
Strictly implements:
- Full Context Universe architecture (Global graph over complete samples, sampling centers only)
- Pre-filters isolated nodes from training/eval centers
- Shuffled-P, Shuffled-G-Bundle, and Shuffled-X-Condition topological controls with target preservation
- PairedView-SAGEPool baseline with true 2-layer neighborhood aggregation
- Validation objective checkpointing with fixed target masks and cross-epoch fixed noise generator
- Training history tracking (train loss, val loss, best epoch, early stopping)
- Validation donor Track-B evaluation for pre-frozen baseline selection
- Embedding saving, hash recording, and fail-fast assertions
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import sys
import time
import json
import hashlib
import copy
import numpy as np
import scipy.sparse as sp
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader

from freqniche_st.graph import (
    build_global_spatial_graph,
    build_within_sample_shuffled_p_graph,
    build_within_sample_shuffled_g_bundle_map,
    build_within_sample_shuffled_x_features
)
from freqniche_st.whitening import TrainOnlyWhitening, TrainOnlyPCA
from freqniche_st.dataset import SubgraphDataset, collate_subgraphs
from freqniche_st.data_cache import SharedPreprocessing
from freqniche_st.data_loading import stage_batch, iter_batches
from freqniche_st.models import (
    FreqNicheDiTModel,
    NicheMAEModel,
    PairedViewSAGEPoolModel
)
from freqniche_st.evaluation import (
    evaluate_track_a_oracle_k,
    compute_unsupervised_clustering_stability,
    evaluate_track_b_leiden
)


def train_single_run(
    config_name: str,
    seed: int,
    X_global: np.ndarray,
    coords_global: np.ndarray,
    sample_ids: np.ndarray,
    A_global: sp.csr_matrix,
    P_global: sp.csr_matrix,
    train_center_indices: np.ndarray,
    val_center_indices: np.ndarray,
    test_center_indices: Optional[np.ndarray],
    obs_df: pd.DataFrame,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: Optional[int] = None,
    min_delta: float = 1e-4,
    output_dir: str = "outputs/freqniche_wave_a",
    eval_test: bool = True,
    optimize_data: bool = True,
    shared_preprocessing: Optional[SharedPreprocessing] = None
) -> Dict[str, Any]:
    """
    Executes a single model training and dual-track evaluation run over the full Context Universe.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    emb_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)

    n_features = X_global.shape[1]

    # 1. Configure Negative Control Graphs, Bundles, and Features
    bundle_map = None
    if config_name == "Shuffled-P":
        P_cond_global, A_cond_global = build_within_sample_shuffled_p_graph(A_global, sample_ids, seed=seed)
        X_cond_global = X_global
    elif config_name in ["Shuffled-G", "Shuffled-G-Bundle"]:
        P_cond_global, A_cond_global = P_global, A_global
        X_cond_global = X_global
        bundle_map = build_within_sample_shuffled_g_bundle_map(len(X_global), sample_ids, seed=seed)
    elif config_name in ["Shuffled-X", "Shuffled-X-Condition"]:
        P_cond_global, A_cond_global = P_global, A_global
        X_cond_global = build_within_sample_shuffled_x_features(X_global, sample_ids, seed=seed)
    else:
        P_cond_global, A_cond_global = P_global, A_global
        X_cond_global = X_global

    A_cond_arg = A_cond_global if config_name == "Shuffled-P" else None

    # 2. Fit Train-only Whitening & PCA on TRUE train residuals
    if optimize_data:
        prep = shared_preprocessing
        if prep is None:
            prep = SharedPreprocessing(X_global, P_global, A_global, train_center_indices)
        prep.validate(X_global, P_global, A_global, train_center_indices)
        whitener, pca_id = prep.whitener, prep.pca_id
        data_cache = prep.true_graph_cache
        diagnostic_cache = prep.true_graph_cache
    else:
        X_train = X_global[train_center_indices]
        P_train = P_global[train_center_indices]
        PX_train = (P_train @ X_global).astype(np.float32)
        H_train = X_train - PX_train
        whitener = TrainOnlyWhitening(k_components=32, lambda_min=1e-4).fit(H_train)
        pca_id = TrainOnlyPCA(k_components=32).fit(X_train)
        data_cache = diagnostic_cache = None
    pin_memory = optimize_data and device.type == "cuda"

    white_hash = hashlib.sha256(whitener.whitening_matrix_.tobytes()).hexdigest()
    pca_hash = hashlib.sha256(pca_id.components_.tobytes()).hexdigest()

    # 3. Instantiate model
    mode = "raw_input" if config_name == "Raw-DiT" else "full_freq"
    is_sagepool = (config_name in ["PairedView-SAGEPool", "PairedView-GraphSAGE", "Expression-CellNiche"])

    if config_name == "Same-Encoder-MAE":
        model = NicheMAEModel(
            in_features=n_features,
            k_target=32,
            d_model=128,
            mode=mode
        ).to(device)
    elif is_sagepool:
        model = PairedViewSAGEPoolModel(
            in_features=n_features,
            d_model=128
        ).to(device)
    else:  # Full-DiT, Raw-DiT, Shuffled-P, Shuffled-G, Shuffled-X
        model = FreqNicheDiTModel(
            in_features=n_features,
            k_target=32,
            d_model=128,
            mode=mode
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 4. Build Train & Validation Objective Datasets
    train_dataset = SubgraphDataset(
        center_indices=train_center_indices,
        A_global=A_global,
        A_cond_global=A_cond_arg,
        P_global=P_global,
        X_global=X_global,
        coords_global=coords_global,
        X_cond_global=X_cond_global,
        bundle_map=bundle_map,
        whitener=whitener,
        k_max=64,
        m_max=32,
        rho=0.5,
        p_center_mask=0.30,
        is_train=True,
        return_paired_view=is_sagepool,
        seed=seed,
        data_cache=data_cache
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_subgraphs,
        num_workers=0,
        pin_memory=pin_memory
    )

    # Dedicated Validation Objective Dataset with deterministic fixed target masks and fixed paired views
    val_obj_dataset = SubgraphDataset(
        center_indices=val_center_indices,
        A_global=A_global,
        A_cond_global=A_cond_arg,
        P_global=P_global,
        X_global=X_global,
        coords_global=coords_global,
        X_cond_global=X_cond_global,
        bundle_map=bundle_map,
        whitener=whitener,
        k_max=64,
        m_max=32,
        rho=0.5,
        p_center_mask=0.30,
        is_train=False,
        is_val_objective=True,
        return_paired_view=is_sagepool,
        seed=seed,
        data_cache=data_cache
    )
    val_obj_loader = DataLoader(val_obj_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_subgraphs, pin_memory=pin_memory)

    # 5. Training Loop with History Tracking and Early Stopping
    t_start = time.time()
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    train_loss_history = []
    val_loss_history = []
    patience_counter = 0

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in iter_batches(train_loader, prefetch=optimize_data):
            batch = stage_batch(batch, device, optimize_data)
            optimizer.zero_grad()
            
            if is_sagepool:
                X_c_1 = batch["X_center_true"].to(device)
                X_v_1 = batch["X_vis"].to(device)
                m_1 = batch["vis_pad_mask"].to(device)
                X_v_2 = batch["X_vis_2"].to(device) if "X_vis_2" in batch else None
                m_2 = batch["vis_pad_mask_2"].to(device) if "vis_pad_mask_2" in batch else None
                
                out = model(
                    X_center_1=X_c_1,
                    X_vis_1=X_v_1,
                    vis_pad_mask_1=m_1,
                    X_vis_2=X_v_2,
                    vis_pad_mask_2=m_2
                )
            elif config_name == "Same-Encoder-MAE":
                out = model(
                    X_vis=batch["X_vis"].to(device),
                    delta_s_vis=batch["delta_s_vis"].to(device),
                    vis_pad_mask=batch["vis_pad_mask"].to(device),
                    H_0_target=batch["H_0_target"].to(device),
                    delta_s_target=batch["delta_s_target"].to(device),
                    target_mask=batch["target_mask"].to(device),
                    L_vis_cond=batch["L_vis_cond"].to(device) if mode == "full_freq" else None,
                    H_vis_cond=batch["H_vis_cond"].to(device) if mode == "full_freq" else None
                )
            else:  # DiT variants
                out = model(
                    X_vis=batch["X_vis"].to(device),
                    delta_s_vis=batch["delta_s_vis"].to(device),
                    vis_pad_mask=batch["vis_pad_mask"].to(device),
                    H_0_target=batch["H_0_target"].to(device),
                    delta_s_target=batch["delta_s_target"].to(device),
                    target_mask=batch["target_mask"].to(device),
                    L_vis_cond=batch["L_vis_cond"].to(device) if mode == "full_freq" else None,
                    H_vis_cond=batch["H_vis_cond"].to(device) if mode == "full_freq" else None
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

        # Evaluate real validation objective loss using cross-epoch fixed noise generator
        model.eval()
        val_losses = []
        val_gen = torch.Generator(device=device).manual_seed(seed + 99999)

        with torch.no_grad():
            for b in iter_batches(val_obj_loader, prefetch=optimize_data):
                b = stage_batch(b, device, optimize_data)
                if is_sagepool:
                    out_val = model(
                        X_center_1=b["X_center_true"].to(device),
                        X_vis_1=b["X_vis"].to(device),
                        vis_pad_mask_1=b["vis_pad_mask"].to(device),
                        X_vis_2=b["X_vis_2"].to(device) if "X_vis_2" in b else None,
                        vis_pad_mask_2=b["vis_pad_mask_2"].to(device) if "vis_pad_mask_2" in b else None
                    )
                    val_losses.append(out_val["loss"].item())
                elif config_name == "Same-Encoder-MAE":
                    out_val = model(
                        X_vis=b["X_vis"].to(device),
                        delta_s_vis=b["delta_s_vis"].to(device),
                        vis_pad_mask=b["vis_pad_mask"].to(device),
                        H_0_target=b["H_0_target"].to(device),
                        delta_s_target=b["delta_s_target"].to(device),
                        target_mask=b["target_mask"].to(device),
                        L_vis_cond=b["L_vis_cond"].to(device) if mode == "full_freq" else None,
                        H_vis_cond=b["H_vis_cond"].to(device) if mode == "full_freq" else None
                    )
                    val_losses.append(out_val["loss"].item())
                else:
                    # For DiT: compute cross-epoch fixed validation diffusion loss on fixed targets
                    H_0 = b["H_0_target"].to(device)
                    B_val = H_0.shape[0]
                    t_val = torch.randint(1, model.timesteps + 1, size=(B_val,), device=device, generator=val_gen)
                    noise_val = torch.randn(H_0.shape, device=device, generator=val_gen)
                    H_t_val = model.schedule.q_sample(x_0=H_0, t=t_val, noise=noise_val)

                    z_val_niche = model.encoder(
                        X_vis=b["X_vis"].to(device),
                        delta_s_vis=b["delta_s_vis"].to(device),
                        vis_pad_mask=b["vis_pad_mask"].to(device),
                        L_vis_cond=b["L_vis_cond"].to(device) if mode == "full_freq" else None,
                        H_vis_cond=b["H_vis_cond"].to(device) if mode == "full_freq" else None
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

        mean_val = np.mean(val_losses) if val_losses else avg_train_loss
        val_loss_history.append(float(mean_val))

        if mean_val < (best_val_loss - min_delta):
            best_val_loss = mean_val
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # Progress is persisted every epoch, without changing checkpoint policy.
        progress = {
            "config": config_name, "seed": seed, "epoch": epoch + 1,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "best_epoch": best_epoch, "best_val_loss": best_val_loss,
            "batch_size": batch_size, "optimize_data": optimize_data,
        }
        progress_dir = os.path.join(output_dir, "progress")
        os.makedirs(progress_dir, exist_ok=True)
        progress_path = os.path.join(progress_dir, f"{config_name}_seed_{seed}.json")
        with open(progress_path + ".tmp", "w") as f:
            json.dump(progress, f, indent=2, allow_nan=False)
        os.replace(progress_path + ".tmp", progress_path)
        print(f"[{config_name} seed={seed}] epoch={epoch+1} train={avg_train_loss:.6f} "
              f"val={mean_val:.6f} best={best_epoch} seconds={progress['epoch_seconds']:.1f}", flush=True)

        if patience is not None and patience_counter >= patience:
            print(f"[{config_name} Seed {seed}] Early stopping triggered at Epoch {epoch+1} (Best Epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f})")
            break

    train_time = time.time() - t_start

    # Load best checkpoint before inference
    model.load_state_dict(best_state_dict)
    model.eval()

    ckpt_path = os.path.join(ckpt_dir, f"{config_name}_seed_{seed}.pt")
    torch.save(best_state_dict, ckpt_path)

    # 6. Inference & Embedding Extraction Helper (Full Subgraph, M = empty)
    def extract_niche_embeddings(indices: np.ndarray) -> np.ndarray:
        eval_dataset = SubgraphDataset(
            center_indices=indices,
            A_global=A_global,
            A_cond_global=A_cond_arg,
            P_global=P_global,
            X_global=X_global,
            coords_global=coords_global,
            X_cond_global=X_cond_global,
            bundle_map=bundle_map,
            whitener=whitener,
            k_max=64,
            m_max=32,
            is_train=False,
            is_val_objective=False,
            data_cache=data_cache
        )
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_subgraphs, pin_memory=pin_memory)
        z_list = []
        with torch.no_grad():
            for b in iter_batches(eval_loader, prefetch=optimize_data):
                b = stage_batch(b, device, optimize_data)
                if is_sagepool:
                    z = model.encode(
                        b["X_center_true"].to(device),
                        b["X_vis"].to(device),
                        b["vis_pad_mask"].to(device)
                    )
                elif config_name == "Same-Encoder-MAE":
                    z = model.encoder(
                        X_vis=b["X_vis"].to(device),
                        delta_s_vis=b["delta_s_vis"].to(device),
                        vis_pad_mask=b["vis_pad_mask"].to(device),
                        L_vis_cond=b["L_vis_cond"].to(device) if mode == "full_freq" else None,
                        H_vis_cond=b["H_vis_cond"].to(device) if mode == "full_freq" else None
                    )
                else:
                    z = model.encode_niche(
                        X_vis=b["X_vis"].to(device),
                        delta_s_vis=b["delta_s_vis"].to(device),
                        vis_pad_mask=b["vis_pad_mask"].to(device),
                        L_vis_cond=b["L_vis_cond"].to(device) if mode == "full_freq" else None,
                        H_vis_cond=b["H_vis_cond"].to(device) if mode == "full_freq" else None
                    )
                z_np = z.cpu().numpy()
                assert np.isfinite(z_np).all(), "Non-finite embedding produced during evaluation!"
                z_list.append(z_np)
        return np.concatenate(z_list, axis=0)

    # 7. Unsupervised Resolution Selection on Validation Donors (Track B)
    z_val = extract_niche_embeddings(val_center_indices)
    best_res, stability_report = compute_unsupervised_clustering_stability(
        z_val,
        resolutions=[0.2, 0.4, 0.6, 0.8, 1.0],
        n_bootstraps=10,
        subsample_ratio=0.8,
        k_min=5,
        k_max=25,
        seed=seed
    )

    # Compute Validation Donor Macro ARI (for primary baseline pre-selection)
    niche_col = "niche" if "niche" in obs_df.columns else "cellcharter_cluster.k_10"
    val_donors = obs_df.iloc[val_center_indices]["donor_id"].unique()
    val_ari_list = []
    for d_id in val_donors:
        d_mask = (obs_df.iloc[val_center_indices]["donor_id"].values == d_id)
        d_val_idx = val_center_indices[d_mask]
        z_val_d = extract_niche_embeddings(d_val_idx)
        y_true_val_d = obs_df.iloc[d_val_idx][niche_col].values
        res_val_d = evaluate_track_b_leiden(z_val_d, y_true_val_d, resolution=best_res, seed=seed)
        val_ari_list.append(res_val_d["leiden_ari"])
    val_donor_macro_ari = float(np.mean(val_ari_list)) if val_ari_list else 0.0

    # 8. Evaluate Test Donors Blindly (Track B on ALL test donor cells) if eval_test=True
    donor_macro_ari = 0.0
    donor_median_ari = 0.0
    worst_donor_ari = 0.0
    test_donor_metrics = {}
    emb_path = None

    if eval_test and test_center_indices is not None and len(test_center_indices) > 0:
        test_donors = obs_df.iloc[test_center_indices]["donor_id"].unique()
        test_ari_list = []
        test_nmi_list = []

        for d_id in test_donors:
            donor_mask = (obs_df.iloc[test_center_indices]["donor_id"].values == d_id)
            d_local_indices = test_center_indices[donor_mask]
            z_test_d = extract_niche_embeddings(d_local_indices)
            y_true_d = obs_df.iloc[d_local_indices][niche_col].values

            n_classes = len(np.unique(y_true_d[~pd.isna(y_true_d)]))

            res_track_a = evaluate_track_a_oracle_k(z_test_d, y_true_d, n_clusters=max(n_classes, 2), seed=seed)
            res_track_b = evaluate_track_b_leiden(z_test_d, y_true_d, resolution=best_res, seed=seed)

            test_donor_metrics[d_id] = {
                "track_b_leiden_ari": res_track_b["leiden_ari"],
                "track_b_leiden_nmi": res_track_b["leiden_nmi"],
                "track_a_kmeans_ari": res_track_a["kmeans_ari"],
                "track_a_gmm_ari": res_track_a["gmm_ari"],
                "n_cells": len(d_local_indices),
                "n_clusters": res_track_b["n_clusters"]
            }
            test_ari_list.append(res_track_b["leiden_ari"])
            test_nmi_list.append(res_track_b["leiden_nmi"])

        donor_macro_ari = float(np.mean(test_ari_list))
        donor_median_ari = float(np.median(test_ari_list))
        worst_donor_ari = float(np.min(test_ari_list))

        emb_path = os.path.join(emb_dir, f"{config_name}_seed_{seed}_test_z.npy")
        z_test_all = extract_niche_embeddings(test_center_indices)
        np.save(emb_path, z_test_all)

    # 9. Zero-z Diagnostic (for Full-DiT on valid validation targets)
    zero_z_diag = None
    if config_name == "Full-Freq-DiT":
        val_sub_centers = val_center_indices[:min(1000, len(val_center_indices))]
        val_diag_dataset = SubgraphDataset(
            center_indices=val_sub_centers,
            A_global=A_global,
            P_global=P_global,
            X_global=X_global,
            coords_global=coords_global,
            whitener=whitener,
            k_max=64,
            m_max=32,
            rho=0.5,
            p_center_mask=0.30,
            is_train=True,
            seed=seed,
            data_cache=diagnostic_cache
        )
        val_diag_loader = DataLoader(val_diag_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_subgraphs, pin_memory=pin_memory)
        loss_norm_list, loss_zero_list = [], []
        for b in iter_batches(val_diag_loader, prefetch=optimize_data):
            b = stage_batch(b, device, optimize_data)
            if not b["target_mask"].any():
                continue
            diag = model.evaluate_zero_z_loss(
                X_vis=b["X_vis"].to(device),
                delta_s_vis=b["delta_s_vis"].to(device),
                vis_pad_mask=b["vis_pad_mask"].to(device),
                H_0_target=b["H_0_target"].to(device),
                delta_s_target=b["delta_s_target"].to(device),
                target_mask=b["target_mask"].to(device),
                L_vis_cond=b["L_vis_cond"].to(device),
                H_vis_cond=b["H_vis_cond"].to(device)
            )
            loss_norm_list.append(diag["loss_normal"])
            loss_zero_list.append(diag["loss_zero_z"])

        mean_norm = float(np.mean(loss_norm_list)) if loss_norm_list else 1.0
        mean_zero = float(np.mean(loss_zero_list)) if loss_zero_list else 1.0
        zero_z_diag = {
            "loss_normal": mean_norm,
            "loss_zero_z": mean_zero,
            "loss_increase_pct": float(((mean_zero - mean_norm) / max(mean_norm, 1e-6)) * 100.0)
        }

    run_result = {
        "config": config_name,
        "seed": seed,
        "train_time_sec": train_time,
        "best_epoch": best_epoch,
        "total_epochs_trained": len(train_loss_history),
        "best_val_objective_loss": float(best_val_loss),
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "val_donor_macro_ari": val_donor_macro_ari,
        "best_resolution": best_res,
        "donor_macro_ari": donor_macro_ari,
        "donor_median_ari": donor_median_ari,
        "worst_donor_ari": worst_donor_ari,
        "per_donor_metrics": test_donor_metrics,
        "zero_z_diagnostic": zero_z_diag,
        "whitening_hash": white_hash,
        "pca_hash": pca_hash,
        "checkpoint_path": ckpt_path,
        "embedding_path": emb_path
    }
    run_result["data_optimization"] = {
        "enabled": optimize_data, "pin_memory": pin_memory, "num_workers": 0,
        "prefetch_batches": 1 if optimize_data else 0,
        "batch_size": batch_size,
        "cache": data_cache.stats() if data_cache is not None else None,
    }

    print(f"[{config_name} Seed {seed}] Best Epoch: {best_epoch}/{len(train_loss_history)} | Val Loss: {best_val_loss:.4f} | Val Macro ARI: {val_donor_macro_ari:.4f} | Time: {train_time:.1f}s")
    return run_result
