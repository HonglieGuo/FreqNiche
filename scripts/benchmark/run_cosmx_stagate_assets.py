"""Complete the missing CosMx STAGATE evaluation with reloadable assets.

Uses the official STAGATE module, its reconstruction MSE, Adam, clipping and
500 epochs. Disconnected physical samples contribute cell-weighted gradients
to ONE optimizer step per epoch. Sparse attention aggregation avoids the
official E-by-512 message allocation; equivalence is checked before training.
No subsampling, coordinate calibration, normalization or hyperparameter search.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines/STAGATE_pyG"))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax
from STAGATE_pyG.STAGATE import STAGATE
from src.benchmark.adapters.common import build_sample_isolated_spatial_graph, source_tree_sha256
from scripts.benchmark import run_cosmx_baselines as protocol

DEFAULT_OUT = ROOT / "baselines/Baselines_outputs/runs/cosmx_nsclc/2026-10-17/20261017_stagate"
CONTRACT = ROOT / "outputs/runs/freqniche/cosmx_nsclc/2026-09-17/20260917_cosmx_nsclc_final"
SEEDS = [42, 123, 2024]


def now():
    return datetime.now().astimezone().isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def torch_save(path, value):
    tmp = path.with_suffix(".tmp.pt")
    torch.save(value, tmp)
    os.replace(tmp, path)


class SparseAggregate(torch.autograd.Function):
    """Exact weighted sum; chunk the edge-weight derivative in backward."""
    @staticmethod
    def forward(ctx, x, weights, edges):
        # PyG flow is source_to_target; rows are targets, columns are sources.
        matrix = torch.sparse_coo_tensor(edges.flip(0), weights, (len(x), len(x)))
        ctx.save_for_backward(x, weights, edges)
        return torch.sparse.mm(matrix, x)

    @staticmethod
    def backward(ctx, grad):
        x, weights, edges = ctx.saved_tensors
        reverse = torch.sparse_coo_tensor(edges, weights, (len(x), len(x)))
        grad_x = torch.sparse.mm(reverse, grad)
        grad_weights = torch.empty_like(weights)
        for start in range(0, len(weights), 32768):
            stop = min(start + 32768, len(weights))
            src, dst = edges[:, start:stop]
            grad_weights[start:stop] = (x[src] * grad[dst]).sum(-1)
        return grad_x, grad_weights, None


def sparse_propagate(self, edge_index, x, alpha, size=None):
    src, dst = edge_index
    a_src, a_dst = alpha
    logits = a_src[src] if a_dst is None else a_src[src] + a_dst[dst]
    weights = softmax(torch.sigmoid(logits), dst, num_nodes=len(x[0]))
    self._alpha = weights
    weights = F.dropout(weights, p=self.dropout, training=self.training)
    # Official STAGATE uses one attention head without dropout.
    if self.heads != 1 or self.dropout != 0:
        raise RuntimeError("Sparse execution is validated only for the official one-head/no-dropout model")
    return SparseAggregate.apply(x[0][:, 0], weights[:, 0], edge_index).unsqueeze(1)


def efficient(model):
    for layer in (model.conv1, model.conv3):
        layer.propagate = MethodType(sparse_propagate, layer)
    return model


def equivalence(device):
    """Compare outputs, every gradient, and one clipped Adam update to upstream."""
    torch.manual_seed(917)
    x = torch.randn(27, 7, device=device)
    e1 = torch.randint(0, 13, (2, 65), device=device)
    e2 = torch.randint(13, 27, (2, 69), device=device)
    edges = torch.cat([e1, e2, torch.arange(27, device=device).repeat(2, 1)], 1)
    original = STAGATE([7, 11, 3]).to(device)
    sparse = efficient(copy.deepcopy(original))
    split = efficient(copy.deepcopy(original))
    original.train(); sparse.train(); split.train()
    oz, ox = original(x, edges)
    sz, sx = sparse(x, edges)
    torch.testing.assert_close(oz, sz, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(ox, sx, atol=2e-6, rtol=2e-5)
    F.mse_loss(ox, x).backward()
    F.mse_loss(sx, x).backward()
    for lo, hi in [(0, 13), (13, 27)]:
        select = (edges[0] >= lo) & (edges[0] < hi)
        _, out = split(x[lo:hi], edges[:, select] - lo)
        (F.mse_loss(out, x[lo:hi]) * ((hi - lo) / 27)).backward()
    grad_diff = 0.0
    for (_, a), (_, b), (_, c) in zip(original.named_parameters(), sparse.named_parameters(), split.named_parameters()):
        if a.grad is None:
            assert b.grad is None and c.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(a.grad, c.grad, atol=3e-6, rtol=3e-5)
            grad_diff = max(grad_diff, float((a.grad - c.grad).abs().max()))
    for model in (original, sparse, split):
        # Upstream decoder .data shares encoder storage: overlapping foreach
        # CUDA updates race. Sequential kernels preserve Adam's update order.
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4, foreach=False)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    for a, b, c in zip(original.parameters(), sparse.parameters(), split.parameters()):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(a, c, atol=3e-6, rtol=3e-5)
    return {"status": "passed", "device": str(device), "max_gradient_absolute_difference": grad_diff,
            "checked": ["embedding", "reconstruction", "all_parameter_gradients", "sample_weighted_full_batch_gradients", "clipped_Adam_update"]}


def binding():
    return {"runner_sha256": sha(__file__), "readout_sha256": sha(protocol.__file__),
            "official_source_sha256": source_tree_sha256(ROOT / "baselines/STAGATE_pyG"),
            "graph_source_sha256": sha(ROOT / "src/benchmark/adapters/common.py")}


def prepare(out):
    manifest = out / "prepared.json"
    if manifest.exists():
        value = json.loads(manifest.read_text())
        if value["code"] != binding():
            raise RuntimeError("Source changed since preparation; refusing silent reuse")
        for relative, digest in value["files"].items():
            if sha(out / relative) != digest:
                raise RuntimeError(f"Input changed: {relative}")
        if sha(protocol.DATASET_PATH) != value["h5ad_sha256"]:
            raise RuntimeError("Dataset changed")
        return value
    if (out / "inputs").exists():
        raise RuntimeError("Incomplete preparation exists; preserve it and choose a fresh output directory")
    protocol.AUDIT_MANIFEST_PATH = CONTRACT / "cosmx_data_audit_manifest.json"
    protocol.VAL_CENTERS_PATH = CONTRACT / "val_centers.json"
    protocol.TEST_CENTERS_PATH = CONTRACT / "test_centers.json"
    protocol.VAL_FOLDS_PATH = CONTRACT / "val_bootstrap_folds.json"
    a = ad.read_h5ad(protocol.DATASET_PATH)
    contracts = {name: json.loads((CONTRACT / name).read_text()) for name in
                 ["val_centers.json", "test_centers.json", "val_bootstrap_folds.json", "cosmx_data_audit_manifest.json"]}
    digest = protocol.verify_cosmx_baseline_preflight(a, contracts["val_centers.json"], contracts["test_centers.json"], contracts["val_bootstrap_folds.json"])
    for name, value in contracts.items():
        shutil.copyfile(CONTRACT / name, out / name)
        assert sha(out / name) == sha(CONTRACT / name), "Contract copy must be byte-identical"
    return _prepare_inputs(out, a, contracts, digest)


def _prepare_inputs(out, a, contracts, digest):
    a.obs[["donor_id", "sample_id", "split"]].to_csv(out / "cells.csv", index_label="cell_barcode")
    pd.DataFrame({"gene": a.var_names.astype(str)}).to_csv(out / "genes.csv", index=False)
    np.save(out / "spatial_um.npy", np.asarray(a.obsm["spatial_um"], dtype=np.float32))
    inputs = []
    groups = a.obs.sample_id.astype(str).to_numpy()
    for i, sample in enumerate(pd.unique(groups)):
        idx = np.flatnonzero(groups == sample)
        sub = a[idx].copy()
        sub.obsm["spatial"] = np.asarray(sub.obsm["spatial_um"], dtype=np.float32)
        graph = build_sample_isolated_spatial_graph(sub, sample_key="sample_id", radius=50.0)
        rows, cols = graph.nonzero()
        directory = out / "inputs" / f"sample_{i}"
        directory.mkdir(parents=True)
        x = sub.X.toarray() if sp.issparse(sub.X) else np.asarray(sub.X)
        np.save(directory / "x.npy", np.asarray(x, dtype=np.float32))
        np.save(directory / "edge_index.npy", np.asarray([rows, cols], dtype=np.int64))
        np.save(directory / "global_indices.npy", idx)
        sp.save_npz(directory / "graph.npz", graph)
        inputs.append({"directory": str(directory.relative_to(out)), "sample_id": sample, "n_cells": len(idx), "n_edges": graph.nnz})
        print(f"Prepared {sample}: {len(idx)} cells, {graph.nnz} edges", flush=True)
    value = {"created_at": now(), "h5ad_sha256": digest, "n_cells": a.n_obs, "n_genes": a.n_vars,
             "code": binding(), "samples": inputs,
             "config": {"seeds": SEEDS, "epochs": 500, "hidden_dims": [512, 30], "radius_um": 50.0,
                        "lr": 1e-4, "weight_decay": 1e-4, "gradient_clipping": 5.0, "Adam_foreach": False,
                        "input": "unchanged float32 expression; no normalization/scaling, matching controlled adapter",
                        "fit_scope": "all 765771 cells, physical-sample-isolated full cohort, transductive",
                        "coordinate_source": "spatial_um without Visium pitch calibration",
                        "execution": "shared official model; cell-weighted sample gradients; one Adam step per full epoch; exact sparse attention aggregation"}}
    value["files"] = {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file() and
                      ("inputs" in p.parts or p.name in ["cells.csv", "genes.csv", "spatial_um.npy", *contracts])}
    write(out / "prepared.json", value)
    return value


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_inputs(out, prepared, device):
    result = []
    for info in prepared["samples"]:
        directory = out / info["directory"]
        result.append((torch.from_numpy(np.load(directory / "x.npy")).to(device),
                       torch.from_numpy(np.load(directory / "edge_index.npy")).to(device),
                       np.load(directory / "global_indices.npy")))
    return result


def train_seed(out, prepared, inputs, seed, device):
    directory = out / f"seed_{seed}"
    directory.mkdir(exist_ok=True)
    completed = directory / "training.json"
    if completed.exists():
        value = json.loads(completed.read_text())
        for file, digest in value["files"].items():
            if sha(directory / file) != digest:
                raise RuntimeError(f"Corrupt trained asset: {directory / file}")
        return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = efficient(STAGATE([prepared["n_genes"], 512, 30]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4, foreach=False)
    checkpoint = directory / "checkpoint.pt"
    losses = []
    start_epoch = 0
    elapsed_before = 0.0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        start_epoch, losses, elapsed_before = saved["epoch"], saved["losses"], saved["elapsed_sec"]
        print(f"Resume seed {seed} at epoch {start_epoch}", flush=True)
    started = time.monotonic()
    for epoch in range(start_epoch + 1, 501):
        model.train(); optimizer.zero_grad()
        total = 0.0
        for x, edges, indices in inputs:
            z, recon = model(x, edges)
            loss = F.mse_loss(recon, x) * (len(indices) / prepared["n_cells"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss seed={seed}, epoch={epoch}")
            loss.backward()
            total += float(loss.detach())
            del z, recon, loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(total)
        elapsed = elapsed_before + time.monotonic() - started
        write(out / "run_state.json", {"status": "running", "phase": "training", "seed": seed,
              "epoch": epoch, "epochs": 500, "loss": total, "seed_elapsed_sec": elapsed,
              "completed_seeds": [s for s in SEEDS if (out / f"seed_{s}/training.json").exists()], "updated_at": now()})
        if epoch == 1 or epoch % 10 == 0:
            print(f"seed={seed} epoch={epoch}/500 mse={total:.7f} elapsed={elapsed:.1f}s", flush=True)
        if epoch % 25 == 0 or epoch == 1 or epoch == 500:
            torch_save(checkpoint, {"model": cpu_state(model), "optimizer": optimizer.state_dict(),
                                   "epoch": epoch, "losses": losses, "elapsed_sec": elapsed})
    model.eval()
    embedding = np.empty((prepared["n_cells"], 30), dtype=np.float32)
    with torch.no_grad():
        for x, edges, indices in inputs:
            z, _ = model(x, edges)
            embedding[indices] = z.cpu().numpy()
    if not np.isfinite(embedding).all():
        raise RuntimeError("Nonfinite embedding")
    torch_save(directory / "model.pt", {"state_dict": cpu_state(model), "hidden_dims": [prepared["n_genes"], 512, 30], "seed": seed, "epochs": 500})
    np.save(directory / "full_embedding.npy", embedding)
    for phase, contract in [("val", "val_centers.json"), ("test", "test_centers.json")]:
        idx = json.loads((out / contract).read_text())["global_indices"]
        np.save(directory / f"{phase}_embedding.npy", embedding[idx])
    write(directory / "loss_history.json", losses)
    write(completed, {"seed": seed, "completed_at": now(), "elapsed_sec": elapsed, "epochs": 500,
                      "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
                      "files": {name: sha(directory / name) for name in ["model.pt", "checkpoint.pt", "full_embedding.npy", "val_embedding.npy", "test_embedding.npy", "loss_history.json"]}})
    del model, optimizer


def evaluate(out):
    folds = json.loads((out / "val_bootstrap_folds.json").read_text())
    # Freeze ALL validation resolutions before the first test readout.
    for seed in SEEDS:
        directory = out / f"seed_{seed}"
        if (directory / "selection.json").exists():
            continue
        write(out / "run_state.json", {"status": "running", "phase": "validation_selection", "seed": seed, "updated_at": now()})
        r, report = protocol.select_unsupervised_resolution(np.load(directory / "val_embedding.npy"), folds, seed=seed)
        write(directory / "selection.json", {"seed": seed, "r_star": r, "selection_report": report,
              "val_embedding_sha256": sha(directory / "val_embedding.npy"), "frozen_at": now()})
        print(f"seed={seed} validation frozen r={r}", flush=True)
    cohort = json.loads((out / "test_centers.json").read_text())
    # The evaluator sees reference labels only after all validation choices froze.
    original = ad.read_h5ad(protocol.DATASET_PATH, backed="r")
    results = []
    for seed in SEEDS:
        directory = out / f"seed_{seed}"
        selected = json.loads((directory / "selection.json").read_text())
        if (directory / "test_result.json").exists():
            results.append(json.loads((directory / "test_result.json").read_text()))
            continue
        write(out / "run_state.json", {"status": "running", "phase": "test_readout", "seed": seed, "updated_at": now()})
        metrics, assignments = protocol.evaluate_test_clustering(np.load(directory / "test_embedding.npy"), selected["r_star"], cohort, original, "stagate", seed)
        assignments["method_variant"] = "STAGATE"
        assignments.to_csv(directory / "assignments.csv", index=False)
        metrics.update({"method_variant": "STAGATE", "selection_sha256": sha(directory / "selection.json"),
                        "test_embedding_sha256": sha(directory / "test_embedding.npy"), "assignments_sha256": sha(directory / "assignments.csv"), "evaluated_at": now()})
        write(directory / "test_result.json", metrics)
        results.append(metrics)
        print(f"seed={seed} ARI={metrics['ari']:.8f} NMI={metrics['nmi']:.8f}", flush=True)
    original.file.close()
    summary = {"method": "STAGATE", "dataset": "cosmx_nsclc", "seeds": SEEDS, "n_test_centers": 10000,
               "test_donor": "Lung9", "seed_results": results, "created_at": now()}
    for metric in ["ari", "nmi", "reference_niche_asw"]:
        values = [result[metric] for result in results]
        summary[f"mean_{metric}"] = float(np.mean(values))
        summary[f"std_{metric}"] = float(np.std(values, ddof=1))
    write(out / "summary.json", summary)


def audit(out, prepared, device):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    inputs = load_inputs(out, prepared, device)
    records = []
    if (out / "asset_audit.json").exists():
        old_audit = json.loads((out / "asset_audit.json").read_text())
        for relative, digest in old_audit["files"].items():
            if sha(out / relative) != digest:
                raise RuntimeError(f"Published asset changed: {relative}")
    for seed in SEEDS:
        directory = out / f"seed_{seed}"
        trained = json.loads((directory / "training.json").read_text())
        for relative, digest in trained["files"].items():
            if sha(directory / relative) != digest:
                raise RuntimeError(f"Trained asset changed: {seed}/{relative}")
        selection = json.loads((directory / "selection.json").read_text())
        assert selection["val_embedding_sha256"] == sha(directory / "val_embedding.npy")
        saved = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        model = efficient(STAGATE(saved["hidden_dims"]).to(device))
        model.load_state_dict(saved["state_dict"], strict=True); model.eval()
        stored = np.load(directory / "full_embedding.npy", mmap_mode="r")
        max_diff = 0.0
        with torch.no_grad():
            for x, edges, idx in inputs:
                z, _ = model(x, edges)
                actual = z.cpu().numpy()
                np.testing.assert_allclose(actual, stored[idx], atol=1e-5, rtol=1e-5)
                max_diff = max(max_diff, float(np.max(np.abs(actual - stored[idx]))))
        for name, contract in [("val", "val_centers.json"), ("test", "test_centers.json")]:
            idx = json.loads((out / contract).read_text())["global_indices"]
            np.testing.assert_array_equal(stored[idx], np.load(directory / f"{name}_embedding.npy"))
        assignments = pd.read_csv(directory / "assignments.csv")
        cohort = json.loads((out / "test_centers.json").read_text())
        assert assignments.cell_barcode.tolist() == cohort["cell_barcodes"]
        result = json.loads((directory / "test_result.json").read_text())
        assert result["selection_sha256"] == sha(directory / "selection.json")
        assert result["test_embedding_sha256"] == sha(directory / "test_embedding.npy")
        assert result["assignments_sha256"] == sha(directory / "assignments.csv")
        for metric, function in [("ari", adjusted_rand_score), ("nmi", normalized_mutual_info_score)]:
            actual = function(assignments.reference_niche, assignments.predicted_cluster)
            assert abs(actual - result[metric]) < 1e-12
        records.append({"seed": seed, "reload_max_abs_error": max_diff, "metric_recompute": "passed"})
        del model
    summary = json.loads((out / "summary.json").read_text())
    for metric in ["ari", "nmi"]:
        values = [json.loads((out / f"seed_{seed}/test_result.json").read_text())[metric] for seed in SEEDS]
        assert abs(summary[f"mean_{metric}"] - np.mean(values)) < 1e-12
        assert abs(summary[f"std_{metric}"] - np.std(values, ddof=1)) < 1e-12
    # Every asset is hash bound, including source, input, checkpoint and outputs.
    value = {"status": "passed", "verified_at": now(), "model_bundles": 3, "records": records,
             "files": {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file() and p.suffix in [".npy", ".npz", ".pt", ".csv", ".json"] and p.name not in ["asset_audit.json", "run_state.json"]}}
    write(out / "asset_audit.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.self_test:
        print(json.dumps(equivalence(device), indent=2)); return
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "controller.lock"
    handle = os.open(lock, os.O_CREAT | os.O_RDWR)
    import fcntl
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.resume and not args.audit_only and (out / "run_state.json").exists():
        raise RuntimeError("Existing run; use --resume to verify and resume, never overwrite")
    try:
        if args.audit_only:
            prepared = prepare(out)
            print(json.dumps(audit(out, prepared, device), indent=2)[:1500]); return
        write(out / "run_state.json", {"status": "running", "phase": "preflight", "updated_at": now(), "pid": os.getpid()})
        write(out / "equivalence.json", equivalence(device))
        prepared = prepare(out)
        write(out / "environment.json", {"torch": torch.__version__, "numpy": np.__version__, "device": str(device),
              "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None, "code": binding(), "recorded_at": now()})
        inputs = load_inputs(out, prepared, device)
        for seed in SEEDS:
            train_seed(out, prepared, inputs, seed, device)
        del inputs
        torch.cuda.empty_cache()
        evaluate(out)
        audit(out, prepared, device)
        write(out / "run_state.json", {"status": "complete", "completed_seeds": SEEDS, "completed_at": now()})
    except Exception:
        write(out / "run_state.json", {"status": "failed", "error": traceback.format_exc(), "failed_at": now()})
        raise
    finally:
        os.close(handle)


if __name__ == "__main__":
    main()
