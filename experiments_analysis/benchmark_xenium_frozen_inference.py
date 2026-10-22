"""Measure frozen Xenium encoder inference on the historical 1k/5k/10k patches.

No fitting, clustering, checkpoint changes or diffusion forward calls. Each
scale/checkpoint runs in a fresh process. CUDA allocator peaks are not the
historical baseline's nvidia-smi process-memory measurements.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / "outputs/runs/freqniche/xenium_prime5k/2026-10-15/20261015-194823-263849_xenium_prime5k_014-test"
DATA = ROOT / "datasets/h5ad/xenium_ffpe_lungbreast_gse311609/samples/GSM9509134_NSCLC_5k_L1_L1.h5ad"
DEFAULT_OUT = ROOT / "outputs/analysis/xenium_inference_scaling_20261021"
SCALES = (1000, 5000, 10000)
SEEDS = (42, 123, 2024)
BATCH = 64


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def source_paths():
    return [DATA, RUN / "model_config.json", RUN / "run_manifest.json",
            RUN / "xenium_test_evaluation_report.json",
            *[RUN / f"checkpoints/xenium_freqniche_trial24_seed_{s}.pt" for s in SEEDS],
            *[ROOT / p for p in ["src/freqniche_st/models/encoder.py",
                                 "src/freqniche_st/models/freqniche_model.py",
                                 "src/freqniche_st/dataset.py", "src/freqniche_st/subgraph.py",
                                 "src/freqniche_st/graph.py", "src/freqniche_st/model_config.py"]]]


def verify_sources():
    report = json.loads((RUN / "xenium_test_evaluation_report.json").read_text())
    assert sha(DATA) == report["bound_input_assets"]["test_h5ad"]["sha256"]
    for seed in SEEDS:
        assert sha(RUN / f"checkpoints/xenium_freqniche_trial24_seed_{seed}.pt") == report["bound_input_assets"]["checkpoints"][str(seed)]["sha256"]
    return {str(p.relative_to(ROOT)): sha(p) for p in source_paths()}


def prepare(out):
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp
    from scipy.spatial import cKDTree
    from src.freqniche_st.graph import build_global_spatial_graph

    a = ad.read_h5ad(DATA, backed="r")
    assert a.shape == (67705, 5001)
    coords = np.asarray(a.obsm["spatial"], dtype=np.float32)
    tree = cKDTree(coords)
    center = np.median(coords, axis=0, keepdims=True)
    times = {}
    for n in SCALES:
        start = time.perf_counter()
        folder = out / f"patch_{n}"
        folder.mkdir()
        # Same ordered, compact L1 patch selection as run_xenium_baselines.py.
        idx = np.sort(tree.query(center, k=n)[1][0])
        x = a.X[idx]
        x = x.toarray().astype(np.float32) if sp.issparse(x) else np.asarray(x, dtype=np.float32)
        x = np.log1p(x / np.maximum(x.sum(axis=1, keepdims=True), 1.0) * 1e4)
        p, graph = build_global_spatial_graph(coords[idx], np.array(["L1_patch"] * n), radius=50.0, sigma=50.0 / 3.0)
        np.save(folder / "indices.npy", idx)
        np.save(folder / "expression.npy", x)
        np.save(folder / "coords.npy", coords[idx])
        (folder / "barcodes.json").write_text(json.dumps(a.obs_names[idx].tolist()))
        sp.save_npz(folder / "A.npz", graph)
        sp.save_npz(folder / "P.npz", p)
        assert np.allclose(np.asarray(p.sum(axis=1)).ravel(), 1, atol=1e-5)
        times[str(n)] = {"preparation_sec": time.perf_counter() - start, "graph_nnz": graph.nnz}
        print(f"PREPARED {n} cells", flush=True)
    a.file.close()
    return times


def load_encoder(seed):
    import torch
    from src.freqniche_st.model_config import load_model_config, build_freqniche_model

    cfg = load_model_config(output_dir=RUN)
    full = build_freqniche_model(5001, cfg)
    state = torch.load(RUN / f"checkpoints/xenium_freqniche_trial24_seed_{seed}.pt", map_location="cpu", weights_only=True)
    state = state.get("model_state_dict", state)
    full.load_state_dict(state, strict=True)
    encoder = full.encoder
    del full, state
    encoder.requires_grad_(False).eval()
    return encoder, cfg


def get_loader(out, n, cfg):
    import numpy as np
    import scipy.sparse as sp
    from torch.utils.data import DataLoader
    from src.freqniche_st.dataset import SubgraphDataset, collate_subgraphs

    folder = out / f"patch_{n}"
    ds = SubgraphDataset(np.arange(n), sp.load_npz(folder / "A.npz"),
                         sp.load_npz(folder / "P.npz"), np.load(folder / "expression.npy"),
                         np.load(folder / "coords.npy"), k_max=cfg["k_max"],
                         m_max=cfg["m_max"], whitener=None, is_train=False, is_val_objective=False)
    return DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0,
                      collate_fn=collate_subgraphs, pin_memory=True)


def forward(encoder, batch):
    return encoder(**{key: batch[key].to("cuda") for key in
                      ("X_vis", "delta_s_vis", "vis_pad_mask", "L_vis_cond", "H_vis_cond")})


def worker(out, n, seed):
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits

    assert torch.cuda.is_available(), "CUDA is required for this measurement"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    with threadpool_limits(limits=1):
        encoder, cfg = load_encoder(seed)
        encoder.to("cuda")
        loader = get_loader(out, n, cfg)
        # Warm-up is excluded from reported steady-state inference time.
        batch = next(iter(loader))
        with torch.no_grad():
            warmup = forward(encoder, batch).cpu().numpy().copy()
        del batch
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_alloc = torch.cuda.memory_allocated()
        start = time.perf_counter()
        rows = []
        with torch.no_grad():
            for batch in loader:
                rows.append(forward(encoder, batch).cpu().numpy())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2**20
        reserved = torch.cuda.max_memory_reserved() / 2**20
        emb = np.concatenate(rows)
        assert emb.shape == (n, cfg["d_model"]) and np.isfinite(emb).all()
        assert np.allclose(emb[:len(warmup)], warmup, atol=1e-6, rtol=1e-5)
        dest = out / f"patch_{n}" / f"seed_{seed}"
        dest.mkdir()
        np.save(dest / "embeddings.npy", emb)
        record = {"cells": n, "seed": seed, "status": "passed", "batch_size": BATCH,
                  "embedding_dim": cfg["d_model"], "runtime_sec": elapsed,
                  "cells_per_sec": n / elapsed, "cuda_peak_allocated_mib": peak,
                  "cuda_peak_reserved_mib": reserved,
                  "cuda_resident_encoder_mib": baseline_alloc / 2**20,
                  "warmup_max_abs_error": float(np.max(np.abs(emb[:len(warmup)] - warmup))),
                  "embedding_sha256": sha(dest / "embeddings.npy"), "model_config": cfg,
                  "hardware": torch.cuda.get_device_name(), "torch": torch.__version__,
                  "cuda": torch.version.cuda, "python": platform.python_version(),
                  "cpu_threads": 1, "pid": os.getpid()}
        write_json(dest / "measurement.json", record)
        print(f"MEASURED n={n} seed={seed}: {peak:.2f} MiB, {elapsed:.3f} s", flush=True)


def audit(out):
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits

    manifest = json.loads((out / "telemetry.json").read_text())
    assert verify_sources() == manifest["inputs"]
    torch.set_num_threads(1)
    checks = []
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            encoder, cfg = load_encoder(seed)
            encoder.to("cuda")
            for n in SCALES:
                folder = out / f"patch_{n}" / f"seed_{seed}"
                record = json.loads((folder / "measurement.json").read_text())
                emb = np.load(folder / "embeddings.npy")
                assert sha(folder / "embeddings.npy") == record["embedding_sha256"]
                assert emb.shape == (n, cfg["d_model"]) and np.isfinite(emb).all()
                assert record["cuda_peak_reserved_mib"] >= record["cuda_peak_allocated_mib"] > 0
                loader = get_loader(out, n, cfg)
                with torch.no_grad():
                    replay = forward(encoder, next(iter(loader))).cpu().numpy()
                assert np.allclose(replay, emb[:len(replay)], atol=1e-6, rtol=1e-5)
                checks.append({"cells": n, "seed": seed, "max_abs_error": float(np.max(np.abs(replay - emb[:len(replay)])))})
            del encoder
            torch.cuda.empty_cache()
    assert len(checks) == 9
    write_json(out / "audit.json", {"status": "passed", "independent_process": True,
                                   "checkpoint_strict_load": True, "checks": checks})
    print("AUDIT PASSED: all 9 saved embeddings independently replayed", flush=True)


def run(out):
    import numpy as np

    out.mkdir(parents=True, exist_ok=False)
    inputs = verify_sources()
    write_json(out / "status.json", {"status": "running", "completed": []})
    prep = prepare(out)
    measurements = []
    for n in SCALES:
        for seed in SEEDS:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--output-dir", str(out),
                   "--worker", "--cells", str(n), "--seed", str(seed)]
            with (out / f"worker_{n}_{seed}.log").open("w") as f:
                subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
            record = json.loads((out / f"patch_{n}/seed_{seed}/measurement.json").read_text())
            measurements.append(record)
            write_json(out / "status.json", {"status": "running", "completed": [(m["cells"], m["seed"]) for m in measurements]})
            print(f"DONE {n}/{seed}: {record['runtime_sec']:.3f}s, {record['cuda_peak_allocated_mib']:.2f} MiB", flush=True)
    summary = []
    for n in SCALES:
        records = [m for m in measurements if m["cells"] == n]
        row = {"cells": n, "n_checkpoints": 3}
        for key in ("runtime_sec", "cuda_peak_allocated_mib", "cuda_peak_reserved_mib", "cells_per_sec"):
            row[key + "_mean"] = float(np.mean([m[key] for m in records]))
            row[key + "_sd"] = float(np.std([m[key] for m in records], ddof=1))
        summary.append(row)
    assert verify_sources() == inputs
    write_json(out / "telemetry.json", {"status": "complete", "method": "FreqNiche", "inputs": inputs,
        "measurement_script_sha256": sha(Path(__file__)), "scales": SCALES, "seeds": SEEDS,
        "protocol": {"stage": "frozen_encoder_inference", "batch_size": BATCH, "precision": "float32",
            "spatial_context": "same compact L1 patches as historical baseline smoke; patch-only graph",
            "timed": "CPU subgraph extraction and dual-frequency filtering, batch collation, H2D, encoder forward and D2H",
            "excluded": "checkpoint loading, data I/O, graph construction, warmup, output serialization, clustering, training and diffusion",
            "memory": "torch.cuda.max_memory_allocated; resident encoder included; reserved peak also recorded",
            "old_baseline_memory_comparable": False,
            "comparison_note": "historical baselines used process-level nvidia-smi and fit_transform; no matched controlled speed/memory comparison claimed"},
        "preparation": prep, "measurements": measurements, "summary": summary})
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    write_json(out / "status.json", {"status": "complete", "measurements": len(measurements), "audit": "pending"})
    print("COMPLETE: 9 measurements saved; independent audit pending", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--cells", type=int, choices=SCALES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    try:
        if args.worker:
            worker(args.output_dir, args.cells, args.seed)
        elif args.audit_only:
            audit(args.output_dir)
            write_json(args.output_dir / "status.json", {"status": "complete", "measurements": 9, "audit": "passed"})
        else:
            run(args.output_dir)
    except Exception as exc:
        if not args.worker and args.output_dir.exists():
            write_json(args.output_dir / "failure.json", {"error": repr(exc)})
        raise
