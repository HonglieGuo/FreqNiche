#!/usr/bin/env python3
"""Recover saved Human Skin Validation embeddings, then resume pending seeds.

--recover-only never trains a model. It requires a checkpoint that already
reached early stopping or its epoch limit. The normal route continues the same
dated run and records its metrics once all requested seeds have completed.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from src.freqniche_st.model_config import internal_from_public, load_model_config
from src.freqniche_st.orchestration import RunRequest, atomic_json, finalize_run, pipeline_commands
from src.freqniche_st.validation_worker import file_sha256, isolated_validation

VAL_BARCODE_SHA256 = "b37d6c6a4e2240eb5082be739ca2347f2d55fa6afbe7d36284b03d73ace051e9"
VAL_DONORS = {"D004", "D077", "D083", "D107"}


def load_run(run_dir: Path) -> tuple[RunRequest, dict]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    req = manifest["request"]
    if req["dataset"] != "human_skin" or req["stages"] != ["prepare", "train"]:
        raise ValueError("This resume entry point only accepts Human Skin Train/Validation runs")
    if not req["seeds"] or len(set(req["seeds"])) != len(req["seeds"]) or set(req["seeds"]) - {42, 123, 2024}:
        raise ValueError("Invalid frozen seed list")
    config = load_model_config(run_dir / "model_config.json", output_dir=run_dir)
    if config != internal_from_public(req["model_config"]):
        raise RuntimeError("Run manifest/model configuration mismatch")
    request = RunRequest(
        dataset=req["dataset"], seeds=tuple(req["seeds"]), device=req["device"],
        max_epochs=req["max_epochs"], patience=req["patience"],
        batch_size=req["batch_size"], min_delta=req["min_delta"],
        stages=tuple(req["stages"]), tag=req["tag"], model_config=req["model_config"],
    )
    expected = [{"step": name, "argv": argv} for name, argv in pipeline_commands(request, run_dir)]
    if manifest["commands"] != expected:
        raise RuntimeError("Frozen commands do not match this run; refusing to execute arbitrary manifest commands")
    return request, manifest


def selected_validation_metadata() -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Read barcodes and selected obs columns only; never open expression/coordinates."""
    import h5py
    import pandas as pd

    path = ROOT / "outputs/contracts/human_skin/validation_barcodes_8k.csv"
    if file_sha256(path) != VAL_BARCODE_SHA256:
        raise RuntimeError("Frozen Validation barcodes changed")
    barcodes = pd.read_csv(path)["cell_barcode"].astype(str).to_numpy()
    if len(barcodes) != 8000 or len(np.unique(barcodes)) != len(barcodes):
        raise RuntimeError("Invalid Validation center list")
    data = ROOT / "datasets/h5ad/human_skin_merfish/human_skin_merfish.h5ad"
    with h5py.File(data, "r") as handle:
        obs = handle["obs"]
        index_key = obs.attrs["_index"]
        if isinstance(index_key, bytes):
            index_key = index_key.decode()
        indices = pd.Index(obs[index_key].asstr()[:]).get_indexer(barcodes)
        if np.any(indices < 0):
            raise RuntimeError("Validation barcode absent from H5AD")
        order = np.argsort(indices)
        restore = np.argsort(order)

        def column(name: str) -> np.ndarray:
            node = obs[name]
            if isinstance(node, h5py.Group):
                categories = node["categories"].asstr()[:]
                codes = np.asarray(node["codes"][indices[order]])[restore]
                if np.any(codes < 0):
                    raise RuntimeError(f"Missing Validation {name}")
                return categories[codes]
            return node.asstr()[indices[order]][restore]

        donors, truth = column("donor_id"), column("niche")
        if "split" in obs and not np.isin(column("split"), ["validation", "val"]).all():
            raise RuntimeError("Non-Validation cells in the frozen center list")
    if set(donors) != VAL_DONORS or any(np.sum(donors == donor) != 2000 for donor in VAL_DONORS):
        raise RuntimeError("Validation donor/count contract mismatch")
    return barcodes, donors, truth, file_sha256(path)


def verify_result(folder: Path, seed: int, request: RunRequest) -> dict:
    result = json.loads((folder / "eval_result.json").read_text(encoding="utf-8"))
    expected = internal_from_public(request.model_config)
    if result.get("seed") != seed or result.get("trial_id") != 24:
        raise RuntimeError("Completed result identity mismatch")
    if any(result.get("config", {}).get(key) != value for key, value in expected.items()):
        raise RuntimeError("Completed result model configuration mismatch")
    for name, key in [("best_val_model.pt", "best_val_model_sha256"), ("val_embeddings.npy", "val_embeddings_sha256")]:
        if file_sha256(folder / name) != result.get(key):
            raise RuntimeError(f"Completed result asset hash mismatch: {name}")
    if not 1 <= result["best_val_epoch"] <= result["stopped_epoch"] <= request.max_epochs:
        raise RuntimeError("Completed result epoch contract mismatch")
    if not np.isfinite(result["donor_macro_ari"]) or set(result["donor_aris"]) != VAL_DONORS:
        raise RuntimeError("Invalid completed Validation metrics")
    return result


def recover_validation(run_dir: Path, seed: int, request: RunRequest) -> dict:
    import torch
    from sklearn.metrics import adjusted_rand_score

    folder = run_dir / f"trial_24_seed_{seed}"
    state = torch.load(folder / "resume_state_last.pt", map_location="cpu", weights_only=False)
    if state.get("seed") != seed or state.get("trial_id") != 24:
        raise RuntimeError("Checkpoint identity mismatch")
    config = internal_from_public(request.model_config)
    if any(state["trial_config"].get(key) != value for key, value in config.items()):
        raise RuntimeError("Checkpoint model configuration mismatch")
    for key in ("max_epochs", "patience", "batch_size", "min_delta"):
        if state.get(key) != getattr(request, key):
            raise RuntimeError(f"Checkpoint training-budget mismatch: {key}")
    if not (state["patience_counter"] >= request.patience or state["epoch"] >= request.max_epochs):
        raise RuntimeError("Training is incomplete; Validation-only recovery cannot skip training")
    best_model = folder / "best_val_model.pt"
    if file_sha256(best_model) != state.get("best_val_model_sha256"):
        raise RuntimeError("Best checkpoint hash mismatch")
    embedding = folder / "val_embeddings.npy"
    z = np.load(embedding, mmap_mode="r")
    if z.shape != (8000, config["d_model"]) or not np.isfinite(z).all():
        raise RuntimeError("Invalid saved Validation embedding")
    _, donors, truth, barcode_sha = selected_validation_metadata()
    print(f"[RECOVER] seed={seed}, completed training at epoch {state['epoch']}; evaluating saved embedding only", flush=True)
    readout_path = folder / "validation_readout.json"
    readout = isolated_validation(embedding, seed, readout_path)
    pred = np.asarray(readout["pred_clusters"])
    aris = {donor: float(adjusted_rand_score(truth[donors == donor], pred[donors == donor])) for donor in sorted(VAL_DONORS)}
    r_star = readout["r_star"]
    times = state.get("epoch_times", [])
    masks = state.get("actual_mask_ratios", [])
    result = {
        "trial_id": 24, "seed": seed, "desc": state["trial_config"]["desc"],
        "best_val_epoch": state["best_val_epoch"], "stopped_epoch": state["epoch"],
        "stopped_early": state["patience_counter"] >= request.patience,
        "best_val_loss": state["best_val_loss"], "r_star": r_star,
        "stability_score": readout["stability"]["stability_scores"][str(r_star)],
        "donor_macro_ari": float(np.mean(list(aris.values()))), "worst_donor_ari": min(aris.values()),
        "donor_ari_std_within_donors": float(np.std(list(aris.values()), ddof=1)), "donor_aris": aris,
        "mean_epoch_time_sec": float(np.mean(times)) if times else None,
        "total_train_time_sec": float(np.sum(times)) if times else None,
        "peak_gpu_mem_mb": None,  # This measurement was not saved before the crash.
        "actual_mask_ratio_mean": float(np.mean(masks)) if masks else None,
        "expected_mask_ratio": state["trial_config"]["rho"], "config": state["trial_config"],
        "graph_fingerprint": state["graph_fingerprint"], "whitener_fingerprint": state["whitener_fingerprint"],
        "train_manifest_hash": state["train_manifest_hash"], "top3_selection_hash": state.get("top3_selection_hash"),
        "best_val_model_sha256": file_sha256(best_model), "val_embeddings_sha256": file_sha256(embedding),
        "runner_code_sha256": state["runner_code_sha256"],  # Preserve the original training source identity.
        "validation_readout_sha256": file_sha256(readout_path), "validation_barcodes_sha256": barcode_sha,
        "recovery_runner_sha256": file_sha256(Path(__file__)), "recovered_validation_only": True,
        "embedding_binding_created_during_recovery": True,
        "recovered_at": datetime.now().astimezone().isoformat(),
    }
    atomic_json(folder / "eval_result.json", result)
    print(f"[RECOVERED] seed={seed}, Validation macro ARI={result['donor_macro_ari']:.6f}, r*={r_star}", flush=True)
    return result


def main() -> None:
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--recover-only", action="store_true", help="Only evaluate already-finished training; do not start pending seeds")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_dir = (ROOT / args.run_dir).resolve()
    run_dir.relative_to(ROOT / "outputs/runs/freqniche")
    request, manifest = load_run(run_dir)
    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "seeds": request.seeds, "recover_only": args.recover_only,
                          "stages": request.stages, "completed_steps": manifest.get("completed_steps", [])}, indent=2))
        return
    lock = run_dir / ".resume.lock"
    # Kernel locks are released even after SIGSEGV/SIGKILL. Keep the lock inode
    # in place so another process can never bypass the lock by recreating it.
    import fcntl
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise RuntimeError(f"Another recovery/resume process is using {run_dir}")
    try:
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        if manifest["status"] == "complete":
            for seed in request.seeds:
                verify_result(run_dir / f"trial_24_seed_{seed}", seed, request)
            print("[ALREADY COMPLETE] Verified existing results; no training or duplicate ledger entries.", flush=True)
            return
        snapshot = run_dir / "run_manifest_before_recovery.json"
        if not snapshot.exists():
            atomic_json(snapshot, manifest)
        completed = {entry["step"] for entry in manifest.get("completed_steps", [])}
        if "prepare_contracts" not in completed:
            raise RuntimeError("Run preparation did not complete; cannot resume its training assets")
        manifest["status"] = "recovering" if args.recover_only else "running"
        manifest["continuation_runner_sha256"] = file_sha256(Path(__file__))
        atomic_json(run_dir / "run_manifest.json", manifest)
        environment = dict(os.environ, FREQNICHE_RUN_DIR=str(run_dir), FREQNICHE_TRAIN_DIR=str(run_dir))
        environment["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src"), environment.get("PYTHONPATH", "")])
        for seed in request.seeds:
            name = f"train_seed_{seed}"
            folder = run_dir / f"trial_24_seed_{seed}"
            started = time.monotonic()
            recovered = False
            if (folder / "eval_result.json").exists():
                verify_result(folder, seed, request)
            elif (folder / "val_embeddings.npy").exists():
                recover_validation(run_dir, seed, request)
                recovered = True
            elif args.recover_only:
                print(f"[PENDING] seed={seed}; left unstarted", flush=True)
                continue
            else:
                argv = dict(pipeline_commands(request, run_dir))[name]
                path = run_dir / "logs" / f"{name}_continued_{int(time.time())}.log"
                print(f"[CONTINUE] seed={seed}: {path}", flush=True)
                with path.open("w", encoding="utf-8") as handle:
                    process = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    assert process.stdout is not None
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        handle.write(line)
                        handle.flush()
                    code = process.wait()
                if code:
                    raise RuntimeError(f"Continuation {name} failed with exit code {code}; see {path}")
                verify_result(folder, seed, request)
            if name not in completed:
                manifest.setdefault("completed_steps", []).append({"step": name, "elapsed_seconds": time.monotonic() - started,
                                                                  "recovered_validation_only": recovered})
                completed.add(name)
                atomic_json(run_dir / "run_manifest.json", manifest)
        all_done = all(f"train_seed_{seed}" in completed for seed in request.seeds)
        if all_done:
            if (run_dir / "result.csv").exists():
                raise RuntimeError("Result ledger already exists for this unfinished run; review before final publication")
            manifest.pop("error", None)
            finalize_run(request, run_dir, manifest)
            print(f"[COMPLETE] {run_dir}", flush=True)
        else:
            manifest["status"] = "interrupted"
            if "error" in manifest:
                manifest["recovered_error"] = manifest.pop("error")
            atomic_json(run_dir / "run_manifest.json", manifest)
            print("[RECOVERY COMPLETE] Saved Validation results recovered. Pending seeds were not trained.", flush=True)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        atomic_json(run_dir / "run_manifest.json", manifest)
        raise
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


if __name__ == "__main__":
    main()
