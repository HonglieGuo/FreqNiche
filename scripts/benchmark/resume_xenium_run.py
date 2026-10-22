#!/usr/bin/env python3
"""Resume one canonical Xenium Train/Validation run without overwriting complete seeds."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.freqniche_st.model_config import internal_from_public, load_model_config
from src.freqniche_st.orchestration import RunRequest, atomic_json, finalize_run, pipeline_commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    path = run / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    req = manifest["request"]
    if req["dataset"] != "xenium_prime5k" or req["stages"] != ["prepare", "train"]:
        raise ValueError("Only canonical Xenium Train/Validation runs can be resumed")
    if req["seeds"] != [42, 123, 2024]:
        raise ValueError("Expected original three-seed run")
    cfg = load_model_config(run / "model_config.json", output_dir=run)
    if cfg != internal_from_public(req["model_config"]):
        raise ValueError("Run/model configuration mismatch")
    request = RunRequest(dataset=req["dataset"], seeds=tuple(req["seeds"]), device=req["device"],
                         max_epochs=req["max_epochs"], patience=req["patience"], batch_size=req["batch_size"],
                         min_delta=req["min_delta"], stages=tuple(req["stages"]), tag=req["tag"],
                         model_config=req["model_config"])
    original_command = dict(pipeline_commands(request, run))["train"]
    recorded = next(entry["argv"] for entry in manifest["commands"] if entry["step"] == "train")
    if original_command != recorded:
        raise ValueError("Original training command no longer matches this run")
    command = [*original_command, "--resume", "--single-thread-backward"]
    print(json.dumps({"run_dir": str(run), "command": command, "reuse_complete_seeds": True,
                      "per_epoch_resume": True, "single_thread_backward": True}, indent=2), flush=True)
    if args.dry_run:
        return
    import fcntl
    with (run / "xenium_recovery.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest["status"] == "complete":
            print(f"[ALREADY COMPLETE] {run}", flush=True)
            return
        if (run / "result.csv").exists():
            raise RuntimeError("Unfinished run already has a result ledger; review before publication")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        log = run / "logs" / f"train_recovery_{stamp}.log"
        shutil.copy2(path, run / "logs" / f"manifest_before_recovery_{stamp}.json")
        environment = dict(os.environ, FREQNICHE_RUN_DIR=str(run), FREQNICHE_TRAIN_DIR=str(run),
                           PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1")
        environment["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src"), environment.get("PYTHONPATH", "")])
        from scripts.benchmark.train_xenium import compute_file_sha256
        complete_before = {str(checkpoint): compute_file_sha256(checkpoint)
                           for checkpoint in (run / "checkpoints").glob("xenium_freqniche_trial24_seed_*.pt")}
        recovery = {"started_at": datetime.now().astimezone().isoformat(), "controller_pid": os.getpid(),
                    "command": command, "log": str(log), "status": "running",
                    "previous_error": manifest.pop("error", None), "preserved_checkpoints": complete_before,
                    "runner_sha256": compute_file_sha256(ROOT / "scripts/benchmark/train_xenium.py")}
        manifest.setdefault("recovery_history", []).append(recovery)
        manifest["status"] = "running"
        atomic_json(path, manifest)
        started = time.perf_counter()
        try:
            with log.open("w", encoding="utf-8") as handle:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                recovery["training_pid"] = process.pid
                atomic_json(path, manifest)
                for line in process.stdout:
                    print(line, end="", flush=True)
                    handle.write(line)
                    handle.flush()
                code = process.wait()
            if code:
                raise RuntimeError(f"Xenium recovery training exited {code}; see {log}")
            for filename, expected in complete_before.items():
                if compute_file_sha256(Path(filename)) != expected:
                    raise RuntimeError(f"A completed checkpoint changed during recovery: {filename}")
            summary = json.loads((run / "checkpoints/xenium_training_summary.json").read_text())
            if sorted(entry["seed"] for entry in summary) != [42, 123, 2024]:
                raise RuntimeError("Recovered training summary is incomplete")
            recovery.update(status="complete", finished_at=datetime.now().astimezone().isoformat())
            manifest.setdefault("completed_steps", []).append({"step": "train", "recovered": True,
                                                               "log": str(log.relative_to(ROOT)),
                                                               "elapsed_seconds": time.perf_counter() - started})
            finalize_run(request, run, manifest)
            print(f"[RECOVERY COMPLETE] {run}", flush=True)
        except BaseException as error:
            recovery.update(status="failed", error=repr(error))
            manifest.update(status="failed", error=repr(error))
            atomic_json(path, manifest)
            raise


if __name__ == "__main__":
    main()
