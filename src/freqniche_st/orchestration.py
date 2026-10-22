"""Canonical FreqNiche experiment orchestration and metric ledger."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

from .metrics_ledger import FREQNICHE_FIELDS, append_wide_rows, generic_csv_rows, generic_json_rows
from .model_config import MODEL_PARAMETER_MAP, internal_from_public


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/runs/freqniche"
CENTRAL_METRICS = ROOT / "outputs/result.csv"
TRAINING_PARAMETERS = ("max_epochs", "patience", "batch_size", "min_delta")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    if not normalized:
        raise ValueError(f"Cannot create a run slug from {value!r}")
    return normalized.lower()


@dataclass(frozen=True)
class RunRequest:
    dataset: str
    seeds: tuple[int, ...]
    device: str
    max_epochs: int
    patience: int
    batch_size: int
    min_delta: float
    stages: tuple[str, ...]
    tag: str
    output_root: Path = DEFAULT_OUTPUT_ROOT
    dry_run: bool = False
    model_config: dict[str, Any] | None = None


def create_run_dir(request: RunRequest) -> tuple[str, Path]:
    timestamp = datetime.now().astimezone()
    stamp = timestamp.strftime("%Y%m%d-%H%M%S-%f")
    run_id = f"{stamp}_{slug(request.dataset)}_{slug(request.tag)}"
    run_dir = request.output_root / slug(request.dataset) / timestamp.strftime("%Y-%m-%d") / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    return run_id, run_dir


def _python() -> str:
    return sys.executable


def pipeline_commands(request: RunRequest, run_dir: Path) -> list[tuple[str, list[str]]]:
    py = _python()
    seeds = [str(seed) for seed in request.seeds]
    commands: list[tuple[str, list[str]]] = []
    model_args = ["--model-config-path", str(run_dir / "model_config.json")] if request.model_config is not None else []

    if request.dataset == "human_skin":
        if "prepare" in request.stages:
            commands.append(("prepare_contracts", [
                py, "scripts/benchmark/build_human_skin_contracts.py",
                "--phase", "train_validation",
            ]))
        if "train" in request.stages:
            for seed in seeds:
                train_args = [
                    py, "-m", "src.freqniche_st.human_skin_backend",
                    "--final-model", "--diagnostic-mode", "--trial-id", "24",
                    "--seed", seed, "--output-base", str(run_dir),
                    "--device", request.device,
                    "--max-epochs", str(request.max_epochs),
                    "--patience", str(request.patience),
                    "--min-delta", str(request.min_delta),
                    "--batch-size", str(request.batch_size),
                    *model_args,
                ]
                commands.append(
                    (
                        f"train_seed_{seed}",
                        train_args,
                    )
                )
        if "evaluate" in request.stages:
            commands.append(("prepare_test_contracts", [
                py, "scripts/benchmark/build_human_skin_contracts.py",
                "--phase", "test", "--authorized",
            ]))
            commands.append(("evaluate", [py, "scripts/benchmark/eval_freqniche_champion.py"]))

    elif request.dataset == "sea_ad":
        if "prepare" in request.stages:
            commands.extend(
                [
                    ("build_cohorts", [py, "scripts/benchmark/build_sea_ad_cohorts.py"]),
                    ("fit_whitener", [py, "scripts/benchmark/fit_sea_ad_whitener.py"]),
                ]
            )
        if "train" in request.stages:
            commands.append(("train", [
                py, "scripts/benchmark/train_sea_ad.py", "--seeds", *seeds,
                "--device", request.device,
                "--max-epochs", str(request.max_epochs),
                "--patience", str(request.patience),
                "--batch-size", str(request.batch_size),
                "--min-delta", str(request.min_delta),
                *model_args,
            ]))
        if "evaluate" in request.stages:
            commands.extend(
                [
                    ("select_resolution", [py, "scripts/benchmark/eval_sea_ad_stability.py", "--phase", "validation", "--device", request.device]),
                    ("evaluate_test", [py, "scripts/benchmark/eval_sea_ad_stability.py", "--phase", "test", "--device", request.device]),
                ]
            )

    elif request.dataset == "cosmx_nsclc":
        if "prepare" in request.stages:
            commands.extend(
                [
                    ("build_cohorts", [py, "scripts/benchmark/build_cosmx_cohorts.py"]),
                    ("fit_whitener", [py, "scripts/benchmark/fit_cosmx_whitener.py"]),
                ]
            )
        if "train" in request.stages:
            commands.append(
                (
                    "train",
                    [
                        py,
                        "scripts/benchmark/train_cosmx.py",
                        "--seeds",
                        *seeds,
                        "--device",
                        request.device,
                        "--batch-size",
                        str(request.batch_size),
                        "--max-epochs",
                        str(request.max_epochs),
                        "--patience",
                        str(request.patience),
                        "--min-delta",
                        str(request.min_delta),
                        *model_args,
                    ],
                )
            )
        if "evaluate" in request.stages:
            commands.extend(
                [
                    ("select_resolution", [py, "scripts/benchmark/eval_cosmx_stability.py", "--phase", "validation", "--device", request.device]),
                    ("evaluate_test", [py, "scripts/benchmark/eval_cosmx_stability.py", "--phase", "test", "--device", request.device]),
                ]
            )

    elif request.dataset == "xenium_prime5k":
        if "prepare" in request.stages:
            commands.extend(
                [
                    ("audit_data", [py, "scripts/benchmark/build_xenium_audit_manifest.py"]),
                    ("build_cohorts", [py, "scripts/benchmark/build_xenium_cohorts.py"]),
                    ("build_graphs", [py, "scripts/benchmark/build_xenium_graphs.py"]),
                    ("fit_whitener", [py, "scripts/benchmark/fit_xenium_whitener.py", "--save", "--authorized", "--device", request.device]),
                ]
            )
        if "train" in request.stages:
            commands.append(
                (
                    "train",
                    [
                        py,
                        "scripts/benchmark/train_xenium.py",
                        "--authorized",
                        "--seeds",
                        *seeds,
                        "--device",
                        request.device,
                        "--batch_size",
                        str(request.batch_size),
                        "--max_epochs",
                        str(request.max_epochs),
                        "--patience",
                        str(request.patience),
                        "--min_delta",
                        str(request.min_delta),
                        *model_args,
                    ],
                )
            )
        if "evaluate" in request.stages:
            commands.extend(
                [
                    ("select_resolution", [py, "scripts/benchmark/eval_xenium_stability.py", "--phase", "validation", "--device", request.device]),
                    ("evaluate_test", [py, "scripts/benchmark/eval_xenium_test.py", "--authorized", "--device", request.device]),
                ]
            )
    else:
        raise KeyError(f"Unknown dataset: {request.dataset}")

    if "evaluate" in request.stages and request.dataset in {"human_skin", "sea_ad"}:
        commands.append(("export_downstream_inputs", [
            py, "scripts/benchmark/export_downstream_inputs.py", "--run-dir", str(run_dir),
        ]))
    return commands


def collect_metrics(run_id: str, request: RunRequest, run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    preferred_csv = {
        "sea_ad": "table4_sea_ad_freqniche_results.csv",
        "cosmx_nsclc": "table6_cosmx_freqniche_results.csv",
    }.get(request.dataset)
    if preferred_csv and (run_dir / preferred_csv).exists():
        path = run_dir / preferred_csv
        return generic_csv_rows(
            path,
            run_id=run_id,
            dataset=request.dataset,
            source_file=str(path.relative_to(ROOT)).replace("\\", "/"),
            asset_dir=str(run_dir.relative_to(ROOT)).replace("\\", "/"),
            default_method="FreqNiche",
            default_variant="Full",
        )

    candidates = sorted(run_dir.rglob("*.json"))
    for path in candidates:
        lower_name = path.name.lower()
        if path.name == "run_manifest.json" or not any(
            token in lower_name for token in ("result", "report", "evaluation")
        ):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        source_file = str(path.relative_to(ROOT)).replace("\\", "/")
        records.extend(
            generic_json_rows(
                payload,
                run_id=run_id,
                dataset=request.dataset,
                source_file=source_file,
                asset_dir=str(run_dir.relative_to(ROOT)).replace("\\", "/"),
                default_method="FreqNiche",
                default_variant="full",
            )
        )
    if not records and "train" in request.stages and request.dataset in {"sea_ad", "cosmx_nsclc", "xenium_prime5k"}:
        summary_path = run_dir / "checkpoints/xenium_training_summary.json"
        xenium_summary = None
        if request.dataset == "xenium_prime5k":
            if not summary_path.is_file():
                raise FileNotFoundError(f"Missing training summary after successful training: {summary_path}")
            xenium_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(xenium_summary, list):
                raise ValueError(f"Unexpected Xenium training summary format: {summary_path}")
        for seed in request.seeds:
            if request.dataset == "xenium_prime5k":
                path = summary_path
                matches = [entry for entry in xenium_summary if isinstance(entry, dict) and entry.get("seed") == seed]
                if len(matches) != 1:
                    raise ValueError(f"Missing or duplicate Xenium training summary for seed {seed}")
                payload = matches[0]
            else:
                path = run_dir / "checkpoints" / f"seed_{seed}" / "training_history.json"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing training history after successful training: {path}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("seed") != seed:
                    raise ValueError(f"Training history seed mismatch: {path}")
            row = generic_json_rows(
                payload,
                run_id=run_id,
                dataset=request.dataset,
                source_file=str(path.relative_to(ROOT)).replace("\\", "/"),
                asset_dir=str(run_dir.relative_to(ROOT)).replace("\\", "/"),
                default_method="FreqNiche",
                default_variant="Full",
            )[0]
            row.update(task="validation_denoising", split="validation", status="complete")
            records.append(row)
    return records


def actual_epochs_for_seed(dataset: str, run_dir: Path, seed: Any) -> int | str:
    """Read completed training epochs, never infer them from the epoch budget.

    Aggregate rows and analyses without an unambiguous per-seed training record
    intentionally remain blank. ``best_epoch`` is not the number of epochs run.
    """
    try:
        seed_number = int(seed)
    except (TypeError, ValueError):
        return ""

    if dataset == "human_skin":
        source = run_dir / f"trial_24_seed_{seed_number}" / "eval_result.json"
        if source.is_file():
            value = json.loads(source.read_text(encoding="utf-8")).get("stopped_epoch")
        else:
            return ""
    elif dataset == "sea_ad":
        source = run_dir / "checkpoints" / f"seed_{seed_number}" / "training_history.json"
        if not source.is_file():
            return ""
        history = json.loads(source.read_text(encoding="utf-8"))
        value = history.get("total_epochs_run")
        if value is None and isinstance(history.get("train_loss_history"), list):
            value = len(history["train_loss_history"])
    elif dataset == "cosmx_nsclc":
        source = run_dir / "checkpoints" / f"seed_{seed_number}" / "training_history.json"
        if not source.is_file():
            return ""
        history = json.loads(source.read_text(encoding="utf-8"))
        losses = history.get("train_loss_history")
        value = len(losses) if isinstance(losses, list) else None
    elif dataset == "xenium_prime5k":
        source = run_dir / "checkpoints" / "xenium_training_summary.json"
        if not source.is_file():
            return ""
        summary = json.loads(source.read_text(encoding="utf-8"))
        value = next(
            (entry.get("total_epochs") for entry in summary
             if isinstance(entry, dict) and entry.get("seed") == seed_number),
            None,
        ) if isinstance(summary, list) else None
    else:
        return ""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return ""
    return value


def run_pipeline(request: RunRequest) -> Path:
    run_id, run_dir = create_run_dir(request)
    commands = pipeline_commands(request, run_dir)
    manifest = {
        "manifest_version": "1.0",
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "planned" if request.dry_run else "running",
        "request": {
            "dataset": request.dataset,
            "seeds": list(request.seeds),
            "device": request.device,
            "max_epochs": request.max_epochs,
            "patience": request.patience,
            "batch_size": request.batch_size,
            "min_delta": request.min_delta,
            "model_config": request.model_config or {},
            "stages": list(request.stages),
            "tag": request.tag,
        },
        "commands": [{"step": step, "argv": argv} for step, argv in commands],
        "code_sha256": sha256_file(Path(__file__)),
    }
    if request.dry_run:
        print(json.dumps({"run_dir": str(run_dir), **manifest}, indent=2))
        return run_dir

    run_dir.mkdir(parents=True, exist_ok=False)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir()
    if request.model_config is not None:
        model_path = run_dir / "model_config.json"
        atomic_json(model_path, internal_from_public(request.model_config))
        manifest["model_config_sha256"] = sha256_file(model_path)
    atomic_json(run_dir / "run_manifest.json", manifest)
    environment = dict(os.environ)
    environment["FREQNICHE_RUN_DIR"] = str(run_dir)
    environment["FREQNICHE_TRAIN_DIR"] = str(run_dir)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "src"), environment.get("PYTHONPATH", "")]
    )

    try:
        for step, argv in commands:
            log_path = logs_dir / f"{step}.log"
            started = time.time()
            with log_path.open("w", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    argv,
                    cwd=ROOT,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="")
                    log_handle.write(line)
                    log_handle.flush()
                return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"Step {step} failed with exit code {return_code}; see {log_path}")
            manifest.setdefault("completed_steps", []).append(
                {"step": step, "elapsed_seconds": time.time() - started, "log": str(log_path.relative_to(ROOT))}
            )
            atomic_json(run_dir / "run_manifest.json", manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        atomic_json(run_dir / "run_manifest.json", manifest)
        raise

    return finalize_run(request, run_dir, manifest)


def finalize_run(request: RunRequest, run_dir: Path, manifest: dict[str, Any]) -> Path:
    """Publish the same metric ledger after either initial execution or resume."""
    run_id = manifest["run_id"]
    metrics = collect_metrics(run_id, request, run_dir)
    hyperparameters = {
        **(request.model_config or {}),
        "radius_um": 50.0 if request.dataset == "human_skin" else "",
        "max_epochs": request.max_epochs,
        "patience": request.patience,
        "batch_size": request.batch_size,
        "min_delta": request.min_delta,
    }
    for row in metrics:
        row.update({key: value for key, value in hyperparameters.items() if key in FREQNICHE_FIELDS})
        row["actual_epochs"] = actual_epochs_for_seed(request.dataset, run_dir, row.get("seed"))
        if request.dataset == "human_skin":
            row["task"] = row.get("task") or "spatial_domain"
            row["paradigm"] = row.get("paradigm") or "inductive"
    append_wide_rows(run_dir / "result.csv", metrics, fieldnames=FREQNICHE_FIELDS)
    append_wide_rows(CENTRAL_METRICS, metrics, manifest_path=ROOT / "outputs/metrics_manifest.json", fieldnames=FREQNICHE_FIELDS)
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    manifest["metrics_rows"] = len(metrics)
    atomic_json(run_dir / "run_manifest.json", manifest)
    return run_dir


def grid_combinations(grid: dict[str, Any], base: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Expand explicitly listed dimensions while retaining non-varied defaults."""
    allowed = set(TRAINING_PARAMETERS) | set(MODEL_PARAMETER_MAP)
    if not grid or set(grid) - allowed:
        raise ValueError(f"Unknown or empty grid dimensions: {sorted(set(grid) - allowed)}")
    keys = list(grid)
    values = []
    for key in keys:
        raw = grid.get(key)
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"Grid key {key!r} must contain a non-empty list")
        values.append(raw)
    return [{**(base or {}), **dict(zip(keys, combination))} for combination in itertools.product(*values)]
