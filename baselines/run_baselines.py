#!/usr/bin/env python3
"""Unified launcher for the baseline methods retained by the manuscript."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.freqniche_st.orchestration import (  # noqa: E402
    atomic_json,
    sha256_file,
    slug,
)
from src.freqniche_st.metrics_ledger import append_wide_rows, generic_json_rows  # noqa: E402
from src.freqniche_st.model_config import frozen_internal_config  # noqa: E402


AVAILABILITY = {
    "human_skin": {"banksy", "cellcharter", "cellniche", "stagate", "sedr", "graphst", "spagcn"},
    "sea_ad": {"banksy", "cellcharter", "cellniche", "stagate", "sedr", "graphst", "spagcn"},
    "cosmx_nsclc": {"banksy", "cellcharter", "cellniche", "stagate", "sedr", "graphst", "spagcn"},
    "xenium_prime5k": {"banksy", "cellcharter", "stagate"},
}
BASELINE_METRICS = ROOT / "baselines/result_baselines.csv"

GENERAL_BASELINE_PYTHON = os.environ.get(
    "FREQNICHE_BASELINE_PYTHON",
    str(Path.home() / "anaconda3/envs/FreqNiche-baselines-sm120-pt28/bin/python"),
)
CELLCHARTER_PYTHON = os.environ.get(
    "FREQNICHE_CELLCHARTER_PYTHON",
    str(Path.home() / "anaconda3/envs/FreqNiche-cellcharter-sm120/bin/python"),
)
SPATIAL_BASELINE_PYTHON = os.environ.get(
    "FREQNICHE_SPATIAL_BASELINE_PYTHON",
    str(Path.home() / ".venvs/FreqNiche-spatial-baselines/bin/python"),
)


def python_for(method: str) -> str:
    if method == "cellcharter":
        return CELLCHARTER_PYTHON
    if method in {"sedr", "graphst", "spagcn"}:
        return SPATIAL_BASELINE_PYTHON
    return GENERAL_BASELINE_PYTHON


ADDITIONAL_SPATIAL = {"sedr", "graphst", "spagcn"}


def additional_steps(args: argparse.Namespace, methods: list[str], phase: str) -> list[tuple[str, list[str]]]:
    result = []
    for method in methods:
        for seed in args.seeds:
            result.append((f"{method}_{phase}_seed_{seed}", [
                python_for(method), "scripts/benchmark/run_additional_spatial_baselines.py",
                "--dataset", args.dataset, "--method", method, "--phase", phase,
                "--seed", str(seed), "--device", args.device,
                *(["--authorized"] if phase == "test" else []),
            ]))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retained FreqNiche baselines")
    parser.add_argument("--dataset", choices=sorted(AVAILABILITY), required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="final")
    parser.add_argument("--output-root", type=Path, default=ROOT / "baselines/Baselines_outputs/runs")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    py = GENERAL_BASELINE_PYTHON
    seeds = [str(seed) for seed in args.seeds]
    methods = args.methods
    steps: list[tuple[str, list[str]]] = []

    if args.dataset == "human_skin":
        steps.append(("prepare_contracts", [py, "scripts/benchmark/build_human_skin_contracts.py", "--phase", "train_validation"]))
        steps.extend(additional_steps(args, [m for m in methods if m in ADDITIONAL_SPATIAL], "validation"))
        steps.append(("prepare_test_contracts", [py, "scripts/benchmark/build_human_skin_contracts.py", "--phase", "test", "--authorized"]))
        for method in methods:
            if method in ADDITIONAL_SPATIAL:
                steps.extend(additional_steps(args, [method], "test"))
                continue
            if method == "banksy":
                steps.append((method, [python_for(method), "scripts/benchmark/run_banksy_benchmark.py", "--seeds", *seeds]))
            elif method == "cellcharter":
                for seed in seeds:
                    steps.append((f"{method}_seed_{seed}", [python_for(method), "scripts/benchmark/run_cellcharter_benchmark.py", "--seed", seed]))
            elif method == "cellniche":
                steps.append((method, [python_for(method), "scripts/benchmark/run_cellniche_benchmark.py", "--seeds", *seeds]))
            elif method == "stagate":
                steps.append((method, [python_for(method), "scripts/benchmark/run_stagate_benchmark.py", "--seeds", *seeds]))

    elif args.dataset == "sea_ad":
        steps.append(("build_cohorts", [py, "scripts/benchmark/build_sea_ad_cohorts.py"]))
        steps.extend(additional_steps(args, [m for m in methods if m in ADDITIONAL_SPATIAL], "validation"))
        for method in methods:
            if method in ADDITIONAL_SPATIAL:
                steps.extend(additional_steps(args, [method], "test"))
                continue
            steps.append((method, [python_for(method), "scripts/benchmark/run_sea_ad_baselines.py", "--methods", method, "--seeds", *seeds]))

    elif args.dataset == "cosmx_nsclc":
        steps.append(("build_cohorts", [py, "scripts/benchmark/build_cosmx_cohorts.py"]))
        steps.extend(additional_steps(args, [m for m in methods if m in ADDITIONAL_SPATIAL], "validation"))
        for method in methods:
            if method in ADDITIONAL_SPATIAL:
                steps.extend(additional_steps(args, [method], "test"))
                continue
            steps.append((method, [python_for(method), "scripts/benchmark/run_cosmx_baselines.py", "--methods", method, "--seeds", *seeds]))

    elif args.dataset == "xenium_prime5k":
        steps.extend(
            [
                ("audit_data", [py, "scripts/benchmark/build_xenium_audit_manifest.py"]),
                ("build_cohorts", [py, "scripts/benchmark/build_xenium_cohorts.py"]),
                ("build_graphs", [py, "scripts/benchmark/build_xenium_graphs.py"]),
            ]
        )
        for method in methods:
            method_python = python_for(method)
            steps.append((f"{method}_select_resolution", [method_python, "scripts/benchmark/run_xenium_baselines.py", "--phase", "val_selection", "--methods", method, "--seeds", *seeds, "--device", args.device]))
            steps.append((f"{method}_evaluate_test", [method_python, "scripts/benchmark/run_xenium_baselines.py", "--phase", "test_eval", "--authorized_stage4_full_run", "--methods", method, "--seeds", *seeds, "--device", args.device]))
    return steps


def collect(run_id: str, dataset: str, run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*.json")):
        lower_name = path.name.lower()
        if path.name == "run_manifest.json" or not any(
            token in lower_name for token in ("result", "report", "summary", "evaluation")
        ):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        method = str(payload.get("method") or payload.get("method_name") or path.stem.split("_seed")[0])
        records.extend(
            generic_json_rows(
                payload,
                run_id=run_id,
                dataset=dataset,
                source_file=str(path.relative_to(ROOT)).replace("\\", "/"),
                asset_dir=str(run_dir.relative_to(ROOT)).replace("\\", "/"),
                default_method=method,
                default_variant="baseline",
            )
        )
    return records


def prepare_cohort_model_config(run_dir: Path, manifest: dict[str, Any]) -> None:
    """Bind the SEA-AD cohort builder's default topology settings to this run.

    The builder shares FreqNiche's model-config gate even when only preparing
    centers for baselines. The baseline run has no model to tune, so use the
    repository's frozen defaults and record their physical hash in its manifest.
    """
    config_path = run_dir / "model_config.json"
    atomic_json(config_path, frozen_internal_config())
    manifest["model_config_sha256"] = sha256_file(config_path)


def main() -> None:
    args = parse_args()
    requested = set(args.methods)
    unavailable = requested - AVAILABILITY[args.dataset]
    if unavailable:
        raise ValueError(f"Methods unavailable for {args.dataset}: {sorted(unavailable)}")
    if len(requested) != len(args.methods):
        raise ValueError("Duplicate method names are not allowed")

    now = datetime.now().astimezone()
    run_id = f"{now:%Y%m%d-%H%M%S}_{slug(args.dataset)}_{slug(args.tag)}"
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    run_dir = output_root / args.dataset / now.strftime("%Y-%m-%d") / run_id
    steps = commands(args)
    manifest = {
        "manifest_version": "1.0",
        "run_id": run_id,
        "created_at": now.isoformat(),
        "status": "planned" if args.dry_run else "running",
        "dataset": args.dataset,
        "methods": args.methods,
        "seeds": args.seeds,
        "device": args.device,
        "commands": [{"step": name, "argv": argv} for name, argv in steps],
        "launcher_sha256": sha256_file(Path(__file__)),
    }
    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), **manifest}, indent=2))
        return
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing baseline run: {run_dir}")
    (run_dir / "logs").mkdir(parents=True)
    if args.dataset == "sea_ad":
        prepare_cohort_model_config(run_dir, manifest)
    atomic_json(run_dir / "run_manifest.json", manifest)
    environment = dict(os.environ)
    environment["FREQNICHE_RUN_DIR"] = str(run_dir)
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src"), environment.get("PYTHONPATH", "")])

    try:
        for name, argv in steps:
            started = time.time()
            log_path = run_dir / "logs" / f"{name}.log"
            with log_path.open("w", encoding="utf-8") as log_handle:
                process = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="")
                    log_handle.write(line)
                    log_handle.flush()
                code = process.wait()
            if code != 0:
                raise RuntimeError(f"Baseline step {name} failed with exit code {code}; see {log_path}")
            manifest.setdefault("completed_steps", []).append({"step": name, "elapsed_seconds": time.time() - started})
            atomic_json(run_dir / "run_manifest.json", manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        atomic_json(run_dir / "run_manifest.json", manifest)
        raise

    metrics = collect(run_id, args.dataset, run_dir)
    append_wide_rows(run_dir / "result.csv", metrics)
    append_wide_rows(BASELINE_METRICS, metrics, manifest_path=ROOT / "baselines/metrics_manifest.json")
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    manifest["metrics_rows"] = len(metrics)
    atomic_json(run_dir / "run_manifest.json", manifest)
    print(f"BASELINE_RUN_COMPLETE {run_dir}")


if __name__ == "__main__":
    main()
