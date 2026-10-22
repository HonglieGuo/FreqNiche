#!/usr/bin/env python3
"""Train, infer Test embeddings and run downstream tasks without historical outputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import train

ROOT = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=("human_skin", "sea_ad", "cosmx_nsclc", "xenium_prime5k"))
    result.add_argument("--stage", choices=("all", "model", "downstream"), default="all")
    result.add_argument("--run-dir", type=Path, help="Existing completed run; only for --stage downstream")
    result.add_argument("--tasks", nargs="+", choices=(
        "neighborhood_composition_prediction", "cortical_depth_prediction", "niche_associated_expression",
    ), help="Manuscript analyses: neighborhood cell composition, cortical depth, or within-cell-type niche-associated expression")
    result.add_argument("--profile", choices=("manuscript", "current"), default="manuscript",
                        help="manuscript: dataset-specific paper defaults; current: editable MODEL/TRAINING in train.py")
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--max-epochs", type=int)
    result.add_argument("--batch-size", type=int)
    result.add_argument("--expression-perivasc-cluster", default="0")
    result.add_argument("--expression-stroma-cluster", default="2")
    result.add_argument("--dry-run", action="store_true", help="Print commands only; no training or writes")
    return result


def task_commands(dataset: str, tasks: list[str], run_dir: Path, *, perivasc="0", stroma="2") -> list[list[str]]:
    supported = {
        "human_skin": {"neighborhood_composition_prediction", "niche_associated_expression"},
        "sea_ad": {"cortical_depth_prediction"}, "cosmx_nsclc": set(), "xenium_prime5k": set(),
    }
    if set(tasks) - supported[dataset]:
        raise ValueError(f"Tasks {tasks} do not apply to {dataset}")
    commands = []
    for task in dict.fromkeys(tasks):
        script = {
            "neighborhood_composition_prediction": "neighborhood_composition_prediction/run_neighborhood_composition_prediction.py",
            "cortical_depth_prediction": "cortical_depth_prediction/run_cortical_depth_prediction.py",
            "niche_associated_expression": "niche_associated_expression/run_niche_associated_expression.py",
        }[task]
        argv = [sys.executable, str(ROOT / "experiments_analysis" / script), "--run-dir", str(run_dir)]
        if task == "niche_associated_expression":
            argv += ["--perivasc-cluster", perivasc, "--stroma-cluster", stroma]
        commands.append(argv)
    return commands


def main() -> None:
    argument_parser = parser()
    args = argument_parser.parse_args()
    if args.stage == "downstream":
        if args.run_dir is None:
            argument_parser.error("--stage downstream requires --run-dir")
        run_dir = args.run_dir.expanduser().resolve()
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        dataset = manifest["request"]["dataset"]
        if args.dataset is not None and args.dataset != dataset:
            argument_parser.error("--dataset differs from the selected run")
    else:
        if args.run_dir is not None:
            argument_parser.error("--run-dir is only valid with --stage downstream")
        dataset = args.dataset or "human_skin"
        run_dir = None
        configured_data = os.environ.get("FREQNICHE_DATASET_PATH")
        if configured_data:
            config = train.load_yaml(ROOT / "configs/datasets" / f"{dataset}.yaml")
            if Path(configured_data).expanduser().resolve() != (ROOT / config["data_path"]).resolve():
                argument_parser.error(
                    "FREQNICHE_DATASET_PATH points outside the selected dataset configuration; "
                    "unset it before running public reproduction"
                )
    tasks = args.tasks if args.tasks is not None else {
        "human_skin": ["neighborhood_composition_prediction", "niche_associated_expression"],
        "sea_ad": ["cortical_depth_prediction"],
    }.get(dataset, [])
    if args.stage == "model":
        if args.tasks:
            argument_parser.error("--tasks requires --stage all or downstream")
        tasks = []
    # Validate task/dataset compatibility before starting expensive training.
    task_commands(dataset, tasks, run_dir or ROOT / "planned")
    if args.stage != "downstream":
        train.DATASET = dataset
        train.DEVICE = args.device
        train.RUN_TEST = True
        train.SEEDS = (42, 123, 2024)
        train.MODEL = dict(train.MODEL)
        train.TRAINING = dict(train.TRAINING)
        if args.profile == "manuscript":
            train.MODEL, train.TRAINING = train.manuscript_defaults(dataset)
        if args.max_epochs is not None:
            train.TRAINING["max_epochs"] = args.max_epochs
        if args.batch_size is not None:
            train.TRAINING["batch_size"] = args.batch_size
        train.TAG = f"reproduce-{args.profile}"
        request = train.build_request(dry_run=args.dry_run)
        run_dir = train.run_pipeline(request)
    environment = dict(os.environ)
    # Analysis scripts receive explicit inputs; do not inherit unrelated run overrides.
    environment.pop("FREQNICHE_RUN_DIR", None)
    environment.pop("FREQNICHE_TRAIN_DIR", None)
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src"), environment.get("PYTHONPATH", "")])
    environment.setdefault("MPLBACKEND", "Agg")
    for command in task_commands(dataset, tasks, run_dir,
                                 perivasc=args.expression_perivasc_cluster, stroma=args.expression_stroma_cluster):
        print("DOWNSTREAM:", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
    print(f"{'PLANNED' if args.dry_run else 'COMPLETED'}: {run_dir}")


if __name__ == "__main__":
    if "--help" not in sys.argv and "-h" not in sys.argv:
        train.ensure_training_python(Path(__file__).resolve())
    main()
