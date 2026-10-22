#!/usr/bin/env python3
"""Run one explicitly configured FreqNiche experiment (no grid search).

Edit the single values below, then run ``python train.py`` from the project
root. If the current interpreter lacks the training packages, this launcher
uses ``~/anaconda3/envs/FreqNiche/bin/python`` when available. Use
``python train.py --dry-run`` to inspect commands without creating outputs.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import importlib.util
import math
import os
import sys
import traceback
from pathlib import Path
from typing import TextIO


def ensure_training_python(entrypoint: Path | None = None) -> None:
    """Select an installed FreqNiche interpreter before creating a run dir."""
    required_modules = ("anndata", "scanpy", "torch", "yaml")
    missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if not missing:
        return

    candidate = Path(os.environ.get("FREQNICHE_PYTHON", "~/anaconda3/envs/FreqNiche/bin/python")).expanduser()
    if (
        candidate.is_file()
        and candidate.resolve() != Path(sys.executable).resolve()
        and os.environ.get("FREQNICHE_RUNTIME_REEXEC") != "1"
    ):
        print(
            f"Current Python {sys.executable} lacks {', '.join(missing)}; "
            f"restarting with {candidate}",
            flush=True,
        )
        os.environ["FREQNICHE_RUNTIME_REEXEC"] = "1"
        os.execv(str(candidate), [str(candidate), str(entrypoint or Path(__file__).resolve()), *sys.argv[1:]])

    raise RuntimeError(
        f"Python {sys.executable} is missing {', '.join(missing)}. "
        "Activate the FreqNiche environment, use its Python directly, or set "
        "FREQNICHE_PYTHON to an interpreter with the required packages. "
        "No run directory was created."
    )


if __name__ == "__main__":
    ensure_training_python()

from src.freqniche_st.model_config import MODEL_PARAMETER_MAP, internal_from_public, manuscript_defaults
from src.freqniche_st.orchestration import (
    DEFAULT_OUTPUT_ROOT,
    RunRequest,
    load_yaml,
    run_pipeline,
)


ROOT = Path(__file__).resolve().parent


class _Tee:
    """Write each terminal message to both the console and a live text log."""

    def __init__(self, terminal: TextIO, logfile: TextIO) -> None:
        self.terminal = terminal
        self.logfile = logfile

    def write(self, message: str) -> int:
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()
        return len(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.logfile.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

# ---------------------------------------------------------------------------
# 1. Select exactly one dataset and one fixed set of hyperparameters.
#    Datasets: human_skin, sea_ad, cosmx_nsclc, xenium_prime5k.
# ---------------------------------------------------------------------------
DATASET = "human_skin"
SEEDS = (42, 123, 2024)  # For final three-seed evaluation: (42, 123, 2024).
DEVICE = "cuda"
TAG = "manuscript"
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT

# Training always includes Validation and saves the best checkpoint and
# Validation embeddings. Keep False during tuning to avoid accessing Test.
# For sea_ad / cosmx_nsclc / xenium_prime5k, Test requires all three seeds.
RUN_TEST = False

# Dataset-specific manuscript defaults (Table S02 and supplementary methods):
# Human Skin/CosMx: 128 dimensions, 3/2 layers, lr=1e-3, max 50 epochs.
# SEA-AD/Xenium:    256 dimensions, 6/4 layers, lr=1e-4, max 200 epochs.
# Both use batch_size=64, patience=7, min_delta=1e-4 and the three seeds above.
# To customize, override values here, e.g. MODEL["learning_rate"] = 5e-4.
MODEL, TRAINING = manuscript_defaults(DATASET)


def build_request(*, dry_run: bool = False) -> RunRequest:
    """Validate one configuration before any data preparation or file writes."""
    config_path = ROOT / "configs" / "datasets" / f"{DATASET}.yaml"
    if not config_path.is_file():
        raise ValueError(f"Unknown dataset {DATASET!r}; expected a config at {config_path}")
    dataset_config = load_yaml(config_path)
    data_path = ROOT / dataset_config["data_path"]
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset is missing: {data_path}")
    split_path = ROOT / dataset_config["splits_path"]
    if not split_path.is_file():
        raise FileNotFoundError(f"Donor split file is missing: {split_path}")
    for sample in dataset_config.get("required_samples", []):
        sample_path = data_path / sample
        if not sample_path.is_file():
            raise FileNotFoundError(f"Required processed H5AD is missing: {sample_path}")

    if set(MODEL) != set(MODEL_PARAMETER_MAP):
        raise ValueError("MODEL must contain each of the 12 public model parameters exactly once")
    internal_from_public(MODEL)  # Architecture and numeric-range validation.
    if set(TRAINING) != {"max_epochs", "patience", "batch_size", "min_delta"}:
        raise ValueError("TRAINING must contain exactly the four training-budget parameters")
    for key in ("max_epochs", "patience", "batch_size"):
        value = TRAINING[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"TRAINING[{key!r}] must be a positive integer")
    delta = TRAINING["min_delta"]
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta < 0:
        raise ValueError("TRAINING['min_delta'] must be a finite nonnegative number")
    if not SEEDS or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in SEEDS):
        raise ValueError("SEEDS must be a nonempty tuple of integer seeds")
    if len(set(SEEDS)) != len(SEEDS):
        raise ValueError("SEEDS must not contain duplicates")
    if RUN_TEST and DATASET != "human_skin" and SEEDS != (42, 123, 2024):
        raise ValueError("Formal Test on this dataset requires seeds (42, 123, 2024)")
    if DEVICE not in {"cpu", "cuda"}:
        raise ValueError("DEVICE must be 'cpu' or 'cuda'")

    output_root = Path(OUTPUT_ROOT)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    stages = ("prepare", "train", "evaluate") if RUN_TEST else ("prepare", "train")
    return RunRequest(
        dataset=DATASET,
        seeds=tuple(SEEDS),
        device=DEVICE,
        max_epochs=TRAINING["max_epochs"],
        patience=TRAINING["patience"],
        batch_size=TRAINING["batch_size"],
        min_delta=float(delta),
        stages=stages,
        tag=TAG,
        output_root=output_root,
        dry_run=dry_run,
        model_config=dict(MODEL),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the single run plan without writing or training")
    args = parser.parse_args()
    request = build_request(dry_run=args.dry_run)
    if args.dry_run:
        run_dir = run_pipeline(request)
        print(f"PLANNED single configuration: {run_dir}", flush=True)
        return

    log_path = ROOT / f"train_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}_{os.getpid()}.txt"
    with log_path.open("x", encoding="utf-8", buffering=1) as logfile:
        with redirect_stdout(_Tee(sys.stdout, logfile)), redirect_stderr(_Tee(sys.stderr, logfile)):
            print(f"[STARTED] {datetime.now().astimezone().isoformat(timespec='seconds')}", flush=True)
            print(f"[LOG] {log_path}", flush=True)
            try:
                run_dir = run_pipeline(request)
            except BaseException as error:
                traceback.print_exc()
                print(f"[FAILED] See {log_path}", flush=True)
                raise SystemExit(130 if isinstance(error, KeyboardInterrupt) else 1) from None
            print(f"[COMPLETED] Single configuration: {run_dir}", flush=True)
            print(f"[LOG] {log_path}", flush=True)


if __name__ == "__main__":
    main()
