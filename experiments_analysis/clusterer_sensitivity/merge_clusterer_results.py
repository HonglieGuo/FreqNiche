"""Verify and merge local FreqNiche and remote baseline result shards."""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import clusterer_common as common


COMBINED_DIR = common.OUTPUT_ROOT / "combined_v1"


def verify_input_records(value) -> None:
    if isinstance(value, dict) and {"path", "sha256"}.issubset(value):
        path = common.ROOT / value["path"]
        if common.compute_sha256(path) != value["sha256"]:
            raise RuntimeError(f"Bound input SHA mismatch: {path}")
    elif isinstance(value, dict):
        for nested in value.values():
            verify_input_records(nested)


def verify_evaluation(suite: str) -> tuple[dict, pd.DataFrame, Path, Path]:
    root = common.evaluation_dir(suite)
    report_path = root / "evaluation_report.json"
    csv_path = root / "clusterer_results.csv"
    report = common.load_signed_json(report_path)
    if report.get("report_version") != "1.0" or report.get("suite") != suite:
        raise RuntimeError(f"Unexpected {suite} evaluation report")
    verify_input_records(report.get("inputs", {}))
    for relative, expected in report["outputs"].items():
        path = root / relative
        if common.compute_sha256(path) != expected:
            raise RuntimeError(f"{suite} output SHA mismatch: {relative}")
    frame = pd.read_csv(csv_path, dtype={"seed": str})
    if set(frame["clusterer"]) != set(common.CLUSTERERS):
        raise RuntimeError(f"Incomplete clusterers in {suite}")
    if set(frame["track"]) != {"validation_frozen", "oracle_true_k"}:
        raise RuntimeError(f"Incomplete tracks in {suite}")
    return report, frame, report_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge clusterer-sensitivity result shards")
    parser.add_argument("--authorized", action="store_true")
    args = parser.parse_args()
    if not args.authorized:
        parser.error("Result merge requires --authorized")
    if COMBINED_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite combined results: {COMBINED_DIR}")

    local_report, local_frame, local_report_path, local_csv_path = verify_evaluation("freqniche")
    remote_report, remote_frame, remote_report_path, remote_csv_path = verify_evaluation("baselines")
    local_environment = local_report.get("environment", {})
    remote_environment = remote_report.get("environment", {})
    mismatches = {
        key: {"local": local_environment.get(key), "remote": remote_environment.get(key)}
        for key in common.COMPARABLE_ENVIRONMENT_KEYS
        if local_environment.get(key) != remote_environment.get(key)
    }
    if mismatches:
        raise RuntimeError(f"Local/remote algorithm-environment mismatch: {mismatches}")
    combined = pd.concat([local_frame, remote_frame], ignore_index=True)
    # Archived reports can include an unused adaptation arm; omit it from paper outputs.
    combined = combined.loc[~combined["method"].eq("FreqNiche-Transductive")].copy()
    expected_methods = {
        "FreqNiche-Inductive",
        "CellCharter-Canonical", "BANKSY-Controlled",
        "STAGATE-Controlled", "CellNiche-Controlled",
    }
    if set(combined["method"]) != expected_methods:
        raise RuntimeError("Combined method registry is incomplete")

    summary_rows = []
    metric_columns = ["donor_macro_ari", "donor_macro_nmi", "worst_donor_ari"]
    for (method, clusterer, track), group in combined.groupby(["method", "clusterer", "track"], sort=True):
        row = {
            "method": method,
            "clusterer": clusterer,
            "track": track,
            "n_runs": int(len(group)),
        }
        for metric in metric_columns:
            values = group[metric].astype(float).to_numpy()
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"sample_sd_{metric}"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    staging = common.OUTPUT_ROOT / f".combined_v1.tmp.{os.getpid()}"
    staging.mkdir(parents=True)
    try:
        runs_path = staging / "all_runs.csv"
        summary_path = staging / "summary.csv"
        combined.to_csv(runs_path, index=False)
        summary.to_csv(summary_path, index=False)
        report = {
            "report_version": "1.0",
            "analysis": "Human Skin clusterer-sensitivity combined report",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "interpretation_boundary": "Validation-frozen track is the primary label-free sensitivity result; Oracle true-K is diagnostic only; no best clusterer is selected post hoc",
            "methods": sorted(expected_methods),
            "clusterers": list(common.CLUSTERERS),
            "tracks": ["validation_frozen", "oracle_true_k"],
            "environment_comparison": {
                "required_equal_keys": list(common.COMPARABLE_ENVIRONMENT_KEYS),
                "local": local_environment,
                "remote": remote_environment,
                "python_patch_difference_allowed_but_recorded": True,
            },
            "inputs": {
                "merge_runner": common.input_record(Path(__file__)),
                "common": common.input_record(Path(common.__file__)),
                "local_report": common.input_record(local_report_path),
                "local_csv": common.input_record(local_csv_path),
                "remote_report": common.input_record(remote_report_path),
                "remote_csv": common.input_record(remote_csv_path),
            },
            "outputs": {
                "all_runs.csv": common.compute_sha256(runs_path),
                "summary.csv": common.compute_sha256(summary_path),
            },
        }
        common.atomic_write_json(staging / "combined_report.json", report)
        os.replace(staging, COMBINED_DIR)
        print(f"Combined results published: {COMBINED_DIR}")
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
