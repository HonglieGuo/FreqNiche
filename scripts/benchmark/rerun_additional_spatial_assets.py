#!/usr/bin/env python3
"""Rerun only SEDR/GraphST/SpaGCN with complete immutable donor asset bundles."""
from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.benchmark import run_additional_spatial_baselines as runner
from src.benchmark.adapters import spatial_artifacts as assets

REFERENCES = {
    "human_skin": ROOT / "baselines/Baselines_outputs/runs/human_skin/2026-09-30/20260930-152429_human_skin_final",
    "sea_ad": ROOT / "baselines/Baselines_outputs/runs/sea_ad/2026-09-30/20260930-161430_sea_ad_final",
    "cosmx_nsclc": ROOT / "baselines/Baselines_outputs/runs/cosmx_nsclc/2026-09-30/20260930-173322_cosmx_nsclc_final",
}


def tasks(datasets, methods, seeds):
    for dataset in datasets:
        for phase in ["validation", "test"]:
            for method in methods:
                for seed in seeds:
                    yield dataset, method, seed, phase


def preflight(datasets, methods, seeds):
    """Read metadata/selected cells only; never train or evaluate a model."""
    import h5py
    import numpy as np
    import pandas as pd
    import torch
    report = {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "datasets": {},
              "h5ad_embedding_keys": {}, "dataset_files": {},
              "versions": {name: metadata.version(name) for name in
                           ["torch", "numpy", "scipy", "scikit-learn", "scanpy", "anndata"]}}
    for path in sorted((ROOT / "datasets/h5ad").rglob("*.h5ad")):
        with h5py.File(path, "r") as handle:
            report["h5ad_embedding_keys"][str(path.relative_to(ROOT))] = list(handle.get("obsm", {}).keys())
    for dataset in datasets:
        reference = REFERENCES[dataset]
        source = Path(runner.DATASETS[dataset]["data"])
        stat = source.stat()
        report["dataset_files"][str(source)] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        info = {"reference": str(reference), "phases": {}}
        for phase, split in [("validation", "val"), ("test", "test")]:
            groups = runner._load_donor_center_data(dataset, split, reference)
            info["phases"][phase] = {donor: {"cells": data.n_obs, "genes": data.n_vars}
                                     for donor, data in groups.items()}
            for method in methods:
                for seed in seeds:
                    directory = reference / "additional_spatial" / method / f"seed_{seed}"
                    payload = runner.signed_read(directory / ("selection.json" if split == "val" else "test_result.json"))
                    expected = payload["per_validation_donor"] if split == "val" else payload["per_donor"]
                    if set(expected) != set(groups):
                        raise RuntimeError(f"Frozen donor list changed: {directory}")
                    for donor, data in groups.items():
                        if split == "val":
                            cells = pd.read_csv(directory / f"val_barcodes_{donor}.csv", dtype=str)
                            if not np.array_equal(cells["cell_barcode"].to_numpy(), data.obs_names.astype(str)):
                                raise RuntimeError(f"Frozen validation cell order changed: {directory}/{donor}")
                        elif expected[donor]["cells"] != data.n_obs:
                            raise RuntimeError(f"Frozen test cell count changed: {directory}/{donor}")
            del groups
        report["datasets"][dataset] = info
    for current, historical in [
        ("outputs/contracts/human_skin/validation_barcodes_8k.csv",
         "outputs/freqniche_wave_a_diagnostics/original_val_barcodes.csv"),
        ("outputs/contracts/human_skin/test_centers_8k.csv",
         "outputs/freqniche_wave_a_diagnostics/wave_a_main/test_centers_mapping.csv"),
    ]:
        if "human_skin" in datasets:
            a = pd.read_csv(ROOT / current, dtype=str)["cell_barcode"].to_numpy()
            b = pd.read_csv(ROOT / historical, dtype=str)["cell_barcode"].to_numpy()
            if not np.array_equal(a, b):
                raise RuntimeError("Human Skin current and historical center lists differ")
    report["status"] = "passed"
    return report


def audit_results(output: Path, datasets, methods, seeds, *, replay=False):
    inventory, comparison = [], []
    for dataset in datasets:
        for method in methods:
            old_values, new_values, new_nmi = [], [], []
            for seed in seeds:
                directory = output / dataset / "additional_spatial" / method / f"seed_{seed}"
                binding = runner._code_binding(method)
                selection = runner.signed_read(directory / "selection.json")
                test = runner.signed_read(directory / "test_result.json")
                reference_directory = REFERENCES[dataset] / "additional_spatial" / method / f"seed_{seed}"
                old_selection = runner.signed_read(reference_directory / "selection.json")
                old_test = runner.signed_read(reference_directory / "test_result.json")
                for payload in [selection, test]:
                    identity = argparse.Namespace(dataset=dataset, method=method, seed=seed)
                    runner._verify_identity(payload, identity, binding)
                if test["selection_sha256"] != assets.sha256(directory / "selection.json"):
                    raise RuntimeError(f"Validation selection binding changed: {directory}")
                for phase, payload in [("validation", selection), ("test", test)]:
                    expected = set(payload["per_validation_donor"] if phase == "validation" else payload["per_donor"])
                    original_donors = old_selection["per_validation_donor"] if phase == "validation" else old_test["per_donor"]
                    if expected != set(original_donors):
                        raise RuntimeError(f"Donor coverage differs from frozen reference: {directory}/{phase}")
                    if set(payload.get("asset_manifests", {})) != expected:
                        raise RuntimeError(f"Incomplete donor coverage: {directory}/{phase}")
                    for donor, record in payload["asset_manifests"].items():
                        manifest_path = directory / record["path"]
                        if not manifest_path.resolve().is_relative_to(directory.resolve()):
                            raise RuntimeError(f"Asset path escapes result directory: {manifest_path}")
                        if assets.sha256(manifest_path) != record["sha256"]:
                            raise RuntimeError(f"Asset manifest changed: {manifest_path}")
                        bundle = assets.verify_artifact(
                            manifest_path.parent, replay=replay,
                            expected_method=method, expected_seed=seed,
                        )
                        if bundle["smoke"] or "clustering_resolution" not in bundle:
                            raise RuntimeError(f"Incomplete formal asset bundle: {manifest_path}")
                        if phase == "validation":
                            original_cells = runner.pd.read_csv(
                                reference_directory / f"val_barcodes_{donor}.csv", dtype=str,
                            )["cell_barcode"].to_numpy()
                            saved_cells = runner.pd.read_csv(
                                manifest_path.parent / "cells.csv", dtype=str, keep_default_na=False,
                            )["cell_barcode"].to_numpy()
                            if not runner.np.array_equal(original_cells, saved_cells):
                                raise RuntimeError(f"Validation cell order differs from reference: {donor}")
                        elif bundle["embedding_shape"][0] != old_test["per_donor"][donor]["cells"]:
                            raise RuntimeError(f"Test cell count differs from frozen reference: {donor}")
                        inventory.append({
                            "dataset": dataset, "method": method, "seed": seed, "phase": phase,
                            "donor": donor, "cells": bundle["embedding_shape"][0],
                            "dimension": bundle["embedding_shape"][1], "path": str(manifest_path.parent),
                        })
                old_values.append(old_test["donor_macro_ari"])
                new_values.append(test["donor_macro_ari"])
                new_nmi.append(test["donor_macro_nmi"])
            comparison.append({
                "dataset": dataset, "method": method, "seeds": "/".join(map(str, seeds)),
                "paper_reference_ari_mean": statistics.mean(old_values),
                "rerun_ari_mean": statistics.mean(new_values),
                "rerun_ari_sd": statistics.stdev(new_values) if len(seeds) > 1 else 0.0,
                "ari_mean_delta": statistics.mean(new_values) - statistics.mean(old_values),
                "rerun_nmi_mean": statistics.mean(new_nmi),
            })
    report = {"status": "passed", "donor_bundles": len(inventory), "replay": replay, "assets": inventory}
    assets.signed_write(output / "asset_audit.json", report)
    temporary = output / ".comparison.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    os.replace(temporary, output / "comparison.csv")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(REFERENCES), default=list(REFERENCES))
    parser.add_argument("--methods", nargs="+", choices=["sedr", "graphst", "spagcn"],
                        default=["sedr", "graphst", "spagcn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="Check data and frozen centers without training")
    parser.add_argument("--replay", action="store_true", help="Audit also replays all donor weights")
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    base = ROOT / "baselines/Baselines_outputs"
    if not output.resolve().is_relative_to(base.resolve()) or output.resolve() == base.resolve():
        parser.error("--output-dir must be a new subdirectory of baselines/Baselines_outputs")
    for values in [args.datasets, args.methods, args.seeds]:
        if len(values) != len(set(values)):
            parser.error("Duplicate dataset, method or seed")
    identity = {"datasets": args.datasets, "methods": args.methods, "seeds": args.seeds, "device": args.device}
    planned = list(tasks(args.datasets, args.methods, args.seeds))
    if args.dry_run:
        print(json.dumps({"output": str(output), "phase_processes": len(planned), "configuration": identity,
                          "tasks": planned, "reference_runs": {d: str(REFERENCES[d]) for d in args.datasets}}, indent=2))
        return
    if args.preflight_only:
        print(json.dumps(preflight(args.datasets, args.methods, args.seeds), indent=2))
        return
    manifest_path = output / "asset_backfill.json"
    code_bindings = {method: runner._code_binding(method) for method in args.methods}
    def compatible_bindings(saved):
        return isinstance(saved, dict) and set(saved) == set(code_bindings) and all(
            assets.code_binding_matches(saved[method], binding) for method, binding in code_bindings.items())
    if args.audit_only:
        manifest = assets.signed_read(manifest_path)
        if manifest["configuration"] != identity:
            raise RuntimeError("Audit options differ from saved run configuration")
        if not compatible_bindings(manifest.get("code_bindings")):
            raise RuntimeError("Source code changed since this backfill")
        report = audit_results(output, args.datasets, args.methods, args.seeds, replay=args.replay)
        print(f"ASSET_AUDIT_PASS bundles={report['donor_bundles']} replay={args.replay}")
        return
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    if "spagcn" in args.methods and importlib.util.find_spec("louvain") is None:
        raise RuntimeError("Formal SpaGCN needs louvain in this same Python environment")
    for dataset in args.datasets:
        if not Path(runner.DATASETS[dataset]["data"]).is_file() or not REFERENCES[dataset].is_dir():
            raise FileNotFoundError(f"Data or frozen reference run missing: {dataset}")
    checks = preflight(args.datasets, args.methods, args.seeds)
    print("PREFLIGHT_PASS frozen donor/cell lists verified", flush=True)
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("Existing run requires --resume; old paper results are never overwritten")
        manifest = assets.signed_read(manifest_path)
        if manifest["configuration"] != identity:
            raise RuntimeError("Resume options differ from original run")
        if not compatible_bindings(manifest.get("code_bindings")):
            raise RuntimeError("Source code changed during this backfill; use a new run directory")
        if manifest["preflight"]["versions"] != checks["versions"]:
            raise RuntimeError("Python package versions changed during this backfill; use a new run directory")
        if manifest["preflight"]["dataset_files"] != checks["dataset_files"]:
            raise RuntimeError("Source H5AD files changed during this backfill")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output directory contains unrecognized assets: {output}")
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"status": "running", "configuration": identity,
                    "created_at": datetime.now().astimezone().isoformat(), "completed_steps": [],
                    "reference_runs": {d: str(REFERENCES[d]) for d in args.datasets}, "input_files": {},
                    "preflight": checks, "code_bindings": code_bindings}
        for dataset in args.datasets:
            run = output / dataset
            run.mkdir()
            for split in ["val", "test"]:
                source = runner.DATASETS[dataset][split]
                source = Path(source) if isinstance(source, Path) else REFERENCES[dataset] / source
                destination = run / source.name
                destination.write_bytes(source.read_bytes())
                manifest["input_files"][str(destination.relative_to(output))] = assets.sha256(destination)
        assets.signed_write(manifest_path, manifest)
    # Kernel-held lock is released even if the process crashes; no concurrent writes.
    with (output / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is already running this asset backfill") from error
        for relative, digest in manifest["input_files"].items():
            if assets.sha256(output / relative) != digest:
                raise RuntimeError(f"Frozen center list changed: {relative}")
        manifest["status"] = "running"
        manifest.pop("error", None)
        assets.signed_write(manifest_path, manifest)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src"), environment.get("PYTHONPATH", "")])
        try:
            for dataset, method, seed, phase in planned:
                run = output / dataset
                log_path = run / "logs" / f"{method}_{phase}_seed_{seed}.log"
                log_path.parent.mkdir(exist_ok=True)
                command = [
                    sys.executable, str(ROOT / "scripts/benchmark/run_additional_spatial_baselines.py"),
                    "--dataset", dataset, "--method", method, "--seed", str(seed), "--phase", phase,
                    "--device", args.device, "--run-dir", str(run),
                    *(["--resume"] if args.resume else []), *(["--authorized"] if phase == "test" else []),
                ]
                print(f"RUN {dataset} {method} seed={seed} phase={phase} log={log_path}", flush=True)
                with log_path.open("a", encoding="utf-8") as handle:
                    subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
                key = f"{dataset}/{method}/{seed}/{phase}"
                if key not in manifest["completed_steps"]:
                    manifest["completed_steps"].append(key)
                assets.signed_write(manifest_path, manifest)
            for source, expected in manifest["preflight"]["dataset_files"].items():
                stat = Path(source).stat()
                if {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != expected:
                    raise RuntimeError(f"Source H5AD changed while models were being fitted: {source}")
            report = audit_results(output, args.datasets, args.methods, args.seeds, replay=args.replay)
            manifest.update(status="complete", finished_at=datetime.now().astimezone().isoformat(),
                            donor_bundles=report["donor_bundles"])
            assets.signed_write(manifest_path, manifest)
            print(f"BACKFILL_COMPLETE {output} donor_bundles={report['donor_bundles']}", flush=True)
        except Exception as error:
            manifest.update(status="failed", error=repr(error))
            assets.signed_write(manifest_path, manifest)
            raise


if __name__ == "__main__":
    main()
