"""Read-only environment and asset preflight for one execution suite."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess

import numpy as np

import clusterer_common as common


def main() -> None:
    parser = argparse.ArgumentParser(description="Clusterer sensitivity preflight")
    parser.add_argument("--suite", choices=("freqniche", "baselines"), required=True)
    args = parser.parse_args()

    for module in ("scanpy", "anndata", "igraph", "leidenalg", "sklearn"):
        importlib.import_module(module)
    r_check = subprocess.run(
        ["Rscript", "-e", "cat(requireNamespace('mclust', quietly=TRUE))"],
        capture_output=True, text=True, check=False,
    )
    if r_check.returncode != 0 or r_check.stdout.strip().upper() != "TRUE":
        raise RuntimeError(f"R mclust unavailable: {r_check.stdout} {r_check.stderr}")

    folds, manifest = common.load_folds()
    if len(folds) != 10 or manifest["n_cells"] != 8000:
        raise RuntimeError("Shared fold preflight failed")
    records = []
    for pair in common.registry(args.suite):
        for split, path in (("validation", pair.validation_path), ("test", pair.test_path)):
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if values.ndim != 2 or values.shape[0] != 8000:
                raise RuntimeError(f"Bad {split} embedding: {path} {values.shape}")
            records.append((pair.method, pair.seed_label, split, tuple(values.shape), common.compute_sha256(path)))
    if common.selection_dir(args.suite).exists() or common.evaluation_dir(args.suite).exists():
        raise FileExistsError("Formal suite output already exists")
    versions = common.assert_python_environment_matches_lock()
    print(f"Clusterer preflight passed: suite={args.suite}, embedding_assets={len(records)}")
    print("Environment versions:", json.dumps(versions, sort_keys=True))
    for method, seed, split, shape, sha in records:
        print(f"  {method} seed={seed} {split} shape={shape} sha256={sha}")


if __name__ == "__main__":
    main()
