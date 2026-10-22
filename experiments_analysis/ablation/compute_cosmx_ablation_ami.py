"""Compute reference-label AMI from frozen CosMx ablation assignments; no fitting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    audit_path = run / "asset_audit.json"
    audit = json.loads(audit_path.read_text())
    status = json.loads((run / "status.json").read_text())
    if status["status"] != "complete" or audit["status"] != "passed" or audit["packages"] != 18:
        raise ValueError("Requires a complete, audited 18-package run")
    centers = read_rows(run / "inputs/test/centers.csv")
    expected = [(r["cell_barcode"], r["reference_niche"]) for r in centers]
    if len(expected) != 10000:
        raise ValueError("Unexpected test-center count")
    packages = {(p["variant"], p["seed"]): p for p in audit["assets"]}
    original = json.loads((run / "cosmx_ablation_summary.json").read_text())
    seed_results, summary, sources = [], [], []
    for config in original["results"]:
        variant = config["variant"]
        values = []
        for seed in original["seeds"]:
            directory = run / variant / f"seed_{seed}"
            path = directory / "test_assignments.csv"
            digest = sha(path)
            if digest != packages[(variant, seed)]["files"][path.name]:
                raise ValueError(f"Assignment hash mismatch: {path}")
            result_path = directory / "test_result.json"
            if sha(result_path) != packages[(variant, seed)]["files"][result_path.name]:
                raise ValueError(f"Test-result hash mismatch: {result_path}")
            result = json.loads(result_path.read_text())
            rows = read_rows(path)
            if [(r["cell_barcode"], r["reference_niche"]) for r in rows] != expected:
                raise ValueError(f"Center order or reference mismatch: {path}")
            reference = [r["reference_niche"] for r in rows]
            predicted = [r["predicted_cluster"] for r in rows]
            for key, scorer in [("ari", adjusted_rand_score), ("nmi", normalized_mutual_info_score)]:
                if not np.isclose(scorer(reference, predicted), result[key], atol=1e-12, rtol=0):
                    raise ValueError(f"Original {key} replay mismatch: {path}")
            ami = float(adjusted_mutual_info_score(reference, predicted, average_method="arithmetic"))
            if not np.isfinite(ami):
                raise ValueError(f"Nonfinite AMI: {path}")
            values.append(ami)
            seed_results.append({"variant": variant, "seed": seed, "ami": ami})
            sources.append({"variant": variant, "seed": seed,
                            "assignment_sha256": digest, "test_result_sha256": sha(result_path)})
        summary.append({"variant": variant, "ami_mean": float(np.mean(values)),
                        "ami_std": float(np.std(values, ddof=1))})
    output = run / "additional_metrics/ami"
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "seed_results.csv", seed_results)
    write_rows(output / "summary.csv", summary)
    provenance = {"metric": "adjusted_mutual_info_score", "average_method": "arithmetic",
                  "comparison": "reference_niche versus predicted_cluster, within each seed",
                  "test_donor": "Lung9", "n_test_centers": len(centers),
                  "standard_deviation": "sample SD across three seeds (ddof=1)",
                  "training_or_clustering_performed": False,
                  "asset_audit_sha256": sha(audit_path), "script_sha256": sha(Path(__file__)),
                  "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
                  "sources": sources, "outputs": {name: sha(output / name) for name in
                                                    ["seed_results.csv", "summary.csv"]}}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "results": summary}, indent=2))


if __name__ == "__main__":
    main()
