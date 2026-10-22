"""Run the unchanged Human Skin Leiden readout in a fresh CPU process.

The worker receives a saved Validation embedding only. A native clustering
failure cannot destroy training state, and the readout can be rerun separately.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_readout(report: dict, embedding_path: Path, seed: int) -> None:
    import numpy as np

    payload = {key: value for key, value in report.items() if key != "self_signature_sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    if report.get("self_signature_sha256") != digest:
        raise RuntimeError("Validation readout self-check mismatch")
    z = np.load(embedding_path, mmap_mode="r")
    if report.get("seed") != seed or report.get("embedding_sha256") != file_sha256(embedding_path):
        raise RuntimeError("Validation readout does not match its embedding/seed")
    if report.get("worker_sha256") != file_sha256(Path(__file__)):
        raise RuntimeError("Validation worker changed after this readout")
    engine = Path(__file__).with_name("evaluation.py")
    if report.get("evaluation_sha256") != file_sha256(engine):
        raise RuntimeError("Validation evaluation engine changed after this readout")
    versions = {name: importlib.metadata.version(name) for name in ("scanpy", "igraph", "leidenalg", "numpy", "scipy")}
    if report.get("versions") != versions:
        raise RuntimeError("Clustering dependencies changed after this readout")
    labels = report.get("pred_clusters", [])
    if len(labels) != len(z) or report.get("embedding_shape") != list(z.shape):
        raise RuntimeError("Validation readout shape mismatch")
    r_star = report.get("r_star")
    stability = report.get("stability", {})
    score = stability.get("stability_scores", {}).get(str(r_star))
    if r_star not in (0.2, 0.4, 0.6, 0.8, 1.0) or score is None or not np.isfinite(score):
        raise RuntimeError("Invalid Validation resolution/stability result")


def isolated_validation(embedding_path: Path, seed: int, report_path: Path) -> dict:
    """Reuse a verified readout or run it once; never change clustering backends."""
    embedding_path, report_path = Path(embedding_path).resolve(), Path(report_path).resolve()
    if not report_path.exists():
        command = [sys.executable, "-X", "faulthandler", "-u", "-m",
                   "src.freqniche_st.validation_worker", "--embedding", str(embedding_path),
                   "--seed", str(seed), "--output", str(report_path)]
        process = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], check=False)
        if process.returncode:
            raise RuntimeError(
                f"Isolated Validation worker exited with {process.returncode}. "
                "The checkpoint and embedding are retained. Recover this run with "
                "scripts/benchmark/resume_human_skin_run.py --run-dir <run> --recover-only."
            )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    verify_readout(report, embedding_path, seed)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite Validation readout: {args.output}")

    import anndata as ad
    import numpy as np
    import scanpy as sc
    from .evaluation import compute_unsupervised_clustering_stability

    input_sha = file_sha256(args.embedding)
    z = np.load(args.embedding)
    if z.ndim != 2 or len(z) < 20 or not np.isfinite(z).all():
        raise ValueError("Validation embedding must be a finite two-dimensional array")
    print(f"[VALIDATION WORKER] fresh process pid={os.getpid()}, seed={args.seed}, shape={z.shape}", flush=True)
    r_star, stability = compute_unsupervised_clustering_stability(z, seed=args.seed, verbose=True)
    print(f"[VALIDATION WORKER] final clustering at r*={r_star}", flush=True)
    cells = ad.AnnData(X=z)
    sc.pp.neighbors(cells, n_neighbors=15, use_rep="X")
    sc.tl.leiden(cells, resolution=r_star, key_added="leiden", random_state=args.seed)
    if file_sha256(args.embedding) != input_sha:
        raise RuntimeError("Validation embedding changed while it was being evaluated")
    report = {
        "seed": args.seed, "embedding_shape": list(z.shape), "embedding_sha256": input_sha,
        "r_star": float(r_star), "stability": stability,
        "pred_clusters": cells.obs["leiden"].astype(str).tolist(),
        "worker_sha256": file_sha256(Path(__file__)),
        "evaluation_sha256": file_sha256(Path(__file__).with_name("evaluation.py")),
        "versions": {name: importlib.metadata.version(name) for name in ("scanpy", "igraph", "leidenalg", "numpy", "scipy")},
        "protocol": "original Leiden/leidenalg readout; raw embedding; 15 neighbors; 5 resolutions; 10x80% resampling; seed+b",
    }
    report["self_signature_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[VALIDATION WORKER] completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
