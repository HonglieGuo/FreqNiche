"""Persist and verify donor-specific transductive spatial-baseline assets."""
from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module, metadata
from pathlib import Path
import hashlib
import json
import os
import random
import tempfile

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def artifact_code_binding(method: str) -> dict:
    from . import legacy_spatial
    from .common import source_tree_sha256
    return {
        "adapter_sha256": sha256(Path(legacy_spatial.__file__)),
        "artifact_writer_sha256": sha256(Path(__file__)),
        "vendor_source_sha256": source_tree_sha256(legacy_spatial.SOURCES[method]),
    }


def code_binding_matches(saved: dict | None, current: dict) -> bool:
    """Accept exact code, or an explicitly recorded serialization-only repair."""
    if saved == current:
        return True
    registry = Path(__file__).with_name("spatial_artifact_compatibility.json")
    if not registry.is_file():
        return False
    return any(saved == item["before"] and current == item["after"]
               for item in json.loads(registry.read_text(encoding="utf-8"))["bindings"])


def h5ad_compatible_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize nullable strings on a serialization copy; preserve values and order."""
    result = frame.copy(deep=True)
    if isinstance(result.index.dtype, pd.StringDtype):
        result.index = pd.Index(result.index.to_numpy(dtype=object), dtype=object, name=result.index.name)
    for name in result.columns:
        values = result[name]
        if isinstance(values.dtype, pd.StringDtype):
            result[name] = pd.Categorical(values)
        values = result[name]
        if isinstance(values.dtype, pd.CategoricalDtype) and isinstance(values.cat.categories.dtype, pd.StringDtype):
            result[name] = pd.Categorical.from_codes(
                values.cat.codes,
                categories=pd.Index(values.cat.categories.to_numpy(dtype=object), dtype=object),
                ordered=values.cat.ordered,
            )
    return result


def signed_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(payload)
    value["self_signature_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def signed_read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    signature = value.pop("self_signature_sha256")
    if signature != hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest():
        raise RuntimeError(f"Invalid artifact signature: {path}")
    return value


def data_fingerprint(data: ad.AnnData) -> str:
    """Bind unlabeled expression, cell/gene ordering, coordinates and sample identity."""
    h = hashlib.sha256()
    matrix = data.X.tocsr() if sp.issparse(data.X) else np.asarray(data.X)
    arrays = [matrix.data, matrix.indices, matrix.indptr] if sp.issparse(matrix) else [matrix]
    for value in [*arrays, np.asarray(data.obsm["spatial"])]:
        value = np.ascontiguousarray(value)
        h.update(str((value.shape, value.dtype.str)).encode())
        h.update(memoryview(value).cast("B"))
    for values in [data.obs_names, data.var_names]:
        h.update(json.dumps(list(map(str, values)), ensure_ascii=False).encode("utf-8"))
    for key in ["donor_id", "sample_id"]:
        if key in data.obs:
            h.update(json.dumps(data.obs[key].astype(str).tolist(), ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


@contextmanager
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def cpu_tensor(value, sparse=False):
    value = value.detach().cpu().clone() if isinstance(value, torch.Tensor) else torch.as_tensor(value).clone()
    return value.to_sparse().coalesce() if sparse and value.layout == torch.strided else value


def replay_native_embedding(directory: Path, device: str | None = None) -> np.ndarray:
    """Reload learned weights and exact graph/features; no training or PCA refitting."""
    directory = Path(directory)
    saved = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    inputs = torch.load(directory / "model_inputs.pt", map_location="cpu", weights_only=True)
    method = saved["method"]
    from .legacy_spatial import _source
    _source(method)
    device = device or saved["effective_device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[name] = value.to(device)
    if method == "sedr":
        model = import_module("SEDR.SEDR_module").SEDR_module(**saved["init_kwargs"])
    elif method == "graphst":
        model = import_module("GraphST.model").Encoder_sparse(
            **saved["init_kwargs"], graph_neigh=inputs["graph_neigh"].to_dense()
        )
    elif method == "spagcn":
        model = import_module("SpaGCN.models").simple_GC_DEC(**saved["init_kwargs"])
        model.mu = torch.nn.Parameter(torch.empty_like(saved["state_dict"]["mu"]))
    else:
        raise ValueError(method)
    model = model.to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        if method == "sedr":
            # Author code randomly masks even during process()/eval(). Preserve the
            # actual sampled mask, rather than silently sampling a new embedding.
            x = inputs["X"].clone()
            x[inputs["mask_nodes"]] += model.enc_mask_token
            mu, _, feat = model.encode(x, inputs["adj_norm"])
            result = torch.cat((feat, mu), dim=1)
        elif method == "graphst":
            result = model(inputs["features"], inputs["features_a"], inputs["adj"])[1]
            result = F.normalize(result, p=2, dim=1)
        else:
            if (directory / "graph_sparse.npz").exists():
                graph = sp.load_npz(directory / "graph_sparse.npz").toarray()
            else:
                with np.load(directory / "graph_dense.npz", allow_pickle=False) as archive:
                    graph = archive["adj_exp"]
            result = model(
                inputs["embed"].float(), torch.as_tensor(graph, dtype=torch.float32, device=device)
            )[0]
    return result.detach().cpu().numpy().astype(np.float32)


def verify_artifact(directory: Path, *, replay=False, expected_method=None,
                    expected_seed=None, expected_fingerprint=None) -> dict:
    directory = Path(directory)
    manifest = signed_read(directory / "artifact_manifest.json")
    required = {"model.pt", "model_inputs.pt", "embedding.npy", "preprocess_pca.joblib",
                "processed_input.h5ad", "cells.csv", "input_genes.csv", manifest["native_embedding_file"]}
    if not required.issubset(manifest["files"]):
        raise RuntimeError(f"Required fitted assets were not recorded: {directory}")
    if manifest["method"] == "spagcn" and not {"graph_sparse.npz", "graph_dense.npz"}.intersection(manifest["files"]):
        raise RuntimeError(f"SpaGCN graph is missing: {directory}")
    for key, expected in [("method", expected_method), ("seed", expected_seed),
                          ("input_fingerprint", expected_fingerprint)]:
        if expected is not None and manifest[key] != expected:
            raise RuntimeError(f"Artifact {key} mismatch: {directory}")
    for relative, record in manifest["files"].items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()):
            raise RuntimeError(f"Invalid artifact path: {relative}")
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Missing/corrupt artifact: {path}")
    z = np.load(directory / "embedding.npy", allow_pickle=False)
    cells = pd.read_csv(directory / "cells.csv", dtype=str, keep_default_na=False)
    if z.shape != tuple(manifest["embedding_shape"]) or len(cells) != len(z) or not np.isfinite(z).all():
        raise RuntimeError(f"Embedding/cell-order mismatch: {directory}")
    order_hash = hashlib.sha256(json.dumps(cells["cell_barcode"].tolist()).encode()).hexdigest()
    if order_hash != manifest["cell_order_sha256"]:
        raise RuntimeError(f"Cell-order signature mismatch: {directory}")
    if replay:
        with preserve_rng():
            restored = replay_native_embedding(directory)
        native = np.load(directory / manifest["native_embedding_file"], allow_pickle=False)
        if not np.allclose(restored, native, atol=1e-4, rtol=1e-4):
            raise RuntimeError(f"Reloaded model does not reproduce native embedding: {directory}")
    return manifest


def save_artifact(directory: Path, *, method, seed, source, processed, model,
                  init_kwargs, model_inputs, embedding, native_embedding,
                  pca, protocol, code_binding, graph=None, smoke=False) -> dict:
    """Publish an atomic complete bundle, after hash and actual weight-reload checks."""
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite fitted model: {directory}")
    with preserve_rng(), tempfile.TemporaryDirectory(prefix=f".{directory.name}.", dir=directory.parent) as temporary:
        staging = Path(temporary)
        effective_device = str(next(model.parameters()).device)
        torch.save({
            "schema_version": 1, "method": method, "seed": int(seed),
            "model_class": f"{type(model).__module__}.{type(model).__name__}",
            "init_kwargs": init_kwargs, "effective_device": effective_device,
            "state_dict": {key: cpu_tensor(value) for key, value in model.state_dict().items()},
        }, staging / "model.pt")
        torch.save(model_inputs, staging / "model_inputs.pt")
        np.save(staging / "embedding.npy", np.asarray(embedding, dtype=np.float32), allow_pickle=False)
        native_file = "embedding.npy"
        if method == "graphst":
            native_file = "native_embedding.npy"
            np.save(staging / native_file, np.asarray(native_embedding, dtype=np.float32), allow_pickle=False)
        joblib.dump(pca, staging / "preprocess_pca.joblib")
        small = ad.AnnData(X=processed.X.copy(), obs=h5ad_compatible_metadata(processed.obs),
                          var=h5ad_compatible_metadata(processed.var))
        small.obsm["spatial"] = np.asarray(processed.obsm["spatial"]).copy()
        if set(small.obs.columns) - {"donor_id", "sample_id"}:
            raise RuntimeError("Reference labels must not enter a fitted-model artifact")
        small.write_h5ad(staging / "processed_input.h5ad", compression="gzip")
        cells = pd.DataFrame({"cell_barcode": source.obs_names.astype(str)})
        for name in ["donor_id", "sample_id"]:
            if name in source.obs:
                cells[name] = source.obs[name].astype(str).to_numpy()
        cells.to_csv(staging / "cells.csv", index=False)
        pd.DataFrame({"gene": source.var_names.astype(str)}).to_csv(staging / "input_genes.csv", index=False)
        if graph is not None:
            if np.count_nonzero(graph) < graph.size / 4:
                sp.save_npz(staging / "graph_sparse.npz", sp.csr_matrix(graph))
            else:
                np.savez_compressed(staging / "graph_dense.npz", adj_exp=graph)
        files = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                 for p in staging.iterdir() if p.is_file()}
        versions = {}
        for package in ["torch", "numpy", "scipy", "scikit-learn", "scanpy", "anndata"]:
            versions[package] = metadata.version(package)
        manifest = {
            "schema_version": 1, "method": method, "seed": int(seed), "smoke": bool(smoke),
            "scope": "transductive_donor_wise", "protocol": protocol,
            "code_binding": code_binding,
            "input_fingerprint": data_fingerprint(source),
            "embedding_shape": list(embedding.shape), "native_embedding_file": native_file,
            "readout": "pca_fit_transform" if method == "graphst" else "identity",
            "cell_order_sha256": hashlib.sha256(json.dumps(cells["cell_barcode"].tolist()).encode()).hexdigest(),
            "versions": versions, "files": files,
            "preprocessing_note": "Actual fitted PCA and exact model input tensors are saved. "
                "GraphST readout uses fit_transform; use embedding.npy for the original coordinates.",
        }
        signed_write(staging / "artifact_manifest.json", manifest)
        verify_artifact(staging, replay=True, expected_method=method, expected_seed=seed)
        if artifact_code_binding(method) != code_binding:
            raise RuntimeError("Source code changed while fitting; fitted assets were not published")
        os.replace(staging, directory)
    return verify_artifact(directory)


def save_assignments(directory: Path, labels, resolution, *, reference=None) -> None:
    directory = Path(directory)
    manifest = verify_artifact(directory)
    cells = pd.read_csv(directory / "cells.csv", dtype=str, keep_default_na=False)
    if len(labels) != len(cells):
        raise ValueError("Assignment row count differs from embedding rows")
    cells["predicted_cluster"] = np.asarray(labels).astype(str)
    if reference is not None:
        cells["reference_label"] = np.asarray(reference).astype(str)
    temporary = directory / f".assignments.csv.tmp.{os.getpid()}"
    cells.to_csv(temporary, index=False)
    os.replace(temporary, directory / "assignments.csv")
    p = directory / "assignments.csv"
    manifest["files"][p.name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    manifest["clustering_resolution"] = float(resolution)
    signed_write(directory / "artifact_manifest.json", manifest)
    verify_artifact(directory)
