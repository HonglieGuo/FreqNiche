"""Human-readable wide metric ledgers for FreqNiche and baselines."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


WIDE_FIELDS = [
    "run_id",
    "times",
    "dataset",
    "task",
    "method",
    "variant",
    "paradigm",
    "seed",
    "split",
    "evaluation_track",
    "clusterer",
    "cluster_parameter",
    "cluster_parameter_semantics",
    "kmeans_k",
    "mclust_components",
    "louvain_resolution",
    "leiden_resolution",
    "n_clusters",
    "n_clusters_sd",
    "embedding_dim",
    "ari",
    "ari_sd",
    "nmi",
    "nmi_sd",
    "worst_donor_ari",
    "worst_donor_ari_sd",
    "donor_ari_sd",
    "median_donor_ari",
    "asw",
    "asw_sd",
    "js_distance",
    "js_distance_sd",
    "mae",
    "mae_sd",
    "rmse",
    "rmse_sd",
    "spearman_rho",
    "spearman_rho_sd",
    "r2",
    "r2_sd",
    "cross_seed_ari",
    "cross_seed_ari_sd",
    "effective_rank",
    "effective_rank_sd",
    "local_purity",
    "local_purity_sd",
    "moran_i_top5_pcs",
    "moran_i_top5_pcs_sd",
    "moran_i_clusters",
    "moran_i_clusters_sd",
    "best_alpha",
    "elapsed_sec",
    "peak_gpu_mb",
    "peak_rss_mb",
    "status",
    "source_file",
    "asset_dir",
    "extra_metrics_json",
]

# Only the FreqNiche ledger records our model and training configuration.
# Baseline ledgers deliberately retain the original WIDE_FIELDS schema.
FREQNICHE_HPARAM_FIELDS = [
    "radius_um", "k_max", "rho", "p_center_mask", "m_max",
    "hidden_dim", "n_encoder_layers", "n_diffusion_layers", "n_heads",
    "diffusion_timesteps", "learning_rate", "weight_decay",
    "covariance_weight", "max_epochs", "actual_epochs", "patience", "batch_size", "min_delta",
]
# The explicit kmeans_k / mclust_components / louvain_resolution /
# leiden_resolution columns already carry every clusterer's chosen parameter.
# Keep the generic field only in the unchanged baseline ledger schema.
_FREQNICHE_PREFIX = WIDE_FIELDS[:WIDE_FIELDS.index("seed") + 1]
_FREQNICHE_METRICS = [
    field for field in WIDE_FIELDS[len(_FREQNICHE_PREFIX):]
    if field not in {"cluster_parameter", "extra_metrics_json"}
]
FREQNICHE_FIELDS = _FREQNICHE_PREFIX + FREQNICHE_HPARAM_FIELDS + _FREQNICHE_METRICS + ["extra_metrics_json"]


def format_times(value: Any = None) -> str:
    """Format a run time to local wall-clock year, month, day, hour and minute."""
    if value in (None, ""):
        moment = datetime.now().astimezone()
    elif isinstance(value, datetime):
        moment = value
    else:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return moment.strftime("%Y-%m-%d %H:%M")


def empty_row(**metadata: Any) -> dict[str, Any]:
    row = {field: "" for field in WIDE_FIELDS}
    row.update(metadata)
    row["times"] = format_times(row.get("times") or row.get("timestamp"))
    row.pop("timestamp", None)
    return row


def flatten_numeric(value: Any, prefix: str = "") -> Iterable[tuple[str, float]]:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_numeric(item, child)
    elif isinstance(value, list):
        return
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, float(value)


def _normalized(metric: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", metric.lower()).strip("_")


def metric_column(metric: str) -> str | None:
    key = _normalized(metric)
    # Per-donor details and configuration values are not aggregate endpoints.
    # A sorted recovery JSON used to assign donor_aris.D004 to the ARI column
    # before donor_macro_ari was visited, making scores depend on key order.
    if re.search(r"(?:^|_)(?:donor_aris|donor_nmis|per_donor|donor_results|config|model_config|request)(?:_|$)", key):
        return None
    is_sample_sd = key.endswith(("sample_sd", "_std", "_sd")) or key.startswith(("sample_sd_", "std_")) or "_std_" in key
    base = re.sub(r"(?:_sample_sd|_std|_sd)$", "", key)
    base = re.sub(r"^std_", "", base)

    if base in {"r", "r_star", "optimal_r_star", "locked_r_star", "resolution"} or base.endswith("_r_star"):
        return "leiden_resolution"
    if "cross_seed" in base and "ari" in base:
        return "cross_seed_ari_sd" if is_sample_sd else "cross_seed_ari"
    if "worst" in base and "ari" in base:
        return "worst_donor_ari_sd" if is_sample_sd else "worst_donor_ari"
    if "donor_ari_std" in key or key.endswith("donor_ari_sd"):
        return "donor_ari_sd"
    if "median_donor_ari" in key:
        return "median_donor_ari"
    if re.search(r"(?:^|_)ari(?:_|$)", base) or base in {"macroari", "meanari"}:
        return "ari_sd" if is_sample_sd else "ari"
    if re.search(r"(?:^|_)nmi(?:_|$)", base) or base in {"macronmi", "meannmi"}:
        return "nmi_sd" if is_sample_sd else "nmi"
    if "asw" in base or "silhouette" in base:
        return "asw_sd" if is_sample_sd else "asw"
    if "js_distance" in base or base.endswith("jsd"):
        return "js_distance_sd" if is_sample_sd else "js_distance"
    if "rmse" in base:
        return "rmse_sd" if is_sample_sd else "rmse"
    if "mae" in base:
        return "mae_sd" if is_sample_sd else "mae"
    if "spearman" in base:
        return "spearman_rho_sd" if is_sample_sd else "spearman_rho"
    if base == "r2" or base.endswith("_r2") or "macro_r2" in base:
        return "r2_sd" if is_sample_sd else "r2"
    if "effective_rank" in base:
        return "effective_rank_sd" if is_sample_sd else "effective_rank"
    if "local_purity" in base:
        return "local_purity_sd" if is_sample_sd else "local_purity"
    if "moran_i_top5" in base:
        return "moran_i_top5_pcs_sd" if is_sample_sd else "moran_i_top5_pcs"
    if "moran_i_cluster" in base:
        return "moran_i_clusters_sd" if is_sample_sd else "moran_i_clusters"
    if base in {"dim", "embedding_dim"}:
        return "embedding_dim"
    if ("cluster" in base or base.endswith("final_k")) and not any(token in base for token in ("moran", "cross_seed")):
        return "n_clusters_sd" if is_sample_sd else "n_clusters"
    if "best_alpha" in base:
        return "best_alpha"
    if "elapsed" in base and ("sec" in base or base == "elapsed"):
        return "elapsed_sec"
    if "gpu" in base and ("mb" in base or "mib" in base):
        return "peak_gpu_mb"
    if "rss" in base and ("mb" in base or "mib" in base):
        return "peak_rss_mb"
    return None


def assign_metrics(row: dict[str, Any], metrics: Iterable[tuple[str, float]]) -> dict[str, Any]:
    def endpoint_priority(item: tuple[str, float]) -> int:
        key = _normalized(item[0])
        # Prefer the requested donor-macro readout over other aggregate aliases.
        if re.search(r"(?:^|_)donor_macro_(?:ari|nmi)(?:_|$)", key):
            return 0
        if re.search(r"(?:^|_)macro_(?:ari|nmi)(?:_|$)", key):
            return 1
        if re.search(r"(?:^|_)mean_(?:ari|nmi)(?:_|$)", key):
            return 2
        return 3

    extra: dict[str, float] = {}
    for metric, value in sorted(metrics, key=endpoint_priority):
        column = metric_column(metric)
        if column is None:
            extra[metric] = value
        elif row.get(column, "") in ("", None):
            row[column] = value
        elif float(row[column]) != float(value):
            extra[metric] = value
    if extra:
        existing = row.get("extra_metrics_json")
        if existing:
            previous = json.loads(existing)
            previous.update(extra)
            extra = previous
        row["extra_metrics_json"] = json.dumps(extra, sort_keys=True, separators=(",", ":"))
    return row


def append_wide_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    manifest_path: Path | None = None,
    fieldnames: list[str] | None = None,
) -> None:
    """Create a result CSV with a header or append rows under a file lock."""
    fields = fieldnames if fieldnames is not None else WIDE_FIELDS
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    deadline = time.time() + 120.0
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"pid={os.getpid()} time={time.time()}\n".encode("utf-8"))
        except FileExistsError:
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for metric-ledger lock: {lock_path}")
            time.sleep(0.25)
    try:
        if path.exists() and path.stat().st_size:
            with path.open("r", newline="", encoding="utf-8-sig") as existing:
                header = next(csv.reader(existing), None)
            if header != fields:
                raise ValueError(f"Result CSV schema mismatch: {path}")
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if handle.tell() == 0:
                writer.writeheader()
            for row in rows:
                record = {field: row.get(field, "") for field in fields}
                record["times"] = format_times(record["times"] or row.get("timestamp"))
                writer.writerow(record)
            handle.flush()
            os.fsync(handle.fileno())
        if manifest_path is not None:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            if manifest_path.exists():
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            else:
                payload = {"manifest_version": "2.0-wide"}
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            with path.open("r", encoding="utf-8", newline="") as handle:
                row_count = sum(1 for _ in csv.DictReader(handle))
            payload.update(
                {
                    "ledger": f"{path.parent.name}/{path.name}",
                    "ledger_sha256": digest.hexdigest(),
                    "row_count": row_count,
                    "columns": fields,
                    "updated_at": datetime.now().astimezone().isoformat(),
                }
            )
            temporary = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary, manifest_path)
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def generic_json_rows(
    payload: dict[str, Any],
    *,
    run_id: str,
    dataset: str,
    source_file: str,
    asset_dir: str,
    default_method: str,
    default_variant: str = "",
) -> list[dict[str, Any]]:
    """Create compact rows for ordinary future run JSON outputs.

    Seed-indexed mappings become one row per seed. Unusual nested details are
    preserved in ``extra_metrics_json`` rather than generating a long table.
    """

    common = {
        "run_id": run_id,
        "dataset": dataset,
        "task": str(payload.get("task", "")),
        "method": str(payload.get("method") or payload.get("method_name") or default_method),
        "variant": str(payload.get("variant") or payload.get("method_variant") or default_variant),
        "paradigm": str(payload.get("paradigm", "")),
        "split": (
            str(payload["split"])
            if payload.get("split") in {"train", "validation", "test"}
            else "test" if (
                "test" in Path(source_file).name.lower()
                or isinstance(payload.get("common_readout"), dict)
                and "test_evaluation" in payload["common_readout"]
            ) else "validation"
        ),
        "clusterer": str(payload.get("clusterer", "")),
        "status": str(payload.get("status", "")),
        "source_file": source_file,
        "asset_dir": asset_dir,
    }
    if default_method == "FreqNiche" and "freqniche" in str(common["method"]).lower():
        common["variant"] = common["variant"] or "Full"
        common["method"] = "FreqNiche"
    seed_mapping = payload.get("per_seed_results") or payload.get("per_seed")
    if isinstance(seed_mapping, dict) and seed_mapping:
        rows = []
        for seed, metrics in seed_mapping.items():
            if not isinstance(metrics, dict):
                continue
            row = empty_row(**common, seed=seed)
            assign_metrics(row, flatten_numeric(metrics))
            if row["leiden_resolution"] not in ("", None) and not row["clusterer"]:
                row["clusterer"] = "leiden"
                row["cluster_parameter"] = row["leiden_resolution"]
                row["cluster_parameter_semantics"] = "resolution"
            rows.append(row)
        if rows:
            return rows
    row = empty_row(**common, seed=payload.get("seed", ""))
    assign_metrics(row, flatten_numeric(payload))
    if row["leiden_resolution"] not in ("", None) and not row["clusterer"]:
        row["clusterer"] = "leiden"
        row["cluster_parameter"] = row["leiden_resolution"]
        row["cluster_parameter_semantics"] = "resolution"
    return [row]


MEAN_SD_PATTERN = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:±|\+/-)\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)


def generic_csv_rows(
    path: Path,
    *,
    run_id: str,
    dataset: str,
    source_file: str,
    asset_dir: str,
    default_method: str,
    default_variant: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    for source in source_rows:
        lookup = {key.lower(): value for key, value in source.items()}
        method = (
            lookup.get("method")
            or lookup.get("method_name")
            or default_method
        )
        variant = lookup.get("method_variant") or lookup.get("variant") or default_variant
        if default_method == "FreqNiche" and "freqniche" in method.lower():
            variant = variant or "Full"
            method = "FreqNiche"
        seed = lookup.get("seed", "")
        row = empty_row(
            run_id=run_id,
            dataset=dataset,
            task="spatial_domain",
            method=method,
            variant=variant,
            paradigm=lookup.get("paradigm", ""),
            seed=seed,
            split="test",
            clusterer=lookup.get("clusterer", ""),
            source_file=source_file,
            asset_dir=asset_dir,
            status=lookup.get("status", ""),
        )
        numeric: list[tuple[str, float]] = []
        for key, value in source.items():
            if key.lower() in {"method", "method_name", "method_variant", "variant", "paradigm", "seed", "status"}:
                continue
            text = str(value).strip()
            match = MEAN_SD_PATTERN.match(text)
            if match:
                numeric.append((key, float(match.group(1))))
                numeric.append((key + ".sample_sd", float(match.group(2))))
                continue
            try:
                numeric.append((key, float(text)))
            except ValueError:
                if key.lower() in {"optimal_r_star", "locked_r_stars"} and text:
                    row["clusterer"] = "leiden"
                    row["leiden_resolution"] = text
                    row["cluster_parameter"] = text
                    row["cluster_parameter_semantics"] = "seed_specific_resolution"
        assign_metrics(row, numeric)
        if row["leiden_resolution"] not in ("", None) and not row["clusterer"]:
            row["clusterer"] = "leiden"
            row["cluster_parameter"] = row["leiden_resolution"]
            row["cluster_parameter_semantics"] = "resolution"
        rows.append(row)
    return rows
