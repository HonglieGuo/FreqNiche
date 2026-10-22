"""Portable project and runtime paths; filesystem aliases are never required."""
from __future__ import annotations

import json
import os
from pathlib import Path, PureWindowsPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_MIGRATIONS_FILE = REPOSITORY_ROOT / "configs/assets/path_migrations.json"
_ASSET_MIGRATIONS = json.loads(_MIGRATIONS_FILE.read_text(encoding="utf-8"))
_PROJECT_FOLDERS = {"outputs", "baselines", "datasets", "experiments_analysis",
                    "Paper", "configs", "src", "scripts", "tests", "environments", "data"}


def project_path(value: str | Path) -> Path:
    """Read a recorded asset path under the current checkout, including old manifests.

    Historical metadata stays byte-for-byte unchanged. Only its path is rebased
    at read time; asset hashes are still checked by the caller.
    """
    raw = str(value).replace("\\", "/")
    candidate = Path(raw)
    absolute = candidate.is_absolute() or PureWindowsPath(raw).is_absolute()
    if absolute and candidate.exists():
        return candidate
    parts = raw.split("/")
    if absolute:
        start = next((i for i, part in enumerate(parts) if part in _PROJECT_FOLDERS), None)
        if start is None:
            return candidate
        raw = "/".join(parts[start:])
    for _ in range(len(_ASSET_MIGRATIONS) + 1):
        changed = False
        for previous, current in sorted(_ASSET_MIGRATIONS.items(), key=lambda item: -len(item[0])):
            if raw == previous or raw.startswith(previous + "/"):
                raw = current + raw[len(previous):]
                changed = True
                break
        if not changed:
            break
    else:
        raise ValueError(f"Asset path migration cycle: {value}")
    return REPOSITORY_ROOT / raw


def resolve_run_output(default_relative: str) -> Path:
    """Use the canonical run override, or a real project-relative asset folder."""
    configured = os.environ.get("FREQNICHE_RUN_DIR")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else REPOSITORY_ROOT / path
    return REPOSITORY_ROOT / default_relative


def resolve_baseline_output(default_relative: str) -> Path:
    """Keep new baseline runs isolated; archived baselines use their real folder."""
    if os.environ.get("FREQNICHE_RUN_DIR"):
        return resolve_run_output(default_relative) / "baselines"
    return REPOSITORY_ROOT / default_relative


def resolve_dataset_path(environment_key: str, default_relative: str) -> Path:
    configured = os.environ.get(environment_key)
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else REPOSITORY_ROOT / path
    return REPOSITORY_ROOT / default_relative


def validated_source_bindings(recorded: dict[str, str]) -> dict[str, str]:
    """Verify archived inputs after relocation, including audited path-only edits."""
    import hashlib
    changes_file = REPOSITORY_ROOT / "configs/assets/source_path_migrations.json"
    changes = json.loads(changes_file.read_text(encoding="utf-8")) if changes_file.exists() else {}
    result = {}
    for name, expected in recorded.items():
        path = project_path(name)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        change = changes.get(relative, {})
        if actual != expected and not (
            path.suffix in {".py", ".sh"}
            and change.get("before") == expected and change.get("after") == actual
        ):
            raise ValueError(f"Recorded asset changed: {relative}")
        result[relative] = actual
    return result
