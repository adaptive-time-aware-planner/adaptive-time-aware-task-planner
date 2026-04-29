"""Helpers for aggregating repeated analysis outputs across multiple task folders."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


def resolve_source_paths(
    *,
    single_path: Path | None,
    analysis_root: Path | None,
    task_folders: Sequence[str] | None,
    filename: str,
) -> list[Path]:
    """Resolve one or many analysis artifact paths.

    When ``analysis_root`` is provided, expects artifacts under
    ``analysis_root/<task_folder>/<filename>`` and returns one path per
    requested task folder. If no folders are specified, discovers all direct
    subdirectories that contain the requested file.
    """

    if analysis_root is None:
        if single_path is None:
            raise ValueError("Either single_path or analysis_root must be provided.")
        return [single_path]

    root = Path(analysis_root)
    if task_folders:
        folder_names = [str(task_folder) for task_folder in task_folders]
    else:
        folder_names = [
            child.name
            for child in sorted(root.iterdir())
            if child.is_dir() and (child / filename).is_file()
        ]

    paths = [root / folder_name / filename for folder_name in folder_names]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing analysis artifacts: {rendered}")
    return paths


def load_json_objects(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load one JSON object per path."""

    payloads: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object at {path}")
        payloads.append(payload)
    return payloads


def mean_std(values: Sequence[float]) -> tuple[float | None, float | None]:
    """Return sample mean/std, with std=0 for a single value."""

    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def summarize_metric(values: Sequence[float | None]) -> tuple[float | None, float | None]:
    """Compute mean/std after dropping missing values."""

    present = [float(value) for value in values if value is not None]
    return mean_std(present)


def collect_metric_values(
    payloads: Sequence[Mapping[str, Any]],
    *,
    setting_key: str,
    case_name: str,
    field: str,
) -> list[float]:
    """Collect one numeric metric across multiple summary payloads."""

    values: list[float] = []
    for payload in payloads:
        metrics = payload.get(setting_key, {})
        if not isinstance(metrics, Mapping):
            continue
        case_metrics = metrics.get(case_name, {})
        if not isinstance(case_metrics, Mapping):
            continue
        value = case_metrics.get(field)
        if value is None:
            continue
        values.append(float(value))
    return values
