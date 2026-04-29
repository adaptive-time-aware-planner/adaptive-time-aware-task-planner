"""Comparison helpers for offline experiment reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _load_report(path: Path) -> dict[str, Any]:
    """Load an offline experiment report from JSON."""

    return json.loads(path.read_text(encoding="utf-8"))


def compare_result_files(before_path: Path, after_path: Path) -> dict[str, Any]:
    """Compare two offline experiment result files.

    Args:
        before_path: Baseline report path.
        after_path: Candidate report path.

    Returns:
        Structured comparison payload.
    """

    before_report = _load_report(before_path)
    after_report = _load_report(after_path)
    before_summary = before_report.get("summary_by_setting", {})
    after_summary = after_report.get("summary_by_setting", {})
    setting_deltas: dict[str, dict[str, Any]] = {}
    for setting in sorted(set(before_summary) | set(after_summary)):
        before_metrics = before_summary.get(setting, {})
        after_metrics = after_summary.get(setting, {})
        before_schedule = before_metrics.get("avg_schedule_time")
        after_schedule = after_metrics.get("avg_schedule_time")
        before_compute = before_metrics.get("avg_compute_time")
        after_compute = after_metrics.get("avg_compute_time")
        setting_deltas[setting] = {
            "before": before_metrics,
            "after": after_metrics,
            "schedule_time_delta": (
                None
                if before_schedule is None or after_schedule is None
                else after_schedule - before_schedule
            ),
            "compute_time_delta": (
                None
                if before_compute is None or after_compute is None
                else after_compute - before_compute
            ),
        }

    before_best = before_report.get("comparison", {}).get("best_by_task", {})
    after_best = after_report.get("comparison", {}).get("best_by_task", {})
    task_best_deltas: dict[str, dict[str, Any]] = {}
    for instruction in sorted(set(before_best) | set(after_best)):
        before_entry = before_best.get(instruction, {})
        after_entry = after_best.get(instruction, {})
        before_schedule = before_entry.get("final_schedule_time")
        after_schedule = after_entry.get("final_schedule_time")
        task_best_deltas[instruction] = {
            "before_best": before_entry,
            "after_best": after_entry,
            "best_schedule_delta": (
                None
                if before_schedule is None or after_schedule is None
                else after_schedule - before_schedule
            ),
        }

    improved_settings = [
        setting
        for setting, payload in setting_deltas.items()
        if payload["schedule_time_delta"] is not None
        and payload["schedule_time_delta"] < 0
    ]
    regressed_settings = [
        setting
        for setting, payload in setting_deltas.items()
        if payload["schedule_time_delta"] is not None
        and payload["schedule_time_delta"] > 0
    ]
    return {
        "before_path": str(before_path),
        "after_path": str(after_path),
        "setting_deltas": setting_deltas,
        "task_best_deltas": task_best_deltas,
        "improved_settings": improved_settings,
        "regressed_settings": regressed_settings,
    }


def save_comparison_report(report: Mapping[str, Any], output_path: Path) -> None:
    """Persist a comparison report as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
