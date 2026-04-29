"""Oracle reference storage and comparison helpers for offline experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from src.utils.config.constants import RESULT_PATH

ORACLE_REFERENCE_SCHEMA_VERSION = "offline_oracle_reference.v1"
DEFAULT_ORACLE_REFERENCE_DIR = RESULT_PATH / "offline_oracle_reference"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class OracleReferenceCoverage:
    """Coverage summary for oracle references used by a baseline report.

    Args:
        matched_instructions: Number of selected instructions with a reference file.
        total_selected_instructions: Total requested instruction count.
        exact_instructions: Number of matched instructions proven exact.
        missing_task_keys: Stable task identifiers that have no reference file.
    """

    matched_instructions: int
    total_selected_instructions: int
    exact_instructions: int
    missing_task_keys: list[str]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return {
            "matched_instructions": self.matched_instructions,
            "total_selected_instructions": self.total_selected_instructions,
            "exact_instructions": self.exact_instructions,
            "missing_task_keys": list(self.missing_task_keys),
        }


def resolve_oracle_reference_dir(path_like: str | None) -> Path:
    """Return the directory used for persisted oracle reference files."""

    if path_like is None:
        return DEFAULT_ORACLE_REFERENCE_DIR
    candidate = Path(path_like).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def build_oracle_task_key(case_name: str, instruction: str) -> str:
    """Return a stable task key shared by oracle and baseline reports."""

    return f"{case_name}::{instruction}"


def build_offline_task_directory(
    base_dir: Path,
    *,
    task_folder_name: str,
    scene_name: str,
    case_name: str,
    instruction_name: str,
) -> Path:
    """Return the shared task directory for oracle and baseline artifacts."""

    instruction_stem = Path(instruction_name).stem
    return (
        base_dir
        / task_folder_name
        / scene_name
        / case_name
        / instruction_stem
    )


def build_oracle_reference_output_path(
    base_dir: Path,
    *,
    task_folder_name: str,
    scene_name: str,
    case_name: str,
    instruction_name: str,
) -> Path:
    """Construct the per-instruction oracle reference file path."""

    return (
        base_dir
        / task_folder_name
        / scene_name
        / case_name
        / Path(instruction_name).name
    )


def save_oracle_reference_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    base_dir: Path,
    task_folder_name: str,
    scene_name: str,
) -> None:
    """Persist oracle rows as one JSON file per instruction."""

    for row in rows:
        persisted_row: dict[str, Any] = {
            "meta_data": dict(row["meta_data"]),
            "saved_time": row["saved_time"],
            "approach": row["approach"],
            "scene_name": row["scene_name"],
            "plans": [dict(plan) for plan in row["plans"]],
            "computation_time": row["computation_time"],
            "scheduler_makespan": row["scheduler_makespan"],
            "timing_success_rate_sched": row["timing_success_rate_sched"],
            "detail_log": dict(row["detail_log"]),
        }
        for key in (
            "case",
            "instruction",
            "completed",
            "abort_reason",
            "optimal_sequence",
            "optimal_schedule_time",
            "exact",
            "timeout_hit",
            "search_nodes",
            "pruned_nodes",
            "idle_advances",
        ):
            if key in row:
                persisted_row[key] = row[key]
        output_path = build_oracle_reference_output_path(
            base_dir,
            task_folder_name=task_folder_name,
            scene_name=scene_name,
            case_name=str(persisted_row["case"]),
            instruction_name=str(persisted_row["instruction"]),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(persisted_row, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )


def load_oracle_reference_row(
    base_dir: Path,
    *,
    task_folder_name: str,
    scene_name: str,
    case_name: str,
    instruction_name: str,
) -> dict[str, Any] | None:
    """Load one oracle reference row from disk when it exists."""

    input_path = build_oracle_reference_output_path(
        base_dir,
        task_folder_name=task_folder_name,
        scene_name=scene_name,
        case_name=case_name,
        instruction_name=instruction_name,
    )
    if not input_path.exists():
        return None
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Oracle reference payload must be a mapping: {input_path}")
    return payload


def summarize_oracle_results(
    oracle_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate deterministic oracle results across instructions."""

    schedule_times = [
        float(row["optimal_schedule_time"])
        for row in oracle_results
        if row["optimal_schedule_time"] is not None
    ]
    solve_times = [
        float(row.get("solve_time", row.get("computation_time")))
        for row in oracle_results
        if row.get("solve_time", row.get("computation_time")) is not None
    ]
    total_compute_times = [
        float(
            row.get(
                "total_compute_time",
                row.get("computation_time", row.get("solve_time")),
            )
        )
        for row in oracle_results
        if row.get("total_compute_time", row.get("computation_time", row.get("solve_time")))
        is not None
    ]
    tcsr_values = [
        float(row["schedule_tcsr"])
        for row in oracle_results
        if row.get("schedule_tcsr") is not None
    ]
    exact_count = sum(1 for row in oracle_results if bool(row["exact"]))
    perfect_tcsr_count = sum(
        1 for row in oracle_results if row.get("tcsr_is_one") is True
    )
    return {
        "num_instructions": len(oracle_results),
        "exact_instructions": exact_count,
        "avg_optimal_schedule_time": mean(schedule_times) if schedule_times else None,
        "avg_solve_time": mean(solve_times) if solve_times else None,
        "avg_total_compute_time": (
            mean(total_compute_times) if total_compute_times else None
        ),
        "avg_schedule_tcsr": mean(tcsr_values) if tcsr_values else None,
        "perfect_tcsr_instructions": perfect_tcsr_count,
    }


def summarize_oracle_gaps(
    scheduler_results: Sequence[Mapping[str, Any]],
    oracle_results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate scheduler-oracle gaps by beam setting."""

    oracle_by_task = {
        build_oracle_task_key(str(row["case"]), str(row["instruction"])): row
        for row in oracle_results
    }
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in scheduler_results:
        beam_key = f"w{int(row['beam_width'])}_d{int(row['beam_depth'])}"
        grouped.setdefault(beam_key, []).append(row)

    summary: dict[str, dict[str, Any]] = {}
    for beam_key, rows in grouped.items():
        scheduler_times: list[float] = []
        oracle_times: list[float] = []
        absolute_gaps: list[float] = []
        relative_gaps: list[float] = []
        exact_matches = 0
        for row in rows:
            oracle_row = oracle_by_task.get(
                build_oracle_task_key(str(row["case"]), str(row["instruction"]))
            )
            if oracle_row is None or oracle_row["optimal_schedule_time"] is None:
                continue
            scheduler_time = float(row["final_schedule_time"])
            oracle_time = float(oracle_row["optimal_schedule_time"])
            absolute_gap = scheduler_time - oracle_time
            relative_gap = absolute_gap / oracle_time if oracle_time else 0.0
            scheduler_times.append(scheduler_time)
            oracle_times.append(oracle_time)
            absolute_gaps.append(absolute_gap)
            relative_gaps.append(relative_gap)
            if bool(oracle_row["exact"]):
                exact_matches += 1
        summary[beam_key] = {
            "num_runs": len(rows),
            "matched_oracle_runs": len(absolute_gaps),
            "exact_oracle_runs": exact_matches,
            "avg_scheduler_time": mean(scheduler_times) if scheduler_times else None,
            "avg_oracle_time": mean(oracle_times) if oracle_times else None,
            "avg_absolute_gap": mean(absolute_gaps) if absolute_gaps else None,
            "avg_relative_gap": mean(relative_gaps) if relative_gaps else None,
        }
    return summary


def build_oracle_comparison_section(
    *,
    scheduler_results: Sequence[Mapping[str, Any]],
    selected_instructions: Mapping[str, Sequence[str]],
    oracle_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the embedded oracle comparison section for a baseline report."""

    oracle_by_task = {
        build_oracle_task_key(str(row["case"]), str(row["instruction"])): row
        for row in oracle_results
    }
    missing_task_keys: list[str] = []
    best_by_task: dict[str, dict[str, Any]] = {}
    for case_name, instructions in selected_instructions.items():
        for instruction in instructions:
            task_key = build_oracle_task_key(case_name, instruction)
            oracle_row = oracle_by_task.get(task_key)
            if oracle_row is None:
                missing_task_keys.append(task_key)

    scheduler_best_by_task: dict[str, Mapping[str, Any]] = {}
    for row in scheduler_results:
        task_key = build_oracle_task_key(str(row["case"]), str(row["instruction"]))
        incumbent = scheduler_best_by_task.get(task_key)
        if incumbent is None or float(row["final_schedule_time"]) < float(
            incumbent["final_schedule_time"]
        ):
            scheduler_best_by_task[task_key] = row

    for task_key, row in scheduler_best_by_task.items():
        oracle_row = oracle_by_task.get(task_key)
        if oracle_row is None or oracle_row["optimal_schedule_time"] is None:
            continue
        oracle_time = float(oracle_row["optimal_schedule_time"])
        scheduler_time = float(row["final_schedule_time"])
        absolute_gap = scheduler_time - oracle_time
        relative_gap = absolute_gap / oracle_time if oracle_time else 0.0
        best_by_task[task_key] = {
            "case": row["case"],
            "instruction": row["instruction"],
            "beam_width": row["beam_width"],
            "beam_depth": row["beam_depth"],
            "scheduler_time": scheduler_time,
            "oracle_time": oracle_time,
            "absolute_gap": absolute_gap,
            "relative_gap": relative_gap,
            "oracle_exact": bool(oracle_row["exact"]),
            "oracle_timeout_hit": bool(oracle_row["timeout_hit"]),
        }

    coverage = OracleReferenceCoverage(
        matched_instructions=len(oracle_results),
        total_selected_instructions=sum(len(items) for items in selected_instructions.values()),
        exact_instructions=sum(1 for row in oracle_results if bool(row["exact"])),
        missing_task_keys=missing_task_keys,
    )
    return {
        "schema_version": ORACLE_REFERENCE_SCHEMA_VERSION,
        "coverage": coverage.as_dict(),
        "summary": summarize_oracle_results(oracle_results),
        "gap_by_setting": summarize_oracle_gaps(scheduler_results, oracle_results),
        "best_by_task": best_by_task,
    }


def load_oracle_results_for_selection(
    *,
    base_dir: Path,
    task_folder_name: str,
    scene_name: str,
    selected_instructions: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    """Load all available oracle rows for the selected case/instruction set."""

    loaded_rows: list[dict[str, Any]] = []
    for case_name, instructions in selected_instructions.items():
        for instruction_name in instructions:
            row = load_oracle_reference_row(
                base_dir,
                task_folder_name=task_folder_name,
                scene_name=scene_name,
                case_name=case_name,
                instruction_name=instruction_name,
            )
            if row is not None:
                loaded_rows.append(row)
    return loaded_rows
