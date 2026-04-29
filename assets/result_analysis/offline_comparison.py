"""Compare offline batch results against oracle reference.

Produces two output files:

1. offline_comparison_raw.json
   Per scene/case/instruction: oracle fields + per-approach metrics and oracle gaps.

2. offline_analysis_summary.json
   Per approach/case: aggregated metrics (sr, tsr, makespan, makespan_sr_1,
   makespan_gap, makespan_gap_sr_1, computation_time, computation_time_gap,
   n_instructions).

Optional (--tolerance-sweep):

3. offline_analysis_tol_sweep.json
   Per tolerance/approach/case: TSR re-computed from saved detail_log.
   Does not require re-running the batch scheduler.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.utils.common.logger import create_module_logger
from src.utils.config.constants import (
    ASSETS_PATH,
    CONSECUTIVE_TASK_WAIT_TOLERANCE,
    TIMING_TOLERANCE_RATIO,
)
from src.utils.task.primitive_action_semantics import (
    classify_issue_group,
    find_first_task_sequence_issue,
)

logger = create_module_logger(__name__)

EPSILON = 1e-6


@lru_cache(maxsize=None)
def _load_translation_validation(
    task_folder: str,
    case: str,
    scene: str,
    instruction: str,
) -> dict[str, Any]:
    """Return action-semantic translation validity for one task JSON."""

    task_path = ASSETS_PATH / "tasks" / task_folder / case / scene / f"{instruction}.json"
    if not task_path.exists():
        return {
            "translation_valid": False,
            "translation_issue_group": "Missing translated task file",
        }

    try:
        task_data = json.loads(task_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "translation_valid": False,
            "translation_issue_group": f"Unreadable translated task file: {exc}",
        }

    issue = find_first_task_sequence_issue(task_data)
    if issue is None:
        return {
            "translation_valid": True,
            "translation_issue_group": None,
        }
    return {
        "translation_valid": False,
        "translation_issue_group": classify_issue_group(issue),
    }


def _sanitize_token(value: Any) -> str:
    """Return a filesystem-friendly token used in summary keys."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _build_approach_key_from_stem(stem: str) -> str:
    """Fallback parser for legacy result filenames without rich metadata."""

    parts = stem.split("__")
    approach = parts[0]
    init_prior = parts[2] if len(parts) > 2 else None

    if approach == "cpm":
        return f"{init_prior}__cpm" if init_prior is not None else "cpm"
    if approach == "edf":
        return f"{init_prior}__edf" if init_prior is not None else "edf"
    if approach == "bayesian":
        if len(parts) < 4:
            return stem
        ablation = parts[1]
        beam = parts[3]
        suffixes = parts[4:]
        baseline_name = "particle_filter" if "pf" in suffixes else "bayesian"
        key = f"{baseline_name}__{ablation}__{beam}"
        for suffix in suffixes:
            if (
                suffix.startswith("eta")
                or suffix.startswith("gt")
                or suffix.startswith("pdist")
                or suffix.startswith("plik")
                or suffix.startswith("mb")
            ):
                key += f"__{suffix}"
        return f"{init_prior}__{key}" if init_prior is not None else key

    return stem


def build_approach_key(
    stem: str,
    meta_data: dict[str, Any] | None = None,
) -> str:
    """Convert result filename stem to a short approach key.

    Examples:
        cpm__DEFAULT__CORRECT_ESTIMATE
            -> CORRECT_ESTIMATE__cpm
        edf__DEFAULT__CORRECT_ESTIMATE
            -> CORRECT_ESTIMATE__edf
        DEFAULT__w10_d10__eta0.1__gtlognormal + metadata(prior=UNDER_ESTIMATE)
            -> UNDER_ESTIMATE__bayesian__DEFAULT__w10_d10__eta0.1__gtlognormal
    """
    if not isinstance(meta_data, dict):
        return _build_approach_key_from_stem(stem)

    init_prior = meta_data.get("init_prior_config")
    prior_prefix = (
        f"{_sanitize_token(init_prior)}__" if init_prior is not None else ""
    )

    baseline_name = str(
        meta_data.get("baseline_name")
        or meta_data.get("approach")
        or stem.split("__")[0]
    )
    if baseline_name == "cpm":
        return f"{prior_prefix}cpm"
    if baseline_name == "edf":
        return f"{prior_prefix}edf"
    if baseline_name not in {"bayesian", "particle_filter"}:
        return stem

    ablation = meta_data.get("ablation_config")
    beam_width = meta_data.get("beam_width")
    beam_depth = meta_data.get("beam_depth")
    if ablation is None or beam_width is None or beam_depth is None:
        return _build_approach_key_from_stem(stem)

    key = f"{baseline_name}__{ablation}__w{beam_width}_d{beam_depth}"
    eta = meta_data.get("eta")
    if eta is not None:
        key += f"__eta{eta}"

    gt_distribution = str(meta_data.get("gt_distribution", "constant"))
    particle_distribution = meta_data.get("particle_distribution")
    particle_likelihood_family = meta_data.get("particle_likelihood_family")

    if baseline_name == "particle_filter":
        key += f"__gt{_sanitize_token(gt_distribution)}"
        if particle_distribution not in {None, gt_distribution}:
            key += f"__pdist{_sanitize_token(particle_distribution)}"
        if particle_likelihood_family not in {None, "gaussian"}:
            key += f"__plik{_sanitize_token(particle_likelihood_family)}"
        monitoring_budget = meta_data.get("monitoring_budget_per_critical")
        if monitoring_budget is not None:
            key += f"__mb{_sanitize_token(monitoring_budget)}"
        return f"{prior_prefix}{key}"

    if gt_distribution != "constant":
        key += f"__gt{_sanitize_token(gt_distribution)}"
    monitoring_budget = meta_data.get("monitoring_budget_per_critical")
    if monitoring_budget is not None:
        key += f"__mb{_sanitize_token(monitoring_budget)}"
    return f"{prior_prefix}{key}"


def _oracle_has_constraint_violation(data: dict[str, Any]) -> bool:
    """Return True when any constraint in the oracle's own detail_log is marked [False]."""
    for step in data.get("steps", []):
        for constraint_result in step.get("detail_log", {}).values():
            if isinstance(constraint_result, str) and constraint_result.startswith(
                "[False]"
            ):
                return True
    return False


def load_oracle(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    violated = _oracle_has_constraint_violation(data)
    optimal_schedule_time = data.get("optimal_schedule_time")
    exact = bool(data.get("exact", False))
    return {
        "optimal_schedule_time": optimal_schedule_time,
        "computation_time": data["computation_time"],
        "exact": exact,
        "constraint_violated": violated,
        # Oracle references without an exact optimal schedule cannot support
        # makespan-gap comparison, even if the file itself was written cleanly.
        "reference_available": exact and optimal_schedule_time is not None,
    }


def load_baseline(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    tsr_raw = data.get("timing_success_rate_sched")
    meta_data = data.get("meta_data", {})
    if not isinstance(meta_data, dict):
        meta_data = {}
    return {
        "completed": bool(data.get("completed", False)),
        "tsr": float(tsr_raw) if tsr_raw is not None else 0.0,
        "makespan": data.get("scheduler_makespan"),
        "computation_time": data.get("computation_time"),
        "scene_name": data.get("scene_name"),
        "case": data.get("case", data.get("case_name")),
        "instruction": data.get("instruction", data.get("instruction_name")),
        "detail_log": data.get("detail_log", {}),
        "meta_data": meta_data,
    }


def _round(value: float | None, ndigits: int) -> float | None:
    return round(value, ndigits) if value is not None else None


def _extract_batch_identity(
    batch_file: Path,
    *,
    batch_root: Path,
    baseline: dict[str, Any],
) -> tuple[str, str, str] | None:
    """Return ``(scene, case, instruction_stem)`` for one saved batch file."""

    scene = baseline.get("scene_name")
    case = baseline.get("case")
    instruction = baseline.get("instruction")
    if scene is not None and case is not None and instruction is not None:
        return str(scene), str(case), Path(str(instruction)).stem

    relative_parts = batch_file.relative_to(batch_root).parts
    if len(relative_parts) >= 6:
        return relative_parts[-5], relative_parts[-4], relative_parts[-3]
    if len(relative_parts) >= 4:
        return relative_parts[-4], relative_parts[-3], relative_parts[-2]
    return None


def _iter_batch_result_files(batch_root: Path) -> list[Path]:
    """Return all saved batch result JSON files under one task folder."""

    return sorted(path for path in batch_root.rglob("*.json") if path.is_file())


def build_raw(
    base_dir: Path,
    task_folder: str,
    *,
    batch_dirname: str = "offline_batch",
    oracle_dirname: str = "offline_oracle_reference",
) -> dict[str, Any]:
    """Walk oracle reference tree and match each instruction to batch results.

    Returns nested dict: raw[scene][case][instruction] = {oracle: ..., approach: ...}
    """
    oracle_root = base_dir / oracle_dirname / task_folder
    batch_root = base_dir / batch_dirname / task_folder

    if not oracle_root.exists():
        raise FileNotFoundError(f"Oracle reference directory not found: {oracle_root}")
    if not batch_root.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_root}")

    raw: dict[str, Any] = {}
    batch_index: dict[tuple[str, str, str], list[tuple[Path, dict[str, Any]]]] = (
        defaultdict(list)
    )

    for batch_file in _iter_batch_result_files(batch_root):
        try:
            baseline = load_baseline(batch_file)
        except Exception as e:
            logger.warning("Failed to load %s: %s", batch_file, e)
            continue
        identity = _extract_batch_identity(
            batch_file,
            batch_root=batch_root,
            baseline=baseline,
        )
        if identity is None:
            logger.warning("Failed to infer batch identity for %s", batch_file)
            continue
        batch_index[identity].append((batch_file, baseline))

    for scene_dir in sorted(oracle_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene = scene_dir.name

        for case_dir in sorted(scene_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            case = case_dir.name

            for oracle_file in sorted(case_dir.glob("*.json")):
                instruction = oracle_file.stem

                try:
                    oracle_data = load_oracle(oracle_file)
                except Exception as e:
                    logger.warning("Failed to load oracle %s: %s", oracle_file, e)
                    continue

                matched_batch_files = batch_index.get((scene, case, instruction), [])
                if not matched_batch_files:
                    logger.warning(
                        "No batch results: %s / %s / %s", scene, case, instruction
                    )
                    continue

                oracle_valid = (
                    not oracle_data.get("constraint_violated", False)
                    and oracle_data.get("reference_available", False)
                )
                translation_check = _load_translation_validation(
                    task_folder=task_folder,
                    case=case,
                    scene=scene,
                    instruction=instruction,
                )
                entry: dict[str, Any] = {
                    "oracle": oracle_data,
                    "oracle_valid": oracle_valid,
                    "translation_valid": translation_check["translation_valid"],
                    "translation_issue_group": translation_check["translation_issue_group"],
                }

                for batch_file, baseline in sorted(
                    matched_batch_files,
                    key=lambda item: item[0].as_posix(),
                ):
                    meta_data = baseline.get("meta_data", {})
                    approach_key = build_approach_key(batch_file.stem, meta_data)

                    makespan = baseline["makespan"]
                    comp_time = baseline["computation_time"]
                    oracle_makespan = oracle_data["optimal_schedule_time"]
                    oracle_comp_time = oracle_data["computation_time"]
                    makespan_gap = None
                    if makespan is not None and oracle_makespan is not None:
                        makespan_gap = _round(float(makespan) - float(oracle_makespan), 4)

                    computation_time_gap = None
                    if comp_time is not None and oracle_comp_time is not None:
                        computation_time_gap = _round(
                            float(comp_time) - float(oracle_comp_time), 6
                        )

                    entry[approach_key] = {
                        "completed": baseline["completed"],
                        "tsr": baseline["tsr"],
                        "makespan": _round(makespan, 4),
                        "makespan_gap": makespan_gap,
                        "computation_time": _round(comp_time, 6),
                        "computation_time_gap": computation_time_gap,
                        "belief_update_method": meta_data.get("belief_update_method"),
                        "gt_distribution": meta_data.get("gt_distribution"),
                        "particle_distribution": meta_data.get("particle_distribution"),
                        "particle_likelihood_family": meta_data.get(
                            "particle_likelihood_family"
                        ),
                        "eta": meta_data.get("eta"),
                    }

                raw.setdefault(scene, {}).setdefault(case, {})[instruction] = entry

    return raw


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(
    raw: dict[str, Any], *, skip_oracle_violated: bool = False, skip_translation_invalid: bool = True
) -> dict[str, Any]:
    """Aggregate raw per-instruction data into per-approach/case summary.

    Args:
        raw: Output of :func:`build_raw`.
        skip_oracle_violated: When True, exclude instructions where the oracle's
            own schedule violates a constraint (``oracle_valid: False``).  These
            entries have inflated ``optimal_schedule_time`` values that make
            approach makespan gaps artificially negative.
    """

    # Collect records: records[approach][case] = list of per-instruction dicts
    records: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    case_filter_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "n_instructions_total": 0,
            "n_translation_valid": 0,
            "n_oracle_valid": 0,
            "n_after_filters": 0,
        }
    )

    for _scene, cases in raw.items():
        for case, instructions in cases.items():
            for _instruction, entry in instructions.items():
                case_filter_counts[case]["n_instructions_total"] += 1
                translation_valid = bool(entry.get("translation_valid", True))
                oracle_valid = bool(entry.get("oracle_valid", True))
                if translation_valid:
                    case_filter_counts[case]["n_translation_valid"] += 1
                if oracle_valid:
                    case_filter_counts[case]["n_oracle_valid"] += 1
                if skip_translation_invalid and not translation_valid:
                    continue
                if skip_oracle_violated and not oracle_valid:
                    continue
                case_filter_counts[case]["n_after_filters"] += 1
                for approach_key, metrics in entry.items():
                    if approach_key in (
                        "oracle",
                        "oracle_valid",
                        "translation_valid",
                        "translation_issue_group",
                    ):
                        continue
                    records[approach_key][case].append(metrics)

    summary: dict[str, Any] = {}

    for approach_key, cases in sorted(records.items()):
        summary[approach_key] = {}
        for case, entries in sorted(cases.items()):
            n = len(entries)
            if n == 0:
                continue

            completed = [e for e in entries if e["completed"]]

            sr = len(completed) / n * 100
            tsr = _mean([e["tsr"] for e in entries]) or 0.0
            tsr_pct = tsr * 100

            makespan = _mean(
                [e["makespan"] for e in entries if e["makespan"] is not None]
            )
            makespan_sr_1 = _mean(
                [e["makespan"] for e in completed if e["makespan"] is not None]
            )
            makespan_gap = _mean(
                [e["makespan_gap"] for e in entries if e["makespan_gap"] is not None]
            )
            makespan_gap_sr_1 = _mean(
                [e["makespan_gap"] for e in completed if e["makespan_gap"] is not None]
            )
            computation_time = _mean(
                [
                    e["computation_time"]
                    for e in entries
                    if e["computation_time"] is not None
                ]
            )
            computation_time_gap = _mean(
                [
                    e["computation_time_gap"]
                    for e in entries
                    if e["computation_time_gap"] is not None
                ]
            )

            summary[approach_key][case] = {
                "sr": round(sr, 4),
                "tsr": round(tsr_pct, 4),
                "makespan": _round(makespan, 4),
                "makespan_sr_1": _round(makespan_sr_1, 4),
                "makespan_gap": _round(makespan_gap, 4),
                "makespan_gap_sr_1": _round(makespan_gap_sr_1, 4),
                "computation_time": _round(computation_time, 6),
                "computation_time_gap": _round(computation_time_gap, 6),
                "n_instructions": n,
                "n_instructions_total": case_filter_counts[case]["n_instructions_total"],
                "n_translation_valid": case_filter_counts[case]["n_translation_valid"],
                "n_oracle_valid": case_filter_counts[case]["n_oracle_valid"],
                "n_after_filters": case_filter_counts[case]["n_after_filters"],
            }

    return summary


def _recompute_tsr_from_detail_log(
    detail_log: dict[str, Any],
    tolerance_abs: float,
) -> float | None:
    """Re-compute schedule TSR from a saved detail_log with a different tolerance_abs.

    Only non-zero interval constraints are affected by tolerance_abs.
    (0, True) consecutive constraints always use CONSECUTIVE_TASK_WAIT_TOLERANCE=2.0s
    and their result is read directly from the [True/False] flag.

    Args:
        detail_log: The ``detail_log`` dict from a batch result JSON.
        tolerance_abs: Absolute tolerance cap for non-zero interval constraints.

    Returns:
        TSR as a float in [0, 1], or None when no constraints exist.
    """
    total = 0
    succeeded = 0

    for _edge_key, edge_val in detail_log.items():
        constraint_str = edge_val.get("Original Timing Constraint", "")
        schedule_str = edge_val.get("Schedule Result", "")

        # Parse constraint: "(interval, is_critical)"
        c_match = re.match(r"\(([^,]+),\s*(True|False)\)", constraint_str)
        if not c_match:
            continue
        interval = float(c_match.group(1))
        is_critical = c_match.group(2) == "True"

        # Parse schedule: "[True/False] : (pred_end) -> (succ_interaction_start)"
        t_match = re.match(
            r"\[(True|False)\] : \(([\d.]+)\) -> \(([\d.]+)s?\)", schedule_str
        )
        if not t_match:
            continue

        total += 1

        if is_critical and interval < EPSILON:
            # (0, True) result is independent of tolerance_abs — use saved flag.
            if t_match.group(1) == "True":
                succeeded += 1
            continue

        pred_end = float(t_match.group(2))
        succ_interaction_start = float(t_match.group(3))
        actual_diff = succ_interaction_start - pred_end

        ratio_allowance = interval * TIMING_TOLERANCE_RATIO
        tolerance = max(5.0, min(tolerance_abs, ratio_allowance))

        if is_critical:
            if abs(interval - actual_diff) <= tolerance:
                succeeded += 1
        else:
            if (interval - actual_diff) <= tolerance:
                succeeded += 1

    return succeeded / total if total > 0 else None


def build_tol_sweep(
    base_dir: Path,
    task_folder: str,
    tolerances: list[float],
    *,
    batch_dirname: str = "offline_batch",
) -> dict[str, Any]:
    """Re-compute TSR for each tolerance from existing batch detail_logs.

    No scheduler re-run is needed — only the evaluation threshold changes.

    Returns:
        Nested dict:
        result[tolerance_str][approach_key][case] = list of {tsr, makespan, ...}
    """
    batch_root = base_dir / batch_dirname / task_folder

    # records[tol][approach][case] = list of per-instruction dicts
    records: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
        str(tol): defaultdict(lambda: defaultdict(list)) for tol in tolerances
    }

    for batch_file in _iter_batch_result_files(batch_root):
        try:
            baseline = load_baseline(batch_file)
        except Exception as e:
            logger.warning("Failed to load %s: %s", batch_file, e)
            continue
        identity = _extract_batch_identity(
            batch_file,
            batch_root=batch_root,
            baseline=baseline,
        )
        if identity is None:
            logger.warning("Failed to infer batch identity for %s", batch_file)
            continue
        _scene, case, _instruction = identity
        approach_key = build_approach_key(
            batch_file.stem,
            baseline.get("meta_data"),
        )
        detail_log = baseline.get("detail_log", {})
        makespan = baseline.get("makespan")
        completed = bool(baseline.get("completed", False))

        for tol in tolerances:
            tsr = _recompute_tsr_from_detail_log(detail_log, tol)
            records[str(tol)][approach_key][case].append(
                {
                    "tsr": tsr if tsr is not None else 0.0,
                    "makespan": makespan,
                    "completed": completed,
                }
            )

    # Aggregate per approach/case
    result: dict[str, Any] = {}
    for tol_str, approaches in records.items():
        result[tol_str] = {}
        for approach_key, cases in sorted(approaches.items()):
            result[tol_str][approach_key] = {}
            for case, entries in sorted(cases.items()):
                n = len(entries)
                if n == 0:
                    continue
                tsr_mean = _mean([e["tsr"] for e in entries]) or 0.0
                makespan_mean = _mean(
                    [e["makespan"] for e in entries if e["makespan"] is not None]
                )
                sr = len([e for e in entries if e["completed"]]) / n * 100
                result[tol_str][approach_key][case] = {
                    "sr": round(sr, 4),
                    "tsr": round(tsr_mean * 100, 4),
                    "makespan": _round(makespan_mean, 4),
                }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare offline batch results against oracle reference."
    )
    parser.add_argument(
        "--task_folder",
        type=str,
        default="sampled_10_instruction_set_for_final_experiment_set_b",
        help="Task folder name inside oracle_reference and offline_batch directories.",
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path("assets/results/offline_exp_result"),
        help="Root directory containing offline_oracle_reference/ and offline_batch/.",
    )
    parser.add_argument(
        "--batch_dirname",
        type=str,
        default="offline_batch",
        help="Batch result subdirectory under --base_dir.",
    )
    parser.add_argument(
        "--oracle_dirname",
        type=str,
        default="offline_oracle_reference",
        help="Oracle reference subdirectory under --base_dir.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to --base_dir.",
    )
    parser.add_argument(
        "--skip_oracle_violated",
        action="store_true",
        default=False,
        help=(
            "Exclude instructions where the oracle reference is unusable "
            "(constraint-violated, timed out, or missing optimal_schedule_time)."
        ),
    )
    parser.add_argument(
        "--include_translation_invalid",
        action="store_true",
        default=False,
        help=(
            "Include translation-invalid instructions in aggregate denominators. "
            "By default, action-semantically invalid translated tasks are excluded."
        ),
    )
    parser.add_argument(
        "--tolerance-sweep",
        nargs="+",
        type=float,
        default=[5.0, 8.0, 12.5],
        metavar="TOL",
        help=(
            "Re-compute TSR for each given tolerance_abs value from saved detail_logs. "
            "No scheduler re-run needed. Example: --tolerance-sweep 5.0 8.0 12.5"
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building raw comparison...")
    raw = build_raw(
        args.base_dir,
        args.task_folder,
        batch_dirname=args.batch_dirname,
        oracle_dirname=args.oracle_dirname,
    )

    raw_path = output_dir / "offline_comparison_raw.json"
    with raw_path.open("w") as f:
        json.dump(raw, f, indent=2)
    print(f"Saved: {raw_path}")

    # Report unusable oracle reference stats
    total = invalid = 0
    for cases in raw.values():
        for instructions in cases.values():
            for entry in instructions.values():
                total += 1
                if not entry.get("oracle_valid", True):
                    invalid += 1
    print(f"Invalid oracle references: {invalid}/{total} instructions")

    print("Aggregating summary...")
    summary = aggregate(
        raw,
        skip_oracle_violated=args.skip_oracle_violated,
        skip_translation_invalid=not args.include_translation_invalid,
    )

    summary_path = output_dir / "offline_analysis_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"Saved: {summary_path}")

    if args.tolerance_sweep:
        print(f"Running tolerance sweep: {args.tolerance_sweep} ...")
        tol_sweep = build_tol_sweep(
            args.base_dir,
            args.task_folder,
            args.tolerance_sweep,
            batch_dirname=args.batch_dirname,
        )
        tol_path = output_dir / "offline_analysis_tol_sweep.json"
        with tol_path.open("w") as f:
            json.dump(tol_sweep, f, indent=2, sort_keys=True)
        print(f"Saved: {tol_path}")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
