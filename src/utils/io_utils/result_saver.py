### utils/io_utils/result_saver.py
from __future__ import annotations

import ast
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

from networkx import DiGraph

if TYPE_CHECKING:
    from src.models.dataclass import CompletedEntry

from src.utils.common.logger import create_module_logger
from src.utils.config import constants
from src.utils.config.constants import (
    CONSECUTIVE_TASK_WAIT_TOLERANCE,
    EPSILON,
    RESULT_PATH,
    TIMING_TOLERANCE_ABS,
    TIMING_TOLERANCE_RATIO,
)
from src.utils.visualizers.visualizer import visualize

log = create_module_logger(__name__, module_log=True, level=logging.INFO)


def get_now_str(fmt: str = "%Y-%m-%d %H:%M") -> str:
    return datetime.now().strftime(fmt)


def compose_subtasks(
    result_schedule: List[CompletedEntry],
) -> Tuple[List[dict], int, int]:
    success_count = 0
    total_count = 0

    for ce in result_schedule:
        execution_status = getattr(ce, "execution_status", None)
        if execution_status is not None:
            total_count += 1
            if execution_status:
                success_count += 1

    return success_count, total_count


def compose_plans(
    result_schedule: List[CompletedEntry], task_name: str
) -> Tuple[float, float, float]:
    success_count, total_count = compose_subtasks(result_schedule)

    simulation_makespan = result_schedule[-1].sim_end_time if result_schedule else None
    scheduler_makespan = (
        result_schedule[-1].schedule_end_time if result_schedule else None
    )
    success_rate = round(success_count / total_count, 3) if total_count > 0 else 0.0

    return success_rate, simulation_makespan, scheduler_makespan


def calculate_timing_success_rate(
    constraints: DiGraph,
    result_schedule: List[CompletedEntry],
    ground_truth_overrides: Optional[Mapping[str, float]] = None,
    tolerance_abs: float = TIMING_TOLERANCE_ABS,
) -> Tuple[float | None, float | None, Dict[str, Any]]:
    """
    Calculates the success rate of timing constraints based on simulation and schedule results.

    Args:
        constraints: A DiGraph representing the constraints between subtasks.
        result_schedule: A list of CompletedEntry objects containing execution results.
        ground_truth_overrides: Optional mapping from object type to sampled
            runtime interval used to evaluate critical timing constraints.

    Returns:
        A tuple containing:
        - The timing success rate for the simulation (float | None).
        - The timing success rate for the schedule (float | None).
        - A dictionary with detailed logging information.
    """

    total_timing_constraints = 0
    succeeded_timing_constraints_sim_cnt = 0
    succeeded_timing_constraints_sched_cnt = 0
    detail_log = {}

    # Create a mapping from task names to completed entries for efficient lookup
    entry_map = {ce.subtask.name: ce for ce in result_schedule}

    for u, v, data in constraints.edges(data=True):
        timing_success_flag_sched = False

        if u.lower().startswith("monitoring") or v.lower().startswith("monitoring"):
            log.debug(f"Skipping monitoring task edge: {u} -> {v}")
            continue

        pred_entry = entry_map.get(u)
        succ_entry = entry_map.get(v)

        if not pred_entry or not succ_entry:

            continue

        total_timing_constraints += 1
        # log.info(f"total_timing_constraints increased: {total_timing_constraints}")
        edge_info = data.get("info", {})
        interval = edge_info.get("Interval", 0)
        is_critical = edge_info.get("IsCritical", False)

        if interval != 0 and is_critical:
            interval = _resolve_ground_truth_interval(
                pred_entry=pred_entry,
                succ_entry=succ_entry,
                default_interval=constants.GT_INTERVAL,
                ground_truth_overrides=ground_truth_overrides,
            )
        elif interval != 0 and not is_critical:
            interval = constants.INIT_PRIOR_MEAN

        # --- Check timing constraints for simulation results ---
        pred_end_time_sim = pred_entry.sim_end_time
        succ_start_time_sim = succ_entry.sim_start_time
        sim_nav_time = (
            succ_entry.sim_nav_time if succ_entry.sim_nav_time is not None else 0.0
        )
        ratio_allowance = interval * TIMING_TOLERANCE_RATIO
        tolerance = max(5, min(tolerance_abs, ratio_allowance))

        actual_diff_sim = (succ_start_time_sim + sim_nav_time) - pred_end_time_sim

        if is_critical:
            if interval < EPSILON:
                # (0, True): check that successor nav started immediately after pred.
                # Nav time itself is physical; only wait_time (before nav starts) matters.
                wait_time_sim = succ_start_time_sim - pred_end_time_sim
                if abs(wait_time_sim) <= CONSECUTIVE_TASK_WAIT_TOLERANCE:
                    succeeded_timing_constraints_sim_cnt += 1
            else:
                if abs(interval - actual_diff_sim) <= tolerance:
                    succeeded_timing_constraints_sim_cnt += 1
        else:  # Non-critical
            if (interval - actual_diff_sim) <= tolerance:
                succeeded_timing_constraints_sim_cnt += 1

        # --- Check timing constraints for schedule results ---
        pred_end_time_sched = pred_entry.schedule_end_time
        succ_start_time_sched = succ_entry.schedule_start_time
        schedule_nav_time = (
            succ_entry.schedule_nav_time
            if succ_entry.schedule_nav_time is not None
            else 0.0
        )

        # The actual interval is between the end of the predecessor and the start of the successor's INTERACTION.
        succ_interaction_start_time_sched = succ_start_time_sched + schedule_nav_time
        actual_diff_sched = succ_interaction_start_time_sched - pred_end_time_sched

        sched_constraint_met = False
        if is_critical:
            # For (0, True) constraint, tasks must be consecutive, meaning the successor's
            # navigation starts immediately after the predecessor finishes. The "wait time" should be near zero.
            # Nav time is physically inevitable; only the scheduler wait before nav is evaluated.
            if interval < EPSILON:
                wait_time = succ_start_time_sched - pred_end_time_sched
                if abs(wait_time) <= CONSECUTIVE_TASK_WAIT_TOLERANCE:
                    succeeded_timing_constraints_sched_cnt += 1
                    sched_constraint_met = True
            # For other critical constraints, the total gap must match the interval.
            else:
                if abs(interval - actual_diff_sched) <= tolerance:
                    succeeded_timing_constraints_sched_cnt += 1
                    sched_constraint_met = True
        else:  # Non-critical
            if (interval - actual_diff_sched) <= tolerance:
                succeeded_timing_constraints_sched_cnt += 1
                sched_constraint_met = True

        timing_success_flag_sched = sched_constraint_met

        # --- Logging ---
        log.info(f"Original Timing Constraint : {u} -> {v} ({interval}, {is_critical})")
        log.info(
            f"Schedule Result [{timing_success_flag_sched}] - {pred_entry.subtask.name} ({pred_end_time_sched:.2f}) -> {succ_entry.subtask.name} (interacts at {succ_interaction_start_time_sched:.2f})s\n\n"
        )
        detail_log[f"{u} -> {v}"] = {
            "Original Timing Constraint": f"({interval}, {is_critical})",
            "Schedule Result": f"[{timing_success_flag_sched}] : ({pred_end_time_sched:.2f}) -> ({succ_interaction_start_time_sched:.2f}s)",
        }

    timing_success_rate_sim = (
        succeeded_timing_constraints_sim_cnt / total_timing_constraints
        if total_timing_constraints != 0
        else None
    )
    timing_success_rate_sched = (
        succeeded_timing_constraints_sched_cnt / total_timing_constraints
        if total_timing_constraints != 0
        else None
    )
    return timing_success_rate_sim, timing_success_rate_sched, detail_log


def serialize_completed_entries(result_schedule: List[CompletedEntry]) -> List[dict]:
    """
    Convert CompletedEntry objects to JSON-serializable format
    """
    serialized_entries = []
    for entry in result_schedule:
        serialized_entry = {
            "subtask_name": entry.subtask.name,
            "start_time_simulation": round(entry.sim_start_time, 2),
            "end_time_simulation": round(entry.sim_end_time, 2),
            "start_time_scheduled": round(entry.schedule_start_time, 2),
            "end_time_scheduled": round(entry.schedule_end_time, 2),
            "execution_status": (
                "SUCCESS"
                if entry.execution_status is True
                else (
                    "FAILURE"
                    if entry.execution_status is False
                    else str(entry.execution_status.name)
                )
            ),
        }
        if hasattr(entry, "primitive_action_log"):
            serialized_entry["primitive_action_log"] = [
                {"action": log["action"], "duration": round(log["duration"], 2)}
                for log in entry.primitive_action_log
            ]
        monitored_subtask = getattr(entry, "monitored_subtask", None)
        if monitored_subtask is not None:
            serialized_entry["monitored_subtask"] = monitored_subtask
        serialized_entries.append(serialized_entry)
    return serialized_entries


def build_hybrid_single_run_payload(
    *,
    meta_data: Mapping[str, Any],
    saved_time: str,
    approach: str,
    scene_name: str,
    plans: List[dict[str, Any]],
    computation_time: float | None,
    scheduler_makespan: float | None,
    timing_success_rate_sched: float | None,
    detail_log: Mapping[str, Any],
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a legacy-style single-run payload with offline extensions.

    Args:
        meta_data: Execution metadata for the run.
        saved_time: Human-readable save timestamp.
        approach: Planner label written to the legacy top-level field.
        scene_name: Scene identifier.
        plans: Serialized plan entries.
        computation_time: Total wall-clock planning time.
        scheduler_makespan: Final scheduled makespan.
        timing_success_rate_sched: Scheduled timing-success ratio.
        detail_log: Timing constraint detail log.
        extra_fields: Additional offline-specific fields to merge into the payload.

    Returns:
        Hybrid single-run payload.
    """

    payload: dict[str, Any] = {
        "meta_data": dict(meta_data),
        "saved_time": saved_time,
        "approach": approach,
        "scene_name": scene_name,
        "plans": list(plans),
        "computation_time": computation_time,
        "scheduler_makespan": scheduler_makespan,
        "timing_success_rate_sched": timing_success_rate_sched,
        "detail_log": dict(detail_log),
    }
    if extra_fields:
        payload.update(dict(extra_fields))
    return payload


def _extract_object_types(objects: Any) -> set[str]:
    """Extract object types from an execution objects payload.

    Args:
        objects: Execution objects payload from a subtask.

    Returns:
        Set of object type names.
    """

    if isinstance(objects, dict):
        raw_objects = list(objects.keys())
    elif isinstance(objects, list):
        raw_objects = objects
    else:
        raw_objects = []

    return {
        str(raw_object).split("|")[0]
        for raw_object in raw_objects
        if raw_object is not None
    }


def _resolve_ground_truth_interval(
    *,
    pred_entry: CompletedEntry,
    succ_entry: CompletedEntry,
    default_interval: float,
    ground_truth_overrides: Optional[Mapping[str, float]],
) -> float:
    """Resolve the runtime interval used for critical-edge evaluation.

    Args:
        pred_entry: Completed predecessor entry.
        succ_entry: Completed successor entry.
        default_interval: Fallback interval when no sampled GT is available.
        ground_truth_overrides: Optional sampled GT mapping keyed by object type.

    Returns:
        Interval used for evaluation.
    """

    if not ground_truth_overrides:
        return float(default_interval)

    pred_objects = _extract_object_types(pred_entry.subtask.execution.objects)
    succ_objects = _extract_object_types(succ_entry.subtask.execution.objects)
    candidate_object_types = list(pred_objects.intersection(succ_objects))
    if not candidate_object_types:
        candidate_object_types = list(succ_objects or pred_objects)

    for object_type in candidate_object_types:
        sampled_interval = ground_truth_overrides.get(object_type)
        if sampled_interval is not None:
            return float(sampled_interval)

    return float(default_interval)


def result_save(
    task_name: str,
    approach_name: str,
    result_schedule: List[Any],
    computation_time: float,
    scene_name: str,
    constraints: DiGraph,
    initial_plan_data: List[Dict],
    log_level: str = "INFO",
    base_result_path: Path | None = None,
    init_prior_mean: float | None = None,
    case_name: Optional[str] = None,
    dag_bayesian_meta_data: Optional[Dict] = None,
    ground_truth_overrides: Optional[Mapping[str, float]] = None,
):
    global log
    if init_prior_mean is None:
        init_prior_mean = constants.INIT_PRIOR_MEAN

    success_rate, simulation_makespan, scheduler_makespan = compose_plans(
        result_schedule, task_name
    )

    timing_success_rate_sim, timing_success_rate_sched, detail_log = (
        calculate_timing_success_rate(
            constraints,
            result_schedule,
            ground_truth_overrides=ground_truth_overrides,
        )
    )

    # Serialize the result schedule
    serialized_plans = serialize_completed_entries(result_schedule)

    total_primitive_actions = sum(
        len(entry.subtask.execution.primitive_actions)
        for entry in result_schedule
        if (
            hasattr(entry, "subtask")
            and hasattr(entry.subtask, "execution")
            and entry.subtask.execution is not None
            and hasattr(entry.subtask.execution, "primitive_actions")
            and entry.subtask.execution.primitive_actions is not None
        )
    )
    print(f"simulation_makespan: {simulation_makespan}")
    print(f"computation_time: {computation_time}")
    print(f"timing_success_rate_sim: {timing_success_rate_sim}")
    print(f"approach_name: {approach_name}")
    realworld_makespan = None
    if "ros" in approach_name:
        simulation_makespan = 0
        realworld_makespan = round(simulation_makespan, 2)

    result_data = {
        "saved_time": get_now_str(),
        "approach": approach_name,
        "scene_name": scene_name,
        "plans": serialized_plans,
        "computation_time": round(computation_time, 5),
        "simulation_makespan": round(simulation_makespan, 2),
        "scheduler_makespan": round(scheduler_makespan, 2),
        "total_primitive_actions": total_primitive_actions,
        "realworld_makespan": realworld_makespan,
        "success_rate": round(success_rate, 2),
        "timing_success_rate_sim": (
            None
            if timing_success_rate_sim is None
            else round(timing_success_rate_sim, 2)
        ),
        "timing_success_rate_sched": (
            None
            if timing_success_rate_sched is None
            else round(timing_success_rate_sched, 2)
        ),
        "detail_log": detail_log,
    }

    # Determine the base path for results
    result_path_base = base_result_path if base_result_path is not None else RESULT_PATH
    result_path_with_prior = result_path_base / f"init_{int(init_prior_mean)}"

    # Find the next available number for the task name
    num = 1
    while True:
        if case_name:
            output_path = (
                result_path_with_prior
                / f"{case_name}"
                / f"{task_name}_{num}"
                / scene_name
            )
        else:
            output_path = result_path_with_prior / f"{task_name}_{num}" / scene_name
        approach_path = output_path / "approach"
        file_path = approach_path / f"{approach_name}.json"
        if not file_path.exists():
            break
        num += 1
    # Create directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    approach_path.mkdir(parents=True, exist_ok=True)

    # visualize(
    #     approach_name,
    #     output_path,
    #     constraints,
    #     result_schedule,
    #     initial_plan_data,
    #     scene_name,
    # )
    if dag_bayesian_meta_data:
        result_data = {"meta_data": dag_bayesian_meta_data, **result_data}

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4)

    print(f"JSON file saved at {file_path}")


def parse_prog_log(lines: List[str]) -> Tuple[List[dict], float, int, int]:
    actions = []
    current_action = None
    start_time, end_time = None, None
    execution_status = None
    last_end_time = 0
    total_count = 0
    success_count = 0

    for line in lines:
        line = line.strip()

        if line.startswith("Executing action:"):
            if current_action:
                current_action.update(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "execution_status": execution_status,
                    }
                )
                actions.append(current_action)

            action = re.findall(r"\['(.*?)'\]", line)
            if action:
                action = action[0].split("', '")
                current_action = {"Executing_action": action}
                start_time, end_time, execution_status = None, None, None
            else:
                current_action = None

        elif line.startswith("start_time:"):
            start_time = float(line.split(":")[1].strip())

        elif line.startswith("end_time:"):
            end_time = float(line.split(":")[1].strip())
            last_end_time = max(last_end_time, end_time)

        elif line.startswith("execution_status:"):
            status = line.split(":")[1].strip()
            if status == "True":
                execution_status = True
                success_count += 1
            elif status == "False":
                execution_status = False
            total_count += 1

    if current_action:
        current_action.update(
            {
                "start_time": start_time,
                "end_time": end_time,
                "execution_status": execution_status,
            }
        )
        actions.append(current_action)

    return actions, last_end_time, success_count, total_count


def parse_cap_log(lines: List[str]) -> Tuple[List[dict], float, int, int]:
    """Parse log from CAP approach."""
    actions = []
    current_action = None
    start_time, end_time = None, None
    execution_status = None
    last_end_time = 0.0
    total_count = 0
    success_count = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("Executing action:"):
            if current_action is not None and start_time is not None:
                current_action.update(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "execution_status": execution_status,
                    }
                )
                actions.append(current_action)

            # Support both legacy list format and new quoted string format.
            # 1) Legacy: Executing action: ['pickup', 'Apple']
            # 2) New   : Executing action: "pickup Apple"
            tokens: List[str] = []

            # Try to parse list-like content first using ast.literal_eval for robustness
            bracket_match = re.search(r"\[.*\]", line)
            if bracket_match:
                try:
                    parsed = ast.literal_eval(bracket_match.group(0))
                    if isinstance(parsed, list):
                        tokens = [str(x) for x in parsed]
                except Exception:
                    # Fallback to regex tokenization if literal_eval fails
                    tokens = re.findall(r"'([^']+)'", bracket_match.group(0))

            if not tokens:
                # Fallback: parse quoted string after the prefix
                m = re.search(r'Executing action:\s*"([^"]*)"', line)
                if not m:
                    m = re.search(r"Executing action:\s*'([^']*)'", line)
                action_str = m.group(1).strip() if m else line.partition(":")[2].strip()
                # Strip surrounding quotes if any remain
                if (action_str.startswith('"') and action_str.endswith('"')) or (
                    action_str.startswith("'") and action_str.endswith("'")
                ):
                    action_str = action_str[1:-1]
                # Split by whitespace into tokens
                tokens = [t for t in action_str.split() if t]

            current_action = {"Executing_action": tokens if tokens else [""]}
            start_time, end_time, execution_status = None, None, None

        elif line.startswith("start_time:"):
            start_time = float(line.split(":")[1].strip())
        elif line.startswith("end_time:"):
            end_time = float(line.split(":")[1].strip())
            last_end_time = max(last_end_time, end_time)
        elif line.startswith("execution_status:"):
            status = line.split(":")[1].strip()
            if status.lower() == "true":
                execution_status = True
                success_count += 1
            elif status.lower() == "false":
                execution_status = False
            total_count += 1

    if current_action is not None and start_time is not None:
        current_action.update(
            {
                "start_time": start_time,
                "end_time": end_time,
                "execution_status": execution_status,
            }
        )
        actions.append(current_action)

    return actions, last_end_time, success_count, total_count


def result_save_llm(
    approach_name: str,
    user_input: str,
    result: str,
    json_output_path: str,
    computation_time: float,
    scene_name: str,
    attempt: int = 1,
    base_result_path: Path | None = None,
    init_prior_mean: float | None = None,
    case_name: Optional[str] = None,
):
    if init_prior_mean is None:
        init_prior_mean = constants.INIT_PRIOR_MEAN

    if approach_name == "cap_ai2thor_simulation":
        with open(result, "r", encoding="utf-8") as f:
            lines = f.readlines()
        actions, last_end_time, success_count, total_count = parse_cap_log(lines)
    else:
        with open(result, "r", encoding="utf-8") as f:
            lines = f.readlines()
        actions, last_end_time, success_count, total_count = parse_prog_log(lines)

    result_data = {
        "saved_time": get_now_str(),
        "approach": approach_name,
        "attempt": attempt,
        "scene_name": scene_name,
        "plans": [{"plan_name": user_input, "actions": actions}],
        "computation_time": computation_time,
        "success_rate": (
            round(success_count / total_count, 3) if total_count > 0 else None
        ),
        "timing_success_rate": None,
        "scheduler_makespan": None,
        "simulation_makespan": last_end_time,
        "realworld_makespan": None,
    }

    # Determine the base path for results
    result_path_base = base_result_path if base_result_path is not None else RESULT_PATH
    result_path_with_prior = result_path_base / f"init_{int(init_prior_mean)}"

    # Find the next available number for the task name
    num = 1
    while True:
        if case_name:
            output_path = (
                result_path_with_prior
                / f"{case_name}"
                / f"{user_input}_{num}"
                / scene_name
            )
        else:
            output_path = result_path_with_prior / f"{user_input}_{num}" / scene_name
        approach_path = output_path / "approach"
        file_path = approach_path / f"{approach_name}.json"
        if not file_path.exists():
            break
        num += 1
    # Create directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    approach_path.mkdir(parents=True, exist_ok=True)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4)
    print(f"JSON file saved at {file_path}")
    print(f"result_path:{file_path}")
