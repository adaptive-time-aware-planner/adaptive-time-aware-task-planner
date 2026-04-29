from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

from assets.result_analysis.utils.evaluator import compute_trial_metrics, evaluate_tasks

# Import with fallback for script execution
from assets.result_analysis.utils.instruction_parser import (
    load_task_info,
    parse_instruction_to_tasks,
)
from assets.result_analysis.utils.specs import TASK_SPECS
from assets.result_analysis.utils.state_change_simulate import load_events_from_file
from assets.result_analysis.utils.summary import aggregate_summary
from src.utils.common.logger import create_module_logger

logger = create_module_logger(__name__)


def _to_spec_key(task_name: str) -> str:
    """Normalize human-readable task name to TASK_SPECS key."""

    return task_name.lower().replace(" and ", "_and_").replace(" ", "_")


def state_base_eval(target_directory: Path | None = None) -> None:
    """Traverse all states folders (states60/100/140), evaluate tasks, and emit single summary in JSON/LaTeX.

    Args:
        target_directory (Path | None): Optional root directory to search for 'states*' folders.
                                        Defaults to 'assets/results/'.
    """

    results_folder: Path = (
        target_directory if target_directory else Path("assets/results/")
    )
    tasks_json_path = (
        Path(__file__).resolve().parents[1] / "tasks" / "floorplan_tasks.json"
    )
    all_task_names, _critical = load_task_info(tasks_json_path)

    if not results_folder.exists():
        logger.error("Results folder not found: %s", results_folder)
        return

    # states60, 80, 100, 120, 140 폴더만 처리
    states_folder_names = [
        "states60",
        "states70",
        "states80",
        "states90",
        "states100",
        "states110",
        "states120",
        "states130",
        "states140",
    ]
    states_folders = []

    # 1. Check direct children (if root_dir points to a specific experiment folder)
    for folder_name in states_folder_names:
        folder_path = results_folder / folder_name
        if folder_path.exists() and folder_path.is_dir():
            states_folders.append(folder_path)

    # 2. If no states folders found directly, check subdirectories (if root_dir points to assets/results/)
    if not states_folders:
        for sub_dir in results_folder.iterdir():
            if sub_dir.is_dir():
                for folder_name in states_folder_names:
                    folder_path = sub_dir / folder_name
                    if folder_path.exists() and folder_path.is_dir():
                        states_folders.append(folder_path)

    if not states_folders:
        logger.error("No valid states folders found in: %s", results_folder)
        return

    logger.info("Processing states folders: %s", [f.name for f in states_folders])

    # 모든 states 폴더의 모든 trials를 하나로 수집
    all_trials: List[Mapping[str, Any]] = []

    for states_folder in states_folders:
        states_name = states_folder.name  # e.g., "states60", "states100", "states140"
        # logger.info("Processing %s...", states_name)

        for difficulty_dir in sorted(d for d in states_folder.iterdir() if d.is_dir()):
            for task_dir in sorted(t for t in difficulty_dir.iterdir() if t.is_dir()):
                instruction_raw = re.sub(r"^\d{2}_", "", task_dir.name)

                parsed_tasks = parse_instruction_to_tasks(
                    instruction_raw, all_task_names
                )
                if not parsed_tasks:
                    logger.warning(
                        "No parsed tasks for instruction_raw: %s", instruction_raw
                    )
                    continue
                spec_task_names = [_to_spec_key(t) for t in parsed_tasks]
                valid_task_names = [t for t in spec_task_names if t in TASK_SPECS]
                if not valid_task_names:
                    logger.warning(
                        "No valid TASK_SPECS for parsed tasks: %s (instruction_raw=%s)",
                        parsed_tasks,
                        instruction_raw,
                    )
                    continue

                for scene_dir in sorted(s for s in task_dir.iterdir() if s.is_dir()):
                    for approach_dir in sorted(
                        a for a in scene_dir.iterdir() if a.is_dir()
                    ):
                        traj_path = approach_dir / "trajectory_log.json"
                        if not traj_path.exists():
                            logger.warning(
                                "trajectory_log.json not found: %s", traj_path
                            )
                            continue
                        try:
                            events_data = load_events_from_file(traj_path)
                        except Exception as e:
                            logger.error(
                                "Failed to load trajectory: %s (%s)", traj_path, e
                            )
                            continue

                        # makespan 계산: trajectory의 모든 action duration 합
                        makespan: float = sum(
                            event.get("duration", 0.0) for event in events_data
                        )

                        task_results = evaluate_tasks(
                            events=events_data,
                            task_names=valid_task_names,
                        )

                        # 각 task별 평가 결과를 저장할 구조
                        evaluation_results = {
                            "metadata": {
                                "states": states_name,
                                "difficulty": difficulty_dir.name,
                                "instruction": instruction_raw,
                                "scene": scene_dir.name,
                                "approach": approach_dir.name,
                                "parsed_tasks": parsed_tasks,
                            },
                            "makespan": makespan,
                            "tasks": {},
                        }

                        for task_name, task_result in task_results.items():

                            # 각 task의 평가 결과 저장
                            task_eval = {
                                "gcr_satisfied": task_result.gcr_pass,
                                "gcr_mid_satisfied_step": task_result.gcr_mid_satisfied_step,
                            }

                            # Multiple TSR 결과 추가
                            if task_result.tsr_results:
                                task_eval["tsrs"] = {}
                                for (
                                    tsr_name,
                                    tsr_result,
                                ) in task_result.tsr_results.items():
                                    task_eval["tsrs"][tsr_name] = {
                                        "passed": tsr_result.passed,
                                        "duration": tsr_result.duration,
                                        "trigger_step": tsr_result.trigger_step,
                                        "end_step": tsr_result.end_step,
                                    }

                            evaluation_results["tasks"][task_name] = task_eval

                        # 평가 결과를 같은 디렉토리에 저장
                        eval_result_path = approach_dir / "evaluation_result.json"
                        with eval_result_path.open("w") as f:
                            json.dump(evaluation_results, f, indent=2)
                        # logger.info("Saved evaluation result to: %s", eval_result_path)

                        trial_metrics = compute_trial_metrics(
                            parsed_tasks=valid_task_names,
                            task_results=task_results,
                            events=events_data,
                        )
                        trial_metrics.update(
                            {
                                "states": states_name,
                                "difficulty": difficulty_dir.name,
                                "instruction": instruction_raw,
                                "scene": scene_dir.name,
                                "approach": approach_dir.name,
                            }
                        )
                        all_trials.append(trial_metrics)

    # 모든 trials를 모아서 하나의 최종 summary 생성
    if all_trials:
        # logger.info("Generating final summary from %d trials...", len(all_trials))
        final_summary = aggregate_summary(all_trials)

        # 최종 요약 파일 저장 (results 폴더 바로 아래)
        summary_path = results_folder / "unified_analysis_summary.json"
        with summary_path.open("w") as f:
            json.dump(final_summary, f, indent=2, sort_keys=True)
        print("\n=== Final Summary (JSON) ===")
        print(f"Saved to: {summary_path}")

        logger.info("Summary generation complete!")
    else:
        logger.warning("No trials were collected across all states folders.")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--root_dir", type=Path, default=Path("assets/results/"))

    args = args.parse_args()
    state_base_eval(args.root_dir)
