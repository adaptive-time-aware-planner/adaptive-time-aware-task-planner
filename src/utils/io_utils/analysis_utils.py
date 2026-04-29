"""
This module provides shared utilities for analyzing simulation result files.

It includes common constants like approach lists and metric keys, as well as
functions for iterating over result files and loading JSON data. This helps
to reduce code duplication across different analysis scripts.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set, Tuple

# Add the project root to the Python path to enable imports from 'src'
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.utils.common import create_module_logger

# --- Logger ---
log = create_module_logger(module_name="analysis_utils", module_log=True)

# --- Common Constants ---
APPROACH_LIST: Set[str] = {
    "prog_ai2thor_simulation.json",
    "cap_ai2thor_simulation.json",
    "dag_bayesian_simulation.json",
    "cpm_simulation.json",
    "dag_edf_simulation.json",
}

METRIC_KEYS: List[str] = [
    "computation_time",
    "simulation_makespan",
    "scheduler_makespan",
    "total_primitive_actions",
    "success_rate",
    "timing_success_rate_sim",
    "timing_success_rate_sched",
]

KITCHEN_SCENES: Set[str] = {
    "FloorPlan1",
    "FloorPlan7",
    "FloorPlan13",
    "FloorPlan18",
    "FloorPlan27",
    "FloorPlan_kitchen",
}
BATHROOM_SCENES: Set[str] = {
    "FloorPlan419",
    "FloorPlan422",
    "FloorPlan426",
    "FloorPlan427",
    "FloorPlan_bathroom",
}


# --- Common Functions ---
def load_json_data(file_path: Path) -> Dict[str, Any]:
    """
    Loads and returns data from a JSON file, handling potential errors.

    Args:
        file_path: The path to the JSON file.

    Returns:
        A dictionary containing the JSON data, or an empty dictionary on error.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, PermissionError, json.JSONDecodeError) as e:
        log.error(f"Failed to read or parse file: {file_path} - {e}")
        return {}


def iter_result_files(
    base_dir: Path,
) -> Iterator[Tuple[Path, Path, Path]]:
    """
    Iterates over result JSON files and yields their paths.

    Args:
        base_dir: The base directory containing the simulation results.

    Yields:
        A tuple containing (task_dir, scene_dir, json_file_path).
    """
    log.info(f"Scanning for result files in: {base_dir}")
    for task_dir in base_dir.iterdir():
        if not task_dir.is_dir() or task_dir.name in ["average", ".git"]:
            continue

        for scene_dir in task_dir.iterdir():
            if not scene_dir.is_dir():
                continue

            approach_dir = scene_dir / "approach"
            if not approach_dir.is_dir():
                continue

            for json_file in approach_dir.glob("*.json"):
                if json_file.name in APPROACH_LIST:
                    yield task_dir, scene_dir, json_file


def get_scene_type(scene_name: str) -> str:
    """
    Determines scene type ('kitchen' or 'bathroom') from scene name.

    Args:
        scene_name: The name of the scene.

    Returns:
        A string indicating the scene type ('kitchen', 'bathroom', or 'unknown').
    """
    if scene_name in KITCHEN_SCENES:
        return "kitchen"
    if scene_name in BATHROOM_SCENES:
        return "bathroom"
    log.warning(
        f"Scene '{scene_name}' could not be classified as 'kitchen' or 'bathroom'."
    )
    return "unknown"


def get_difficulty(task_name: str) -> str:
    """
    Determines the difficulty of a task based on critical sub-tasks in its name.

    Args:
        task_name: The name of the task directory.

    Returns:
        A string representing the difficulty ('easy', 'medium', 'hard').
    """
    critical_list = [
        "fill_bathtub_with_shower_head",
        "clean_the_toilet_with_spray_bottle_and_scrub_brush",
        "clean_the_sink_with_spray_and_dish_sponge",
        "boil_water_with_kettle",
        "cook_egg",
        "boil_potato",
        "fill_pot_with_water",
    ]
    critical_count = sum(1 for task in critical_list if task in task_name)

    if critical_count == 0:
        return "easy"
    if critical_count == 1:
        return "medium"
    return "hard"


def common_cli_parser(description: str) -> argparse.ArgumentParser:
    """
    Creates a common command-line parser for analysis scripts.

    Args:
        description: The description of the script for the help message.

    Returns:
        An argparse.ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=str(Path(__file__).resolve().parent),
        help="The base directory to analyze. Defaults to the script's directory.",
    )
    return parser
