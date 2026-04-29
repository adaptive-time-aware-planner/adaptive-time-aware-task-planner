"""Refactors the floorplan_tasks.json file.
This script identifies common tasks across all floorplans, moves them into a
'common' section, and leaves only the unique tasks under each floorplan.
"""

import json
from pathlib import Path
from typing import Dict, List, Set

# Assuming this script is in the 'scripts' directory and utils is accessible
# If not, you might need to adjust the path.
from utils.config.constants import ASSETS_PATH

FloorplanTaskData = Dict[str, Dict[str, List[str]]]


def load_floorplan_tasks(path: Path) -> FloorplanTaskData:
    """Load floorplan task data from the JSON config file."""
    with open(path) as f:
        return json.load(f)


def save_floorplan_tasks(path: Path, data: Dict) -> None:
    """Save the refactored task data to the JSON config file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Successfully refactored and saved tasks to {path}")


def get_common_instructions(
    tasks_dict: FloorplanTaskData,
) -> Dict[str, Set[str]]:
    """Get common instructions shared across all scenes for each task type.
    Args:
        tasks_dict: A dictionary containing tasks for each floorplan.
    Returns:
        A dictionary with sets of common tasks for each type.
    """
    scene_names = list(tasks_dict.keys())
    if not scene_names:
        return {}

    # Initialize common_tasks with the tasks from the first scene
    common_tasks: Dict[str, Set[str]] = {
        "critical": set(tasks_dict[scene_names[0]].get("critical", [])),
        "non_critical": set(tasks_dict[scene_names[0]].get("non_critical", [])),
        "not_constrained": set(tasks_dict[scene_names[0]].get("not_constrained", [])),
    }

    # Find the intersection with the rest of the scenes
    for scene_name in scene_names[1:]:
        for task_type in common_tasks:
            common_tasks[task_type].intersection_update(
                tasks_dict[scene_name].get(task_type, [])
            )

    return common_tasks


def main() -> None:
    """Main function to execute the refactoring."""
    tasks_path = ASSETS_PATH / "tasks" / "floorplan_tasks.json"
    tasks_data = load_floorplan_tasks(tasks_path)

    # 1. Find common tasks
    common_tasks = get_common_instructions(tasks_data)

    # 2. Create the new refactored data structure
    refactored_data = {}

    # 3. Remove common tasks from each floorplan to find unique tasks
    unique_floorplan_tasks: Dict[str, Dict[str, List[str]]] = {}
    for scene_name, tasks in tasks_data.items():
        unique_tasks: Dict[str, List[str]] = {}
        for task_type, task_list in tasks.items():
            unique_tasks[task_type] = sorted(
                list(set(task_list) - common_tasks[task_type])
            )
        unique_floorplan_tasks[scene_name] = unique_tasks

    refactored_data = {
        "common": {
            task_type: sorted(list(tasks)) for task_type, tasks in common_tasks.items()
        },
        **unique_floorplan_tasks,
    }

    # 4. Save the new structure back to the file
    save_floorplan_tasks(tasks_path, refactored_data)


if __name__ == "__main__":
    main()
