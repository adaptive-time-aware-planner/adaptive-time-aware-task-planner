from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Prefer shared project logger
try:
    from src.utils.common.logger import create_module_logger
except Exception:
    import sys as _sys
    from pathlib import Path as _Path

    _ROOT = _Path(__file__).resolve().parents[3]
    if str(_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_ROOT))
    from src.utils.common.logger import create_module_logger  # type: ignore

logger = create_module_logger(__name__)


def load_task_info(json_path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    """Load all task names and critical tasks by floorplan from JSON.

    Args:
        json_path: Path to tasks JSON file (e.g., floorplan_tasks.json).

    Returns:
        Tuple of (all_task_names, critical_tasks_by_floorplan).
    """
    if not json_path.exists():
        return [], {}
    with json_path.open("r") as f:
        data: Dict[str, Any] = json.load(f)
    task_names = {
        t
        for v in data.values()
        if isinstance(v, dict)
        for tasks in v.values()
        if isinstance(tasks, list)
        for t in tasks
    }
    critical_tasks = {
        fp: data.get("common", {}).get("critical", []) + v.get("critical", [])
        for fp, v in data.items()
        if isinstance(fp, str) and fp.startswith("FloorPlan")
    }
    return list(task_names), critical_tasks


def parse_instruction_to_tasks(instruction: str, all_task_names: List[str]) -> List[str]:
    """Parse instruction id to list of task names using known task names.

    Args:
        instruction: Instruction id string (underscored).
        all_task_names: Known canonical task names.

    Returns:
        List of parsed task names (human-readable).
    """
    remaining = instruction
    parsed: List[str] = []
    # match longer names first
    sorted_tasks = sorted(all_task_names, key=len, reverse=True)
    while remaining:
        found = False
        for task in sorted_tasks:
            task_id = task.replace(" ", "_").replace(" and ", "_and_")
            if remaining.startswith(task_id):
                parsed.append(task)
                remaining = remaining[len(task_id) :].lstrip("_and_")
                found = True
                break
        if not found:
            if remaining:
                logger.warning("Unparsable instruction part: '%s'", remaining)
            break
    return parsed


