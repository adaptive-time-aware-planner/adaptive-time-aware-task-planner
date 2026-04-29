import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Assuming this script is in the 'scripts' directory
try:
    # Running from project root
    from src.utils.config.constants import ASSETS_PATH
except ImportError:
    # Running as a standalone script
    ASSETS_PATH = Path(__file__).resolve().parent.parent / "assets"

METADATA_FILE_PATH = ASSETS_PATH / "tasks" / "floorplan_tasks.json"


def _load_task_definitions(
    json_path: Path,
) -> Tuple[Dict[str, Set[str]], Dict[str, List[str]]]:
    """Loads task definitions from floorplan_tasks.json."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    all_tasks_by_floorplan: Dict[str, Set[str]] = defaultdict(set)
    common_tasks = set()
    if "common" in data:
        for task_list in data["common"].values():
            common_tasks.update(task_list)

    floorplan_keys = [k for k in data.keys() if k.startswith("FloorPlan")]
    for key in floorplan_keys:
        fp_tasks = set()
        if key in data:
            for task_list in data[key].values():
                fp_tasks.update(task_list)
        all_tasks_by_floorplan[key] = common_tasks.union(fp_tasks)

    sorted_tasks_by_floorplan = {
        fp: sorted(list(tasks), key=len, reverse=True)
        for fp, tasks in all_tasks_by_floorplan.items()
    }
    # For parsing, we also need a global list for cases where floorplan isn't specified
    global_tasks = set()
    for tasks in all_tasks_by_floorplan.values():
        global_tasks.update(tasks)

    return {}, sorted_tasks_by_floorplan


def _parse_instruction_from_filename(filename: str, all_tasks: List[str]) -> List[str]:
    """Parses an instruction filename to extract constituent tasks."""
    match = re.match(r"\d+_(.+)\.json", filename)
    if not match:
        return []

    instruction_str = match.group(1)
    parsed_tasks: List[str] = []
    temp_str = instruction_str
    while temp_str:
        found_match = False
        for task in all_tasks:
            if temp_str.startswith(task):
                parsed_tasks.append(task)
                temp_str = temp_str[len(task) :]
                if temp_str.startswith("_and_"):
                    temp_str = temp_str[len("_and_") :]
                found_match = True
                break
        if not found_match and temp_str:
            return []  # Failed to parse completely
    return parsed_tasks


def decomposed_to_metadata(decomposed_folder_path: Path) -> Dict[str, Any]:
    """decomposed folder path에서 metadata를 생성합니다.

    Args:
        decomposed_folder_path (Path): decomposed folder path

    Returns:
        Dict[str, Any]: metadata
    """
    _, all_tasks_by_floorplan = _load_task_definitions(METADATA_FILE_PATH)

    metadata = {
        "metadata": {
            "date": datetime.now().strftime("%Y-%m-%d, %H:%M:%S"),
            "generation_criteria": "tasks_and_constraints_count",
            "task_counts": set(),
            "constraint_counts": set(),
            "instruction_counts_by_case": {},
        },
        "instructions_by_case": {},
    }

    case_dirs = sorted(
        [
            d
            for d in decomposed_folder_path.iterdir()
            if d.is_dir() and d.name.startswith("tasks_")
        ]
    )
    floorplan_names = sorted(all_tasks_by_floorplan.keys())

    for case_dir in case_dirs:
        case_name = case_dir.name
        match = re.match(r"tasks_(\d+)_constraints_(\d+)", case_name)
        if match:
            metadata["metadata"]["task_counts"].add(int(match.group(1)))
            metadata["metadata"]["constraint_counts"].add(int(match.group(2)))

        metadata["metadata"]["instruction_counts_by_case"][case_name] = {
            fp: 0 for fp in ["common"] + floorplan_names
        }
        metadata["instructions_by_case"][case_name] = {
            fp: [] for fp in ["common"] + floorplan_names
        }

        instructions_per_floorplan = {}
        floorplan_dirs = sorted([d for d in case_dir.iterdir() if d.is_dir()])

        for fp_dir in floorplan_dirs:
            fp_name = fp_dir.name
            if fp_name not in floorplan_names:
                continue

            tasks_for_this_fp = all_tasks_by_floorplan.get(fp_name, [])
            instructions = []
            for filepath in sorted(fp_dir.glob("*.json")):
                tasks = _parse_instruction_from_filename(
                    filepath.name, tasks_for_this_fp
                )
                if tasks:
                    instruction_str = " and ".join(tasks)
                    instructions.append(instruction_str)
            instructions_per_floorplan[fp_name] = instructions

        if not instructions_per_floorplan:
            continue

        instruction_sets = {
            fp: set(inst) for fp, inst in instructions_per_floorplan.items()
        }

        common_instructions = set()
        is_complete_set = set(instructions_per_floorplan.keys()) == set(floorplan_names)

        if is_complete_set and instruction_sets:
            common_instructions = set.intersection(*instruction_sets.values())

        if common_instructions:
            sorted_common = sorted(list(common_instructions))
            metadata["instructions_by_case"][case_name]["common"] = sorted_common
            metadata["metadata"]["instruction_counts_by_case"][case_name]["common"] = (
                len(sorted_common)
            )

        for fp_name, inst_set in instruction_sets.items():
            specific_instructions = inst_set - common_instructions
            if specific_instructions:
                sorted_specific = sorted(list(specific_instructions))
                metadata["instructions_by_case"][case_name][fp_name] = sorted_specific
                metadata["metadata"]["instruction_counts_by_case"][case_name][
                    fp_name
                ] = len(sorted_specific)

    metadata["metadata"]["task_counts"] = sorted(
        list(metadata["metadata"]["task_counts"])
    )
    metadata["metadata"]["constraint_counts"] = sorted(
        list(metadata["metadata"]["constraint_counts"])
    )

    return metadata


def main():
    """Main function to generate metadata from a decomposed folder."""
    target_folder = ASSETS_PATH / "tasks" / "decomposed_final_251031"
    output_path = target_folder.parent / f"{target_folder.name}_metadata.json"

    print(f"Generating metadata for: {target_folder}")

    metadata_result = decomposed_to_metadata(target_folder)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_result, f, indent=4)

    print(f"\nMetadata file successfully generated at: {output_path}")


if __name__ == "__main__":
    main()
