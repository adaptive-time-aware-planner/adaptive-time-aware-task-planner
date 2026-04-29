"""Reclassify decomposed task JSON files based on their actual content.

This script iterates through all JSON files in the 'assets/tasks/decomposed' directory,
counts the number of high-level tasks and critical temporal constraints
(where "Urgency" is true) within each file, and then copies the file to a new
directory structure under 'assets/tasks/reclassified' that reflects these counts.
After processing, it also generates a `reclassification_metadata.json` file
summarizing the number of instructions in each reclassified category.

The new structure will be:
assets/tasks/reclassified/t{M}_c{N}/{scene_name}/{renumbered_filename}.json
- {M}: Number of high-level tasks
- {N}: Number of critical temporal constraints
- {scene_name}: The original scene name (e.g., FloorPlan1)
"""

import hashlib
import json
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# --- Setup ---
ROOT_PATH = Path(__file__).parent.parent
ASSETS_PATH = ROOT_PATH / "assets"
SOURCE_DIR_NAMES = [
    "decomposed_original",
    "decomposed_critical_coffee",
    "decomposed_critical_microwave",
    "decomposed_critical_coffee_microwave",
]
SOURCE_DIRS = [ASSETS_PATH / "tasks" / name for name in SOURCE_DIR_NAMES]
DEST_DIR = ASSETS_PATH / "tasks" / "reclassified"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def count_criteria(task_data: List[Dict[str, Any]], task_file_path: Path) -> (int, int):
    """Counts high-level tasks and critical temporal constraints in a task JSON structure.
    - num_instructions: Number of instructions in the task file
    - num_tasks: Number of high-level tasks in the task file
    - num_subtasks: Number of subtasks in the task file
    - num_critical_constraints: Number of critical temporal constraints in the task file
    Args:
        task_data: List[Dict[str, Any]]
        task_file_path: Path
    Returns:
        tuple: (num_instructions, num_tasks, num_subtasks, num_critical_constraints)
    """
    num_instructions = int(task_file_path.parent.parent.name.split("_")[1])
    num_tasks = len(task_data)
    num_subtasks = 0
    num_critical_constraints = 0

    for task in task_data:
        num_subtasks += len(task.get("Subtasks", []))
        for subtask in task.get("Subtasks", []):
            for constraint in subtask.get("TemporalConstraints", []):
                if constraint.get("Urgency") is True and constraint.get("Interval") > 0:
                    num_critical_constraints += 1

    return num_instructions, num_tasks, num_subtasks, num_critical_constraints


def get_base_filename(path: Path) -> str:
    """Removes the leading number prefix (e.g., '25_') from a filename."""
    return re.sub(r"^\d+_", "", path.name)


def main() -> None:
    """Main function to perform the reclassification."""
    # Collect files from all source directories
    json_files = []
    logging.info(f"Scanning for JSON files in source directories: {SOURCE_DIR_NAMES}")
    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            logging.warning(f"Source directory not found, skipping: {source_dir}")
            continue
        files_in_dir = list(source_dir.glob("**/*.json"))
        json_files.extend(files_in_dir)
        logging.info(f"  Found {len(files_in_dir)} files in '{source_dir.name}'")

    if not json_files:
        logging.error("No JSON files found in any of the source directories. Aborting.")
        return

    if DEST_DIR.exists():
        logging.warning(
            f"Destination directory {DEST_DIR} already exists. Deleting it."
        )
        shutil.rmtree(DEST_DIR)
    DEST_DIR.mkdir(parents=True)

    logging.info(
        f"Found a total of {len(json_files)} files to reclassify and renumber."
    )

    # Group files by their new destination directory
    destination_map = defaultdict(list)
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            num_instructions, num_tasks, num_subtasks, num_critical_constraints = (
                count_criteria(data, file_path)
            )
            scene_name = file_path.parent.name
            new_folder_name = f"i{num_instructions}_tc{num_critical_constraints}"
            new_dir = DEST_DIR / new_folder_name / scene_name
            destination_map[new_dir].append(file_path)

        except (json.JSONDecodeError, TypeError, IndexError) as e:
            logging.error(f"Could not read or parse file {file_path}, skipping: {e}")
        except Exception as e:
            logging.error(
                f"An unexpected error occurred while processing {file_path}, skipping: {e}"
            )

    # --- Deduplication Step ---
    logging.info("Deduplicating files based on content hash...")
    unique_destination_map = defaultdict(list)
    seen_hashes_per_destination = defaultdict(set)
    deduplicated_count = 0

    for dest_dir, source_files in destination_map.items():
        for file_path in source_files:
            try:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                    file_hash = hashlib.sha256(file_bytes).hexdigest()

                if file_hash not in seen_hashes_per_destination[dest_dir]:
                    seen_hashes_per_destination[dest_dir].add(file_hash)
                    unique_destination_map[dest_dir].append(file_path)
                else:
                    deduplicated_count += 1
            except IOError as e:
                logging.error(f"Could not read file for hashing {file_path}: {e}")

    logging.info(
        f"Deduplication complete. Removed {deduplicated_count} duplicate file(s)."
    )
    logging.info(
        f"Proceeding with {sum(len(files) for files in unique_destination_map.values())} unique files."
    )

    # --- Re-organize map by case and identify common instructions ---
    case_to_scenes_map = defaultdict(dict)
    for dest_dir, files in unique_destination_map.items():
        case_name = dest_dir.parent.name
        scene_name = dest_dir.name
        case_to_scenes_map[case_name][scene_name] = files

    file_hashes = {}
    all_files = [f for files in unique_destination_map.values() for f in files]
    for path in all_files:
        if path not in file_hashes:
            with open(path, "rb") as f:
                file_bytes = f.read()
                file_hashes[path] = (
                    get_base_filename(path),
                    hashlib.sha256(file_bytes).hexdigest(),
                )

    case_common_hashes = defaultdict(set)
    for case_name, scenes_data in case_to_scenes_map.items():
        scene_content_sets = [
            {file_hashes[p] for p in files} for files in scenes_data.values()
        ]
        if scene_content_sets:
            common_content_tuples = set.intersection(*scene_content_sets)
            case_common_hashes[case_name] = {h for _, h in common_content_tuples}

    # --- Create final destination map with 'common' folders ---
    final_destination_map = defaultdict(list)
    seen_common_hashes = set()

    for dest_dir, files in unique_destination_map.items():
        case_name = dest_dir.parent.name
        scene_name = dest_dir.name
        common_hashes_for_case = case_common_hashes.get(case_name, set())

        for file_path in files:
            _, file_hash = file_hashes[file_path]
            if file_hash in common_hashes_for_case:
                if file_hash not in seen_common_hashes:
                    common_dest_dir = DEST_DIR / case_name / "common"
                    final_destination_map[common_dest_dir].append(file_path)
                    seen_common_hashes.add(file_hash)
            else:
                final_destination_map[dest_dir].append(file_path)

    # --- Copy, rename, and renumber files to final destinations ---
    processed_count = 0
    for new_dir, source_files in final_destination_map.items():
        new_dir.mkdir(parents=True, exist_ok=True)
        sorted_files = sorted(source_files, key=lambda x: get_base_filename(x))

        for i, old_file_path in enumerate(sorted_files, 1):
            try:
                base_name = get_base_filename(old_file_path)
                new_file_name = f"{i:02d}_{base_name}"
                new_file_path = new_dir / new_file_name

                shutil.copy2(str(old_file_path), str(new_file_path))
                processed_count += 1
            except Exception as e:
                logging.error(f"Failed to copy/rename file {old_file_path}: {e}")

    logging.info(
        f"Reclassification and separation complete. Processed {processed_count} unique files."
    )
    logging.info(f"Results are in: {DEST_DIR}")

    # --- Generate Metadata File reflecting the new structure ---
    logging.info("Generating metadata file...")

    sorted_cases = {}
    for case_name in sorted(case_to_scenes_map.keys()):
        scenes_data = case_to_scenes_map[case_name]
        common_hashes_for_case = case_common_hashes.get(case_name, set())

        # Get common instructions info
        common_content_tuples = {
            file_hashes[p]
            for p_list in scenes_data.values()
            for p in p_list
            if file_hashes[p][1] in common_hashes_for_case
        }
        common_instructions_list = sorted({base for base, _ in common_content_tuples})
        num_common = len(common_hashes_for_case)

        # Get scene-specific counts
        scene_counts = {}
        total_specific = 0
        for scene_name, files in scenes_data.items():
            specific_count = sum(
                1 for p in files if file_hashes[p][1] not in common_hashes_for_case
            )
            scene_counts[scene_name] = specific_count
            total_specific += specific_count

        sorted_scene_counts = {
            name: scene_counts[name] for name in sorted(scene_counts.keys())
        }

        sorted_cases[case_name] = {
            "total": num_common + total_specific,
            "common_instruction_count": len(common_instructions_list),
            "common_instructions": common_instructions_list,
            "counts": {"common": num_common, "scenes": sorted_scene_counts},
        }

    metadata = {
        "total_files_processed": processed_count,
        "instruction_counts_by_case": sorted_cases,
    }

    metadata_path = DEST_DIR.parent / "reclassification_metadata.json"
    try:
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        logging.info(f"Metadata successfully saved to {metadata_path}")
    except Exception as e:
        logging.error(f"Failed to write metadata file: {e}")


if __name__ == "__main__":
    main()
