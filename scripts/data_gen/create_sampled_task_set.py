"""Create a fixed, sampled task set for experiments to ensure reproducibility.

This script performs the following actions:
1. Reads a predefined list of experimental cases (`SAMPLING_CANDIDATES`).
2. For each case, it identifies instructions that are common across all scenes.
3. Based on a defined policy (`SAMPLE_SIZE`, `PRESERVE_COMMON`), it samples
   instructions from each scene, prioritizing the common ones.
4. Copies the selected and renumbered instruction files to a new output directory.
5. Generates a final metadata JSON file summarizing the content of the new sampled set.
"""

import json
import logging
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

# Ensure the project root is in the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config.constants import ASSETS_PATH

# --- Configuration Constants ---

# 1. Source and Destination
SOURCE_DIR = ASSETS_PATH / "tasks" / "decomposed_final_revision_metadata_260402_v1"
OUTPUT_DIR = (
    ASSETS_PATH / "tasks" / "sampled_18_instruction_set_for_final_experiment_260402"
)
SOURCE_METADATA_PATH = ASSETS_PATH / "tasks" / "decomposed_final_revision_metadata_260402.json"


# 2. Sampling Policy
NUM_COMMON_TO_SAMPLE_PER_CASE = 18
RANDOM_SEED = 42


# 3. Case Filtering Policy
MIN_COMMON_INSTRUCTIONS_REQUIRED = 18


# --- End of Configuration ---

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def get_base_filename(path: Path) -> str:
    """Removes the leading number prefix (e.g., '01_') and .json extension from a filename."""
    name_without_ext = path.name.replace(".json", "")
    match = re.match(r"\d+_(.*)", name_without_ext)
    if match:
        return match.group(1)
    return name_without_ext


def main() -> None:
    """Main function to create the sampled task set."""
    # --- 1. Initialization and Validation ---
    if not SOURCE_METADATA_PATH.exists():
        logging.error(f"Source metadata file not found: {SOURCE_METADATA_PATH}")
        return

    try:
        with open(SOURCE_METADATA_PATH, "r", encoding="utf-8") as f:
            source_metadata = json.load(f)
        logging.info(f"Successfully loaded metadata from {SOURCE_METADATA_PATH}")
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Failed to read or parse metadata file: {e}")
        return

    if OUTPUT_DIR.exists():
        logging.warning(f"Output directory {OUTPUT_DIR} already exists. Deleting it.")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    random.seed(RANDOM_SEED)
    final_metadata = {
        "metadata": {
            "source_directory": str(SOURCE_DIR),
            "source_metadata": str(SOURCE_METADATA_PATH),
            "num_common_sampled_per_case": NUM_COMMON_TO_SAMPLE_PER_CASE,
            "random_seed": RANDOM_SEED,
        },
        "cases": {},
    }

    # --- 2. Case Filtering ---
    logging.info(
        f"Filtering for cases with at least {MIN_COMMON_INSTRUCTIONS_REQUIRED} common instructions..."
    )
    candidate_cases = {}
    instructions_by_case = source_metadata.get("instructions_by_case", {})
    for case_name, case_data in instructions_by_case.items():
        common_instructions = case_data.get("common", [])
        if len(common_instructions) >= MIN_COMMON_INSTRUCTIONS_REQUIRED:
            candidate_cases[case_name] = common_instructions

    logging.info(
        f"Found {len(candidate_cases)} candidate cases: {sorted(candidate_cases.keys())}"
    )

    # --- 3. Sampling, Copying, and Renaming ---
    for case_name, common_instructions in candidate_cases.items():
        case_path = SOURCE_DIR / case_name
        if not case_path.is_dir():
            logging.warning(f"Directory for case '{case_name}' not found. Skipping.")
            logging.warning(f"use nl_to_json.py to generate the folder first.")
            continue

        logging.info(f"--- Processing case: {case_name} ---")

        # 3.1. Sample common instructions for the case
        sampled_instructions_text = sorted(
            random.sample(common_instructions, NUM_COMMON_TO_SAMPLE_PER_CASE)
        )
        # Convert to filename-compatible format
        sampled_instructions = [s.replace(" ", "_") for s in sampled_instructions_text]
        logging.info(f"Sampled {len(sampled_instructions)} instructions.")

        # 3.2. Process each scene individually to preserve unique instruction files
        scene_dirs = sorted([d for d in case_path.iterdir() if d.is_dir()])
        all_scenes_valid = True
        case_file_structure = {}

        for scene_dir in scene_dirs:
            source_files_map = {
                get_base_filename(p): p for p in scene_dir.glob("*.json")
            }
            files_to_copy_for_scene = [
                source_files_map[base]
                for base in sampled_instructions
                if base in source_files_map
            ]

            if len(files_to_copy_for_scene) != NUM_COMMON_TO_SAMPLE_PER_CASE:
                logging.warning(
                    f"In '{case_name}', scene '{scene_dir.name}' is missing some instructions. "
                    f"Found {len(files_to_copy_for_scene)}/{NUM_COMMON_TO_SAMPLE_PER_CASE}. "
                    "Skipping this entire case."
                )
                all_scenes_valid = False
                break
            case_file_structure[scene_dir.name] = sorted(
                files_to_copy_for_scene, key=lambda p: p.name
            )

        if not all_scenes_valid:
            continue

        # 3.3. If all scenes are valid, create directories and copy files
        final_metadata["cases"][case_name] = {
            "common_instructions_sampled": sampled_instructions_text,
            "scenes": {},
        }
        total_copied_for_case = 0
        for scene_name, files_to_copy in case_file_structure.items():
            dest_scene_path = OUTPUT_DIR / case_name / scene_name
            dest_scene_path.mkdir(parents=True, exist_ok=True)

            for i, source_file in enumerate(files_to_copy, 1):
                base_name = get_base_filename(source_file)
                new_filename = f"{i:02d}_{base_name}.json"
                dest_file = dest_scene_path / new_filename
                shutil.copy2(source_file, dest_file)
                total_copied_for_case += 1

            final_metadata["cases"][case_name]["scenes"][scene_name] = {
                "instruction_count": len(files_to_copy)
            }
        logging.info(
            f"Successfully copied {total_copied_for_case} files for case '{case_name}'."
        )

    # --- 4. Generate Final Metadata File ---
    logging.info("Generating final metadata file...")
    output_filename = "sampled_dataset_metadata.json"
    output_filepath = OUTPUT_DIR.parent / f"{OUTPUT_DIR.name}_metadata.json"

    try:
        with output_filepath.open("w", encoding="utf-8") as f:
            json.dump(final_metadata, f, indent=4, ensure_ascii=False)
        logging.info(f"Final metadata successfully saved to {output_filepath}")
    except IOError as e:
        logging.error(f"Failed to write final output file: {e}")

    logging.info("Sampling script finished successfully.")


if __name__ == "__main__":
    main()
