import argparse
import json
import re
from typing import Dict, Iterable, List

from src.utils.common import create_module_logger
from src.utils.config.constants import ASSETS_PATH
from src.utils.task.task_generator import TaskGenerator
from src.utils.task.task_util import TaskUtil

logger = create_module_logger(__name__, module_log=True)

INSTRUCTION_FILE = ASSETS_PATH / "tasks" / "decomposed_final_revision_metadata_260402.json"
OUTPUT_DIR_PREFIX = "decomposed_final_revision_metadata_260402"

KITCHEN_SCENES = [
    "FloorPlan1",
    "FloorPlan7",
    "FloorPlan13",
    "FloorPlan18",
    "FloorPlan27",
]


def load_instructions() -> Dict:
    """Load all instructions from the generated criteria-based JSON file."""
    try:
        with open(INSTRUCTION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Instruction file not found: {INSTRUCTION_FILE}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse {INSTRUCTION_FILE}: {e}")
        raise


def filter_cases(
    all_case_keys: Iterable[str], task_counts: List[int], constraint_counts: List[int]
) -> List[str]:
    """Filter case keys based on user-provided task and constraint counts."""
    if not task_counts and not constraint_counts:
        return sorted(list(all_case_keys))

    filtered_keys = []
    for key in all_case_keys:
        match = re.match(r"tasks_(\d+)_constraints_(\d+)", key)
        if not match:
            continue
        num_t, num_c = map(int, match.groups())

        task_match = (not task_counts) or (num_t in task_counts)
        constraint_match = (not constraint_counts) or (num_c in constraint_counts)

        if task_match and constraint_match:
            filtered_keys.append(key)

    return sorted(filtered_keys)


def main() -> None:
    """Process instructions from the criteria-based JSON file to generate structured task data."""
    argparser = argparse.ArgumentParser(
        description="Generate structured tasks from generated_instructions_by_criteria.json.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    argparser.add_argument(
        "--task-counts",
        type=int,
        nargs="*",
        default=[2, 3],
        help="A list of task counts to process (e.g., 2 3). Processes all if not provided.",
    )
    argparser.add_argument(
        "--constraint-counts",
        type=int,
        nargs="*",
        default=[1, 2],
        help="A list of constraint counts to process (e.g., 0 1). Processes all if not provided.",
    )
    argparser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Trial version suffix for output directory (e.g., v1, v2, ..., v5).",
    )
    argparser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Prompt file name under assets/prompts to override the default scene prompt.",
    )
    args = argparser.parse_args()

    output_base = ASSETS_PATH / "tasks" / f"{OUTPUT_DIR_PREFIX}_{args.version}"

    # --- Load Data ---
    instruction_data = load_instructions()
    all_cases = instruction_data.get("instructions_by_case", {})
    cases_to_process = filter_cases(
        all_cases.keys(), args.task_counts, args.constraint_counts
    )

    generator = TaskGenerator(
        is_rag=False,
        is_test=True,
        use_cache=False,
        prompt_file_override=args.prompt_file,
    )

    failed_instructions: List[str] = []

    # --- Main Processing Loop ---
    for case_key in cases_to_process:
        logger.info(f"### Processing Case: {case_key.upper()} ###")
        case_data = all_cases[case_key]
        common_instructions = case_data.get("common", [])

        for scene_name in KITCHEN_SCENES:
            logger.info(f"--- Processing scene: {scene_name} ---")
            scene_instructions = case_data.get(scene_name, [])
            combined_instructions = list(
                dict.fromkeys(scene_instructions + common_instructions)
            )

            if not combined_instructions:
                logger.warning(
                    f"No instructions found for scene {scene_name} in case {case_key}"
                )
                continue

            output_dir = output_base / case_key / scene_name
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Processing {len(combined_instructions)} instructions for scene {scene_name}."
            )

            for idx, instruction in enumerate(combined_instructions, start=1):
                logger.info(
                    f"[{idx}/{len(combined_instructions)}] {instruction}"
                )
                try:
                    result = generator.generate_task(instruction, scene_name)
                except Exception as e:
                    logger.error(f"Failed to generate '{instruction}': {e}. Skipping.")
                    failed_instructions.append(f"{case_key}/{scene_name}: {instruction}")
                    continue

                output_file_name = instruction.replace(" ", "_").replace("'", "")
                output_file = output_dir / f"{idx:02d}_{output_file_name}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                logger.info(f"Saved: {output_file.name}")

                try:
                    subtasks, constraints, _ = TaskUtil.build_tasks_and_constraints(
                        task_data=result,
                        scene_file_name=f"{scene_name}_physics_environment.json",
                        enable_decomposition=True,
                    )
                    logger.debug(f"Subtasks: {subtasks}")
                    logger.debug(f"Constraints: {constraints}")
                except Exception as e:
                    logger.error(
                        f"TaskUtil validation failed for '{instruction}': {e}"
                    )

                logger.info("-" * 20)

            logger.info(f"Finished processing scene: {scene_name}")
            logger.info("=" * 40)

        logger.info(f"### Finished Case: {case_key.upper()} ###\n")

    if failed_instructions:
        logger.warning(f"Failed instructions ({len(failed_instructions)}):")
        for item in failed_instructions:
            logger.warning(f"  - {item}")
    else:
        logger.info("All instructions processed successfully.")


if __name__ == "__main__":
    main()
