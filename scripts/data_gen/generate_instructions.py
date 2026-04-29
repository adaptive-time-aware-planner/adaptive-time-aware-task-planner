"""Generates task instructions of varying difficulty for different floorplans."""

import json
import random
from datetime import datetime
from itertools import combinations
from typing import Any, Dict, List, Set, Tuple

from src.utils.config.constants import ASSETS_PATH

# --- Constraint Definitions ---
# 각 세트는 상호 배타적인 태스크 그룹을 나타냅니다.
# 이는 주로 단일 자원(e.g., Pot, Microwave)의 동시 사용 제약을 반영합니다.
# instruction 리스트는 각 그룹에서 최대 하나의 태스크만 포함해야 유효합니다.
EXCLUSIVE_TASK_GROUPS: List[Set[str]] = [
    # 1. 'Pot' 자원 제약 그룹: Pot은 하나이므로 관련된 행위는 동시에 일어날 수 없습니다.
    #    'boil_potato'는 물을 채우고 끓이는 과정을 포함하는 상위 개념입니다.
    {"boil_potato", "fill_pot_with_water", "boil_water_with_pot"},
    {"cook_egg", "boil_potato", "boil_water_with_pot"},
    # faucet 으로 인한 제약 그룹 
    {
        "fill_pot_with_water",
        "fill_bowl_with_water",
        "boil_water_with_pot",
        "boil_potato",
        "wash_apple_and_lettuce"
    },
    {
        "fill_pot_with_water",
        "fill_bowl_with_water",
        "boil_water_with_pot",
        "boil_potato",
        "wash_a_tomato"
    },
    {
        "fill_pot_with_water",
        "fill_bowl_with_water",
        "boil_water_with_pot",
        "boil_potato",
        "wash_a_butterknife"
    },
    {
        "fill_pot_with_water",
        "fill_bowl_with_water",
        "boil_water_with_pot",
        "boil_potato",
        "wash_a_spatula"
    },
    {
        "fill_pot_with_water",
        "fill_bowl_with_water",
        "boil_water_with_pot",
        "boil_potato",
        "wash_all_fork_and_spoon"
    },
    {
        "fill_pot_with_water",
        "fill_bowl_with_water",
        "boil_water_with_pot",
        "boil_potato",
        "wash_plate_and_cup"
    },
    {
        "fill_pot_with_water",
        "fill_bowl_with_water",
        "boil_water_with_pot",
        "boil_potato",
        "wash_plate_and_cup"
    },
    {
        "fill_pot_with_water",
        "fill_bowl_with_water",
        "boil_water_with_pot",
        "boil_potato",
        "wash_apple_and_lettuce"
    },
    # 2. 'Potato' 조리법 제약 그룹: 감자를 삶는 것과 전자레인지에 데우는 것은 동시에 할 수 없습니다.
    {"boil_potato", "heat_the_potato_using_microwave"},
    # 3. 'Bread' 조리법 제약 그룹: 빵을 데우는 두 가지 방법은 동시에 사용할 수 없습니다.
    {"heat_the_bread_using_microwave", "slice_bread_and_toast_it_using_toaster"},
    # 4. 'Microwave' 자원 제약 그룹: 전자레인지는 하나이므로 동시에 두 가지 다른 음식을 데울 수 없습니다.
    {"heat_the_bread_using_microwave", "heat_the_potato_using_microwave"},
    # 5. '음료' 준비 제약 그룹: 물컵을 준비하는 것과 커피를 만드는 것은 다른 종류의 음료 준비입니다.
    {"prepare_a_water_cup", "make_a_coffee"},
    # 6. '농산물 처리' 논리 제약 그룹: 씻는 행위와 냉장고에 넣는 행위를 동시에 명령하지 않습니다.
    {"wash_apple_and_lettuce", "put_apple_and_lettuce_in_fridge"},
    ####
    #real world constraints
    # pan으로 인한 constraint
    {
        "cook_sausage",
        "cook_chicken",
        "place_pan_on_stove",
    },
    # plate로 인한 constraint
    {
        "put_banana_on_plate",
        "cook_sausage",
        "cook_chicken",
        "put_tomato_on_plate",
    },
    # blue bowl로 인한 constraint
    {
        "put_tomato_in_blue_bowl",
        "move_blue_bowl_to_sink"
    },
    # tomato로 인한 constraint
    {
        "put_tomato_on_plate",
        "put_tomato_in_blue_bowl",
    },


]


def load_and_prepare_floor_plan_tasks() -> Dict[str, Any]:
    """Load and prepare task combinations.
    Returns:
        A dictionary containing 'common_tasks' and scene-specific 'mixed_tasks'.
    """
    config_path = ASSETS_PATH / "tasks" / "floorplan_tasks_realworld.json"
    with open(config_path) as f:
        data: Dict[str, Any] = json.load(f)

    common_tasks = data.get("common", {})

    floorplan_specific_tasks = {
        k: v for k, v in data.items() if k not in ["common", "metadata"]
    }

    common_tasks_tuple = (
        common_tasks.get("critical", []),
        common_tasks.get("non_critical", []),
        common_tasks.get("not_constrained", []),
    )

    mixed_tasks: Dict[str, Tuple[List[str], List[str], List[str]]] = {}
    for scene_name, specific_tasks in floorplan_specific_tasks.items():
        mixed_tasks[scene_name] = (
            common_tasks_tuple[0] + specific_tasks.get("critical", []),
            common_tasks_tuple[1] + specific_tasks.get("non_critical", []),
            common_tasks_tuple[2] + specific_tasks.get("not_constrained", []),
        )

    return {"common_tasks": common_tasks_tuple, "mixed_tasks": mixed_tasks}


def apply_filters(instruction_list: List[str]) -> bool:
    """Instruction 리스트에 대해 정의된 자원 및 논리 제약조건들을 적용합니다.

    이 함수는 상호 배타적인 태스크 그룹(EXCLUSIVE_TASK_GROUPS)을 기반으로,
    주어진 instruction 리스트가 유효한지 검사합니다.
    하나의 instruction 리스트에 각 그룹의 태스크가 1개를 초과하여 포함될 수 없습니다.

    예시:
    - ["boil_potato", "make_a_coffee"] -> 유효 (다른 그룹)
    - ["boil_potato", "fill_pot_with_water"] -> 유효하지 않음 ('Pot' 자원 그룹 내 2개)

    Args:
        instruction_list: 검사할 태스크 문자열의 리스트.

    Returns:
        instruction 리스트가 모든 제약조건을 만족하면 True, 그렇지 않으면 False.
    """
    instruction_set = set(instruction_list)

    for group in EXCLUSIVE_TASK_GROUPS:
        # 현재 instruction 리스트에 포함된 그룹 내 태스크들을 찾습니다.
        found_tasks = instruction_set.intersection(group)
        # 만약 한 그룹에서 1개보다 많은 태스크가 발견되면, 제약조건 위반입니다.
        if len(found_tasks) > 1:
            return False

    # 모든 제약조건 그룹을 통과하면 유효한 instruction 리스트입니다.
    return True


def _generate_combinations(
    base_list: List[str],
    combo_list: List[str],
    combo_size: int,
) -> List[str]:
    """Generate all possible instruction combinations based on a recipe."""
    temp_instructions = []
    for combo in combinations(combo_list, combo_size):
        instruction_list = base_list + list(combo)
        if apply_filters(instruction_list):
            instruction = " and ".join(instruction_list)
            temp_instructions.append(instruction)
    return temp_instructions


def generate_instructions_for_case(
    num_tasks: int,
    num_constraints: int,
    critical_list: List[str],
    non_critical_list: List[str],
    not_constrained_list: List[str],
) -> List[str]:
    """Generate instructions for a specific case of task and constraint counts."""
    # Validate if generation is possible
    if num_constraints > num_tasks:
        return []
    if num_constraints > len(critical_list):
        return []

    num_non_critical_tasks = num_tasks - num_constraints
    non_critical_pool = non_critical_list + not_constrained_list

    if num_non_critical_tasks < 0 or num_non_critical_tasks > len(non_critical_pool):
        return []

    generated_instructions: List[str] = []

    # Iterate through all combinations of critical tasks
    for critical_combo in combinations(critical_list, num_constraints):
        # For each critical combo, generate combinations of non-critical tasks
        base_list = list(critical_combo)
        generated_instructions.extend(
            _generate_combinations(base_list, non_critical_pool, num_non_critical_tasks)
        )

    return list(set(generated_instructions))  # Return unique instructions


def main() -> None:
    """Generate instructions for a 4x4 matrix of task and constraint counts."""
    # 0. Load existing instructions if the file exists
    output_path = ASSETS_PATH / "tasks" / "decomposed_final_revision_metadata_251202_realworld.json"
    existing_instructions_path = (
        ASSETS_PATH / "tasks" / "decomposed_merged_251031_metadata.json"
    )
    existing_instructions_by_case = {}
    if existing_instructions_path.exists():
        print(
            f"Found existing instruction file at {existing_instructions_path}. Loading to merge."
        )
        with open(existing_instructions_path, "r", encoding="utf-8") as f:
            existing_instructions_by_case = json.load(f).get("instructions_by_case", {})

    # 1. Load and prepare task data
    prepared_data = load_and_prepare_floor_plan_tasks()
    common_tasks = prepared_data["common_tasks"]
    mixed_tasks_by_scene = prepared_data["mixed_tasks"]

    # This dictionary will hold all generated instructions, structured by case
    all_generated_instructions: Dict[str, Any] = {}

    task_counts = range(2, 5)  # Number of tasks: 2, 3, 4, 5
    constraint_counts = range(
        0, 3
    )  # Number of constraints (critical tasks): 0, 1, 2, 3

    # Define sampling limits based on intent in comments
    MAX_COMMON_INSTRUCTIONS = 15
    MAX_SCENE_INSTRUCTIONS = 0

    # 2. Loop through all 16 cases
    for num_t in task_counts:
        for num_c in constraint_counts:
            case_key = f"tasks_{num_t}_constraints_{num_c}"
            print(f"--- Processing instructions for: {case_key} ---")

            existing_case_data = existing_instructions_by_case.get(case_key, {})

            # --- Part 3: Common Instructions ---
            # Generate all possible common instructions
            all_common_instructions = generate_instructions_for_case(
                num_tasks=num_t,
                num_constraints=num_c,
                critical_list=common_tasks[0],
                non_critical_list=common_tasks[1],
                not_constrained_list=common_tasks[2],
            )
            # Get existing instructions and create a pool of new candidates
            existing_common_set = set(existing_case_data.get("common", []))
            new_common_pool = sorted(
                list(set(all_common_instructions) - existing_common_set)
            )

            # Sample a fixed number of purely new instructions
            num_to_sample = min(MAX_COMMON_INSTRUCTIONS, len(new_common_pool))
            sampled_new_common = random.sample(new_common_pool, num_to_sample)

            # The result is ONLY the newly sampled instructions
            final_common = sorted(sampled_new_common)
            case_result: Dict[str, Any] = {"common": final_common}
            final_common_set = set(final_common)

            # --- Part 4: Scene-Specific Instructions ---
            for scene_name, tasks in mixed_tasks_by_scene.items():
                # Generate all possible mixed instructions
                mixed_instructions = generate_instructions_for_case(
                    num_tasks=num_t,
                    num_constraints=num_c,
                    critical_list=tasks[0],
                    non_critical_list=tasks[1],
                    not_constrained_list=tasks[2],
                )
                # Filter out any that are now in the final common set
                distinct_mixed = [
                    inst for inst in mixed_instructions if inst not in final_common_set
                ]

                # Get existing instructions and find new candidates
                existing_scene_set = set(existing_case_data.get(scene_name, []))
                new_scene_pool = sorted(list(set(distinct_mixed) - existing_scene_set))

                # Sample a fixed number of purely new instructions
                num_to_sample = min(MAX_SCENE_INSTRUCTIONS, len(new_scene_pool))
                sampled_new_scene = random.sample(new_scene_pool, num_to_sample)

                # The result is ONLY the newly sampled instructions
                final_scene = sorted(sampled_new_scene)
                case_result[scene_name] = final_scene

            all_generated_instructions[case_key] = case_result

    # 5. Assemble the final output with metadata
    instruction_counts_by_case = {
        case_key: {
            scope: len(instructions) for scope, instructions in case_data.items()
        }
        for case_key, case_data in all_generated_instructions.items()
    }

    final_output = {
        "metadata": {
            "date": datetime.now().strftime("%Y-%m-%d, %H:%M:%S"),
            "generation_criteria": "tasks_and_constraints_count",
            "task_counts": list(task_counts),
            "constraint_counts": list(constraint_counts),
            "instruction_counts_by_case": instruction_counts_by_case,
        },
        "instructions_by_case": all_generated_instructions,
    }

    # 6. Save the final dictionary to a single JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)

    print(f"\nSuccessfully generated and merged instructions to {output_path}")


if __name__ == "__main__":
    main()
