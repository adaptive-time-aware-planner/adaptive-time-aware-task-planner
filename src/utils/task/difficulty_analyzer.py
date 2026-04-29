
from typing import Dict, List

# These lists are derived from scripts/generate_instructions.py
# to determine task difficulty based on the presence of critical tasks.

Floorplan_kitchen_critical_list: List[str] = [
    "boil_potato",
    "cook_egg",
    "fill_pot_with_water"
]

Floorplan1_critical_list: List[str] = [
    "boil_potato", 
    "cook_egg",
    "fill_pot_with_water", 
    "boil_water_with_kettle", 
]

Floorplan7_critical_list: List[str] = [
    "boil_potato", 
    "cook_egg",
    "fill_pot_with_water", 
    "boil_water_with_kettle"
]

Floorplan13_critical_list: List[str] = [
    "boil_potato", 
    "cook_egg",
    "fill_pot_with_water"
]

Floorplan18_critical_list: List[str] = [
    "boil_potato", 
    "cook_egg",
    "fill_pot_with_water",
    "boil_water_with_kettle"
]

Floorplan27_critical_list: List[str] = [
    "boil_potato", 
    "cook_egg",
    "fill_pot_with_water"
]

Floorplan_bathroom_critical_list: List[str] = [
    "fill_bathtub_with_shower_head",
    "clean_the_toilet_with_spray_bottle_and_scrub_brush"
]

Floorplan419_critical_list: List[str] = [
    "fill_bathtub_with_shower_head",
    "clean_the_toilet_with_spray_bottle_and_scrub_brush",
]

Floorplan422_critical_list: List[str] = [
    "fill_bathtub_with_shower_head",
    "clean_the_toilet_with_spray_bottle_and_scrub_brush",
]

Floorplan426_critical_list: List[str] = [
    "fill_bathtub_with_shower_head",
    "clean_the_toilet_with_spray_bottle_and_scrub_brush",
]

Floorplan427_critical_list: List[str] = [
    "fill_bathtub_with_shower_head",
    "clean_the_toilet_with_spray_bottle_and_scrub_brush",
    "clean_the_sink_with_spray_and_dish_sponge"
    
]

# A comprehensive mapping of scene names to their respective critical task lists.
# This dictionary is case-insensitive for scene names to ensure robust matching.
ALL_CRITICAL_LISTS: Dict[str, List[str]] = {
    "floorplan_kitchen": Floorplan_kitchen_critical_list,
    "floorplan1": Floorplan1_critical_list,
    "floorplan7": Floorplan7_critical_list,
    "floorplan13": Floorplan13_critical_list,
    "floorplan18": Floorplan18_critical_list,
    "floorplan27": Floorplan27_critical_list,
    "floorplan_bathroom": Floorplan_bathroom_critical_list,
    "floorplan419": Floorplan419_critical_list,
    "floorplan422": Floorplan422_critical_list,
    "floorplan426": Floorplan426_critical_list,
    "floorplan427": Floorplan427_critical_list,
}


def get_instruction_difficulty(instruction: str, scene_name: str) -> str:
    """
    Determines the difficulty of an instruction based on the number of critical tasks it contains.

    The difficulty is categorized as 'simple', 'normal', or 'hard' based on the count of
    critical tasks:
    - 0 critical tasks: 'simple'
    - 1 critical task: 'normal'
    - 2 or more critical tasks: 'hard'
    - 'unknown' if the scene name is not found in the critical list mapping.

    Args:
        instruction (str): The instruction containing individual tasks separated by ' and '.
        scene_name (str): The name of the scene (e.g., 'FloorPlan1') to look up the
                          corresponding critical task list. The lookup is case-insensitive.

    Returns:
        str: The calculated difficulty level of the instruction ('simple', 'normal', 'hard', 'unknown').
    """
    # instruction을 개별 task로 분리
    tasks = [task.strip() for task in instruction.split(" and ")]
    
    critical_list = ALL_CRITICAL_LISTS.get(scene_name.lower())
    if critical_list is None:
        return "unknown"

    # 각 task가 critical한지 확인 (numeric suffix 제거)
    critical_task_count = 0
    for task in tasks:
        processed_task = task
        if task.rpartition('_')[-1].isdigit():
            processed_task = task.rpartition('_')[0]
        
        if processed_task in critical_list:
            critical_task_count += 1

    if critical_task_count == 0:
        return "simple"
    elif critical_task_count == 1:
        return "normal"
    else:
        return "hard" 