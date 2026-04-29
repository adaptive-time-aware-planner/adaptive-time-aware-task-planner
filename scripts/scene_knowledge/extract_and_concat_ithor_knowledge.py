import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from src.simulation.runner_ai2thor import init_ai2thor_controller
from src.simulation.scene_info_extractor import (
    extract_environment,
    extract_navigation_time,
    extract_object_positions,
)
from src.utils.config import SCENE_KNOWLEDGE_PATH
from src.utils.io_utils import (
    save_environment_data,
    save_navigation_time,
    save_object_positions,
)


def get_scene_type(scene_name: str) -> str:
    """Determine scene type based on scene number.

    Args:
        scene_name (str): Scene name in format "FloorPlan{n}"

    Returns:
        str: Scene type ("kitchen", "living_room", "bedroom", or "bathroom")
    """
    scene_num = int(scene_name.replace("FloorPlan", ""))

    if 1 <= scene_num <= 30:
        return "kitchen"
    elif 201 <= scene_num <= 230:
        return "living_room"
    elif 301 <= scene_num <= 330:
        return "bedroom"
    elif 401 <= scene_num <= 430:
        return "bathroom"
    else:
        raise ValueError(f"Invalid scene number: {scene_num}")


def get_all_scenes() -> Dict[str, List[str]]:
    """Get all scene names organized by scene type."""
    all_scenes = {}

    # Kitchen scenes (1-30)
    all_scenes["kitchen"] = [f"FloorPlan{i}" for i in range(1, 31)]

    # Living room scenes (201-230)
    all_scenes["living_room"] = [f"FloorPlan{i}" for i in range(201, 231)]

    # Bedroom scenes (301-330)
    all_scenes["bedroom"] = [f"FloorPlan{i}" for i in range(301, 331)]

    # Bathroom scenes (401-430)
    all_scenes["bathroom"] = [f"FloorPlan{i}" for i in range(401, 431)]

    return all_scenes


def process_scene(scene_name: str, scene_type: str) -> bool:
    """Process a single scene and return True if successful, False if skipped."""
    scene_path = (
        SCENE_KNOWLEDGE_PATH / scene_type / "environment" / f"{scene_name}_physics.json"
    )

    # Skip if the file already exists
    if scene_path.exists():
        print(f"Skipping {scene_name} - data already exists")
        return False

    print(f"Processing scene: {scene_name}")
    controller = init_ai2thor_controller(scene_name)
    scene = controller.last_event.metadata["sceneName"]

    # Extract and save object positions
    positions = extract_object_positions(controller)
    save_object_positions(scene_type, scene, positions, SCENE_KNOWLEDGE_PATH)

    # Extract and save environment data
    env, object_ids = extract_environment(controller)
    save_environment_data(scene_type, scene, env, SCENE_KNOWLEDGE_PATH)

    controller.stop()
    print(f"Successfully processed scene: {scene}")
    return True


def concatenate_scene_data(scene_type: str, scene_list: List[str]):
    """Concatenate all scene data for a given scene type."""
    concatenated_data = {}
    output_dir = SCENE_KNOWLEDGE_PATH / scene_type / "environment" / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"concatenated_scenes_{scene_type}.json"

    # Skip if concatenated file already exists
    if output_path.exists():
        print(f"Skipping concatenation for {scene_type} - file already exists")
        return

    for scene_name in scene_list:
        scene_path = (
            SCENE_KNOWLEDGE_PATH
            / scene_type
            / "environment"
            / f"{scene_name}_physics.json"
        )

        if not scene_path.exists():
            print(f"Warning: {scene_path} does not exist, skipping...")
            continue

        with open(scene_path, "r") as f:
            scene_data = json.load(f)
            concatenated_data[scene_name] = scene_data

    with open(output_path, "w") as f:
        json.dump(concatenated_data, f, indent=2)
    print(f"Successfully saved concatenated data to {output_path}")


def main():
    all_scenes = get_all_scenes()

    # Process all scenes
    for scene_type, scene_list in all_scenes.items():
        print(f"\nProcessing {scene_type} scenes...")
        for scene_name in scene_list:
            process_scene(scene_name, scene_type)

    # Concatenate data for each scene type
    for scene_type, scene_list in all_scenes.items():
        print(f"\nConcatenating {scene_type} scenes...")
        concatenate_scene_data(scene_type, scene_list)


if __name__ == "__main__":
    main()
