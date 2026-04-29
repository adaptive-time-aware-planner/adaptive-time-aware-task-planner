### utils/io_utils/ithor_io.py
import json
from pathlib import Path
from typing import Dict, Tuple


def save_object_positions(
    scene_type: str,
    scene_name: str,
    obj_positions: Dict[str, Tuple[float, float, float]],
    save_path: Path,
):
    output_file = save_path / scene_type / "object_init_positions" / f"{scene_name}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as f:
        json.dump(obj_positions, f, indent=4)
        f.write("\n")


def save_environment_data(scene_type: str,scene_name: str, env_data: dict, save_path: Path):
    output_file = save_path / scene_type / "environment" / f"{scene_name}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(env_data, f, indent=4)


def save_navigation_time(scene_type: str, scene_name: str, move_time: dict, save_path: Path):
    output_file = save_path / scene_type / "navigation_time" / f"{scene_name}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(move_time, f, indent=4)
