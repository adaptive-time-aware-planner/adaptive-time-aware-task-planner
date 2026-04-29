### utils/task/task_io.py
import json
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

from src.utils.config import constants
from src.utils.config.constants import ASSETS_PATH, ROOM_TYPE, SCENE_KNOWLEDGE_PATH
from src.utils.task.task_generator import TaskGenerator


class SceneData(NamedTuple):
    """
    특정 방 타입에 해당하는 데이터를 저장하는 NamedTuple
    """

    file_name: str
    interaction_objects: dict[str, list[str]]
    object_positions: dict[str, tuple[float, float, float]]


def load_task_data_from_sampled_set(
    case_name: str, scene_name: str, instruction: str
) -> dict:
    """
    Load task data from sampled set.
    """
    # Use TASK_PATH from constants to avoid hardcoded folder names.
    task_path = constants.TASK_PATH / case_name / scene_name / instruction
    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")
    with task_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_task_data_from_file(task_file_name: str) -> dict:
    """
    특정 task 파일에서 JSON 데이터를 불러옴.
    """
    target_path = constants.TASK_PATH / task_file_name
    if not target_path.exists():
        try:
            with open(task_file_name, "r", encoding="utf-8") as file:
                print(f"Task file found: {task_file_name}")
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Task file not found: {target_path}")

    with target_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_file(file_path: Path, file_type: str) -> dict | list | str:
    """
    지정된 파일 경로에서 데이터를 로딩한다.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"{file_type.capitalize()} file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        if file_type == "json":
            return json.load(f)
        else:
            return f.read()


def list_task_files(scene_name=None) -> list[Path]:
    """
    TASK_PATH 디렉토리 내 JSON 파일 목록을 이름 기준으로 정렬하여 반환.
    """
    if scene_name is None:
        return sorted(constants.TASK_PATH.glob("*.json"), key=lambda p: p.name)
    else:
        return sorted(
            (constants.TASK_PATH / scene_name).glob("*.json"), key=lambda p: p.name
        )


def load_scene_data(room_type: str, file_name: str) -> SceneData:
    """
    특정 방 타입에 해당하는 데이터를 로드
    """

    def _load_scene_interaction_objects(
        room_type: str, file_name: str
    ) -> dict[str, tuple[float, float, float]]:
        """
        상호작용 가능한 객체를 담은 JSON 파일 로드
        """
        file_path = SCENE_KNOWLEDGE_PATH / room_type / "environment" / file_name
        raw_data = load_file(file_path, "json")
        return raw_data

    def _load_scene_positions(
        room_type: str, file_name: str
    ) -> dict[str, tuple[float, float, float]]:
        """
        객체별 3D 위치를 담은 JSON 파일 로드
        """
        file_path = (
            SCENE_KNOWLEDGE_PATH / room_type / "object_init_positions" / file_name
        )
        raw_data = load_file(file_path, "json")
        return {k: tuple(v) for k, v in raw_data.items()}

    return SceneData(
        file_name,
        _load_scene_interaction_objects(room_type, file_name),
        _load_scene_positions(room_type, file_name),
    )


def get_user_scene_choice() -> SceneData:
    """
    Scene Type을 받고 해당하는 폴더 내 파일 목록을 유저가 선택할 수 있도록 함

    Returns:
        SceneData[str, dict[str, list[str]], dict[str, tuple[float, float, float]]]
        : 유저가 선택한 파일 이름과 상호작용 가능한 객체 목록, 객체별 3D 위치
    """

    while True:
        print("Select a scene from the list below:")
        print("0. Default Scene (FloorPlan1)")
        for idx, room_type in enumerate(ROOM_TYPE, start=1):
            print(f"{idx}. {room_type}")
        try:
            user_input = int(input("Enter the number of your choice: "))
        except ValueError:
            print("숫자를 입력해 주세요.")
            continue

        if user_input == 0:
            return load_scene_data("kitchen", "FloorPlan1_physics.json")
        elif 1 <= user_input <= len(ROOM_TYPE):
            scene_type = ROOM_TYPE[user_input - 1]
            break
        else:
            print(f"Invalid choice. Please select between 1 and {len(ROOM_TYPE)}.")

    scene_path = SCENE_KNOWLEDGE_PATH / scene_type / "object_init_positions"
    scene_file_list = list(scene_path.glob("*.json"))

    if not scene_file_list:
        print(f"⚠️ No JSON files found in: {scene_path}")
        return None

    while True:
        print("Select a scene from the list below:")
        for idx, scene_file in enumerate(scene_file_list, start=1):
            print(f"{idx}. {scene_file.name}")
        try:
            user_input = int(input("Enter the number of your choice: "))
        except ValueError:
            print("숫자를 입력해 주세요.")
            continue

        if 1 <= user_input <= len(scene_file_list):
            scene_file_name = scene_file_list[user_input - 1].name
            return load_scene_data(scene_type, scene_file_name)
        else:
            print(
                f"Invalid choice. Please select between 1 and {len(scene_file_list)}."
            )


def get_user_task_choice(
    task_files: list[Path],
    choice: Optional[int] = None,
    is_rag: bool = False,
    scene_name: Optional[str] = "FloorPlan1",
) -> Tuple[str, Optional[int]]:
    """
    유저에게 task 파일을 선택하거나 새 instruction을 생성할 수 있도록 입력 받음.

    Returns:
        - (task_file_name, index) 또는 (generated_task_name, None)
    """
    print("Select a file from the list below:")
    print("0. new instruction")
    for idx, file_path in enumerate(task_files, start=1):
        print(f"{idx}. {file_path.name}")

    while True:
        try:
            if choice is None:
                choice = int(input("Enter the number of your choice: "))

            if choice == 0:
                user_input = input("Enter your instruction: ")
                TaskGenerator(is_rag).generate_task(user_input, scene_name)

                return user_input, choice
            elif 1 <= choice <= len(task_files):
                return task_files[choice - 1].name, choice
            else:
                print(f"Invalid choice. Please select between 0 and {len(task_files)}.")
        except ValueError as exc:
            print(f"Invalid input. Please enter a number. Error: {exc}")
        choice = None


# def load_task_data_from_file(task_file_name: str) -> dict:
#     """
#     특정 task 파일에서 JSON 데이터를 불러옴.
#     """

#     if not task_file_name.endswith(".json"):
#         task_file_name = f"{task_file_name}.json"
#     target_path = TASK_PATH / task_file_name
#     if not target_path.exists():
#         raise FileNotFoundError(f"Task file not found: {target_path}")

#     with target_path.open("r", encoding="utf-8") as file:
#         return json.load(file)


def load_scene_positions(file_name: str) -> dict[str, tuple[float, float, float]]:
    """
    객체별 3D 위치를 담은 JSON 파일 로드
    """
    scene_num = int(
        file_name.replace("FloorPlan", "")
        .replace("_positions.json", "")
        .replace("_physics_environment.json", "")
    )
    if scene_num > 400:
        scene_type = "bathroom"
    elif scene_num > 300:
        scene_type = "real_world"
    else:
        scene_type = "kitchen"

    file_path = (
        SCENE_KNOWLEDGE_PATH
        / scene_type
        / "object_init_positions"
        / file_name.replace("_environment", "")
    )
    with file_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)
    return {k: tuple(v) for k, v in raw_data.items()}


def get_natural_language_from_task_file(task_file_name: str) -> str:
    """
    주어진 task 파일 이름에 해당하는 자연어 설명을 반환
    """
    nl_path = constants.TASK_PATH / "task_natural_languages.json"
    with nl_path.open("r", encoding="utf-8") as f:
        task_nl_dict = json.load(f)

    task_nl_dict = {k.strip(":"): v for k, v in task_nl_dict.items()}
    return task_nl_dict.get(task_file_name, None)
