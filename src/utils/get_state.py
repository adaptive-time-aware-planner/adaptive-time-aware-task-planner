from __future__ import annotations

import json
import logging
from pathlib import Path
from site import abs_paths
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ai2thor.controller import Controller


def get_all_object_states(controller: Controller) -> List[Dict[str, Any]]:
    """AI2-THOR scene에 있는 모든 객체의 지정된 상태 속성들을 가져옵니다.

    이 함수는 AI2-THOR 환경에 쿼리하여 모든 객체에 대해 필요한
    속성(예: name, position, isCooked, isOpen)만 필터링하여 가져옵니다.

    Args:
        controller: scene에 연결된 AI2-THOR controller 인스턴스.

    Returns:
        각각 객체의 필터링된 상태를 나타내는 딕셔너리 리스트.
    """
    event = controller.last_event
    # 마지막 이벤트가 없으면 'Pass' 액션을 수행하여 현재 상태를 가져옵니다.
    if not event or "objects" not in event.metadata:
        event = controller.step(action="Pass")

    # 메타데이터와 객체가 존재하는지 확인
    if not event.metadata or "objects" not in event.metadata:
        # 객체를 찾을 수 없는 경우 경고를 기록하거나 오류를 발생시킬 수 있습니다.
        return []

    objects_metadata = event.metadata["objects"]
    filtered_states = []
    keys_to_keep = [
        "objectId",
        "name",
        "isCooked",
        "isSliced",
        "isOpen",
        "parentReceptacles",
        "isToggled",
        "isFilledWithLiquid",
    ]
    for obj in objects_metadata:
        state = {key: obj[key] for key in keys_to_keep if key in obj}
        filtered_states.append(state)

    return filtered_states


#
def get_changed_object_states(
    state_before: list[dict[str, Any]], state_after: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """두 객체 상태 목록을 비교하여 변경된 객체의 목록을 반환합니다.

    Args:
        state_before: 이전 상태의 객체 목록입니다.
        state_after: 이후 상태의 객체 목록입니다.

    Returns:
        상태가 변경된 객체의 목록.
    """
    before_map = {obj["objectId"]: obj for obj in state_before}
    after_map = {obj["objectId"]: obj for obj in state_after}

    changed_objects_formatted = []
    all_ids = set(before_map.keys()) | set(after_map.keys())

    for obj_id in all_ids:
        state1 = before_map.get(obj_id)
        state2 = after_map.get(obj_id)

        is_changed = False
        if state1 is None or state2 is None:
            # 객체가 추가되거나 삭제된 경우 변경으로 간주
            is_changed = True
        else:
            # 지정된 키에 대해서만 변경 여부 확인
            for key in [
                "isCooked",
                "isSliced",
                "isOpen",
                "parentReceptacles",
                "isToggled",
                "isFilledWithLiquid",
            ]:
                if state1.get(key) != state2.get(key):
                    is_changed = True
                    break

        if is_changed:
            if state2:  # Object changed or was added
                properties = state2.copy()
                object_name = properties.pop("name")
                properties.pop(
                    "objectId"
                )  # Don't include objectId in the properties dict

                formatted_change = {
                    "object_name": object_name,
                    "property": properties,
                }
                changed_objects_formatted.append(formatted_change)
    return changed_objects_formatted


def save_scene_state(
    controller: Optional[Controller],
    output_path: Path,
    case_name: str,
    scene_name: str,
    instruction: str,
    approach_name: str,
    state_label: str,
) -> None:
    """scene에 있는 모든 객체의 현재 상태를 가져와 JSON 파일로 저장합니다.

    상태는 지정된 출력 디렉토리 내의 '{scene_name}_state.json' 파일에 저장됩니다.

    Args:
        controller: The AI2-THOR controller instance for the scene.
        output_path: 상태 파일이 저장될 경로.
        scene_name: 출력 파일 이름에 사용될 scene의 이름.
    """
    output_path = (
        output_path
        / case_name
        / instruction
        / scene_name
        / approach_name
        / f"{state_label}_state.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 컨트롤러가 없고 FloorPlan301이면 빈 JSON 파일을 생성하고 종료
    if controller is None and scene_name == "FloorPlan301":
        with output_path.open("w", encoding="utf-8") as f:
            json.dump([], f, indent=4)
        print(
            f"'{scene_name}'의 scene 상태가 {output_path}에 빈 JSON으로 저장되었습니다."
        )
        return

    object_states = get_all_object_states(controller)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(object_states, f, indent=4)

    print(f"'{scene_name}'의 scene 상태가 {output_path.resolve()} 저장되었습니다.")
    print(
        f"trajectory_log.json 파일이 {output_path.parent / 'trajectory_log.json'} 에 저장되었습니다."
    )
    print(
        f"evaluation_result.json 파일이 {output_path.parent / 'evaluation_result.json'} 에 저장되었습니다."
    )


def get_changed_object_states_ros(
    state_before: tuple[dict[str, Any], Optional[str]],
    state_after: tuple[dict[str, Any], Optional[str]],
) -> list[dict[str, Any]]:
    """ROS object position and held object state change check"""

    obj_state_before, held_object_before = state_before
    obj_state_after, held_object_after = state_after

    state_changes = []

    # Identify all objects that exist or are held
    all_objects = set(obj_state_before.keys()) | set(obj_state_after.keys())
    if held_object_before:
        all_objects.add(held_object_before)
    if held_object_after:
        all_objects.add(held_object_after)

    for obj_name in all_objects:
        # 1. Check State Change
        state_before = obj_state_before.get(obj_name, {})
        state_after = obj_state_after.get(obj_name, {})

        # 비교할 속성 키 목록
        keys_to_check = ["position", "isCooked", "isToggled", "parentReceptacles"]

        is_changed = False
        changed_properties = {}

        for key in keys_to_check:
            val_before = state_before.get(key)
            val_after = state_after.get(key)

            # 값이 다르면 변경으로 간주 (둘 다 None인 경우는 제외)
            if val_before != val_after:
                is_changed = True
                changed_properties[key] = val_after

        if is_changed:
            state_changes.append(
                {"object_name": obj_name, "property": changed_properties}
            )

    return state_changes


if __name__ == "__main__":
    # 이 함수들을 사용하는 방법에 대한 예제입니다.
    # ai2thor가 설치되어 있어야 합니다.
    try:
        from ai2thor.controller import Controller
    except ImportError:
        print("이 예제를 실행하려면 ai2thor를 설치해주세요: pip install ai2thor")
    else:
        # 사용 예시:
        # 특정 scene에 대한 controller를 초기화합니다.
        # FloorPlan1은 테스트에 흔히 사용되는 scene입니다.
        scene = "FloorPlan1"
        controller = Controller(scene=scene)

        # 상태 파일을 저장할 위치를 정의합니다.
        save_directory = Path("./scene_states")

        # 상태를 가져와 저장합니다.
        save_scene_state(controller, save_directory, scene, instruction="test")

        # controller를 중지합니다.
        controller.stop()
