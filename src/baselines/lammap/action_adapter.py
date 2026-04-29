"""
LaMMA-P Action Adapter

LaMMA-P의 생성 코드(GoToObject, PickupObject 등)를 파싱하여
우리 시스템의 primitive action 형식(NAVIGATE_TO, GRASP 등)으로 변환하고
Action handler를 통해 ai2thor에서 실행합니다.
"""

import logging
import re
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

from ai2thor.controller import Controller

if TYPE_CHECKING:
    from ithor.handlers.action import Action


def resolve_object_id(controller: Controller, object_name: str) -> Optional[str]:
    """ai2thor 씬에서 object_name에 매칭되는 objectId를 찾습니다.

    Args:
        controller: AI2-THOR controller
        object_name: 오브젝트 이름 (예: 'Apple', 'Fridge', 'Sink|+01.39|...')

    Returns:
        매칭된 objectId 또는 None
    """
    controller.step("Pass")
    objects = controller.last_event.metadata["objects"]

    # 이미 full objectId인 경우
    if "|" in object_name:
        for obj in objects:
            if obj["objectId"] == object_name:
                return obj["objectId"]
        # 부분 매칭 시도
        for obj in objects:
            if object_name in obj["objectId"]:
                return obj["objectId"]

    # objectType 매칭
    for obj in objects:
        if re.match(object_name, obj["objectId"], re.IGNORECASE):
            return obj["objectId"]

    # objectType 이름으로 매칭
    for obj in objects:
        if obj["objectType"].lower() == object_name.lower():
            return obj["objectId"]

    # 부분 매칭
    for obj in objects:
        if object_name.lower() in obj["objectType"].lower():
            return obj["objectId"]

    return None


def resolve_receptacle_id(
    controller: Controller, receptacle_name: str
) -> Tuple[Optional[str], str]:
    """receptacle objectId를 찾고 PLACE_INSIDE vs PLACE_ON_TOP을 결정합니다.

    Returns:
        (objectId, action_type) where action_type is 'PLACE_INSIDE' or 'PLACE_ON_TOP'
    """
    controller.step("Pass")
    objects = controller.last_event.metadata["objects"]

    target_obj = None

    if "|" in receptacle_name:
        for obj in objects:
            if obj["objectId"] == receptacle_name or receptacle_name in obj["objectId"]:
                target_obj = obj
                break
    else:
        for obj in objects:
            if obj["objectType"].lower() == receptacle_name.lower():
                target_obj = obj
                break
        if not target_obj:
            for obj in objects:
                if receptacle_name.lower() in obj["objectType"].lower():
                    target_obj = obj
                    break

    if not target_obj:
        return None, "PLACE_ON_TOP"

    # openable이면 PLACE_INSIDE, 아니면 PLACE_ON_TOP
    if target_obj.get("openable", False):
        action_type = "PLACE_INSIDE"
    else:
        action_type = "PLACE_ON_TOP"

    return target_obj["objectId"], action_type


def parse_lammap_code_to_actions(code: str) -> List[dict]:
    """LaMMA-P 생성 코드를 파싱하여 액션 리스트로 변환합니다.

    LaMMA-P가 생성하는 코드 형태:
        GoToObject(robots[0], 'Apple')
        PickupObject(robots[0], 'Apple')
        GoToObject(robots[0], 'Sink')
        PutObject(robots[0], 'Apple', 'Sink')
        ToggleObjectOn(robots[0], 'Faucet')
        time.sleep(5)
        ToggleObjectOff(robots[0], 'Faucet')

    Returns:
        List of action dicts: [{'action': 'GoToObject', 'args': ['Apple']}, ...]
    """
    actions = []

    # 함수 호출 패턴 매칭
    # GoToObject(robots[0], 'Apple') 또는 GoToObject(robot, 'Apple')
    func_pattern = re.compile(
        r"(GoToObject|PickupObject|PutObject|ToggleObjectOn|ToggleObjectOff|"
        r"OpenObject|CloseObject|SliceObject|BreakObject|CleanObject|"
        r"DropHandObject|ThrowObject|PushObject|PullObject)"
        r"\s*\(\s*(?:robots?\[?\d*\]?|robots?)\s*,\s*(.*?)\s*\)",
        re.IGNORECASE,
    )

    sleep_pattern = re.compile(r"time\.sleep\s*\(\s*(\d+(?:\.\d+)?)\s*\)")

    for line in code.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("def "):
            continue

        # 함수 호출 매칭
        match = func_pattern.search(line)
        if match:
            func_name = match.group(1)
            args_str = match.group(2)
            # 인자 파싱 - 따옴표로 감싸진 문자열들
            args = re.findall(r"['\"]([^'\"]+)['\"]", args_str)
            actions.append({"action": func_name, "args": args})
            continue

        # time.sleep 매칭
        sleep_match = sleep_pattern.search(line)
        if sleep_match:
            duration = float(sleep_match.group(1))
            actions.append({"action": "Wait", "args": [str(duration)]})
            continue

    return actions


def convert_to_primitive_actions(
    controller: Controller, lammap_actions: List[dict], logger: logging.Logger
) -> List[str]:
    """LaMMA-P 액션 리스트를 우리 시스템의 primitive action 문자열로 변환합니다.

    Mapping:
        GoToObject    → NAVIGATE_TO <objectId>
        PickupObject  → GRASP <objectId>
        PutObject     → PLACE_INSIDE/PLACE_ON_TOP <receptacleId>
        ToggleObjectOn  → TOGGLE_ON <objectId>
        ToggleObjectOff → TOGGLE_OFF <objectId>
        OpenObject    → OPEN <objectId>
        CloseObject   → CLOSE <objectId>
        SliceObject   → SLICE <objectId>
        time.sleep    → WAIT <duration>
    """
    primitive_actions = []

    for action in lammap_actions:
        action_name = action["action"]
        args = action["args"]

        if action_name == "GoToObject" and args:
            obj_id = resolve_object_id(controller, args[0])
            if obj_id:
                primitive_actions.append(f"NAVIGATE_TO {obj_id}")
            else:
                logger.warning(f"Object not found: {args[0]}, skipping NAVIGATE_TO")

        elif action_name == "PickupObject" and args:
            obj_id = resolve_object_id(controller, args[0])
            if obj_id:
                primitive_actions.append(f"GRASP {obj_id}")
            else:
                logger.warning(f"Object not found: {args[0]}, skipping GRASP")

        elif action_name == "PutObject" and len(args) >= 2:
            # PutObject(robot, object, receptacle) - receptacle is args[1]
            recp_id, place_type = resolve_receptacle_id(controller, args[-1])
            if recp_id:
                primitive_actions.append(f"{place_type} {recp_id}")
            else:
                logger.warning(
                    f"Receptacle not found: {args[-1]}, skipping {place_type}"
                )

        elif action_name == "ToggleObjectOn" and args:
            obj_id = resolve_object_id(controller, args[0])
            if obj_id:
                primitive_actions.append(f"TOGGLE_ON {obj_id}")
            else:
                logger.warning(f"Object not found: {args[0]}, skipping TOGGLE_ON")

        elif action_name == "ToggleObjectOff" and args:
            obj_id = resolve_object_id(controller, args[0])
            if obj_id:
                primitive_actions.append(f"TOGGLE_OFF {obj_id}")
            else:
                logger.warning(f"Object not found: {args[0]}, skipping TOGGLE_OFF")

        elif action_name == "OpenObject" and args:
            obj_id = resolve_object_id(controller, args[0])
            if obj_id:
                primitive_actions.append(f"OPEN {obj_id}")
            else:
                logger.warning(f"Object not found: {args[0]}, skipping OPEN")

        elif action_name == "CloseObject" and args:
            obj_id = resolve_object_id(controller, args[0])
            if obj_id:
                primitive_actions.append(f"CLOSE {obj_id}")
            else:
                logger.warning(f"Object not found: {args[0]}, skipping CLOSE")

        elif action_name == "SliceObject" and args:
            obj_id = resolve_object_id(controller, args[0])
            if obj_id:
                primitive_actions.append(f"SLICE {obj_id}")
            else:
                logger.warning(f"Object not found: {args[0]}, skipping SLICE")

        elif action_name == "Wait" and args:
            primitive_actions.append(f"WAIT {args[0]}")

        elif action_name in ("BreakObject", "ThrowObject", "DropHandObject", "PushObject", "PullObject", "CleanObject"):
            if args:
                obj_id = resolve_object_id(controller, args[0])
                logger.warning(
                    f"Action {action_name} not directly supported, skipping"
                )
        else:
            logger.warning(f"Unknown action: {action_name}, skipping")

    return primitive_actions


def execute_primitive_actions(
    controller: Controller,
    action_interface: "Action",
    primitive_actions: List[str],
    logger: logging.Logger,
) -> Tuple[float, bool]:
    """primitive action 리스트를 Action handler를 통해 실행합니다.

    Returns:
        (total_elapsed_time, all_succeeded)
    """
    action_mapping = {
        "NAVIGATE_TO": lambda target: action_interface.move_to(target),
        "GRASP": lambda target: action_interface.pickup(target),
        "PLACE_INSIDE": lambda target: action_interface.put(target),
        "PLACE_ON_TOP": lambda target: action_interface.put(target),
        "OPEN": lambda target: action_interface.open(target),
        "CLOSE": lambda target: action_interface.close(target),
        "TOGGLE_ON": lambda target: action_interface.toggle_on(target),
        "TOGGLE_OFF": lambda target: action_interface.toggle_off(target),
        "SLICE": lambda target: action_interface.slice(target),
        "WAIT": lambda duration: action_interface.wait(round(float(duration), 2)),
    }

    elapsed_time = 0.0
    all_succeeded = True

    for action_str in primitive_actions:
        parts = action_str.split(" ", 1)
        action_type = parts[0]
        target = parts[1] if len(parts) > 1 else None

        if action_type in action_mapping:
            try:
                action_duration = action_mapping[action_type](target)
                elapsed_time += action_duration
            except Exception as e:
                logger.error(f"Action '{action_str}' raised exception: {e}")
                all_succeeded = False
                continue

            success = controller.last_event.metadata.get("lastActionSuccess", False)
            if not success:
                error_msg = controller.last_event.metadata.get("errorMessage", "")
                logger.warning(f"Action '{action_str}' failed: {error_msg}")
                all_succeeded = False

            logger.info(
                f"Action: {action_str}, duration: {round(action_duration, 2)}, "
                f"success: {success}"
            )
            controller.step(action="Pass")
        else:
            logger.warning(f"Unknown action type: {action_type}. Skipping.")

    return round(elapsed_time, 2), all_succeeded
