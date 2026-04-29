import copy
import functools
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from ai2thor.controller import Controller

from src.utils.get_state import (
    get_all_object_states,
    get_changed_object_states,
    get_changed_object_states_ros,
)


def log_ros_action_state(func: Callable) -> Callable:
    """ROS 액션 상태 로깅, JSON 파일에 저장"""
    """@log_ros_action_state
    def _simulate_grasp(self, target_obj_id: Optional[str]) -> None:
        self.held_object = target_obj_id

    @log_ros_action_state
    def _simulate_place(self, receptacle_id: Optional[str]) -> None:
        
        if not self.held_object:
            logger.warning("Agent not holding anything. Cannot place. Action FAILED.")
        elif not receptacle_id or receptacle_id.lower() not in self.object_positions:
            raise ValueError(
                f"Place target receptacle '{receptacle_id}' not found in scene positions."
            )
        else:
            logger.debug(f"  Placing '{self.held_object}' on/in '{receptacle_id}'.")
            if self.held_object in self.object_positions:
                self.object_positions[self.held_object] = self.object_positions[
                    receptacle_id.lower()
                ]
            # Ensure directory exists before writing updated positions
            os.makedirs("assets/runtime_state/dynamic", exist_ok=True)
            with open("assets/runtime_state/dynamic/object_states.json", "w") as f:
                json.dump(self.object_positions, f, indent=4)
            self.held_object = None
            
            {
    "banana": 7,
    "plate" : 15,
    "pan" : 31,
    "sausage" : 32,
    "stove": 33,
    "orange_bowl": 34,
    "blue_bowl": 35,
    "countertop": 36,
    "sink|sinkbasin": 37,
    "laundry_machine": 38,
    "chicken": 39,
    "cup": 40,
    "toggle": 41
}
            """

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        log_path: Path = self.trajectory_log_path
        # 1. Action 함수의 로직 시작 전, 현재 상태 확인
        state_before = (copy.deepcopy(self.object_states), self.held_object)
        # Execute the original action
        result = func(self, *args, **kwargs)

        # 외부에서 실행시간을 받아와서 사용
        duration = kwargs.pop("duration", None)

        # 2. Action 함수 로직 종료 후, 상태 확인
        state_after = (copy.deepcopy(self.object_states), self.held_object)

        # 3. 상태 변경된 것만 찾기
        state_changes = get_changed_object_states_ros(state_before, state_after)

        # Log entry 형식 준비
        action_name = func.__name__.replace("_simulate_", "")
        action_args_str = ", ".join(
            [str(a) for a in args] + [f"{k}={v}" for k, v in kwargs.items()]
        )
        primitive_action_str = f"{action_name.upper()} {action_args_str}"

        # 기존 로그 데이터 불러오기
        log_data = []
        if log_path.exists() and log_path.stat().st_size > 0:
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = []

        if not isinstance(log_data, list):
            log_data = []

        next_index = len(log_data)

        # Calculate accumulated duration
        current_time = 0.0
        if log_data:
            last_entry = log_data[-1]
            if "current_time" in last_entry:
                current_time = last_entry["current_time"]
            else:
                current_time = sum(
                    (entry.get("duration") or 0.0)
                    for entry in log_data
                    if isinstance(entry.get("duration"), (int, float))
                )

        step_duration = duration if isinstance(duration, (int, float)) else 0.0
        current_time += step_duration

        log_entry = {
            "index": next_index,
            "primitive_action": primitive_action_str.strip(),
            "duration": duration,
            "current_time": current_time,
            "state_change": state_changes,
        }

        log_data.append(log_entry)

        # 로그 파일 저장
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)

        return result

    return wrapper


def log_action_state(func: Callable) -> Callable:
    """1. Action 함수의 로직 시작 전, 현재 상태 확인 (get_state.py의 get_all_object_states 함수 사용)
            2. Action 함수 로직 종료 후, 상태 확인 (get_state.py의 get_changed_object_states 함수 사용)
            3. 상태 변경된 것만 JSON 파일에 저장
            예) [
        {
            "index": 0,
            "primitive_action": "NAVIGATE_TO Pot",
            "duration": 10.0,
            "state_change": []
        },
        {
            "index": 1,
            "primitive_action": "GRASP Pot",
            "duration": 10.0,
            "state_change": [
                {
                    "object_name": "Pot_aaa",
                    "property": {
                        "parentReceptacles": ["agent"],
                        "isFilledWithLiquid": false,
                        "iscooked": false,
                        "isToggled": false
                    }
                }
            ]
        }
    ]
    """

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        controller: Controller = self.controller
        log_path: Path = self.trajectory_log_json_path

        # 1. Action 함수의 로직 시작 전, 현재 상태 확인
        # controller.step(action="Pass")
        state_before = get_all_object_states(controller)
        # Execute the original action
        result = func(self, *args, **kwargs)
        duration = result if isinstance(result, (int, float)) else 0.0

        # 2. Action 함수 로직 종료 후, 상태 확인
        controller.step(action="Pass")
        state_after = get_all_object_states(controller)

        # 3. 상태 변경된 것만 찾기
        state_changes = get_changed_object_states(state_before, state_after)

        # Log entry 형식 준비
        action_name = func.__name__
        action_args_str = ", ".join(
            [str(a) for a in args] + [f"{k}={v}" for k, v in kwargs.items()]
        )
        primitive_action_str = f"{action_name.upper()} {action_args_str}"

        # 기존 로그 데이터 불러오기
        log_data = []
        if log_path.exists() and log_path.stat().st_size > 0:
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = []

        if not isinstance(log_data, list):
            log_data = []

        next_index = len(log_data)

        # Calculate accumulated duration
        current_time = 0.0
        if log_data:
            last_entry = log_data[-1]
            if "current_time" in last_entry:
                current_time = last_entry["current_time"]
            else:
                current_time = sum(
                    (entry.get("duration") or 0.0)
                    for entry in log_data
                    if isinstance(entry.get("duration"), (int, float))
                )

        step_duration = duration if isinstance(duration, (int, float)) else 0.0
        current_time += step_duration

        log_entry = {
            "index": next_index,
            "primitive_action": primitive_action_str.strip(),
            "duration": duration,
            "current_time": current_time,
            "state_change": state_changes,
        }

        log_data.append(log_entry)

        # 로그 파일 저장
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)

        return result

    return wrapper
