import argparse
import copy
import gc
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TextIO

import numpy as np

# import for ai2thor
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

import src.baselines.cap.util.LMPgen as gen
from src.simulation.runner_ai2thor import init_ai2thor_controller
from src.utils.common import create_module_logger
from src.utils.config import constants
from src.utils.config.constants import set_init_prior_mean, set_init_prior_variance
from src.utils.get_state import save_scene_state
from src.utils.io_utils.result_saver import result_save_llm
from src.utils.io_utils.task_io import load_task_data_from_sampled_set
from src.utils.ros_executor import RosExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from ithor.handlers.action import Action


class ActionFailedError(Exception):
    """AI2-THOR 환경에서 물리적 액션이 실패했을 때 발생하는 예외."""

    def __init__(self, action_name: str, args: tuple, message: str):
        """ActionFailedError를 초기화합니다.

        Args:
            action_name (str): 실패한 액션의 이름.
            args (tuple): 액션에 전달된 인자.
            message (str): 시뮬레이터가 반환한 오류 메시지.
        """
        self.action_name = action_name
        self.args = args
        self.message = message
        super().__init__(f"Action '{action_name}' with args {args} failed: {message}")


def diagnose_failure(controller: Controller, action_name: str, args: tuple) -> str:
    """주어진 액션 실패의 원인을 진단하여 상세한 오류 메시지를 반환합니다.

    Args:
        controller (Controller): 현재 AI2-THOR 컨트롤러 인스턴스.
        action_name (str): 실패한 액션의 이름.
        args (tuple): 실패한 액션에 전달된 인자.

    Returns:
        str: 진단된 실패 원인이 담긴 상세 메시지.
    """
    if controller is None:  # ROS mode
        return f"ROS action '{action_name}' with args {args} failed."

    metadata = controller.last_event.metadata
    last_error = metadata.get("errorMessage", "")

    if action_name.lower() == "pickup":
        object_id = args[0]
        # 1. 손이 이미 차 있는지 확인
        if metadata.get("inventoryObjects"):
            held_obj_type = metadata["inventoryObjects"][0]["objectType"]
            return f"Hand is already full, holding {held_obj_type}."

        # 2. 객체 상태 확인
        target_obj = next(
            (o for o in metadata["objects"] if o["objectId"] == object_id), None
        )
        if not target_obj:
            return f"Object '{object_id}' not found in the current scene."
        if not target_obj["pickupable"]:
            return f"Object '{target_obj['objectType']}' is not pickupable."
        if not target_obj["visible"]:
            return f"Object '{target_obj['objectType']}' is not visible."

        # 3. 부모 컨테이너가 닫혀 있는지 확인
        if target_obj.get("parentReceptacles"):
            for parent_id in target_obj["parentReceptacles"]:
                parent_obj = next(
                    (o for o in metadata["objects"] if o["objectId"] == parent_id),
                    None,
                )
                if (
                    parent_obj
                    and parent_obj.get("openable")
                    and not parent_obj.get("isOpen")
                ):
                    return f"Parent receptacle '{parent_obj['objectType']}' is closed."

        if not target_obj["reachable"]:
            return "Object is too far to reach."

    elif action_name.lower() == "put":
        receptacle_id = args[0]
        # 1. 손이 비어 있는지 확인
        if not metadata.get("inventoryObjects"):
            return "Hand is empty, nothing to put down."

        # 2. 수납 객체 상태 확인
        receptacle_obj = next(
            (o for o in metadata["objects"] if o["objectId"] == receptacle_id), None
        )
        if not receptacle_obj:
            return f"Receptacle '{receptacle_id}' not found."
        if not receptacle_obj["receptacle"]:
            return f"Target '{receptacle_obj['objectType']}' is not a receptacle."
        if receptacle_obj.get("openable") and not receptacle_obj.get("isOpen"):
            return f"Target receptacle '{receptacle_obj['objectType']}' is closed."
        if not receptacle_obj["reachable"]:
            return "Target receptacle is too far to reach."

    # 진단된 원인이 없으면, 시뮬레이터의 마지막 에러 메시지를 반환
    if last_error:
        return last_error

    return "Action failed for an unknown reason."


last_end_time: float = 0.0  # 마지막 액션의 종료 시간을 추적

## LMP Prompts
# LMP(Language Model Program)를 위한 프롬프트 파일 경로
_this_file = Path(__file__).resolve()
_this_dir = _this_file.parent
prompt_scene_ui_path = _this_dir / "data/prompt_scene_ui.txt"
prompt_parse_obj_name_path = _this_dir / "data/prompt_parse_obj_name.txt"
prompt_parse_question_path = _this_dir / "data/prompt_parse_question.txt"
prompt_fgen_path = _this_dir / "data/prompt_fgen.txt"


def read_txt(file_path: Path) -> Optional[str]:
    """지정된 경로의 텍스트 파일을 읽어 내용을 반환합니다.

    Args:
        file_path (Path): 읽어올 파일의 경로.

    Returns:
        Optional[str]: 파일의 내용. 파일을 찾을 수 없거나 오류 발생 시 None을 반환합니다.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
        return None


# 프롬프트 파일 읽기
prompt_scene_ui_content = read_txt(prompt_scene_ui_path)
prompt_parse_obj_name_content = read_txt(prompt_parse_obj_name_path)
prompt_parse_question_content = read_txt(prompt_parse_question_path)
prompt_fgen_content = read_txt(prompt_fgen_path)

# 필수 프롬프트 파일 확인
if not all(
    [
        prompt_scene_ui_content,
        prompt_parse_obj_name_content,
        prompt_parse_question_content,
        prompt_fgen_content,
    ]
):
    print("Error: One or more essential prompt files could not be read. Exiting.")
    sys.exit(1)

prompt_scene_ui = prompt_scene_ui_content.strip()
prompt_parse_obj_name = prompt_parse_obj_name_content.strip()
prompt_parse_question = prompt_parse_question_content.strip()
prompt_fgen = prompt_fgen_content.strip()


def build_temporal_logic_guidelines(wait_units: int) -> str:
    """Temporal logic guideline text with dynamic wait time units.

    Args:
        wait_units (int): Canonical integer wait time to reference in examples.

    Returns:
        str: Multi-line guideline text to be appended to prompts.
    """
    return (
        """
    # Guidelines for Temporal Logic in Task Planning

    - Task Sequencing:
        - Ensure all tasks follow logical order (A before B when prerequisite).
        - Identify and respect prerequisites between tasks.

    - Autonomous Processing & Wait Times:
        - Infer reasonable integer wait times for autonomous operations or process completion.
        - Typical ranges:
            - Microwave/Stove cooking: ~{wait} time units
            - Filling a Pot: ~{wait} time units
            - Filling a Mug: ~{wait} time units

    - Immediate Sequential Actions:
        - If a single continuous operation is split (e.g., pick up then place),
          the time gap between them is 0 (immediate succession).

    - Urgency and Immediate Follow-up:
        - Some actions must immediately follow the prior one due to time-sensitive states.
        - Examples:
            - Turn off stove immediately after cooking completes.
            - Turn off faucet immediately when a container is full.
            - Promptly remove items when machine cycles complete.
        - If a follow-up is not time-critical (e.g., serving/eating later), it need not be immediate.

    - Prohibited Functions:
        - Do not use `time.sleep()`. Use the provided `wait(duration)` function for all delays.
    """
    ).format(wait=wait_units)


def timed_action(
    logger: logging.Logger,
    action_name: str,
    action_func: Callable,
    controller: Controller,
) -> Callable[..., float]:
    """주어진 액션 함수를 감싸 시간 측정 및 로깅을 수행합니다.

    액션 실행 전후로 시간을 측정하고, 그 결과를 지정된 로그 파일에
    `result_save_llm`가 파싱할 수 있는 형식으로 기록합니다.
    액션의 시작 시간은 이전 액션이 끝난 시간으로 설정됩니다.

    Args:
        logger (logging.Logger): 로그를 기록할 로거 객체.
        action_name (str): 로깅에 사용될 액션의 이름.
        action_func (Callable): 실행 시간을 측정하고 로깅할 실제 액션 함수.
        controller (Controller): AI2-THOR 컨트롤러 인스턴스. 액션 성공 여부 확인에 사용됩니다.

    Returns:
        Callable[..., float]: 시간을 측정하고 결과를 로깅하는 래퍼 함수.
                            이 래퍼 함수는 실행된 액션의 소요 시간을 반환합니다.
    """

    def wrapper(*args: Any, **kwargs: Any) -> float:
        """래퍼 함수는 실제 액션을 실행하고 로깅합니다."""
        global last_end_time
        action_log_str = " ".join([action_name] + [str(arg) for arg in args])

        # 행동 전 인벤토리 상태 로깅
        inventory_objs = controller.last_event.metadata.get("inventoryObjects", [])
        held_obj_str = (
            f"holding {inventory_objs[0]['objectType']}"
            if inventory_objs
            else "holding nothing"
        )
        logger.info(f"State before action: {held_obj_str}")

        logger.info(f'Executing action: "{action_log_str}"')

        start_time = last_end_time

        # 실제 액션 함수를 실행하고 반환 값을 확인합니다.
        action_result = action_func(*args, **kwargs)
        event_after_action = controller.last_event

        # 3중 확인: action.py의 반환 값, 시뮬레이터의 공식 성공 여부, 그리고 오류 메시지 존재 여부까지 확인합니다.
        action_failed_explicitly = action_result is False
        last_action_was_unsuccessful = not event_after_action.metadata.get(
            "lastActionSuccess", True
        )
        error_message_exists = bool(event_after_action.metadata.get("errorMessage"))

        if (
            action_failed_explicitly
            or last_action_was_unsuccessful
            or error_message_exists
        ):
            logger.info(f"start_time: {round(start_time, 2)}")
            # 실패 시 시간은 흐르지 않은 것으로 간주합니다.
            logger.info(f"end_time: {round(start_time, 2)}")
            logger.info(f"execution_status: {False}")

            # 실패 원인을 진단하여 상세한 오류 메시지를 생성합니다.
            error_message = diagnose_failure(controller, action_name, args)

            raise ActionFailedError(action_name, args, error_message)

        # 성공 시, 반환 값은 elapsed_time입니다.
        elapsed_time = action_result
        if elapsed_time is None:
            # action_func가 None을 반환하는 경우, 시간 변화가 없음을 의미합니다.
            elapsed_time = 0.0

        end_time = start_time + elapsed_time
        last_end_time = end_time  # 다음 액션을 위해 마지막 종료 시간 업데이트

        # 액션 시간 및 실행 결과 로그
        logger.info(f"start_time: {round(start_time, 2)}")
        logger.info(f"end_time: {round(end_time, 2)}")
        logger.info(f"execution_status: {True}")

        # Synchronize state after the action has been executed and logged.
        controller.step(action="Pass")

        return elapsed_time

    return wrapper


# CAP action name → ROS primitive action verb mapping
_CAP_TO_ROS_VERB = {
    "move_to": "navigate_to",
    "pickup": "grasp",
    "put": "place_on_top",
    "toggle_on": "toggle_on",
    "toggle_off": "toggle_off",
    "open": "open",
    "close": "close",
    "wait": "wait",
    "slice": "slice",
    "monitoring": "monitoring",
}


def parse_cap_to_ros(action_name: str, args: tuple) -> str:
    """CAP의 LLM-생성 액션 호출을 ROS primitive action 문자열로 변환합니다.

    Args:
        action_name (str): CAP 액션 이름 (e.g., "pickup", "move_to").
        args (tuple): 액션에 전달된 인자.

    Returns:
        str: ROS primitive action 문자열 (e.g., "grasp egg").
    """
    if action_name == "drop":
        # drop()은 ROS에서 직접 지원하지 않음; place 대체 사용
        return "place_on_top countertop"

    ros_verb = _CAP_TO_ROS_VERB.get(action_name)
    if ros_verb is None:
        raise ValueError(f"Unknown CAP action '{action_name}' — no ROS mapping defined.")

    if action_name == "wait":
        # wait(duration) — 인자는 숫자
        duration = args[0] if args else 0
        return f"wait {duration}"

    if args:
        target = str(args[0]).lower()
        return f"{ros_verb} {target}"

    return ros_verb


def timed_action_ros(
    logger: logging.Logger,
    action_name: str,
    ros_executor: "RosExecutor",
) -> Callable[..., float]:
    """ROS 모드에서 액션 함수를 RosExecutor를 통해 실행하고 시간을 측정하는 래퍼입니다.

    Args:
        logger (logging.Logger): 로그를 기록할 로거 객체.
        action_name (str): 로깅에 사용될 액션의 이름.
        ros_executor (RosExecutor): ROS 실행기 인스턴스.

    Returns:
        Callable[..., float]: 시간을 측정하고 결과를 로깅하는 래퍼 함수.
    """

    def wrapper(*args: Any, **kwargs: Any) -> float:
        global last_end_time
        # Convert CAP action to ROS primitive
        ros_action = parse_cap_to_ros(action_name, args)
        action_log_str = f"{action_name} {' '.join(str(a) for a in args)}"
        logger.info(f'Executing ROS action: "{action_log_str}" -> "{ros_action}"')

        start_time = last_end_time

        success, elapsed_time, action_logs = ros_executor.execute_primitive_actions(
            [ros_action]
        )

        if not success:
            error_msg = f"ROS action '{ros_action}' failed."
            if action_logs:
                error_msg = action_logs[-1].get("error", error_msg)
            raise ActionFailedError(action_name, args, error_msg)

        end_time = start_time + elapsed_time
        last_end_time = end_time

        logger.info(f"start_time: {round(start_time, 2)}")
        logger.info(f"end_time: {round(end_time, 2)}")
        logger.info(f"execution_status: {True}")

        return elapsed_time

    return wrapper


def build_cfg_scene(temporal_guidelines: str) -> Dict[str, Any]:
    """LMP 설정을 런타임에 생성해 반환합니다.

    Args:
        temporal_guidelines (str): 프롬프트에 주입할 시간 논리 가이드라인.

    Returns:
        Dict[str, Any]: LMP 설정 딕셔너리.
    """
    return {
        "lmps": {
            "scene_ui": {
                "prompt_text": prompt_scene_ui
                + "\n"
                + temporal_guidelines
                + "\nobjects = [{objects}]",
                "engine": "gpt-4o",
                "max_tokens": 2048,
                "temperature": 0,
                "query_prefix": "# ",
                "query_suffix": ".",
                "stop": ["#", "objects = ["],
                "maintain_session": True,
                "debug_mode": False,
                "include_context": True,
                "has_return": False,
                "return_val_name": "ret_val",
            },
            "parse_obj_name": {
                "prompt_text": prompt_parse_obj_name,
                "engine": "gpt-4o",
                "max_tokens": 2048,
                "temperature": 0,
                "query_prefix": "# ",
                "query_suffix": ".",
                "stop": ["#", "objects = ["],
                "maintain_session": False,
                "debug_mode": False,
                "include_context": True,
                "has_return": True,
                "return_val_name": "ret_val",
            },
            "parse_question": {
                "prompt_text": prompt_parse_question,
                "engine": "gpt-4o",
                "max_tokens": 2048,
                "temperature": 0,
                "query_prefix": "# ",
                "query_suffix": ".",
                "stop": ["#", "objects = ["],
                "maintain_session": False,
                "debug_mode": False,
                "include_context": True,
                "has_return": True,
                "return_val_name": "ret_val",
            },
            "fgen": {
                "prompt_text": prompt_fgen + "\n" + temporal_guidelines,
                "engine": "gpt-4o",
                "max_tokens": 2048,
                "temperature": 0,
                "query_prefix": "# define function: ",
                "query_suffix": ".",
                "stop": ["# define", "# example"],
                "maintain_session": False,
                "debug_mode": False,
                "include_context": True,
            },
        }
    }


# LMP가 사용할 수 있는 액션 이름 목록
# 이 목록은 `variable_vars` 초기화와 `timed_action` 적용에 모두 사용됩니다.
_ACTION_NAMES = [
    "pickup",
    "slice",
    "put",
    "drop",
    "toggle_on",
    "toggle_off",
    "open",
    "close",
    "monitoring",
    "wait",
    # "fill",
    "move_to",
]


def setup_LMP(
    controller: Controller,
    action_interface: Action,
    cfg_scene: Dict[str, Any],
    logger: logging.Logger,
    vars_log: TextIO,
    ros_executor: Optional["RosExecutor"] = None,
) -> gen.LMP:
    """LMP (Language Model Program) 환경을 설정하고 초기화합니다.

    AI2-THOR 컨트롤러와 액션 핸들러를 기반으로 LMP 환경을 구성합니다.
    환경에는 초기 객체 목록과 좌표 정보가 포함됩니다.
    또한, LMP가 상호작용할 수 있는 API(액션 함수, 상태 조회 함수 등)를
    정의하고, 시간 측정 및 로깅을 위해 액션 함수들을 `timed_action`으로 감쌉니다.
    마지막으로, 다양한 LMP(함수 생성, 파싱, 상위 레벨 명령어 처리)를
    생성하여 반환합니다.

    Args:
        controller (Controller): AI2-THOR 시뮬레이션 컨트롤러.
        action_interface (Action): 시뮬레이션 환경에서 에이전트의 액션을 정의하는 핸들러.
        cfg_scene (Dict[str, Any]): LMP 설정을 담고 있는 딕셔너리.
        logger (logging.Logger): 액션 로그를 기록할 로거 객체.
        vars_log (TextIO): 변수 로깅을 위한 파일 객체.
        ros_executor (Optional[RosExecutor]): ROS 실행기 인스턴스. None이면 시뮬레이션 모드.

    Returns:
        gen.LMP: 사용자의 상위 레벨 언어 명령을 처리하도록 설정된 메인 LMP 객체.
    """
    # LMP env wrapper
    # 설정 파일을 깊은 복사하여 원본을 유지합니다.
    cfg_scene = copy.deepcopy(cfg_scene)

    # LMP 환경을 감싸는 래퍼를 생성합니다.
    # ROS 모드에서는 controller가 None이며, LMP_wrapper가 내부적으로 처리합니다.
    LMP_env = gen.LMP_wrapper(controller, cfg_scene)

    # LMP가 사용할 수 있는 API(고정 변수)를 정의합니다.
    fixed_vars = {"np": np}
    # fixed_vars.update({"time": time}) # time.sleep() 대신 시뮬레이션의 wait()를 사용하도록 유도하기 위해 주석 처리
    fixed_vars.update({"controller": Controller})

    # LMP가 사용할 수 있는 API(가변 변수, 주로 액션 함수)를 정의합니다.
    if ros_executor is not None:
        # ROS 모드: 모든 액션을 timed_action_ros로 감쌈
        variable_vars = {}
        for action_name in _ACTION_NAMES:
            variable_vars[action_name] = timed_action_ros(
                logger, action_name, ros_executor
            )
    else:
        # 시뮬레이션 모드: action_interface에서 액션 함수를 가져와 timed_action으로 감쌈
        # pickup, slice, put, drop, toggle_on, toggle_off, open, close, monitoring, wait, fill, move to
        variable_vars = {
            k: getattr(action_interface, k)
            for k in _ACTION_NAMES
            if hasattr(action_interface, k)
        }

        # 정의된 액션 함수들을 timed_action으로 감싸 시간 측정 및 로깅을 추가합니다.
        for action_name in _ACTION_NAMES:
            if action_name in variable_vars:
                original_func = variable_vars[action_name]
                variable_vars[action_name] = timed_action(
                    logger, action_name, original_func, controller
                )

    # 환경 상태를 조회하는 함수들을 가변 변수에 추가합니다.
    variable_vars.update(
        {
            k: getattr(LMP_env, k)
            for k in [
                "is_obj_visible",
                "get_obj_names",
                "get_obj_id",
                "get_true_states",
                "get_ability_states",
                "get_parentReceptacles",
                "get_obj_in_hand",
            ]
        }
    )
    for var_name, var_value in variable_vars.items():
        vars_log.write(f"{var_name}: {var_value}\n")

    # 로봇이 메시지를 출력하는 'say' 함수를 추가합니다.
    variable_vars["say"] = lambda msg: print(f"robot says: {msg}")

    # 함수 생성 LMP(lmp_fgen)를 생성합니다.
    lmp_fgen = gen.LMPFGen(cfg_scene["lmps"]["fgen"], fixed_vars, variable_vars)

    # 다른 저수준 LMP(객체 이름 파싱, 질문 파싱)들을 생성합니다.
    variable_vars.update(
        {
            k: gen.LMP(k, cfg_scene["lmps"][k], lmp_fgen, fixed_vars, variable_vars)
            for k in [
                "parse_obj_name",
                "parse_question",
            ]
        }
    )

    # 고수준 언어 명령을 처리하는 메인 LMP(lmp_scene_ui)를 생성합니다.
    lmp_scene_ui = gen.LMP(
        "scene_ui",
        cfg_scene["lmps"]["scene_ui"],
        lmp_fgen,
        fixed_vars,
        variable_vars,
    )

    return lmp_scene_ui


def parse_arguments() -> argparse.Namespace:
    """스크립트 실행을 위한 명령행 인자를 파싱합니다.

    Returns:
        argparse.Namespace: 파싱된 명령행 인자를 담고 있는 객체.
    """
    parser = argparse.ArgumentParser(description="Task Scheduler")
    parser.add_argument(
        "-d",
        "--decomposition",
        action="store_true",
        help="태스크 분해 실행 여부. 지정하지 않으면 실행하지 않습니다.",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="시각화 실행 여부. 지정하지 않으면 실행하지 않습니다.",
    )
    parser.add_argument(
        "-r",
        "--reset",
        action="store_true",
        help="리셋 실행 여부. 지정하지 않으면 실행하지 않습니다.",
    )
    parser.add_argument(
        "-s",
        "--simulation",
        action="store_true",
        help="시뮬레이션 실행 여부. 지정하지 않으면 실행하지 않습니다.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="로그 출력 수준 설정 (default: DEBUG)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="FloorPlan1",
        # 추후에 scene 목록이 생기면 choices = [] 으로 구현한다.
        help="시뮬레이션에 사용할 씬 이름 (default: FloorPlan1)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="01_boil_potato_and_set_the_table.json",
        help="실행할 자연어 명령어",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Path to the log file for this specific run.",
    )
    parser.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="The attempt number for a run.",
    )
    parser.add_argument(
        "--cloud-rendering",
        action="store_true",
        help="Use CloudRendering platform for AI2-THOR.",
    )
    parser.add_argument(
        "--init_prior_mean",
        type=float,
        default=100,
        help="베이지안 추정을 위한 초기 평균값 (기본값: 60.0)",
    )
    parser.add_argument(
        "--init_prior_variance",
        type=float,
        default=100,
        help="베이지안 추정을 위한 초기 분산값 (기본값: constants.py 값)",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="tasks_4_constraints_3",
        help="The name of the case.",
    )
    parser.add_argument(
        "--ablation-name",
        type=str,
        default=None,
        help="The name of the ablation configuration.",
    )

    parser.add_argument(
        "--task-folder-name",
        type=str,
        default=None,
        help="Task folder name for organizing results",
    )

    parser.add_argument(
        "--ros",
        default=False,
        action="store_true",
        help="ROS 실행 여부 (default: False)",
    )

    return parser.parse_args()


def main():
    """메인 실행 함수."""
    # --- 스크립트 설정 및 초기화 ---
    approach_name = "cap_ai2thor_simulation"
    args = parse_arguments()

    scene_name = args.scene
    instruction = args.instruction

    if args.init_prior_mean is not None:
        set_init_prior_mean(args.init_prior_mean)
    if args.init_prior_variance is not None:
        set_init_prior_variance(args.init_prior_variance)

    if args.task_folder_name:
        dynamic_task_path = constants.ASSETS_PATH / "tasks" / args.task_folder_name
        constants.set_task_path(dynamic_task_path)
        base_result_path = constants.RESULT_PATH / args.task_folder_name
    else:
        base_result_path = constants.RESULT_PATH

    # ROS 모드에서는 approach_name에 _ros 접미사 추가
    if args.ros:
        approach_name = f"{approach_name}_ros"

    logger = create_module_logger(
        module_name=approach_name,
        log_file_path=Path(args.log_path) if args.log_path else None,
        level=args.log_level,
    )

    controller = None
    ros_executor = None
    action_handler = None

    try:
        if args.case:
            # Load task data
            load_task_data_from_sampled_set(args.case, scene_name, args.instruction)

        # Use a consistent directory name for state saving (keep numeric prefix when case)
        instruction_dir_name = (
            Path(args.instruction).stem
            if (args.case and args.instruction)
            else (Path(instruction).stem if instruction else instruction)
        )
        # 실패 시 불완전한 trajectory 로그 파일이 남지 않도록 삭제합니다.
        trajectory_path = (
            base_result_path
            / f"states{int(args.init_prior_mean)}/{args.case}/{instruction_dir_name}/{scene_name}/{approach_name}/trajectory_log.json"
        )
        if trajectory_path.exists():
            trajectory_path.unlink()

        if args.ros:
            # ROS 모드: AI2-THOR 컨트롤러 대신 RosExecutor 사용
            controller = None
            ros_executor = RosExecutor(trajectory_log_path=trajectory_path)
            action_handler = None
        else:
            # 시뮬레이션 모드: AI2-THOR 컨트롤러 초기화
            platform_obj = None
            if args.cloud_rendering:
                platform_obj = CloudRendering
            controller = init_ai2thor_controller(scene_name, platform=platform_obj)
            ros_executor = None

            save_scene_state(
                controller=controller,
                output_path=base_result_path / f"states{int(args.init_prior_mean)}",
                case_name=args.case,
                scene_name=scene_name,
                instruction=instruction_dir_name,
                approach_name=approach_name,
                state_label="init",
            )
            # Action 핸들러 초기화
            action_handler = Action(
                controller,
                logger=logger,
                trajectory_log_json_path=trajectory_path,
            )

        # LMP 환경 설정
        wait_units = (
            int(args.init_prior_mean) if args.init_prior_mean is not None else 60
        )
        temporal_guidelines = build_temporal_logic_guidelines(wait_units)
        cfg_scene_runtime = build_cfg_scene(temporal_guidelines)
        with open("vars_log.txt", "w", buffering=1) as vars_log:
            lmp_scene_ui = setup_LMP(
                controller,
                action_handler,
                cfg_scene_runtime,
                logger,
                vars_log,
                ros_executor=ros_executor,
            )

            # --- 태스크 실행 ---
            # 현재 장면에 있는 객체 목록 가져오기
            if args.ros:
                with open("assets/runtime_state/static/object_init_states.json") as f:
                    objs = list(json.load(f).keys())
            else:
                objs = list(
                    set(
                        obj["objectType"]
                        for obj in controller.step("Pass").metadata["objects"]
                    )
                )
            print(f"objs: {objs}")

            print(f"'{instruction}' 명령을 실행합니다...")
            computation_time_start = time.time()

            # LMP를 통해 명령어 실행
            lmp_scene_ui(instruction, objects=f"{objs}")

        # --- 결과 저장 ---
        # 현재 computaion_time은 시뮬레이션 타임을 포함해서 정확하지 않음.
        # 추후에 llmgeneration 방식의 computation_time을 폐기할 수 있으므로 일단 스킵
        computation_time = time.time() - computation_time_start
        result_path = f"{instruction_dir_name}"
        result_args = {
            "approach_name": approach_name,
            "user_input": instruction_dir_name,
            "result": str(
                base_result_path
                / f"states{int(args.init_prior_mean)}/{args.case}/{instruction_dir_name}/{scene_name}/{approach_name}/trajectory_log.json"
            ),
            "json_output_path": result_path,
            "computation_time": computation_time,
            "scene_name": scene_name,
            "attempt": args.attempt,
            "init_prior_mean": args.init_prior_mean,
            "case_name": args.case,
        }

        result_save_llm(**result_args)

        if not args.ros:
            save_scene_state(
                controller=controller,
                output_path=base_result_path / f"states{int(args.init_prior_mean)}",
                case_name=args.case,
                scene_name=scene_name,
                instruction=instruction_dir_name,
                approach_name=approach_name,
                state_label="end",
            )
        print("실행이 완료되었습니다.")
    except Exception as e:
        print("실행 중 치명적인 오류 발생: %s", e)
        if controller:
            controller.stop()
        # 실패 시 불완전한 trajectory 로그 파일이 남지 않도록 삭제합니다.
        trajectory_path = (
            base_result_path
            / f"states{int(args.init_prior_mean)}/{args.case}/{instruction_dir_name}/{scene_name}/{approach_name}/trajectory_log.json"
        )
        if trajectory_path.exists():
            trajectory_path.unlink()
        sys.exit(1)
    finally:
        if ros_executor:
            try:
                ros_executor.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down ROS executor: {e}")
        if controller:
            try:
                controller.stop()
            except Exception as e:
                logger.error(f"Error stopping controller: {e}")
        # Force garbage collection to free memory
        gc.collect()
        print("Controller stopped. Exiting.")


if __name__ == "__main__":
    main()
