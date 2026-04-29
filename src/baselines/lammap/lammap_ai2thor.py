"""
LaMMA-P AI2-THOR Integration

LaMMA-P baseline을 우리 프로젝트의 ai2thor 환경에서 실행하는 메인 엔트리포인트.
기존 baseline (CAP, ProgPrompt)과 동일한 인터페이스 패턴을 따릅니다.

Pipeline:
    1. ai2thor 컨트롤러 초기화 및 씬 설정
    2. 씬에서 객체 정보 추출 → LaMMA-P에 전달
    3. LaMMA-P PDDL 계획 생성 (LLM + Fast Downward)
    4. 계획을 코드로 변환 (LLM)
    5. 생성된 코드를 파싱하여 primitive actions로 변환
    6. Action handler를 통해 ai2thor에서 실행
    7. 결과 저장

Usage:
    python -m src.baselines.lammap.lammap_ai2thor \\
        --scene FloorPlan1 \\
        --instruction "Place the apple in the fridge" \\
        --openai-api-key-file api_key \\
        --gpt-version gpt-4o
"""

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from ithor.handlers.action import Action
from src.baselines.lammap.action_adapter import (
    convert_to_primitive_actions,
    execute_primitive_actions,
    parse_lammap_code_to_actions,
)
from src.simulation.runner_ai2thor import init_ai2thor_controller
from src.utils.common import create_module_logger
from src.utils.config import constants
from src.utils.config.constants import set_init_prior_mean
from src.utils.get_state import save_scene_state
from src.utils.io_utils.result_saver import result_save_llm
from src.utils.ros_executor import RosExecutor


def get_scene_objects_for_lammap(controller: Controller) -> str:
    """ai2thor 씬에서 객체 정보를 추출하여 LaMMA-P가 사용하는 형식으로 반환합니다."""
    controller.step("Pass")
    metadata = controller.last_event.metadata

    obj_list = []
    for obj in metadata["objects"]:
        obj_info = {
            "name": obj["objectType"],
            "objectId": obj["objectId"],
            "mass": obj.get("mass", 0),
            "pickupable": obj.get("pickupable", False),
            "receptacle": obj.get("receptacle", False),
            "openable": obj.get("openable", False),
            "toggleable": obj.get("toggleable", False),
            "sliceable": obj.get("sliceable", False),
        }
        obj_list.append(obj_info)

    return f"\n\nobjects = {json.dumps(obj_list, indent=2)}"


def run_lammap_planning(
    base_path: str,
    task: str,
    floor_plan: int,
    objects_ai: str,
    gpt_version: str,
    api_key_file: str,
    logger: logging.Logger,
    wait_units: int = 100,
) -> str:
    """LaMMA-P의 PDDL 계획 생성 → 코드 변환 파이프라인을 실행합니다.

    Returns:
        생성된 실행 코드 (Python string)
    """
    # LaMMA-P 경로 설정
    lammap_scripts = os.path.join(base_path, "scripts")
    sys.path.insert(0, base_path)
    sys.path.insert(0, lammap_scripts)

    # pddlrun_llmseparate의 핵심 클래스 import
    from scripts.pddlrun_llmseparate import TaskManager
    from plantocode import MimicFormatTranslator

    # 로봇 설정 (단일 로봇, 모든 스킬)
    available_robots = [
        {
            "name": "robot1",
            "skills": [
                "GoToObject", "OpenObject", "CloseObject", "BreakObject",
                "SliceObject", "ToggleObjectOn", "ToggleObjectOff", "PickupObject",
                "PutObject", "DropHandObject", "ThrowObject", "PushObject",
                "PullObject",
            ],
            "mass": 100,
        }
    ]

    # api_key 경로 resolve: 절대경로가 아니면 lammap base 기준으로 찾기
    if not os.path.isabs(api_key_file):
        candidates = [
            api_key_file,                                      # CWD 기준
            os.path.join(base_path, api_key_file),             # lammap 디렉토리 기준
            api_key_file + ".txt",                             # CWD 기준 .txt
            os.path.join(base_path, api_key_file + ".txt"),    # lammap 디렉토리 기준 .txt
        ]
        resolved_key = api_key_file
        for c in candidates:
            if os.path.isfile(c):
                resolved_key = c
                break
        api_key_file = resolved_key

    logger.info(f"[LAMMAP] Planning 시작: {task}")
    logger.info(f"[LAMMAP] api_key_file: {api_key_file}")

    # Step 1: PDDL 계획 생성
    t0 = time.time()
    logger.info("[LAMMAP] TaskManager 초기화 중...")
    task_manager = TaskManager(
        base_path=base_path,
        gpt_version=gpt_version,
        api_key_file=api_key_file,
        wait_units=wait_units,
    )
    logger.info(f"[LAMMAP] TaskManager 초기화 완료 ({time.time()-t0:.1f}s)")

    t1 = time.time()
    logger.info("[LAMMAP] process_tasks 시작 (LLM decompose/allocate/pddl/plan)...")
    task_manager.process_tasks(
        test_tasks=[task],
        available_robots=[available_robots],
        objects_ai=objects_ai,
    )
    logger.info(f"[LAMMAP] process_tasks 완료 ({time.time()-t1:.1f}s)")

    # Step 2: 계획을 실행 코드로 변환
    if task_manager.code_planpddl:
        combined_plan = task_manager.code_planpddl[0]
        logger.info(f"[LAMMAP] code_planpddl 사용 (길이: {len(combined_plan)})")
    elif task_manager.combined_plan:
        combined_plan = task_manager.combined_plan[0]
        logger.info(f"[LAMMAP] combined_plan 사용 (길이: {len(combined_plan)})")
    else:
        raise RuntimeError("LaMMA-P failed to generate a plan")

    t2 = time.time()
    logger.info("[LAMMAP] MimicFormatTranslator 초기화 중...")
    translator = MimicFormatTranslator(
        api_key_file=api_key_file,
        gpt_version=gpt_version,
        wait_units=wait_units,
    )

    logger.info("[LAMMAP] translate_to_mimic_format 호출 중 (LLM)...")
    mimic_code = translator.translate_to_mimic_format(task, combined_plan)
    logger.info(f"[LAMMAP] 코드 변환 완료 ({time.time()-t2:.1f}s, 길이: {len(mimic_code)})")

    # Validate & fix
    t3 = time.time()
    logger.info("[LAMMAP] validate_and_fix 호출 중 (LLM)...")
    is_valid, msg, corrected_code = translator.validate_and_fix_mimic_code(
        mimic_code, task
    )
    if corrected_code and len(corrected_code.strip()) > 0:
        mimic_code = corrected_code
    logger.info(f"[LAMMAP] 코드 검증 완료 ({time.time()-t3:.1f}s, valid={is_valid})")

    return mimic_code


def run_lammap_from_task_file(
    base_path: str,
    task_file: str,
    task_index: int,
    objects_ai: str,
    gpt_version: str,
    api_key_file: str,
    logger: logging.Logger,
) -> str:
    """exp/ 폴더의 태스크 파일에서 특정 태스크를 로드하여 실행합니다."""
    with open(task_file, "r") as f:
        tasks = json.load(f)

    if task_index >= len(tasks):
        raise ValueError(f"Task index {task_index} out of range (max {len(tasks)-1})")

    task_data = tasks[task_index]
    instruction = task_data["instruction"]
    logger.info(f"Loading task from file: {instruction}")

    return run_lammap_planning(
        base_path=base_path,
        task=instruction,
        floor_plan=0,
        objects_ai=objects_ai,
        gpt_version=gpt_version,
        api_key_file=api_key_file,
        logger=logger,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LaMMA-P AI2-THOR Baseline")
    parser.add_argument(
        "--scene",
        type=str,
        default="FloorPlan1",
        help="AI2-THOR 씬 이름 (default: FloorPlan1)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="실행할 태스크 명령어",
    )
    parser.add_argument(
        "--task-file",
        type=str,
        default=None,
        help="exp/ 폴더의 태스크 파일 경로 (--instruction 대신 사용)",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=0,
        help="태스크 파일에서의 인덱스 (default: 0)",
    )
    parser.add_argument(
        "--openai-api-key-file",
        type=str,
        default="api_key",
        help="OpenAI API 키 파일 경로",
    )
    parser.add_argument(
        "--gpt-version",
        type=str,
        default="gpt-4o",
        choices=["gpt-3.5-turbo", "gpt-4o", "gpt-3.5-turbo-16k", "gpt-5"],
        help="사용할 GPT 모델",
    )
    parser.add_argument(
        "--cloud-rendering",
        default=False,
        action="store_true",
        help="CloudRendering 사용 여부",
    )
    parser.add_argument(
        "--ros",
        default=False,
        action="store_true",
        help="ROS 실행 모드 (default: False)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="로깅 레벨",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="로그 파일 경로",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="tasks_3_constraints_2",
        help="The name of the case.",
    )
    parser.add_argument(
        "--ablation-name",
        type=str,
        default=None,
        help="The name of the ablation configuration.",
    )
    parser.add_argument(
        "--simulation",
        default=False,
        action="store_true",
        help="Simulation mode flag.",
    )
    parser.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="시도 횟수 (default: 1)",
    )
    parser.add_argument(
        "--init_prior_mean",
        type=float,
        default=60,
        help="초기 prior mean 값 (default: 60)",
    )
    parser.add_argument(
        "--init_prior_variance",
        type=float,
        default=900,
        help="초기 prior variance 값 (default: 900)",
    )
    parser.add_argument(
        "--task-folder-name",
        type=str,
        default=None,
        help="태스크 폴더 이름",
    )
    parser.add_argument(
        "--llm-cache-file",
        type=str,
        default=None,
        help="사전 계산된 LLM 결과 캐시 파일 경로 (없으면 실제 LLM 호출)",
    )
    return parser.parse_args()


def main():
    approach_name = "lammap_ai2thor_simulation"
    args = parse_arguments()

    scene_name = args.scene
    instruction = args.instruction

    if args.init_prior_mean is not None:
        set_init_prior_mean(args.init_prior_mean)

    if args.task_folder_name:
        dynamic_task_path = constants.ASSETS_PATH / "tasks" / args.task_folder_name
        constants.set_task_path(dynamic_task_path)
        base_result_path = constants.RESULT_PATH / args.task_folder_name
    else:
        base_result_path = constants.RESULT_PATH

    if args.ros:
        approach_name = f"{approach_name}_ros"

    logger = create_module_logger(
        module_name=approach_name,
        log_file_path=Path(args.log_path) if args.log_path else None,
        level=args.log_level,
    )

    # LaMMA-P 기본 경로
    lammap_base = os.path.dirname(os.path.abspath(__file__))

    controller = None
    ros_executor = None
    action_handler = None
    t_main_start = time.time()

    try:
        # instruction_dir_name 및 task_string 결정
        if instruction and instruction.endswith(".json"):
            # run_all 모드: instruction이 JSON 파일명
            instruction_dir_name = Path(instruction).stem
            m = re.match(r"\d+_(.*)", instruction_dir_name)
            task_string = m.group(1).replace("_", " ") if m else instruction_dir_name.replace("_", " ")
        elif instruction:
            # standalone 모드: instruction이 자연어
            instruction_dir_name = re.sub(r"[^a-zA-Z0-9]+", "_", instruction).strip("_")
            task_string = instruction
        elif args.task_file:
            instruction_dir_name = f"lammap_task_{args.task_index}"
            task_string = None  # task_file에서 나중에 설정
        else:
            raise ValueError("--instruction 또는 --task-file이 필요합니다")

        # trajectory 경로
        trajectory_path = (
            base_result_path
            / f"states{int(args.init_prior_mean)}/{args.case}/{instruction_dir_name}/{scene_name}/{approach_name}/trajectory_log.json"
        )
        if trajectory_path.exists():
            trajectory_path.unlink()

        # --- 컨트롤러 초기화 ---
        t_ctrl = time.time()
        logger.info(f"[LAMMAP] 컨트롤러 초기화 시작 (scene={scene_name})...")
        if args.ros:
            ros_executor = RosExecutor(trajectory_log_path=trajectory_path)
        else:
            platform_obj = CloudRendering if args.cloud_rendering else None
            controller = init_ai2thor_controller(scene_name, platform=platform_obj)
            logger.info(f"[LAMMAP] ai2thor Controller 생성 완료 ({time.time()-t_ctrl:.1f}s)")

            save_scene_state(
                controller=controller,
                output_path=base_result_path / f"states{int(args.init_prior_mean)}",
                case_name=args.case,
                scene_name=scene_name,
                instruction=instruction_dir_name,
                approach_name=approach_name,
                state_label="init",
            )

            action_handler = Action(
                controller,
                logger=logger,
                trajectory_log_json_path=trajectory_path,
            )
        logger.info(f"[LAMMAP] 컨트롤러 초기화 완료 ({time.time()-t_ctrl:.1f}s)")

        # --- 씬 객체 정보 추출 ---
        t_obj = time.time()
        logger.info("[LAMMAP] 씬 객체 정보 추출 중...")
        if args.ros:
            with open("assets/runtime_state/static/object_init_states.json") as f:
                ros_objs = json.load(f)
            objects_ai = f"\n\nobjects = {json.dumps(list(ros_objs.keys()))}"
        else:
            objects_ai = get_scene_objects_for_lammap(controller)
        logger.info(f"[LAMMAP] 씬 객체 정보 추출 완료 ({time.time()-t_obj:.1f}s)")

        # --- LaMMA-P 계획 생성 (캐시 우선 조회) ---
        computation_time_start = time.time()
        mimic_code = None
        wait_units = int(args.init_prior_mean) if args.init_prior_mean is not None else 100

        if args.llm_cache_file and Path(args.llm_cache_file).exists():
            version_idx = 0
            if args.task_folder_name:
                vm = re.search(r"_v(\d+)$", args.task_folder_name)
                if vm:
                    version_idx = int(vm.group(1)) - 1

            instr_name = instruction if instruction.endswith(".json") else instruction_dir_name + ".json"
            cache_key = f"{args.case}|{instr_name}|{wait_units}"
            with open(args.llm_cache_file) as _cf:
                _cache = json.load(_cf)

            cached_list = _cache.get(cache_key, [])
            if version_idx < len(cached_list) and cached_list[version_idx]:
                mimic_code = cached_list[version_idx]
                logger.info(
                    f"[LAMMAP] 캐시 히트: {cache_key} [v{version_idx+1}] "
                    f"(코드 길이: {len(mimic_code)})"
                )
            else:
                logger.warning(
                    f"[LAMMAP] 캐시 미스: {cache_key} [v{version_idx+1}] → 실제 LLM 호출"
                )

        if mimic_code is None:
            if args.task_file:
                mimic_code = run_lammap_from_task_file(
                    base_path=lammap_base,
                    task_file=args.task_file,
                    task_index=args.task_index,
                    objects_ai=objects_ai,
                    gpt_version=args.gpt_version,
                    api_key_file=args.openai_api_key_file,
                    logger=logger,
                )
                if not task_string:
                    with open(args.task_file) as f:
                        tasks = json.load(f)
                    task_string = tasks[args.task_index]["instruction"]
            else:
                mimic_code = run_lammap_planning(
                    base_path=lammap_base,
                    task=task_string,
                    floor_plan=int(scene_name.replace("FloorPlan", "")),
                    objects_ai=objects_ai,
                    gpt_version=args.gpt_version,
                    api_key_file=args.openai_api_key_file,
                    logger=logger,
                    wait_units=wait_units,
                )

        logger.info(f"LaMMA-P 계획 생성 완료")
        logger.info(f"생성된 코드:\n{mimic_code[:500]}...")

        # --- 생성된 코드를 파싱 → primitive actions 변환 ---
        lammap_actions = parse_lammap_code_to_actions(mimic_code)
        logger.info(f"파싱된 LaMMA-P 액션 수: {len(lammap_actions)}")

        if not lammap_actions:
            logger.error("LaMMA-P가 실행 가능한 액션을 생성하지 못했습니다")
            sys.exit(1)

        primitive_actions = convert_to_primitive_actions(
            controller, lammap_actions, logger
        )
        logger.info(f"변환된 primitive action 수: {len(primitive_actions)}")
        for i, pa in enumerate(primitive_actions):
            logger.info(f"  [{i+1}] {pa}")

        # --- ai2thor에서 실행 ---
        logger.info("ai2thor 실행 시작...")
        elapsed_time, all_succeeded = execute_primitive_actions(
            controller=controller,
            action_interface=action_handler,
            primitive_actions=primitive_actions,
            logger=logger,
        )

        computation_time = time.time() - computation_time_start

        logger.info(
            f"실행 완료 - elapsed_time: {elapsed_time}s, "
            f"success: {all_succeeded}, "
            f"computation_time: {computation_time:.2f}s"
        )

        # --- 결과 저장 ---
        result_args = {
            "approach_name": approach_name,
            "user_input": instruction_dir_name,
            "result": str(trajectory_path),
            "json_output_path": instruction_dir_name,
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

        logger.info("[LAMMAP] 실행이 완료되었습니다.")

    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        if controller:
            controller.stop()
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
        gc.collect()
        logger.info(f"[LAMMAP] 전체 소요시간: {time.time()-t_main_start:.1f}s. Exiting.")


if __name__ == "__main__":
    main()
