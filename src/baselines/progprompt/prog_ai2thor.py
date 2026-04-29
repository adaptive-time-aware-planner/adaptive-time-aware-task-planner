import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import openai
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from util.utils_execute import LM, ProgPromptActionFailedError, simulate_execution

from ithor.handlers.action import Action
from src.simulation.runner_ai2thor import init_ai2thor_controller
from src.utils.common import create_module_logger
from src.utils.config import constants
from src.utils.config.constants import set_init_prior_mean
from src.utils.get_state import save_scene_state
from src.utils.io_utils.result_saver import result_save_llm
from src.utils.io_utils.task_io import list_task_files
from src.utils.ros_executor import RosExecutor

current_dir = os.path.dirname(os.path.abspath(__file__))  # 이 파일의 현재 경로


def build_temporal_logic_guidelines(wait_units: int) -> str:
    """Create temporal logic guidelines text with dynamic wait time units.

    The returned text is appended to prompts so that the LLM decomposes tasks
    with correct sequencing, wait time inference, and immediate follow-ups.


    Args:
        wait_units (int): The canonical integer wait time to reference in examples.

    Returns:
        str: Multi-line guideline text to inject into prompts.
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
            - Turn off laundry machine immediately after cycle completes.
            - Promptly remove items when machine cycles complete.
        - If a follow-up is not time-critical (e.g., serving/eating later), it need not be immediate.

    - Prohibited Functions:
        - Do not use `time.sleep()`. Use the provided `wait(duration)` function for all delays.
    
    - Object Placement Constraints:
        - For `PLACE_ON_TOP <obj>` or `PLACE_INSIDE <obj>`, the `<obj>` MUST be a valid receptacle (e.g., Table, Sink, Stove, Shelf, Cabinet).
        - Do NOT place items on pickupable objects (e.g., do not place on Chicken, Apple, Bowl).
        - Ensure the target receptacle is appropriate for the item (e.g., place food on plates or in pans, not directly on random furniture if not specified).
    """
    ).format(wait=wait_units)


def parse_arguments() -> argparse.Namespace:
    """
    명령행 인자를 파싱합니다.
    """
    parser = argparse.ArgumentParser(description="Task Scheduler")
    parser.add_argument(
        "-d",
        "--decomposition",
        default=True,
        action="store_true",
        help="태스크 분해 여부 (default: True)",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        default=True,
        action="store_true",
        help="시각화 실행 여부 (default: True)",
    )
    parser.add_argument(
        "-r",
        "--reset",
        default=True,
        action="store_true",
        help="리셋 실행 여부 (default: True)",
    )
    parser.add_argument(
        "-s",
        "--simulation",
        default=True,
        action="store_true",
        help="시뮬레이션 실행 여부 (default: True)",
    )
    parser.add_argument(
        "--ros",
        default=False,
        action="store_true",
        help="ROS 실행 여부 (default: False)",
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
        "--gpt-version",
        type=str,
        default="gpt-4o",
        choices=["gpt-4o", "gpt-4o-mini"],
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="FloorPlan13",
        # 추후에 scene 목록이 생기면 choices = [] 으로 구현한다.
        help="시뮬레이션에 사용할 씬 이름 (default: FloorPlan1)",
    )
    parser.add_argument(
        "--prompt-num-examples", type=int, default=4, choices=range(1, 5)
    )
    parser.add_argument(
        "--prompt-task-examples-ablation",
        type=str,
        default="none",
        choices=["none", "no_comments", "no_feedback", "no_comments_feedback"],
    )
    parser.add_argument(
        "--openai-api-key", type=str, default=os.getenv("OPENAI_API_KEY")
    )

    parser.add_argument("--prompt-task-examples", type=str, default="default")
    parser.add_argument(
        "--instruction",
        type=str,
        default="08_heat_the_potato_using_microwave_and_make_a_coffee_and_wash_all_fork_and_spoon.json",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="assets/results/states100/tasks_4_constraints_3/01_boil_potato_and_set_the_table/FloorPlan1/progprompt/prog_logs_01_boil_potato_and_set_the_table.txt",
        help="Path to the log file for this specific run.",
    )
    parser.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="The attempt number for the run.",
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
        default=900,
        help="베이지안 추정을 위한 초기 분산값 (기본값: constants.py 값)",
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
        "--task-folder-name",
        type=str,
        default=None,
        help="Task folder name for organizing results",
    )

    return parser.parse_args()


def generate_plan(
    controller: Optional[Controller],
    task: str,
    args: argparse.Namespace,
    logger: logging.Logger,
    action_interface: Optional[Action],
) -> None:
    """Generate, save, and simulate a plan for a given task.

    The prompt includes available actions, visible objects, few-shot examples,
    and explicit temporal logic guidelines to ensure correct sequencing, wait
    time inference, and immediate follow-ups when needed.

    Args:
        controller (Controller): Active AI2-THOR controller instance.
        task (str): Natural language task/instruction to accomplish.
        args (argparse.Namespace): Parsed CLI arguments controlling generation and simulation.
        logger (logging.Logger): Logger for module-level logging.

    Returns:
        None: This function persists artifacts (plan/logs) and triggers simulation.
    """
    # 현재 scene에 있는 object들을 가져오고, prompt와 example_task_path를 설정
    if args.simulation:
        # 1. 시뮬레이션 환경의 object들을 가져옴
        obj = list(
            set(
                obj["objectType"] for obj in controller.step("Pass").metadata["objects"]
            )
        )
        # 2. iTHOR에서 사용 가능한 action들
        prompt = "from actions import walk <obj>, pickup <obj>, put <obj> <obj>, drop <obj>, open <obj>, close <obj>, toggle_on <obj>, toggle_off <obj>, slice <obj>, wait <duration>"
        # 3. 예제 태스크 경로
        example_task_path = os.path.join(current_dir, "example_task.json")

    elif args.ros:
        # 1. ROS 환경의 object들을 가져옴
        with open(
            "assets/scene_knowledge/real_world/object_init_positions/FloorPlan301_physics.json",
            "r",
        ) as f:
            physics_data = json.load(f)
            obj = list(physics_data.keys())
        # 2. ROS에서 사용 가능한 action들
        prompt = "from actions import TOGGLE_ON <obj>, TOGGLE_OFF <obj>, GRASP <obj>, PLACE_ON_TOP <obj>, PLACE_INSIDE <obj>, WAIT <duration>, MONITORING <obj>, NAVIGATE_TO <obj>"
        # 3. 예제 태스크 경로
        example_task_path = os.path.join(current_dir, "example_task_real.json")

    # 현재 scene에 있는 objects를 prompt에 추가
    prompt += f"\nobjects(name) = {obj}\n\n"

    # Temporal logic guidelines 주입
    wait_units = int(args.init_prior_mean) if args.init_prior_mean is not None else 60
    prompt += "\n" + build_temporal_logic_guidelines(wait_units) + "\n"
    with open(example_task_path, "r") as f:
        tmp = json.load(f)
        prompt_egs = {}
        for k, v in tmp.items():
            prompt_egs[k] = v

    if args.prompt_task_examples == "default":
        if args.ros:
            default_examples = [
                "cook_chicken",
                "put orange bowl on sink",
                "do_laundry",
            ]
            # ROS 예시는 개수가 적을 수 있으므로 args.prompt_num_examples 조정이 필요할 수도 있음
            # 현재 파일에는 3개만 있으므로 len(default_examples)와 비교하여 안전하게 순회하도록 수정 필요
        else:
            default_examples = [
                "Heat Potato using Microwave and set the table for lunch",
                "use coffee machine to make coffee then pick up the apple",
                "Fill the bathtub with water",
                "wash tomato, potato and egg, and cook egg fry",
                "put tomato and apple in fridge and put book in shelf",
            ]

        num_examples_to_use = min(args.prompt_num_examples, len(default_examples))
        for i in range(num_examples_to_use):
            prompt += (
                "task : "
                + default_examples[i]
                + " \n"
                + prompt_egs[default_examples[i]]
                + "\n\n"
            )

    computation_time_start = time.time()

    # Read task from input
    print(f"Generating plan for: {task}\n")
    curr_prompt = (
        f"{prompt}\ntask : {task}\n"  ## 주어진 정보 + 수행할 task 이어서 prompt 만듦
    )
    logger.info(f"ProgPrompt Plan Generation PROMPT:\n\n{curr_prompt}\n")
    _, text = LM(
        curr_prompt,
        args.gpt_version,
        max_tokens=2048,
        stop=["def"],
        frequency_penalty=0.15,
    )
    # save generated plan
    line = {}
    print(f"Saving generated plan at: {task}.json\n")
    plan_of_task_path = os.path.join(current_dir, f"result/plans_of_{task}.json")
    with open(plan_of_task_path, "w") as f:
        line[task] = text
        json.dump(line, f)

    computation_time = time.time() - computation_time_start

    prog_log_path = os.path.join(current_dir, f"result/prog_logs_{task}.txt")
    os.makedirs(os.path.dirname(prog_log_path), exist_ok=True)
    log_file = open(prog_log_path, "w", buffering=1)
    try:
        approach_name = "prog_ai2thor_simulation"
        # For saving, prefer numeric-prefixed instruction stem when running with --case
        if args.case and args.instruction:
            folder_key = Path(args.instruction).stem
        else:
            folder_key = task

        result_path = f"{folder_key}"

        # Execute simulation; ensure any internal buffers are flushed to log_file
        simulate_execution(
            controller, [task], [text], log_file, args, logger, action_interface
        )
        log_file.flush()

        result_args = {
            "approach_name": approach_name,
            "user_input": folder_key,
            "result": prog_log_path,
            "json_output_path": result_path,
            "computation_time": computation_time,
            "scene_name": args.scene,
            "attempt": args.attempt,
            "init_prior_mean": args.init_prior_mean,
            "case_name": args.case,
        }
        result_save_llm(**result_args)
        save_scene_state(
            controller=controller,
            output_path=base_result_path / f"states{int(args.init_prior_mean)}",
            case_name=args.case,
            scene_name=args.scene,
            instruction=folder_key,
            approach_name="progprompt",
            state_label="end",
        )
    finally:
        try:
            log_file.flush()
        except Exception:
            pass
        try:
            log_file.close()
        except Exception:
            pass


if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()

    # Handle INIT_PRIOR_MEAN override
    if args.init_prior_mean is not None:
        set_init_prior_mean(args.init_prior_mean)

    if args.task_folder_name:
        dynamic_task_path = constants.ASSETS_PATH / "tasks" / args.task_folder_name
        constants.set_task_path(dynamic_task_path)
        base_result_path = constants.RESULT_PATH / args.task_folder_name
    else:
        base_result_path = constants.RESULT_PATH

    logger = create_module_logger(
        module_name="prog_ai2thor",
        log_file_path=Path(args.log_path) if args.log_path else None,
        level=args.log_level,
    )

    instruction = args.instruction
    task = ""
    if args.case:
        # run_all에서 case와 함께 전달되는 instruction은 파일명일 수 있으므로 자연어로 정규화
        if instruction is None:
            print("Error: --case가 지정되면 --instruction도 필요합니다.")
            sys.exit(1)
        stem = Path(instruction).stem
        m = re.match(r"\d+_(.*)", stem)
        task = m.group(1) if m else stem
    else:
        if instruction:
            try:
                choice = int(instruction)
                task_files = list_task_files(args.scene)
                if 1 <= choice <= len(task_files):
                    task = Path(task_files[choice - 1]).stem
                else:
                    print(
                        f"Error: Invalid number. Please choose a number between 1 and {len(task_files)}."
                    )
                    sys.exit(1)
            except ValueError:
                # It's a natural language instruction, not a number
                task = instruction.strip()
        else:
            print("명령어가 인자로 제공되지 않았습니다. 사용자 입력을 기다립니다...")
            task = input().strip()

    openai.api_key = args.openai_api_key

    scene_name = args.scene
    platform_obj = None
    if args.cloud_rendering:
        platform_obj = CloudRendering
    if args.simulation:
        controller = init_ai2thor_controller(scene_name, platform=platform_obj)
    elif args.ros:
        controller = None
    # Use a consistent directory name for state saving
    instruction_dir_name = (
        Path(instruction).stem
        if (args.case and instruction)
        else (task if instruction else task)
    )
    trajectory_path = (
        base_result_path
        / f"states{int(args.init_prior_mean)}/{args.case}/{instruction_dir_name}/{scene_name}/progprompt/trajectory_log.json"
    )
    if trajectory_path.exists():
        trajectory_path.unlink()
    save_scene_state(
        controller=controller,
        output_path=base_result_path / f"states{int(args.init_prior_mean)}",
        case_name=args.case,
        scene_name=scene_name,
        instruction=instruction_dir_name,
        approach_name="progprompt",
        state_label="init",
    )
    task_successful = True
    try:
        if args.simulation:
            action_interface = Action(
                controller,
                logger=logger,
                trajectory_log_json_path=(
                    base_result_path
                    / f"states{int(args.init_prior_mean)}/{args.case}/{instruction_dir_name}/{scene_name}/progprompt/trajectory_log.json"
                ),
            )
        elif args.ros:
            action_interface = RosExecutor(
                trajectory_log_path=(
                    base_result_path
                    / f"states{int(args.init_prior_mean)}/{args.case}/{instruction_dir_name}/{scene_name}/progprompt/trajectory_log.json"
                )
            )
        generate_plan(controller, task, args, logger, action_interface)
    except ProgPromptActionFailedError as e:
        logger.error(f"Task failed due to an explicit action failure: {e}")
        task_successful = False
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        task_successful = False
    finally:
        try:
            if args.simulation:
                controller.stop()

        except Exception as e:
            logger.error(f"Error stopping controller: {e}")

        # Force garbage collection to free memory
        gc.collect()

        if not task_successful:
            logger.info("Exiting with status 1 (failure).")
            sys.exit(1)
        else:
            logger.info("Exiting with status 0 (success).")
            sys.exit(0)
