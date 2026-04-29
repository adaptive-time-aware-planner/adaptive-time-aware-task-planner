import argparse
import gc
import heapq
import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import networkx as nx
from ai2thor.platform import CloudRendering
from dataclass import ActionResult, CompletedEntry, SchedulerState, SimulationNode

from ithor.handlers.action import Action
from ithor.utils.math_utils import load_navigation_graph
from src.models.task import *
from src.scheduler.action_handler import ActionHandler
from src.simulation.runner_ai2thor import execute_subtask, init_ai2thor_controller
from src.utils.common.logger import create_module_logger
from src.utils.config.constants import RESULT_PATH
from src.utils.get_state import save_scene_state
from src.utils.io_utils import task_io
from src.utils.io_utils.result_saver import result_save
from src.utils.io_utils.task_io import (
    get_natural_language_from_task_file,
    get_user_task_choice,
    list_task_files,
    load_scene_positions,
    load_task_data_from_file,
    load_task_data_from_sampled_set,
)
from src.utils.ros_executor import RosExecutor
from utils.task.task_util import TaskUtil


def is_executable(subtask: Subtask, current_state: SchedulerState) -> bool:
    """
    subtask가 dependency 때문에 실행 가능한지 여부 반환
    """
    constraints = current_state.constraints
    incoming = list(constraints.in_edges(subtask.name))
    if incoming:  # dependency가 있는 경우
        # 단순히 첫번째 dependency가 완료되었는지 여부로 판단
        pred_name = incoming[0][0]
        completed = {entry.subtask.name for entry in current_state.completed_entries}
        return pred_name in completed
    return True  # dependency가 없으면 실행 가능


def compute_nav_time(
    subtask: Subtask, current_state: SchedulerState, action_handler: ActionHandler
) -> float:
    """
    subtask의 첫 번째 primitive action이 NAVIGATE_TO일 경우, 해당 액션의 소요 시간을 계산합니다.

    Args:
        subtask: 현재 실행할 Subtask 객체
        current_state: 현재 SchedulerState 객체
        action_handler: 액션 실행 관련 정보를 제공하는 ActionHandler 객체

    Returns:
        NAVIGATE_TO 액션의 소요 시간 (없으면 0)

    Raises:
        ValueError: subtask의 첫 번째 액션이 NAVIGATE_TO가 아닐 경우
    """
    nav_time = 0.0

    if not subtask.execution or not subtask.execution.primitive_actions:
        return nav_time

    first_action = subtask.execution.primitive_actions[0]
    if not first_action.startswith("NAVIGATE_TO"):
        raise ValueError(f"[{subtask.name}] 첫 번째 액션이 NAVIGATE_TO가 아닙니다.")

    # SimulationNode를 생성하여 해당 액션만 시뮬레이션
    temp_node = SimulationNode(
        deadline=current_state.current_time,
        simulation_subtask=subtask,
        state=current_state,
        execution_time=0.0,
    )
    nav_info = action_handler.get_actions_info(temp_node, [first_action])
    if nav_info:
        nav_time = nav_info.cumulative_time
        nav_positions = nav_info.scene_positions

    return nav_time, nav_positions


def offline_subtask_execution(
    subtask: Subtask, current_state: SchedulerState, action_handler: ActionHandler
) -> float:
    """
    현재 subtask의 실행 시간을 시뮬레이션하여 반환
    """
    temp_node = SimulationNode(
        deadline=current_state.current_time,
        simulation_subtask=subtask,
        state=current_state,
        execution_time=0.0,
    )
    actions = subtask.execution.primitive_actions or []
    exec_info = action_handler.get_actions_info(temp_node, actions) if actions else None

    # exec_info를 추출하는데 실패할경우 None을 return
    return exec_info if exec_info else None


def update_state(
    current_state: SchedulerState,
    next_subtask: Subtask,
    exec_info: ActionResult,
    nav_time: Optional[float] = None,
) -> SchedulerState:
    subtask_duration = exec_info.cumulative_time
    subtask_entry = CompletedEntry(
        subtask=next_subtask,
        schedule_start_time=current_state.current_time,
        schedule_end_time=current_state.current_time + subtask_duration,
        schedule_nav_time=nav_time,
    )
    new_completed = current_state.completed_entries + [subtask_entry]
    new_remaining = [
        st for st in current_state.remaining_subtasks if st.name != next_subtask.name
    ]
    next_state = SchedulerState(
        subtask=next_subtask,
        completed_entries=new_completed,
        remaining_subtasks=new_remaining,
        constraints=current_state.constraints,
        current_time=current_state.current_time + subtask_duration,
        scene_positions=exec_info.scene_positions,
        held_object=exec_info.held_object,
        agent_location=current_state.agent_location,
    )
    return next_state


def nav_and_wait_during_interval(
    current_state: SchedulerState,
    interval: float,
    next_subtask: Subtask,
    is_critical: bool,
    action_handler: ActionHandler,
) -> Tuple[SchedulerState, bool]:
    """
    주어진 시간(interval) 동안 이동(NAVIGATE)과 대기(WAIT)를 위한 서브태스크를 생성합니다.
    next_subtask의 첫 번째 액션이 NAVIGATE_TO여야 하며,
    네비게이션 소요 시간이 interval보다 작을 경우에만,
    네비게이션 후 남은 시간만큼 대기하는 WAIT 서브태스크를 생성합니다.
    Args:
        current_state: 현재 스케줄러 상태 (시간, 위치 등 포함)
        interval: 현재 간선에 주어진 시간 간격
        next_subtask: 다음에 실행할 서브태스크. 첫번째 액션은 반드시 NAVIGATE_TO여야 함.
        is_critical: 해당 간선이 critical인지 여부
        action_handler: 액션 실행을 위한 핸들러
    Returns:
        - 생성된 CompletedEntry들의 리스트 (NAVIGATE, WAIT 서브태스크)
        - 업데이트된 SchedulerState (현재 시간 및 위치 갱신)
    Raises:
        ValueError: next_subtask의 첫 번째 액션이 NAVIGATE_TO가 아닐 경우
    """
    entries: List[CompletedEntry] = []
    current_time = current_state.current_time

    # Get the first NAVIGATE_TO action from the next subtask
    first_action = next_subtask.execution.primitive_actions[0]
    if not first_action.startswith("NAVIGATE_TO"):
        raise ValueError(
            f"[{next_subtask.name}] 첫 번째 액션이 NAVIGATE_TO가 아닙니다."
        )
    # Calculate navigation time
    nav_time, nav_positions = compute_nav_time(
        next_subtask, current_state, action_handler
    )
    # Only proceed if interval is greater than navigation time
    if 0.0 < nav_time <= interval:
        # Create NAVIGATE subtask
        nav_subtask = Subtask(
            task_name=next_subtask.task_name,
            name=f"NAVIGATE_TO_{first_action.split()[1]}",
            repetition=1,
            subtask_type="NAVIGATE",
            execution=Execution(objects={}, primitive_actions=[first_action]),
            duration=Duration(type="NAVIGATE", interval=nav_time),
            temporal_constraints=[],
        )
        nav_entry = CompletedEntry(
            subtask=nav_subtask,
            schedule_start_time=current_time,
            schedule_end_time=current_time + nav_time,
            schedule_nav_time=nav_time,
        )
        entries.append(nav_entry)
        current_time += nav_time
        # Create WAIT subtask for remaining time
    wait_time = interval - nav_time
    if wait_time >= 0:
        wait_subtask = Subtask(
            task_name=next_subtask.task_name,
            name=f"WAIT {wait_time} to {next_subtask.name}",
            repetition=1,
            subtask_type="WAIT",
            execution=Execution(objects={}, primitive_actions=[f"WAIT {wait_time}"]),
            duration=Duration(type="WAIT", interval=wait_time),
            temporal_constraints=[],
        )
        wait_entry = CompletedEntry(
            subtask=wait_subtask,
            schedule_start_time=current_time,
            schedule_end_time=current_time + wait_time,
        )
        entries.append(wait_entry)
        # Create new state with updated time and positions
        new_state = SchedulerState(
            subtask=current_state.subtask,
            completed_entries=current_state.completed_entries + entries,
            remaining_subtasks=current_state.remaining_subtasks,
            constraints=current_state.constraints,
            current_time=current_time + wait_time,
            scene_positions=nav_positions,
            held_object=current_state.held_object,
            agent_location=current_state.agent_location,
        )
        return new_state, True
    # If interval < nav_time, unchanged state
    return current_state, False


def compute_deadline_for_subtask(
    subtask: Subtask,
    current_state: SchedulerState,
    nav_time: float,
    execution_time: float,
) -> float:
    """
    subtask에 대한 deadline을 timeslot 유형과 dependency 조건에 따라 계산한다.

    [규칙]
        1. 기본규칙
            deadline = current_time + execution_time(이동시간 포함)
        2. subtask가 non critical in edge만 있으면:
            deadline = max(선행 subtask의 end_time + edge의 interval)
        3. subtask가 critical in edge를 포함한다면:
            deadline = max(critical edge의(선행 subtask의 end_time + edge interval))
    """
    constraints = current_state.constraints
    current_time = current_state.current_time
    incoming_edges = list(constraints.in_edges(subtask.name, data=True))

    # 기본 deadline
    deadline = current_time + execution_time

    if incoming_edges:
        # Critical edges 처리
        critical_edges = [
            (u, v, data)
            for u, v, data in incoming_edges
            if data.get("info", {}).get("IsCritical", False)
        ]

        if critical_edges:
            # 모든 critical edge의 deadline을 계산
            critical_deadlines = []
            for u, v, data in critical_edges:
                pred_end_time = next(
                    (
                        entry.schedule_end_time
                        for entry in current_state.completed_entries
                        if entry.subtask.name == u
                    ),
                    current_time,
                )
                critical_deadlines.append(pred_end_time + data["info"]["Interval"])
            # 가장 긴 deadline을 선택 (모든 critical constraint를 만족해야 함)
            deadline = max(critical_deadlines)
        else:
            # Non-critical edges 처리
            non_critical_deadlines = []
            for u, v, data in incoming_edges:
                pred_end_time = next(
                    (
                        entry.schedule_end_time
                        for entry in current_state.completed_entries
                        if entry.subtask.name == u
                    ),
                    current_time,
                )
                non_critical_deadlines.append(pred_end_time + data["info"]["Interval"])
            # 가장 긴 deadline을 선택
            deadline = max(non_critical_deadlines)

    return deadline


def get_next_subtask_edf(
    current_state: SchedulerState, action_handler: ActionHandler
) -> Optional[Subtask]:
    """
    각 실행 가능한 subtask에 대해 deadline을 산출한 후, deadline이 가장 짧은 subtask를 선택한다.
    """
    import heapq

    queue = []
    for subtask in current_state.remaining_subtasks:
        if not is_executable(subtask, current_state):
            continue

        nav_time, _ = compute_nav_time(subtask, current_state, action_handler)
        exec_info = offline_subtask_execution(subtask, current_state, action_handler)
        execution_time = exec_info.cumulative_time
        deadline = compute_deadline_for_subtask(
            subtask, current_state, nav_time, execution_time
        )
        sim_node = SimulationNode(
            deadline=deadline,
            simulation_subtask=subtask,
            state=current_state,
            execution_time=execution_time,
        )
        heapq.heappush(queue, sim_node)
        # remaining_subtask들의 deadline 정보를 삽입
        subtask.deadline = deadline

    if queue:
        chosen_node = heapq.heappop(queue)
        chosen_node.simulation_subtask.duration.interval = chosen_node.execution_time
        return chosen_node.simulation_subtask

    return None


def update(
    current_state: SchedulerState, next_subtask: Subtask, action_handler: ActionHandler
) -> SchedulerState:
    """
    1) next_subtask 실행 시간 시뮬레이션 후 subtask.duration.interval 에 반영
    2) After 제약(interval=X)이 있으면, designated_start = predecessor.end_time + X
        - current_time < designated_start => NAVIGATE_TO + WAIT subtask를 CompletedEntry로 추가
    3) next_subtask 실행 CompletedEntry 추가
    """
    # -------------------------------
    # 1) 실제 실행 시간 시뮬레이션
    # -------------------------------
    current_time = current_state.current_time
    # -------------------------------
    # 2) After 제약(interval=X) 처리
    # -------------------------------
    constraints = current_state.constraints
    incoming_edges = list(constraints.in_edges(next_subtask.name, data=True))
    designated_start = current_time  # 기본적으로는 지금 바로 시작

    nav_time, _ = compute_nav_time(next_subtask, current_state, action_handler)

    # incoming_edge 가 있으면 deadline 규칙에 따라 연산된 deadline 시간에 subtask 시작
    if incoming_edges:
        designated_start = next_subtask.deadline

        # 만약 current_time < designated_start 라면 => 대기 필요
        if current_time < designated_start:
            # 남은 시간(= designated_start - current_time) 만큼 NAVIGATE + WAIT
            interval = designated_start - current_time
            new_state, is_nav_and_wait = nav_and_wait_during_interval(
                current_state, interval, next_subtask, False, action_handler
            )
            current_state = new_state
            if is_nav_and_wait:
                nav_time = 0.0

    exec_info = offline_subtask_execution(next_subtask, current_state, action_handler)
    # -------------------------------
    # 3) next_subtask 실행
    # -------------------------------
    return update_state(current_state, next_subtask, exec_info, nav_time)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Task Scheduler")

    # 현재 미사용중인 argument들은 주석처리되었다.
    parser.add_argument(
        "-d",
        "--decomposition",
        help="Enable or disable decomposition",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--reset",
        default=True,
        help="Reset the knowledge base to Gaussian",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--simulation",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="tasks_3_constraints_2",
        help="The name of the case.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="로그 출력 수준 설정 (default: ERROR)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="FloorPlan13",
        help="시뮬레이션에 사용할 씬 이름 (default: FloorPlan1)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="08_heat_the_potato_using_microwave_and_make_a_coffee_and_wash_all_fork_and_spoon.json",
        help="실행할 태스크 instruction 문자열 또는 번호 (default: None)",
    )
    parser.add_argument(
        "--ablation-name",
        type=str,
        default=None,
        help="The name of the ablation configuration.",
    )
    parser.add_argument(
        "--ros",
        default=False,
        action="store_true",
        help="ROS 실행 여부 (default: False)",
    )
    parser.add_argument(
        "--cloud-rendering",
        action="store_true",
        help="Use CloudRendering platform for AI2-THOR.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Path to the log file for this specific run.",
    )
    parser.add_argument(
        "--init_prior_mean",
        type=float,
        default=100,
        help="베이지안 추정을 위한 초기 평균값 (기본값: constants.py 값)",
    )
    parser.add_argument(
        "--init_prior_variance",
        type=float,
        default=900,
        help="베이지안 추정을 위한 초기 분산값 (기본값: constants.py 값)",
    )
    parser.add_argument(
        "--task-folder-name",
        type=str,
        default=None,
        help="Task folder name for organizing results",
    )

    return parser.parse_args()


def main():
    # Set up the AI2-THOR controller and navigation graph
    approach_name = "dag_edf"

    args = parse_arguments()

    # Dynamically override constants based on command-line arguments
    from src.utils.config import constants

    base_result_path = constants.RESULT_PATH
    if args.task_folder_name:
        # [Added] Override TASK_PATH dynamically based on the argument
        # This ensures we load tasks from the specified folder, not the default constant.
        dynamic_task_path = constants.ASSETS_PATH / "tasks" / args.task_folder_name
        constants.set_task_path(dynamic_task_path)

        base_result_path = constants.RESULT_PATH / args.task_folder_name
        base_result_path.mkdir(parents=True, exist_ok=True)

    if args.init_prior_mean is not None:
        constants.set_init_prior_mean(args.init_prior_mean)
    if args.init_prior_variance is not None:
        constants.set_init_prior_variance(args.init_prior_variance)

    logger = create_module_logger(
        module_name=approach_name,
        log_file_path=Path(args.log_path) if args.log_path else None,
        level=logging.ERROR,
    )
    scene_name = args.scene
    controller = None

    platform_obj = None
    if args.cloud_rendering:
        platform_obj = CloudRendering

    try:
        if args.ros:
            controller = None
            nav_graph = {(0, 0, 0): {(0, 0, 0)}}
            action_handler = ActionHandler(nav_graph, real_world_mode=True)
        else:
            controller = init_ai2thor_controller(scene_name, platform=platform_obj)
            nav_graph = load_navigation_graph(controller)
            action_handler = ActionHandler(nav_graph, real_world_mode=False)

        scene_poses = load_scene_positions(f"{scene_name}_positions.json")
        if args.case:
            # Load task data
            input_natural_language = re.match(r"\d+_(.*)", args.instruction).group(1)
            task_data = load_task_data_from_sampled_set(
                args.case, scene_name, args.instruction
            )
            action_interface = Action(
                controller,
                logger=logger,
                trajectory_log_json_path=Path(
                    base_result_path
                    / f"states{int(args.init_prior_mean)}/{args.case}/{args.instruction.split('.json')[0]}/{scene_name}/{approach_name}/trajectory_log.json"
                ),
            )

            save_scene_state(
                controller=controller,
                output_path=base_result_path / f"states{int(args.init_prior_mean)}",
                case_name=args.case,
                scene_name=scene_name,
                instruction=args.instruction.split(".json")[0],
                approach_name=approach_name,
                state_label="init",
            )

        elif args.instruction:
            # Load the chosen task data
            task_files = list_task_files(scene_name=scene_name)
            instruction = args.instruction
            input_natural_language = instruction
            task_data = None

            try:
                choice = int(instruction)
                if 1 <= choice <= len(task_files):
                    task_file_name = task_files[choice - 1]
                    task_data = load_task_data_from_file(task_file_name)
                    input_natural_language = Path(task_file_name).stem
            except ValueError:
                # It's a natural language instruction, not a number
                pass
            save_scene_state(
                controller=controller,
                output_path=base_result_path / f"states{int(args.init_prior_mean)}",
                scene_name=scene_name,
                instruction=args.instruction.split(".json")[0],
                approach_name=approach_name,
                state_label="init",
            )
            if task_data is None:
                # It was a natural language instruction or an invalid number choice.
                # In both cases, we treat it as a natural language instruction.
                task_data = {"instruction": instruction}
        else:
            task_file_name, choice = get_user_task_choice(
                task_files, scene_name=scene_name
            )
            task_data = load_task_data_from_file(task_file_name)
            input_natural_language = task_file_name
            if choice != 0:
                input_natural_language = task_file_name
        # Build tasks and constraints
        subtasks, constraints, bayesian_load = TaskUtil.build_tasks_and_constraints(
            task_data, scene_file_name=f"{scene_name}_physics_environment.json"
        )

        computation_time = 0
        init_state = TaskUtil.get_init_state(subtasks, constraints, scene_poses)
        current_state = init_state
        result_schedule = []

        # Phase 1: Complete scheduling
        for i in range(len(subtasks)):
            subtask_scheduling_time_start = time.time()
            next_subtask = get_next_subtask_edf(current_state, action_handler)

            if next_subtask is None:
                break
            current_state = update(current_state, next_subtask, action_handler)
            subtask_scheduling_time = time.time() - subtask_scheduling_time_start
            computation_time += subtask_scheduling_time

        result_schedule = current_state.completed_entries
        result_schedule.pop(0)  # Remove the init entry

        # Print execution times for each entry
        print("\n=== Execution Times for Each Entry ===")
        current_state = init_state
        for entry in result_schedule:

            action_handler = ActionHandler(nav_graph)
            exec_info = offline_subtask_execution(
                entry.subtask, current_state, action_handler
            )
            action_handler = ActionHandler(nav_graph)

            current_state = update_state(
                current_state, entry.subtask, exec_info, entry.schedule_nav_time
            )

        # Phase 2: Execute simulation if requested
        if args.simulation:

            simulation_current_time = 0.0
            # Execute each subtask in the schedule
            for entry in result_schedule:
                subtask_time, execution_status, sim_nav_time = execute_subtask(
                    controller, entry.subtask, logger, action_interface
                )
                # Update the entry with simulation times and execution status
                entry.sim_start_time = simulation_current_time
                entry.sim_end_time = simulation_current_time + subtask_time
                entry.execution_status = execution_status
                entry.sim_nav_time = sim_nav_time
                simulation_current_time += subtask_time

            save_scene_state(
                controller=controller,
                output_path=base_result_path / f"states{int(args.init_prior_mean)}",
                case_name=args.case,
                scene_name=scene_name,
                instruction=args.instruction.split(".json")[0],
                approach_name=approach_name,
                state_label="end",
            )

            approach_name = f"{approach_name}_simulation"
            result_args = {
                "task_name": input_natural_language,
                "approach_name": approach_name,
                "result_schedule": result_schedule,
                "computation_time": computation_time,
                "scene_name": scene_name,
                "constraints": constraints,
                "initial_plan_data": task_data,
                "init_prior_mean": args.init_prior_mean,
            }

            if args.case:
                result_args.update(
                    {
                        "task_name": args.instruction.split(".json")[0],
                        "case_name": args.case,
                    }
                )

            result_save(**result_args, base_result_path=base_result_path)

        if args.ros:
            ros_executor = RosExecutor(
                trajectory_log_path=Path(
                    base_result_path
                    / f"states{int(args.init_prior_mean)}/{args.case}/{args.instruction.split('.json')[0]}/{scene_name}/{approach_name}/trajectory_log.json"
                )
            )
            real_executed_result_schedule = ros_executor.execute_schedule(
                result_schedule
            )

            result_args = {
                "task_name": input_natural_language,
                "approach_name": f"{approach_name}_ros",
                "result_schedule": real_executed_result_schedule,
                "computation_time": computation_time,
                "scene_name": scene_name,
                "constraints": constraints,
                "initial_plan_data": task_data,
                "init_prior_mean": args.init_prior_mean,
            }
            result_save(**result_args, base_result_path=base_result_path)
    finally:
        if controller:
            try:
                controller.stop()
            except Exception as e:
                logger.error(f"Error stopping controller: {e}")
        # Force garbage collection to free memory
        gc.collect()


if __name__ == "__main__":
    main()
