import argparse
import gc
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ai2thor.platform import CloudRendering

from ithor.handlers.action import Action
from ithor.utils.math_utils import load_navigation_graph
from simulation.runner_ai2thor import execute_subtask, init_ai2thor_controller
from src.core import Agent, Scheduler
from src.core.monitoring import (
    create_ground_truth_store,
    create_monitoring_backend,
    create_observation_model,
)
from src.scheduler import ActionHandler, ConstraintHandler, HeuristicManager
from src.utils.get_state import save_scene_state
from src.utils.ros_executor import RosExecutor
from utils.common.logger import create_module_logger
from utils.config import LOG_ROUND
from utils.io_utils import (
    get_user_task_choice,
    list_task_files,
    load_task_data_from_file,
    result_save,
)
from utils.io_utils.task_io import (
    get_user_scene_choice,
    load_scene_positions,
    load_task_data_from_sampled_set,
)
from utils.task import TaskUtil

log = create_module_logger(__name__, module_log=True)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Task Scheduler")

    parser.add_argument(
        "-r",
        "--reset",
        default=True,
        action="store_true",
        help="(Deprecated, no-op) Kept for backward compatibility with run_all.py.",
    )
    parser.add_argument(
        "--ablation-name",
        type=str,
        default=None,
        help="The name of the ablation configuration.",
    )
    parser.add_argument(
        "--init_prior_mean",
        type=float,
        default=100,
        help="베이지안 추정을 위한 초기 평균값 (기본값: constants.py 값)",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="tasks_3_constraints_2",
        help="The name of the case.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="08_heat_the_potato_using_microwave_and_make_a_coffee_and_wash_all_fork_and_spoon.json",
        help="실행할 태스크 instruction 문자열 또는 번호 (default: None)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="FloorPlan13",
        help="input scene name (default: FloorPlan1)",
    )

    parser.add_argument(
        "--simulation",
        default=True,
        action="store_true",
        help="Simulation 모드 사용 여부 (default: False)",
    )
    parser.add_argument(
        "--ros",
        default=False,
        action="store_true",
        help="ROS 통신 사용 여부 (default: False)",
    )
    parser.add_argument(
        "--cloud-rendering",
        default=False,
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
        "--init_prior_variance",
        type=float,
        default=900,
        help="베이지안 추정을 위한 초기 분산값 (기본값: constants.py 값)",
    )
    parser.add_argument(
        "--factor_alpha",
        type=float,
        default=None,
        help="Factor alpha 값 (기본값: constants.py 값)",
    )
    parser.add_argument(
        "--bayesian-threshold-probability",
        type=float,
        default=None,
        help="Monitoring risk tolerance eta 값 (기본값: constants.py 값)",
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=10,
        help="Scheduler beam width (기본값: constants.py 값)",
    )
    parser.add_argument(
        "--beam_depth",
        type=int,
        default=10,
        help="Scheduler beam depth (simulation_depth) (기본값: constants.py 값)",
    )
    parser.add_argument(
        "--disable_monitoring",
        action="store_true",
        help="Disable Bayesian monitoring.",
    )
    parser.add_argument(
        "--belief-update-method",
        type=str,
        default="bayesian",
        choices=["bayesian", "particle_filter"],
        help="Monitoring timing/update backend to use.",
    )
    parser.add_argument(
        "--trajectory_log_path",
        type=str,
        default=None,
        help="Path to save per-action trajectory logs for downstream evaluation.",
    )
    parser.add_argument(
        "--task-folder-name",
        type=str,
        default="sampled_10_instruction_set_for_final_experiment_set_b",
        help="Task folder name for organizing results",
    )
    parser.add_argument(
        "--gt-distribution",
        type=str,
        default="constant",
        choices=["constant", "gaussian", "lognormal", "gamma", "mixture"],
        help="Ground-truth duration distribution used during monitoring updates.",
    )
    parser.add_argument(
        "--gt-seed",
        type=int,
        default=42,
        help="Random seed used for runtime ground-truth sampling.",
    )
    parser.add_argument(
        "--observation-mode",
        type=str,
        default="auto",
        choices=["auto", "synthetic_gaussian", "openai_vlm"],
        help="Observation backend used for monitoring updates.",
    )
    parser.add_argument(
        "--observation-image-path",
        type=str,
        default=None,
        help="Optional image path used by real-world observation backends.",
    )
    parser.add_argument(
        "--observation-openai-model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI multimodal model used for VLM observations.",
    )
    parser.add_argument(
        "--observation-openai-api-key",
        type=str,
        default=None,
        help="Optional OpenAI API key override for VLM observations.",
    )
    return parser.parse_args()


def _sanitize_filename(value: str) -> str:
    """Return a filesystem-safe slug."""

    if not value:
        return "task"
    slug = re.sub(r"[^0-9a-zA-Z_\-]+", "_", value)
    slug = slug.strip("_")
    return slug or "task"


def _resolve_observation_mode(args: argparse.Namespace) -> str:
    """Resolve the runtime observation backend from CLI arguments."""

    if args.observation_mode != "auto":
        return args.observation_mode
    if args.ros:
        return "openai_vlm"
    return "synthetic_gaussian"


def _build_observation_image_provider(
    *,
    controller: Any,
    observation_image_path: Optional[str],
) -> Optional[Any]:
    """Create a callback that returns the latest image payload."""

    if observation_image_path:
        image_path = Path(observation_image_path)

        def _load_image_from_path() -> Path:
            return image_path

        return _load_image_from_path

    if controller is not None and getattr(controller, "last_event", None) is not None:

        def _load_controller_frame() -> Any:
            return getattr(controller.last_event, "frame", None)

        return _load_controller_frame

    return None


def main() -> None:
    """Main entry point for the Task Scheduler."""
    args = parse_arguments()
    base_approach_name = f"dag_{args.belief_update_method}"
    approach_name = base_approach_name

    # Dynamically override constants based on command-line arguments
    from src.utils.config import constants

    # Define base result path based on task folder name
    base_result_path = constants.RESULT_PATH
    if args.task_folder_name:
        # [Added] Override TASK_PATH dynamically based on the argument
        # This ensures we load tasks from the specified folder, not the default constant.
        dynamic_task_path = constants.ASSETS_PATH / "tasks" / args.task_folder_name
        constants.set_task_path(dynamic_task_path)

        base_result_path = constants.RESULT_PATH / args.task_folder_name
        base_result_path.mkdir(parents=True, exist_ok=True)

    init_prior_mean = (
        args.init_prior_mean
        if args.init_prior_mean is not None
        else constants.INIT_PRIOR_MEAN
    )
    init_prior_variance = (
        args.init_prior_variance
        if args.init_prior_variance is not None
        else constants.INIT_PRIOR_VARIANCE
    )

    if args.init_prior_mean is not None:
        constants.set_init_prior_mean(args.init_prior_mean)
    else:
        constants.set_init_prior_mean(init_prior_mean)
    if args.init_prior_variance is not None:
        constants.set_init_prior_variance(args.init_prior_variance)
    else:
        constants.set_init_prior_variance(init_prior_variance)
    if args.factor_alpha is not None:
        constants.set_factor_alpha(args.factor_alpha)
    if args.bayesian_threshold_probability is not None:
        constants.set_bayesian_threshold_probability(
            args.bayesian_threshold_probability
        )
    if args.beam_width is not None:
        constants.set_beam_width(args.beam_width)
    if args.beam_depth is not None:
        constants.set_simulation_depth(args.beam_depth)
    if args.disable_monitoring:
        constants.set_monitoring_enabled(False)

    logger = create_module_logger(
        module_name=approach_name,
        log_file_path=Path(args.log_path) if args.log_path else None,
        level=logging.ERROR,
    )
    scene_name = args.scene
    controller = None
    ros_executor = None  # ros_executor를 None으로 미리 초기화

    platform_obj = None  # Set up the AI2-THOR controller and navigation graph

    if args.cloud_rendering:
        platform_obj = CloudRendering

    try:
        if scene_name is None:
            scene_data = get_user_scene_choice()
            scene_name = scene_data.file_name.split("_")[0]

        if args.ros:
            controller = None
            nav_graph = {(0, 0, 0): {(0, 0, 0)}}
            action_handler = ActionHandler(nav_graph, real_world_mode=True)
        else:
            controller = init_ai2thor_controller(scene_name, platform=platform_obj)
            nav_graph = load_navigation_graph(controller)
            action_handler = ActionHandler(nav_graph, real_world_mode=False)

        trajectory_log_path: Optional[Path] = (
            Path(args.trajectory_log_path).expanduser()
            if args.trajectory_log_path
            else None
        )

        action_interface: Optional[Action] = None
        task_files = list_task_files(scene_name=scene_name)

        if args.case:
            # Load task data
            input_natural_language = re.match(r"\d+_(.*)", args.instruction).group(1)
            task_data = load_task_data_from_sampled_set(
                args.case, scene_name, args.instruction
            )
            approach_name_final = (
                f"{approach_name}_{args.ablation_name}"
                if args.ablation_name
                else approach_name
            )
            save_scene_state(
                controller=controller,
                output_path=base_result_path / f"states{int(init_prior_mean)}",
                case_name=args.case,
                scene_name=scene_name,
                instruction=args.instruction.split(".json")[0],
                approach_name=f"{approach_name_final}",
                state_label="init",
            )
            if trajectory_log_path is None:
                instr_stem = Path(args.instruction).stem
                approach_suffix = (
                    f"{approach_name}_{args.ablation_name}"
                    if args.ablation_name
                    else approach_name
                )
                trajectory_log_path = (
                    base_result_path
                    / f"states{int(init_prior_mean)}"
                    / args.case
                    / instr_stem
                    / scene_name
                    / approach_suffix
                    / "trajectory_log.json"
                )

            trajectory_log_path.parent.mkdir(parents=True, exist_ok=True)
            if trajectory_log_path.exists():
                trajectory_log_path.unlink()
            action_interface = Action(
                controller,
                logger=logger,
                trajectory_log_json_path=trajectory_log_path,
            )

        elif args.instruction:
            # Load the chosen task data
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
                output_path=base_result_path / f"states{int(init_prior_mean)}",
                case_name=args.case,
                scene_name=scene_name,
                instruction=input_natural_language,
                approach_name=approach_name,
                state_label="init",
            )
            if task_data is None:
                # It was a natural language instruction or an invalid number choice.
                # In both cases, we treat it as a natural language instruction.
                task_data = {"instruction": instruction}

            if trajectory_log_path is None:
                instruction_slug = _sanitize_filename(input_natural_language)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                trajectory_log_path = (
                    Path("assets/results/tune")
                    / instruction_slug
                    / f"{timestamp}_trajectory_log.json"
                )
            trajectory_log_path.parent.mkdir(parents=True, exist_ok=True)
            if trajectory_log_path.exists():
                trajectory_log_path.unlink()
            action_interface = Action(
                controller,
                logger=logger,
                trajectory_log_json_path=trajectory_log_path,
            )

        else:
            task_file_name, choice = get_user_task_choice(task_files)
            task_data = load_task_data_from_file(task_file_name)
            input_natural_language = task_file_name
            if choice != 0:
                input_natural_language = task_file_name
            if trajectory_log_path is None:
                instruction_slug = _sanitize_filename(Path(task_file_name).stem)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                trajectory_log_path = (
                    Path("assets/results/tune")
                    / instruction_slug
                    / f"{timestamp}_trajectory_log.json"
                )
            trajectory_log_path.parent.mkdir(parents=True, exist_ok=True)
            if trajectory_log_path.exists():
                trajectory_log_path.unlink()
            action_interface = Action(
                controller,
                logger=logger,
                trajectory_log_json_path=trajectory_log_path,
            )

        if action_interface is None:
            raise RuntimeError(
                "Failed to initialize action interface for simulation run."
            )

        # Build tasks and constraints
        # subtasks, constraints = TaskUtil.build_tasks_and_constraints(
        #     task_data, scene_file_name=scene_data.file_name,
        # )

        subtasks, constraints, bayesian_load = TaskUtil.build_tasks_and_constraints(
            task_data,
            scene_file_name=f"{scene_name}_physics_environment.json",
        )

        # Initialize the agent and scheduler
        constraint_handler = ConstraintHandler(action_handler)
        observation_mode = _resolve_observation_mode(args)
        observation_image_provider = _build_observation_image_provider(
            controller=controller,
            observation_image_path=args.observation_image_path,
        )
        observation_model = create_observation_model(
            observation_mode,
            image_provider=observation_image_provider,
            openai_model_name=args.observation_openai_model,
            openai_api_key=args.observation_openai_api_key,
            random_seed=args.gt_seed,
        )
        belief_store, monitoring_policy, belief_updater = create_monitoring_backend(
            args.belief_update_method,
            bayesian_load,
            particle_distribution="gaussian",
            observation_model=observation_model,
        )
        ground_truth_store = create_ground_truth_store(
            constants.CRITICAL_OBJECT_GROUND_TRUTH,
            distribution=args.gt_distribution,
            random_seed=args.gt_seed,
        )
        ground_truth_store.ensure_intervals(bayesian_load)
        agent = Agent(
            constraint_handler,
            bayesian_load,
            belief_updater=belief_updater,
            belief_store=belief_store,
            ground_truth_store=ground_truth_store,
            real_world_mode=args.ros,
        )
        cost_calculator = HeuristicManager(
            action_handler,
            real_world_mode=args.ros,
        )
        scheduler = Scheduler(
            action_handler=action_handler,
            constraint_handler=constraint_handler,
            heuristic_manager=cost_calculator,
            monitoring_policy=monitoring_policy,
            beam_width=constants.BEAM_WIDTH,
            simulation_depth=constants.SIMULATION_DEPTH,
            real_world_mode=args.ros,
        )
        scene_poses: Dict[str, Any] = load_scene_positions(
            f"{scene_name}_positions.json"
        )
        # current_state = TaskUtil.get_init_state(
        #     subtasks, constraints, scene_data.object_positions
        # )
        current_state = TaskUtil.get_init_state(subtasks, constraints, scene_poses)

        is_end = False

        total_compute_time, total_sim_time = 0, 0

        ros_executor = (
            RosExecutor(trajectory_log_path=trajectory_log_path) if args.ros else None
        )

        while not is_end:

            next_state, computation_elapsed_time = scheduler.get_next_state(
                current_state
            )
            total_compute_time += computation_elapsed_time

            if next_state is None:
                logger.error("No feasible solution found.")
                break

            if args.simulation:
                sim_elapsed_time, execution_status, sim_nav_time = execute_subtask(
                    controller, next_state.subtask, logger, action_interface
                )
                # 시뮬레이션에서 흐른 시간과 실행 상태를 저장.
                last_entry = next_state.completed_entries[-1]
                last_entry.sim_start_time = total_sim_time
                last_entry.sim_end_time = total_sim_time + sim_elapsed_time
                last_entry.execution_status = execution_status
                last_entry.sim_nav_time = sim_nav_time
                total_sim_time += sim_elapsed_time
                intervallist = []
                # 모니터 끄려면 이 안쪽을 주석화.
                if (
                    not args.disable_monitoring
                    and next_state.subtask.subtask_type == "Monitor"
                ):
                    next_state, monitored_subtask = agent.update_monitoring_belief(
                        next_state
                    )
                    next_state.completed_entries[-1].monitored_subtask = (
                        monitored_subtask
                    )
                current_state = next_state
                if not current_state.remaining_subtasks:
                    is_end = True
                intervallist = []
                for name1 in current_state.constraints.in_edges._adjdict.keys():
                    for name2 in current_state.constraints.in_edges._adjdict[
                        name1
                    ].keys():
                        if current_state.constraints.in_edges._adjdict[name1][name2][
                            "info"
                        ]["IsCritical"]:
                            intervallist.append(
                                {
                                    name2: current_state.constraints.in_edges._adjdict[
                                        name1
                                    ][name2]["info"]["Interval"]
                                }
                            )
                last_entry = current_state.completed_entries[-1]
                if last_entry.subtask.name != "Init":
                    logger.info(
                        f"{last_entry.subtask.name} ({round(last_entry.sim_start_time, LOG_ROUND)} ~ {round(last_entry.sim_end_time,LOG_ROUND)})"
                    )
                    logger.info(
                        f"Primitive actions: {last_entry.subtask.execution.primitive_actions}\n"
                    )
                    last_entry.start_time_scheduled = round(
                        last_entry.sim_start_time, LOG_ROUND
                    )
                    last_entry.end_time_scheduled = round(
                        last_entry.sim_end_time, LOG_ROUND
                    )

            if args.ros and ros_executor:
                ros_start_offset = ros_executor.total_ros_time
                success, elapsed_time, action_logs = ros_executor.execute_subtask(
                    next_state.subtask
                )

                last_entry = next_state.completed_entries[-1]
                last_entry.sim_start_time = ros_start_offset
                last_entry.sim_end_time = ros_start_offset + elapsed_time
                last_entry.execution_status = success
                last_entry.primitive_action_log = action_logs
                last_entry.schedule_start_time = last_entry.sim_start_time
                last_entry.schedule_end_time = last_entry.sim_end_time
                next_state = next_state._replace(
                    current_time=last_entry.sim_end_time
                )

                if not success:
                    break

                if (
                    not args.disable_monitoring
                    and next_state.subtask.subtask_type == "Monitor"
                ):
                    next_state, monitored_subtask = agent.update_monitoring_belief(
                        next_state
                    )
                    next_state.completed_entries[-1].monitored_subtask = (
                        monitored_subtask
                    )

                current_state = next_state
                if not current_state.remaining_subtasks:
                    is_end = True
    finally:
        if ros_executor and args.ros:
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

        if args.ros:
            result_schedule = [
                entry
                for entry in current_state.completed_entries
                if entry.subtask.name != "Init"
            ]
            approach_name = f"{approach_name}_ros"
            result_args = {
                "task_name": input_natural_language,
                "approach_name": approach_name,
                "result_schedule": result_schedule,
                "computation_time": total_compute_time,
                "scene_name": scene_name,
                "constraints": current_state.constraints,
                "initial_plan_data": task_data,
                "init_prior_mean": args.init_prior_mean,
                "ground_truth_overrides": ground_truth_store.as_dict(),
            }
            result_save(
                **result_args,
                base_result_path=base_result_path,
            )

    if args.simulation:
        result_schedule = [
            entry
            for entry in current_state.completed_entries
            if entry.subtask.name != "Init"
        ]

        approach_name = f"{base_approach_name}_simulation"
        result_args = {
            "task_name": input_natural_language,
            "approach_name": approach_name,
            "result_schedule": result_schedule,
            "computation_time": total_compute_time,
            "scene_name": scene_name,
            "constraints": current_state.constraints,
            "initial_plan_data": task_data,
            "init_prior_mean": init_prior_mean,
            "ground_truth_overrides": ground_truth_store.as_dict(),
            # "simulationTime": total_sim_time,
        }
        if args.case:
            approach_name = base_approach_name
            meta_data = {
                "init_prior_name": constants.INIT_PRIOR_MEAN,
                "init_prior_variance": constants.INIT_PRIOR_VARIANCE,
                "factor_alpha": constants.FACTOR_ALPHA,
                "bayesian_threshold_probability": (
                    constants.BAYESIAN_THRESHOLD_PROBABILITY
                ),
                "beam_width": constants.BEAM_WIDTH,
                "beam_depth": constants.SIMULATION_DEPTH,
                "disable_monitoring": not constants.MONITORING_ENABLED,
                "belief_update_method": args.belief_update_method,
                "gt_distribution": args.gt_distribution,
                "gt_seed": args.gt_seed,
                "pf_prior_distribution": "gaussian",
                "observation_mode": observation_mode,
                "observation_openai_model": (
                    args.observation_openai_model
                    if observation_mode == "openai_vlm"
                    else None
                ),
                "sampled_ground_truths": ground_truth_store.as_dict(),
            }
            result_args.update(
                {
                    "task_name": args.instruction.split(".json")[0],
                    "approach_name": f"{approach_name}_{args.ablation_name}",
                    "case_name": args.case,
                    "dag_bayesian_meta_data": meta_data,
                }
            )
        approach_name_final = (
            f"{approach_name}_{args.ablation_name}"
            if args.ablation_name
            else approach_name
        )
        save_scene_state(
            controller=controller,
            output_path=base_result_path / f"states{int(init_prior_mean)}",
            case_name=args.case,
            scene_name=scene_name,
            instruction=args.instruction.split(".json")[0],
            approach_name=f"{approach_name_final}",
            state_label="end",
        )
        result_save(
            **result_args,
            base_result_path=base_result_path,
        )


if __name__ == "__main__":
    main()
