"""
ai2thor_simulation.py

This module provides:
1. A unified controller initialization function (`init_ai2thor_controller`)
   that can create an AI2-THOR Controller with default or overridden parameters.
2. A utility function (`execute_subtask`) to perform a given subtask with primitive actions.

Example usage:
    from ai2thor_simulation import init_ai2thor_controller, execute_subtask

    controller = init_ai2thor_controller()
    elapsed_time, success = execute_subtask(controller, subtask_obj)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ai2thor.controller import Controller

from src.models.dataclass import TaskExecutionStatus

# Action handler import
if TYPE_CHECKING:
    from ithor.handlers.action import Action
    from src.models.task import Subtask


# Constants (unify your constants in one place)
from src.utils.config.constants import (
    DEFAULT_SCENE_NAME,
    GRID_SIZE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)

# # Logging utility
# from src.utils.common.logger import create_module_logger


def init_ai2thor_controller(
    scene: str = DEFAULT_SCENE_NAME,
    platform=None,
    agent_mode: str = "default",
    mass_threshold: float = 0.04,
    grid_size: float = GRID_SIZE,
    movement_gaussian_sigma: float = 0.005,
    render_depth_image: bool = False,
    render_instance_segmentation: bool = False,
    width: int = SCREEN_WIDTH,
    height: int = SCREEN_HEIGHT,
    render_third_party_cameras: bool = False,
    field_of_view: int = 60,
) -> Controller:
    """
    Initializes and returns an AI2-THOR controller instance with default or user-defined settings.

    Args:
        scene (str, optional): Scene name to load. Defaults to FloorPlan1.
        platform (str, optional): Platform to use. Defaults to None.
        agent_mode (str, optional): Agent mode ("default", "locobot", "drone", or "arm"). Defaults to "default".
        mass_threshold (float, optional): Minimum mass required for objects to be moved. Defaults to 0.04.
        grid_size (float, optional): Movement grid size (mean for Move Actions). Defaults to GRID_SIZE.
        movement_gaussian_sigma (float, optional): Movement sigma for Move Actions. Defaults to 0.005.
        render_depth_image (bool, optional): Whether to render depth images. Defaults to False.
        render_instance_segmentation (bool, optional): Whether to render instance segmentation. Defaults to False.
        width (int, optional): Screen width. Defaults to SCREEN_WIDTH.
        height (int, optional): Screen height. Defaults to SCREEN_HEIGHT.
        render_third_party_cameras (bool, optional): Whether to render third-party cameras. Defaults to False.
        field_of_view (int, optional): Camera field of view. Defaults to 60.

    Returns:
        Controller: Configured AI2-THOR Controller.
    """
    print(f"platform: {platform}")
    controller = Controller(
        agentMode=agent_mode,
        massThreshold=mass_threshold,
        scene=scene,
        gridSize=grid_size,
        movementGaussianSigma=movement_gaussian_sigma,
        renderDepthImage=render_depth_image,
        renderInstanceSegmentation=render_instance_segmentation,
        width=width,
        height=height,
        renderThirdPartyCameras=render_third_party_cameras,
        fieldOfView=field_of_view,
        platform=platform,
    )
    print(f"controller: {controller}")
    return controller


def execute_subtask(
    controller: Controller,
    subtask: Subtask,
    logger: logging.Logger,
    action_interface: Action,
) -> tuple[float, TaskExecutionStatus, float]:
    """
    Executes a given subtask using the provided AI2-THOR controller.

    Returns:
        tuple[float, TaskExecutionStatus, float]:
            - float: The total elapsed time taken to execute the subtask.
            - TaskExecutionStatus: Whether the subtask succeeded ('SUCCESS' or 'FAILURE').
            - float: 실제 첫 NAVIGATE_TO primitive action의 소요 시간(sim_nav_time)
    """

    if subtask.name == "Init":
        return 0.0, TaskExecutionStatus.SUCCESS, 0.0

    logger.info(f"Executing Subtask: {subtask.name}")

    execution = subtask.execution
    primitive_actions = execution.primitive_actions
    objects = execution.objects

    for action_str in primitive_actions:
        logger.debug(f"Primitive action: {action_str}")

    object_registry = {}
    if objects is not None:
        controller.step("Pass")
        all_obj_ids = {
            obj["objectId"] for obj in controller.last_event.metadata["objects"]
        }
        for obj_id in objects:
            obj_id_copy_1 = obj_id.replace("00.00", "+00.00")
            obj_id_copy_2 = obj_id.replace("+00.00", "00.00")
            if (
                obj_id not in all_obj_ids
                and obj_id_copy_1 not in all_obj_ids
                and obj_id_copy_2 not in all_obj_ids
            ):
                raise ValueError(f"Object '{obj_id}' not found in the environment.")
            object_registry[obj_id] = (
                obj_id_copy_1 if obj_id_copy_1 in all_obj_ids else obj_id_copy_2
            )

    action_mapping = {
        "NAVIGATE_TO": lambda target_obj: action_interface.move_to(target_obj),
        "GRASP": lambda target_obj: action_interface.pickup(target_obj),
        "PLACE_INSIDE": lambda target_obj: action_interface.put(target_obj),
        "PLACE_ON_TOP": lambda target_obj: action_interface.put(target_obj),
        "OPEN": lambda target_obj: action_interface.open(target_obj),
        "CLOSE": lambda target_obj: action_interface.close(target_obj),
        "TOGGLE_ON": lambda target_obj: action_interface.toggle_on(target_obj),
        "TOGGLE_OFF": lambda target_obj: action_interface.toggle_off(target_obj),
        "SLICE": lambda target_obj: action_interface.slice(target_obj),
        "MONITORING": lambda target_obj: action_interface.monitoring(target_obj),
        "WAIT": lambda duration: action_interface.wait(round(float(duration), 2)),
        # "FILL": lambda target_obj: action_interface.fill(target_obj),
    }

    elapsed_time = 0.0
    execution_status = TaskExecutionStatus.SUCCESS
    sim_nav_time = 0.0
    nav_time_found = False

    for action_str in primitive_actions:
        parts = action_str.split(" ", 1)
        action_type = parts[0]
        target_obj_id = parts[1] if len(parts) > 1 else None

        if action_type in action_mapping:
            action_duration = action_mapping[action_type](target_obj_id)
            elapsed_time += action_duration
            if not nav_time_found and action_type == "NAVIGATE_TO":
                sim_nav_time = action_duration
                nav_time_found = True
        else:
            logger.warning(f"Unknown action type: {action_type}. Skipping.")
            continue

        success = controller.last_event.metadata.get("lastActionSuccess", False)
        if not success:
            logger.warning(f"Action '{action_str}' failed.")
            execution_status = TaskExecutionStatus.FAILURE

        logger.info(
            f"Action: {action_str}, duration: {round(action_duration, 2)}, success: {success}"
        )
        controller.step(action="Pass")

    elapsed_time = round(elapsed_time, 2)
    logger.info(f"Subtask '{subtask.name}' completed. Elapsed time: {elapsed_time}")
    return elapsed_time, execution_status, sim_nav_time
