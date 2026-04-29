"""This module provides a class to handle ROS communication for executing subtasks."""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import requests

from src.models.dataclass import CompletedEntry
from src.models.task import Subtask
from src.utils.common.logger import create_module_logger
from src.utils.decorators import log_ros_action_state
from src.utils.translate import InstructionTranslator
EXPERIMENT_WAIT_DURATION = 10
logger = create_module_logger(module_name=__name__, module_log=True)

Position: TypeAlias = Tuple[float, float, float]


class SimulateObjectState:
    """Simulates changes in object positions based on actions."""

    def __init__(self, trajectory_log_path: Path) -> None:
        """Initializes the object position simulator."""
        try:
            with open("assets/runtime_state/static/object_init_states.json") as f:
                self.object_states = json.load(f)
        except FileNotFoundError:
            logger.error("object_init_states.json not found. Please check the path.")
            self.object_states = {}
        self.held_object: Optional[str] = None
        self.agent_location: List[float] = [0.0, 0.0, 0.0]
        self.trajectory_log_path: Path = trajectory_log_path

    @log_ros_action_state
    def _simulate_grasp(self, target_obj_id: Optional[str], duration: Optional[float]) -> None:
        """Simulates grasping an object."""
        self.held_object = target_obj_id
        self.object_states[target_obj_id]['parentReceptacles'] = ["agent"]

    @log_ros_action_state
    def _simulate_navigate(self, target_obj_id: Optional[str], duration: Optional[float]) -> None:
        """Simulates grasping an object."""
        logger.info(f"Navigating to {target_obj_id}")

    @log_ros_action_state
    def _simulate_wait(self, wait_duration: Optional[float], duration: Optional[float]) -> None:
        """Simulates waiting for a duration."""
        logger.info(f"Waiting for {wait_duration} seconds")

    
    @log_ros_action_state
    def _simulate_monitoring(self, duration: Optional[float]) -> None:
        """Simulates monitoring."""
        logger.info("Monitoring")

    @log_ros_action_state
    def _simulate_toggle_on(self, target_obj_id: Optional[str], duration: Optional[float]) -> None:
        """Simulates toggling on an object."""
        logger.info(f"Toggling on {target_obj_id}")
        # Check if 'isToggled' property exists before toggling
        if 'isToggled' in self.object_states.get(target_obj_id, {}):
            current_state: Optional[bool] = self.object_states[target_obj_id].get('isToggled')
            if isinstance(current_state, bool):
                self.object_states[target_obj_id]['isToggled'] = not current_state
                logger.info(
                    f"Toggled '{target_obj_id}' state to {self.object_states[target_obj_id]['isToggled']}"
                )
            else:
                logger.warning(
                    f"'isToggled' attribute for '{target_obj_id}' is not a boolean; cannot toggle."
                )
        else:
            logger.warning(
                f"Object '{target_obj_id}' does not have an 'isToggled' property, skipping toggle."
            )
    
    @log_ros_action_state
    def _simulate_toggle_off(self, target_obj_id: Optional[str], duration: Optional[float]) -> None:
        """Simulates toggling off an object."""
        logger.info(f"Toggling off {target_obj_id}")
        # Check if 'isToggled' property exists before toggling
        if 'isToggled' in self.object_states.get(target_obj_id, {}):
            current_state: Optional[bool] = self.object_states[target_obj_id].get('isToggled')
            if isinstance(current_state, bool):
                self.object_states[target_obj_id]['isToggled'] = not current_state
                logger.info(
                    f"Toggled '{target_obj_id}' state to {self.object_states[target_obj_id]['isToggled']}"
                )
            else:
                logger.warning(
                    f"'isToggled' attribute for '{target_obj_id}' is not a boolean; cannot toggle."
                )
        else:
            logger.warning(
                f"Object '{target_obj_id}' does not have an 'isToggled' property, skipping toggle."
            )
    @log_ros_action_state
    def _simulate_place(self, receptacle_id: Optional[str], duration: Optional[float]) -> None:
        """Simulates placing an object in or on a receptacle."""
        if not self.held_object:
            logger.warning("Agent not holding anything. Cannot place. Action FAILED.")
        elif not receptacle_id or receptacle_id.lower() not in self.object_states.keys():
            raise ValueError(
                f"Place target receptacle '{receptacle_id}' not found in object states."
            )
        else:
            logger.debug(f"  Placing '{self.held_object}' on/in '{receptacle_id}'.")
            if self.held_object in self.object_states:
                self.object_states[self.held_object]['parentReceptacles'] = [receptacle_id]
                self.object_states[self.held_object]['position'] = self.object_states[receptacle_id]['position']
            self.held_object = None




class RosExecutor:
    """Handles execution of subtasks via a remote ROS bridge."""

    def __init__(self, trajectory_log_path: Path) -> None:
        """Initializes the RosExecutor."""
        self.ros_bridge_url = os.getenv("ROS_BRIDGE_URL", "http://localhost:8000")
        self.object_state_simulator = SimulateObjectState(trajectory_log_path)
        self.held_object: Optional[str] = None
        self.ros_start_time: Optional[float] = None
        self.total_ros_time: float = 0.0
        # Initialize client-side translator
        self.translator = InstructionTranslator()
        logger.info(f"ROS Bridge URL set to: {self.ros_bridge_url}")
        self._check_server_health()

    def _check_server_health(self) -> None:
        """Checks if the ROS bridge server is running."""
        try:
            response = requests.get(f"{self.ros_bridge_url}/docs", timeout=5)
            response.raise_for_status()
            logger.info("Successfully connected to ROS bridge server.")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.error(
                f"Could not connect to ROS bridge server at {self.ros_bridge_url}. "
                f"Please ensure the server is running. Error: {e}"
            )
            raise

    def execute_primitive_actions(
        self, primitive_actions: List[str]
    ) -> Tuple[bool, float, List[Dict[str, Any]]]:
        """
        Executes a list of primitive actions via the ROS bridge and records timing.

        Args:
            primitive_actions: A list of primitive action strings to execute.

        Returns:
            A tuple containing:
            - bool: True if all actions succeed, False otherwise.
            - float: The total time elapsed for executing these actions.
            - List[Dict[str, Any]]: A log of each action and its execution time.
        """
        action_log: List[Dict[str, Any]] = []
        total_elapsed_time = 0.0

        for primitive_action in primitive_actions:
            action_start_time = time.time()
            success = False

            primitive_action_parts = primitive_action.split(" ")
            action_verb = primitive_action_parts[0].lower()

            if action_verb == "wait":
                wait_duration = float(primitive_action_parts[1])
                time.sleep(EXPERIMENT_WAIT_DURATION)
                logger.info(f"Waiting for {wait_duration} seconds")
                success = True
            else:
                try:
                    translated_parts = self.translator.translate(primitive_action)
                    logger.critical(
                        f"primitive_action: {primitive_action} translated to translated_parts: {translated_parts}"
                    )
                    response = requests.post(
                        f"{self.ros_bridge_url}/execute_translated_action",
                        json={"action_parts": translated_parts},
                        timeout=300,
                    )
                    response.raise_for_status()
                    result = response.json()
                    success = result.get("success", False)
                except requests.RequestException as e:
                    logger.error(
                        f"HTTP request for action '{primitive_action}' failed: {e}"
                    )
                    success = False

            action_end_time = time.time()
            elapsed_time = action_end_time - action_start_time
            logger.info(f"Action '{primitive_action}' took {elapsed_time:.2f} seconds")
            total_elapsed_time += elapsed_time
            action_log.append(
                {
                    "action": primitive_action,
                    "duration": elapsed_time,
                    "success": success,
                }
            )

            if not success:
                logger.error(f"Action '{primitive_action}' failed. Stopping task.")
                return False, total_elapsed_time, action_log

            # Simulate object state changes
            if primitive_action_parts[1].lower() == "sink":
                primitive_action_parts[1] = "sink|sinkbasin"    
            if action_verb == "grasp":
                self.object_state_simulator._simulate_grasp(
                    primitive_action_parts[1].lower(), duration=elapsed_time
                )
                self.held_object = primitive_action_parts[1]
                logger.info(f"Held object: {self.held_object}")
            elif action_verb.startswith("place"):
                self.object_state_simulator._simulate_place(
                    primitive_action_parts[1].lower(), duration=elapsed_time
                )

                self.held_object = None
            elif action_verb.startswith("navigate_to"):
                self.object_state_simulator._simulate_navigate(
                    primitive_action_parts[1].lower(), duration=elapsed_time
                )
            elif action_verb.startswith("wait"):
                self.object_state_simulator._simulate_wait(
                    wait_duration=EXPERIMENT_WAIT_DURATION, duration=float(primitive_action_parts[1])
                )
            elif action_verb.startswith("monitoring"):
                self.object_state_simulator._simulate_monitoring(duration=elapsed_time)
            elif action_verb.startswith("toggle_on"):
                self.object_state_simulator._simulate_toggle_on(
                    primitive_action_parts[1].lower(), duration=elapsed_time
                )
            elif action_verb.startswith("toggle_off"):
                self.object_state_simulator._simulate_toggle_off(
                    primitive_action_parts[1].lower(), duration=elapsed_time
                )
            os.makedirs("assets/runtime_state/dynamic", exist_ok=True)
            with open("assets/runtime_state/dynamic/object_states.json", "w") as f:
                json.dump(self.object_state_simulator.object_states, f, indent=4)

        return True, total_elapsed_time, action_log

    def execute_subtask(
        self, subtask: Subtask
    ) -> Tuple[bool, float, List[Dict[str, Any]]]:
        """
        Executes the primitive actions of a single subtask.

        Args:
            subtask: The subtask to execute.

        Returns:
            A tuple containing:
            - bool: True if execution is successful, False otherwise.
            - float: The total time elapsed for the subtask.
            - List[Dict[str, Any]]: A log of each action and its execution time.
        """
        if self.ros_start_time is None:
            self.ros_start_time = time.time()

        primitive_actions = subtask.execution.primitive_actions
        if not primitive_actions:
            return True, 0.0, []

        success, elapsed_time, action_logs = self.execute_primitive_actions(
            primitive_actions
        )
        self.total_ros_time += elapsed_time
        return success, elapsed_time, action_logs

    def execute_schedule(self, schedule: List[CompletedEntry]) -> List[CompletedEntry]:
        """
        Executes a pre-defined schedule of subtasks.

        This method iterates through a list of completed entries and executes
        the corresponding subtasks. It ensures that ROS communication is
        properly shut down afterwards.

        Args:
            schedule: A list of CompletedEntry objects representing the schedule.


        Returns:
            The schedule with updated execution information.
        """
        try:
            for entry in schedule:
                ros_start_offset = self.total_ros_time
                success, elapsed_time, action_logs = self.execute_subtask(entry.subtask)

                entry.sim_start_time = ros_start_offset
                entry.sim_end_time = ros_start_offset + elapsed_time
                entry.execution_status = success
                # Storing detailed action logs in a new attribute
                entry.primitive_action_log = action_logs

                if not success:
                    break
        finally:
            self.shutdown()
        return schedule

    def shutdown(self) -> None:
        """Placeholder for shutdown logic. Currently does nothing."""
        logger.info("ROS Executor shutdown.")
        pass
