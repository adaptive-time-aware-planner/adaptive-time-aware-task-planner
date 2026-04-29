import argparse
import concurrent.futures
import gc
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import psutil
import requests
import yaml

from src.utils.common import create_module_logger
from src.utils.config.constants import (
    ASSETS_PATH,
    INIT_PRIOR_VARIANCE,
    LOG_PATH,
    RESULT_PATH,
    SCRIPTS_PATH,
)

# Create a single timestamp for the entire script run
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

# Initialize a global logger for the main script
global logger
logger = create_module_logger(
    module_name=__name__,
    module_log=True,
    level=logging.ERROR,
    run_timestamp=RUN_TIMESTAMP,
)
MAX_RETRIES = 10

# Memory management constants
MEMORY_THRESHOLD_PERCENT = 85.0  # Pause new tasks if memory usage exceeds this
MEMORY_CHECK_INTERVAL = 5  # Seconds between memory checks
GC_COLLECT_AFTER_TASKS = 5  # Force garbage collection after this many tasks


class BaselineType(Enum):
    """Represents the type of baseline."""

    SCHEDULER = 1
    LLM = 0


class AblationConfig(Enum):
    """Enumerations for ablation study configurations."""

    NONE_URGENCY = {
        "alpha_heuristic": 1.0,
        "beta_heuristic": 0.0,
        "gamma_heuristic": 100.0,
    }
    NONE_REMAINING_WORK = {
        "alpha_heuristic": 1.0,
        "beta_heuristic": 10.0,
        "gamma_heuristic": 0.0,
    }
    NONE_MONITORING = {
        "disable_monitoring": True,
    }
    GREEDY = {
        "beam_width": 1,
        "beam_depth": 1,
    }

    DEFAULT = {}


class InitPriorConfig(Enum):
    """Enumerations for initial prior configurations."""

    OVER_ESTIMATE = {
        "init_prior_mean": 140.0,
        "init_prior_variance": INIT_PRIOR_VARIANCE,
    }
    OVER_MEDIUM_ESTIMATE = {
        "init_prior_mean": 120.0,
        "init_prior_variance": INIT_PRIOR_VARIANCE,
    }
    CORRECT_ESTIMATE = {
        "init_prior_mean": 100.0,
        "init_prior_variance": INIT_PRIOR_VARIANCE,
    }
    UNDER_MEDIUM_ESTIMATE = {
        "init_prior_mean": 80.0,
        "init_prior_variance": INIT_PRIOR_VARIANCE,
    }
    UNDER_ESTIMATE = {
        "init_prior_mean": 60.0,
        "init_prior_variance": INIT_PRIOR_VARIANCE,
    }

    OVER_ESTIMATE_130 = {
        "init_prior_mean": 130.0,
        "init_prior_variance": INIT_PRIOR_VARIANCE,
    }
    OVER_ESTIMATE_110 = {
        "init_prior_mean": 110.0,
        "init_prior_variance": INIT_PRIOR_VARIANCE,
    }
    UNDER_ESTIMATE_90 = {
        "init_prior_mean": 90.0,
        "init_prior_variance": INIT_PRIOR_VARIANCE,
    }
    UNDER_ESTIMATE_70 = {
        "init_prior_mean": 70.0,
        "init_prior_variance": INIT_PRIOR_VARIANCE,
    }


@dataclass
class ExperimentTask:
    """Represents a single experiment task to be run by a worker.

    Attributes:
        baseline_info (Tuple[BaselineType, Path]): The baseline type and script path.
        case_name (str): The name of the test case.
        scene_name (str): The name of the scene.
        instruction_path (str): The path to the instruction file.
        try_idx (int): The attempt index for the run.
        is_simulation (bool): Flag indicating if running in simulation mode.
        cloud_rendering (bool): Flag for enabling cloud rendering.
        ablation_name (str): The name of the ablation configuration.
        ablation_params (Dict[str, Any]): The parameters for the ablation.
        init_prior_name (str): The name of the initial prior configuration.
        init_prior_params (Dict[str, Any]): The parameters for the initial prior.
        file_copy_lock (threading.Lock): A lock for thread-safe file copying.
        max_retries (int): The maximum number of retries for the task.
        log_dir_timestamp (str): The timestamp for the log directory.
    """

    baseline_info: Tuple[BaselineType, Path]
    case_name: str
    scene_name: str
    instruction_path: str
    try_idx: int
    is_simulation: bool
    cloud_rendering: bool
    ablation_name: str
    ablation_params: Dict[str, Any]
    init_prior_name: str
    init_prior_params: Dict[str, Any]
    file_copy_lock: threading.Lock
    max_retries: int
    log_dir_timestamp: str
    gpu_id: int  # Assigned GPU ID for this task
    task_folder_name: str = "default_task_folder"  # The task folder name


def get_memory_usage() -> float:
    """Get current memory usage percentage.

    Returns:
        float: Current memory usage as a percentage.
    """
    return psutil.virtual_memory().percent


def wait_for_memory_available(
    threshold: float = MEMORY_THRESHOLD_PERCENT, timeout: int = 600
) -> None:
    """Wait until memory usage drops below threshold.

    Args:
        threshold (float): Memory usage percentage threshold.
        timeout (int): Maximum time to wait in seconds.
    """
    start_time = time.time()
    last_log_time = start_time

    while get_memory_usage() > threshold:
        current_time = time.time()
        elapsed = current_time - start_time

        if elapsed > timeout:
            logger.critical(
                f"Memory usage ({get_memory_usage():.1f}%) still high after {timeout}s. "
                "Forcing execution to proceed (ignoring threshold)."
            )
            break

        if current_time - last_log_time >= 30:
            logger.warning(
                f"Memory usage at {get_memory_usage():.1f}%. Waiting for memory to free up... ({int(elapsed)}s elapsed)"
            )
            last_log_time = current_time

        time.sleep(MEMORY_CHECK_INTERVAL)
        gc.collect()


def cleanup_subprocess(process: subprocess.Popen) -> None:
    """Forcefully cleanup a subprocess and its children.

    Args:
        process (subprocess.Popen): The subprocess to cleanup.
    """
    try:
        # Check if process is already dead
        if process.poll() is not None:
            return

        parent = psutil.Process(process.pid)
        try:
            children = parent.children(recursive=True)
        except psutil.NoSuchProcess:
            children = []

        # Kill children first (SIGKILL)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

        # Wait briefly for children to die
        _, alive = psutil.wait_procs(children, timeout=1)

        # Kill any survivors
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass

        # Kill parent
        try:
            parent.kill()
            parent.wait(timeout=1)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            pass

    except Exception as e:
        logger.warning(f"Error cleaning up subprocess: {e}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run experimental scenarios.")
    parser.add_argument(
        "--config",
        type=str,
        default="run_all_config.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, lists the experiments to be run without executing them.",
    )
    parser.add_argument(
        "--skip-completed",
        default=True,
        action="store_true",
        help=(
            "Skip tasks when a corresponding result JSON already exists (mirrors run_all.py behavior)."
        ),
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from the specified yaml file in the scripts directory.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        Dict[str, Any]: The loaded configuration dictionary.
    """
    config_path = SCRIPTS_PATH / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def _run_script_and_log(
    baseline_path: Path,
    ablation_name: str,
    case_name: str,
    scene_name: str,
    instruction_path: Path,
    is_simulation: bool,
    cloud_rendering: bool,
    log_path: Path,
    ablation_params: Dict[str, Any],
    init_prior_params: Dict[str, Any],
    attempt: int,
    gpu_id: int,
    task_folder_name: str,
) -> subprocess.CompletedProcess:
    """
    Constructs and runs a script command, capturing and logging the output.

    Args:
        baseline_path (Path): The path to the baseline script to execute.
        ablation_name (str): The name of the ablation configuration.
        case_name (str): The name of the case.
        scene_name (str): The name of the scene.
        instruction_path (Path): The path to the instruction file.
        is_simulation (bool): Flag for simulation mode.
        cloud_rendering (bool): Flag for cloud rendering.
        log_path (Path): The path to the log file for this run.
        ablation_params (Dict[str, Any]): Ablation configuration parameters.
        init_prior_params (Dict[str, Any]): Initial prior configuration parameters.
        attempt (int): The current attempt number.
        gpu_id (int): The GPU ID to use for this task.
        task_folder_name (str): The task folder name.

    Returns:
        subprocess.CompletedProcess: The result of the subprocess run.
    """
    wrapper_script = (
        SCRIPTS_PATH
        / "infra"
        / "run_project.sh"
    )

    cmd = [
        str(wrapper_script),
        "python3",
        str(baseline_path),
        "--scene",
        scene_name,
        "--case",
        case_name,
        "--ablation-name",
        ablation_name,
        "--instruction",
        instruction_path.name,
        "--log-path",
        str(log_path),
        "--task-folder-name",
        task_folder_name,
    ]

    if not is_simulation:
        cmd.append("--ros")
    else:
        cmd.append("--simulation")
        if cloud_rendering:
            cmd.append("--cloud-rendering")

    for key, value in (ablation_params | init_prior_params).items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    logger.info(f"Executing command (GPU {gpu_id}): {' '.join(cmd)}")

    # Set specific GPU for this process if assigned
    import os

    env = os.environ.copy()
    if gpu_id >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logger.info(f"Executing command: {' '.join(cmd)}")

    # Use Popen for better process control
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    try:
        stdout, stderr = process.communicate(timeout=1200)  # 20 minutes timeout
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        logger.error(f"Process timeout after 5 minutes. Killing process...")
        cleanup_subprocess(process)
        stdout, stderr = "", "Process killed due to timeout"
        returncode = -1
    except Exception as e:
        logger.error(f"Error during process execution: {e}")
        cleanup_subprocess(process)
        raise
    finally:
        # Ensure process is cleaned up
        cleanup_subprocess(process)

    # Create result object compatible with subprocess.run
    class ProcessResult:
        def __init__(self, returncode, stdout, stderr):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    result = ProcessResult(returncode, stdout, stderr)

    log_header = f"\n--- {'SUCCESS' if result.returncode == 0 else 'FAILURE'} (Attempt {attempt}) ---\n"
    log_content = (
        result.stdout
        if result.returncode == 0
        else f"Return Code: {result.returncode}\n--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}"
    )

    with log_path.open("a", encoding="utf-8") as f:
        f.write(log_header)
        f.write(log_content)

    logger.critical(
        f"Finished {baseline_path.name} (Ablation: {ablation_name}) (Inst) (Attempt {attempt}) with return code: {result.returncode}"
        f"Case: {case_name} (Scene: {scene_name}) (Instruction: {instruction_path.name})"
        f"Return Code: {result.returncode}"
        f"Stdout: {result.stdout}"
        f"Stderr: {result.stderr}"
    )
    return result


def return_to_initial_position(ros_bridge_url: str = "http://localhost:8000") -> None:
    """Sends a command to the robot to return to the initial position."""
    try:
        # Action parts: [robot_model, action_id, object_id, position]
        # NAVIGATE_TO is 10. Initial position is Node 100.
        action_parts = [0, 10, 100, 100]
        logger.info("Sending 'Return to Initial Position' command...")
        response = requests.post(
            f"{ros_bridge_url}/execute_translated_action",
            json={"action_parts": action_parts},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("success"):
            logger.info("Successfully returned to initial position.")
        else:
            logger.warning("Robot failed to return to initial position.")
    except Exception as e:
        logger.error(f"Error returning to initial position: {e}")


def worker(task: ExperimentTask) -> None:
    """
    Worker function to run a single experiment task.

    It handles object position resets for real-world experiments and executes
    the baseline script, with retry logic for LLM-based baselines.

    Args:
        task (ExperimentTask): The experiment task to execute.
    """
    # Wait for memory to be available before starting
    wait_for_memory_available()

    # Log memory usage at start
    mem_start = get_memory_usage()
    logger.info(f"Starting task with memory usage: {mem_start:.1f}%")

    try:
        # For real-world experiments, reset object positions to default before each run.
        if not task.is_simulation:
            with task.file_copy_lock:
                object_mapping_path = (
                    SCRIPTS_PATH.parent / "assets/runtime_state/static/object_init_states.json"
                )
                object_positions_path = (
                    SCRIPTS_PATH.parent / "assets/runtime_state/dynamic/object_states.json"
                )
                if object_mapping_path.exists():
                    shutil.copy2(object_mapping_path, object_positions_path)
                    logger.info(
                        f"Initialized object positions for task: {Path(task.instruction_path).name}"
                    )
                else:
                    logger.warning(
                        f"Source file for object positions not found: {object_mapping_path}"
                    )

        b_type, b_path = task.baseline_info
        instr_path_obj = Path(task.instruction_path)

        log_file_name = f"{b_path.stem}_{task.ablation_name}_{task.init_prior_name}_{task.case_name}_{task.scene_name}_{instr_path_obj.stem}_{task.try_idx + 1}.log"
        log_file_path = (
            LOG_PATH / f"{task.log_dir_timestamp}-worker_logs" / log_file_name
        )
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.critical(
            f"WORKER START | Baseline: {b_path.name} | Ablation: {task.ablation_name} | "
            f"Prior: {task.init_prior_name} | Case: {task.case_name} | Scene: {task.scene_name} | "
            f"Run: {task.try_idx + 1} | Task: {instr_path_obj.name}"
        )

        if b_type == BaselineType.SCHEDULER:
            buffer_between_instructions = 0 if task.is_simulation else 30
            if buffer_between_instructions > 0:
                logger.info(
                    f"Waiting for {buffer_between_instructions} seconds before starting."
                )
                time.sleep(buffer_between_instructions)

            for attempt in range(1, task.max_retries + 1):
                result = _run_script_and_log(
                    b_path,
                    task.ablation_name,
                    task.case_name,
                    task.scene_name,
                    instr_path_obj,
                    task.is_simulation,
                    task.cloud_rendering,
                    log_file_path,
                    task.ablation_params,
                    task.init_prior_params,
                    attempt=attempt,
                    gpu_id=task.gpu_id,
                    task_folder_name=task.task_folder_name,
                )
                if result.returncode == 0:
                    logger.info(
                        f"Scheduler baseline {b_path.name} succeeded on attempt {attempt}."
                    )
                    break

                if attempt < task.max_retries:
                    logger.warning(
                        f"Scheduler baseline {b_path.name} failed on attempt {attempt}. Retrying in 10 seconds..."
                    )
                    time.sleep(10)
            else:  # No-break
                logger.error(
                    f"Scheduler baseline {b_path.name} failed after {task.max_retries} attempts."
                )
        elif b_type == BaselineType.LLM:
            buffer_between_instructions = 2 if task.is_simulation else 20
            if buffer_between_instructions > 0:
                logger.info(
                    f"Waiting for {buffer_between_instructions} seconds before starting."
                )
                time.sleep(buffer_between_instructions)
            for attempt in range(1, task.max_retries + 1):
                result = _run_script_and_log(
                    b_path,
                    task.ablation_name,
                    task.case_name,
                    task.scene_name,
                    instr_path_obj,
                    task.is_simulation,
                    task.cloud_rendering,
                    log_file_path,
                    task.ablation_params,
                    task.init_prior_params,
                    attempt=attempt,
                    gpu_id=task.gpu_id,
                    task_folder_name=task.task_folder_name,
                )
                if result.returncode == 0:
                    logger.info(
                        f"LLM baseline {b_path.name} succeeded on attempt {attempt}."
                    )
                    break

                if attempt < task.max_retries:
                    logger.warning(
                        f"LLM baseline {b_path.name} failed on attempt {attempt}. Retrying in {buffer_between_instructions} seconds..."
                    )
                    time.sleep(buffer_between_instructions)
            else:  # No-break
                logger.error(
                    f"LLM baseline {b_path.name} failed after {task.max_retries} attempts."
                )
    finally:
        # Return robot to initial position after each real-world task (success or failure)
        if not task.is_simulation:
            ros_bridge_url = os.getenv("ROS_BRIDGE_URL", "http://localhost:8000")
            return_to_initial_position(ros_bridge_url)

        # Force garbage collection after each task
        gc.collect()

        # Log memory usage at end
        mem_end = get_memory_usage()
        logger.info(
            f"Task completed. Memory usage: {mem_end:.1f}% (change: {mem_end - mem_start:+.1f}%)"
        )


def load_instruction_case_mapping_from_scenes(
    target_scenes: List[str],
    task_folder_names: str | List[str],
    min_task_count: int = 0,
    max_task_count: int = 99,
    min_constraint_count: int = 0,
    max_constraint_count: int = 99,
) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    """
    Loads instruction files and maps them to cases and scenes.

    It scans specified task folders, filtering by task and constraint counts,
    and organizes the found instruction files by case and scene.

    Args:
        target_scenes (List[str]): A list of scene names to include.
        task_folder_names (str | List[str]): The name(s) of the folder(s) within 'assets/tasks/' to search.
        min_task_count (int): The minimum number of tasks for a case to be included (inclusive).
        max_task_count (int): The maximum number of tasks for a case to be included (inclusive).
        min_constraint_count (int): The minimum number of constraints for a case to be included (inclusive).
        max_constraint_count (int): The maximum number of constraints for a case to be included (inclusive).

    Returns:
        Dict[str, Dict[str, List[Tuple[str, str]]]]: A nested dictionary mapping case names to
        a dictionary of scene names to a list of tuples (instruction file path, source task folder name).
    """
    if isinstance(task_folder_names, str):
        task_folder_names = [task_folder_names]

    case_instruction_mapping: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for task_folder_name in task_folder_names:
        task_folder = ASSETS_PATH / "tasks" / task_folder_name
        if not task_folder.exists():
            logger.error(f"Task folder not found: {task_folder}")
            # Instead of raising, we just continue to next folder or raise if none found?
            # Original code raised. Let's log and continue, but if all fail maybe raise.
            continue

        for test_case in sorted(task_folder.iterdir(), key=lambda x: x.name):
            if not test_case.is_dir():
                continue

            match = re.match(r"tasks_(\d+)_constraints_(\d+)", test_case.name)
            if not match:
                continue

            num_tasks, num_constraints = map(int, match.groups())

            if not (min_task_count <= num_tasks <= max_task_count) or \
               not (min_constraint_count <= num_constraints <= max_constraint_count):
                continue

            for scene_name in target_scenes:
                scene_folder = test_case / scene_name
                if scene_folder.exists() and scene_folder.is_dir():
                    instruction_files = sorted(
                        [f for f in scene_folder.glob("*.json")],
                        key=lambda x: x.name,
                    )
                    case_instruction_mapping[test_case.name][scene_name].extend(
                        [(str(f), task_folder_name) for f in instruction_files]
                    )

    if not case_instruction_mapping:
        logger.warning("No cases found in any of the provided task folders.")

    return case_instruction_mapping


def _iter_states_dirs(
    config: Dict[str, Any], task_folder_name: str | None = None
) -> List[Path]:
    """Return list of states* result directories under RESULT_PATH.

    If ``init_prior_mean`` is provided in the config, only that specific
    directory will be returned; otherwise, all directories starting with
    ``states`` are returned.

    Args:
        config (Dict[str, Any]): The configuration dictionary.
        task_folder_name (str | None): Optional task folder name to scope results.

    Returns:
        List[Path]: List of candidate initial prior result directories.
    """
    states_dirs: List[Path] = []
    base_path = RESULT_PATH
    if task_folder_name:
        base_path = base_path / task_folder_name

    init_prior = config.get("init_prior_mean")
    if isinstance(init_prior, (int, float)):
        states_dirs = [base_path / f"states{int(init_prior)}"]
    else:
        if base_path.exists():
            for p in base_path.iterdir():
                if p.is_dir() and p.name.startswith("states"):
                    states_dirs.append(p)
    return states_dirs


def _find_latest_result_json_for_task(
    baseline_path: Path,
    instruction_path: str,
    scene_name: str,
    config: Dict[str, Any],
    case_name: str | None = None,
    ablation_name: str | None = None,
    task_folder_name: str | None = None,
) -> Path | None:
    """Locate the latest result JSON for a given baseline/instruction/scene.

    This mirrors the skip logic used in run_all.py: it searches under
    assets/results/[task_folder_name]/states*/{instruction_stem}_*/{scene}/{baseline_name}/end_state.json
    and returns the one with the highest trailing number.

    Args:
        baseline_path (Path): Baseline script path.
        instruction_path (str): Path to the instruction JSON file.
        scene_name (str): Scene name.
        config (Dict[str, Any]): Configuration dictionary.
        case_name (str | None): Case name.
        ablation_name (str | None): Ablation name.
        task_folder_name (str | None): Task folder name.

    Returns:
        Path | None: The latest existing result JSON path if found, else None.
    """
    # dag_bayesian special handling: it appends ablation name instead of simulation suffix when case is present
    if "dag_bayesian" in baseline_path.name and case_name and ablation_name:
        approach_name = f"dag_bayesian_{ablation_name}"
    else:
        suffix = "_simulation" if config.get("simulation", False) else ""
        approach_name = f"{baseline_path.stem}{suffix}"

    # Create list of approach names to search
    # BUG FIX: save_scene_state uses "cpm" but result_save uses "cpm_simulation"
    # So we need to search both versions
    approach_names_to_search = [approach_name]
    if approach_name.endswith("_simulation"):
        # Also search without _simulation suffix (e.g., "cpm" for "cpm_simulation")
        approach_names_to_search.append(approach_name.replace("_simulation", ""))

    # Robust instruction keys: keep numeric prefix, normalize spaces to underscores
    raw_stem = Path(instruction_path).stem
    underscore_stem = raw_stem.replace(" ", "_")
    instruction_keys: List[str] = []
    for key in (raw_stem, underscore_stem):
        if key not in instruction_keys:
            instruction_keys.append(key)

    candidates: List[Tuple[int, Path]] = []
    for state_dir in _iter_states_dirs(config, task_folder_name):
        base_dir = state_dir / case_name if case_name else state_dir
        if not base_dir.exists():
            continue
        for key in instruction_keys:
            # Exact folder (no numeric suffix)
            exact_dir = base_dir / key
            if exact_dir.is_dir():
                # Try all approach name variations
                for approach_name_variant in approach_names_to_search:
                    json_path = (
                        exact_dir
                        / scene_name
                        / approach_name_variant
                        / "end_state.json"
                    )
                    if json_path.exists():
                        candidates.append((0, json_path))

            # Numeric suffixed folders: key_1, key_2, ...
            for task_dir in base_dir.glob(f"{key}_*"):
                if not task_dir.is_dir():
                    continue
                m = re.search(r"_(\d+)$", task_dir.name)
                if not m:
                    continue
                num = int(m.group(1))
                # Try all approach name variations
                for approach_name_variant in approach_names_to_search:
                    json_path = (
                        task_dir / scene_name / approach_name_variant / "end_state.json"
                    )
                    if json_path.exists():
                        candidates.append((num, json_path))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _is_completed_result(json_path: Path) -> bool:
    """Return True if a result JSON file exists.

    Args:
        json_path (Path): Path to the JSON result file.

    Returns:
        bool: True if the file exists, else False.
    """
    return json_path.exists()


def should_skip_completed_for_task(
    baseline_path: Path,
    instruction_path: str,
    scene_name: str,
    config: Dict[str, Any],
    case_name: str | None = None,
    ablation_name: str | None = None,
    task_folder_name: str | None = None,
) -> Tuple[bool, Path | None]:
    """Check whether to skip a task due to an existing result JSON.

    The decision is controlled by ``config['skip_completed']``. If enabled, it
    searches for the latest result JSON using the same heuristic as run_all.py.

    Args:
        baseline_path (Path): Baseline script path.
        instruction_path (str): Path to the instruction JSON file.
        scene_name (str): Scene name.
        config (Dict[str, Any]): Configuration dictionary.
        case_name (str | None): Case name.
        ablation_name (str | None): Ablation name.
        task_folder_name (str | None): Task folder name.

    Returns:
        Tuple[bool, Path | None]: (should_skip, found_json_path)
    """
    if not config.get("skip_completed"):
        return False, None
    result_json = _find_latest_result_json_for_task(
        baseline_path,
        instruction_path,
        scene_name,
        config,
        case_name,
        ablation_name,
        task_folder_name,
    )
    if result_json and _is_completed_result(result_json):
        return True, result_json
    return False, None


def main() -> None:
    """
    Main function to orchestrate the experimental runs.

    It parses arguments, loads configurations, prepares experiment tasks,
    and executes them in parallel using a thread pool.
    """
    args = parse_arguments()
    config = load_config(args.config)
    # Use the global timestamp defined at the top of the script
    run_timestamp = RUN_TIMESTAMP

    # Reflect CLI skip flag into config (same behavior as run_all.py)
    if getattr(args, "skip_completed", False):
        config["skip_completed"] = True

    is_simulation: bool = config.get("simulation", False)
    cloud_rendering: bool = config.get("cloud_rendering", False)
    num_runs_per_instruction: int = config.get("num_runs_per_instruction", 1)
    max_retries: int = config.get("max_retries", 10)
    max_workers: int = config.get("max_workers", 10)
    num_gpus: int = config.get("num_gpus", 0)
    start_idx: int = config.get("start_idx", 1)
    end_idx: int = config.get("end_idx", 99999)

    is_dry_run: bool = args.dry_run

    sched_baselines: List[Tuple[BaselineType, Path]] = (
        [(BaselineType.SCHEDULER, Path(p)) for p in config.get("approaches", [])]
        if config.get("approaches", [])
        else []
    )

    llm_baselines: List[Tuple[BaselineType, Path]] = (
        [(BaselineType.LLM, Path(p)) for p in config.get("llm_scripts", [])]
        if config.get("llm_scripts", [])
        else []
    )

    baselines = sched_baselines + llm_baselines

    scene_types_config = config.get("scene_type", "kitchen")
    scene_types = (
        [scene_types_config]
        if isinstance(scene_types_config, str)
        else scene_types_config
    )
    scene_lists = config.get("scene_lists", {})
    target_scenes = [
        scene for scene_type in scene_types for scene in scene_lists.get(scene_type, [])
    ]

    min_task_count = config.get("min_task_count", 0)
    max_task_count = config.get("max_task_count", 99)
    min_constraint_count = config.get("min_constraint_count", 0)
    max_constraint_count = config.get("max_constraint_count", 99)
    task_folder_name = config.get(
        "task_folder_name", "decomposed_rightbefore_final_251031"
    )
    case_instruction_mapping: Dict[str, Dict[str, List[Tuple[str, str]]]] = (
        load_instruction_case_mapping_from_scenes(
            target_scenes, task_folder_name,
            min_task_count, max_task_count,
            min_constraint_count, max_constraint_count
        )
    )

    ablation_names_to_run = config.get("ablation_configs", [])
    if ablation_names_to_run:
        ablation_configs_to_run = {
            name: AblationConfig[name].value
            for name in ablation_names_to_run
            if name in AblationConfig.__members__
        }
    else:
        ablation_configs_to_run = {
            name: config.value for name, config in AblationConfig.__members__.items()
        }

    init_prior_names_to_run = config.get("init_prior_configs", [])
    if init_prior_names_to_run:
        init_prior_configs_to_run = {
            name: InitPriorConfig[name].value
            for name in init_prior_names_to_run
            if name in InitPriorConfig.__members__
        }
    else:
        init_prior_configs_to_run = {
            name: config.value for name, config in InitPriorConfig.__members__.items()
        }

    if not ablation_configs_to_run:
        ablation_configs_to_run = {"DEFAULT": {}}
    if not init_prior_configs_to_run:
        init_prior_configs_to_run = {"DEFAULT": {}}

    execute_dict = config.get("execute_dict", {})
    file_copy_lock = threading.Lock()

    logger.critical(f"Found {len(case_instruction_mapping)} cases to run.")
    logger.critical(f"Found {len(baselines)} baselines to run.")
    logger.critical(f"Found {len(ablation_configs_to_run)} ablation configs to run.")
    logger.critical(
        f"Found {len(init_prior_configs_to_run)} init prior configs to run."
    )
    # logger.critical(f"Found {len(execute_dict)} execute dicts to run.")

    tasks_to_run = []
    gpu_counter = 0
    # num_gpus is now loaded from config above

    for baseline in baselines:
        b_type, b_path = baseline

        # Use all ablation configs for dag_bayesian, only DEFAULT for others.
        ablations_to_use = ablation_configs_to_run
        if "dag_bayesian.py" not in str(b_path):
            ablations_to_use = {"DEFAULT": AblationConfig.DEFAULT.value}

        # Create product of remaining configurations
        experiment_configs = product(
            ablations_to_use.items(),
            init_prior_configs_to_run.items(),
            case_instruction_mapping.items(),
        )

        for (
            (ablation_name, ablation_params),
            (init_prior_name, init_prior_params),
            (case_name, scene_map),
        ) in experiment_configs:
            for scene_name, instructions in scene_map.items():
                for instruction_path, task_src_folder_name in instructions:
                    instr_path_obj = Path(instruction_path)
                    # Apply start_idx filter
                    instr_idx_match = re.match(r"(\d+)_", instr_path_obj.name)
                    if instr_idx_match:
                        instr_idx = int(instr_idx_match.group(1))
                        if instr_idx < start_idx or instr_idx > end_idx:
                            continue
                    # Apply execute_dict filter if it's defined
                    if execute_dict:
                        if case_name not in execute_dict:
                            continue
                        if scene_name not in execute_dict[case_name]:
                            continue
                        if (
                            instr_path_obj.name
                            not in execute_dict[case_name][scene_name]
                        ):
                            continue

                    # Skip if a completed result already exists (when enabled)
                    # Merge current init_prior params so _iter_init_dirs targets the correct init_* dir
                    merged_config_for_skip = {**config, **init_prior_params}
                    do_skip, found_json = should_skip_completed_for_task(
                        b_path,
                        instruction_path,
                        scene_name,
                        merged_config_for_skip,
                        case_name,
                        ablation_name,
                        task_src_folder_name,
                    )
                    if do_skip:
                        logger.critical(
                            f"Skip completed: {scene_name} | {b_path.stem} | '{instr_path_obj.stem}' -> {found_json}"
                        )
                        continue

                    for try_idx in range(num_runs_per_instruction):
                        # Round-robin GPU assignment if GPUs are available
                        gpu_id = -1
                        if num_gpus > 0:
                            gpu_id = gpu_counter % num_gpus
                            gpu_counter += 1

                        task = ExperimentTask(
                            baseline_info=baseline,
                            case_name=case_name,
                            scene_name=scene_name,
                            instruction_path=instruction_path,
                            try_idx=try_idx,
                            is_simulation=is_simulation,
                            cloud_rendering=cloud_rendering,
                            ablation_name=ablation_name,
                            ablation_params=ablation_params,
                            init_prior_name=init_prior_name,
                            init_prior_params=init_prior_params,
                            file_copy_lock=file_copy_lock,
                            max_retries=max_retries,
                            log_dir_timestamp=run_timestamp,
                            gpu_id=gpu_id,
                            task_folder_name=task_src_folder_name,
                        )
                        tasks_to_run.append(task)

    # Diagnostic logging for task generation
    task_counts = defaultdict(int)
    for t in tasks_to_run:
        task_counts[t.ablation_name] += 1

    logger.critical("Task Generation Summary:")
    for ablation, count in task_counts.items():
        logger.critical(f"  - Ablation '{ablation}': {count} tasks")
    logger.critical(f"  - Total: {len(tasks_to_run)} tasks")

    if is_dry_run:
        logger.critical("=" * 80)
        logger.critical(f"DRY RUN MODE: Found {len(tasks_to_run)} experiments to run.")
        logger.critical("=" * 80)
        count = 0
        for task in tasks_to_run:
            b_type, b_path = task.baseline_info
            instr_path_obj = Path(task.instruction_path)
            log_msg = (
                f"  - Baseline: {b_path.name:<30} | "
                f"Ablation: {task.ablation_name:<15} | "
                f"Prior: {task.init_prior_name:<15} | "
                f"Case: {task.case_name:<8} | "
                f"Scene: {task.scene_name:<15} | "
                f"Instruction: {instr_path_obj.name:<80} | "
                f"Run: {task.try_idx + 1}"
            )
            logger.critical(log_msg)
            count += 1
        logger.critical(f"Total {count} experiments to run.")
        logger.critical("=" * 80)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: List[concurrent.futures.Future] = [
            executor.submit(worker, task) for task in tasks_to_run
        ]

        pending_futures = set(futures)
        completed_count = 0
        last_log_time = time.time()

        while pending_futures:
            # Check for completed futures with a short timeout to allow periodic logging
            done, not_done = concurrent.futures.wait(
                pending_futures,
                timeout=30,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            # Process completed tasks
            for future in done:
                try:
                    future.result()
                    completed_count += 1

                    # Periodically force garbage collection
                    if completed_count % GC_COLLECT_AFTER_TASKS == 0:
                        logger.info(
                            f"Completed {completed_count} tasks. Running garbage collection..."
                        )
                        gc.collect()
                        mem_usage = get_memory_usage()
                        logger.info(f"Current memory usage: {mem_usage:.1f}%")
                except Exception as e:
                    logger.critical(f"A task generated an exception: {e}")
                    logger.critical(traceback.format_exc())

            pending_futures = not_done

            # Log status if no tasks completed recently or periodically
            current_time = time.time()
            if current_time - last_log_time >= 60 or not done:
                logger.critical(
                    f"Progress Update: {completed_count}/{len(tasks_to_run)} completed. "
                    f"{len(pending_futures)} pending/running. "
                    f"(Memory: {get_memory_usage():.1f}%)"
                )
                last_log_time = current_time

        # Final garbage collection
        logger.info("All tasks completed. Final garbage collection...")
        gc.collect()


if __name__ == "__main__":
    main()
