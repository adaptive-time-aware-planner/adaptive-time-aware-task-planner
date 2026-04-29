"""In-process offline scheduler experiment harness."""

from __future__ import annotations

import copy
import heapq
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import yaml
from src.core import Agent, Scheduler
from src.experiments.exact_oracle import (
    DeterministicExactOracle,
    build_initial_oracle_upper_bound,
)
from src.experiments.offline_oracle_reference import (
    ORACLE_REFERENCE_SCHEMA_VERSION,
    build_oracle_comparison_section,
    load_oracle_results_for_selection,
    resolve_oracle_reference_dir,
    save_oracle_reference_rows,
    summarize_oracle_results,
)
from src.core.monitoring import (
    create_ground_truth_store,
    create_monitoring_backend,
    create_observation_model,
)
from src.models.dataclass import (
    ActionResult,
    CompletedEntry,
    SchedulerState,
    SimulationNode,
    TaskExecutionStatus,
)
from src.models.task import Duration, Execution, Subtask
from src.scheduler import ActionHandler, ConstraintHandler, HeuristicManager
from src.utils.config import constants
from src.utils.io_utils.result_saver import (
    build_hybrid_single_run_payload,
    calculate_timing_success_rate,
    serialize_completed_entries,
)
from src.utils.io_utils.task_io import load_scene_positions, load_task_data_from_sampled_set
from src.utils.task.task_util import TaskUtil

Position = Tuple[float, float, float]
NavGraph = Dict[Position, list[Position]]
BeamBound = tuple[int, int]

DEFAULT_BEAM_BOUND: list[BeamBound] = [(1, 1), (5, 5), (10, 10)]
DEFAULT_INIT_PRIOR_CONFIG = "CORRECT_ESTIMATE"
DEFAULT_ABLATION_CONFIG = "DEFAULT"
DEFAULT_ORACLE_TIMEOUT_SECONDS = 300.0
NAV_GRAPH_CACHE_SCHEMA_VERSION = "ai2thor_nav_graph.v1"
SUPPORTED_BELIEF_UPDATE_METHODS = {"bayesian", "particle_filter"}
SUPPORTED_DURATION_DISTRIBUTIONS = {
    "constant",
    "gaussian",
    "lognormal",
    "gamma",
    "mixture",
}
INIT_PRIOR_CONFIG_MAP: dict[str, tuple[float, float]] = {
    "OVER_ESTIMATE": (140.0, constants.INIT_PRIOR_VARIANCE),
    "OVER_MEDIUM_ESTIMATE": (120.0, constants.INIT_PRIOR_VARIANCE),
    "CORRECT_ESTIMATE": (100.0, constants.INIT_PRIOR_VARIANCE),
    "UNDER_MEDIUM_ESTIMATE": (80.0, constants.INIT_PRIOR_VARIANCE),
    "UNDER_ESTIMATE": (60.0, constants.INIT_PRIOR_VARIANCE),
    "OVER_ESTIMATE_130": (130.0, constants.INIT_PRIOR_VARIANCE),
    "OVER_ESTIMATE_110": (110.0, constants.INIT_PRIOR_VARIANCE),
    "UNDER_ESTIMATE_90": (90.0, constants.INIT_PRIOR_VARIANCE),
    "UNDER_ESTIMATE_70": (70.0, constants.INIT_PRIOR_VARIANCE),
}
ABLATION_CONFIG_MONITORING: dict[str, bool] = {
    "DEFAULT": True,
    "NONE_MONITORING": False,
}


def _quantize_position(position: Position) -> Position:
    """Snap a position onto the global navigation grid."""

    return tuple(
        round(float(coord) / constants.GRID_SIZE) * constants.GRID_SIZE
        for coord in position
    )


def _normalize_beam_bound(raw_value: Sequence[Any]) -> list[BeamBound]:
    """Normalize beam bound entries into integer ``(width, depth)`` tuples."""

    normalized: list[BeamBound] = []
    for raw_entry in raw_value:
        if isinstance(raw_entry, str):
            width_str, depth_str = raw_entry.split(",", maxsplit=1)
            normalized.append((int(width_str), int(depth_str)))
            continue
        if isinstance(raw_entry, Sequence) and not isinstance(raw_entry, (str, bytes)):
            if len(raw_entry) != 2:
                raise ValueError(f"Invalid beam_bound entry: {raw_entry}")
            normalized.append((int(raw_entry[0]), int(raw_entry[1])))
            continue
        raise ValueError(f"Unsupported beam_bound entry: {raw_entry}")
    if not normalized:
        raise ValueError("beam_bound must contain at least one (width, depth) pair.")
    return normalized


def _resolve_effective_beam_bounds(config: ExperimentConfig) -> list[BeamBound]:
    """Collapse beam sweeps for planners that do not use beam search."""

    if config.approach == "bayesian":
        return list(config.beam_bound)
    return [config.beam_bound[0]]


@dataclass
class ExperimentConfig:
    """Serializable configuration for an offline experiment session."""

    experiment_name: str = "offline_experiment"
    suite_name: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    task_folder_name: str = "sampled_10_instruction_set_for_final_experiment_set_b"
    approach: str = "bayesian"
    ablation_config: str = DEFAULT_ABLATION_CONFIG
    init_prior_config: str = DEFAULT_INIT_PRIOR_CONFIG
    case: str = "tasks_3_constraints_2"
    cases: list[str] = field(default_factory=list)
    scene: str = "FloorPlan13"
    instructions: list[str] = field(default_factory=list)
    max_tasks: int = 3
    beam_bound: list[BeamBound] = field(
        default_factory=lambda: list(DEFAULT_BEAM_BOUND)
    )
    belief_update_method: str = "bayesian"
    gt_distribution: str = "constant"
    gt_seed: int = 42
    particle_distribution: Optional[str] = None
    particle_likelihood_family: Optional[str] = None
    factor_alpha: Optional[float] = None
    eta: Optional[float] = None
    max_monitoring_per_critical_interval: Optional[int] = None
    observation_mode: str = "synthetic_gaussian"
    nav_graph_source: str = "ai2thor_controller"
    output_path: Optional[str] = None
    oracle_reference_dir: Optional[str] = None
    oracle_time_limit_seconds: float = DEFAULT_ORACLE_TIMEOUT_SECONDS
    schema_version: str = "offline_harness.v1"

    def __post_init__(self) -> None:
        """Normalize config payloads loaded from YAML/CLI."""

        if self.suite_name is not None:
            self.suite_name = str(self.suite_name)
        self.tags = [str(tag) for tag in self.tags]
        self.cases = [str(case_name) for case_name in self.cases]
        self.instructions = [str(instruction) for instruction in self.instructions]
        self.approach = str(self.approach)
        self.ablation_config = str(self.ablation_config)
        self.init_prior_config = str(self.init_prior_config)
        self.belief_update_method = str(self.belief_update_method)
        self.gt_distribution = str(self.gt_distribution)
        if self.particle_distribution is not None:
            self.particle_distribution = str(self.particle_distribution)
        if self.particle_likelihood_family is not None:
            self.particle_likelihood_family = str(self.particle_likelihood_family)
        if isinstance(self.max_monitoring_per_critical_interval, str):
            raw_budget = self.max_monitoring_per_critical_interval.strip().lower()
            if raw_budget in {"uncap", "unbounded", "inf", "infinity"}:
                self.max_monitoring_per_critical_interval = -1
            else:
                self.max_monitoring_per_critical_interval = int(raw_budget)
        elif self.max_monitoring_per_critical_interval is not None:
            self.max_monitoring_per_critical_interval = int(
                self.max_monitoring_per_critical_interval
            )
        self.beam_bound = _normalize_beam_bound(self.beam_bound)
        if self.approach not in {"bayesian", "edf", "cpm", "oracle"}:
            raise ValueError(f"Unsupported approach: {self.approach}")
        if self.ablation_config not in ABLATION_CONFIG_MONITORING:
            raise ValueError(f"Unsupported ablation_config: {self.ablation_config}")
        if self.init_prior_config not in INIT_PRIOR_CONFIG_MAP:
            raise ValueError(f"Unsupported init_prior_config: {self.init_prior_config}")
        if self.belief_update_method not in SUPPORTED_BELIEF_UPDATE_METHODS:
            raise ValueError(
                f"Unsupported belief_update_method: {self.belief_update_method}"
            )
        if self.gt_distribution not in SUPPORTED_DURATION_DISTRIBUTIONS:
            raise ValueError(f"Unsupported gt_distribution: {self.gt_distribution}")
        if (
            self.particle_distribution is not None
            and self.particle_distribution not in SUPPORTED_DURATION_DISTRIBUTIONS
        ):
            raise ValueError(
                f"Unsupported particle_distribution: {self.particle_distribution}"
            )
        if (
            self.particle_likelihood_family is not None
            and self.particle_likelihood_family not in SUPPORTED_DURATION_DISTRIBUTIONS
        ):
            raise ValueError(
                "Unsupported particle_likelihood_family: "
                f"{self.particle_likelihood_family}"
            )


@dataclass(frozen=True)
class ExperimentTask:
    """Single rollout unit inside an offline experiment grid."""

    instruction: str
    beam_width: int
    beam_depth: int
    task_index: int
    case: str = ""


def _read_config_file(config_path: Path) -> dict[str, Any]:
    """Load a JSON or YAML experiment config file."""

    suffix = config_path.suffix.lower()
    raw_text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(raw_text)
    if suffix in {".yaml", ".yml"}:
        loaded = yaml.safe_load(raw_text)
        return loaded or {}
    raise ValueError(f"Unsupported config file extension: {config_path.suffix}")


def load_experiment_config(config_path: Optional[Path]) -> ExperimentConfig:
    """Load experiment configuration from disk or defaults.

    Args:
        config_path: Optional JSON/YAML path.

    Returns:
        ExperimentConfig: Parsed configuration object.
    """

    if config_path is None:
        return ExperimentConfig()

    payload = _read_config_file(config_path)
    payload.pop("execution_mode", None)
    return ExperimentConfig(**payload)


def apply_cli_overrides(
    config: ExperimentConfig,
    overrides: Mapping[str, Any],
) -> ExperimentConfig:
    """Apply non-``None`` CLI overrides onto a config object."""

    merged = asdict(config)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return ExperimentConfig(**merged)


def _resolve_particle_distribution(config: ExperimentConfig) -> str:
    """Resolve the PF particle prior family for one run.

    When the user does not explicitly choose a particle distribution, PF runs
    default to the active GT family so the particle posterior can represent the
    same non-Gaussian support being tested. Non-PF runs keep the legacy
    Gaussian approximation.
    """

    if config.particle_distribution is not None:
        return config.particle_distribution
    if config.belief_update_method == "particle_filter":
        return config.gt_distribution
    return "gaussian"


def _resolve_monitoring_budget_label(
    max_monitoring_per_critical_interval: int | None,
) -> str | None:
    """Return the serialized monitoring-budget label for one run."""

    if max_monitoring_per_critical_interval is None:
        return None
    if int(max_monitoring_per_critical_interval) < 0:
        return "uncap"
    return str(int(max_monitoring_per_critical_interval))


def _mean_or_none(values: Sequence[float]) -> float | None:
    """Return the arithmetic mean or ``None`` when the input is empty."""

    return mean(values) if values else None


def _p95_or_none(values: Sequence[float]) -> float | None:
    """Return the empirical p95 or ``None`` for empty inputs."""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95))))
    return ordered[index]


def build_grid_nav_graph(scene_positions: Mapping[str, Position]) -> NavGraph:
    """Build a lightweight grid navigation graph from scene object positions."""

    if "agent" not in scene_positions:
        raise ValueError("Scene positions must contain an 'agent' entry.")

    grid_size = constants.GRID_SIZE
    agent_position = tuple(scene_positions["agent"])
    agent_y = _quantize_position(agent_position)[1]
    floor_points = {
        (
            _quantize_position(tuple(position))[0],
            agent_y,
            _quantize_position(tuple(position))[2],
        )
        for position in scene_positions.values()
    }
    floor_points.add(
        (
            _quantize_position(agent_position)[0],
            agent_y,
            _quantize_position(agent_position)[2],
        )
    )
    x_values = sorted({point[0] for point in floor_points})
    z_values = sorted({point[2] for point in floor_points})
    if not x_values or not z_values:
        raise ValueError("Unable to derive navigation bounds from scene positions.")

    x_min, x_max = min(x_values), max(x_values)
    z_min, z_max = min(z_values), max(z_values)
    nav_graph: NavGraph = {}
    x_steps = int(round((x_max - x_min) / grid_size))
    z_steps = int(round((z_max - z_min) / grid_size))
    for x_index in range(x_steps + 1):
        for z_index in range(z_steps + 1):
            x_coord = round(x_min + (x_index * grid_size), 3)
            z_coord = round(z_min + (z_index * grid_size), 3)
            node = (x_coord, agent_y, z_coord)
            nav_graph[node] = []

    for node in list(nav_graph.keys()):
        x_coord, y_coord, z_coord = node
        candidate_neighbors = [
            (round(x_coord + grid_size, 3), y_coord, z_coord),
            (round(x_coord - grid_size, 3), y_coord, z_coord),
            (x_coord, y_coord, round(z_coord + grid_size, 3)),
            (x_coord, y_coord, round(z_coord - grid_size, 3)),
        ]
        nav_graph[node] = [
            neighbor for neighbor in candidate_neighbors if neighbor in nav_graph
        ]
    return nav_graph


def _serialize_nav_graph(nav_graph: NavGraph) -> dict[str, Any]:
    """Convert a navigation graph into a stable JSON payload."""

    serialized_nodes = [
        {
            "position": [float(coord) for coord in node],
            "neighbors": [
                [float(coord) for coord in neighbor] for neighbor in neighbors
            ],
        }
        for node, neighbors in sorted(nav_graph.items())
    ]
    return {
        "schema_version": NAV_GRAPH_CACHE_SCHEMA_VERSION,
        "grid_size": float(constants.GRID_SIZE),
        "nodes": serialized_nodes,
    }


def _deserialize_nav_graph(payload: Mapping[str, Any]) -> NavGraph:
    """Restore a navigation graph from its cached JSON representation."""

    if payload.get("schema_version") != NAV_GRAPH_CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported nav graph cache schema version.")
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError("Nav graph cache payload must contain a node list.")
    nav_graph: NavGraph = {}
    for raw_node in raw_nodes:
        if not isinstance(raw_node, Mapping):
            raise ValueError("Nav graph node entries must be mappings.")
        position = tuple(float(coord) for coord in raw_node["position"])
        neighbors = [
            tuple(float(coord) for coord in neighbor)
            for neighbor in raw_node.get("neighbors", [])
        ]
        nav_graph[position] = neighbors
    return nav_graph


def _build_nav_graph_cache_path(scene_name: str) -> Path:
    """Return the cache file path for an AI2-THOR navigation graph."""

    return constants.NAV_GRAPH_CACHE_PATH / f"{scene_name}.json"


def _load_cached_ai2thor_nav_graph(scene_name: str) -> Optional[NavGraph]:
    """Load a cached AI2-THOR navigation graph when available."""

    cache_path = _build_nav_graph_cache_path(scene_name)
    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    return _deserialize_nav_graph(payload)


def _save_cached_ai2thor_nav_graph(scene_name: str, nav_graph: NavGraph) -> None:
    """Persist an AI2-THOR navigation graph cache to disk."""

    cache_path = _build_nav_graph_cache_path(scene_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialize_nav_graph(nav_graph)
    payload["scene_name"] = scene_name
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_ai2thor_nav_graph(scene_name: str) -> NavGraph:
    """Load the navigation graph from cache or directly from AI2-THOR.

    Args:
        scene_name: AI2-THOR scene identifier such as ``FloorPlan13``.

    Returns:
        Navigation graph extracted from the controller.

    Raises:
        RuntimeError: If the controller cannot be initialized or the graph cannot
            be loaded.
    """

    from ithor.utils.math_utils import load_navigation_graph
    from src.simulation.runner_ai2thor import init_ai2thor_controller

    try:
        cached_nav_graph = _load_cached_ai2thor_nav_graph(scene_name)
    except (json.JSONDecodeError, OSError, ValueError):
        cached_nav_graph = None
    if cached_nav_graph is not None:
        return cached_nav_graph

    controller = None
    try:
        controller = init_ai2thor_controller(scene_name)
        nav_graph = load_navigation_graph(controller)
        _save_cached_ai2thor_nav_graph(scene_name, nav_graph)
        return nav_graph
    except Exception as exc:  # pragma: no cover - exercised via integration usage.
        raise RuntimeError(
            f"Failed to load AI2-THOR navigation graph for scene '{scene_name}'."
        ) from exc
    finally:
        if controller is not None:
            try:
                controller.stop()
            except Exception:
                pass


def resolve_nav_graph(
    config: ExperimentConfig,
    scene_positions: Mapping[str, Position],
) -> NavGraph:
    """Resolve the navigation graph source for offline planning.

    Args:
        config: Experiment configuration.
        scene_positions: Preloaded scene positions used by the synthetic graph path.

    Returns:
        Navigation graph chosen by ``config.nav_graph_source``.

    Raises:
        ValueError: If the nav graph source is unsupported.
    """

    if config.nav_graph_source == "synthetic_grid":
        return build_grid_nav_graph(scene_positions)
    if config.nav_graph_source == "ai2thor_controller":
        return load_ai2thor_nav_graph(config.scene)
    raise ValueError(f"Unsupported nav_graph_source: {config.nav_graph_source}")


def resolve_cases(config: ExperimentConfig) -> list[str]:
    """Resolve case names for the current experiment.

    Args:
        config: Experiment configuration.

    Returns:
        Ordered list of cases to run.
    """

    if config.cases:
        return list(config.cases)
    return [config.case]


def resolve_instructions(
    config: ExperimentConfig,
    case_name: Optional[str] = None,
) -> list[str]:
    """Resolve instruction file names for a case.

    Args:
        config: Experiment configuration.
        case_name: Optional explicit case override.

    Returns:
        Sorted instruction file names for the selected case.
    """

    resolved_case = case_name or config.case

    if config.instructions:
        return list(config.instructions)

    case_dir = (
        constants.ASSETS_PATH
        / "tasks"
        / config.task_folder_name
        / resolved_case
        / config.scene
    )
    if not case_dir.exists():
        raise FileNotFoundError(f"Task case directory not found: {case_dir}")

    task_files = sorted(path.name for path in case_dir.glob("*.json"))
    if not task_files:
        raise FileNotFoundError(f"No task files found under: {case_dir}")
    return task_files[: config.max_tasks]


def build_selected_instruction_map(config: ExperimentConfig) -> dict[str, list[str]]:
    """Build the selected instruction set for every requested case.

    Args:
        config: Experiment configuration.

    Returns:
        Mapping from case name to selected instruction file names.
    """

    return {
        case_name: resolve_instructions(config, case_name)
        for case_name in resolve_cases(config)
    }


def build_experiment_tasks(config: ExperimentConfig) -> list[ExperimentTask]:
    """Build the cartesian product of instructions and beam settings."""

    tasks: list[ExperimentTask] = []
    effective_beam_bound = _resolve_effective_beam_bounds(config)
    for case_name in resolve_cases(config):
        for task_index, instruction in enumerate(resolve_instructions(config, case_name)):
            for beam_width, beam_depth in effective_beam_bound:
                tasks.append(
                    ExperimentTask(
                        instruction=instruction,
                        beam_width=int(beam_width),
                        beam_depth=int(beam_depth),
                        task_index=task_index,
                        case=case_name,
                    )
                )
    return tasks


def _copy_schedule_to_sim_fields(state: SchedulerState) -> None:
    """Mirror schedule timestamps into simulation fields for offline evaluation."""

    for entry in state.completed_entries:
        if entry.sim_start_time == float("inf"):
            entry.sim_start_time = entry.schedule_start_time
        if entry.sim_end_time == float("inf"):
            entry.sim_end_time = entry.schedule_end_time
        if entry.execution_status == TaskExecutionStatus.NOT_EXECUTED:
            entry.execution_status = TaskExecutionStatus.SUCCESS
        if entry.sim_nav_time is None:
            entry.sim_nav_time = entry.schedule_nav_time


def _serialize_schedule_steps(
    completed_entries: Sequence[CompletedEntry],
    *,
    monitored_subtasks: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Serialize scheduled entries with explicit start/end timing fields."""

    monitored_subtasks = monitored_subtasks or {}
    serialized_steps: list[dict[str, Any]] = []
    normalized_entries: list[CompletedEntry] = []
    for entry in completed_entries:
        if entry.sim_start_time == float("inf"):
            entry.sim_start_time = entry.schedule_start_time
        if entry.sim_end_time == float("inf"):
            entry.sim_end_time = entry.schedule_end_time
        if entry.sim_nav_time is None:
            entry.sim_nav_time = entry.schedule_nav_time
        normalized_entries.append(entry)
    for entry_payload, entry in zip(
        serialize_completed_entries(list(normalized_entries)),
        normalized_entries,
    ):
        if entry.subtask.subtask_type == "Init":
            continue
        serialized_steps.append(
            {
                "subtask_name": entry_payload["subtask_name"],
                "subtask_type": entry.subtask.subtask_type,
                "start_time_scheduled": entry_payload["start_time_scheduled"],
                "end_time_scheduled": entry_payload["end_time_scheduled"],
                "start_time_simulation": entry_payload["start_time_simulation"],
                "end_time_simulation": entry_payload["end_time_simulation"],
                "schedule_nav_time": (
                    round(float(entry.schedule_nav_time), 2)
                    if entry.schedule_nav_time is not None
                    else None
                ),
                "execution_status": entry_payload["execution_status"],
                "monitored_subtask": monitored_subtasks.get(entry.subtask.name),
            }
        )
    return serialized_steps


def _is_tcsr_perfect(schedule_tcsr: float | None) -> bool | None:
    """Return whether the schedule TCSR is exactly satisfied."""

    if schedule_tcsr is None:
        return None
    return abs(float(schedule_tcsr) - 1.0) <= 1e-9


def _count_monitoring_events_by_interval(
    completed_entries: Sequence[CompletedEntry],
) -> dict[str, int]:
    """Count executed monitoring actions keyed by their critical-end target name."""

    from src.utils.common import extract_monitoring_target_name

    counts: dict[str, int] = {}
    for entry in completed_entries:
        if entry.subtask.subtask_type != "Monitor":
            continue
        try:
            target_name = extract_monitoring_target_name(entry.subtask.name)
        except ValueError:
            continue
        counts[target_name] = counts.get(target_name, 0) + 1
    return counts


def _build_result_meta_data(
    config: ExperimentConfig,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the hybrid legacy-style metadata block for one run result."""

    init_prior_mean, init_prior_variance = INIT_PRIOR_CONFIG_MAP[config.init_prior_config]
    baseline_name = (
        "particle_filter"
        if config.approach == "bayesian"
        and config.belief_update_method == "particle_filter"
        else config.approach
    )
    payload: dict[str, Any] = {
        "suite_name": config.suite_name,
        "baseline_name": baseline_name,
        "approach": config.approach,
        "ablation_config": config.ablation_config,
        "init_prior_config": config.init_prior_config,
        "init_prior_name": init_prior_mean,
        "init_prior_mean": init_prior_mean,
        "init_prior_variance": init_prior_variance,
        "beam_width": result.get("beam_width"),
        "beam_depth": result.get("beam_depth"),
        "disable_monitoring": not ABLATION_CONFIG_MONITORING[config.ablation_config],
        "nav_graph_source": config.nav_graph_source,
        "belief_update_method": config.belief_update_method,
        "gt_distribution": config.gt_distribution,
        "gt_seed": config.gt_seed,
        "particle_distribution": result.get("particle_distribution"),
        "particle_likelihood_family": result.get("particle_likelihood_family"),
        "factor_alpha": config.factor_alpha,
        "eta": config.eta,
        "monitoring_budget_per_critical": _resolve_monitoring_budget_label(
            config.max_monitoring_per_critical_interval
        ),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _build_hybrid_result_row(
    config: ExperimentConfig,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Wrap one rollout result in the hybrid single-run schema."""

    plans = [dict(step) for step in result["steps"]]
    payload = build_hybrid_single_run_payload(
        meta_data=_build_result_meta_data(config, result),
        saved_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
        approach=(
            config.approach
            if config.approach == "oracle"
            else f"{config.approach}__{config.ablation_config}"
        ),
        scene_name=config.scene,
        plans=plans,
        computation_time=(
            float(result["total_compute_time"])
            if result.get("total_compute_time") is not None
            else None
        ),
        scheduler_makespan=(
            float(result["final_schedule_time"])
            if result.get("final_schedule_time") is not None
            else None
        ),
        timing_success_rate_sched=(
            float(result["schedule_tcsr"])
            if result.get("schedule_tcsr") is not None
            else None
        ),
        detail_log=result.get("timing_detail", {}),
        extra_fields={
            **dict(result),
            "case_name": result.get("case"),
            "instruction_name": result.get("instruction"),
            "scene_name": config.scene,
            "plans": plans,
            "steps": plans,
            "simulation_makespan": result.get("final_schedule_time"),
            "timing_success_rate_sim": result.get("schedule_tcsr"),
            "timing_success_rate_sched": result.get("schedule_tcsr"),
        },
    )
    return payload


def _build_persisted_single_run_payload(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Collapse an in-memory hybrid row into a slim persisted payload."""

    payload: dict[str, Any] = {
        "meta_data": dict(row["meta_data"]),
        "saved_time": row["saved_time"],
        "approach": row["approach"],
        "scene_name": row["scene_name"],
        "plans": [dict(plan) for plan in row["plans"]],
        "computation_time": row["computation_time"],
        "scheduler_makespan": row["scheduler_makespan"],
        "timing_success_rate_sched": row["timing_success_rate_sched"],
        "detail_log": dict(row["detail_log"]),
    }
    optional_keys = [
        "case",
        "instruction",
        "completed",
        "abort_reason",
        "optimal_sequence",
        "optimal_schedule_time",
        "exact",
        "timeout_hit",
        "search_nodes",
        "pruned_nodes",
        "idle_advances",
        "actual_monitor_count_total",
        "actual_monitor_count_by_interval",
        "initial_planning_time",
        "replanning_time_total",
        "replanning_time_mean",
        "replanning_time_p95",
        "belief_update_count",
        "belief_update_time_total",
        "belief_update_time_mean",
        "monitoring_budget_per_critical",
    ]
    for key in optional_keys:
        if key in row:
            payload[key] = row[key]
    return payload


def _apply_runtime_overrides(config: ExperimentConfig) -> dict[str, Any]:
    """Apply temporary constants overrides and return previous values."""

    init_prior_mean, init_prior_variance = INIT_PRIOR_CONFIG_MAP[config.init_prior_config]
    previous_values = {
        "task_path": constants.TASK_PATH,
        "monitoring_enabled": constants.MONITORING_ENABLED,
        "init_prior_mean": constants.INIT_PRIOR_MEAN,
        "init_prior_variance": constants.INIT_PRIOR_VARIANCE,
        "factor_alpha": constants.FACTOR_ALPHA,
        "eta": constants.BAYESIAN_THRESHOLD_PROBABILITY,
    }
    constants.set_task_path(constants.ASSETS_PATH / "tasks" / config.task_folder_name)
    constants.set_monitoring_enabled(
        ABLATION_CONFIG_MONITORING[config.ablation_config]
    )
    constants.set_init_prior_mean(init_prior_mean)
    constants.set_init_prior_variance(init_prior_variance)
    if config.factor_alpha is not None:
        constants.set_factor_alpha(config.factor_alpha)
    if config.eta is not None:
        constants.set_bayesian_threshold_probability(config.eta)
    return previous_values


def _restore_runtime_overrides(previous_values: Mapping[str, Any]) -> None:
    """Restore mutable runtime constants after an experiment."""

    constants.set_task_path(previous_values["task_path"])
    constants.set_monitoring_enabled(previous_values["monitoring_enabled"])
    constants.set_init_prior_mean(previous_values["init_prior_mean"])
    constants.set_init_prior_variance(previous_values["init_prior_variance"])
    constants.set_factor_alpha(previous_values["factor_alpha"])
    constants.set_bayesian_threshold_probability(previous_values["eta"])


def run_single_rollout(
    config: ExperimentConfig,
    task: ExperimentTask,
    *,
    nav_graph: NavGraph,
    scene_positions: Mapping[str, Position],
) -> dict[str, Any]:
    """Execute one offline rollout for a task/beam combination."""

    if config.approach == "edf":
        return run_single_rollout_edf(
            config,
            task,
            nav_graph=nav_graph,
            scene_positions=scene_positions,
        )
    if config.approach == "cpm":
        return run_single_rollout_cpm(
            config,
            task,
            nav_graph=nav_graph,
            scene_positions=scene_positions,
        )
    if config.approach != "bayesian":
        raise ValueError(f"Unsupported approach for rollout: {config.approach}")

    task_data = load_task_data_from_sampled_set(
        task.case or config.case,
        config.scene,
        task.instruction,
    )
    subtasks, constraints, bayesian_load = TaskUtil.build_tasks_and_constraints(
        task_data,
        scene_file_name=f"{config.scene}_physics_environment.json",
    )
    current_state = TaskUtil.get_init_state(
        subtasks,
        constraints,
        copy.deepcopy(dict(scene_positions)),
    )

    action_handler = ActionHandler(nav_graph, real_world_mode=False)
    constraint_handler = ConstraintHandler(action_handler)
    observation_model = create_observation_model(
        config.observation_mode,
        random_seed=config.gt_seed,
    )
    particle_distribution = _resolve_particle_distribution(config)
    belief_store, monitoring_policy, belief_updater = create_monitoring_backend(
        config.belief_update_method,
        bayesian_load,
        particle_distribution=particle_distribution,
        particle_likelihood_family=config.particle_likelihood_family,
        observation_model=observation_model,
        random_seed=config.gt_seed,
    )
    ground_truth_store = create_ground_truth_store(
        constants.CRITICAL_OBJECT_GROUND_TRUTH,
        distribution=config.gt_distribution,
        random_seed=config.gt_seed,
    )
    ground_truth_store.ensure_intervals(bayesian_load)
    agent = Agent(
        constraint_handler,
        bayesian_load,
        belief_updater=belief_updater,
        belief_store=belief_store,
        ground_truth_store=ground_truth_store,
    )
    scheduler = Scheduler(
        action_handler=action_handler,
        constraint_handler=constraint_handler,
        heuristic_manager=HeuristicManager(action_handler),
        monitoring_policy=monitoring_policy,
        beam_width=task.beam_width,
        simulation_depth=task.beam_depth,
        max_monitoring_per_critical_interval=(
            config.max_monitoring_per_critical_interval
        ),
    )

    total_compute_time = 0.0
    replanning_durations: list[float] = []
    belief_update_durations: list[float] = []
    chunk_sizes: list[int] = []
    replanning_count = 0
    aborted = False
    abort_reason = ""
    monitored_subtasks: dict[str, Any] = {}
    while current_state.remaining_subtasks:
        next_state, compute_elapsed_time = scheduler.get_next_state(current_state)
        planned_states = [] if next_state is None else [next_state]

        total_compute_time += compute_elapsed_time
        replanning_durations.append(float(compute_elapsed_time))
        replanning_count += 1
        if not planned_states:
            aborted = True
            abort_reason = "no_feasible_solution"
            break

        executed_in_chunk = 0
        for next_state in planned_states:
            _copy_schedule_to_sim_fields(next_state)
            monitored_subtask: Optional[dict[str, Any]] = None
            if (
                ABLATION_CONFIG_MONITORING[config.ablation_config]
                and next_state.subtask.subtask_type == "Monitor"
            ):
                belief_update_started_at = perf_counter()
                next_state, monitored_subtask = agent.update_monitoring_belief(next_state)
                belief_update_durations.append(
                    perf_counter() - belief_update_started_at
                )
                _copy_schedule_to_sim_fields(next_state)

            if monitored_subtask is not None:
                monitored_subtasks[next_state.subtask.name] = monitored_subtask
            current_state = next_state
            executed_in_chunk += 1
            if len(current_state.completed_entries) > 500:
                aborted = True
                abort_reason = "step_guard_exceeded"
                break
        chunk_sizes.append(executed_in_chunk)
        if aborted:
            break

    _copy_schedule_to_sim_fields(current_state)
    _, schedule_tcsr, detail_log = calculate_timing_success_rate(
        current_state.constraints,
        current_state.completed_entries,
        ground_truth_overrides=ground_truth_store.as_dict(),
    )
    steps = _serialize_schedule_steps(
        current_state.completed_entries,
        monitored_subtasks=monitored_subtasks,
    )
    wait_count = sum(1 for step in steps if step["subtask_type"] == "WAIT")
    monitor_count = sum(1 for step in steps if step["subtask_type"] == "Monitor")
    action_count = sum(
        1 for step in steps if step["subtask_type"] not in {"WAIT", "Monitor"}
    )
    monitor_count_by_interval = _count_monitoring_events_by_interval(
        current_state.completed_entries
    )
    avg_committed_steps = (
        sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0.0
    )
    monitoring_budget_label = _resolve_monitoring_budget_label(
        config.max_monitoring_per_critical_interval
    )

    result = {
        "case": task.case or config.case,
        "instruction": task.instruction,
        "beam_width": task.beam_width,
        "beam_depth": task.beam_depth,
        "completed": not aborted and not current_state.remaining_subtasks,
        "abort_reason": abort_reason,
        "total_compute_time": total_compute_time,
        "final_schedule_time": current_state.current_time,
        "schedule_tcsr": schedule_tcsr,
        "tcsr_is_one": _is_tcsr_perfect(schedule_tcsr),
        "action_count": action_count,
        "wait_count": wait_count,
        "monitor_count": monitor_count,
        "actual_monitor_count_total": monitor_count,
        "actual_monitor_count_by_interval": monitor_count_by_interval,
        "replanning_count": replanning_count,
        "initial_planning_time": (
            replanning_durations[0] if replanning_durations else None
        ),
        "replanning_time_total": total_compute_time,
        "replanning_time_mean": _mean_or_none(replanning_durations),
        "replanning_time_p95": _p95_or_none(replanning_durations),
        "belief_update_count": len(belief_update_durations),
        "belief_update_time_total": (
            sum(belief_update_durations) if belief_update_durations else 0.0
        ),
        "belief_update_time_mean": _mean_or_none(belief_update_durations),
        "avg_committed_steps_per_replan": avg_committed_steps,
        "steps": steps,
        "timing_detail": detail_log,
        "ground_truth_intervals": ground_truth_store.as_dict(),
        "particle_distribution": particle_distribution,
        "particle_likelihood_family": config.particle_likelihood_family,
        "monitoring_budget_per_critical": monitoring_budget_label,
    }
    return _build_hybrid_result_row(config, result)


def _simulate_subtask_actions(
    current_state: SchedulerState,
    subtask: Subtask,
    action_handler: ActionHandler,
) -> Optional[ActionResult]:
    """Simulate a subtask from the current state and return the final action log."""

    primitive_actions = subtask.execution.primitive_actions if subtask.execution else None
    if not primitive_actions:
        return None
    temp_node = SimulationNode(
        heuristic_cost=0.0,
        depth=0,
        tie_breaker=0,
        parent_node=None,
        state=current_state,
        risk_level=0,
    )
    return action_handler.get_actions_info(temp_node, primitive_actions)


def _edf_compute_nav_time(
    subtask: Subtask,
    current_state: SchedulerState,
    action_handler: ActionHandler,
) -> tuple[float, Mapping[str, Position]]:
    """Compute the first navigation duration exactly like the EDF baseline."""

    primitive_actions = subtask.execution.primitive_actions if subtask.execution else None
    if not primitive_actions:
        return 0.0, current_state.scene_positions
    first_action = primitive_actions[0]
    if not first_action.startswith("NAVIGATE_TO"):
        return 0.0, current_state.scene_positions
    nav_info = action_handler.get_actions_info(
        SimulationNode(
            heuristic_cost=0.0,
            depth=0,
            tie_breaker=0,
            parent_node=None,
            state=current_state,
            risk_level=0,
        ),
        [first_action],
    )
    if nav_info is None:
        return 0.0, current_state.scene_positions
    return float(nav_info.cumulative_time), nav_info.scene_positions


def _edf_offline_subtask_execution(
    subtask: Subtask,
    current_state: SchedulerState,
    action_handler: ActionHandler,
) -> Optional[ActionResult]:
    """Simulate the full subtask exactly like the EDF baseline."""

    return _simulate_subtask_actions(current_state, subtask, action_handler)


def _edf_update_state(
    current_state: SchedulerState,
    next_subtask: Subtask,
    exec_info: ActionResult,
    nav_time: Optional[float] = None,
) -> SchedulerState:
    """Append a completed entry exactly like the EDF baseline."""

    subtask_duration = float(exec_info.cumulative_time)
    subtask_entry = CompletedEntry(
        subtask=next_subtask,
        schedule_start_time=current_state.current_time,
        schedule_end_time=current_state.current_time + subtask_duration,
        schedule_nav_time=nav_time,
        execution_status=TaskExecutionStatus.SUCCESS,
    )
    new_completed = current_state.completed_entries + [subtask_entry]
    new_remaining = [
        subtask
        for subtask in current_state.remaining_subtasks
        if subtask.name != next_subtask.name
    ]
    held_object = exec_info.held_object if exec_info.held_object is not None else current_state.held_object
    return SchedulerState(
        subtask=next_subtask,
        completed_entries=new_completed,
        remaining_subtasks=new_remaining,
        constraints=current_state.constraints,
        current_time=current_state.current_time + subtask_duration,
        scene_positions=copy.deepcopy(exec_info.scene_positions),
        held_object=held_object,
        agent_location=current_state.agent_location,
    )


def _edf_is_executable(subtask: Subtask, current_state: SchedulerState) -> bool:
    """Mirror the baseline executable check for EDF."""

    incoming = list(current_state.constraints.in_edges(subtask.name))
    if incoming:
        predecessor_name = incoming[0][0]
        completed = {entry.subtask.name for entry in current_state.completed_entries}
        return predecessor_name in completed
    return True


def _edf_nav_and_wait_during_interval(
    current_state: SchedulerState,
    interval: float,
    next_subtask: Subtask,
    action_handler: ActionHandler,
) -> tuple[SchedulerState, bool]:
    """Insert NAVIGATE and WAIT entries exactly like the EDF baseline."""

    entries: list[CompletedEntry] = []
    current_time = current_state.current_time
    primitive_actions = next_subtask.execution.primitive_actions if next_subtask.execution else None
    if not primitive_actions:
        return current_state, False
    first_action = primitive_actions[0]
    if not first_action.startswith("NAVIGATE_TO"):
        return current_state, False

    nav_time, nav_positions = _edf_compute_nav_time(
        next_subtask,
        current_state,
        action_handler,
    )
    if 0.0 < nav_time <= interval:
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
            execution_status=TaskExecutionStatus.SUCCESS,
        )
        entries.append(nav_entry)
        current_time += nav_time

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
            execution_status=TaskExecutionStatus.SUCCESS,
        )
        entries.append(wait_entry)
        return (
            SchedulerState(
                subtask=current_state.subtask,
                completed_entries=current_state.completed_entries + entries,
                remaining_subtasks=current_state.remaining_subtasks,
                constraints=current_state.constraints,
                current_time=current_time + wait_time,
                scene_positions=copy.deepcopy(dict(nav_positions)),
                held_object=current_state.held_object,
                agent_location=current_state.agent_location,
            ),
            True,
        )
    return current_state, False


def _edf_compute_deadline(
    subtask: Subtask,
    current_state: SchedulerState,
    nav_time: float,
    execution_time: float,
) -> float:
    """Compute the EDF release target using interaction-start semantics."""

    constraints = current_state.constraints
    current_time = current_state.current_time
    incoming_edges = list(constraints.in_edges(subtask.name, data=True))
    deadline = current_time + execution_time
    if incoming_edges:
        critical_edges = [
            (source_name, target_name, data)
            for source_name, target_name, data in incoming_edges
            if data.get("info", {}).get("IsCritical", False)
        ]
        if critical_edges:
            critical_deadlines = []
            for source_name, _, data in critical_edges:
                predecessor_end_time = next(
                    (
                        entry.schedule_end_time
                        for entry in current_state.completed_entries
                        if entry.subtask.name == source_name
                    ),
                    current_time,
                )
                critical_deadlines.append(
                    predecessor_end_time + float(data["info"]["Interval"])
                )
            deadline = max(critical_deadlines)
        else:
            non_critical_deadlines = []
            for source_name, _, data in incoming_edges:
                predecessor_end_time = next(
                    (
                        entry.schedule_end_time
                        for entry in current_state.completed_entries
                        if entry.subtask.name == source_name
                    ),
                    current_time,
                )
                non_critical_deadlines.append(
                    predecessor_end_time + float(data["info"]["Interval"])
                )
            deadline = max(non_critical_deadlines)
    return deadline


def _edf_select_next_subtask(
    current_state: SchedulerState,
    action_handler: ActionHandler,
) -> Optional[Subtask]:
    """Select the next subtask exactly like the EDF baseline."""

    queue: list[tuple[float, int, Subtask]] = []
    for subtask in current_state.remaining_subtasks:
        if not _edf_is_executable(subtask, current_state):
            continue
        nav_time, _ = _edf_compute_nav_time(subtask, current_state, action_handler)
        exec_info = _edf_offline_subtask_execution(subtask, current_state, action_handler)
        if exec_info is None:
            continue
        execution_time = float(exec_info.cumulative_time)
        deadline = _edf_compute_deadline(
            subtask,
            current_state,
            nav_time,
            execution_time,
        )
        setattr(subtask, "deadline", deadline)
        heapq.heappush(queue, (deadline, len(queue), subtask))
    if not queue:
        return None
    chosen_subtask = heapq.heappop(queue)[2]
    chosen_exec_info = _edf_offline_subtask_execution(
        chosen_subtask,
        current_state,
        action_handler,
    )
    if chosen_exec_info is not None:
        chosen_subtask.duration.interval = float(chosen_exec_info.cumulative_time)
    return chosen_subtask


def _edf_update(
    current_state: SchedulerState,
    next_subtask: Subtask,
    action_handler: ActionHandler,
) -> SchedulerState:
    """Update EDF state exactly like the baseline implementation."""

    current_time = current_state.current_time
    incoming_edges = list(current_state.constraints.in_edges(next_subtask.name, data=True))
    designated_start = current_time
    nav_time, _ = _edf_compute_nav_time(next_subtask, current_state, action_handler)
    if incoming_edges:
        designated_start = float(getattr(next_subtask, "deadline", current_time))
        if current_time < designated_start:
            interval = designated_start - current_time
            current_state, inserted_nav_and_wait = _edf_nav_and_wait_during_interval(
                current_state,
                interval,
                next_subtask,
                action_handler,
            )
            if inserted_nav_and_wait:
                nav_time = 0.0
    exec_info = _edf_offline_subtask_execution(next_subtask, current_state, action_handler)
    if exec_info is None:
        raise ValueError(f"Failed to simulate EDF subtask: {next_subtask.name}")
    return _edf_update_state(current_state, next_subtask, exec_info, nav_time)


def run_single_rollout_edf(
    config: ExperimentConfig,
    task: ExperimentTask,
    *,
    nav_graph: NavGraph,
    scene_positions: Mapping[str, Position],
) -> dict[str, Any]:
    """Execute one offline rollout using the EDF baseline policy."""

    task_data = load_task_data_from_sampled_set(
        task.case or config.case,
        config.scene,
        task.instruction,
    )
    subtasks, constraints, _ = TaskUtil.build_tasks_and_constraints(
        task_data,
        scene_file_name=f"{config.scene}_physics_environment.json",
    )
    current_state = TaskUtil.get_init_state(
        subtasks,
        constraints,
        copy.deepcopy(dict(scene_positions)),
    )
    action_handler = ActionHandler(nav_graph, real_world_mode=False)

    total_compute_time = 0.0
    aborted = False
    abort_reason = ""

    while current_state.remaining_subtasks:
        selection_started_at = perf_counter()
        selected = _edf_select_next_subtask(current_state, action_handler)
        total_compute_time += perf_counter() - selection_started_at
        if selected is None:
            aborted = True
            abort_reason = "no_feasible_solution"
            break

        current_state = _edf_update(current_state, selected, action_handler)
        if len(current_state.completed_entries) > 500:
            aborted = True
            abort_reason = "step_guard_exceeded"
            break

    _copy_schedule_to_sim_fields(current_state)
    _, schedule_tcsr, detail_log = calculate_timing_success_rate(
        current_state.constraints,
        current_state.completed_entries,
    )
    steps = _serialize_schedule_steps(current_state.completed_entries)
    wait_count = sum(1 for entry in current_state.completed_entries if entry.subtask.subtask_type == "WAIT")
    action_count = sum(
        1
        for entry in current_state.completed_entries
        if entry.subtask.subtask_type not in {"WAIT", "Monitor", "Init"}
    )
    result = {
        "case": task.case or config.case,
        "instruction": task.instruction,
        "beam_width": task.beam_width,
        "beam_depth": task.beam_depth,
        "completed": not aborted and not current_state.remaining_subtasks,
        "abort_reason": abort_reason,
        "total_compute_time": total_compute_time,
        "final_schedule_time": current_state.current_time,
        "schedule_tcsr": schedule_tcsr,
        "tcsr_is_one": _is_tcsr_perfect(schedule_tcsr),
        "action_count": action_count,
        "wait_count": wait_count,
        "monitor_count": 0,
        "replanning_count": 1 if steps else 0,
        "avg_committed_steps_per_replan": float(len(steps)) if steps else 0.0,
        "steps": steps,
        "timing_detail": detail_log,
        "ground_truth_intervals": {},
    }
    return _build_hybrid_result_row(config, result)


def run_single_rollout_cpm(
    config: ExperimentConfig,
    task: ExperimentTask,
    *,
    nav_graph: NavGraph,
    scene_positions: Mapping[str, Position],
) -> dict[str, Any]:
    """Execute one offline rollout using the CPM baseline policy."""

    from src.baselines import cpm as cpm_baseline

    task_data = load_task_data_from_sampled_set(
        task.case or config.case,
        config.scene,
        task.instruction,
    )
    subtasks, constraints, _ = TaskUtil.build_tasks_and_constraints(
        task_data,
        scene_file_name=f"{config.scene}_physics_environment.json",
        enable_decomposition=True,
    )
    cpm_baseline.constraints = constraints
    current_state = TaskUtil.get_init_state(
        subtasks,
        constraints,
        copy.deepcopy(dict(scene_positions)),
    )
    action_handler = ActionHandler(nav_graph, real_world_mode=False)

    subtasks_without_edge = [
        subtask
        for subtask in subtasks
        if all(
            subtask.name != source_name and subtask.name != target_name
            for source_name, target_name in list(constraints.edges)
        )
    ]

    planning_started_at = perf_counter()
    critical_path = cpm_baseline.find_critical_path(subtasks)
    result_schedule = cpm_baseline.get_final_entries(
        critical_path,
        subtasks_without_edge,
        current_state,
        action_handler,
    )
    total_compute_time = perf_counter() - planning_started_at

    current_state = TaskUtil.get_init_state(
        subtasks,
        constraints,
        copy.deepcopy(dict(scene_positions)),
    )
    for entry in result_schedule:
        exec_info = cpm_baseline.offline_subtask_execution(
            current_state,
            entry.subtask,
            action_handler,
        )
        current_state = cpm_baseline.update_state(current_state, entry.subtask, exec_info)

    _copy_schedule_to_sim_fields(current_state)
    _, schedule_tcsr, detail_log = calculate_timing_success_rate(
        current_state.constraints,
        current_state.completed_entries,
    )
    steps = _serialize_schedule_steps(current_state.completed_entries)
    wait_count = sum(1 for entry in current_state.completed_entries if entry.subtask.subtask_type == "WAIT")
    action_count = sum(
        1 for entry in current_state.completed_entries if entry.subtask.subtask_type not in {"WAIT", "Monitor"}
    )
    result = {
        "case": task.case or config.case,
        "instruction": task.instruction,
        "beam_width": task.beam_width,
        "beam_depth": task.beam_depth,
        "completed": True,
        "abort_reason": "",
        "total_compute_time": total_compute_time,
        "final_schedule_time": current_state.current_time,
        "schedule_tcsr": schedule_tcsr,
        "tcsr_is_one": _is_tcsr_perfect(schedule_tcsr),
        "action_count": action_count,
        "wait_count": wait_count,
        "monitor_count": 0,
        "replanning_count": 1 if steps else 0,
        "avg_committed_steps_per_replan": float(len(steps)) if steps else 0.0,
        "steps": steps,
        "timing_detail": detail_log,
        "ground_truth_intervals": {},
    }
    return _build_hybrid_result_row(config, result)


def summarize_results(results: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate per-run results into a setting-level summary."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        beam_key = f"w{int(result['beam_width'])}_d{int(result['beam_depth'])}"
        grouped.setdefault(beam_key, []).append(result)

    summary: dict[str, dict[str, Any]] = {}
    for beam_key, rows in grouped.items():
        completed_rows = [row for row in rows if row["completed"]]
        schedule_times = [float(row["final_schedule_time"]) for row in completed_rows]
        compute_times = [float(row["total_compute_time"]) for row in rows]
        tcsr_values = [
            float(row["schedule_tcsr"])
            for row in completed_rows
            if row["schedule_tcsr"] is not None
        ]
        wait_counts = [int(row["wait_count"]) for row in rows]
        monitor_counts = [int(row["monitor_count"]) for row in rows]
        replanning_counts = [int(row["replanning_count"]) for row in rows]
        committed_steps = [
            float(row["avg_committed_steps_per_replan"]) for row in rows
        ]
        summary[beam_key] = {
            "num_runs": len(rows),
            "completed_runs": len(completed_rows),
            "avg_schedule_time": mean(schedule_times) if schedule_times else None,
            "median_schedule_time": median(schedule_times) if schedule_times else None,
            "avg_compute_time": mean(compute_times) if compute_times else None,
            "avg_schedule_tcsr": mean(tcsr_values) if tcsr_values else None,
            "avg_wait_count": mean(wait_counts) if wait_counts else None,
            "avg_monitor_count": mean(monitor_counts) if monitor_counts else None,
            "avg_replanning_count": (
                mean(replanning_counts) if replanning_counts else None
            ),
            "avg_committed_steps_per_replan": (
                mean(committed_steps) if committed_steps else None
            ),
        }
    return summary


def summarize_results_by_case_setting(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate per-run results at the case-setting level.

    Args:
        results: Rollout result rows.

    Returns:
        Mapping keyed by ``case:wX_dY`` to aggregated metrics.
    """

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        beam_key = f"w{int(result['beam_width'])}_d{int(result['beam_depth'])}"
        case_key = f"{result['case']}:{beam_key}"
        grouped.setdefault(case_key, []).append(result)

    summary: dict[str, dict[str, Any]] = {}
    for case_key, rows in grouped.items():
        completed_rows = [row for row in rows if row["completed"]]
        schedule_times = [float(row["final_schedule_time"]) for row in completed_rows]
        compute_times = [float(row["total_compute_time"]) for row in rows]
        tcsr_values = [
            float(row["schedule_tcsr"])
            for row in completed_rows
            if row["schedule_tcsr"] is not None
        ]
        wait_counts = [int(row["wait_count"]) for row in rows]
        monitor_counts = [int(row["monitor_count"]) for row in rows]
        replanning_counts = [int(row["replanning_count"]) for row in rows]
        committed_steps = [
            float(row["avg_committed_steps_per_replan"]) for row in rows
        ]
        summary[case_key] = {
            "num_runs": len(rows),
            "completed_runs": len(completed_rows),
            "avg_schedule_time": mean(schedule_times) if schedule_times else None,
            "median_schedule_time": median(schedule_times) if schedule_times else None,
            "avg_compute_time": mean(compute_times) if compute_times else None,
            "avg_schedule_tcsr": mean(tcsr_values) if tcsr_values else None,
            "avg_wait_count": mean(wait_counts) if wait_counts else None,
            "avg_monitor_count": mean(monitor_counts) if monitor_counts else None,
            "avg_replanning_count": (
                mean(replanning_counts) if replanning_counts else None
            ),
            "avg_committed_steps_per_replan": (
                mean(committed_steps) if committed_steps else None
            ),
        }
    return summary


def compare_ready_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build compact comparison helpers from run results."""

    best_by_task: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        task_key = f"{result['case']}::{result['instruction']}"
        grouped.setdefault(task_key, []).append(result)

    for task_key, rows in grouped.items():
        best_row = sorted(
            rows,
            key=lambda row: (row["final_schedule_time"], row["total_compute_time"]),
        )[0]
        best_by_task[task_key] = {
            "case": best_row["case"],
            "instruction": best_row["instruction"],
            "beam_width": best_row["beam_width"],
            "beam_depth": best_row["beam_depth"],
            "final_schedule_time": best_row["final_schedule_time"],
            "total_compute_time": best_row["total_compute_time"],
        }
    return {"best_by_task": best_by_task}


def run_grid_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Run an offline experiment grid and return a structured report."""

    previous_values = _apply_runtime_overrides(config)
    try:
        scene_positions = load_scene_positions(f"{config.scene}_positions.json")
        nav_graph = resolve_nav_graph(config, scene_positions)
        tasks = build_experiment_tasks(config)
        selected_instructions = build_selected_instruction_map(config)
        results = [
            run_single_rollout(
                config,
                task,
                nav_graph=nav_graph,
                scene_positions=scene_positions,
            )
            for task in tasks
        ]
        report = {
            "schema_version": config.schema_version,
            "saved_time": datetime.now().isoformat(timespec="seconds"),
            "experiment": {
                "name": config.experiment_name,
                "tags": list(config.tags),
            },
            "config": asdict(config),
            "selected_instructions": selected_instructions,
            "summary_by_setting": summarize_results(results),
            "summary_by_case_setting": summarize_results_by_case_setting(results),
            "comparison": compare_ready_summary(results),
            "results": results,
        }
        oracle_results = load_oracle_results_for_selection(
            base_dir=resolve_oracle_reference_dir(config.oracle_reference_dir),
            task_folder_name=config.task_folder_name,
            scene_name=config.scene,
            selected_instructions=selected_instructions,
        )
        report["oracle_comparison"] = build_oracle_comparison_section(
            scheduler_results=results,
            selected_instructions=selected_instructions,
            oracle_results=oracle_results,
        )
        return report
    finally:
        _restore_runtime_overrides(previous_values)


def _build_oracle_reference_config(config: ExperimentConfig) -> ExperimentConfig:
    """Build the config used for standalone oracle reference generation."""

    merged = asdict(config)
    merged["approach"] = "oracle"
    merged["ablation_config"] = "NONE_MONITORING"
    merged["gt_distribution"] = "constant"
    merged["schema_version"] = ORACLE_REFERENCE_SCHEMA_VERSION
    return ExperimentConfig(**merged)


def _build_deterministic_scheduler_config(config: ExperimentConfig) -> ExperimentConfig:
    """Build a monitoring-free deterministic config for oracle comparison.

    Args:
        config: User-provided experiment configuration.

    Returns:
        Copy of the config aligned with deterministic initial-schedule comparison.
    """

    merged = asdict(config)
    merged["ablation_config"] = "NONE_MONITORING"
    if merged["approach"] == "oracle":
        # Oracle comparison still needs a deterministic scheduler baseline.
        merged["approach"] = "edf"
    merged["gt_distribution"] = "constant"
    merged["schema_version"] = "offline_harness.v1"
    return ExperimentConfig(**merged)


def _run_exact_oracle_rollout(
    config: ExperimentConfig,
    *,
    case_name: str,
    instruction: str,
    nav_graph: NavGraph,
    scene_positions: Mapping[str, Position],
    incumbent_upper_bound: Optional[float],
) -> dict[str, Any]:
    """Run the deterministic exact oracle for one instruction.

    Args:
        config: Deterministic experiment config.
        case_name: Case identifier.
        instruction: Instruction file name.
        nav_graph: Deterministic navigation graph.
        scene_positions: Scene positions for the selected floorplan.
        incumbent_upper_bound: Optional incumbent makespan for pruning.

    Returns:
        JSON-serializable oracle result dictionary.
    """

    task_data = load_task_data_from_sampled_set(
        case_name,
        config.scene,
        instruction,
    )
    subtasks, constraints, _ = TaskUtil.build_tasks_and_constraints(
        task_data,
        scene_file_name=f"{config.scene}_physics_environment.json",
    )
    initial_state = TaskUtil.get_init_state(
        subtasks,
        constraints,
        copy.deepcopy(dict(scene_positions)),
    )
    action_handler = ActionHandler(nav_graph, real_world_mode=False)
    constraint_handler = ConstraintHandler(action_handler)
    oracle = DeterministicExactOracle(
        action_handler=action_handler,
        constraint_handler=constraint_handler,
        heuristic_manager=HeuristicManager(action_handler),
        time_limit_seconds=config.oracle_time_limit_seconds,
    )
    solution = oracle.solve(
        initial_state,
        instruction=instruction,
        case=case_name,
        incumbent_upper_bound=build_initial_oracle_upper_bound(incumbent_upper_bound),
    )
    _, schedule_tcsr, detail_log = calculate_timing_success_rate(
        constraints,
        solution.completed_entries,
    )
    result = solution.as_dict()
    result["steps"] = _serialize_schedule_steps(solution.completed_entries)
    result["schedule_tcsr"] = schedule_tcsr
    result["tcsr_is_one"] = _is_tcsr_perfect(schedule_tcsr)
    result["timing_detail"] = detail_log
    return _build_hybrid_result_row(config, result)


def run_oracle_reference_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Generate and persist deterministic oracle reference rows."""

    reference_config = _build_oracle_reference_config(config)
    selected_instructions = build_selected_instruction_map(reference_config)
    previous_values = _apply_runtime_overrides(reference_config)
    try:
        scene_positions = load_scene_positions(f"{reference_config.scene}_positions.json")
        nav_graph = resolve_nav_graph(reference_config, scene_positions)
        oracle_results: list[dict[str, Any]] = []
        for case_name, instructions in selected_instructions.items():
            for instruction in instructions:
                oracle_results.append(
                    _run_exact_oracle_rollout(
                        reference_config,
                        case_name=case_name,
                        instruction=instruction,
                        nav_graph=nav_graph,
                        scene_positions=scene_positions,
                        incumbent_upper_bound=None,
                    )
                )
    finally:
        _restore_runtime_overrides(previous_values)

    reference_dir = resolve_oracle_reference_dir(reference_config.oracle_reference_dir)
    save_oracle_reference_rows(
        oracle_results,
        base_dir=reference_dir,
        task_folder_name=reference_config.task_folder_name,
        scene_name=reference_config.scene,
    )
    return {
        "schema_version": ORACLE_REFERENCE_SCHEMA_VERSION,
        "saved_time": datetime.now().isoformat(timespec="seconds"),
        "experiment": {
            "name": config.experiment_name,
            "tags": list(config.tags),
        },
        "config": asdict(reference_config),
        "selected_instructions": selected_instructions,
        "oracle_results": oracle_results,
        "oracle_summary": summarize_oracle_results(oracle_results),
    }


def save_experiment_report(report: Mapping[str, Any], output_path: Path) -> None:
    """Persist an experiment report as JSON."""

    payload: Any = report
    if isinstance(report.get("results"), list) and len(report["results"]) == 1:
        payload = _build_persisted_single_run_payload(report["results"][0])
    elif (
        isinstance(report.get("oracle_results"), list)
        and len(report["oracle_results"]) == 1
    ):
        payload = _build_persisted_single_run_payload(report["oracle_results"][0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def iter_report_lines(report: Mapping[str, Any]) -> Iterable[str]:
    """Yield a compact CLI report preview."""

    yield f"Experiment: {report['experiment']['name']}"
    for setting, metrics in report["summary_by_setting"].items():
        yield (
            f"- {setting}: completed={metrics['completed_runs']}/{metrics['num_runs']}, "
            f"avg_schedule_time={metrics['avg_schedule_time']}, "
            f"median_schedule_time={metrics['median_schedule_time']}, "
            f"avg_compute_time={metrics['avg_compute_time']}, "
            f"avg_schedule_tcsr={metrics['avg_schedule_tcsr']}"
        )


def iter_oracle_report_lines(report: Mapping[str, Any]) -> Iterable[str]:
    """Yield a compact CLI preview for oracle reference reports."""

    yield f"Experiment: {report['experiment']['name']}"
    oracle_summary = report["oracle_summary"]
    yield (
        "Oracle: "
        f"exact={oracle_summary['exact_instructions']}/{oracle_summary['num_instructions']}, "
        f"avg_optimal_schedule_time={oracle_summary['avg_optimal_schedule_time']}, "
        f"avg_solve_time={oracle_summary['avg_solve_time']}, "
        f"avg_total_compute_time={oracle_summary['avg_total_compute_time']}, "
        f"avg_schedule_tcsr={oracle_summary['avg_schedule_tcsr']}"
    )
