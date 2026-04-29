"""Thin orchestrator for offline experiment suites."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

# Repo root (contains src/); parents[0]=offline, parents[1]=scripts, parents[2]=root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.offline_oracle_reference import (  # noqa: E402
    DEFAULT_ORACLE_REFERENCE_DIR,
    resolve_oracle_reference_dir,
)
from src.utils.common import create_module_logger  # noqa: E402

DEFAULT_SUITE_ORDER = (
    "scalability",
    "eta_sensitivity",
    "monitoring_budget",
    "pf_vs_bayesian",
)
DEFAULT_ORACLE_CONFIG = "scripts/offline/config/oracle_reference_config.yaml"
DEFAULT_TASK_FOLDER = "sampled_10_instruction_set_for_final_experiment_set_b"
DEFAULT_OUTPUT_DIR = "assets/results/offline_batch"
DEFAULT_NAV_GRAPH_SOURCE = "ai2thor_controller"

SUITE_LATEX_MODULE: dict[str, str] = {
    "scalability": "assets.result_analysis.scalability_case_table",
    "eta_sensitivity": "assets.result_analysis.eta_sensitivity_tables",
    "pf_vs_bayesian": "assets.result_analysis.pf_vs_bayesian_table",
    "monitoring_budget": "assets.result_analysis.monitoring_budget_table",
}

SUITE_REGISTRY: dict[str, tuple[str, ...]] = {
    "pf_vs_bayesian": (
        "scripts/offline/config/pf_vs_bayesian_gaussian_bayesian_config.yaml",
        "scripts/offline/config/pf_vs_bayesian_lognormal_bayesian_config.yaml",
        "scripts/offline/config/pf_vs_bayesian_mixture_bayesian_config.yaml",
        "scripts/offline/config/pf_vs_bayesian_gaussian_particle_filter_config.yaml",
        "scripts/offline/config/pf_vs_bayesian_lognormal_particle_filter_config.yaml",
        "scripts/offline/config/pf_vs_bayesian_mixture_particle_filter_config.yaml",
    ),
    "scalability": ("scripts/offline/config/scalability_config.yaml",),
    "eta_sensitivity": ("scripts/offline/config/eta_sensitivity_config.yaml",),
    "monitoring_budget": ("scripts/offline/config/monitoring_budget_config.yaml",),
}


@dataclass(frozen=True)
class SuiteDefinition:
    """Resolved suite metadata used by the thin orchestrator."""

    name: str
    config_paths: tuple[Path, ...]
    raw_configs: tuple[dict[str, Any], ...]
    output_dir: Path
    oracle_reference_dir: Path
    task_folder_names: tuple[str, ...]
    base_dir: Path
    batch_dirname: str
    oracle_dirname: str
    analysis_output_dir: Path


@dataclass(frozen=True)
class StageSpec:
    """One subprocess stage in the orchestrated suite plan."""

    kind: str
    suite: str | None
    command: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class ExecutionPlan:
    """Fully materialized execution plan for one suite request."""

    suite_names: tuple[str, ...]
    suites: tuple[SuiteDefinition, ...]
    oracle_config_path: Path
    oracle_config: dict[str, Any]
    stages: tuple[StageSpec, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for suite orchestration."""

    parser = argparse.ArgumentParser(
        description="Run offline experiment suites with oracle preflight and analysis."
    )
    parser.add_argument(
        "--suite",
        choices=[*DEFAULT_SUITE_ORDER, "all"],
        required=True,
        help="Experiment family to run, or 'all' for the full offline suite set.",
    )
    parser.add_argument(
        "--oracle-config",
        type=str,
        default=DEFAULT_ORACLE_CONFIG,
        help=(
            "Base oracle batch config. "
            "Its oracle-specific settings are preserved while scope is merged "
            "from the selected suite configs."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the staged subprocess plan without executing it.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Force skip-completed behavior on offline batch stages.",
    )
    return parser.parse_args(argv)


def _resolve_config_path(config_path: str | Path) -> Path:
    """Resolve CLI config paths like the existing offline runners."""

    candidate = Path(config_path).expanduser()
    if candidate.is_absolute():
        return candidate
    offline_dir = Path(__file__).resolve().parent
    scripts_dir = offline_dir.parent
    parts = candidate.parts
    if parts and parts[0] == "scripts":
        return PROJECT_ROOT / candidate
    if len(parts) == 1:
        return offline_dir / candidate
    return scripts_dir / candidate


def _resolve_repo_path(path_like: str | Path) -> Path:
    """Resolve repo-relative paths against the repository root."""

    candidate = Path(path_like).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a mapping."""

    resolved_path = _resolve_config_path(config_path)
    with resolved_path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must deserialize into a mapping: {resolved_path}")
    return loaded


def _unique_preserve_order(values: Sequence[str]) -> list[str]:
    """Return a stable de-duplicated list while preserving first occurrence."""

    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def _normalize_string_list(value: Any, *, default: str | None = None) -> list[str]:
    """Normalize YAML scalar/list values into a string list."""

    if value is None:
        return [default] if default is not None else []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _resolve_suite_names(requested_suite: str) -> tuple[str, ...]:
    """Expand the CLI suite selector into an ordered tuple of suite names."""

    if requested_suite == "all":
        return DEFAULT_SUITE_ORDER
    return (requested_suite,)


def _collect_common_path(
    raw_configs: Sequence[Mapping[str, Any]],
    *,
    key: str,
    default: str | Path,
    context: str,
    resolver: Callable[[str | Path], Path],
) -> Path:
    """Resolve one path-valued config key and require a single unique result."""

    candidates = {resolver(config.get(key, default)) for config in raw_configs}
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in sorted(candidates))
        raise ValueError(f"{context} must share a single {key}: {rendered}")
    return next(iter(candidates))


def _derive_analysis_paths(
    *,
    output_dir: Path,
    oracle_reference_dir: Path,
) -> tuple[Path, str, str]:
    """Return the common base_dir plus batch/oracle relative dirnames."""

    common_base = Path(os.path.commonpath([str(output_dir), str(oracle_reference_dir)]))
    batch_dirname = output_dir.relative_to(common_base).as_posix()
    oracle_dirname = oracle_reference_dir.relative_to(common_base).as_posix()
    return common_base, batch_dirname, oracle_dirname


def _load_suite_definition(
    suite_name: str,
    config_paths: Sequence[str | Path],
) -> SuiteDefinition:
    """Load, validate, and normalize the configs that make up one suite."""

    resolved_paths = tuple(_resolve_config_path(path) for path in config_paths)
    raw_configs = tuple(_load_yaml_config(path) for path in resolved_paths)

    output_dir = _collect_common_path(
        raw_configs,
        key="output_dir",
        default=DEFAULT_OUTPUT_DIR,
        context=f"Suite '{suite_name}'",
        resolver=_resolve_repo_path,
    )
    oracle_reference_dir = _collect_common_path(
        raw_configs,
        key="oracle_reference_dir",
        default=DEFAULT_ORACLE_REFERENCE_DIR,
        context=f"Suite '{suite_name}'",
        resolver=resolve_oracle_reference_dir,
    )
    task_folder_names = _unique_preserve_order(
        [
            task_folder
            for config in raw_configs
            for task_folder in _normalize_string_list(
                config.get("task_folder_name"),
                default=DEFAULT_TASK_FOLDER,
            )
        ]
    )
    base_dir, batch_dirname, oracle_dirname = _derive_analysis_paths(
        output_dir=output_dir,
        oracle_reference_dir=oracle_reference_dir,
    )
    analysis_output_dir = output_dir.parent / "analysis" / suite_name
    return SuiteDefinition(
        name=suite_name,
        config_paths=resolved_paths,
        raw_configs=raw_configs,
        output_dir=output_dir,
        oracle_reference_dir=oracle_reference_dir,
        task_folder_names=tuple(task_folder_names),
        base_dir=base_dir,
        batch_dirname=batch_dirname,
        oracle_dirname=oracle_dirname,
        analysis_output_dir=analysis_output_dir,
    )


def _require_common_string_value(
    raw_configs: Sequence[Mapping[str, Any]],
    *,
    key: str,
    default: str,
    context: str,
) -> str:
    """Return one normalized scalar value shared by all selected configs."""

    values = {str(config.get(key, default)) for config in raw_configs}
    if len(values) != 1:
        rendered = ", ".join(sorted(values))
        raise ValueError(f"{context} must share a single {key}: {rendered}")
    return next(iter(values))


def _merge_scene_scope(
    raw_configs: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, list[str]]]:
    """Union the selected suites' scene_type/scene_lists scope."""

    scene_type_order: list[str] = []
    merged_scene_lists: dict[str, list[str]] = {}
    for config in raw_configs:
        scene_types = _normalize_string_list(
            config.get("scene_type"), default="kitchen"
        )
        scene_lists = dict(config.get("scene_lists", {}))
        for scene_type in scene_types:
            if scene_type not in scene_type_order:
                scene_type_order.append(scene_type)
            merged_scene_lists.setdefault(scene_type, [])
            merged_scene_lists[scene_type].extend(
                str(scene_name) for scene_name in scene_lists.get(scene_type, [])
            )
    return scene_type_order, {
        scene_type: _unique_preserve_order(scene_names)
        for scene_type, scene_names in merged_scene_lists.items()
    }


def _merge_oracle_config(
    *,
    base_oracle_config: Mapping[str, Any],
    suites: Sequence[SuiteDefinition],
) -> dict[str, Any]:
    """Create the oracle preflight config by overlaying suite scope on a base config."""

    selected_configs = [config for suite in suites for config in suite.raw_configs]
    oracle_reference_dirs = {suite.oracle_reference_dir for suite in suites}
    if len(oracle_reference_dirs) != 1:
        rendered = ", ".join(str(path) for path in sorted(oracle_reference_dirs))
        raise ValueError(
            "Selected suites must share one oracle_reference_dir for a common "
            f"oracle preflight: {rendered}"
        )

    merged = dict(base_oracle_config)
    scene_type, scene_lists = _merge_scene_scope(selected_configs)
    cases = _unique_preserve_order(
        [
            case_name
            for config in selected_configs
            for case_name in _normalize_string_list(config.get("cases"))
        ]
    )
    task_folder_names = _unique_preserve_order(
        [task_folder for suite in suites for task_folder in suite.task_folder_names]
    )
    instructions = _unique_preserve_order(
        [
            instruction_name
            for config in selected_configs
            for instruction_name in _normalize_string_list(config.get("instructions"))
        ]
    )
    max_tasks_values = [
        int(config["max_tasks"])
        for config in selected_configs
        if config.get("max_tasks") is not None
    ]

    merged["scene_type"] = scene_type
    merged["scene_lists"] = scene_lists
    merged["cases"] = cases
    merged["task_folder_name"] = task_folder_names
    merged["nav_graph_source"] = _require_common_string_value(
        selected_configs,
        key="nav_graph_source",
        default=DEFAULT_NAV_GRAPH_SOURCE,
        context="Selected suites",
    )
    merged["oracle_reference_dir"] = str(next(iter(oracle_reference_dirs)))
    merged["skip_completed"] = True
    if max_tasks_values:
        merged["max_tasks"] = max(max_tasks_values)
    if instructions:
        merged["instructions"] = instructions
    else:
        merged.pop("instructions", None)
    return merged


def _build_oracle_stage_command(
    oracle_config_path: Path,
) -> tuple[str, ...]:
    """Return the standalone oracle preflight subprocess command."""

    return (
        os.environ.get("PYTHON", sys.executable),
        str(PROJECT_ROOT / "scripts/offline/run_oracle_reference_batch.py"),
        "--config",
        str(oracle_config_path),
        "--skip-completed",
    )


def _build_batch_stage_command(
    config_path: Path,
    *,
    force_skip_completed: bool,
) -> tuple[str, ...]:
    """Return the batch subprocess command for one suite config."""

    command = [
        os.environ.get("PYTHON", sys.executable),
        str(PROJECT_ROOT / "scripts/offline/run_batch.py"),
        "--config",
        str(config_path),
        "--mode",
        "offline",
    ]
    if force_skip_completed:
        command.append("--skip-completed")
    return tuple(command)


def _build_analysis_stage_command(
    suite: SuiteDefinition,
    *,
    task_folder_name: str,
) -> tuple[str, ...]:
    """Return the analysis subprocess command for one task folder."""

    analysis_output_dir = suite.analysis_output_dir / task_folder_name

    return (
        os.environ.get("PYTHON", sys.executable),
        "-m",
        "assets.result_analysis.offline_comparison",
        "--base_dir",
        str(suite.base_dir),
        "--batch_dirname",
        suite.batch_dirname,
        "--oracle_dirname",
        suite.oracle_dirname,
        "--task_folder",
        task_folder_name,
        "--output_dir",
        str(analysis_output_dir),
    )


def _build_latex_stage_command(
    suite: SuiteDefinition,
    *,
    task_folder_names: Sequence[str],
) -> tuple[str, ...] | None:
    """Return the LaTeX export subprocess command for a suite, or None if unsupported."""

    module = SUITE_LATEX_MODULE.get(suite.name)
    if module is None:
        return None

    latex_out_dir = suite.analysis_output_dir / "latex_tables"
    python = os.environ.get("PYTHON", sys.executable)
    common_args = (
        "--analysis-root",
        str(suite.analysis_output_dir),
        "--task-folders",
        *task_folder_names,
    )

    if suite.name == "scalability":
        return (
            python,
            "-m",
            module,
            *common_args,
            "--output",
            str(latex_out_dir / "scalability_case_table.tex"),
        )

    if suite.name == "eta_sensitivity":
        return (
            python,
            "-m",
            module,
            *common_args,
            "--batch-root",
            str(suite.output_dir),
            "--out-dir",
            str(latex_out_dir),
        )

    if suite.name == "pf_vs_bayesian":
        return (
            python,
            "-m",
            module,
            *common_args,
            "--output",
            str(latex_out_dir / "pf_vs_bayesian_overall.tex"),
        )

    if suite.name == "monitoring_budget":
        return (
            python,
            "-m",
            module,
            *common_args,
            "--out-dir",
            str(latex_out_dir),
        )

    return None


def write_merged_oracle_config(
    config: Mapping[str, Any],
    *,
    output_path: Path,
) -> Path:
    """Persist the generated oracle preflight config."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return output_path


def build_execution_plan(
    *,
    requested_suite: str,
    oracle_config_path: str | Path,
    force_skip_completed: bool,
    temp_dir: str | Path,
) -> ExecutionPlan:
    """Build the full staged plan for one suite invocation."""

    suite_names = _resolve_suite_names(requested_suite)
    suites = tuple(
        _load_suite_definition(suite_name, SUITE_REGISTRY[suite_name])
        for suite_name in suite_names
    )
    base_oracle_config = _load_yaml_config(oracle_config_path)
    merged_oracle_config = _merge_oracle_config(
        base_oracle_config=base_oracle_config,
        suites=suites,
    )
    merged_oracle_config_path = write_merged_oracle_config(
        merged_oracle_config,
        output_path=Path(temp_dir) / "merged_oracle_reference_config.yaml",
    )

    stages: list[StageSpec] = [
        StageSpec(
            kind="oracle",
            suite=None,
            command=_build_oracle_stage_command(merged_oracle_config_path),
            description="oracle preflight",
        )
    ]
    for suite in suites:
        for config_path in suite.config_paths:
            stages.append(
                StageSpec(
                    kind="batch",
                    suite=suite.name,
                    command=_build_batch_stage_command(
                        config_path,
                        force_skip_completed=force_skip_completed,
                    ),
                    description=f"offline batch: {config_path.name}",
                )
            )
        for task_folder_name in suite.task_folder_names:
            stages.append(
                StageSpec(
                    kind="analysis",
                    suite=suite.name,
                    command=_build_analysis_stage_command(
                        suite,
                        task_folder_name=task_folder_name,
                    ),
                    description=f"offline analysis: {task_folder_name}",
                )
            )
        latex_command = _build_latex_stage_command(
            suite,
            task_folder_names=suite.task_folder_names,
        )
        if latex_command is not None:
            stages.append(
                StageSpec(
                    kind="latex",
                    suite=suite.name,
                    command=latex_command,
                    description=f"latex export: {suite.name} (aggregated)",
                )
            )
    return ExecutionPlan(
        suite_names=suite_names,
        suites=suites,
        oracle_config_path=merged_oracle_config_path,
        oracle_config=merged_oracle_config,
        stages=tuple(stages),
    )


def _log_dry_run(plan: ExecutionPlan, logger: Any) -> None:
    """Emit a concise dry-run view of the planned subprocess stages."""

    logger.critical("=" * 80)
    logger.critical(
        "DRY RUN MODE: offline suite plan for %s",
        ", ".join(plan.suite_names),
    )
    logger.critical("Generated oracle config: %s", plan.oracle_config_path)
    logger.critical("=" * 80)
    for index, stage in enumerate(plan.stages, start=1):
        suite_label = stage.suite or "shared"
        logger.critical(
            "[%02d/%02d] %s | suite=%s | %s",
            index,
            len(plan.stages),
            stage.kind,
            suite_label,
            stage.description,
        )
        logger.critical("         %s", " ".join(stage.command))
    logger.critical("=" * 80)


def _run_stage(stage: StageSpec, logger: Any) -> None:
    """Execute one planned subprocess stage."""

    suite_label = stage.suite or "shared"
    logger.critical(
        "Running %s stage | suite=%s | %s",
        stage.kind,
        suite_label,
        stage.description,
    )
    subprocess.run(
        list(stage.command),
        check=True,
        cwd=PROJECT_ROOT,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for the offline suite orchestrator."""

    args = parse_args(argv)
    logger = create_module_logger(module_name=__name__, module_log=True)
    with tempfile.TemporaryDirectory(prefix="offline_suite_") as temp_dir:
        plan = build_execution_plan(
            requested_suite=args.suite,
            oracle_config_path=args.oracle_config,
            force_skip_completed=args.skip_completed,
            temp_dir=temp_dir,
        )
        if args.dry_run:
            _log_dry_run(plan, logger)
            return
        for stage in plan.stages:
            _run_stage(stage, logger)


if __name__ == "__main__":
    main()
