"""CLI entrypoint for standalone offline oracle reference generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

# parents[0]=offline, parents[1]=scripts, parents[2]=repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.offline_harness import (  # noqa: E402
    ExperimentConfig,
    apply_cli_overrides,
    iter_oracle_report_lines,
    load_experiment_config,
    run_oracle_reference_experiment,
    save_experiment_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for oracle reference generation."""

    parser = argparse.ArgumentParser(
        description="Generate standalone exact oracle references for offline tasks."
    )
    parser.add_argument("--config", type=str, default=None, help="JSON/YAML config path.")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--task-folder-name", type=str, default=None)
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--instructions", nargs="*", default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument(
        "--nav-graph-source",
        type=str,
        choices=["synthetic_grid", "ai2thor_controller"],
        default=None,
    )
    parser.add_argument("--oracle-reference-dir", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--oracle-time-limit-seconds", type=float, default=None)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for oracle reference generation."""

    parser = build_parser()
    parsed = parser.parse_args(list(argv) if argv is not None else None)
    if parsed.instruction and not parsed.instructions:
        parsed.instructions = [parsed.instruction]
    return parsed


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build the effective oracle reference config from file and CLI values."""

    base_config = load_experiment_config(
        Path(args.config).expanduser() if args.config else None
    )
    overrides: dict[str, Any] = {
        "experiment_name": args.experiment_name,
        "task_folder_name": args.task_folder_name,
        "case": args.case,
        "cases": args.cases,
        "scene": args.scene,
        "instructions": args.instructions,
        "max_tasks": args.max_tasks,
        "nav_graph_source": args.nav_graph_source,
        "oracle_reference_dir": args.oracle_reference_dir,
        "output_path": args.output_path,
        "oracle_time_limit_seconds": args.oracle_time_limit_seconds,
    }
    return apply_cli_overrides(base_config, overrides)


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    config = build_config(args)
    report = run_oracle_reference_experiment(config)
    for line in iter_oracle_report_lines(report):
        print(line)
    if config.output_path:
        save_experiment_report(report, Path(config.output_path).expanduser())


if __name__ == "__main__":
    main()
