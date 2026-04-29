"""CLI entrypoint for in-process offline scheduler experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

# parents[0]=offline, parents[1]=scripts, parents[2]=repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.offline_compare import (  # noqa: E402
    compare_result_files,
    save_comparison_report,
)
from src.experiments.offline_harness import (  # noqa: E402
    ExperimentConfig,
    apply_cli_overrides,
    iter_report_lines,
    load_experiment_config,
    run_grid_experiment,
    save_experiment_report,
)

SUBCOMMAND_CHOICES = {"run", "compare"}


def _add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Register shared CLI arguments for offline runs."""

    parser.add_argument(
        "--config", type=str, default=None, help="JSON/YAML config path."
    )
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--suite-name", type=str, default=None)
    parser.add_argument("--task-folder-name", type=str, default=None)
    parser.add_argument(
        "--approach",
        type=str,
        choices=["bayesian", "edf", "cpm"],
        default=None,
    )
    parser.add_argument("--ablation-config", type=str, default=None)
    parser.add_argument("--init-prior-config", type=str, default=None)
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--instructions", nargs="*", default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--beam-bound", nargs="+", default=None)
    parser.add_argument("--belief-update-method", type=str, default=None)
    parser.add_argument("--gt-distribution", type=str, default=None)
    parser.add_argument("--gt-seed", type=int, default=None)
    parser.add_argument("--particle-distribution", type=str, default=None)
    parser.add_argument("--particle-likelihood-family", type=str, default=None)
    parser.add_argument("--factor-alpha", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument(
        "--max-monitoring-per-critical-interval",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--nav-graph-source",
        type=str,
        choices=["synthetic_grid", "ai2thor_controller"],
        default=None,
    )
    parser.add_argument("--oracle-reference-dir", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--oracle-time-limit-seconds", type=float, default=None)


def _build_subcommand_parser() -> argparse.ArgumentParser:
    """Build the existing subcommand-based parser."""

    parser = argparse.ArgumentParser(
        description="Offline scheduler experiment harness."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an offline experiment grid.")
    _add_common_run_arguments(run_parser)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare two result JSON files."
    )
    compare_parser.add_argument("--before", type=str, required=True)
    compare_parser.add_argument("--after", type=str, required=True)
    compare_parser.add_argument("--output-path", type=str, default=None)

    return parser


def _build_flat_parser() -> argparse.ArgumentParser:
    """Build a flat compatibility parser aligned with run_all-style flags."""

    parser = argparse.ArgumentParser(
        description="Offline scheduler experiment harness (flat compatibility mode)."
    )
    _add_common_run_arguments(parser)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--ros", action="store_true")
    parser.add_argument("--cloud-rendering", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--attempt", type=int, default=None)
    parser.add_argument("--log-level", type=str, default=None)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for both legacy and flat compatibility workflows."""

    cli_args = list(argv) if argv is not None else sys.argv[1:]
    if cli_args and cli_args[0] in SUBCOMMAND_CHOICES:
        parser = _build_subcommand_parser()
        parsed = parser.parse_args(cli_args)
        parsed.entry_mode = "subcommand"
        return parsed

    parser = _build_flat_parser()
    parsed = parser.parse_args(cli_args)
    parsed.command = "run"
    if parsed.instruction and not parsed.instructions:
        parsed.instructions = [parsed.instruction]
    parsed.entry_mode = "flat"
    return parsed


def _build_run_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build the effective run config from file defaults and CLI overrides."""

    base_config = load_experiment_config(
        Path(args.config).expanduser() if args.config else None
    )
    overrides: dict[str, Any] = {
        "experiment_name": args.experiment_name,
        "suite_name": args.suite_name,
        "task_folder_name": args.task_folder_name,
        "approach": args.approach,
        "ablation_config": args.ablation_config,
        "init_prior_config": args.init_prior_config,
        "case": args.case,
        "cases": args.cases,
        "scene": args.scene,
        "instructions": args.instructions,
        "max_tasks": args.max_tasks,
        "beam_bound": args.beam_bound,
        "belief_update_method": args.belief_update_method,
        "gt_distribution": args.gt_distribution,
        "gt_seed": args.gt_seed,
        "particle_distribution": args.particle_distribution,
        "particle_likelihood_family": args.particle_likelihood_family,
        "factor_alpha": args.factor_alpha,
        "eta": args.eta,
        "max_monitoring_per_critical_interval": (
            args.max_monitoring_per_critical_interval
        ),
        "nav_graph_source": args.nav_graph_source,
        "oracle_reference_dir": args.oracle_reference_dir,
        "output_path": args.output_path,
        "oracle_time_limit_seconds": args.oracle_time_limit_seconds,
    }
    return apply_cli_overrides(base_config, overrides)


def _run_command(args: argparse.Namespace) -> None:
    """Execute the run subcommand."""

    config = _build_run_config(args)
    report = run_grid_experiment(config)
    for line in iter_report_lines(report):
        print(line)
    if config.output_path:
        save_experiment_report(report, Path(config.output_path).expanduser())


def _compare_command(args: argparse.Namespace) -> None:
    """Execute the compare subcommand."""

    report = compare_result_files(
        Path(args.before).expanduser(),
        Path(args.after).expanduser(),
    )
    print(f"Improved settings: {len(report['improved_settings'])}")
    print(f"Regressed settings: {len(report['regressed_settings'])}")
    if args.output_path:
        save_comparison_report(report, Path(args.output_path).expanduser())


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    if args.command == "run":
        _run_command(args)
        return
    _compare_command(args)


if __name__ == "__main__":
    main()
