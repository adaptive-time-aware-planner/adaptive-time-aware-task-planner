"""Batch runner for standalone offline oracle reference generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

# Repo root (contains src/); parents[0]=offline, parents[1]=scripts, parents[2]=root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.batch import (  # noqa: E402
    build_oracle_reference_batch_summary_path,
    build_oracle_reference_tasks,
    execute_batch_tasks,
    generate_run_timestamp,
    log_dry_run,
    write_batch_summary,
)
from src.utils.common import create_module_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the oracle reference batch runner."""

    parser = argparse.ArgumentParser(
        description="Run offline oracle reference generation as a batch."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="scripts/offline/config/oracle_reference_config.yaml",
        help=(
            "YAML config path. "
            "repo-relative if starting with scripts/; "
            "else under scripts/ for multi-segment paths; "
            "else next to this script (scripts/offline/config/)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated task list without executing it.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Force skip-completed behavior for oracle reference tasks.",
    )
    return parser.parse_args()


def _resolve_config_path(config_path: str) -> Path:
    """Resolve a config path for CLI use (aligned with ``run_batch.py``)."""

    candidate = Path(config_path).expanduser()
    if candidate.is_absolute():
        return candidate
    offline_dir = Path(__file__).resolve().parent
    scripts_dir = offline_dir.parent
    project_root = scripts_dir.parent
    parts = candidate.parts
    if parts and parts[0] == "scripts":
        return project_root / candidate
    if len(parts) == 1:
        return offline_dir / "config" / candidate
    return scripts_dir / candidate


def load_config(config_path: str) -> dict[str, Any]:
    """Load and parse a YAML config file."""

    resolved_path = _resolve_config_path(config_path)
    with resolved_path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Oracle reference config must deserialize into a mapping.")
    return loaded


def main() -> None:
    """CLI entrypoint for oracle reference batching."""

    args = parse_args()
    raw_config = load_config(args.config)
    if args.skip_completed:
        raw_config["skip_completed"] = True
    run_timestamp = generate_run_timestamp()
    logger = create_module_logger(
        module_name=__name__,
        module_log=True,
        run_timestamp=run_timestamp,
    )
    tasks = build_oracle_reference_tasks(
        config=raw_config,
        run_timestamp=run_timestamp,
        logger=logger,
    )
    logger.critical("Oracle reference batch mode")
    logger.critical("Total queued tasks: %s", len(tasks))
    if args.dry_run:
        log_dry_run(tasks, logger)
        return
    max_workers = int(raw_config.get("max_workers") or 4)
    execute_batch_tasks(tasks, max_workers=max_workers, logger=logger)
    summary_path = build_oracle_reference_batch_summary_path(
        raw_config,
        run_timestamp=run_timestamp,
    )
    write_batch_summary(
        tasks,
        summary_path=summary_path,
        run_timestamp=run_timestamp,
        mode="offline_oracle_reference",
    )
    logger.critical("Saved oracle reference batch summary: %s", summary_path)


if __name__ == "__main__":
    main()
