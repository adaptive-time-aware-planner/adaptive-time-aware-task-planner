"""Unified subprocess batch runner for offline and AI2-THOR experiments."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml

# Repo root (contains src/); parents[0]=offline, parents[1]=scripts, parents[2]=root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.batch import (
    BatchTask,
    build_offline_batch_summary_path,
    build_ai2thor_tasks,
    build_offline_tasks,
    execute_batch_tasks,
    generate_run_timestamp,
    log_dry_run,
    write_batch_summary,
)
from src.utils.common import create_module_logger

CONFIG_SECTION_KEYS = {"ai2thor", "offline", "mode"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the unified batch runner."""

    parser = argparse.ArgumentParser(
        description="Run offline and AI2-THOR experiment batches from one queue."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="scripts/offline/config/batch_config.yaml",
        help=(
            "YAML config path. "
            "repo-relative if starting with scripts/; "
            "else under scripts/ for multi-segment paths; "
            "else next to this script (scripts/offline/config/)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "ai2thor", "both"],
        default=None,
        help="Optional mode override for the loaded config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated task list without executing it.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Force skip-completed behavior on enabled sections.",
    )
    return parser.parse_args()


def _resolve_config_path(config_path: str) -> Path:
    """Resolve a config path for CLI use.

    - Absolute paths are used as given.
    - Paths starting with ``scripts/`` are resolved from the repository root
      (e.g. ``scripts/offline/config/batch_config.yaml``).
    - Multi-segment paths otherwise resolve from ``scripts/`` (e.g.
      ``offline/config/batch_config.yaml``).
    - A single filename resolves from ``scripts/offline/config/``.
    """

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
        raise ValueError("Batch config must deserialize into a mapping.")
    return loaded


def _is_unified_config(raw_config: Mapping[str, Any]) -> bool:
    """Return whether the loaded YAML already uses the unified schema."""

    return any(key in raw_config for key in CONFIG_SECTION_KEYS)


def _infer_mode(
    *,
    requested_mode: str | None,
    raw_mode: Any,
    has_ai2thor: bool,
    has_offline: bool,
) -> str:
    """Determine the effective execution mode."""

    if requested_mode is not None:
        return requested_mode
    if isinstance(raw_mode, str):
        return raw_mode
    if has_ai2thor and has_offline:
        return "both"
    if has_offline:
        return "offline"
    return "ai2thor"


def _normalize_section_configs(
    raw_config: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    """Normalize unified or legacy YAML into per-mode config mappings."""

    if not _is_unified_config(raw_config):
        legacy_config = dict(raw_config)
        if args.skip_completed:
            legacy_config["skip_completed"] = True
        return args.mode or "offline", None, legacy_config

    shared_config = {
        key: value
        for key, value in raw_config.items()
        if key not in CONFIG_SECTION_KEYS
    }
    ai2thor_config = None
    if isinstance(raw_config.get("ai2thor"), dict):
        ai2thor_config = {**shared_config, **dict(raw_config["ai2thor"])}
    offline_config = None
    if isinstance(raw_config.get("offline"), dict):
        offline_config = {**shared_config, **dict(raw_config["offline"])}
    effective_mode = _infer_mode(
        requested_mode=args.mode,
        raw_mode=raw_config.get("mode"),
        has_ai2thor=ai2thor_config is not None,
        has_offline=offline_config is not None,
    )
    if args.skip_completed:
        if ai2thor_config is not None:
            ai2thor_config["skip_completed"] = True
        if offline_config is not None:
            offline_config["skip_completed"] = True
    return effective_mode, ai2thor_config, offline_config


def _build_tasks_for_mode(
    *,
    mode: str,
    ai2thor_config: Mapping[str, Any] | None,
    offline_config: Mapping[str, Any] | None,
    run_timestamp: str,
    logger: logging.Logger,
) -> list[BatchTask]:
    """Build the task list for the effective run mode."""

    tasks: list[BatchTask] = []
    if mode in {"ai2thor", "both"}:
        if ai2thor_config is None:
            raise ValueError("Selected mode requires an 'ai2thor' config section.")
        tasks.extend(
            build_ai2thor_tasks(
                config=ai2thor_config,
                run_timestamp=run_timestamp,
                logger=logger,
            )
        )
    if mode in {"offline", "both"}:
        if offline_config is None:
            raise ValueError("Selected mode requires an 'offline' config section.")
        tasks.extend(
            build_offline_tasks(
                config=offline_config,
                run_timestamp=run_timestamp,
                logger=logger,
            )
        )
    return tasks


def main() -> None:
    """CLI entrypoint for unified experiment batching."""

    args = parse_args()
    raw_config = load_config(args.config)
    run_timestamp = generate_run_timestamp()
    logger = create_module_logger(
        module_name=__name__,
        module_log=True,
        run_timestamp=run_timestamp,
    )
    mode, ai2thor_config, offline_config = _normalize_section_configs(raw_config, args)
    tasks = _build_tasks_for_mode(
        mode=mode,
        ai2thor_config=ai2thor_config,
        offline_config=offline_config,
        run_timestamp=run_timestamp,
        logger=logger,
    )
    logger.critical("Unified batch mode: %s", mode)
    logger.critical("Total queued tasks: %s", len(tasks))
    if args.dry_run:
        log_dry_run(tasks, logger)
        return
    max_workers = int(
        raw_config.get("max_workers")
        or (ai2thor_config or {}).get("max_workers")
        or (offline_config or {}).get("max_workers")
        or 4
    )
    execute_batch_tasks(tasks, max_workers=max_workers, logger=logger)
    if mode in {"offline", "both"} and offline_config is not None:
        offline_tasks = [
            task for task in tasks if task.metadata.get("mode") == "offline"
        ]
        summary_path = build_offline_batch_summary_path(
            offline_config,
            run_timestamp=run_timestamp,
        )
        write_batch_summary(
            offline_tasks,
            summary_path=summary_path,
            run_timestamp=run_timestamp,
            mode=mode,
        )
        logger.critical("Saved offline batch summary: %s", summary_path)


if __name__ == "__main__":
    main()
