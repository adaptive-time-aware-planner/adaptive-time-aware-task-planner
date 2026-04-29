from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class MovePlan:
    """Represents a single file move plan.

    Attributes:
        source: The source file path to move from.
        destination: The destination file path to move to.
    """

    source: Path
    destination: Path


def setup_logger(verbosity: int) -> None:
    """Configure the root logger based on verbosity level.

    Args:
        verbosity: Verbosity level (0: WARNING, 1: INFO, 2+: DEBUG).
    """

    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def discover_json_files(root: Path) -> Iterable[Path]:
    """Yield all JSON files under a given root directory recursively.

    Args:
        root: Root directory to search.

    Yields:
        Paths to JSON files.
    """

    yield from root.rglob("*.json")


def build_move_plans(
    base_dir: Path,
    misplaced_root: Path,
    target_root: Path,
) -> List[MovePlan]:
    """Build a list of file move plans from misplaced_root to target_root.

    The relative path under misplaced_root is preserved under target_root.

    Args:
        base_dir: Base directory containing the states folders.
        misplaced_root: The incorrect root (e.g., base_dir/"states"/"60").
        target_root: The correct root (e.g., base_dir/"states60").

    Returns:
        A list of MovePlan entries.
    """

    plans: List[MovePlan] = []
    for src in discover_json_files(misplaced_root):
        try:
            rel = src.relative_to(misplaced_root)
        except ValueError:
            # Should not happen, but guard just in case
            logging.debug("Skipping non-relative path: %s", src)
            continue

        dst = target_root / rel
        plans.append(MovePlan(source=src, destination=dst))

    return plans


def execute_move_plans(
    plans: List[MovePlan], *, dry_run: bool, overwrite: bool
) -> Tuple[int, int]:
    """Execute file move plans.

    Args:
        plans: List of MovePlan entries to execute.
        dry_run: If True, only log actions without performing moves.
        overwrite: If True, overwrite existing destination files.

    Returns:
        A tuple of (moved_count, skipped_count).
    """

    moved = 0
    skipped = 0

    for plan in plans:
        dst_parent = plan.destination.parent
        action_desc = f"{plan.source} -> {plan.destination}"

        if plan.destination.exists():
            if overwrite:
                logging.info("Overwriting: %s", action_desc)
            else:
                logging.warning("Exists, skipping (use --overwrite): %s", action_desc)
                skipped += 1
                continue

        logging.info("Move: %s", action_desc)
        if not dry_run:
            dst_parent.mkdir(parents=True, exist_ok=True)
            # If destination exists and overwrite is True, remove before move
            if plan.destination.exists() and overwrite:
                plan.destination.unlink()
            shutil.move(str(plan.source), str(plan.destination))
        moved += 1

    return moved, skipped


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Move misplaced JSON files from assets/results/states/{60,100,140} "
            "into the correct assets/results/states{60,100,140} folders."
        )
    )

    parser.add_argument(
        "--base-dir",
        default="assets/results",
        help="Base directory containing states, states60, states100, states140.",
        type=str,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned moves without executing them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (use -v or -vv).",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point to move misplaced JSON files to the correct states folders.

    It scans under assets/results/states/{60,100,140} and moves JSON files to
    assets/results/states{60,100,140} respectively, preserving relative paths.
    """

    args = parse_args()
    setup_logger(args.verbose)

    base_dir = Path(args.base_dir)
    misplaced_states_dir = base_dir / "states"

    mapping: Dict[str, str] = {
        "60": "states60",
        "100": "states100",
        "140": "states140",
    }

    all_plans: List[MovePlan] = []
    for key, target_name in mapping.items():
        mis_root = misplaced_states_dir / key
        target_root = base_dir / target_name

        if not mis_root.exists():
            logging.debug("No misplaced directory found (skipping): %s", mis_root)
            continue

        logging.info("Planning moves from %s to %s", mis_root, target_root)
        plans = build_move_plans(base_dir=base_dir, misplaced_root=mis_root, target_root=target_root)
        all_plans.extend(plans)

    if not all_plans:
        logging.warning("No JSON files found to move. Nothing to do.")
        return

    logging.info("Total JSON files planned for move: %d", len(all_plans))
    moved, skipped = execute_move_plans(all_plans, dry_run=args.dry_run, overwrite=args.overwrite)
    logging.warning("Moved: %d, Skipped: %d", moved, skipped)


if __name__ == "__main__":
    main()

