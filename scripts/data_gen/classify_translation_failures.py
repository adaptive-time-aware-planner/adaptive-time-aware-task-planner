"""Classify translation/decomposition failures from primitive action semantics.

This script validates translated task JSON files against the same held-object
preconditions enforced by ``ActionHandler``:

- ``GRASP`` is illegal while already holding an object.
- ``PLACE_INSIDE`` / ``PLACE_ON_TOP`` are illegal while holding nothing.

It can optionally intersect those semantic issues with planner failures
(``completed = false``) and oracle-invalid results to verify that the reported
failure categories are grounded in runnable experiment outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.common import create_module_logger
from src.utils.config.constants import ASSETS_PATH
from src.utils.task.primitive_action_semantics import (
    classify_issue_group,
    find_held_object_semantic_issues,
)

logger = create_module_logger(__name__, module_log=True)

TASK_DIR = ASSETS_PATH / "tasks"
DEFAULT_TASK_PREFIX = "decomposed_final_revision_metadata_260402"
DEFAULT_VERSIONS = ["v1", "v2", "v3", "v4", "v5"]
DEFAULT_RESULTS_ROOT = ASSETS_PATH / "results" / "offline_exp_result" / "offline_batch_scalability"
DEFAULT_ORACLE_ROOT = ASSETS_PATH / "results" / "offline_oracle_reference"
DEFAULT_RESULTS_GLOB = (
    "{prefix}_{version}/CORRECT_ESTIMATE/FloorPlan*/tasks_*/*/bayesian/"
    "DEFAULT__w10_d10__eta0.1.json"
)
PRIMARY_LABEL_PRIORITY = [
    "Container placed onto stove after ingredient placement, without re-grasp",
    "Pot placed into sink without first grasping pot",
    "Coffee machine prepared without grasping mug/cup",
    "StoveBurner placement without held object",
    "Double grasp while already holding an object",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify translation failures using primitive-action semantics."
    )
    parser.add_argument(
        "--task-prefix",
        default=DEFAULT_TASK_PREFIX,
        help="Base task folder prefix. Versions are appended as _v1, _v2, ...",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=DEFAULT_VERSIONS,
        help="Trial/version suffixes to inspect.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Offline batch result root for planner failure intersection.",
    )
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=DEFAULT_ORACLE_ROOT,
        help="Oracle reference root for optional cross-check.",
    )
    parser.add_argument(
        "--skip-results-check",
        action="store_true",
        help="Only inspect task JSON files; do not intersect with planner results.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path to save the summary.",
    )
    return parser.parse_args()


def _task_key_from_task_path(path: Path) -> tuple[str, str, str, str]:
    return (path.parts[-4], path.parts[-2], path.parts[-3], path.name)


def _task_key_from_result_path(path: Path) -> tuple[str, str, str, str]:
    return (path.parts[-7], path.parts[-5], path.parts[-4], f"{path.parts[-3]}.json")


def _task_key_from_oracle_path(path: Path) -> tuple[str, str, str, str]:
    return (path.parts[-4], path.parts[-3], path.parts[-2], path.name)


def iter_task_paths(task_prefix: str, versions: Iterable[str]) -> Iterable[Path]:
    for version in versions:
        pattern = f"{task_prefix}_{version}/*/FloorPlan*/*.json"
        yield from TASK_DIR.glob(pattern)


def collect_semantic_issues(
    task_prefix: str,
    versions: list[str],
) -> tuple[dict[tuple[str, str, str, str], list[dict[str, object]]], Counter]:
    issues_by_file: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    issue_counter: Counter = Counter()

    for task_path in iter_task_paths(task_prefix, versions):
        task_data = json.loads(task_path.read_text(encoding="utf-8"))
        issues = find_held_object_semantic_issues(task_data)
        if not issues:
            continue

        task_key = _task_key_from_task_path(task_path)
        serialized = []
        for issue in issues:
            label = classify_issue_group(issue)
            issue_counter[label] += 1
            serialized.append(
                {
                    "task_name": issue.task_name,
                    "subtask_name": issue.subtask_name,
                    "issue_type": issue.issue_type,
                    "action_index": issue.action_index,
                    "action": issue.action,
                    "receptacle": issue.receptacle,
                    "current_held_object": issue.current_held_object,
                    "group": label,
                }
            )
        issues_by_file[task_key] = serialized

    return issues_by_file, issue_counter


def collect_incomplete_results(
    results_root: Path,
    task_prefix: str,
    versions: list[str],
) -> set[tuple[str, str, str, str]]:
    incomplete: set[tuple[str, str, str, str]] = set()
    for version in versions:
        pattern = DEFAULT_RESULTS_GLOB.format(prefix=task_prefix, version=version)
        for result_path in results_root.glob(pattern):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("completed") is False:
                incomplete.add(_task_key_from_result_path(result_path))
    return incomplete


def collect_invalid_oracle(
    oracle_root: Path,
    task_prefix: str,
    versions: list[str],
) -> set[tuple[str, str, str, str]]:
    invalid: set[tuple[str, str, str, str]] = set()
    for version in versions:
        pattern = f"{task_prefix}_{version}/FloorPlan*/tasks_*/*.json"
        for oracle_path in oracle_root.glob(pattern):
            oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
            if oracle.get("optimal_schedule_time") is None or oracle.get("exact") is False:
                invalid.add(_task_key_from_oracle_path(oracle_path))
    return invalid


def build_summary(args: argparse.Namespace) -> dict[str, object]:
    issues_by_file, issue_counter = collect_semantic_issues(args.task_prefix, args.versions)
    summary: dict[str, object] = {
        "task_prefix": args.task_prefix,
        "versions": list(args.versions),
        "files_with_semantic_issues": len(issues_by_file),
        "issue_groups_all": dict(issue_counter.most_common()),
    }

    if args.skip_results_check:
        return summary

    incomplete = collect_incomplete_results(args.results_root, args.task_prefix, args.versions)
    invalid_oracle = collect_invalid_oracle(args.oracle_root, args.task_prefix, args.versions)
    issue_keys = set(issues_by_file)
    overlap = incomplete & issue_keys

    primary_counter: Counter = Counter()
    for key in sorted(overlap):
        labels = {issue["group"] for issue in issues_by_file[key]}
        primary_label = next(
            (label for label in PRIMARY_LABEL_PRIORITY if label in labels),
            sorted(labels)[0] if labels else None,
        )
        if primary_label is not None:
            primary_counter[primary_label] += 1

    summary.update(
        {
            "incomplete_results": len(incomplete),
            "invalid_oracle_results": len(invalid_oracle),
            "semantic_issue_overlap_with_incomplete": len(overlap),
            "incomplete_without_semantic_issue": len(incomplete - issue_keys),
            "semantic_issue_without_incomplete": len(issue_keys - incomplete),
            "oracle_equals_incomplete": invalid_oracle == incomplete,
            "primary_failure_groups_for_incomplete": dict(primary_counter.most_common()),
        }
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = build_summary(args)

    logger.critical("Semantic issue files: %s", summary["files_with_semantic_issues"])
    for label, count in summary["issue_groups_all"].items():
        logger.critical("  %4d  %s", count, label)

    if not args.skip_results_check:
        logger.critical("Incomplete planner results: %s", summary["incomplete_results"])
        logger.critical(
            "Oracle invalid results: %s (matches incomplete=%s)",
            summary["invalid_oracle_results"],
            summary["oracle_equals_incomplete"],
        )
        logger.critical(
            "Incomplete ∩ semantic issue: %s",
            summary["semantic_issue_overlap_with_incomplete"],
        )
        logger.critical("Primary failure groups among incomplete files:")
        for label, count in summary["primary_failure_groups_for_incomplete"].items():
            logger.critical("  %4d  %s", count, label)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.critical("Saved summary: %s", args.output_json)


if __name__ == "__main__":
    main()
