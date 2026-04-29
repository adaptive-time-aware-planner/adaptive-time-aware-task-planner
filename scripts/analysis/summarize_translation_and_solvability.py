"""Summarize translation reliability and downstream solvability.

This script is designed for paper- and rebuttal-facing analysis. It combines:

1. Translation reliability across repeated LLM translation trials
   - task-count accuracy against the intended T2/T3 case label
   - critical-constraint-count accuracy against the intended C1/C2 case label
   - semantic validity of primitive action sequences
   - task-level consistency across repeated trials
   - exact critical-constraint signature consistency across repeated trials

2. Suite-level solvability summaries for saved run_experiment_suite outputs
   - oracle solvability on translation-valid instances
   - per-config scheduler completion counts
   - per-config evaluation-pool size:
       translation-valid ∩ oracle-solvable ∩ scheduler-completed
   - common evaluation-pool size across all configs in a suite

Example:
    python scripts/analysis/summarize_translation_and_solvability.py \
        --suite-root assets/results/offline_exp_result/offline_batch_scalability_0421_failed \
        --suite-root assets/results/offline_exp_result/offline_batch_eta_sensitivity_0421_failed \
        --output-json /tmp/translation_solvability_summary.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

_PRIMITIVE_MODULE_PATH = (
    PROJECT_ROOT / "src" / "utils" / "task" / "primitive_action_semantics.py"
)
_primitive_spec = importlib.util.spec_from_file_location(
    "primitive_action_semantics",
    _PRIMITIVE_MODULE_PATH,
)
if _primitive_spec is None or _primitive_spec.loader is None:
    raise ImportError(f"Unable to load {_PRIMITIVE_MODULE_PATH}")
_primitive_module = importlib.util.module_from_spec(_primitive_spec)
sys.modules[_primitive_spec.name] = _primitive_module
_primitive_spec.loader.exec_module(_primitive_module)
classify_issue_group = _primitive_module.classify_issue_group
find_first_task_sequence_issue = _primitive_module.find_first_task_sequence_issue

ASSETS_PATH = PROJECT_ROOT / "assets"
TASK_DIR = ASSETS_PATH / "tasks"
DEFAULT_TASK_PREFIX = "decomposed_final_revision_metadata_260402"
DEFAULT_VERSIONS = ["v1", "v2", "v3", "v4", "v5"]
DEFAULT_ORACLE_ROOT = ASSETS_PATH / "results" / "offline_oracle_reference"
DEFAULT_SUITE_ROOTS = [
    ASSETS_PATH / "results" / "offline_exp_result" / "offline_batch_scalability_0421_failed",
    ASSETS_PATH / "results" / "offline_exp_result" / "offline_batch_eta_sensitivity_0421_failed",
]
CASE_RE = re.compile(r"tasks_(\d+)_constraints_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize translation reliability and suite solvability."
    )
    parser.add_argument(
        "--task-prefix",
        default=DEFAULT_TASK_PREFIX,
        help="Base task folder prefix. Trial folders are resolved as <prefix>_v1, ...",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=DEFAULT_VERSIONS,
        help="Translation / oracle / result trial suffixes to inspect.",
    )
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=DEFAULT_ORACLE_ROOT,
        help="Root directory containing offline oracle reference JSON files.",
    )
    parser.add_argument(
        "--suite-root",
        dest="suite_roots",
        type=Path,
        action="append",
        default=None,
        help="Offline experiment suite root to summarize. May be repeated.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the full machine-readable summary.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Optional path to save markdown tables.",
    )
    return parser.parse_args()


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _metric(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "count": numerator,
        "denominator": denominator,
        "rate": _rate(numerator, denominator),
    }


def _parse_case_expectation(case_name: str) -> tuple[int, int]:
    match = CASE_RE.fullmatch(case_name)
    if match is None:
        raise ValueError(f"Unexpected case name: {case_name}")
    return int(match.group(1)), int(match.group(2))


def _instruction_to_stem(instruction: str) -> str:
    return instruction.replace(" ", "_").replace("'", "")


def _find_task_file(scene_dir: Path, stem: str) -> Path | None:
    for json_file in scene_dir.glob("*.json"):
        parts = json_file.stem.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit() and parts[1] == stem:
            return json_file
        if json_file.stem == stem:
            return json_file
    return None


def load_instruction_registry(
    task_prefix: str,
    reference_version: str,
) -> dict[tuple[str, str], list[str]]:
    """Return the expected (scene, case) -> instruction stem registry.

    We intentionally use an existing translated task folder as the registry source
    instead of the metadata JSON. The saved task files reflect the actual
    instruction inventory used by the experiments (360 instances per version for
    the v1--v5 decomposed sets), while the metadata file can include broader
    scene-specific bookkeeping lists that are not all instantiated for each scene.
    """

    version_dir = TASK_DIR / f"{task_prefix}_{reference_version}"
    if not version_dir.exists():
        raise FileNotFoundError(f"Reference task directory missing: {version_dir}")

    registry: dict[tuple[str, str], list[str]] = {}
    for case_dir in sorted(version_dir.glob("tasks_*")):
        if not case_dir.is_dir():
            continue
        for scene_dir in sorted(case_dir.glob("FloorPlan*")):
            if not scene_dir.is_dir():
                continue
            stems = sorted(path.stem for path in scene_dir.glob("*.json"))
            registry[(scene_dir.name, case_dir.name)] = stems

    return registry


def _critical_constraints(task_data: object) -> list[tuple[str, str, str, float, bool]]:
    constraints: list[tuple[str, str, str, float, bool]] = []
    if not isinstance(task_data, list):
        return constraints

    for task in task_data:
        if not isinstance(task, dict):
            continue
        for subtask in task.get("Subtasks", []):
            if not isinstance(subtask, dict):
                continue
            target_name = str(subtask.get("Name", ""))
            for constraint in subtask.get("TemporalConstraints") or []:
                if not isinstance(constraint, dict):
                    continue
                interval = float(constraint.get("Interval", 0.0) or 0.0)
                urgency = bool(constraint.get("Urgency", False))
                if urgency and interval > 0.0:
                    constraints.append(
                        (
                            str(constraint.get("Subtask", "")),
                            target_name,
                            str(constraint.get("Type", "")),
                            interval,
                            True,
                        )
                    )
    constraints.sort()
    return constraints


def _count_tasks(task_data: object) -> int:
    if not isinstance(task_data, list):
        return 0
    return sum(1 for task in task_data if isinstance(task, dict))


def summarize_translation(
    task_prefix: str,
    versions: list[str],
) -> tuple[dict[str, Any], set[tuple[str, str, str, str]]]:
    registry = load_instruction_registry(task_prefix, versions[0])

    per_case_counts: dict[str, Counter] = defaultdict(Counter)
    per_version_counts: dict[str, Counter] = defaultdict(Counter)
    issue_groups: Counter[str] = Counter()

    semantic_by_instruction: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    constraint_sig_by_instruction: dict[
        tuple[str, str, str], list[tuple[tuple[str, str, str, float, bool], ...] | None]
    ] = defaultdict(list)

    translation_valid_instances: set[tuple[str, str, str, str]] = set()
    total_pairs = 0
    total_instructions = 0

    for (scene, case_name), instruction_stems in sorted(registry.items()):
        for instruction_stem in instruction_stems:
                total_instructions += 1
                expected_tasks, expected_constraints = _parse_case_expectation(case_name)

                for version in versions:
                    total_pairs += 1
                    pair_key = (version, scene, case_name, instruction_stem)
                    instruction_key = (scene, case_name, instruction_stem)
                    version_dir = TASK_DIR / f"{task_prefix}_{version}" / case_name / scene
                    json_file = version_dir / f"{instruction_stem}.json"

                    counts = per_case_counts[case_name]
                    version_counts = per_version_counts[version]
                    counts["pairs"] += 1
                    version_counts["pairs"] += 1

                    if json_file is None:
                        semantic_by_instruction[instruction_key].append(False)
                        constraint_sig_by_instruction[instruction_key].append(None)
                        counts["missing"] += 1
                        version_counts["missing"] += 1
                        continue

                    task_data = json.loads(json_file.read_text(encoding="utf-8"))
                    actual_tasks = _count_tasks(task_data)
                    actual_constraints = len(_critical_constraints(task_data))
                    task_count_ok = actual_tasks == expected_tasks
                    constraint_count_ok = actual_constraints == expected_constraints

                    issue = find_first_task_sequence_issue(task_data)
                    semantic_ok = issue is None
                    signature = tuple(_critical_constraints(task_data))

                    if semantic_ok:
                        translation_valid_instances.add(pair_key)
                    else:
                        issue_groups[classify_issue_group(issue)] += 1

                    counts["task_count_ok"] += int(task_count_ok)
                    counts["constraint_count_ok"] += int(constraint_count_ok)
                    counts["both_structural_ok"] += int(task_count_ok and constraint_count_ok)
                    counts["semantic_ok"] += int(semantic_ok)

                    version_counts["task_count_ok"] += int(task_count_ok)
                    version_counts["constraint_count_ok"] += int(constraint_count_ok)
                    version_counts["both_structural_ok"] += int(
                        task_count_ok and constraint_count_ok
                    )
                    version_counts["semantic_ok"] += int(semantic_ok)

                    semantic_by_instruction[instruction_key].append(semantic_ok)
                    constraint_sig_by_instruction[instruction_key].append(signature)

    semantic_valid_counts = Counter(sum(values) for values in semantic_by_instruction.values())
    exact_constraint_sig_counts = Counter(
        int(len(set(signatures)) == 1) if signatures else 0
        for signatures in constraint_sig_by_instruction.values()
    )
    exact_constraint_sig_counts  # keep linter quiet when optimized away

    semantic_at_least = {
        f"valid_in_at_least_{k}_of_{len(versions)}": _metric(
            sum(1 for values in semantic_by_instruction.values() if sum(values) >= k),
            total_instructions,
        )
        for k in range(1, len(versions) + 1)
    }
    semantic_exact = {
        f"valid_in_exactly_{k}_of_{len(versions)}": _metric(
            semantic_valid_counts.get(k, 0),
            total_instructions,
        )
        for k in range(0, len(versions) + 1)
    }

    constraint_sig_at_least = {}
    for k in range(1, len(versions) + 1):
        count = 0
        for signatures in constraint_sig_by_instruction.values():
            signature_counter = Counter(signatures)
            if signature_counter and signature_counter.most_common(1)[0][1] >= k:
                count += 1
        constraint_sig_at_least[f"exact_signature_in_at_least_{k}_of_{len(versions)}"] = (
            _metric(count, total_instructions)
        )

    translation_summary = {
        "n_versions": len(versions),
        "n_instruction_trial_pairs": total_pairs,
        "n_unique_instruction_instances": total_instructions,
        "task_count_accuracy": _metric(
            sum(counter["task_count_ok"] for counter in per_case_counts.values()),
            total_pairs,
        ),
        "critical_constraint_count_accuracy": _metric(
            sum(counter["constraint_count_ok"] for counter in per_case_counts.values()),
            total_pairs,
        ),
        "structural_both_accuracy": _metric(
            sum(counter["both_structural_ok"] for counter in per_case_counts.values()),
            total_pairs,
        ),
        "semantic_translation_validity": _metric(
            len(translation_valid_instances),
            total_pairs,
        ),
        "semantic_issue_groups": dict(issue_groups.most_common()),
        "per_case": {
            case_name: {
                "pairs": counter["pairs"],
                "task_count_accuracy": _metric(counter["task_count_ok"], counter["pairs"]),
                "critical_constraint_count_accuracy": _metric(
                    counter["constraint_count_ok"], counter["pairs"]
                ),
                "structural_both_accuracy": _metric(
                    counter["both_structural_ok"], counter["pairs"]
                ),
                "semantic_translation_validity": _metric(
                    counter["semantic_ok"], counter["pairs"]
                ),
                "missing_files": counter["missing"],
            }
            for case_name, counter in sorted(per_case_counts.items())
        },
        "per_version": {
            version: {
                "pairs": counter["pairs"],
                "task_count_accuracy": _metric(counter["task_count_ok"], counter["pairs"]),
                "critical_constraint_count_accuracy": _metric(
                    counter["constraint_count_ok"], counter["pairs"]
                ),
                "structural_both_accuracy": _metric(
                    counter["both_structural_ok"], counter["pairs"]
                ),
                "semantic_translation_validity": _metric(
                    counter["semantic_ok"], counter["pairs"]
                ),
                "missing_files": counter["missing"],
            }
            for version, counter in sorted(per_version_counts.items())
        },
        "task_level_semantic_consistency": {
            **semantic_at_least,
            **semantic_exact,
        },
        "critical_constraint_signature_consistency": constraint_sig_at_least,
    }
    return translation_summary, translation_valid_instances


def load_oracle_records(
    oracle_root: Path,
    task_prefix: str,
    versions: list[str],
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    records: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for version in versions:
        version_dir = oracle_root / f"{task_prefix}_{version}"
        if not version_dir.exists():
            logger.warning("Oracle version directory missing: %s", version_dir)
            continue

        for oracle_path in version_dir.rglob("*.json"):
            rel = oracle_path.relative_to(version_dir)
            parts = rel.parts
            if len(parts) < 3:
                continue
            scene, case_name, filename = parts[-3], parts[-2], parts[-1]
            instruction_stem = Path(filename).stem
            data = json.loads(oracle_path.read_text(encoding="utf-8"))
            records[(version, scene, case_name, instruction_stem)] = {
                "exact": bool(data.get("exact", False)),
                "optimal_schedule_time": data.get("optimal_schedule_time"),
                "computation_time": data.get("computation_time"),
            }

    return records


def load_suite_results(
    suite_root: Path,
    task_prefix: str,
    versions: list[str],
) -> dict[str, dict[tuple[str, str, str, str], dict[str, Any]]]:
    configs: dict[str, dict[tuple[str, str, str, str], dict[str, Any]]] = defaultdict(dict)

    for version in versions:
        version_dir = suite_root / f"{task_prefix}_{version}"
        if not version_dir.exists():
            logger.warning("Suite version directory missing: %s", version_dir)
            continue

        for result_path in version_dir.rglob("*.json"):
            rel = result_path.relative_to(version_dir)
            parts = rel.parts
            if len(parts) < 5:
                continue

            scene = parts[-5]
            case_name = parts[-4]
            instruction_stem = parts[-3]
            config_key = "/".join(parts[:-5] + parts[-2:])

            try:
                data = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            if not isinstance(data, dict) or "completed" not in data:
                continue

            configs[config_key][(version, scene, case_name, instruction_stem)] = {
                "completed": bool(data.get("completed", False)),
                "abort_reason": data.get("abort_reason"),
                "scheduler_makespan": data.get("scheduler_makespan"),
                "timing_success_rate_sched": data.get("timing_success_rate_sched"),
                "computation_time": data.get("computation_time"),
            }

    return dict(configs)


def summarize_suite(
    suite_root: Path,
    task_prefix: str,
    versions: list[str],
    translation_valid_instances: set[tuple[str, str, str, str]],
    oracle_records: dict[tuple[str, str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    configs = load_suite_results(suite_root, task_prefix, versions)

    oracle_solved_instances = {
        instance
        for instance, record in oracle_records.items()
        if record.get("exact") and record.get("optimal_schedule_time") is not None
    }

    config_summaries: dict[str, Any] = {}
    completion_sets: list[set[tuple[str, str, str, str]]] = []
    present_sets: list[set[tuple[str, str, str, str]]] = []

    for config_key, records in sorted(configs.items()):
        present_instances = set(records)
        completed_instances = {
            instance for instance, record in records.items() if record.get("completed")
        }
        eval_pool = translation_valid_instances & oracle_solved_instances & completed_instances
        oracle_reference_pool = translation_valid_instances & oracle_solved_instances & present_instances

        present_sets.append(present_instances)
        completion_sets.append(completed_instances)

        config_summaries[config_key] = {
            "n_result_files": len(records),
            "n_present_instances": len(present_instances),
            "n_completed_instances": len(completed_instances),
            "n_translation_valid_overlap": len(translation_valid_instances & present_instances),
            "n_oracle_solved_overlap": len(oracle_solved_instances & present_instances),
            "n_reference_eval_pool": len(oracle_reference_pool),
            "n_eval_pool": len(eval_pool),
            "completion_rate_on_translation_valid_overlap": _rate(
                len(translation_valid_instances & completed_instances),
                len(translation_valid_instances & present_instances),
            ),
            "coverage_on_oracle_solved_translation_valid": _rate(
                len(eval_pool),
                len(oracle_reference_pool),
            ),
            "n_oracle_none_but_completed": len(
                completed_instances & present_instances - oracle_solved_instances
            ),
            "n_translation_invalid_but_completed": len(
                completed_instances & present_instances - translation_valid_instances
            ),
        }

    suite_present_union = set().union(*present_sets) if present_sets else set()
    suite_completed_intersection = (
        set.intersection(*completion_sets) if completion_sets else set()
    )
    suite_present_intersection = set.intersection(*present_sets) if present_sets else set()
    suite_reference_pool = translation_valid_instances & oracle_solved_instances & suite_present_union
    suite_common_eval_pool = (
        translation_valid_instances & oracle_solved_instances & suite_completed_intersection
    )

    return {
        "suite_root": str(suite_root),
        "n_configs": len(configs),
        "n_instances_with_any_result": len(suite_present_union),
        "n_instances_present_in_all_configs": len(suite_present_intersection),
        "n_reference_pool_with_any_result": len(suite_reference_pool),
        "n_common_eval_pool_across_all_configs": len(suite_common_eval_pool),
        "configs": config_summaries,
    }


def render_console_summary(summary: dict[str, Any]) -> str:
    translation = summary["translation"]
    n_versions = translation["n_versions"]
    lines = [
        "== Translation Reliability ==",
        (
            "Task-count accuracy: "
            f"{translation['task_count_accuracy']['count']}/"
            f"{translation['task_count_accuracy']['denominator']} "
            f"({translation['task_count_accuracy']['rate']:.1%})"
        ),
        (
            "Critical-constraint-count accuracy: "
            f"{translation['critical_constraint_count_accuracy']['count']}/"
            f"{translation['critical_constraint_count_accuracy']['denominator']} "
            f"({translation['critical_constraint_count_accuracy']['rate']:.1%})"
        ),
        (
            "Structural both-correct: "
            f"{translation['structural_both_accuracy']['count']}/"
            f"{translation['structural_both_accuracy']['denominator']} "
            f"({translation['structural_both_accuracy']['rate']:.1%})"
        ),
        (
            "Semantic translation validity: "
            f"{translation['semantic_translation_validity']['count']}/"
            f"{translation['semantic_translation_validity']['denominator']} "
            f"({translation['semantic_translation_validity']['rate']:.1%})"
        ),
    ]

    semantic_consistency = translation["task_level_semantic_consistency"]
    lines.extend(
        [
            (
                f"Task-level semantic consistency ({n_versions}/{n_versions} valid): "
                f"{semantic_consistency[f'valid_in_exactly_{n_versions}_of_{n_versions}']['count']}/"
                f"{semantic_consistency[f'valid_in_exactly_{n_versions}_of_{n_versions}']['denominator']} "
                f"({semantic_consistency[f'valid_in_exactly_{n_versions}_of_{n_versions}']['rate']:.1%})"
            ),
            (
                f"Task-level semantic consistency (>={min(3, n_versions)}/{n_versions} valid): "
                f"{semantic_consistency[f'valid_in_at_least_{min(3, n_versions)}_of_{n_versions}']['count']}/"
                f"{semantic_consistency[f'valid_in_at_least_{min(3, n_versions)}_of_{n_versions}']['denominator']} "
                f"({semantic_consistency[f'valid_in_at_least_{min(3, n_versions)}_of_{n_versions}']['rate']:.1%})"
            ),
        ]
    )

    constraint_consistency = translation["critical_constraint_signature_consistency"]
    lines.extend(
        [
            (
                f"Constraint signature consistency ({n_versions}/{n_versions} exact): "
                f"{constraint_consistency[f'exact_signature_in_at_least_{n_versions}_of_{n_versions}']['count']}/"
                f"{constraint_consistency[f'exact_signature_in_at_least_{n_versions}_of_{n_versions}']['denominator']} "
                f"({constraint_consistency[f'exact_signature_in_at_least_{n_versions}_of_{n_versions}']['rate']:.1%})"
            ),
            (
                f"Constraint signature consistency (>={max(1, n_versions - 1)}/{n_versions} exact): "
                f"{constraint_consistency[f'exact_signature_in_at_least_{max(1, n_versions - 1)}_of_{n_versions}']['count']}/"
                f"{constraint_consistency[f'exact_signature_in_at_least_{max(1, n_versions - 1)}_of_{n_versions}']['denominator']} "
                f"({constraint_consistency[f'exact_signature_in_at_least_{max(1, n_versions - 1)}_of_{n_versions}']['rate']:.1%})"
            ),
        ]
    )

    if translation["semantic_issue_groups"]:
        lines.append("Top semantic issue groups:")
        for label, count in list(translation["semantic_issue_groups"].items())[:5]:
            lines.append(f"  - {label}: {count}")

    lines.append("")
    lines.append("== Suite Solvability ==")
    for suite_name, suite_summary in summary["suites"].items():
        lines.append(
            f"{suite_name}: configs={suite_summary['n_configs']}, "
            f"reference_pool={suite_summary['n_reference_pool_with_any_result']}, "
            f"common_eval_all_configs={suite_summary['n_common_eval_pool_across_all_configs']}"
        )
    return "\n".join(lines)


def _format_rate(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def render_markdown_tables(summary: dict[str, Any]) -> str:
    translation = summary["translation"]
    lines: list[str] = []

    lines.extend(
        [
            "## Translation Reliability",
            "",
            "| Metric | Count | Rate |",
            "|---|---:|---:|",
            (
                "| Task-count accuracy | "
                f"{translation['task_count_accuracy']['count']} / "
                f"{translation['task_count_accuracy']['denominator']} | "
                f"{_format_rate(translation['task_count_accuracy']['rate'])} |"
            ),
            (
                "| Critical-constraint-count accuracy | "
                f"{translation['critical_constraint_count_accuracy']['count']} / "
                f"{translation['critical_constraint_count_accuracy']['denominator']} | "
                f"{_format_rate(translation['critical_constraint_count_accuracy']['rate'])} |"
            ),
            (
                "| Structural both-correct | "
                f"{translation['structural_both_accuracy']['count']} / "
                f"{translation['structural_both_accuracy']['denominator']} | "
                f"{_format_rate(translation['structural_both_accuracy']['rate'])} |"
            ),
            (
                "| Semantic translation validity | "
                f"{translation['semantic_translation_validity']['count']} / "
                f"{translation['semantic_translation_validity']['denominator']} | "
                f"{_format_rate(translation['semantic_translation_validity']['rate'])} |"
            ),
            (
                "| Task-level semantic consistency (5/5 valid) | "
                f"{translation['task_level_semantic_consistency']['valid_in_exactly_5_of_5']['count']} / "
                f"{translation['task_level_semantic_consistency']['valid_in_exactly_5_of_5']['denominator']} | "
                f"{_format_rate(translation['task_level_semantic_consistency']['valid_in_exactly_5_of_5']['rate'])} |"
            ),
            (
                "| Task-level semantic consistency (>=3/5 valid) | "
                f"{translation['task_level_semantic_consistency']['valid_in_at_least_3_of_5']['count']} / "
                f"{translation['task_level_semantic_consistency']['valid_in_at_least_3_of_5']['denominator']} | "
                f"{_format_rate(translation['task_level_semantic_consistency']['valid_in_at_least_3_of_5']['rate'])} |"
            ),
            (
                "| Constraint signature consistency (5/5 exact) | "
                f"{translation['critical_constraint_signature_consistency']['exact_signature_in_at_least_5_of_5']['count']} / "
                f"{translation['critical_constraint_signature_consistency']['exact_signature_in_at_least_5_of_5']['denominator']} | "
                f"{_format_rate(translation['critical_constraint_signature_consistency']['exact_signature_in_at_least_5_of_5']['rate'])} |"
            ),
            (
                "| Constraint signature consistency (>=4/5 exact) | "
                f"{translation['critical_constraint_signature_consistency']['exact_signature_in_at_least_4_of_5']['count']} / "
                f"{translation['critical_constraint_signature_consistency']['exact_signature_in_at_least_4_of_5']['denominator']} | "
                f"{_format_rate(translation['critical_constraint_signature_consistency']['exact_signature_in_at_least_4_of_5']['rate'])} |"
            ),
            "",
            "## Translation Reliability by Case",
            "",
            "| Case | Task Count | Constraint Count | Structural Both | Semantic Validity |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for case_name, case_stats in translation["per_case"].items():
        lines.append(
            f"| {case_name} | "
            f"{_format_rate(case_stats['task_count_accuracy']['rate'])} | "
            f"{_format_rate(case_stats['critical_constraint_count_accuracy']['rate'])} | "
            f"{_format_rate(case_stats['structural_both_accuracy']['rate'])} | "
            f"{_format_rate(case_stats['semantic_translation_validity']['rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Suite Solvability",
            "",
            "| Suite | Configs | Reference Pool | Common Eval Pool (All Configs) |",
            "|---|---:|---:|---:|",
        ]
    )
    for suite_name, suite_stats in summary["suites"].items():
        lines.append(
            f"| {suite_name} | {suite_stats['n_configs']} | "
            f"{suite_stats['n_reference_pool_with_any_result']} | "
            f"{suite_stats['n_common_eval_pool_across_all_configs']} |"
        )

    lines.extend(
        [
            "",
            "## Per-Config Solvability",
            "",
            "| Suite | Config | Present | Completed | Reference Eval Pool | Eval Pool | Oracle Coverage |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for suite_name, suite_stats in summary["suites"].items():
        for config_name, config_stats in suite_stats["configs"].items():
            lines.append(
                f"| {suite_name} | {config_name} | "
                f"{config_stats['n_present_instances']} | "
                f"{config_stats['n_completed_instances']} | "
                f"{config_stats['n_reference_eval_pool']} | "
                f"{config_stats['n_eval_pool']} | "
                f"{_format_rate(config_stats['coverage_on_oracle_solved_translation_valid'])} |"
            )

    if translation["semantic_issue_groups"]:
        lines.extend(
            [
                "",
                "## Semantic Failure Breakdown",
                "",
                "| Issue Group | Count |",
                "|---|---:|",
            ]
        )
        for issue_group, count in translation["semantic_issue_groups"].items():
            lines.append(f"| {issue_group} | {count} |")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    suite_roots = args.suite_roots or list(DEFAULT_SUITE_ROOTS)

    translation_summary, translation_valid_instances = summarize_translation(
        task_prefix=args.task_prefix,
        versions=args.versions,
    )
    oracle_records = load_oracle_records(
        oracle_root=args.oracle_root,
        task_prefix=args.task_prefix,
        versions=args.versions,
    )

    full_summary = {
        "task_prefix": args.task_prefix,
        "versions": list(args.versions),
        "translation": translation_summary,
        "oracle": {
            "n_oracle_records": len(oracle_records),
            "n_oracle_solved_instances": sum(
                1
                for record in oracle_records.values()
                if record.get("exact") and record.get("optimal_schedule_time") is not None
            ),
        },
        "suites": {
            suite_root.name: summarize_suite(
                suite_root=suite_root,
                task_prefix=args.task_prefix,
                versions=args.versions,
                translation_valid_instances=translation_valid_instances,
                oracle_records=oracle_records,
            )
            for suite_root in suite_roots
        },
    }

    print(render_console_summary(full_summary))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(full_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSaved full summary to {args.output_json}")

    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            render_markdown_tables(full_summary),
            encoding="utf-8",
        )
        print(f"Saved markdown tables to {args.markdown_output}")


if __name__ == "__main__":
    main()
