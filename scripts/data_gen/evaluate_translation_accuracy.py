"""
Evaluate translation validity and cross-trial consistency of nl_to_json across trials.

Metrics
-------
1. Post-refinement validity  — % of (instruction, trial) pairs that pass the full
                               TaskUtil pipeline (object-ID check + constraint refinement).
2. Pre-refinement validity   — % that pass Task.parse_instruction only (JSON structure check).
3. Cross-trial consistency   — % of instructions valid in at least k/N trials.
4. Per-trial breakdown       — validity rate per version (to inspect trial variance).
5. Per-case breakdown        — validity rate per (tasks_N_constraints_M) case.

"Missing file" (generation failed for that trial) is counted as invalid (False),
so every instruction has exactly N boolean values regardless of failures.

Usage
-----
    python scripts/data_gen/evaluate_translation_accuracy.py
    python scripts/data_gen/evaluate_translation_accuracy.py --versions v1 v2 v3
    python scripts/data_gen/evaluate_translation_accuracy.py \\
        --output-csv assets/results/translation_validity.csv
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.common import create_module_logger
from src.utils.config.constants import ASSETS_PATH
from src.utils.task.task_util import TaskUtil

logger = create_module_logger(__name__, module_log=True)

TASK_DIR = ASSETS_PATH / "tasks"
INSTRUCTION_FILE = TASK_DIR / "decomposed_final_revision_metadata_260402.json"
OUTPUT_DIR_PREFIX = "decomposed_final_revision_metadata_260402"

KITCHEN_SCENES = [
    "FloorPlan1",
    "FloorPlan7",
    "FloorPlan13",
    "FloorPlan18",
    "FloorPlan27",
]

CASES = [
    "tasks_2_constraints_1",
    "tasks_2_constraints_2",
    "tasks_3_constraints_1",
    "tasks_3_constraints_2",
]


# ---------------------------------------------------------------------------
# Validity checks
# ---------------------------------------------------------------------------

def check_pre_refinement(task_data: dict) -> bool:
    """True if the JSON can be parsed into Task objects (structure check only)."""
    try:
        from src.models.task import Task
        Task.parse_instruction(task_data)
        return True
    except Exception as e:
        logger.debug(f"Pre-refinement fail: {e}")
        return False


def check_post_refinement(task_data: dict, scene_name: str) -> bool:
    """True if the full TaskUtil pipeline succeeds (object-ID + constraint refinement)."""
    try:
        TaskUtil.build_tasks_and_constraints(
            task_data=task_data,
            scene_file_name=f"{scene_name}_physics_environment.json",
            enable_decomposition=True,
        )
        return True
    except Exception as e:
        logger.debug(f"Post-refinement fail: {e}")
        return False


# ---------------------------------------------------------------------------
# Instruction registry (what files should exist per trial)
# ---------------------------------------------------------------------------

def load_instruction_registry() -> Dict[str, List[str]]:
    """
    Returns { case_key: [instruction, ...] } for CASES.
    Instructions are the 'common' list from the metadata JSON
    (scene-specific lists are merged in, deduplicated).
    This is used to detect missing files (generation failures).
    """
    with open(INSTRUCTION_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    registry: Dict[str, List[str]] = {}
    all_cases = data.get("instructions_by_case", {})

    for case in CASES:
        case_data = all_cases.get(case, {})
        common = case_data.get("common", [])
        scene_specific: List[str] = []
        for scene in KITCHEN_SCENES:
            scene_specific.extend(case_data.get(scene, []))

        seen = set()
        merged: List[str] = []
        for inst in (scene_specific + common):
            if inst not in seen:
                seen.add(inst)
                merged.append(inst)
        registry[case] = merged

    return registry


def _instruction_to_stem(instruction: str) -> str:
    """Convert instruction string to the file stem used by nl_to_json.py."""
    return instruction.replace(" ", "_").replace("'", "")


def _find_file(scene_dir: Path, stem: str) -> Optional[Path]:
    """Find the JSON file whose name matches {index}_{stem}.json."""
    for jf in scene_dir.glob("*.json"):
        parts = jf.stem.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit() and parts[1] == stem:
            return jf
        if jf.stem == stem:
            return jf
    return None


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------

def collect_results(
    versions: List[str],
    registry: Dict[str, List[str]],
) -> Dict[str, Dict[str, Dict[str, Dict[str, List[bool]]]]]:
    """
    Returns:
      results[case][scene][instruction] = {
          "pre":  [bool_v1, bool_v2, ...],
          "post": [bool_v1, bool_v2, ...],
      }
    Missing file for any (version, instruction) -> False for that trial.
    """
    results: Dict = {
        case: {
            scene: {
                inst: {"pre": [], "post": []}
                for inst in registry[case]
            }
            for scene in KITCHEN_SCENES
        }
        for case in CASES
    }

    for version in versions:
        version_dir = TASK_DIR / f"{OUTPUT_DIR_PREFIX}_{version}"
        if not version_dir.exists():
            logger.warning(f"Version dir not found, treating all as invalid: {version_dir}")

        for case in CASES:
            for scene in KITCHEN_SCENES:
                scene_dir = version_dir / case / scene
                scene_exists = scene_dir.exists() if version_dir.exists() else False

                for instruction in registry[case]:
                    stem = _instruction_to_stem(instruction)
                    task_data: Optional[dict] = None

                    if scene_exists:
                        jf = _find_file(scene_dir, stem)
                        if jf:
                            try:
                                with open(jf, "r", encoding="utf-8") as f:
                                    task_data = json.load(f)
                            except Exception as e:
                                logger.debug(f"JSON load error {jf}: {e}")

                    if task_data is None:
                        results[case][scene][instruction]["pre"].append(False)
                        results[case][scene][instruction]["post"].append(False)
                    else:
                        results[case][scene][instruction]["pre"].append(
                            check_pre_refinement(task_data)
                        )
                        results[case][scene][instruction]["post"].append(
                            check_post_refinement(task_data, scene)
                        )

    return results


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    results: Dict,
    versions: List[str],
    registry: Dict[str, List[str]],
) -> Dict:
    N = len(versions)

    trial_pre_valid = [0] * N
    trial_post_valid = [0] * N
    trial_total = [0] * N

    # consistency_buckets[k] = number of instructions valid in exactly k trials
    post_buckets: Dict[int, int] = defaultdict(int)
    pre_buckets: Dict[int, int] = defaultdict(int)

    total_instructions = 0
    total_pre_valid = 0
    total_post_valid = 0

    per_case_stats: Dict[str, Dict] = {}

    for case in CASES:
        case_pre = 0
        case_post = 0
        case_samples = 0

        for scene in KITCHEN_SCENES:
            for instruction in registry[case]:
                pre_bools = results[case][scene][instruction]["pre"]
                post_bools = results[case][scene][instruction]["post"]

                total_instructions += 1
                pre_v = sum(pre_bools)
                post_v = sum(post_bools)

                total_pre_valid += pre_v
                total_post_valid += post_v
                case_pre += pre_v
                case_post += post_v
                case_samples += N

                for i in range(N):
                    trial_total[i] += 1
                    if pre_bools[i]:
                        trial_pre_valid[i] += 1
                    if post_bools[i]:
                        trial_post_valid[i] += 1

                post_buckets[post_v] += 1
                pre_buckets[pre_v] += 1

        per_case_stats[case] = {
            "instructions": len(registry[case]) * len(KITCHEN_SCENES),
            "total_samples": case_samples,
            "pre_valid": case_pre,
            "post_valid": case_post,
            "pre_rate": case_pre / case_samples if case_samples else 0.0,
            "post_rate": case_post / case_samples if case_samples else 0.0,
        }

    total_samples = total_instructions * N

    per_trial_stats = []
    for i, version in enumerate(versions):
        t = trial_total[i]
        per_trial_stats.append({
            "version": version,
            "total": t,
            "pre_valid": trial_pre_valid[i],
            "post_valid": trial_post_valid[i],
            "pre_rate": trial_pre_valid[i] / t if t else 0.0,
            "post_rate": trial_post_valid[i] / t if t else 0.0,
        })

    # Fraction of instructions valid in >= k trials
    consistency_at_k = {}
    for k in range(1, N + 1):
        count = sum(v for kk, v in post_buckets.items() if kk >= k)
        consistency_at_k[k] = count / total_instructions if total_instructions else 0.0

    return {
        "n_versions": N,
        "total_instructions": total_instructions,
        "total_samples": total_samples,
        "pre_rate": total_pre_valid / total_samples if total_samples else 0.0,
        "total_pre_valid": total_pre_valid,
        "post_rate": total_post_valid / total_samples if total_samples else 0.0,
        "total_post_valid": total_post_valid,
        "consistency_at_k": consistency_at_k,
        "post_buckets": dict(post_buckets),
        "pre_buckets": dict(pre_buckets),
        "per_trial": per_trial_stats,
        "per_case": per_case_stats,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(metrics: Dict, versions: List[str]) -> None:
    N = metrics["n_versions"]
    print("\n" + "=" * 65)
    print("  Translation Validity and Cross-Trial Consistency")
    print("=" * 65)
    print(f"Trials             : {versions}")
    print(f"Unique instructions: {metrics['total_instructions']}")
    print(f"Total samples      : {metrics['total_samples']}")
    print()
    print("Validity rates (pooled across all trials):")
    print(f"  Pre-refinement  (JSON structure) : {metrics['pre_rate']:.1%}"
          f"  ({metrics['total_pre_valid']}/{metrics['total_samples']})")
    print(f"  Post-refinement (full pipeline)  : {metrics['post_rate']:.1%}"
          f"  ({metrics['total_post_valid']}/{metrics['total_samples']})")
    print()
    print(f"Cross-trial consistency (post, out of {metrics['total_instructions']} instructions):")
    for k in range(N, 0, -1):
        count = sum(v for kk, v in metrics["post_buckets"].items() if kk >= k)
        label = f"ALL {N}/{N}" if k == N else f">= {k}/{N}"
        print(f"  {label} trials valid: {metrics['consistency_at_k'][k]:.1%}"
              f"  ({count}/{metrics['total_instructions']})")
    print()
    print("Per-trial (post-refinement):")
    for t in metrics["per_trial"]:
        print(f"  {t['version']}: {t['post_valid']}/{t['total']}  ({t['post_rate']:.1%})")
    print()
    print("Per-case (post-refinement):")
    for case, s in metrics["per_case"].items():
        print(f"  {case}: {s['post_valid']}/{s['total_samples']}  ({s['post_rate']:.1%})")
    print("=" * 65 + "\n")


def save_csv(metrics: Dict, versions: List[str], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    N = metrics["n_versions"]
    rows: List[List] = [
        ["type", "name", "pre_valid", "post_valid", "total", "pre_rate", "post_rate"]
    ]
    rows.append([
        "overall", "all",
        metrics["total_pre_valid"], metrics["total_post_valid"],
        metrics["total_samples"],
        f"{metrics['pre_rate']:.4f}", f"{metrics['post_rate']:.4f}",
    ])
    for k in range(1, N + 1):
        count = sum(v for kk, v in metrics["post_buckets"].items() if kk >= k)
        rows.append([
            "consistency", f"at_least_{k}_of_{N}",
            "", count, metrics["total_instructions"], "",
            f"{metrics['consistency_at_k'][k]:.4f}",
        ])
    for t in metrics["per_trial"]:
        rows.append([
            "trial", t["version"],
            t["pre_valid"], t["post_valid"], t["total"],
            f"{t['pre_rate']:.4f}", f"{t['post_rate']:.4f}",
        ])
    for case, s in metrics["per_case"].items():
        rows.append([
            "case", case,
            s["pre_valid"], s["post_valid"], s["total_samples"],
            f"{s['pre_rate']:.4f}", f"{s['post_rate']:.4f}",
        ])

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    logger.info(f"CSV saved: {output_csv}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate translation validity and cross-trial consistency of nl_to_json."
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["v1", "v2", "v3", "v4", "v5"],
        help="Trial version suffixes to evaluate (default: v1 v2 v3 v4 v5).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save CSV summary.",
    )
    args = parser.parse_args()

    logger.info(f"Evaluating versions: {args.versions}")
    registry = load_instruction_registry()
    results = collect_results(args.versions, registry)
    metrics = compute_metrics(results, args.versions, registry)
    print_report(metrics, args.versions)

    if args.output_csv:
        save_csv(metrics, args.versions, args.output_csv)


if __name__ == "__main__":
    main()
