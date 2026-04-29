# validate_lamma_floorplan.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

from rwh_utils.lamma_combined_validator import (
    LaMMACombinedPlanValidator,
    load_dataset_specs,
    extract_index_from_folder_name,
)


def build_pretty_txt(summary: dict) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("LaMMA-P Combined Plan Validation Summary")
    lines.append("=" * 72)
    lines.append(f"FloorPlan         : {summary['floorplan']}")
    lines.append(f"FloorPlan Dir     : {summary['floorplan_dir']}")
    lines.append(f"Dataset JSON      : {summary['dataset_json']}")
    lines.append("")
    lines.append("[Overall]")
    lines.append(f"  Tasks               : {summary['num_tasks']}")
    lines.append(
        f"  Success             : {summary['success_count']}/{summary['num_tasks']} "
        f"({summary['success_rate']:.3f})"
    )
    lines.append(
        f"  Goal Satisfied      : {summary['goal_satisfied_count']}/{summary['num_tasks']} "
        f"({summary['goal_satisfied_rate']:.3f})"
    )
    lines.append(f"  Average Score       : {summary['avg_score']:.3f}")
    lines.append("")

    if summary.get("violation_counter"):
        lines.append("[Top Violations]")
        for k, v in summary["violation_counter"][:10]:
            lines.append(f"  - {k}: {v}")
        lines.append("")

    lines.append("[Per Task]")
    for i, r in enumerate(summary["results"], start=1):
        lines.append("-" * 72)
        lines.append(f"{i}. {r['task']}")
        lines.append(f"   Folder              : {r['folder']}")
        lines.append(f"   Success             : {r['success']}")
        lines.append(f"   Goal Satisfied      : {r['goal_satisfied']}")
        lines.append(f"   Score               : {r['score']:.3f}")
        lines.append(
            f"   Executable Actions  : {r['executable_actions']}/{r['total_actions']}"
        )
        lines.append(
            f"   Matched Steps       : {r['matched_expected_steps']}/{r['expected_steps']}"
        )

        if r.get("violations"):
            lines.append("   Violations:")
            for v in r["violations"]:
                lines.append(f"     - {v}")
        else:
            lines.append("   Violations          : None")

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_root",
        type=str,
        default="exp"
    )
    parser.add_argument("--floorplan", type=str, required=True)
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    floorplan = args.floorplan

    floorplan_dir = exp_root / "baek_dataset" / floorplan
    dataset_json = exp_root / f"{floorplan.lower()}_tasks.json"

    specs = load_dataset_specs(dataset_json)
    validator = LaMMACombinedPlanValidator(specs)

    folders = sorted(
        [p for p in floorplan_dir.iterdir() if p.is_dir() and p.name.startswith("[")]
    )

    results = []
    success_count = 0
    goal_count = 0
    violation_counter = Counter()

    print(f"=== validating {floorplan_dir} ===")
    print(f"dataset spec: {dataset_json}")

    for folder in folders:
        idx = extract_index_from_folder_name(folder.name)
        if idx is None:
            print(f"[SKIP] {folder.name} | cannot parse task index")
            continue

        task_spec = validator.find_task_spec_by_index(idx)
        report = validator.validate_plan_file(
            folder,
            task_spec,
            robots=["robot1"]
        )
        results.append(report.to_dict())

        if report.success:
            success_count += 1
        if report.goal_satisfied:
            goal_count += 1

        for v in report.violations:
            key = v.split(" -> ")[-1] if " -> " in v else v
            violation_counter[key] += 1

        print(
            f"[TASK {idx}] {report.task} | "
            f"success={report.success} | "
            f"goal={report.goal_satisfied} | "
            f"score={report.score:.3f} | "
            f"exec={report.executable_actions}/{report.total_actions} | "
            f"matched={report.matched_expected_steps}/{report.expected_steps}"
        )

        if report.violations:
            for v in report.violations:
                print(f"   - {v}")

    n = len(results)
    avg_score = sum(r["score"] for r in results) / n if n else 0.0

    summary = {
        "floorplan": floorplan,
        "floorplan_dir": str(floorplan_dir),
        "dataset_json": str(dataset_json),
        "num_tasks": n,
        "success_count": success_count,
        "goal_satisfied_count": goal_count,
        "success_rate": (success_count / n) if n else 0.0,
        "goal_satisfied_rate": (goal_count / n) if n else 0.0,
        "avg_score": avg_score,
        "violation_counter": violation_counter.most_common(),
        "results": results,
    }

    print("=== summary ===")
    print(
        f"tasks={n} | success={success_count}/{n} | "
        f"goal={goal_count}/{n} | avg_score={avg_score:.3f}"
    )

    output_json = floorplan_dir / "lamma_validation_summary.json"
    output_txt = floorplan_dir / "lamma_validation_summary.txt"

    output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    pretty_txt = build_pretty_txt(summary)
    output_txt.write_text(pretty_txt, encoding="utf-8")

    print(f"saved json: {output_json}")
    print(f"saved txt : {output_txt}")


if __name__ == "__main__":
    main()