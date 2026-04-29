"""Render monitoring-budget LaTeX table from offline analysis summary.

Sweeps max_monitoring_per_critical_intervals in {1, 2, 3, uncap} under
CORRECT_ESTIMATE / bayesian / DEFAULT / w10_d10 / eta0.1 / gt_gaussian.

Output:
    monitoring_budget_by_case.tex

Usage::

    python -m assets.result_analysis.monitoring_budget_table
    python -m assets.result_analysis.monitoring_budget_table \\
        --summary path/to/offline_analysis_summary.json \\
        --out-dir path/to/latex_tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assets.result_analysis.utils.multi_source import (
    load_json_objects,
    mean_std,
    resolve_source_paths,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SUMMARY = (
    "assets/results/offline_exp_result/analysis/monitoring_budget/"
    "offline_analysis_summary.json"
)
DEFAULT_OUT_DIR = (
    "assets/results/offline_exp_result/analysis/monitoring_budget/latex_tables"
)

BUDGET_ORDER: list[str] = ["1", "2", "3", "uncap"]
BUDGET_LABEL: dict[str, str] = {
    "1": "1",
    "2": "2",
    "3": "3",
    "uncap": r"$\infty$",
}

CASES: list[tuple[str, str]] = [
    ("tasks_2_constraints_1", "T2C1"),
    ("tasks_2_constraints_2", "T2C2"),
    ("tasks_3_constraints_1", "T3C1"),
    ("tasks_3_constraints_2", "T3C2"),
]


def _setting_key(budget: str) -> str:
    return (
        f"CORRECT_ESTIMATE__bayesian__DEFAULT__w10_d10__eta0.1__gtgaussian__mb{budget}"
    )


# ---------------------------------------------------------------------------
# Helpers (shared with eta_sensitivity_tables)
# ---------------------------------------------------------------------------

def _tex(
    mean: float | None,
    std: float | None,
    nd: int = 1,
    *,
    bold: bool = False,
    underline: bool = False,
) -> str:
    if mean is None:
        return "--"
    std_value = 0.0 if std is None else std
    s = f"{mean:.{nd}f} \\pm {std_value:.{nd}f}"
    if bold:
        return rf"{{\boldmath ${s}$}}"
    if underline:
        return rf"$\underline{{{s}}}$"
    return f"${s}$"


def _rank_format(
    values: list[tuple[float | None, float | None]],
    nd: int = 1,
    *,
    higher_is_better: bool,
) -> list[str]:
    eps = 1e-9
    sign = -1.0 if higher_is_better else 1.0
    present = [v[0] for v in values if v[0] is not None]
    sorted_unique = sorted({sign * v for v in present})
    rank0 = sorted_unique[0] if sorted_unique else None
    rank1 = sorted_unique[1] if len(sorted_unique) > 1 else None

    result = []
    for mean, std in values:
        if mean is None:
            result.append("--")
            continue
        sv = sign * mean
        if rank0 is not None and abs(sv - rank0) < eps:
            result.append(_tex(mean, std, nd=nd, bold=True))
        elif rank1 is not None and abs(sv - rank1) < eps:
            result.append(_tex(mean, std, nd=nd, underline=True))
        else:
            result.append(_tex(mean, std, nd=nd))
    return result


def _collect_stats(
    summaries: list[dict[str, Any]],
    *,
    setting_key: str,
    case_name: str,
    field: str,
) -> tuple[float | None, float | None]:
    values: list[float] = []
    for summary in summaries:
        metrics = summary.get(setting_key, {}).get(case_name, {})
        if not isinstance(metrics, dict):
            continue
        value = metrics.get(field)
        if value is None:
            continue
        values.append(float(value))
    return mean_std(values)


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def _by_case_tabular(summaries: list[dict[str, Any]]) -> str:
    present_budgets = [
        b for b in BUDGET_ORDER if any(_setting_key(b) in summary for summary in summaries)
    ]
    n_mb = len(present_budgets)
    budget_header = " & ".join(
        rf"\textbf{{{BUDGET_LABEL[b]}}}" for b in present_budgets
    )

    col_spec = "l|" + "|".join("r" * n_mb for _ in range(3))
    lines: list[str] = []
    lines.append(rf"\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}" "\n")
    lines.append(r"\toprule" "\n")

    # header row 1
    lines.append(
        r"\multirow{2}{*}{\textbf{Case}} & "
        rf"\multicolumn{{{n_mb}}}{{c}}{{\textbf{{TCSR (\%)}} ($\uparrow$)}} & "
        rf"\multicolumn{{{n_mb}}}{{c}}{{\textbf{{Gap$^{{+}}$ (s)}} ($\downarrow$)}} & "
        rf"\multicolumn{{{n_mb}}}{{c}}{{\textbf{{CT (s)}} ($\downarrow$)}} \\" "\n"
    )

    def _cmidrule(s: int, e: int) -> str:
        return rf"\cmidrule(lr){{{s}-{e}}}"

    c1, c2 = 2, 1 + n_mb
    c3, c4 = c2 + 1, c2 + n_mb
    c5, c6 = c4 + 1, c4 + n_mb
    lines.append(f"{_cmidrule(c1,c2)}{_cmidrule(c3,c4)}{_cmidrule(c5,c6)}\n")

    # header row 2: budget labels per metric group
    lines.append(f"& {budget_header} & {budget_header} & {budget_header} \\\\\n")
    lines.append(r"\midrule" "\n")

    for case_name, case_label in CASES:
        tsr_vals: list[tuple[float | None, float | None]] = []
        gap_vals: list[tuple[float | None, float | None]] = []
        ct_vals: list[tuple[float | None, float | None]] = []

        for budget in present_budgets:
            key = _setting_key(budget)
            tsr_vals.append(
                _collect_stats(
                    summaries,
                    setting_key=key,
                    case_name=case_name,
                    field="tsr",
                )
            )
            gap_vals.append(
                _collect_stats(
                    summaries,
                    setting_key=key,
                    case_name=case_name,
                    field="makespan_gap_sr_1",
                )
            )
            ct_vals.append(
                _collect_stats(
                    summaries,
                    setting_key=key,
                    case_name=case_name,
                    field="computation_time",
                )
            )

        tsr_cells = _rank_format(tsr_vals, nd=1, higher_is_better=True)
        gap_cells = _rank_format(gap_vals, nd=1, higher_is_better=False)
        ct_cells = _rank_format(ct_vals, nd=3, higher_is_better=False)

        lines.append(
            f"\\textbf{{{case_label}}} & "
            f"{' & '.join(tsr_cells)} & "
            f"{' & '.join(gap_cells)} & "
            f"{' & '.join(ct_cells)} \\\\\n"
        )

    lines.append(r"\bottomrule" "\n")
    lines.append(r"\end{tabular}" "\n")
    return "".join(lines)


def build_by_case_tex(summaries: list[dict[str, Any]]) -> str:
    tabular = _by_case_tabular(summaries)
    return (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        r"{\small" "\n"
        r"\setlength{\tabcolsep}{4.5pt}" "\n"
        r"\renewcommand{\arraystretch}{1.05}" "\n"
        r"\resizebox{\columnwidth}{!}{%" "\n"
        + tabular
        + "}\n"
        "}\n"
        r"\caption{Effect of monitoring budget (max monitors per critical interval) "
        r"under CORRECT\_ESTIMATE prior (Gaussian GT, DEFAULT planner, $W{=}D{=}10$, $\eta{=}0.1$). "
        r"Entries report mean $\pm$ standard deviation over five decomposed instruction sets "
        r"(v1--v5), computed over translation-valid instructions only. Gap$^{+}$ averages "
        r"$\max(0,\text{makespan}-\text{oracle})$ over successful instructions. Bold: best; "
        r"underline: second-best per row.}"
        "\n"
        r"\label{tab:monitoring_budget_by_case}"
        "\n"
        r"\end{table}"
        "\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_SUMMARY,
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=None,
        help="Root directory containing per-task-folder analysis outputs.",
    )
    parser.add_argument(
        "--task-folders",
        nargs="+",
        default=None,
        help="Task folder names to aggregate under --analysis-root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_OUT_DIR,
    )
    args = parser.parse_args()

    summary_paths = resolve_source_paths(
        single_path=args.summary,
        analysis_root=args.analysis_root,
        task_folders=args.task_folders,
        filename="offline_analysis_summary.json",
    )
    summaries = load_json_objects(summary_paths)

    for budget in BUDGET_ORDER:
        key = _setting_key(budget)
        if not any(key in summary for summary in summaries):
            print(f"[warn] missing in summary: {key}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "monitoring_budget_by_case.tex"
    out_path.write_text(build_by_case_tex(summaries), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
