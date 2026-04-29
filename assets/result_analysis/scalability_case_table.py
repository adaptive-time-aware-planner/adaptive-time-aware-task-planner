"""Render a beam-wise case table for the scalability summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from assets.result_analysis.utils.multi_source import (
    load_json_objects,
    mean_std,
    resolve_source_paths,
)

DEFAULT_SUMMARY = (
    "assets/results/offline_exp_result/offline_batch_scalability/"
    "analysis/scalability/offline_analysis_summary.json"
)
DEFAULT_OUT = (
    "assets/results/offline_exp_result/offline_batch_scalability/"
    "analysis/scalability/latex_tables/scalability_case_table.tex"
)

BEAMS: list[tuple[str, str]] = [
    ("w1_d1", "B=1"),
    ("w10_d10", "B=10"),
    ("w20_d20", "B=20"),
]

METHOD_SPECS: list[tuple[str, str, dict[str, str]]] = [
    (
        "edf",
        r"EDF",
        {
            "w1_d1": "CORRECT_ESTIMATE__edf",
            "w10_d10": "CORRECT_ESTIMATE__edf",
            "w20_d20": "CORRECT_ESTIMATE__edf",
        },
    ),
    (
        "ours",
        r"Ours",
        {
            "w1_d1": "CORRECT_ESTIMATE__bayesian__DEFAULT__w1_d1__eta0.1",
            "w5_d5": "CORRECT_ESTIMATE__bayesian__DEFAULT__w5_d5__eta0.1",
            "w10_d10": "CORRECT_ESTIMATE__bayesian__DEFAULT__w10_d10__eta0.1",
            "w20_d20": "CORRECT_ESTIMATE__bayesian__DEFAULT__w20_d20__eta0.1",
        },
    ),
    (
        "nm",
        r"Ours wo Mon.",
        {
            "w1_d1": "CORRECT_ESTIMATE__bayesian__NONE_MONITORING__w1_d1",
            "w5_d5": "CORRECT_ESTIMATE__bayesian__NONE_MONITORING__w5_d5",
            "w10_d10": "CORRECT_ESTIMATE__bayesian__NONE_MONITORING__w10_d10",
            "w20_d20": "CORRECT_ESTIMATE__bayesian__NONE_MONITORING__w20_d20",
        },
    ),
]

CASES: list[tuple[str, str]] = [
    ("tasks_2_constraints_1", "T2C1"),
    ("tasks_2_constraints_2", "T2C2"),
    ("tasks_3_constraints_1", "T3C1"),
    ("tasks_3_constraints_2", "T3C2"),
]


def _fmt_value(
    mean: float | None,
    std: float | None,
    *,
    digits: int,
    style: str = "",
    scale: float = 1.0,
) -> str:
    if mean is None:
        return "--"
    mean *= scale
    return f"${mean:.{digits}f}$"


def _compute_highlight_styles(
    values: list[float],
    *,
    higher_is_better: bool,
    tolerance: float = 1e-9,
) -> list[str]:
    """Return cell styles (`best`, `second`, or ``) for one metric column."""

    if not values:
        return []

    unique_sorted = sorted(set(values), reverse=higher_is_better)
    best_value = unique_sorted[0]
    second_value = unique_sorted[1] if len(unique_sorted) > 1 else None
    styles: list[str] = []
    for value in values:
        if abs(value - best_value) <= tolerance:
            styles.append("best")
        elif second_value is not None and abs(value - second_value) <= tolerance:
            styles.append("second")
        else:
            styles.append("")
    return styles


def _collect_stats(
    summaries: list[Mapping[str, Any]],
    *,
    summary_key: str,
    case_name: str,
    field: str,
) -> tuple[float | None, float | None]:
    values: list[float] = []
    for summary in summaries:
        metrics = summary.get(summary_key, {}).get(case_name)
        if metrics is None:
            continue
        value = metrics.get(field)
        if value is None:
            continue
        values.append(float(value))
    return mean_std(values)


def _build_highlight_map(
    summaries: list[Mapping[str, Any]],
    present_methods: list[tuple[str, str, dict[str, str]]],
) -> dict[tuple[str, str, str], dict[str, str]]:
    """Build per-case, per-beam metric highlight styles across methods."""

    metric_specs = [
        ("tsr", True),
        ("makespan_gap", False),
        ("computation_time", False),
    ]
    highlight_map: dict[tuple[str, str, str], dict[str, str]] = {}

    for case, _short in CASES:
        for beam_key, _beam_label in BEAMS:
            for metric_name, higher_is_better in metric_specs:
                summary_keys: list[str] = []
                values: list[float] = []
                for _method_id, _label, key_map in present_methods:
                    summary_key = key_map[beam_key]
                    mean, _std = _collect_stats(
                        summaries,
                        summary_key=summary_key,
                        case_name=case,
                        field=metric_name,
                    )
                    if mean is None:
                        continue
                    summary_keys.append(summary_key)
                    values.append(mean)

                styles = _compute_highlight_styles(
                    values,
                    higher_is_better=higher_is_better,
                )
                highlight_map[(case, beam_key, metric_name)] = dict(
                    zip(summary_keys, styles)
                )

    return highlight_map


def _build_tabular(summaries: list[Mapping[str, Any]]) -> str:
    head = (
        r"\begin{tabular}{@{}llrrrrrrrrr@{}}"
        "\n\\toprule\n"
        r"\multirow{2}{*}{\textbf{Method}} & \multirow{2}{*}{\textbf{Task}} & "
        r"\multicolumn{3}{c}{\textbf{TCSR} ($\uparrow$)} & "
        r"\multicolumn{3}{c}{\textbf{Gap (s)} ($\downarrow$)} & "
        r"\multicolumn{3}{c}{\textbf{CT (s)} ($\downarrow$)} \\"
        "\n"
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}"
        "\n"
        r"& & \textbf{B=1} & \textbf{B=10} & \textbf{B=20} "
        r"& \textbf{B=1} & \textbf{B=10} & \textbf{B=20} "
        r"& \textbf{B=1} & \textbf{B=10} & \textbf{B=20} \\"
        "\n\\midrule\n"
    )
    present_methods = [
        (method_id, label, key_map)
        for method_id, label, key_map in METHOD_SPECS
        if any(any(key in summary for summary in summaries) for key in key_map.values())
    ]
    if not present_methods:
        raise KeyError("No supported EDF/Ours/Ours wo Mon. keys found in summary.")

    highlight_map = _build_highlight_map(summaries, present_methods)
    chunks: list[str] = []
    for method_index, (_method_id, label, key_map) in enumerate(present_methods):
        for case_index, (case, short) in enumerate(CASES):
            prefix = (
                rf"\multirow{{4}}{{*}}{{\textbf{{{label}}}}}" if case_index == 0 else ""
            )
            tsr_cells: list[str] = []
            gap_cells: list[str] = []
            ct_cells: list[str] = []
            for beam_key, _beam_label in BEAMS:
                summary_key = key_map[beam_key]
                tsr_mean, tsr_std = _collect_stats(
                    summaries,
                    summary_key=summary_key,
                    case_name=case,
                    field="tsr",
                )
                gap_mean, gap_std = _collect_stats(
                    summaries,
                    summary_key=summary_key,
                    case_name=case,
                    field="makespan_gap",
                )
                ct_mean, ct_std = _collect_stats(
                    summaries,
                    summary_key=summary_key,
                    case_name=case,
                    field="computation_time",
                )
                if tsr_mean is None and gap_mean is None and ct_mean is None:
                    tsr_cells.append("--")
                    gap_cells.append("--")
                    ct_cells.append("--")
                    continue
                tsr_cells.append(
                    _fmt_value(
                        tsr_mean,
                        tsr_std,
                        digits=2,
                        scale=0.01,
                        style=highlight_map[(case, beam_key, "tsr")].get(
                            summary_key, ""
                        ),
                    )
                )
                gap_cells.append(
                    _fmt_value(
                        gap_mean,
                        gap_std,
                        digits=2,
                        style=highlight_map[(case, beam_key, "makespan_gap")].get(
                            summary_key, ""
                        ),
                    )
                )
                ct_cells.append(
                    _fmt_value(
                        ct_mean,
                        ct_std,
                        digits=2,
                        style=highlight_map[(case, beam_key, "computation_time")].get(
                            summary_key, ""
                        ),
                    )
                )
            chunks.append(
                f"{prefix} & \\textbf{{{short}}} & "
                + " & ".join(tsr_cells + gap_cells + ct_cells)
                + " \\\\\n"
            )

        if method_index == len(present_methods) - 1:
            continue
        chunks.append("\\midrule\n")

    if chunks and chunks[-1] in {"\\midrule\n", "\\addlinespace[2pt]\n"}:
        chunks.pop()
    return head + "".join(chunks) + "\\bottomrule\n\\end{tabular}\n"


def build_tex(summaries: list[Mapping[str, Any]]) -> str:
    tabular = _build_tabular(summaries)
    return (
        r"\begin{table}[t]"
        "\n"
        r"\centering"
        "\n"
        r"{\small"
        "\n"
        r"\setlength{\tabcolsep}{3.5pt}"
        "\n"
        r"\renewcommand{\arraystretch}{1.05}"
        "\n" + tabular + "}\n"
        r"\caption{Scalability results by beam size under the CORRECT\_ESTIMATE "
        r"setting with $\eta=0.1$. Entries report mean values over five "
        r"decomposed instruction sets. Columns sweep beam size "
        r"$B \in \{1,10,20\}$.}"
        "\n"
        r"\label{tab:scalability_case_results}"
        "\n"
        r"\end{table}"
        "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_SUMMARY,
        help="Path to offline_analysis_summary.json.",
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
        "-o",
        "--output",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_OUT,
        help="Output .tex path.",
    )
    args = parser.parse_args()

    summary_paths = resolve_source_paths(
        single_path=args.summary,
        analysis_root=args.analysis_root,
        task_folders=args.task_folders,
        filename="offline_analysis_summary.json",
    )
    summaries = load_json_objects(summary_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_tex(summaries), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
