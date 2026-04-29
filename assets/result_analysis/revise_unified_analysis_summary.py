"""
Process experiment summary data and generate a LaTeX table.

This script performs two main functions:
1.  Revises a raw JSON summary file by aggregating results based on task length.
    It calculates the average metrics (SR, TSR, makespan) across different
    constraints for tasks of the same length.
2.  Generates a LaTeX table from the revised summary, comparing different
    scheduling approaches and highlighting the best-performing ones.

The script is intended to be run from the command line, taking the path to the
input summary file as an argument.
"""

import argparse
import json
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Configuration Constants ---

# Metrics to include in the LaTeX table
# Options: "sr", "gcr", "tsr", "makespan"
INCLUDED_METRICS: List[str] = [
    "sr",
    "gcr",
    "tsr",
    "makespan",
]

# Order of initial time estimates (all possible values)
INIT_ORDER: List[str] = [
    "init_60",
    "init_70",
    "init_90",
    "init_80",
    "init_100",
    "init_110",
    "init_120",
    "init_130",
    "init_140",
]

# Display names for different methods/approaches
METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "dag_bayesian_DEFAULT": "Ours",
    "dag_bayesian_GREEDY": "Ours (Greedy)",
    "dag_bayesian_NONE_MONITORING": "Ours (w/o Mon.)",
    "dag_bayesian_NONE_URGENCY": "Ours (w/o Urg.)",
    "dag_bayesian_NONE_REMAINING_WORK": "Ours (w/o Rem.)",
    "dag_edf": "EDF",
    "cpm": "CPM",
    "progprompt": "Prog.",
    "cap_ai2thor_simulation": "CaP",
}

# Methods to be included in the main comparison table and their order
MAIN_METHODS: List[str] = [
    "dag_edf",
    "cpm",
    "progprompt",
    "cap_ai2thor_simulation",
    "dag_bayesian_DEFAULT",
]

# Key for the ablation study method
ABLATION_METHOD_KEY: str = "dag_bayesian_NONE_MONITORING"

# Order and labels for task difficulties (based on number of subtasks)
TASK_ORDER: List[str] = [
    "tasks_2_constraints_1",
    "tasks_2_constraints_2",
    "tasks_3_constraints_1",
    "tasks_3_constraints_2",
    # "tasks_4_constraints_1",
    # "tasks_4_constraints_2",
]
TASK_DISPLAY_NAMES: Dict[str, str] = {
    "tasks_2_constraints_1": "T2 C1",
    "tasks_2_constraints_2": "T2 C2",
    "tasks_3_constraints_1": "T3 C1",
    "tasks_3_constraints_2": "T3 C2",
    # "tasks_4_constraints_1": "Tasks 4 (C=1)",
    # "tasks_4_constraints_2": "Tasks 4 (C=2)",
}

# Approaches to exclude from the final analysis
APPROACHES_TO_EXCLUDE: List[str] = [
    "dag_bayesian_GREEDY",
    "dag_bayesian_NONE_URGENCY",
    "dag_bayesian_NONE_REMAINING_WORK",
]


# --- Argument Parsing ---


def load_argument_parser() -> argparse.Namespace:
    """Set up and parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process summary data and generate a LaTeX table."
    )
    parser.add_argument(
        "--summary_file_path",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results"
        / "unified_analysis_summary.json",
        help="Path to the input JSON summary file.",
    )
    parser.add_argument(
        "--save_revised_summary",
        default=True,
        action="store_true",
        help="If set, save the intermediate revised summary to a '.revised.json' file.",
    )
    parser.add_argument(
        "--ablation_style",
        type=str,
        default="parentheses",
        choices=["parentheses", "new_row"],
        help="Choose how to display ablation study results: 'parentheses' (inline) or 'new_row' (separate row).",
    )
    return parser.parse_args()


# --- Data Processing Functions ---


def reorder_metrics_dict(data: Dict[str, float]) -> "OrderedDict[str, float]":
    """Reorder metric keys in a dictionary for consistent output.

    Args:
        data (Dict[str, float]): A dictionary of metrics.

    Returns:
        OrderedDict[str, float]: The dictionary with sorted keys.
    """
    # Define standard metric order (superset of potentially included metrics)
    standard_order = ["sr", "gcr", "tsr", "makespan", "makespan_sr_1"]

    # Filter based on INCLUDED_METRICS and what's available in data
    # Using standard_order ensures consistent ordering regardless of INCLUDED_METRICS order
    keys_to_keep = [k for k in standard_order if k in data]

    return OrderedDict([(k, data[k]) for k in keys_to_keep])


def merge_by_task_length(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate summary metrics by task and constraint type.

    This function processes a summary dictionary and groups results by
    task length and constraint count (e.g., "tasks_2_constraints_1").

    Note: TSR (Timing Success Rate) is only calculated for constraints >= 1,
    excluding constraints_0 cases.

    Args:
        summary (Dict[str, Any]): The raw summary data structured as
            summary[approach][task_case]["init"][init_case] = metrics.

    Returns:
        Dict[str, Any]: A new summary dictionary with metrics aggregated
            by task/constraint, structured as
            merged[approach][task_constraint_key][init_case] = averaged_metrics.
    """
    sums: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    )
    counts: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    )

    for approach, task_cases in summary.items():
        for task_case, task_data in task_cases.items():
            match = re.match(r"tasks_(\d+)_constraints_(\d+)", task_case)
            if not match:
                continue

            tasks_num = int(match.group(1))
            constraints_num = int(match.group(2))

            if constraints_num == 0:
                continue

            tasks_key = f"tasks_{tasks_num}_constraints_{constraints_num}"
            init_dict = task_data.get("init", {})

            if not isinstance(init_dict, dict):
                continue

            # TSR은 constraint >= 1인 경우에만 계산
            include_tsr = constraints_num >= 1

            for init_case, metrics in init_dict.items():
                for metric_name, value in metrics.items():
                    # TSR의 경우 constraint >= 1일 때만 합산
                    if metric_name == "tsr" and not include_tsr:
                        continue

                    sums[approach][tasks_key][init_case][metric_name] += value
                    counts[approach][tasks_key][init_case][metric_name] += 1

    merged_summary: Dict[str, Any] = defaultdict(dict)
    for approach, tasks_dict in sums.items():
        for tasks_key, init_dict in tasks_dict.items():
            merged_summary[approach][tasks_key] = {}
            for init_case, metric_sums in init_dict.items():
                avg_metrics: Dict[str, float] = {}

                for metric_name, total in metric_sums.items():
                    count = counts[approach][tasks_key][init_case][metric_name]
                    if count > 0:
                        avg_metrics[metric_name] = total / count

                # TSR이 metric_sums에 없는 경우 (constraint >= 1인 데이터가 없음)
                # 다른 메트릭들은 있지만 TSR만 없는 경우를 처리
                if "tsr" not in avg_metrics and metric_sums:
                    # 다른 메트릭들이 있다면 TSR을 0으로 설정
                    avg_metrics["tsr"] = 0.0

                if not avg_metrics:
                    continue

                # Reorder and store
                merged_summary[approach][tasks_key][init_case] = reorder_metrics_dict(
                    avg_metrics
                )

    return dict(merged_summary)


# --- LaTeX Generation Functions ---


def find_best_values(data: Dict[str, Any]) -> Dict[Tuple[str, str, str], float]:
    """Identify the best metric values among main methods AND ablation method.

    Compares the methods listed in `MAIN_METHODS` plus `ABLATION_METHOD_KEY`
    to find the best (max for SR/TSR, min for makespan) value for each metric
    under each task and initial condition. This is used for bolding in the table.

    Args:
        data (Dict[str, Any]): The revised summary data.

    Returns:
        Dict[Tuple[str, str, str], float]: A dictionary mapping
            (task, init, metric) to the best value.
    """
    best_vals: Dict[Tuple[str, str, str], float] = {}

    # Include ablation method in comparison for best value
    methods_to_compare = MAIN_METHODS + [ABLATION_METHOD_KEY]

    for task in TASK_ORDER:
        for init in INIT_ORDER:
            # Collect values for comparison
            sr_vals = [
                data[method][task][init].get("sr", -1.0)
                for method in methods_to_compare
                if data.get(method, {}).get(task, {}).get(init)
            ]
            gcr_vals = [
                data[method][task][init].get("gcr", -1.0)
                for method in methods_to_compare
                if data.get(method, {}).get(task, {}).get(init)
            ]
            tsr_vals = [
                data[method][task][init].get("tsr", -1.0)
                for method in methods_to_compare
                if data.get(method, {}).get(task, {}).get(init)
            ]
            mk_vals = [
                data[method][task][init].get("makespan_sr_1", float("inf"))
                for method in methods_to_compare
                if data.get(method, {}).get(task, {}).get(init)
            ]

            # Determine best values
            if sr_vals:
                best_vals[(task, init, "sr")] = max(sr_vals)
            if gcr_vals:
                best_vals[(task, init, "gcr")] = max(gcr_vals)
            if tsr_vals:
                best_vals[(task, init, "tsr")] = max(tsr_vals)
            if mk_vals:
                best_vals[(task, init, "makespan_sr_1")] = min(mk_vals)

    return best_vals


def fmt_cell(
    main_val: Optional[float],
    metric: str,
    best_val: float,
    ablation_val: Optional[float] = None,
    show_ablation_in_parens: bool = True,
) -> str:
    """Format a single table cell with main value, bolding, and ablation study value.

    Args:
        main_val (Optional[float]): The primary metric value for the cell.
        metric (str): The name of the metric (e.g., 'sr', 'makespan').
        best_val (float): The best value for this metric across methods.
        ablation_val (Optional[float], optional): The corresponding value from
            the ablation study. Defaults to None.
        show_ablation_in_parens (bool): Whether to show ablation value in parentheses.

    Returns:
        str: The formatted string for the LaTeX table cell.
    """
    if main_val is None:
        return "---"

    # Format the main value to one decimal place
    s_main = f"{main_val:.1f}"

    # Apply bolding if the main value is the best (within a small tolerance)
    if abs(main_val - best_val) < 0.001:
        s_main = f"\\textbf{{{s_main}}}"

    # Append ablation value in parentheses if requested
    if show_ablation_in_parens and ablation_val is not None:
        s_abl = f"{ablation_val:.1f}"

        # Bold ablation value if it is the best
        if abs(ablation_val - best_val) < 0.001:
            s_abl = f"\\textbf{{{s_abl}}}"

        return f"{s_main} ({s_abl})"

    return s_main


def generate_latex_table(
    data: Dict[str, Any],
    init_keys: List[str],
    column_labels: List[str],
    ablation_style: str = "parentheses",
) -> str:
    """Generate the complete LaTeX code for the results table.

    Args:
        data (Dict[str, Any]): The revised and filtered summary data.
        init_keys (List[str]): List of init keys to include in this table.
        column_labels (List[str]): List of labels for the init columns.
        ablation_style (str): Style for ablation study display ('parentheses' or 'new_row').

    Returns:
        str: A string containing the full LaTeX table code.
    """
    if len(init_keys) != len(column_labels):
        raise ValueError("Length of init_keys and column_labels must match.")

    best_values = find_best_values(data)
    lines: List[str] = []

    # Determine method list based on style
    methods_to_plot = list(MAIN_METHODS)
    if ablation_style == "new_row":
        # Insert ABLATION_METHOD_KEY after dag_bayesian_DEFAULT
        try:
            idx = methods_to_plot.index("dag_bayesian_DEFAULT")
            methods_to_plot.insert(idx + 1, ABLATION_METHOD_KEY)
        except ValueError:
            pass  # dag_bayesian_DEFAULT not in list

    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Performance comparison in the AI2-THOR simulation environment. "
            r"Results are aggregated by task complexity. "
            r"Values in parentheses indicate the performance of the ablation study \quotes{Ours (w/o Mon.)}. "
            r"Higher SR, GCR and TSR are better ($\uparrow$), and lower Makespan is better ($\downarrow$).}",
            r"\label{tab:simulation_results_merged}",
            r"{\renewcommand{\arraystretch}{1.1}",
            r"\resizebox{\columnwidth}{!}{%",
            # Dynamic column definition based on num_metrics
            # ll | (num_metrics)c | ... * number of init blocks
            f"\\begin{{tabular}}{{@{{}}ll|{'|'.join(['c' * len(INCLUDED_METRICS)] * len(init_keys))}@{{}}}}",
            r"\toprule",
        ]
    )

    # Column Headers

    # Dynamically build the metric header part
    metric_headers = []
    for m in INCLUDED_METRICS:
        if m == "sr":
            metric_headers.append(r"\textbf{SR ($\uparrow$)}")
        elif m == "gcr":
            metric_headers.append(r"\textbf{GCR ($\uparrow$)}")
        elif m == "tsr":
            metric_headers.append(r"\textbf{TSR ($\uparrow$)}")
        elif m == "makespan":
            metric_headers.append(r"\textbf{MS ($\downarrow$)}")

    metric_header_str = " & ".join(metric_headers)
    num_metrics = len(INCLUDED_METRICS)

    # Build the second row (Init Labels)
    header_row_2_parts = [r"\multicolumn{2}{c|}{\multirow{2}{*}{\textbf{Method}}}"]
    header_row_3_parts = []  # Will hold cmidrules

    # Calculate column spans
    for i, label in enumerate(column_labels):
        header_row_2_parts.append(rf"\multicolumn{{{num_metrics}}}{{c|}}{{{label}}}")

        # cmidrule calculation
        # First 2 columns are Method/Difficulty. Then blocks of num_metrics.
        # Start col for block i (0-indexed i): 3 + i * num_metrics
        # End col: 3 + i * num_metrics + num_metrics - 1
        start_col = 3 + i * num_metrics
        end_col = start_col + num_metrics - 1

        header_row_3_parts.append(rf"\cmidrule(l){{{start_col}-{end_col}}}")

    # Build the third row (SR, GCR, TSR, Makespan repeated)
    header_row_4_parts = [r"\multicolumn{2}{c|}{\textbf{Difficulty}}"]
    for _ in range(len(column_labels)):
        header_row_4_parts.append(metric_header_str)

    lines.extend(
        [
            " & ".join(header_row_2_parts) + r" \\",
            " ".join(header_row_3_parts),
            " & ".join(header_row_4_parts) + r" \\",
            r"\midrule",
        ]
    )

    # Table Body
    for method_idx, method_key in enumerate(methods_to_plot):
        method_label = METHOD_DISPLAY_NAMES.get(method_key, method_key)

        for task_idx, task_key in enumerate(TASK_ORDER):
            row_parts = []

            # Method Name Column (with multirow)
            if task_idx == 0:
                row_parts.append(f"\\multirow{{3}}{{*}}{{\\textbf{{{method_label}}}}}")
            else:
                # For subsequent rows in a multirow block, leave the first column empty.
                row_parts.append("")

            # Task Difficulty Column
            row_parts.append(
                f"\\textbf{{{TASK_DISPLAY_NAMES.get(task_key, task_key)}}}"
            )

            # Metric Columns for each requested init key
            for init_key in init_keys:
                metrics = data.get(method_key, {}).get(task_key, {}).get(init_key, {})

                # Get ablation data if applicable AND style is parentheses
                abl_metrics = None
                if (
                    ablation_style == "parentheses"
                    and method_key == "dag_bayesian_DEFAULT"
                ):
                    abl_metrics = (
                        data.get(ABLATION_METHOD_KEY, {})
                        .get(task_key, {})
                        .get(init_key)
                    )

                for metric_key in INCLUDED_METRICS:
                    # Table display logic: Use makespan_sr_1 for makespan column
                    val_key = (
                        "makespan_sr_1" if metric_key == "makespan" else metric_key
                    )
                    val = metrics.get(val_key)

                    abl_val = None
                    if abl_metrics:
                        abl_val = abl_metrics.get(val_key)

                    # For best value comparison, we also need to look up the correct key
                    # create a temporary key for lookup
                    lookup_metric_key = val_key

                    best_val = best_values.get(
                        (task_key, init_key, lookup_metric_key),
                        float("inf") if metric_key == "makespan" else float("-inf"),
                    )

                    row_parts.append(
                        fmt_cell(
                            val,
                            metric_key,
                            best_val,
                            abl_val,
                            show_ablation_in_parens=(ablation_style == "parentheses"),
                        )
                    )

            lines.append(" & ".join(row_parts) + r" \\")

        if method_idx < len(methods_to_plot) - 1:
            lines.append(r"\midrule")

    # Footer
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


# --- Main Execution ---


def main() -> None:
    """Run the main script execution pipeline."""
    args = load_argument_parser()

    # 1. Load data from the source file
    try:
        with open(args.summary_file_path, "r") as f:
            summary_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.summary_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.summary_file_path}")
        return

    # 2. Process and revise the summary
    revised_summary = merge_by_task_length(summary_data)

    # 3. Filter out excluded approaches
    for approach in APPROACHES_TO_EXCLUDE:
        revised_summary.pop(approach, None)

    # 4. Save the revised summary if requested
    if args.save_revised_summary:
        output_path = args.summary_file_path.with_suffix(".revised.json")
        with open(output_path, "w") as f:
            json.dump(revised_summary, f, indent=4)
        print(f"Revised summary saved to: {output_path}")

    # 5. Generate and print the LaTeX tables

    # Table 1: Under-estimate (100, 80, 60)
    init_keys_1 = ["init_100"]
    labels_1 = [r"\textbf{Correct (100s)}"]

    print(
        generate_latex_table(
            revised_summary, init_keys_1, labels_1, args.ablation_style
        )
    )

    # Table 2: Over-estimate (100, 120, 140)
    # init_keys_2 = ["init_100"]
    # labels_2 = [r"\textbf{Correct (100s)}"]

    # print("\n--- LaTeX Table 2 (Over-estimate) ---\n")
    # print(
    #     generate_latex_table(
    #         revised_summary, init_keys_2, labels_2, args.ablation_style
    #     )
    # )


if __name__ == "__main__":
    main()
