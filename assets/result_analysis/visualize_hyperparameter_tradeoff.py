import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASK_CASES = [
    "tasks_2_constraints_1",
    "tasks_2_constraints_2",
    "tasks_3_constraints_1",
    "tasks_3_constraints_2",
]

PRIOR_LABELS = {
    "Under": "Under",
    "Correct": "Correct",
    "Over": "Over",
    "UNDER_ESTIMATE": "Under",
    "CORRECT_ESTIMATE": "Correct",
    "OVER_ESTIMATE": "Over",
}

PRIOR_STYLE = {
    "Under": {"color": "tab:blue", "marker": "o"},
    "Correct": {"color": "tab:green", "marker": "s"},
    "Over": {"color": "tab:orange", "marker": "^"},
}

ETA_STYLE = {
    0.01: {"color": "tab:purple", "marker": "o", "label": r"$\eta=0.01$"},
    0.1: {"color": "crimson", "marker": "*", "label": r"$\eta=0.1$"},
    0.9: {"color": "tab:gray", "marker": "X", "label": r"$\eta=0.9$"},
}


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def parse_pm_value(cell: str) -> float:
    """Parse LaTeX cells like `$1.00 \\pm 0.00$` or `--`."""
    if "--" in cell:
        return float("nan")
    match = re.search(r"[-+]?\d*\.?\d+", cell)
    if not match:
        return float("nan")
    return float(match.group(0))


def parse_eta_overall_table(path: Path) -> list[dict]:
    """Read eta/TCSR/gap/monitoring-count values from the generated LaTeX table."""
    rows: list[dict] = []
    if not path.exists():
        return rows

    current_prior: str | None = None
    row_pattern = re.compile(
        r"(?:\\multirow\{3\}\{\*\}\{\\textbf\{(?P<prior>[^}]+)\}\}\s*&\s*)?"
        r"\$(?P<eta>[^$]+)\$\s*&\s*"
        r"(?P<tcsr>[^&]+)\s*&\s*"
        r"(?P<gap>[^&]+)\s*&\s*"
        r"(?P<monitor>.+?)\\\\"
    )

    for line in path.read_text().splitlines():
        match = row_pattern.search(line)
        if not match:
            continue
        if match.group("prior"):
            current_prior = PRIOR_LABELS[match.group("prior")]
        if current_prior is None:
            continue

        rows.append(
            {
                "prior": current_prior,
                "eta": float(match.group("eta")),
                "tcsr": parse_pm_value(match.group("tcsr")),
                "gap": parse_pm_value(match.group("gap")),
                "monitor_count": parse_pm_value(match.group("monitor")),
            }
        )
    return rows


def aggregate_eta_from_json(eta_dir: Path) -> list[dict]:
    """Fallback if the LaTeX table with monitoring counts is unavailable."""
    grouped: dict[tuple[str, float], list[float]] = {}
    key_re = re.compile(
        r"(UNDER_ESTIMATE|CORRECT_ESTIMATE|OVER_ESTIMATE)__bayesian__DEFAULT__w10_d10__eta(.+)"
    )

    for summary_path in sorted(eta_dir.glob("*/offline_analysis_summary.json")):
        data = json.loads(summary_path.read_text())
        for key, case_results in data.items():
            match = key_re.fullmatch(key)
            if not match:
                continue
            prior_raw, eta_raw = match.groups()
            prior = PRIOR_LABELS[prior_raw]
            eta = float(eta_raw)
            tcsr_values = [
                case_results[case]["tsr"] / 100.0
                for case in TASK_CASES
                if case in case_results
            ]
            grouped.setdefault((prior, eta), []).append(mean(tcsr_values))

    rows = []
    for (prior, eta), values in sorted(grouped.items()):
        rows.append(
            {
                "prior": prior,
                "eta": eta,
                "tcsr": mean(values),
                "gap": float("nan"),
                "monitor_count": eta,
            }
        )
    return rows


def aggregate_scalability(scalability_dir: Path) -> list[dict]:
    grouped: dict[int, dict[str, list[float]]] = {}
    key_re = re.compile(r"CORRECT_ESTIMATE__bayesian__DEFAULT__w(\d+)_d\1__eta0\.1")

    for summary_path in sorted(scalability_dir.glob("*/offline_analysis_summary.json")):
        data = json.loads(summary_path.read_text())
        for key, case_results in data.items():
            match = key_re.fullmatch(key)
            if not match:
                continue
            b = int(match.group(1))
            case_tcsr = []
            case_gap = []
            case_ct = []
            case_makespan = []
            for case in TASK_CASES:
                if case not in case_results:
                    continue
                metrics = case_results[case]
                case_tcsr.append(metrics["tsr"] / 100.0)
                case_gap.append(metrics["makespan_gap_sr_1"])
                case_ct.append(metrics["computation_time"])
                case_makespan.append(metrics["makespan_sr_1"])

            grouped.setdefault(b, {"tcsr": [], "gap": [], "ct": [], "makespan": []})
            grouped[b]["tcsr"].append(mean(case_tcsr))
            grouped[b]["gap"].append(mean(case_gap))
            grouped[b]["ct"].append(mean(case_ct))
            grouped[b]["makespan"].append(mean(case_makespan))

    rows = []
    for b in sorted(grouped):
        rows.append(
            {
                "B": b,
                "tcsr": mean(grouped[b]["tcsr"]),
                "gap": mean(grouped[b]["gap"]),
                "ct": mean(grouped[b]["ct"]),
                "makespan": mean(grouped[b]["makespan"]),
            }
        )
    return rows


def collect_eta_mismatch_points(eta_dir: Path) -> list[dict]:
    points = []
    key_re = re.compile(
        r"(UNDER_ESTIMATE|OVER_ESTIMATE)__bayesian__DEFAULT__w10_d10__eta(.+)"
    )

    for summary_path in sorted(eta_dir.glob("*/offline_analysis_summary.json")):
        data = json.loads(summary_path.read_text())
        split = summary_path.parent.name
        for key, case_results in data.items():
            match = key_re.fullmatch(key)
            if not match:
                continue
            prior_raw, eta_raw = match.groups()
            eta = float(eta_raw)
            prior = PRIOR_LABELS[prior_raw]
            for case, metrics in case_results.items():
                points.append(
                    {
                        "eta": eta,
                        "prior": prior,
                        "case": case,
                        "split": split,
                        "makespan": metrics["makespan"],
                        "tcsr": metrics["tsr"] / 100.0,
                    }
                )
    return points


def collect_eta_condition_points(eta_dir: Path) -> list[dict]:
    points = []
    key_re = re.compile(
        r"(UNDER_ESTIMATE|CORRECT_ESTIMATE|OVER_ESTIMATE)__bayesian__DEFAULT__w10_d10__eta(.+)"
    )

    for summary_path in sorted(eta_dir.glob("*/offline_analysis_summary.json")):
        data = json.loads(summary_path.read_text())
        split = summary_path.parent.name
        for key, case_results in data.items():
            match = key_re.fullmatch(key)
            if not match:
                continue
            prior_raw, eta_raw = match.groups()
            eta = float(eta_raw)
            group = "Correct" if prior_raw == "CORRECT_ESTIMATE" else "Incorrect"
            for case, metrics in case_results.items():
                points.append(
                    {
                        "eta": eta,
                        "group": group,
                        "case": case,
                        "split": split,
                        "makespan": metrics["makespan"],
                        "tcsr": metrics["tsr"] / 100.0,
                    }
                )
    return points


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
        }
    )


def summarize_eta_rows(eta_rows: list[dict]) -> list[dict]:
    grouped: dict[float, dict[str, list[float]]] = {}
    for row in eta_rows:
        eta = row["eta"]
        grouped.setdefault(eta, {"tcsr": [], "monitor_count": []})
        if not np.isnan(row["tcsr"]):
            grouped[eta]["tcsr"].append(row["tcsr"])
        if not np.isnan(row["monitor_count"]):
            grouped[eta]["monitor_count"].append(row["monitor_count"])

    summary = []
    for eta in sorted(grouped):
        values = grouped[eta]
        summary.append(
            {
                "eta": eta,
                "worst_tcsr": min(values["tcsr"]),
                "avg_monitor_count": mean(values["monitor_count"]),
            }
        )
    return summary


def summarize_eta_average(eta_rows: list[dict]) -> list[dict]:
    grouped: dict[float, dict[str, list[float] | int]] = {}
    for row in eta_rows:
        eta = row["eta"]
        grouped.setdefault(eta, {"tcsr": [], "gap": [], "failed": 0})
        grouped[eta]["tcsr"].append(row["tcsr"])
        if np.isnan(row["gap"]):
            grouped[eta]["failed"] += 1
        else:
            grouped[eta]["gap"].append(row["gap"])

    summary = []
    for eta in sorted(grouped):
        values = grouped[eta]
        summary.append(
            {
                "eta": eta,
                "mean_tcsr": mean(values["tcsr"]),
                "mean_gap": mean(values["gap"]),
                "failed": int(values["failed"]),
                "total": len(values["tcsr"]),
            }
        )
    return summary


def summarize_eta_correct_incorrect(eta_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[float, str], dict[str, list[float]]] = {}
    for row in eta_rows:
        group = "Correct" if row["prior"] == "Correct" else "Incorrect"
        key = (row["eta"], group)
        grouped.setdefault(key, {"tcsr": [], "gap": []})
        if not np.isnan(row["tcsr"]):
            grouped[key]["tcsr"].append(row["tcsr"])
        if not np.isnan(row["gap"]):
            grouped[key]["gap"].append(row["gap"])

    summary = []
    for (eta, group), values in sorted(grouped.items()):
        summary.append(
            {
                "eta": eta,
                "group": group,
                "tcsr": mean(values["tcsr"]),
                "gap": mean(values["gap"]),
            }
        )
    return summary


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output_base.with_suffix('.png')}")
    print(f"Saved: {output_base.with_suffix('.pdf')}")


def plot_selection_tradeoff(
    eta_rows: list[dict], scalability_rows: list[dict], output_base: Path
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))

    eta_summary = summarize_eta_rows(eta_rows)
    selected_eta = 0.1
    ax = axes[0]
    x = [row["avg_monitor_count"] for row in eta_summary]
    y = [row["worst_tcsr"] for row in eta_summary]
    labels = [row["eta"] for row in eta_summary]
    ax.plot(x, y, color="tab:blue", marker="o", linewidth=2.1, markersize=6)
    for x_i, y_i, eta in zip(x, y, labels):
        is_selected = abs(eta - selected_eta) < 1e-9
        ax.scatter(
            [x_i],
            [y_i],
            s=95 if is_selected else 44,
            marker="*" if is_selected else "o",
            color="crimson" if is_selected else "tab:blue",
            zorder=4,
        )
        if eta == 0.01:
            dx, dy = (-0.10, -0.055)
        elif eta == 0.1:
            dx, dy = (-0.56, 0.018)
        else:
            dx, dy = (0.06, 0.03)
        ax.text(x_i + dx, y_i + dy, rf"$\eta={eta:g}$", fontsize=8)
    selected = next(row for row in eta_summary if abs(row["eta"] - selected_eta) < 1e-9)
    ax.axvline(
        selected["avg_monitor_count"],
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
    )

    ax.set_title(r"(a) Risk tolerance $\eta$")
    ax.set_xlabel("Monitoring count")
    ax.set_ylabel("Worst-case TCSR")
    ax.set_xlim(-0.1, max(x) + 0.18)
    ax.set_ylim(0.54, 1.045)
    ax.set_yticks([0.6, 0.8, 1.0])
    ax.grid(True, linestyle="--", alpha=0.35)

    ax = axes[1]
    rows = sorted(scalability_rows, key=lambda r: r["B"])
    x = [row["ct"] for row in rows]
    y = [row["gap"] for row in rows]
    b_values = [row["B"] for row in rows]
    ax.plot(x, y, color="crimson", marker="o", linewidth=2.2, markersize=6)
    for x_i, y_i, b in zip(x, y, b_values):
        dx, dy = (0.16, 0.45)
        if b == 20:
            dx, dy = (-1.18, -0.95)
        ax.text(x_i + dx, y_i + dy, rf"$B={b}$", fontsize=8)
    if 10 in b_values:
        idx = b_values.index(10)
        ax.axvline(x[idx], color="gray", linestyle="--", linewidth=1.0, alpha=0.8)

    ax.set_title(r"(b) Search size $B=W=D$")
    ax.set_xlabel("Computation time (s)")
    ax.set_ylabel("Temporal gap (s)")
    ax.set_xlim(0, max(x) + 0.8)
    ax.set_ylim(max(0, min(y) - 1.2), max(y) + 1.4)
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout(w_pad=1.6)
    save_figure(fig, output_base)


def plot_scalability_only(scalability_rows: list[dict], output_base: Path) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    rows = sorted(scalability_rows, key=lambda r: r["B"])
    x = [row["ct"] for row in rows]
    y = [row["makespan"] for row in rows]
    b_values = [row["B"] for row in rows]

    ax.plot(
        x,
        y,
        color="crimson",
        marker="o",
        linewidth=3.0,
        markersize=10,
        zorder=3,
    )
    for x_i, y_i, b in zip(x, y, b_values):
        dx, dy = 0.22, 0.65
        ha = "left"
        if b == 20:
            dx, dy = -0.35, -1.0
            ha = "right"
        ax.text(
            x_i + dx,
            y_i + dy,
            f"B = {b}",
            fontsize=18,
            fontweight="normal",
            fontstyle="normal",
            ha=ha,
        )

    if 10 in b_values:
        idx = b_values.index(10)
        ax.axvline(x[idx], color="gray", linestyle="--", linewidth=1.6, alpha=0.8)

    ax.set_xlabel("Computation time (s)", fontsize=22, labelpad=10)
    ax.set_ylabel("Makespan (s)", fontsize=22, labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_xlim(0, max(x) + 1.0)
    ax.set_ylim(max(0, min(y) - 1.4), max(y) + 1.8)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_figure(fig, output_base)


def plot_eta_correct_incorrect(eta_rows: list[dict], output_base: Path) -> None:
    configure_matplotlib()
    rows = summarize_eta_correct_incorrect(eta_rows)
    groups = ["Correct", "Incorrect"]
    x = np.arange(len(groups))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), sharex=True)

    for ax, metric, ylabel, title in [
        (axes[0], "tcsr", "TCSR", "(a) Temporal robustness"),
        (axes[1], "gap", "Temporal gap (s)", "(b) Efficiency cost"),
    ]:
        for eta in [0.01, 0.1, 0.9]:
            style = ETA_STYLE[eta]
            eta_rows_subset = [row for row in rows if abs(row["eta"] - eta) < 1e-9]
            y = []
            for group in groups:
                value = next(
                    row[metric] for row in eta_rows_subset if row["group"] == group
                )
                y.append(value)
            ax.plot(
                x,
                y,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linewidth=2.2 if eta == 0.1 else 1.8,
                markersize=8 if eta == 0.1 else 6,
            )
            if metric == "gap" and any(np.isnan(value) for value in y):
                for x_i, y_i in zip(x, y):
                    if np.isnan(y_i):
                        ax.scatter(
                            [x_i],
                            [1.8],
                            marker="x",
                            s=48,
                            color=style["color"],
                            linewidths=1.4,
                            zorder=4,
                        )
                        ax.text(
                            x_i - 0.18,
                            4.1,
                            "fail",
                            fontsize=8,
                            color=style["color"],
                        )

        ax.set_xticks(x, groups)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[0].set_ylim(0.54, 1.04)
    axes[0].set_yticks([0.6, 0.8, 1.0])
    axes[1].set_ylim(0, 40.5)
    axes[1].set_yticks([0, 10, 20, 30, 40])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        framealpha=0.92,
    )
    fig.tight_layout(w_pad=1.6, rect=[0, 0.14, 1, 1])
    save_figure(fig, output_base)


def plot_eta_mismatch_makespan(points: list[dict], output_base: Path) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(3.45, 2.9))

    for eta in sorted({point["eta"] for point in points}):
        eta_points = [point for point in points if abs(point["eta"] - eta) < 1e-9]
        style = ETA_STYLE.get(
            eta, {"color": None, "marker": "o", "label": rf"$\eta={eta:g}$"}
        )
        xs = np.array([point["makespan"] for point in eta_points])
        ys = np.array([point["tcsr"] for point in eta_points])

        ax.scatter(
            xs,
            ys,
            s=18,
            marker=style["marker"],
            color=style["color"],
            alpha=0.24,
            linewidths=0,
        )
        ax.errorbar(
            mean(xs.tolist()),
            mean(ys.tolist()),
            xerr=float(np.std(xs)),
            yerr=float(np.std(ys)),
            fmt=style["marker"],
            color=style["color"],
            markersize=11 if eta == 0.1 else 8,
            markeredgecolor="black" if eta == 0.1 else style["color"],
            markeredgewidth=0.5,
            capsize=2.5,
            linewidth=1.2,
            label=style["label"],
            zorder=4,
        )

    ax.set_xlabel("Makespan (s)")
    ax.set_ylabel("TCSR")
    ax.set_xlim(110, 260)
    ax.set_ylim(0.50, 1.045)
    ax.set_yticks([0.6, 0.8, 1.0])
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", framealpha=0.92)
    ax.set_title(r"Mismatch-only $\eta$ sensitivity")
    ax.text(114, 0.535, "Under/Over priors only", fontsize=7)
    fig.tight_layout()
    save_figure(fig, output_base)


def plot_eta_dual_axis(points: list[dict], output_base: Path) -> None:
    configure_matplotlib()
    groups = ["Correct", "Incorrect"]
    x_base = np.arange(len(groups), dtype=float)
    eta_offsets = {0.01: -0.035, 0.1: 0.0, 0.9: 0.035}

    summary: dict[tuple[float, str], dict[str, float]] = {}
    for eta in sorted({point["eta"] for point in points}):
        for group in groups:
            selected = [
                point
                for point in points
                if abs(point["eta"] - eta) < 1e-9 and point["group"] == group
            ]
            summary[(eta, group)] = {
                "tcsr": mean([point["tcsr"] for point in selected]),
                "makespan": mean([point["makespan"] for point in selected]),
            }

    fig, ax_tcsr = plt.subplots(figsize=(3.45, 3.0))
    ax_ms = ax_tcsr.twinx()

    for eta in [0.01, 0.1, 0.9]:
        style = ETA_STYLE[eta]
        x = x_base + eta_offsets[eta]
        tcsr = [summary[(eta, group)]["tcsr"] for group in groups]
        makespan = [summary[(eta, group)]["makespan"] for group in groups]

        ax_tcsr.plot(
            x,
            tcsr,
            linestyle="--",
            color=style["color"],
            marker=style["marker"],
            markersize=7 if eta == 0.1 else 5.5,
            linewidth=1.8,
            alpha=0.95,
        )
        ax_ms.plot(
            x,
            makespan,
            linestyle="-",
            color=style["color"],
            marker=style["marker"],
            markersize=7 if eta == 0.1 else 5.5,
            linewidth=1.8,
            alpha=0.75,
        )

    ax_tcsr.set_xticks(x_base, groups)
    ax_tcsr.set_xlabel("Temporal prior condition")
    ax_tcsr.set_ylabel("TCSR")
    ax_ms.set_ylabel("Makespan (s)")
    ax_tcsr.set_ylim(0.54, 1.04)
    ax_tcsr.set_yticks([0.6, 0.8, 1.0])
    ax_ms.set_ylim(160, 208)
    ax_ms.set_yticks([160, 175, 190, 205])
    ax_tcsr.grid(True, linestyle="--", alpha=0.35)
    ax_tcsr.set_title(r"Sensitivity to $\eta$", pad=6)

    eta_handles = [
        plt.Line2D(
            [0],
            [0],
            color=ETA_STYLE[eta]["color"],
            marker=ETA_STYLE[eta]["marker"],
            linestyle="",
            markersize=7 if eta == 0.1 else 5.5,
            label=ETA_STYLE[eta]["label"],
        )
        for eta in [0.01, 0.1, 0.9]
    ]
    metric_handles = [
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.6, label="TCSR"),
        plt.Line2D(
            [0], [0], color="black", linestyle="-", linewidth=1.6, label="Makespan"
        ),
    ]
    fig.legend(
        handles=eta_handles + metric_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=3,
        framealpha=0.92,
        columnspacing=0.9,
        handletextpad=0.35,
    )

    fig.tight_layout(rect=[0, 0.15, 1, 1])
    save_figure(fig, output_base)


def plot_eta_aggregate(eta_rows: list[dict], output_base: Path) -> None:
    configure_matplotlib()
    rows = summarize_eta_average(eta_rows)

    fig, ax = plt.subplots(figsize=(3.45, 2.85))
    for row in rows:
        eta = row["eta"]
        style = ETA_STYLE.get(
            eta, {"color": None, "marker": "o", "label": rf"$\eta={eta:g}$"}
        )
        marker_size = 115 if eta == 0.1 else 72
        marker = style["marker"]
        ax.scatter(
            row["mean_gap"],
            row["mean_tcsr"],
            s=marker_size,
            marker=marker,
            color=style["color"],
            edgecolors="black" if eta == 0.1 else "none",
            linewidths=0.5,
            zorder=4,
        )

        if eta == 0.01:
            dx, dy = -5.8, -0.038
        elif eta == 0.1:
            dx, dy = -2.8, 0.018
        else:
            dx, dy = 1.0, -0.01
        ax.text(
            row["mean_gap"] + dx,
            row["mean_tcsr"] + dy,
            style["label"],
            fontsize=8,
        )
        if row["failed"] > 0:
            ax.text(
                row["mean_gap"] + 1.0,
                row["mean_tcsr"] - 0.045,
                f"{row['failed']}/{row['total']} fail",
                fontsize=7,
                color=style["color"],
            )

    ax.annotate(
        "conservative\nmonitoring",
        xy=(rows[0]["mean_gap"], rows[0]["mean_tcsr"]),
        xytext=(22.0, 0.88),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        fontsize=7,
        ha="center",
    )
    late_row = next(row for row in rows if abs(row["eta"] - 0.9) < 1e-9)
    ax.annotate(
        "late monitoring",
        xy=(late_row["mean_gap"], late_row["mean_tcsr"]),
        xytext=(7.0, 0.79),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        fontsize=7,
        ha="center",
    )

    ax.set_xlabel("Temporal gap (s, successful only)")
    ax.set_ylabel("Mean TCSR over priors")
    ax.set_xlim(0, 31)
    ax.set_ylim(0.68, 1.04)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_title(r"Aggregated $\eta$ sensitivity")
    fig.tight_layout()
    save_figure(fig, output_base)


def plot_eta_story(eta_rows: list[dict], output_base: Path, layout: str = "wide") -> None:
    configure_matplotlib()
    priors = ["Under", "Correct", "Over"]
    x = np.arange(len(priors))
    by_eta = {
        eta: sorted(
            [row for row in eta_rows if abs(row["eta"] - eta) < 1e-9],
            key=lambda r: priors.index(r["prior"]),
        )
        for eta in sorted({row["eta"] for row in eta_rows})
    }

    if layout == "stacked":
        fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.2), sharex=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))

    ax = axes[0]
    for eta, rows in by_eta.items():
        style = ETA_STYLE.get(eta, {"color": None, "marker": "o", "label": rf"$\eta={eta:g}$"})
        y = [next(row["tcsr"] for row in rows if row["prior"] == prior) for prior in priors]
        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            linewidth=2.0,
            markersize=8 if eta == 0.1 else 6,
        )

    ax.set_title("(a) Temporal robustness")
    ax.set_ylabel("TCSR")
    ax.set_xticks(x, priors)
    ax.set_ylim(0.54, 1.04)
    ax.set_yticks([0.6, 0.8, 1.0])
    ax.grid(True, linestyle="--", alpha=0.35)
    legend_cols = 3 if layout == "wide" else 1
    legend_loc = "lower center" if layout == "wide" else "lower right"
    legend_anchor = (0.5, -0.02) if layout == "wide" else None
    ax.legend(
        loc=legend_loc,
        bbox_to_anchor=legend_anchor,
        ncol=legend_cols,
        framealpha=0.92,
    )
    if layout == "wide":
        ax.annotate(
            "late monitoring",
            xy=(2, 0.60),
            xytext=(1.33, 0.69),
            arrowprops={"arrowstyle": "->", "linewidth": 0.8},
            fontsize=8,
        )

    ax = axes[1]
    for eta, rows in by_eta.items():
        style = ETA_STYLE.get(eta, {"color": None, "marker": "o", "label": rf"$\eta={eta:g}$"})
        y = [next(row["gap"] for row in rows if row["prior"] == prior) for prior in priors]
        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=8 if eta == 0.1 else 6,
        )
        for x_i, y_i in zip(x, y):
            if np.isnan(y_i):
                ax.scatter(
                    [x_i],
                    [1.0],
                    marker="x",
                    color=style["color"],
                    s=52,
                    linewidths=1.4,
                    zorder=4,
                )

    ax.set_title("(b) Efficiency cost")
    ax.set_ylabel("Temporal gap (s)")
    ax.set_xticks(x, priors)
    ax.set_ylim(0, 41)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.text(-0.02, 2.6, "failed", color=ETA_STYLE[0.9]["color"], fontsize=8)
    ax.text(1.72, 2.6, "failed", color=ETA_STYLE[0.9]["color"], fontsize=8)
    if layout == "wide":
        ax.annotate(
            "early monitoring",
            xy=(1, 37.31),
            xytext=(1.15, 30.0),
            arrowprops={"arrowstyle": "->", "linewidth": 0.8},
            fontsize=8,
        )

    if layout == "stacked":
        axes[-1].set_xlabel("Initial prior condition")
        fig.tight_layout(h_pad=1.0)
    else:
        fig.tight_layout(w_pad=1.7)
    save_figure(fig, output_base)


def plot_detailed_tradeoff(
    eta_rows: list[dict], scalability_rows: list[dict], output_base: Path
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))

    ax = axes[0]
    for prior in ["Under", "Correct", "Over"]:
        rows = sorted([row for row in eta_rows if row["prior"] == prior], key=lambda r: r["eta"])
        if not rows:
            continue
        x = [row["monitor_count"] for row in rows]
        y = [row["tcsr"] for row in rows]
        style = PRIOR_STYLE[prior]
        ax.plot(
            x,
            y,
            label=prior,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=6,
        )

    ax.set_title(r"(a) Risk tolerance $\eta$")
    ax.set_xlabel("Monitoring count")
    ax.set_ylabel("TCSR")
    ax.set_xlim(-0.08, 2.12)
    ax.set_ylim(0.54, 1.04)
    ax.set_yticks([0.6, 0.8, 1.0])
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", framealpha=0.92)
    ax.text(0.04, 0.57, r"$\eta=0.9$", fontsize=8)
    ax.text(1.11, 0.96, r"$\eta=0.1$", fontsize=8)
    ax.text(1.74, 1.015, r"$\eta=0.01$", fontsize=8)

    ax = axes[1]
    rows = sorted(scalability_rows, key=lambda r: r["B"])
    x = [row["ct"] for row in rows]
    y = [row["gap"] for row in rows]
    b_values = [row["B"] for row in rows]
    ax.plot(x, y, color="crimson", marker="o", linewidth=2.2, markersize=6)
    for x_i, y_i, b in zip(x, y, b_values):
        dx, dy = (0.16, 0.45)
        if b == 20:
            dx, dy = (-1.18, -0.95)
        ax.text(x_i + dx, y_i + dy, rf"$B={b}$", fontsize=8)
    if 10 in b_values:
        idx = b_values.index(10)
        ax.axvline(x[idx], color="gray", linestyle="--", linewidth=1.0, alpha=0.8)

    ax.set_title(r"(b) Search size $B=W=D$")
    ax.set_xlabel("Computation time (s)")
    ax.set_ylabel("Temporal gap (s)")
    ax.set_xlim(0, max(x) + 0.8)
    ax.set_ylim(max(0, min(y) - 1.2), max(y) + 1.4)
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout(w_pad=1.6)
    save_figure(fig, output_base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot hyperparameter trade-off analysis.")
    parser.add_argument(
        "--eta_dir",
        type=Path,
        default=Path(
            "assets/results/offline_exp_result_0421_before_inst_chage/analysis/eta_sensitivity"
        ),
    )
    parser.add_argument(
        "--scalability_dir",
        type=Path,
        default=Path(
            "assets/results/offline_exp_result_0421_before_inst_chage/analysis/scalability_final"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/revision_comment/figs/hyperparameter_tradeoff_mpl"),
    )
    parser.add_argument(
        "--view",
        choices=[
            "selection",
            "detailed",
            "eta_story",
            "eta_aggregate",
            "eta_correct_incorrect",
            "eta_dual_axis",
            "eta_mismatch_makespan",
            "scalability_only",
            "both",
        ],
        default="selection",
        help="Plot compact selection, eta story, aggregated eta trade-off, correct/incorrect eta sensitivity, dual-axis eta sensitivity, mismatch-only makespan/TCSR, scalability-only, detailed eta figure, or both selection/detailed.",
    )
    parser.add_argument(
        "--layout",
        choices=["wide", "stacked"],
        default="wide",
        help="Layout used by --view eta_story.",
    )
    args = parser.parse_args()

    eta_table = args.eta_dir / "latex_tables" / "eta_sensitivity_overall.tex"
    eta_rows = parse_eta_overall_table(eta_table)
    if not eta_rows:
        eta_rows = aggregate_eta_from_json(args.eta_dir)

    scalability_rows = aggregate_scalability(args.scalability_dir)
    if not eta_rows:
        raise RuntimeError(f"No eta-sensitivity data found under {args.eta_dir}")
    if not scalability_rows:
        raise RuntimeError(f"No scalability data found under {args.scalability_dir}")

    if args.view in {"selection", "both"}:
        output = args.output
        if args.view == "both":
            output = args.output.with_name(f"{args.output.name}_selection")
        plot_selection_tradeoff(eta_rows, scalability_rows, output)
    if args.view == "eta_story":
        suffix = "" if args.layout == "wide" else "_singlecol"
        output = args.output.with_name(f"eta_sensitivity_story{suffix}")
        plot_eta_story(eta_rows, output, args.layout)
    if args.view == "eta_aggregate":
        output = args.output.with_name("eta_sensitivity_aggregate")
        plot_eta_aggregate(eta_rows, output)
    if args.view == "eta_correct_incorrect":
        output = args.output.with_name("eta_sensitivity_correct_incorrect")
        plot_eta_correct_incorrect(eta_rows, output)
    if args.view == "eta_dual_axis":
        points = collect_eta_condition_points(args.eta_dir)
        if not points:
            raise RuntimeError(f"No eta-sensitivity points found under {args.eta_dir}")
        output = args.output.with_name("eta_sensitivity_dual_axis")
        plot_eta_dual_axis(points, output)
    if args.view == "eta_mismatch_makespan":
        points = collect_eta_mismatch_points(args.eta_dir)
        if not points:
            raise RuntimeError(f"No mismatch eta-sensitivity points found under {args.eta_dir}")
        output = args.output.with_name("eta_sensitivity_mismatch_makespan")
        plot_eta_mismatch_makespan(points, output)
    if args.view == "scalability_only":
        output = args.output.with_name("scalability_tradeoff")
        plot_scalability_only(scalability_rows, output)
    if args.view in {"detailed", "both"}:
        output = args.output
        if args.view == "both":
            output = args.output.with_name(f"{args.output.name}_detailed")
        plot_detailed_tradeoff(eta_rows, scalability_rows, output)


if __name__ == "__main__":
    main()
