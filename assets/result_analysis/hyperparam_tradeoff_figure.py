"""2-panel hyperparameter trade-off figure for paper."""

from __future__ import annotations

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "legend.title_fontsize": 8,
})

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Colors & styles ───────────────────────────────────────────────────────────

PRIOR_COLOR = {
    "UNDER_ESTIMATE":   "#E07B54",
    "CORRECT_ESTIMATE": "#4C9B6F",
    "OVER_ESTIMATE":    "#5B8EC4",
}
PRIOR_MARKER = {
    "UNDER_ESTIMATE":   "^",
    "CORRECT_ESTIMATE": "o",
    "OVER_ESTIMATE":    "s",
}
PRIOR_LABEL = {
    "UNDER_ESTIMATE":   "Under",
    "CORRECT_ESTIMATE": "Correct",
    "OVER_ESTIMATE":    "Over",
}

CASE_COLOR = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
CASE_LABEL = {
    "tasks_2_constraints_1": "T2C1",
    "tasks_2_constraints_2": "T2C2",
    "tasks_3_constraints_1": "T3C1",
    "tasks_3_constraints_2": "T3C2",
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_eta_data() -> dict:
    base = PROJECT_ROOT / "assets/results/offline_exp_result_0421_before_inst_chage/offline_batch_eta_sensitivity"
    priors = list(PRIOR_COLOR.keys())
    etas = ["0.01", "0.1", "0.9"]
    acc = defaultdict(lambda: defaultdict(lambda: {"tsr": [], "mon": []}))

    for f in base.rglob("*.json"):
        name = f.stem
        if "w10_d10" not in name:
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if "timing_success_rate_sched" not in d:
            continue

        prior = next((p for p in priors if p in str(f)), None)
        eta = next((e for e in etas if f"eta{e}" in name), None)
        if prior is None or eta is None:
            continue

        tsr = d["timing_success_rate_sched"]
        mon_dict = d.get("actual_monitor_count_by_interval") or {}
        vals = [v for v in mon_dict.values() if v is not None]
        mon = float(np.mean(vals)) if vals else 0.0

        acc[prior][eta]["tsr"].append(float(tsr) if tsr is not None else 0.0)
        acc[prior][eta]["mon"].append(mon)

    result = {}
    for prior in priors:
        result[prior] = {}
        for eta in etas:
            v = acc[prior][eta]
            result[prior][eta] = {
                "tsr": float(np.mean(v["tsr"])) if v["tsr"] else float("nan"),
                "mon": float(np.mean(v["mon"])) if v["mon"] else float("nan"),
            }
    return result


def load_scalability_data() -> tuple[dict, list]:
    folders = [
        PROJECT_ROOT / f"assets/results/offline_exp_result_0421_before_inst_chage"
        f"/analysis/scalability_final"
        f"/decomposed_final_revision_metadata_260402_v{i}/offline_analysis_summary.json"
        for i in range(1, 6)
    ]
    beams = ["w1_d1", "w10_d10", "w20_d20"]
    cases = list(CASE_LABEL.keys())
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for f in folders:
        d = json.loads(f.read_text())
        for beam in beams:
            key = f"CORRECT_ESTIMATE__bayesian__DEFAULT__{beam}__eta0.1"
            if key not in d:
                continue
            for case in cases:
                if case not in d[key]:
                    continue
                acc[beam][case]["gap"].append(d[key][case]["makespan_gap"])
                acc[beam][case]["ct"].append(d[key][case]["computation_time"])

    result = {}
    for beam in beams:
        result[beam] = {}
        for case in cases:
            result[beam][case] = {
                "gap": float(np.mean(acc[beam][case]["gap"])),
                "ct":  float(np.mean(acc[beam][case]["ct"])),
            }
    return result, cases


# ── Panel (a): eta sensitivity ────────────────────────────────────────────────

def plot_eta(ax, eta_data: dict) -> None:
    priors = list(PRIOR_COLOR.keys())
    etas = ["0.01", "0.1", "0.9"]
    ETA_SIZE = {"0.01": 55, "0.1": 110, "0.9": 55}
    ETA_ALPHA = {"0.01": 0.6, "0.1": 1.0, "0.9": 0.6}

    for prior in priors:
        color = PRIOR_COLOR[prior]
        marker = PRIOR_MARKER[prior]
        xs = [eta_data[prior][e]["mon"] for e in etas]
        ys = [eta_data[prior][e]["tsr"] for e in etas]

        ax.plot(xs, ys, "-", color=color, linewidth=1.0, alpha=0.5, zorder=1)
        for i, eta in enumerate(etas):
            ax.scatter(xs[i], ys[i],
                       s=ETA_SIZE[eta], color=color, marker=marker,
                       edgecolors="black" if eta == "0.1" else color,
                       linewidths=1.2 if eta == "0.1" else 0.4,
                       alpha=ETA_ALPHA[eta], zorder=3)

    # eta label annotations (Correct prior 기준)
    label_offset = {
        "0.01": (0.04,  0.006),
        "0.1":  (0.04, -0.012),
        "0.9":  (0.04,  0.006),
    }
    for eta in etas:
        x = eta_data["CORRECT_ESTIMATE"][eta]["mon"]
        y = eta_data["CORRECT_ESTIMATE"][eta]["tsr"]
        ox, oy = label_offset[eta]
        ax.annotate(
            rf"$\eta={eta}$", (x, y),
            xytext=(x + ox, y + oy),
            fontsize=7, color="dimgray",
        )

    # η=0.9 failure zone annotation
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.annotate(
        "Fails under\nprior mismatch",
        xy=(0.05, 0.604), xytext=(0.4, 0.63),
        fontsize=6.5, color="#C44E52",
        arrowprops=dict(arrowstyle="->", color="#C44E52", lw=0.7),
    )

    # legend
    handles = [
        plt.Line2D([0], [0], marker=PRIOR_MARKER[p], color=PRIOR_COLOR[p],
                   linestyle="None", markersize=5, label=PRIOR_LABEL[p])
        for p in priors
    ]
    ax.legend(handles=handles, title="Init Prior", loc="lower right",
              framealpha=0.85, handletextpad=0.3, borderpad=0.4)

    ax.set_xlabel("Monitoring Count per Interval")
    ax.set_ylabel("TCSR")
    ax.set_title(r"(a) Risk Tolerance $\eta$ Selection", fontweight="bold")
    ax.set_xlim(-0.1, 2.2)
    ax.set_ylim(0.56, 1.04)
    ax.grid(True, linestyle="--", alpha=0.35)


# ── Panel (b): scalability ────────────────────────────────────────────────────

def plot_scalability(ax, sc_data: dict, cases: list) -> None:
    beams = ["w1_d1", "w10_d10", "w20_d20"]
    B_LABEL = {"w1_d1": "B=1", "w10_d10": "B=10", "w20_d20": "B=20"}

    for i, case in enumerate(cases):
        color = CASE_COLOR[i]
        cts  = [sc_data[b][case]["ct"]  for b in beams]
        gaps = [sc_data[b][case]["gap"] for b in beams]

        ax.plot(cts, gaps, "-", color=color, linewidth=1.2, alpha=0.7, zorder=2)
        for b, ct, gap in zip(beams, cts, gaps):
            is_b10 = (b == "w10_d10")
            ax.scatter(ct, gap,
                       s=80 if is_b10 else 45,
                       color=color,
                       edgecolors="black" if is_b10 else color,
                       linewidths=1.0 if is_b10 else 0.3,
                       zorder=3)

    # B label annotations (T3C2 기준, 가장 분리됨)
    ref = "tasks_3_constraints_2"
    label_offset = {
        "w1_d1":   (-0.3,  1.5),
        "w10_d10": ( 0.3, -2.0),
        "w20_d20": ( 0.5,  1.0),
    }
    for b in beams:
        ct  = sc_data[b][ref]["ct"]
        gap = sc_data[b][ref]["gap"]
        ox, oy = label_offset[b]
        ax.annotate(B_LABEL[b], (ct, gap),
                    xytext=(ct + ox, gap + oy),
                    fontsize=7.5, fontweight="bold", color="black")

    # legend
    handles = [
        plt.Line2D([0], [0], color=CASE_COLOR[i], linewidth=1.5,
                   marker="o", markersize=4, label=CASE_LABEL[c])
        for i, c in enumerate(cases)
    ]
    ax.legend(handles=handles, loc="upper left",
              framealpha=0.85, handletextpad=0.3, borderpad=0.4)

    ax.set_xlabel("Computation Time (s)")
    ax.set_ylabel("Temporal Gap (s)")
    ax.set_title(r"(b) Search Size $B\,(=\!W\!=\!D)$ Selection", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    eta_data = load_eta_data()
    sc_data, cases = load_scalability_data()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    plt.subplots_adjust(wspace=0.30)

    plot_eta(axes[0], eta_data)
    plot_scalability(axes[1], sc_data, cases)

    out_dir = PROJECT_ROOT / "assets/results/offline_exp_result/analysis/scalability/latex_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = out_dir / f"hyperparam_tradeoff.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
