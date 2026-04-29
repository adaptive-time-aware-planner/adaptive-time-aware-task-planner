import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# --- 데이터 준비 (공통) ---
# 순서: [Under(60s), Under-mid(80s), Correct(100s), Over-mid(120s), Over(140s)]
DATA = {
    "Ours": {
        "sr": [],
        "makespan": [],
        "color": "red",
        "marker": "*",
        "style": "-",
    },
    "Ours (w/o Mon.)": {  # Ablation
        "sr": [],
        "makespan": [],
        "color": "orange",
        "marker": "p",
        "style": "-",
    },
    "tDAG + EDF": {
        "sr": [],
        "makespan": [],
        "color": "brown",
        "marker": "x",
        "style": "-.",
    },
    "tDAG + CPM": {
        "sr": [],
        "makespan": [],
        "color": "green",
        "marker": "s",
        "style": "--",
    },
    "Prog.": {
        "sr": [],
        "makespan": [],
        "color": "blue",
        "marker": "o",
        "style": ":",
    },
    "CaP": {
        "sr": [],
        "makespan": [],
        "color": "purple",
        "marker": "D",
        "style": ":",
    },
    "LaMMAP": {
        "sr": [],
        "makespan": [],
        "color": "teal",
        "marker": "^",
        "style": "--",
    },
}

APPROACH_LIST = {
    "dag_bayesian_DEFAULT": "Ours",
    "dag_bayesian_NONE_MONITORING": "Ours (w/o Mon.)",
    "dag_edf": "tDAG + EDF",
    "cpm": "tDAG + CPM",
    "progprompt": "Prog.",
    "cap_ai2thor_simulation": "CaP",
}

INIT_LIST_SIM = [f"init_{i}" for i in range(60, 150, 20)]
INIT_LIST_REAL = ["init_60", "init_100", "init_140"]

# Default to SIM, will be updated in main based on args
INIT_LIST = INIT_LIST_SIM

CONDITIONS_SIM = [
    "60s",
    "80s",
    "100s\n(Correct)",
    "120s",
    "140s",
]
CONDITIONS_REAL = ["60s", "100s\n(Correct)", "140s"]
TASK_CASE = [
    "tasks_2_constraints_1",
    "tasks_2_constraints_2",
    "tasks_3_constraints_1",
    "tasks_3_constraints_2",
]
METRIC_LIST = ["sr", "makespan"]

# 제거할 베이스라인 목록
EXCLUDE_BASELINES = []


def load_data(data_path: str) -> dict:
    with open(data_path, "r") as f:
        raw_data = json.load(f)

    # DATA 딕셔너리 초기화
    for key in DATA:
        DATA[key]["sr"] = []
        DATA[key]["makespan"] = []

    # 각 접근법에 대해 데이터 처리
    for approach_key, approach_name in APPROACH_LIST.items():
        # 제외할 베이스라인 스킵
        if approach_key in EXCLUDE_BASELINES:
            continue

        if approach_key not in raw_data:
            continue

        # 각 초기 조건(init_60, 80, 100, 120, 140)별로 순회
        for init_cond in INIT_LIST:
            sr_values = []
            makespan_values = []

            # 모든 태스크 케이스(tasks_2, 3, 4)의 값을 수집
            for task_case in TASK_CASE:
                if (
                    task_case in raw_data[approach_key]
                    and init_cond in raw_data[approach_key][task_case]
                ):
                    metrics = raw_data[approach_key][task_case][init_cond]
                    sr_values.append(metrics.get("sr", 0))
                    makespan_value = metrics.get("makespan", 0)

                    # CAP의 makespan이 비정상적으로 높은 경우 클리핑 (500초로 제한)
                    if approach_name == "CaP" and makespan_value > 500:
                        makespan_value = 500

                    makespan_values.append(makespan_value)

            # 수집된 값들의 평균 계산
            avg_sr = np.mean(sr_values) if sr_values else 0.0
            avg_makespan = np.mean(makespan_values) if makespan_values else 0.0

            # 결과 저장
            DATA[approach_name]["sr"].append(avg_sr)
            DATA[approach_name]["makespan"].append(avg_makespan)

    print("Data Loaded Successfully.")
    return DATA


def plot_trajectory(data: dict, output_path: str | None = None) -> None:
    """
    sr과 Makespan의 관계를 보여주는 궤적 그래프를 생성합니다.
    x축은 Makespan(효율성), y축은 sr(강건성)을 나타냅니다.

    Args:
        data (dict): 시각화에 사용할 데이터.
        output_path (str | None): 그래프를 저장할 파일 경로. None이면 화면에 표시.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # 각 Method별 궤적 그리기
    for name, d in data.items():
        # 데이터가 비어있으면(제외된 베이스라인) 그리지 않음
        if not d["sr"] or not d["makespan"]:
            continue

        ax.plot(
            d["makespan"],
            d["sr"],
            label=name,
            color=d["color"],
            marker=d["marker"],
            linestyle=d["style"],
            linewidth=3 if "Default" in name else 1.5,
            markersize=10 if "Default" in name else 8,
            alpha=1.0 if "Default" in name else 0.8,
        )

    # --- 영역 하이라이트 (핵심) ---
    # 1. Ours: Tight Cluster (Robustness)
    ours_center = (
        np.mean(data["Ours"]["makespan"]),
        np.mean(data["Ours"]["sr"]),
    )
    cluster_circle = Ellipse(
        ours_center, width=30, height=6, angle=0, color="red", alpha=0.1
    )
    ax.add_patch(cluster_circle)
    ax.text(
        ours_center[0],
        ours_center[1] + 4,
        "Robust & Efficient\n(Sweet Spot)",
        color="red",
        ha="center",
        fontweight="bold",
        fontsize=14,
    )

    # --- 축 및 설정 ---
    ax.set_xlabel("Makespan (s) ↓ (Efficiency)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Success Rate (%) ↑ (Robustness)", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=14)

    # 그리드 및 범례
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=14)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"그래프가 저장되었습니다: {output_path}")
    else:
        plt.show()


def plot_separate_metrics(
    data: dict, conditions: list, output_path: str | None = None
) -> None:
    """
    sr과 Makespan을 각각의 그래프로 분리하여 보여줍니다.
    x축은 초기 믿음 조건, y축은 각 메트릭의 값을 나타냅니다.

    Args:
        data (dict): 시각화에 사용할 데이터.
        conditions (list): x축 레이블 리스트.
        output_path (str | None): 그래프를 저장할 파일 경로. None이면 화면에 표시.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    metrics_info = [
        (
            "sr",
            "Success Rate across Initial Belief Conditions",
            "Success Rate (%)",
        ),
    ]

    method_order = [
        "Ours",
        "Ours (w/o Mon.)",
        "tDAG + CPM",
        "tDAG + EDF",
        "Prog.",
        "CaP",
    ]

    for col, (metric_key, title, ylabel) in enumerate(metrics_info):
        for name in method_order:
            # 데이터가 비어있으면(제외된 베이스라인) 그리지 않음
            if not data[name][metric_key]:
                continue

            d = data[name]
            y_data = d[metric_key]
            lw = 3.5 if "Default" in name else 1.8
            alpha = 1.0 if "Default" in name else 0.7

            ax.plot(
                conditions,
                y_data,
                label=name,
                color=d["color"],
                linestyle=d["style"],
                marker=d["marker"],
                linewidth=lw,
                alpha=alpha,
                markersize=9,
            )

        ax.set_ylabel(ylabel, fontsize=16, labelpad=10)
        ax.set_xlabel(
            r"Initial Belief Condition ($\Delta_0$)", fontsize=16, labelpad=10
        )
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.grid(True, linestyle="--", alpha=0.4)

        # Correct 조건 강조 (빨간색 세로줄, 약간 투명하게)
        # conditions 리스트에서 "100s\n(Correct)"의 인덱스를 찾아서 표시
        try:
            correct_idx = conditions.index("100s\n(Correct)")
            ax.axvline(x=correct_idx, color="red", alpha=0.3, linewidth=1.5)
        except ValueError:
            pass  # 100s가 없으면 표시하지 않음

    # 범례 통합
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(0.99, 0.65),  # bbox_to_anchor 조정
        ncol=1,
        fontsize=14,
        framealpha=0.9,
    )

    plt.tight_layout()
    # tight_layout 후 하단 여백 확보
    plt.subplots_adjust(bottom=0.25)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"그래프가 저장되었습니다: {output_path}")
    else:
        plt.show()


def save_aggregated_json(
    data: dict, output_path: str = "aggregated_result.json"
) -> None:
    """
    통합된 결과(sr, Makespan)를 JSON 파일로 저장합니다.
    """
    # 저장할 데이터 구조 변환
    export_data = {}
    for method, metrics in data.items():
        if not metrics["sr"] and not metrics["makespan"]:
            continue

        method_results = []
        for i, condition in enumerate(INIT_LIST):
            # 인덱스 에러 방지
            sr_val = metrics["sr"][i] if i < len(metrics["sr"]) else 0.0
            makespan_val = (
                metrics["makespan"][i] if i < len(metrics["makespan"]) else 0.0
            )

            method_results.append(
                {
                    "init_condition": condition,
                    "sr": float(sr_val),
                    "makespan": float(makespan_val),
                }
            )

        export_data[method] = method_results

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=4)
    print(f"통합 결과가 저장되었습니다: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        sr(Success Rate)과 Makespan 성능 지표를 시각화합니다.
        두 가지 플롯 타입을 선택할 수 있습니다:
        1. trajectory: sr과 Makespan의 관계를 2D 궤적으로 표시 (기본값)
        2. separate: sr과 Makespan을 각각 별도의 라인 그래프로 표시
        """
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=None,
        help="데이터 파일 경로를 지정합니다. 지정하지 않으면 experiment_type에 따른 기본 경로를 사용합니다.",
    )

    parser.add_argument(
        "--save_json",
        action="store_true",
        default=True,
        help="통합된 결과를 JSON 파일로 저장합니다 (기본값: True).",
    )

    parser.add_argument(
        "--experiment_type",
        type=str,
        choices=["sim", "real"],
        default="sim",
        help="실험 타입을 지정합니다 (sim: 시뮬레이션, real: 리얼 월드). 기본값: sim",
    )

    args = parser.parse_args()

    # Set global INIT_LIST and prepare conditions based on experiment type
    if args.experiment_type == "sim":
        INIT_LIST[:] = INIT_LIST_SIM
        conditions = CONDITIONS_SIM
        default_path = Path(
            "assets/results/260206_conflict_aware_wait_clean/unified_analysis_summary.revised.json"
        )
    else:
        INIT_LIST[:] = INIT_LIST_REAL
        conditions = CONDITIONS_REAL
        default_path = Path(
            "assets/results/260208_final_real_exp_result/unified_analysis_summary.revised.json"
        )

    # Determine data path
    if args.root_dir:
        data_path = args.root_dir / "unified_analysis_summary.revised.json"
    else:
        data_path = default_path

    print(f"Loading data from: {data_path}")
    load_data(data_path)

    if args.save_json:
        # Save to the same directory as the input file
        output_json_path = data_path.parent / "aggregated_result.json"
        save_aggregated_json(DATA, output_json_path)

    # plot_trajectory(DATA, args.data_path / "trajectory.png")
    # Save plot to the same directory as the input file
    output_plot_path = data_path.parent / "separate.png"
    plot_separate_metrics(DATA, conditions, output_plot_path)
