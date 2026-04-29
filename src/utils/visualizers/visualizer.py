# utils/visualizers/visualizer.py

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Rectangle

from src.models.dataclass import CompletedEntry

from . import dag


def compute_start_times(durations: List[float]):
    """
    각 서브태스크 duration에 따라 시작 시간을 누적 계산.
    Returns: (startTimes, totalTime)
    """
    start_times = []
    current_time = 0.0
    for d in durations:
        start_times.append(current_time)
        current_time += d
    return start_times, current_time


def assign_chain_numbers(
    subtask_names: List[str], dependencies: List[List[str]], durations: List[float]
):
    """
    subtask_names: 실행 순서대로 정렬된 서브태스크 리스트
    dependencies: merge_groups()로 병합된 서브태스크 그룹
    durations: 각 서브태스크의 실행 시간 리스트

    return: (chains, independent_line)
      - chains: 각 subtask별로 할당된 체인 번호 리스트
      - independent_line: 의존성 없는 그룹(= 독립태스크)용 체인 번호
    """
    chain_mapping = {}
    independent_tasks = set(subtask_names)

    # Dependency 그룹 먼저 체인 번호 할당
    for chain_id, group in enumerate(dependencies, start=1):
        for subtask in group:
            chain_mapping[subtask] = chain_id
            if subtask in independent_tasks:
                independent_tasks.remove(subtask)

    # 독립적인 태스크 처리
    if independent_tasks:
        independent_chain_id = len(dependencies) + 1
        for t in independent_tasks:
            chain_mapping[t] = independent_chain_id
        independent_chain_line = independent_chain_id
    else:
        independent_chain_line = None

    # 각 체인의 시작 시간
    chain_start_times = {cid: float("inf") for cid in set(chain_mapping.values())}
    for subtask, cid in chain_mapping.items():
        idx = subtask_names.index(subtask)
        st_time = sum(durations[:idx])  # 단순 누적
        if st_time < chain_start_times[cid]:
            chain_start_times[cid] = st_time

    # 시작 시간이 빠른 체인을 위쪽으로
    sorted_chains = sorted(chain_start_times, key=lambda x: chain_start_times[x])
    max_chain_id = len(sorted_chains)
    new_chain_mapping = {
        chain_id: (max_chain_id - i) for i, chain_id in enumerate(sorted_chains)
    }

    # 최종 체인 리스트
    final_chains = [new_chain_mapping[chain_mapping[n]] for n in subtask_names]

    # independent_chain_line 업데이트
    final_independent_line = (
        new_chain_mapping[independent_chain_line]
        if independent_chain_line in new_chain_mapping
        else None
    )

    return final_chains, final_independent_line


def plot_subtask_timeline(
    ax,
    subtask_names: List[str],
    durations: List[float],
    chains: List[int],
    start_times: List[float],
    x_limit: float,
    independent_line: Optional[int],
):
    """
    여러 서브태스크(혹은 스케줄) 비교용 Gantt 차트.
    ax: matplotlib Axes
    x_limit: x축 최대값
    independent_line: 의존성 없는 체인을 표시할 때 사용
    """
    bar_height = 0.8
    num_chains = max(chains) if chains else 1

    # 파스텔 컬러
    pastel_colors = [
        (0.72, 0.85, 0.98),
        (0.98, 0.75, 0.75),
        (0.80, 0.92, 0.77),
        (0.99, 0.88, 0.70),
        (0.91, 0.79, 0.98),
        (0.75, 0.85, 0.98),
        (0.99, 0.92, 0.75),
        (0.79, 0.88, 0.99),
        (0.94, 0.80, 0.85),
        (0.85, 0.85, 0.85),
    ]

    # 독립태스크용 팔레트
    independent_colors = [
        (0.55, 0.80, 0.90),
        (0.90, 0.60, 0.45),
        (0.65, 0.85, 0.50),
        (0.80, 0.70, 0.40),
        (0.70, 0.60, 0.80),
        (0.80, 0.60, 0.85),
        (0.75, 0.85, 0.60),
        (0.60, 0.80, 0.80),
        (0.65, 0.70, 0.85),
        (0.80, 0.75, 0.65),
    ]

    # 독립 서브태스크 목록
    if independent_line is not None:
        independent_subtasks = [
            subtask_names[i]
            for i in range(len(subtask_names))
            if chains[i] == independent_line
        ]
        # 독립 subtasks 마다 고유 색상
        independent_task_colors = {
            task: independent_colors[i % len(independent_colors)]
            for i, task in enumerate(independent_subtasks)
        }
    else:
        independent_subtasks = []
        independent_task_colors = {}

    # Gantt 막대
    for i, name in enumerate(subtask_names):
        x_pos = start_times[i]
        y_pos = chains[i]
        w = durations[i]

        if (independent_line is not None) and (y_pos == independent_line):
            face_color = independent_task_colors.get(name, (0.8, 0.8, 0.8))
        else:
            color_idx = (y_pos - 1) % len(pastel_colors) if y_pos > 0 else 0
            face_color = pastel_colors[color_idx]

        rect = Rectangle(
            (x_pos, y_pos - bar_height / 2),
            w,
            bar_height,
            facecolor=face_color,
            edgecolor=(0.4, 0.4, 0.4),
            linewidth=0.8,
            alpha=0.9,
        )
        ax.add_patch(rect)
        # text
        ax.text(
            x_pos + w / 2,
            y_pos,
            name,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=(0.2, 0.2, 0.2),
        )

    # 같은 chain에서 연속 / gap 표시
    for i in range(len(subtask_names) - 1):
        if chains[i] == chains[i + 1] and chains[i] != independent_line:
            end_of_i = start_times[i] + durations[i]
            start_of_i1 = start_times[i + 1]
            y_pos = chains[i]

            # 연속(시간 차이 0)
            if abs(end_of_i - start_of_i1) < 1e-12:
                ax.plot(
                    [end_of_i, end_of_i],
                    [y_pos - bar_height / 2, y_pos + bar_height / 2],
                    color="k",
                    linewidth=1.2,
                )
            # gap
            elif end_of_i < start_of_i1:
                gap_start = end_of_i
                gap_width = start_of_i1 - end_of_i

                # base color
                if y_pos == independent_line:
                    base_color = independent_task_colors[subtask_names[i]]
                else:
                    base_color = pastel_colors[(y_pos - 1) % len(pastel_colors)]
                # 더 밝게
                lighter_color = tuple(bc + 0.5 * (1 - bc) for bc in base_color)

                gap_rect = Rectangle(
                    (gap_start, y_pos - bar_height / 2),
                    gap_width,
                    bar_height,
                    facecolor=lighter_color,
                    edgecolor="none",
                    alpha=0.3,
                )
                ax.add_patch(gap_rect)

    ax.set_xlim([0, x_limit])
    ax.set_ylim([0.4, num_chains + 0.6])
    ax.set_yticks([])
    ax.set_xlabel("Time")
    ax.grid(axis="x", linestyle="-", color="0.9")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def plot_completed_subtasks_gantt(
    completed_subtasks: List[CompletedEntry], save_path: Optional[str] = None
):
    """
    완료된 Subtask 리스트(예: CompletedEntry 형태) 기반으로 Gantt를 그린다.
    entry 형식: {"subtask": object, "start_time": float, "end_time": float}
    """
    if not completed_subtasks:
        print("No completed subtasks.")
        return

    # 시작 시간 기준 정렬
    sorted_entries = sorted(completed_subtasks, key=lambda e: e.sim_start_time)
    gantt_data = []
    current_time = 0.0

    for entry in sorted_entries:
        st_time = entry.sim_start_time
        ed_time = entry.sim_end_time
        sb_name = getattr(entry.subtask, "name")

        # 대기 구간
        if st_time > current_time:
            gantt_data.append(
                {"name": f"Wait until {sb_name}", "start": current_time, "end": st_time}
            )
        gantt_data.append({"name": sb_name, "start": st_time, "end": ed_time})
        current_time = ed_time

    # Matplotlib
    _, ax = plt.subplots(figsize=(10, 6))
    task_names = [g["name"] for g in gantt_data]
    start_times = [g["start"] for g in gantt_data]
    durations = [g["end"] - g["start"] for g in gantt_data]

    y_pos = np.arange(len(task_names))
    bars = ax.barh(y_pos, durations, left=start_times, color="skyblue")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(task_names)
    ax.invert_yaxis()  # 처음 수행된 작업이 위
    ax.set_xlabel("Time")
    ax.set_title("Gantt Chart (Completed Subtasks)")

    for i, _ in enumerate(bars):
        st = start_times[i]
        et = st + durations[i]
        ax.text(
            st + durations[i] / 2,
            y_pos[i],
            f"{st:.1f} - {et:.1f}",
            ha="center",
            va="center",
            color="black",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved Gantt to: {save_path}")
        plt.close()
    else:
        plt.show()


def _prepare_simulation_json_data(
    plan: List[CompletedEntry],
    approach_name: str,
    scene_name: str,
    constraints_graph: nx.DiGraph,
) -> Dict[str, Any]:
    """
    CompletedEntry 리스트와 메타데이터를 plot_gantt_final_cutoff가
    요구하는 Python 딕셔너리 형태로 변환합니다.
    """
    subtasks_for_json = []
    simulation_makespan = 0.0
    if plan:
        for entry in plan:
            start_time = (
                entry.sim_start_time
                if entry.sim_start_time != float("inf")
                else entry.schedule_start_time
            )
            end_time = (
                entry.sim_end_time
                if entry.sim_end_time != float("inf")
                else entry.schedule_end_time
            )

            subtasks_for_json.append(
                {
                    "subtask_name": entry.subtask.name,
                    "start_time_simulation": start_time,
                    "end_time_simulation": end_time,
                    "execution_status": entry.execution_status,
                }
            )
            if end_time != float("inf"):
                simulation_makespan = max(simulation_makespan, end_time)
        if simulation_makespan == 0 and subtasks_for_json:
            last_entry_end_time = subtasks_for_json[-1]["end_time_simulation"]
            if last_entry_end_time != float("inf"):
                simulation_makespan = last_entry_end_time

    saved_time_str = datetime.now().strftime("%Y-%m-%d_%H_%M")
    return {
        "name": f"{saved_time_str}_{scene_name}_{approach_name}",
        "approach": approach_name,
        "scene_name": scene_name,
        "saved_time": saved_time_str,
        "simulation_makespan": simulation_makespan,
        "plans": [
            {
                "subtasks": subtasks_for_json,
            }
        ],
        "constraints": constraints_graph,
    }


# --- UnionFind 클래스 ---
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, i):
        if i not in self.parent:
            self.parent[i] = i
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        if i not in self.parent:
            self.add(i)
        if j not in self.parent:
            self.add(j)
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i

    def add(self, i):
        if i not in self.parent:
            self.parent[i] = i


# --- Helper function to get base name for grouping split tasks ---
def _get_base_name_for_lane_grouping(name: str) -> str:
    """
    Extracts a base name from a task name, stripping prefixes like "EARLY_", "REMAIN_"
    and suffixes like "_part_X". This base name is used to group related split tasks
    onto the same lane in the Gantt chart.
    """
    processed_name = name
    # Strip "EARLY_" prefix first
    if processed_name.startswith("EARLY_"):
        processed_name = processed_name[len("EARLY_") :]
    # Then, strip "REMAIN_" prefix from the potentially already processed name
    if processed_name.startswith("REMAIN_"):
        processed_name = processed_name[len("REMAIN_") :]

    # Finally, handle "_part_" splitting
    name_parts = processed_name.split("_part_")
    return name_parts[0]


# --- 레인 할당 함수 ---
def merge_groups_for_lanes(
    constraints_graph: nx.DiGraph, ordered_subtask_names: List[str]
) -> List[List[str]]:
    """
    Groups subtasks for lane assignment in the Gantt chart.
    It first groups tasks based on their base names (e.g., "EARLY_TaskA", "REMAIN_TaskA", "TaskA_part_1"
    all belong to the "TaskA" group).
    Then, it considers explicit constraints from constraints_graph if provided.
    """
    uf = UnionFind()
    for name in ordered_subtask_names:
        uf.add(name)

    # 1. Name-based grouping for split tasks (EARLY_, REMAIN_, _part_)
    # This ensures that parts of the same original task are grouped together.
    original_to_parts = defaultdict(list)
    for name in ordered_subtask_names:
        base_name = _get_base_name_for_lane_grouping(name)
        original_to_parts[base_name].append(name)

    for base_name, parts in original_to_parts.items():
        if len(parts) > 1:
            # Sort parts by their original order to ensure consistent representative selection
            # and union order. This helps in deterministic behavior.
            sorted_parts_for_union = sorted(
                parts,
                key=lambda p_name: (
                    ordered_subtask_names.index(p_name)
                    if p_name in ordered_subtask_names
                    else float("inf")
                ),
            )
            first_part_in_group = sorted_parts_for_union[0]
            for i in range(1, len(sorted_parts_for_union)):
                uf.union(first_part_in_group, sorted_parts_for_union[i])

    # 2. Constraint-based grouping (Optional, can be added if needed)
    if constraints_graph:
        for u, v in constraints_graph.edges():
            # Ensure u and v are known tasks before attempting union
            if u in ordered_subtask_names and v in ordered_subtask_names:
                uf.union(u, v)
            elif u not in ordered_subtask_names:
                # 더 자세한 로그로 디버깅 정보 제공
                print(
                    f"Warning: Task '{u}' in constraint edge ({u} -> {v}) not in ordered_subtask_names. "
                    f"This may indicate that the task was planned but not executed. Skipping this edge for grouping."
                )
            elif v not in ordered_subtask_names:
                print(
                    f"Warning: Task '{v}' in constraint edge not in ordered_subtask_names. Skipping this edge for grouping."
                )

    # 3. Form final chains based on UnionFind groups
    groups = defaultdict(list)
    for name in ordered_subtask_names:
        groups[uf.find(name)].append(name)

    final_chains = []
    # Sort groups by the original index of their first member to maintain overall order
    temp_sorted_groups = sorted(
        groups.items(),
        key=lambda item: (
            ordered_subtask_names.index(item[1][0])
            if item[1] and item[1][0] in ordered_subtask_names
            else float("inf")
        ),
    )
    for _, members in temp_sorted_groups:
        # Sort members within each chain by their original order
        final_chains.append(
            sorted(
                members,
                key=lambda x: (
                    ordered_subtask_names.index(x)
                    if x in ordered_subtask_names
                    else float("inf")
                ),
            )
        )
    return final_chains


def assign_y_lanes_for_gantt_v3(
    ordered_subtask_names: List[str],
    execution_chains: List[List[str]],
    start_times_map: Dict[str, float],
    durations_map_viz: Dict[str, float],
):
    lanes_assignment_map = {}
    lane_available_time = defaultdict(float)
    if execution_chains:
        sorted_chains = sorted(
            execution_chains,
            key=lambda chain: (
                start_times_map.get(chain[0], float("inf")) if chain else float("inf")
            ),
        )
    else:
        sorted_chains = []

    max_lane_used = -1

    for chain in sorted_chains:
        if not chain:
            continue
        chain_can_start_at = start_times_map.get(chain[0], 0)
        assigned_lane = -1
        for lane_num in range(max_lane_used + 2):
            if lane_available_time[lane_num] <= chain_can_start_at + 0.001:
                assigned_lane = lane_num
                break
        if assigned_lane > max_lane_used:
            max_lane_used = assigned_lane

        for task_name in chain:
            lanes_assignment_map[task_name] = assigned_lane
            task_start = start_times_map.get(task_name, 0)
            task_duration_viz = durations_map_viz.get(task_name, 0)
            actual_placement_start = max(lane_available_time[assigned_lane], task_start)
            lane_available_time[assigned_lane] = (
                actual_placement_start + task_duration_viz
            )

    unique_assigned_lanes = sorted(list(set(lanes_assignment_map.values())))
    lane_remap = {old_lane: i + 1 for i, old_lane in enumerate(unique_assigned_lanes)}

    final_lanes_list = []
    last_non_wait_lane = 1
    if not ordered_subtask_names and not lane_remap:
        num_actual_lanes = 0
    elif not lane_remap and ordered_subtask_names:
        num_actual_lanes = 1
    else:
        num_actual_lanes = len(lane_remap) if lane_remap else 0

    for name in ordered_subtask_names:
        assigned_idx = lanes_assignment_map.get(name)
        if name.startswith("Wait Gap ("):
            final_lanes_list.append(last_non_wait_lane)
        elif assigned_idx is not None and lane_remap:
            current_mapped_lane = lane_remap.get(assigned_idx, 1)
            final_lanes_list.append(current_mapped_lane)
            if not name.startswith("Wait Gap ("):
                last_non_wait_lane = current_mapped_lane
        else:
            final_lanes_list.append(last_non_wait_lane)

    return final_lanes_list, (
        num_actual_lanes
        if num_actual_lanes > 0
        else (1 if ordered_subtask_names else 0)
    )


def plot_gantt_final_cutoff(
    simulation_data: Dict[str, Any],
    initial_plan_json_data: List[Dict],
    save_dir: str = "gantt_charts_final_cutoff_executable",
):
    plt.style.use("seaborn-v0_8-paper")
    font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    font_prop = fm.FontProperties(fname=font_path, size=10)
    plt.rcParams["font.family"] = font_prop.get_name()
    if not simulation_data.get("plans") or not simulation_data["plans"][0].get(
        "subtasks"
    ):
        print("오류: 시뮬레이션 데이터에 'plans' 또는 'subtasks' 정보가 부족합니다.")
        if not simulation_data.get("plans"):
            simulation_data["plans"] = [{"subtasks": []}]
        elif not simulation_data["plans"][0].get("subtasks"):
            simulation_data["plans"][0]["subtasks"] = []

    plan_data = simulation_data["plans"][0]
    subtasks_sim_data = plan_data.get("subtasks", [])
    simulation_makespan = float(simulation_data.get("simulation_makespan", 0))
    constraints_g = simulation_data.get("constraints")

    # subtasks_sim_data를 subtask_name을 키로 하는 딕셔너리로 변환 (조회용)
    subtasks_details_map = {st["subtask_name"]: st for st in subtasks_sim_data}

    if not subtasks_sim_data:
        print(
            "경고: 시뮬레이션 데이터에 서브태스크가 없습니다. 빈 차트가 생성될 수 있습니다."
        )

    subtask_names_in_order, start_times_map, durations_map_viz, colors_map = (
        [],
        {},
        {},
        {},
    )

    color_palette = {
        "success": "#5694C6",
        "failure": "#D65F5F",
        "monitoring": "#FFB761",
        "wait": "#E8E8E8",
        "dependency_normal": "#6084A8",
        "dependency_critical": "#C44E52",
        "text_on_bar": "#1A1A1A",
        "text_on_dark_bar": "#F5F5F5",
        "grid_color": "#D0D0D0",
    }

    current_sim_time = 0.0
    # 바 시각화 및 텍스트 표시 관련 임계값 조정
    # 1. 모니터링/분할 태스크의 최소 시각적 길이 (너무 작으면 안보임)
    min_visual_duration_monitoring = (
        simulation_makespan * 0.01 if simulation_makespan > 0 else 0.1
    )
    # 2. 텍스트 레이블을 *아예* 표시하지 않을 바의 최대 시각적 길이 (이것보다 짧으면 텍스트 없음)
    max_duration_for_no_text = (
        simulation_makespan * 0.015 if simulation_makespan > 0 else 0.15
    )
    # 3. 첫 단어만 표시할 바의 최대 시각적 길이 (max_duration_for_no_text < 길이 <= 이 값 이면 첫 단어)
    max_duration_for_short_text = (
        simulation_makespan * 0.05 if simulation_makespan > 0 else 0.5
    )
    # (이것보다 길면 전체 이름 시도)

    for st_data in subtasks_sim_data:
        name = st_data.get("subtask_name", "Unknown Subtask")
        sim_start, sim_end = st_data.get("start_time_simulation"), st_data.get(
            "end_time_simulation"
        )
        status = st_data.get("execution_status", None)

        if (
            sim_start is None
            or sim_start == float("inf")
            or sim_end is None
            or sim_end == float("inf")
        ):
            print(
                f"경고: 태스크 '{name}'의 시작 또는 종료 시간이 유효하지 않습니다. 건너뛰니다."
            )
            continue

        sim_start_f, sim_end_f = float(sim_start), float(sim_end)

        if sim_end_f < sim_start_f:
            print(
                f"경고: 태스크 '{name}'의 종료 시간({sim_end_f})이 시작 시간({sim_start_f})보다 빠릅니다. 건너뛰니다."
            )
            continue

        actual_current_duration = sim_end_f - sim_start_f
        visual_duration = actual_current_duration  # 기본 시각적 길이는 실제 길이

        is_monitoring_or_part = (
            "monitoring for" in name.lower()
            or "Monitoring for" in name
            or name.startswith("EARLY_")
            or name.startswith("REMAIN_")
            or "_part_" in name
            or st_data.get("monitored_subtask")
        )

        if (
            is_monitoring_or_part
            and actual_current_duration < min_visual_duration_monitoring
        ):
            visual_duration = min_visual_duration_monitoring  # 최소 시각적 길이 적용

        durations_map_viz[name] = visual_duration  # 시각화에 사용될 최종 바 길이 저장

        if sim_start_f > current_sim_time + 0.001:
            wait_name = f"Wait Gap ({current_sim_time:.1f}-{sim_start_f:.1f})"
            subtask_names_in_order.append(wait_name)
            start_times_map[wait_name] = current_sim_time
            durations_map_viz[wait_name] = sim_start_f - current_sim_time
            colors_map[wait_name] = color_palette["wait"]

        subtask_names_in_order.append(name)
        start_times_map[name] = sim_start_f
        # colors_map[name]은 아래 is_monitoring_or_part와 status에 따라 설정

        if is_monitoring_or_part and not name.startswith(
            "Wait Gap ("
        ):  # Wait Gap이 아닌 모니터링/분할 태스크
            colors_map[name] = color_palette["monitoring"]
        elif status is False:
            colors_map[name] = color_palette["failure"]
        else:
            colors_map[name] = color_palette["success"]
        current_sim_time = sim_end_f

    if not subtask_names_in_order and simulation_makespan > 0:
        wait_name = f"Total Span ({0.0:.1f}-{simulation_makespan:.1f})"
        subtask_names_in_order.append(wait_name)
        start_times_map[wait_name] = 0.0
        durations_map_viz[wait_name] = simulation_makespan
        colors_map[wait_name] = color_palette["wait"]
    elif not subtask_names_in_order and simulation_makespan == 0:
        print(
            "정보: 시뮬레이션 메이크스팬이 0이고 처리할 태스크가 없습니다. 빈 차트를 생성합니다."
        )
        simulation_makespan = 10

    non_wait_task_names = [
        name for name in subtask_names_in_order if not name.startswith("Wait Gap (")
    ]

    assigned_lanes_for_non_wait, num_potential_lanes_from_assign = [], 0

    if non_wait_task_names:
        execution_chains = merge_groups_for_lanes(constraints_g, non_wait_task_names)
        assigned_lanes_for_non_wait, num_potential_lanes_from_assign = (
            assign_y_lanes_for_gantt_v3(
                non_wait_task_names,
                execution_chains,
                {k: v for k, v in start_times_map.items() if k in non_wait_task_names},
                {
                    k: v
                    for k, v in durations_map_viz.items()
                    if k in non_wait_task_names
                },
            )
        )
    else:
        num_potential_lanes_from_assign = 1 if subtask_names_in_order else 0

    final_y_lanes = []
    non_wait_idx = 0
    last_valid_lane = 1
    for name in subtask_names_in_order:
        if name.startswith("Wait Gap ("):
            final_y_lanes.append(last_valid_lane)
        else:
            if non_wait_idx < len(assigned_lanes_for_non_wait):
                current_lane = assigned_lanes_for_non_wait[non_wait_idx]
                final_y_lanes.append(current_lane)
                last_valid_lane = current_lane
                non_wait_idx += 1
            else:
                final_y_lanes.append(last_valid_lane)

    if final_y_lanes:
        num_total_lanes = max(final_y_lanes)
    elif num_potential_lanes_from_assign > 0:
        num_total_lanes = num_potential_lanes_from_assign
    else:
        num_total_lanes = 1

    fig_height_base = 6
    fig_height_per_lane = 0.8
    fig_height = max(
        fig_height_base,
        num_total_lanes * fig_height_per_lane + (fig_height_base * 0.30 * 2),
    )
    fig_width = (
        max(15, simulation_makespan * 0.2 + 2) if simulation_makespan > 0 else 15
    )
    if fig_width > 30:
        fig_width = 30

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    bar_height = 0.6

    ax.yaxis.grid(
        True,
        linestyle="--",
        which="major",
        color=color_palette["grid_color"],
        alpha=0.5,
        zorder=0,
    )
    ax.xaxis.grid(
        True,
        linestyle=":",
        which="major",
        color=color_palette["grid_color"],
        alpha=0.4,
        zorder=0,
    )

    for i, name in enumerate(subtask_names_in_order):
        start_time = start_times_map[name]
        # durations_map_viz에는 이미 시각적 최소 길이가 반영된 값이 들어있음
        current_bar_visual_duration = durations_map_viz[name]
        color = colors_map[name]
        y_pos = final_y_lanes[i] if i < len(final_y_lanes) else 1

        rect = Rectangle(
            (start_time, y_pos - bar_height / 2),
            current_bar_visual_duration,
            bar_height,
            facecolor=color,
            edgecolor=mcolors.to_rgba(color_palette["text_on_bar"], alpha=0.7),
            linewidth=0.6,
            alpha=0.85,
            zorder=2,
        )
        ax.add_patch(rect)

        text_to_display = ""
        if not name.startswith("Wait Gap ("):
            # 텍스트 표시 조건: 바의 시각적 길이가 max_duration_for_no_text 보다 길어야 함
            if current_bar_visual_duration > max_duration_for_no_text:
                # 기본 이름 (M:, E:, R: 접두사 처리된 이름)
                processed_display_name = name
                prefix = ""
                if name.startswith("Monitoring for ") or name.startswith(
                    "monitoring for "
                ):
                    prefix_len = len("Monitoring for ")
                    prefix = "M: "
                    processed_display_name = name[prefix_len:]
                elif name.startswith("EARLY_"):
                    prefix = "E: "
                    processed_display_name = name[len("EARLY_") :]
                elif name.startswith("REMAIN_"):
                    prefix = "R: "
                    processed_display_name = name[len("REMAIN_") :]

                # _part_X 제거
                processed_display_name = processed_display_name.split("_part_")[0]

                # 짧은 바에 대한 처리: 첫 단어만 표시
                if current_bar_visual_duration <= max_duration_for_short_text:
                    first_word = processed_display_name.split(" ")[0]
                    text_to_display = prefix + first_word
                    if len(first_word) > 10:  # 첫 단어도 너무 길면 자르기
                        text_to_display = prefix + first_word[:8] + "…"
                else:  # 충분히 긴 바: 전체 이름 표시 (길이 제한 적용)
                    text_to_display = prefix + processed_display_name
                    # 전체 이름에 대한 길이 제한 (max_len_heuristic 대신 고정값 또는 다른 로직 사용 가능)
                    max_char_for_full_name = (
                        int(current_bar_visual_duration * 2.2) + 2
                    )  # 바 길이에 비례
                    if len(text_to_display) > max_char_for_full_name:
                        text_to_display = (
                            text_to_display[: max_char_for_full_name - 1] + "…"
                        )

            if text_to_display:  # 표시할 텍스트가 있을 경우에만 그리기
                r_bar, g_bar, b_bar = mcolors.to_rgb(color)
                brightness = (r_bar * 299 + g_bar * 587 + b_bar * 114) / 1000
                text_color_on_bar = (
                    color_palette["text_on_dark_bar"]
                    if brightness < 0.5
                    else color_palette["text_on_bar"]
                )
                ax.text(
                    start_time + current_bar_visual_duration / 2,
                    y_pos,
                    text_to_display,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color_on_bar,
                    clip_on=True,
                    zorder=3,
                    path_effects=[
                        path_effects.withStroke(
                            linewidth=0.5,
                            foreground=(
                                mcolors.to_rgba(text_color_on_bar, alpha=0.2)
                                if brightness < 0.5
                                else mcolors.to_rgba(
                                    color_palette["text_on_dark_bar"], alpha=0.2
                                )
                            ),
                        )
                    ],
                )

    if constraints_g:
        task_positions_for_arrows = {}
        for i, name in enumerate(subtask_names_in_order):
            if not name.startswith("Wait Gap ("):
                y_val_for_arrow = final_y_lanes[i] if i < len(final_y_lanes) else 1

                actual_task_duration = 0
                if name in subtasks_details_map:  # Wait Gap이 아닌 실제 태스크인 경우
                    task_detail = subtasks_details_map[name]
                    actual_start = task_detail.get("start_time_simulation", 0)
                    actual_end = task_detail.get("end_time_simulation", 0)
                    if actual_start != float("inf") and actual_end != float("inf"):
                        actual_task_duration = actual_end - actual_start
                    else:  # Fallback to visual duration if actual times are inf
                        actual_task_duration = durations_map_viz.get(name, 0)
                else:  # Wait Gap 또는 기타 목록에 없는 태스크 (이론상으론 non_wait_task_names에만 해당)
                    actual_task_duration = durations_map_viz.get(name, 0)

                task_positions_for_arrows[name] = {
                    "y": y_val_for_arrow,
                    "start_x": start_times_map.get(
                        name, 0
                    ),  # start_x는 subtask_names_in_order 기준
                    "end_x": start_times_map.get(name, 0)
                    + actual_task_duration,  # end_x는 실제 태스크 지속시간 기준
                }

        for u, v, edge_data in constraints_g.edges(data=True):
            if u in task_positions_for_arrows and v in task_positions_for_arrows:
                pos_u, pos_v = (
                    task_positions_for_arrows[u],
                    task_positions_for_arrows[v],
                )
                if pos_u["end_x"] <= pos_v["start_x"] + 0.01:
                    start_x, start_y = pos_u["end_x"], pos_u["y"]
                    end_x, end_y = pos_v["start_x"], pos_v["y"]
                    edge_info = edge_data.get("info", {})
                    interval_val = edge_info.get("Interval", 0)
                    is_critical_edge = edge_info.get("Urgency", False) or edge_info.get(
                        "IsCritical", False
                    )
                    arrow_color_edge = (
                        color_palette["dependency_critical"]
                        if is_critical_edge
                        else color_palette["dependency_normal"]
                    )
                    arrow_lw = 1.8 if is_critical_edge else 1.0
                    connection_rad = 0.1 if abs(start_y - end_y) > 0.1 else 0
                    ax.annotate(
                        "",
                        xy=(end_x, end_y),
                        xytext=(start_x, start_y),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            shrinkA=2,
                            shrinkB=2,
                            color=arrow_color_edge,
                            lw=arrow_lw,
                            connectionstyle=f"arc3,rad={connection_rad}",
                        ),
                        zorder=1,
                    )
                    text_x_interval, text_y_interval = (start_x + end_x) / 2, start_y
                    if abs(start_y - end_y) > 0.1:
                        text_y_interval = (start_y + end_y) / 2
                    else:
                        text_y_interval -= 0.15 * bar_height
                    interval_display_str = (
                        str(int(interval_val))
                        if isinstance(interval_val, (int, float))
                        and not np.isinf(interval_val)
                        and not np.isnan(interval_val)
                        else str(interval_val)
                    )
                    ax.text(
                        text_x_interval,
                        text_y_interval,
                        interval_display_str,
                        color=arrow_color_edge,
                        fontsize=6.5,
                        ha="center",
                        va="center",
                        bbox=dict(
                            boxstyle="round,pad=0.15",
                            fc=mcolors.to_rgba("white", alpha=0.7),
                            ec=mcolors.to_rgba(arrow_color_edge, alpha=0.5),
                            lw=0.5,
                        ),
                        zorder=3,
                    )

    ax.set_yticks(range(1, num_total_lanes + 1))
    ax.set_yticklabels([f"Lane {i}" for i in range(1, num_total_lanes + 1)], fontsize=9)
    ax.set_ylim(0.4, num_total_lanes + 0.6)
    ax.invert_yaxis()

    ax.set_xlabel("Time (s)", fontsize=10, labelpad=10)

    max_display_time = simulation_makespan if simulation_makespan > 0 else 10
    ax.set_xlim(-max_display_time * 0.02, max_display_time * 1.02)

    legend_elements = [
        plt.Rectangle(
            (0, 0), 1, 1, color=color_palette["success"], label="Success/Other"
        ),
        plt.Rectangle((0, 0), 1, 1, color=color_palette["failure"], label="Failure"),
        plt.Rectangle(
            (0, 0), 1, 1, color=color_palette["monitoring"], label="Monitoring/Part"
        ),  # 범례 수정
        plt.Rectangle((0, 0), 1, 1, color=color_palette["wait"], label="Wait/Gap"),
        plt.Line2D(
            [0],
            [0],
            color=color_palette["dependency_normal"],
            lw=1.5,
            label="Dependency (Non-Critical)",
        ),
        plt.Line2D(
            [0],
            [0],
            color=color_palette["dependency_critical"],
            lw=2.0,
            label="Dependency (Critical)",
        ),
    ]
    num_legend_cols, num_legend_rows = 3, (len(legend_elements) + 2) // 3
    legend_height_per_row = 0.06
    total_legend_height = num_legend_rows * legend_height_per_row
    xaxis_label_height, yaxis_last_tick_label_height, bottom_buffer = 0.05, 0.02, 0.04
    calculated_bottom_margin = (
        total_legend_height
        + xaxis_label_height
        + yaxis_last_tick_label_height
        + bottom_buffer
    )
    subplots_bottom = max(0.20, min(calculated_bottom_margin, 0.45))
    legend_y_anchor = bottom_buffer + 0.01
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y_anchor),
        fancybox=True,
        shadow=True,
        ncol=num_legend_cols,
        fontsize=8,
        frameon=True,
        edgecolor=color_palette["grid_color"],
    )
    fig.tight_layout(rect=[0.06, subplots_bottom, 0.97, 0.95])

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    timestamp_str = simulation_data.get(
        "saved_time", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    timestamp = timestamp_str.replace(":", "_").replace(" ", "_")
    safe_approach_name = "".join(
        c if c.isalnum() else "_"
        for c in simulation_data.get("approach", "DefaultApproach")
    )
    safe_scene_name = "".join(
        c if c.isalnum() else "_"
        for c in simulation_data.get("scene_name", "DefaultScene").replace(
            "_physics.json", ""
        )
    )
    output_file_name = (
        f"{safe_approach_name}_{safe_scene_name}_{timestamp}_gantt_academic.png"
    )
    final_save_path = Path(save_dir) / output_file_name
    try:
        plt.savefig(final_save_path, dpi=300, bbox_inches="tight")
        print(f"개선된 간트 차트가 '{final_save_path}'에 저장되었습니다.")
    except Exception as e:
        print(f"오류: 간트 차트를 저장하는 중 문제 발생 - {e}")
    plt.close(fig)


def plot_gantt_chart(
    completed_subtasks: List, save_folder_path: str, is_display: bool = False
):
    """
    주어진 completed_subtasks를 순서대로 가정, Gantt 차트를 그린다.
    - start/end가 없으므로, make_gantt_data(...)로 start/end를 임시로 계산.
    - 하나의 Gantt로 시각화 (단일 path 시나리오).

    Args:
        completed_subtasks: 순서대로 수행된 Subtask 리스트
        save_folder_path: 결과물을 저장할 폴더
        is_display: True면 plt.show(), False면 plt.close()
    """
    if not completed_subtasks:
        print("No completed subtasks to plot.")
        return

    # 1) Gantt 데이터 구성
    gantt_data = make_gantt_data(completed_subtasks)
    # gantt_data: [ {"name": ..., "start":..., "end":...}, {...}, ...]

    # 2) Matplotlib barh로 시각화
    fig, ax = plt.subplots(figsize=(12, 6))

    task_names = [d["name"] for d in gantt_data]
    start_times = [d["start"] for d in gantt_data]
    durations = [d["end"] - d["start"] for d in gantt_data]

    y_pos = np.arange(len(task_names))
    bars = ax.barh(y_pos, durations, left=start_times, color="skyblue")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(task_names)
    ax.invert_yaxis()  # y축을 뒤집어 위가 0이 되도록
    ax.set_xlabel("Time")
    ax.set_title("Gantt Chart of Completed Subtasks")

    # 바 위에 시작/끝 시간을 표시
    for i, bar in enumerate(bars):
        st = start_times[i]
        et = st + durations[i]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_y() + bar.get_height() / 2,
            f"{st:.1f} - {et:.1f}",
            ha="center",
            va="center",
            color="black",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()

    # 3) 파일 저장
    save_path = (
        Path(save_folder_path) / f"task_schedule_gantt_{len(completed_subtasks)}.png"
    )
    plt.savefig(save_path)
    print(f"Old Gantt chart saved at {save_path}")

    if is_display:
        plt.show()
    else:
        plt.close()


def make_gantt_data(completed_subtasks: List[CompletedEntry]) -> List[dict]:
    """
    주어진 completed_subtasks를 순서대로 가정하고,
    Subtask마다 (start, end)를 누적 계산하여 반환.

    Returns:
        A list of dictionaries, each having:
          {
            "name": <subtask.name>,
            "start": <start_time>,
            "end": <end_time>
          }
    """
    gantt_info = []
    current_time = 0.0

    for ce in completed_subtasks:

        duration = ce.subtask.duration.interval
        start = ce.sim_start_time
        end = ce.sim_end_time

        gantt_info.append({"name": ce.subtask.name, "start": start, "end": end})

        # 다음 Subtask의 시작 시간은 end로 누적
        current_time = end

    return gantt_info


def visualize(
    approach: str,
    output_path: Path,
    constraints: nx.DiGraph,
    plan: List[CompletedEntry],
    initial_plan_json_data: List[Dict],
    scene_name: str,
    is_display_dag: bool = False,
):
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    dag_save_path = output_path / f"{approach}_dag.png"
    try:
        dag.visualize_graph(
            constraints, save_path=str(dag_save_path), is_display=is_display_dag
        )
        print(f"DAG 이미지가 '{dag_save_path}'에 저장되었습니다.")
    except Exception as e:
        print(f"DAG 이미지 저장 중 오류 발생: {e}")

    if plan:
        simulation_data_dict = _prepare_simulation_json_data(
            plan, approach, scene_name, constraints
        )
        plot_gantt_final_cutoff(
            simulation_data=simulation_data_dict,
            initial_plan_json_data=initial_plan_json_data,
            save_dir=str(output_path),
        )
        plot_gantt_chart(
            completed_subtasks=plan,
            save_folder_path=str(output_path),
        )
