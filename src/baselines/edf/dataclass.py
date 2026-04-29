from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Tuple

from networkx import DiGraph

from src.models.task import Subtask


@dataclass
class ActionResult:
    action_full_name: str
    action_type: str
    cumulative_time: float  # 누적 시간 (이 액션이 종료된 시점)
    action_duration: float  # 이 액션에 걸린 소요 시간
    scene_positions: dict[str, Tuple[float, float, float]]
    held_object: Optional[str] = None
    success: bool = False

    def __repr__(self):
        return f"({self.action_full_name}, {self.action_type}, {self.cumulative_time}, {self.action_duration}, {self.held_object})"


@dataclass
class CompletedEntry:
    """
    완료된 Subtask에 대해, (Subtask, schedule_start_time, schedule_end_time)을 함께 저장
    """

    subtask: Subtask  
    # start, end time은 navigation을 포함한 시작 및 종료 시간
    schedule_start_time: float = float("inf")
    schedule_end_time: float = float("inf")
    sim_start_time: float = float("inf")
    sim_end_time: float = float("inf")
    actual_first_nav_duration: Optional[float] = None
    execution_status: bool = False
    schedule_nav_time: Optional[float] = None
    sim_nav_time: Optional[float] = None

    def __repr__(self):
        return f"({self.subtask.name}, {self.schedule_start_time} ~ {self.schedule_end_time})"


class SchedulerState(NamedTuple):
    """
    현재 스케쥴 상태를 저장하는 dataclass
    """

    # 현재 subtask
    subtask: Subtask
    # 수행된 subtask들 (현재 subtask 포함X)
    completed_entries: List[CompletedEntry]
    # 남은 subtask들
    remaining_subtasks: List[Subtask]
    # 현재 constraint
    constraints: DiGraph
    # 현재 절대 시간
    current_time: float
    # 현재 agent, object들의 position
    scene_positions: dict[str, list[float, float, float]]
    # 현재 agent가 들고 있는 object
    held_object: Optional[str]
    # agent의 위치 (landmark)
    agent_location: str = None




class SimulationNode(NamedTuple):
    """
    우선순위 큐에서 사용할 탐색 노드.
    - state: 실제 스케줄 상태 (SchedulerState)
    - deadline: 해당 subtask의 deadline
    - simulation_subtask: 현재 simulation 중인 subtask
    """

    deadline: float
    simulation_subtask: Subtask
    state: SchedulerState
    execution_time: float
    def __lt__(self, other):
        return self.deadline < other.deadline