from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple

from networkx import DiGraph

if TYPE_CHECKING:
    from src.models.task import Subtask


class TaskExecutionStatus(Enum):
    """
    Subtask 실행 상태를 나타내는 Enum
    """

    NOT_EXECUTED = auto()
    SUCCESS = auto()
    FAILURE = auto()


class SchedulerState(NamedTuple):
    """
    현재 스케쥴 상태를 저장하는 dataclass
    """

    # 현재 subtask
    subtask: Subtask
    # 수행된 subtask들 (현재 subtask 포함)
    completed_entries: List["CompletedEntry"]
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
    - heuristic_cost: 지금까지 누적된 비용 (높을수록 우선)
    - depth: 현재 탐색 깊이
    - tie_breaker: 우선순위가 같을 때 순서 결정용
    - state: 실제 스케줄 상태 (SchedulerState)
    - risk_level: 제약 조건 위반 위험도 (0: Safe, 1: Warning, 2: Violation)
    """

    heuristic_cost: float
    depth: int
    tie_breaker: int
    parent_node: Optional["SimulationNode"]
    state: SchedulerState
    # 제약 조건 위반 위험도 (0: Safe, 1: Warning, 2: Violation)
    risk_level: int = 0


class TimeSlot(NamedTuple):
    """
    Subtask 간의 제약 시간을 저장하는 NamedTuple
    """

    # 해당 subtask에서 in/out하는 제약 시간
    interval: int
    # 해당 subtask에서 in/out하는 제약 critical한지 여부
    is_critical: bool
    # 해당 subtask에서 in/out하는 제약과 연결된 subtask 이름
    related_subtask_name: Optional[str]

    def __repr__(self):
        return f"({self.interval}, {self.is_critical}, {self.related_subtask_name},)"


class SchedulingDue(NamedTuple):
    """
    다음 critical subtask로 인해 현재 후보군에게 영향을 주는 스케줄링 마감 정보를 저장.
    """

    # 현재 후보군이 이 시간까지는 완료되는 것이 좋음 (다음 critical subtask의 시작 시간)
    due_date: float
    # 이 마감 시간을 유발한 (다음) critical subtask의 이름
    due_related_sub_name: Optional[str] = None

    def __repr__(self):
        return f"(due_date={self.due_date}, due_related_sub_name='{self.due_related_sub_name}')"


@dataclass
class ActionResult:
    action_full_name: str
    action_type: str
    cumulative_time: float  # 누적 시간 (이 액션이 종료된 시점)
    action_duration: float  # 이 액션에 걸린 소요 시간
    scene_positions: dict[str, Tuple[float, float, float]]
    held_object: Optional[str] = None
    success: bool = False
    first_nav_duration: Optional[float] = None

    def __repr__(self):
        return f"({self.action_full_name}, {self.action_type}, {self.cumulative_time}, {self.action_duration}, {self.held_object})"


@dataclass
class ActionSimulationLog:
    results: list[ActionResult] = field(default_factory=list)

    def add_result(
        self,
        action_full_name: str,
        action_type: str,
        cumulative_time: float,
        action_duration: float,
        scene_positions: dict[str, Tuple[float, float, float]],
        held_object: Optional[str] = None,
        success: bool = False,
    ):
        self.results.append(
            ActionResult(
                action_full_name=action_full_name,
                action_type=action_type,
                cumulative_time=cumulative_time,
                action_duration=action_duration,
                scene_positions=scene_positions,
                held_object=held_object,
                success=success,
            )
        )

    def get_total_navigate_duration(self) -> float:
        """
        action_type이 'NAVIGATE_TO'인 액션들만 골라서 action_duration의 합을 구한다.
        """
        total = 0.0
        for result in self.results:
            if result.action_type.upper() == "NAVIGATE_TO":
                total += result.action_duration
        return total

    def total_time_used(self) -> float:
        """
        전체 액션 중 가장 마지막 액션의 time_used(누적 시간)를 반환.
        없으면 0.0을 반환.
        """
        if not self.results:
            return 0.0
        # 마지막 ActionResult의 time_used가 전체 시뮬레이션 누적 시간
        return self.results[-1].cumulative_time

    def filter_by_action_type(self, action_type: str) -> list[ActionResult]:
        """
        특정 action_type(대소문자 무관)에 해당하는 모든 ActionResult를 리스트로 반환.
        """
        action_type_upper = action_type.upper()
        return [
            res for res in self.results if res.action_type.upper() == action_type_upper
        ]

    def count_actions(self, action_type: Optional[str] = None) -> int:
        """
        특정 action_type에 해당하는 액션의 개수를 세거나,
        action_type이 None이면 전체 액션 개수를 반환한다.
        """
        if action_type is None:
            return len(self.results)
        action_type_upper = action_type.upper()
        return sum(
            1 for res in self.results if res.action_type.upper() == action_type_upper
        )

    def get_actions(self) -> List[str]:
        """
        모든 액션 이름을 리스트로 반환한다.
        """
        return [res.action_full_name for res in self.results]


@dataclass
class CompletedEntry:
    """
    완료된 Subtask에 대해, (Subtask, schedule_start_time, schedule_end_time, sim_start_time, sim_end_time, execution_status)을 함께 저장
    스케쥴러 상에서 완료된 시간, 시뮬레이션 상에서 완료된 시간, 시뮬레이션 실행 상태를 함께 저장
    """

    subtask: Subtask
    # start, end time은 navigation을 포함한 시작 및 종료 시간
    schedule_start_time: float = float("inf")
    schedule_end_time: float = float("inf")
    sim_start_time: float = float("inf")
    sim_end_time: float = float("inf")
    # 첫 번째 navigation 액션의 소요 시간
    actual_first_nav_duration: Optional[float] = None
    sim_nav_time: Optional[float] = None
    schedule_nav_time: Optional[float] = None
    # Simulation / Real-world에서 실행 성공 상태
    execution_status: "TaskExecutionStatus" = TaskExecutionStatus.NOT_EXECUTED

    def __repr__(self):
        return f"({self.subtask.name}, {self.schedule_start_time} ~ {self.schedule_end_time}, {self.sim_start_time} ~ {self.sim_end_time}, {self.execution_status})"


@dataclass
class Candidate:
    """
    Subtask의 실행 가능 여부를 판단하기 위한 NamedTuple
    """

    subtask: Subtask
    # subtask이 critical인지 여부
    is_critical: bool
    # subtask의 실제 상호작용 시작 예상 시간
    actual_interaction_start_time: Optional[float] = None
    # subtask의 시간 제약 로직 상 상호작용 시작 시간
    logical_interaction_start_time: Optional[float] = None
    # subtask의 첫 번째 navigation 액션의 소요 시간
    estimated_first_nav_duration: float = 0.0
    # 고려할 스케줄링 마감시간
    scheduling_due: SchedulingDue = SchedulingDue(
        due_date=float("inf"), due_related_sub_name=None
    )
    critical_context: Optional["CriticalContext"] = None

    def __repr__(self):
        return (
            f"({self.subtask.name}; duration : {self.subtask.duration.interval}, actual_interaction_start_time = {self.actual_interaction_start_time}, "
            f"logical_interaction_start_time = {self.logical_interaction_start_time}, scheduling_due = {self.scheduling_due}, is_critical = {self.is_critical})"
        )


@dataclass
class CriticalContext:
    source_subtask: Optional[str]
    source_end_time: Optional[float]
    interval: float
    logical_start_time: Optional[float]
