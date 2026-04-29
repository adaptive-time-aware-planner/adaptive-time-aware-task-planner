from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, TypeAlias

import networkx as nx
from networkx import DiGraph

from src.models.dataclass import (
    ActionResult,
    Candidate,
    CompletedEntry,
    CriticalContext,
    SchedulingDue,
    SimulationNode,
    TaskExecutionStatus,
    TimeSlot,
)
from src.models.task import Subtask
from src.scheduler.action_handler import ActionHandler
from src.utils.common import create_module_logger
from src.utils.config import EPSILON
from src.utils.config.constants import (
    CONSTRAINT_MERGE_THRESHOLD,
    RISK_GRACE_SECONDS,
    TIMING_TOLERANCE_ABS,
)

log = create_module_logger(__name__, True, logging.DEBUG)

TimeSlotCacheKey: TypeAlias = Tuple[int, str, str]


class ConstraintHandler:

    def __init__(self, action_handler: "ActionHandler"):
        """
        ConstraintHandler 초기화. ActionHandler 인스턴스를 주입받습니다.
        """
        self.action_handler = action_handler
        self._search_time_slot_cache: Optional[
            Dict[TimeSlotCacheKey, Tuple[TimeSlot, ...]]
        ] = None
        self._search_cache_hits = 0
        self._search_cache_misses = 0
        log.debug("ConstraintHandler initialized with ActionHandler.")

    def begin_search_session(
        self, cache: Dict[TimeSlotCacheKey, Tuple[TimeSlot, ...]]
    ) -> None:
        """Attach a search-scoped cache for repeated time-slot lookups.

        Args:
            cache: Cache owned by the scheduler for the current search call.
        """

        self._search_time_slot_cache = cache
        self._search_cache_hits = 0
        self._search_cache_misses = 0

    def end_search_session(self) -> tuple[int, int]:
        """Detach the search-scoped cache and return cache statistics.

        Returns:
            A tuple of `(cache_hits, cache_misses)` collected in the current search.
        """

        stats = (self._search_cache_hits, self._search_cache_misses)
        self._search_time_slot_cache = None
        self._search_cache_hits = 0
        self._search_cache_misses = 0
        return stats

    @staticmethod
    def _has_failed_execution(pred_entry: CompletedEntry) -> bool:
        """Return whether a predecessor entry represents an explicit failure.

        Args:
            pred_entry: Completed predecessor entry to inspect.

        Returns:
            ``True`` when the predecessor explicitly failed, otherwise ``False``.
        """

        execution_status = pred_entry.execution_status
        if isinstance(execution_status, bool):
            return execution_status is False
        if execution_status == TaskExecutionStatus.FAILURE:
            return True
        return getattr(execution_status, "name", None) == "FAILURE"

    @staticmethod
    def _collapse_monitoring_residual_context(
        raw_context: list[tuple[str, float, float, dict]],
    ) -> list[tuple[str, float, float]]:
        """Keep only the latest monitoring residual edge for each critical family."""

        collapsed_context: list[tuple[str, float, float]] = []
        latest_residual_by_family: dict[str, tuple[str, float, float]] = {}

        for pred_name, pred_end_time, interval, info in raw_context:
            if info.get("IsMonitoringResidual", False):
                family_key = info.get("CriticalFamilyKey")
                if family_key is not None:
                    prev = latest_residual_by_family.get(family_key)
                    if prev is None or pred_end_time > prev[1] + EPSILON:
                        latest_residual_by_family[family_key] = (
                            pred_name,
                            pred_end_time,
                            interval,
                        )
                    continue

            collapsed_context.append((pred_name, pred_end_time, interval))

        collapsed_context.extend(latest_residual_by_family.values())
        return collapsed_context

    def get_time_slots(
        self, subtask_name: str, constraints: DiGraph, direction: str
    ) -> List[TimeSlot]:
        """In / Out Edge의 Wrapper. TimeSlot 객체를 반환

        Args:
            subtask_name (str): 타임슬롯을 조회할 태스크 이름
            constraints (DiGraph): 제약 그래프
            direction (str): "in" 또는 "out"
        Returns:
            List[TimeSlot]: TimeSlot 객체를 반환
        """
        cache_key: Optional[TimeSlotCacheKey] = None
        if self._search_time_slot_cache is not None:
            cache_key = (id(constraints), subtask_name, direction)
            cached_slots = self._search_time_slot_cache.get(cache_key)
            if cached_slots is not None:
                self._search_cache_hits += 1
                return list(cached_slots)

        edges = (
            list(constraints.out_edges(subtask_name, data=True))
            if direction == "out"
            else list(constraints.in_edges(subtask_name, data=True))
        )
        if not edges:
            if cache_key is not None and self._search_time_slot_cache is not None:
                self._search_time_slot_cache[cache_key] = ()
                self._search_cache_misses += 1
            return []

        time_slots = []

        for u, v, data in edges:
            interval = data.get("info", {}).get("Interval", 0.0)
            is_crit = data.get("info", {}).get("IsCritical", False)
            linked_subtask = v if direction == "out" else u

            time_slots.append(TimeSlot(float(interval), is_crit, linked_subtask))
        if cache_key is not None and self._search_time_slot_cache is not None:
            self._search_time_slot_cache[cache_key] = tuple(time_slots)
            self._search_cache_misses += 1
        return time_slots

    def get_feasible_candidates(
        self,
        curr_node: SimulationNode,
    ) -> Tuple[List[Candidate], List[Candidate]]:
        """
        Determines feasible and not-yet-feasible candidates from the current state.
        Adjusts start times based on navigation estimates (from ActionHandler)
        and logical constraints, then checks against current time.
        """
        log.debug("[get_feasible_candidates] FEASIBLE CANDIDATES & NOT_YET CANDIDATES")
        feasible_candidates: List[Candidate] = []
        not_yet_candidates: List[Candidate] = []

        current_time = curr_node.state.current_time
        remaining_subtasks = curr_node.state.remaining_subtasks

        for sub in remaining_subtasks:
            # 1. 시간 제약 상 시작 가능한 시간 조건 확인
            (
                logical_interaction_start_time,
                is_critical,
                status,
                critical_info,
            ) = self.get_logical_interaction_start_time(curr_node, sub)

            critical_ctx = CriticalContext(
                source_subtask=critical_info.get("critical_source"),
                source_end_time=critical_info.get("critical_source_end_time"),
                interval=critical_info.get("critical_interval", 0.0),
                logical_start_time=critical_info.get("critical_start_time"),
            )

            # 선행 태스크 실패 또는 제약 조건 오류/충돌 시 후보에서 제외
            if status in ["FAILED_PREDECESSOR", "CONSTRAINT_ERROR", "CONFLICT"]:
                log.warning(
                    f"Subtask '{sub.name}' cannot be scheduled due to status: {status}. Skipping."
                )
                continue

            # 1. NOT_READY status 처리
            if (
                status == "NOT_READY"
            ):  # logical_start_time is None when status is NOT_READY

                # not_yet 후보에 필요한 정보 추가 (예: 예상 준비 시간 등)
                # 여기서는 Candidate 객체만 추가하고, 추후 scheduling_due 할당 등에서 활용 가능
                # NOT_READY 상태 Subtask는 최소 시작 시간이 None임
                candidate = Candidate(
                    subtask=sub,
                    is_critical=is_critical,
                    logical_interaction_start_time=logical_interaction_start_time,
                    critical_context=critical_ctx,
                )

                if (
                    candidate.is_critical
                    and candidate.scheduling_due.due_date == float("inf")
                    and critical_ctx.source_subtask
                    and critical_ctx.source_end_time is not None
                ):
                    inferred_due = critical_ctx.source_end_time + critical_ctx.interval
                    if inferred_due <= curr_node.state.current_time:
                        inferred_due = curr_node.state.current_time + EPSILON
                    candidate.scheduling_due = SchedulingDue(
                        due_date=inferred_due,
                        due_related_sub_name=candidate.subtask.name,
                    )
                log.debug(
                    f"Subtask '{sub.name}' is not yet feasible caused by predecessor task is not finished."
                )

                not_yet_candidates.append(candidate)  # status 추가
                continue

            # 2. 물리적 제약 조건 확인 (네비게이션 시간 등)
            first_nav_action = (
                sub.execution.primitive_actions[0]
                if sub.execution
                and sub.execution.primitive_actions[0].startswith("NAVIGATE_TO")
                else None
            )

            first_nav_duration = 0.0
            if first_nav_action:
                # curr_node의 현재 시간, 위치 등을 기준으로 첫 액션 실행 시간 추정
                navigation_info: Optional[ActionResult] = (
                    self.action_handler.get_actions_info(curr_node, [first_nav_action])
                )
                if navigation_info is None or not navigation_info.success:
                    log.warning(
                        "Navigation simulation failed for '%s'; falling back to zero prep time.",
                        sub.name,
                    )
                else:
                    first_nav_duration = navigation_info.action_duration

            # 3. 최종 시작 가능 시간 계산 및 Feasibility 판단
            #    logical_start_time: 선행 작업 완료 + 제약 간격 이후의 시간 (상호작용 시작 가능 논리적 시간)
            #    current_time + estimated_prep_duration: 현재부터 첫 액션 수행 후의 시간 (물리적 준비 완료 시간)

            # 실제 상호작용이 시작될 수 있는 가장 이른 시간
            # 가능한 상호작용 시간 보다, Logical Interaction Start Time이 더 늦는 경우 -> Logical Interaction Start Time으로 결정 -> 논리적 제약 충족 필요 -> not yet
            # Logical Interaction Start Time보다, 가능한 상호작용 시간이 같거나 늦는 경우 -> 가능한 상호작용 시간으로 결정 -> feasible candidates

            candidate = Candidate(
                subtask=sub,
                is_critical=is_critical,
                logical_interaction_start_time=logical_interaction_start_time,
                actual_interaction_start_time=max(
                    logical_interaction_start_time, current_time + first_nav_duration
                ),
                estimated_first_nav_duration=first_nav_duration,
                critical_context=critical_ctx,
            )

            # 현재 시간에 "상호작용을 시작"할 수 있는 경우만 feasible.
            # Monitoring horizon은 scheduler의 wait/urgent path에서 처리하고,
            # candidate 분류는 interaction-start readiness와 planner grace로만 결정한다.
            if logical_interaction_start_time <= (
                current_time + first_nav_duration + RISK_GRACE_SECONDS
            ):
                log.debug(
                    f"Subtask '{sub.name}' is feasible now (interaction can start at {candidate.actual_interaction_start_time:.2f}, current_time: {current_time:.2f}, first_nav_duration: {first_nav_duration:.2f})."
                )
                feasible_candidates.append(candidate)
            else:
                # 즉시 상호작용 시작은 불가능 (논리적 시간 미도래 또는 준비 시간 필요)
                log.debug(
                    f"Subtask '{sub.name}' is not yet feasible for immediate interaction (interaction can start at {candidate.actual_interaction_start_time:.2f})."
                )
                not_yet_candidates.append(candidate)

        self._assign_scheduling_due(feasible_candidates, not_yet_candidates, curr_node)

        return feasible_candidates, not_yet_candidates

    def get_logical_interaction_start_time(
        self, curr_node: SimulationNode, sub: Subtask
    ) -> tuple:
        """
        태스크 'sub'의 논리적 최소 시작 시간을 계산합니다.(시간 제약 상에서 최소 시작 시간)
        선행 태스크 완료 시간과 제약 시간 간격을 기반으로 합니다.
        선행 작업이 아직 완료되지 않은 경우, feasible_candidates/not_yet_candidates에서 예상 완료 시점을 추정하여 반환.
        반환: (logical_interaction_start_time(절대 시간계), is_critical, status)
        Status: "COMPLETED", "NOT_READY", "FAILED_PREDECESSOR", "CONSTRAINT_ERROR", "CONFLICT"
        """

        curr_constraints = curr_node.state.constraints
        # CONSTRAINT_ERROR: 제약 그래프가 없거나 사이클을 갖는 경우
        if (
            not curr_constraints
            or not isinstance(curr_constraints, nx.DiGraph)
            or not nx.is_directed_acyclic_graph(curr_constraints)
        ):

            return None, False, "CONSTRAINT_ERROR", {}

        # COMPLETED: 태스크가 아직 제약 그래프에 없는 경우 -> 동적으로 생성된 것으로 간주
        if sub.name not in curr_constraints:
            log.debug(
                f"Subtask '{sub.name}' not found in constraint graph. Assuming ready at time 0."
            )
            return (
                0.0,
                False,
                "COMPLETED",
                {
                    "critical_source": None,
                    "critical_source_end_time": None,
                    "critical_interval": 0.0,
                    "critical_start_time": 0.0,
                },
            )

        completed_subtasks_map = {
            ce.subtask.name: ce for ce in curr_node.state.completed_entries
        }
        # COMPLETED : Subtask에 시간 제약이 부재한 경우에는 언제든지 수행되도 되는 것
        in_edges = list(curr_constraints.in_edges(sub.name, data=True))

        # 선행 작업이 있는 Task에 대하여
        critical_times = []
        raw_critical_context = []
        non_critical_earliest_start = 0.0
        all_predecessors_finished = True
        any_predecessor_failed = False
        failure_reason = ""

        for pred_name, _, edge_data in in_edges:
            info = edge_data.get("info", {})
            interval = float(info.get("Interval", 0.0))
            is_crit = info.get("IsCritical", False)

            pred_entry: Optional[CompletedEntry] = completed_subtasks_map.get(
                pred_name, None
            )

            if pred_entry is None:
                all_predecessors_finished = False
                continue
            if self._has_failed_execution(pred_entry):
                any_predecessor_failed = True
                failure_reason = pred_name
                break

            pred_end_time = pred_entry.schedule_end_time
            curr_logical_interaction_start_time = pred_end_time + interval

            # Critical / Non-critical 분리
            if is_crit:
                log.debug(
                    f"current_subtask {sub.name} has critical constraint: {pred_name} -> {sub.name} = {curr_logical_interaction_start_time} ({pred_end_time=}, {interval=})"
                )
                raw_critical_context.append((pred_name, pred_end_time, interval, info))
            else:
                non_critical_earliest_start = max(
                    non_critical_earliest_start, curr_logical_interaction_start_time
                )

        critical_context = self._collapse_monitoring_residual_context(
            raw_critical_context
        )
        critical_times = [
            pred_end_time + interval
            for _, pred_end_time, interval in critical_context
        ]
        # 선행 작업 성공 / 실패 여부 확인
        if any_predecessor_failed:
            log.error(
                f"Final status for '{sub.name}': FAILED_PREDECESSOR ({failure_reason})"
            )
            return None, False, "FAILED_PREDECESSOR", {}

        if all_predecessors_finished:
            final_start_time = 0.0
            is_final_critical = False

            if critical_times:
                # 하나의 Subtask u,v pair 간 복수의 Critical 제약이 존재하는 경우,
                earliest_critical_time = min(critical_times)
                latest_critical_time = max(critical_times)
                if (
                    abs(earliest_critical_time - latest_critical_time)
                    > CONSTRAINT_MERGE_THRESHOLD
                ):
                    # Must satisfy all critical intervals: pick the latest (most restrictive) time
                    log.debug(
                        f"    [Constraint] Multi-start for '{sub.name}': candidates={sorted(critical_times)} -> resolved={latest_critical_time:.2f}"
                    )
                    final_start_time = latest_critical_time
                else:
                    final_start_time = earliest_critical_time

                is_final_critical = True
                if EPSILON < non_critical_earliest_start - final_start_time:
                    # Prefer the stricter (later) non-critical requirement without failing scheduling
                    log.debug(
                        f"    [Constraint] Tension for '{sub.name}': crit {final_start_time:.2f} < non-crit {non_critical_earliest_start:.2f}. Using non-critical."
                    )
                    final_start_time = non_critical_earliest_start
            else:
                final_start_time = non_critical_earliest_start
                is_final_critical = False

            if is_final_critical:
                if critical_context:
                    latest_ctx = max(
                        critical_context,
                        key=lambda ctx: ctx[1] + ctx[2],
                    )
                    pred_name, pred_end, interval = latest_ctx
                    critical_source = pred_name
                    critical_source_end_time = pred_end
                    critical_interval = interval
                else:
                    critical_source = None
                    critical_source_end_time = None
                    critical_interval = 0.0
            else:
                critical_source = None
                critical_source_end_time = None
                critical_interval = 0.0

            collected_critical_info = {
                "critical_source": critical_source,
                "critical_source_end_time": critical_source_end_time,
                "critical_interval": critical_interval,
                "critical_start_time": (
                    non_critical_earliest_start
                    if non_critical_earliest_start > final_start_time
                    else final_start_time
                ),
            }

            return (
                final_start_time,
                is_final_critical,
                "COMPLETED",
                collected_critical_info,
            )

        # If not all predecessors finished, we still want to return critical context if possible
        # to allow urgency checking for blocked tasks.
        final_start_time = 0.0
        if critical_times:
            final_start_time = max(critical_times)

        # Calculate critical info similarly
        critical_source = None
        critical_source_end_time = None
        critical_interval = 0.0

        if critical_context:
            latest_ctx = max(
                critical_context,
                key=lambda ctx: ctx[1] + ctx[2],
            )
            pred_name, pred_end, interval = latest_ctx
            critical_source = pred_name
            critical_source_end_time = pred_end
            critical_interval = interval

        collected_critical_info = {
            "critical_source": critical_source,
            "critical_source_end_time": critical_source_end_time,
            "critical_interval": critical_interval,
            "critical_start_time": final_start_time,
        }

        # [Fix] Determine criticality for NOT_READY tasks
        # If any incoming edge is critical, the task is effectively critical (part of a critical chain)
        # even if we can't calculate the exact start time yet.
        is_critical_end_subtask = False
        for crit_start_sub_name, crit_start_sub_name, edge_data in in_edges:
            if edge_data.get("info", {}).get("IsCritical", False):
                is_critical_end_subtask = True
                break

        return (
            None,
            is_critical_end_subtask,
            "NOT_READY",
            collected_critical_info,
        )

    def _assign_scheduling_due(
        self,
        feasible_candidates: List[Candidate],
        not_yet_candidates: List[Candidate],
        curr_node: SimulationNode,
    ) -> None:
        """
        Assigns scheduling due to feasible candidates based on the logical earliest start time
        of the next upcoming critical task found in the not_yet list.
        Modifies the feasible list IN-PLACE.
        """

        # Find the earliest logical start time among upcoming critical tasks
        valid_crit_candidates = [
            c
            for c in not_yet_candidates
            if c.is_critical
            and c.logical_interaction_start_time is not None
            and c.logical_interaction_start_time >= curr_node.state.current_time
            and c.logical_interaction_start_time != float("inf")
        ]
        if valid_crit_candidates:
            # Sort by logical start time to find the *next* critical scheduling_due
            valid_crit_candidates.sort(key=lambda x: x.logical_interaction_start_time)
            next_crit = valid_crit_candidates[0]

            # Assign the calculated scheduling_due (IN-PLACE) while preserving critical metadata.
            new_scheduling_due = SchedulingDue(
                due_date=next_crit.logical_interaction_start_time,
                due_related_sub_name=next_crit.subtask.name,
            )
        else:
            # 남은 Critical Task가 없는 경우: 무한대(inf) 또는 적절한 기본값 할당
            new_scheduling_due = SchedulingDue(
                due_date=float("inf"),
                due_related_sub_name=None,
            )

        for feasible_candidate in feasible_candidates:
            own_due = feasible_candidate.logical_interaction_start_time
            if (
                feasible_candidate.is_critical
                and own_due is not None
                and own_due != float("inf")
            ):
                feasible_candidate.scheduling_due = SchedulingDue(
                    due_date=float(own_due),
                    due_related_sub_name=feasible_candidate.subtask.name,
                )
            else:
                feasible_candidate.scheduling_due = new_scheduling_due

    def get_required_predecessors(
        self,
        target_subtask_name: str,
        remaining_subtasks: List[Subtask],
        constraints: DiGraph,
    ) -> List[Subtask]:
        """
        Returns a list of remaining subtasks that are ancestors (dependencies)
        of the target_subtask_name in the constraint graph.

        Args:
            target_subtask_name: The name of the target critical subtask.
            remaining_subtasks: List of currently remaining subtasks.
            constraints: The dependency graph.

        Returns:
            List[Subtask]: A list of predecessor subtasks that must be completed
                           before the target subtask can start.
        """
        if not constraints.has_node(target_subtask_name):
            return []

        ancestors = nx.ancestors(constraints, target_subtask_name)
        required_predecessors = [
            sub for sub in remaining_subtasks if sub.name in ancestors
        ]
        return required_predecessors
