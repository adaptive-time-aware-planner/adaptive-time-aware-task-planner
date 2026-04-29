from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, TypeAlias

from src.core.monitoring import (
    BeliefStore,
    MonitoringPolicy,
    MonitoringTriggerContext,
    create_monitoring_policy,
)
from src.models.dataclass import (
    ActionResult,
    Candidate,
    CompletedEntry,
    SchedulerState,
    SchedulingDue,
    SimulationNode,
)
from src.models.task import Duration, Execution, Subtask
from src.utils.common import create_module_logger, extract_monitoring_target_name
from src.utils.common.decorators import time_logger
from src.utils.config import EPSILON, MONITORING_DURATION, RED, RESET, constants
from src.utils.config.constants import BEAM_WIDTH, INIT_PRIOR_VARIANCE, SIMULATION_DEPTH
from src.utils.task import TaskUtil

if TYPE_CHECKING:
    from src.scheduler import ActionHandler, ConstraintHandler, HeuristicManager

log = create_module_logger(module_name=__name__, module_log=True)
ActiveIntervalCacheKey: TypeAlias = Tuple[int, Tuple[Tuple[str, float], ...]]
MonitoringTriggerCacheKey: TypeAlias = Tuple[str, Optional[str], float, float, float]
ActionInfoCache: TypeAlias = Dict[object, Optional[ActionResult]]
TimeSlotCache: TypeAlias = Dict[object, Tuple[object, ...]]


@dataclass
class SchedulerSearchCache:
    """Caches repeated calculations during a single scheduler search call."""

    action_results: ActionInfoCache = field(default_factory=dict)
    time_slots: TimeSlotCache = field(default_factory=dict)
    active_intervals: Dict[
        ActiveIntervalCacheKey, List[Tuple[float, SchedulingDue]]
    ] = field(default_factory=dict)
    monitoring_triggers: Dict[MonitoringTriggerCacheKey, float] = field(
        default_factory=dict
    )
    action_cache_hits: int = 0
    action_cache_misses: int = 0
    time_slot_cache_hits: int = 0
    time_slot_cache_misses: int = 0
    active_interval_cache_hits: int = 0
    active_interval_cache_misses: int = 0
    trigger_cache_hits: int = 0
    trigger_cache_misses: int = 0


@dataclass(frozen=True)
class MonitoringObligation:
    """Resolved active monitoring obligation for one critical interval."""

    trigger_time: float
    variance: float
    due: SchedulingDue


@dataclass(frozen=True)
class WaitExpansionEvent:
    """One event that a blocked-frontier wait expansion may advance to."""

    kind: str
    candidate: Candidate
    event_time: float
    nav_duration: float
    raw_trigger_time: float = float("inf")
    variance: float = float("-inf")
    due_date: float = float("inf")
    obligation: Optional[MonitoringObligation] = None


class Scheduler:
    """
    Beam Search based Scheduler with n-step lookahead.
    Given a current state, it attempts to find the best next subtask to execute
    by simulating expansions of feasible (or soon-to-be-feasible) subtasks.

    Attributes:
        search_width (int): Beam width (number of top expansions to keep).
        simulation_depth (int): Maximum search depth for lookahead.
        nav_graph (dict): Navigation graph for path planning.

        action_handler (ActionHandler): Handles action duration calculations.
        constraint_handler (ConstraintHandler): Checks subtask feasibility.
        cost_calculator (HeuristicManager): Calculates heuristic cost of expansions.
        _counter (itertools.count): A counter to break ties in the priority queue.
    """

    def __init__(
        self,
        action_handler: ActionHandler,
        constraint_handler: ConstraintHandler,
        heuristic_manager: HeuristicManager,
        monitoring_policy: Optional[MonitoringPolicy] = None,
        beam_width: int = BEAM_WIDTH,
        simulation_depth: int = SIMULATION_DEPTH,
        max_monitoring_per_critical_interval: int | None = None,
        real_world_mode: bool = False,
    ) -> None:
        """Initialize the scheduler and its expansion policies.

        Args:
            action_handler: Simulates primitive action durations.
            constraint_handler: Evaluates temporal feasibility.
            heuristic_manager: Scores expanded nodes.
            monitoring_policy: Optional backend-specific monitoring trigger policy.
            beam_width: Layer-wise beam width.
            simulation_depth: Lookahead depth for beam search.
            max_monitoring_per_critical_interval: Optional cap for how many
                monitoring actions may execute within one critical interval.
                Negative values represent an explicit unbounded budget.
        """

        self.search_width = beam_width
        self.simulation_depth = simulation_depth
        log.info(
            f"{RED}[Scheduler Init] search_width={beam_width}, simulation_depth={simulation_depth}{RESET}"
        )
        self.constraint_handler = constraint_handler
        self.action_handler = action_handler
        self.cost_calculator = heuristic_manager
        self.real_world_mode = bool(real_world_mode)
        self.monitoring_policy = monitoring_policy or create_monitoring_policy(
            "bayesian",
            BeliefStore(),
        )
        self.max_monitoring_per_critical_interval = max_monitoring_per_critical_interval
        self._counter = itertools.count()
        self._search_cache: Optional[SchedulerSearchCache] = None

    def _begin_search_session(self) -> None:
        """Create and attach search-scoped caches for the current beam search."""

        self._search_cache = SchedulerSearchCache()
        if hasattr(self.action_handler, "begin_search_session"):
            self.action_handler.begin_search_session(self._search_cache.action_results)
        if hasattr(self.constraint_handler, "begin_search_session"):
            self.constraint_handler.begin_search_session(self._search_cache.time_slots)

    def _end_search_session(self) -> None:
        """Detach search-scoped caches and emit debug statistics."""

        if self._search_cache is None:
            return

        action_hits, action_misses = (0, 0)
        time_slot_hits, time_slot_misses = (0, 0)
        if hasattr(self.action_handler, "end_search_session"):
            action_hits, action_misses = self.action_handler.end_search_session()
        if hasattr(self.constraint_handler, "end_search_session"):
            time_slot_hits, time_slot_misses = (
                self.constraint_handler.end_search_session()
            )
        self._search_cache.action_cache_hits = action_hits
        self._search_cache.action_cache_misses = action_misses
        self._search_cache.time_slot_cache_hits = time_slot_hits
        self._search_cache.time_slot_cache_misses = time_slot_misses
        log.debug(
            "[Scheduler Cache] action hits=%d misses=%d, time_slot hits=%d misses=%d, "
            "active_interval hits=%d misses=%d, trigger hits=%d misses=%d",
            self._search_cache.action_cache_hits,
            self._search_cache.action_cache_misses,
            self._search_cache.time_slot_cache_hits,
            self._search_cache.time_slot_cache_misses,
            self._search_cache.active_interval_cache_hits,
            self._search_cache.active_interval_cache_misses,
            self._search_cache.trigger_cache_hits,
            self._search_cache.trigger_cache_misses,
        )
        self._search_cache = None

    @staticmethod
    def _build_completed_entries_signature(
        completed_entries: List[CompletedEntry],
    ) -> Tuple[Tuple[str, float], ...]:
        """Create a hashable signature for completed entries relevant to monitoring."""

        return tuple(
            (entry.subtask.name, entry.schedule_end_time) for entry in completed_entries
        )

    def _has_real_world_late_due_signal(
        self,
        curr_node: SimulationNode,
        target_subtask_name: str,
    ) -> bool:
        """Return whether the latest ROS monitoring already marked a target due/late."""

        if not self.real_world_mode:
            return False

        completed_names = {
            entry.subtask.name for entry in curr_node.state.completed_entries
        }

        for pred_name, _end_name, data in curr_node.state.constraints.in_edges(
            target_subtask_name, data=True
        ):
            info = data.get("info", {})
            if not info.get("IsCritical", False):
                continue
            if not pred_name.startswith("Monitoring for"):
                continue
            if pred_name not in completed_names:
                continue
            if bool(info.get("LateObservation", False)):
                return True
            if info.get("IsMonitoringResidual", False) and float(
                info.get("Interval", float("inf"))
            ) <= EPSILON:
                return True
        return False

    def _get_active_monitoring_intervals(
        self, curr_node: SimulationNode
    ) -> List[Tuple[float, SchedulingDue]]:
        """Return active critical intervals for the current node with memoization.

        Args:
            curr_node: Node whose completed tasks and constraints define active intervals.

        Returns:
            Active intervals as `(variance, due)` tuples.
        """

        graph = curr_node.state.constraints
        cache_key = (
            id(graph),
            self._build_completed_entries_signature(curr_node.state.completed_entries),
        )
        if self._search_cache is not None:
            cached_intervals = self._search_cache.active_intervals.get(cache_key)
            if cached_intervals is not None:
                self._search_cache.active_interval_cache_hits += 1
                return cached_intervals

        completed_entries_map = {
            ce.subtask.name: ce for ce in curr_node.state.completed_entries
        }
        active_intervals: List[Tuple[float, SchedulingDue]] = []
        for start_name, end_name, data in graph.edges(data=True):
            info = data.get("info", {})
            if info.get("IsCritical") and info.get("Interval") > EPSILON:
                if (
                    start_name in completed_entries_map
                    and end_name not in completed_entries_map
                ):
                    start_entry = completed_entries_map[start_name]
                    if start_entry.subtask.subtask_type != "Monitor":
                        interval = info.get("Interval", 0.0)
                        variance = info.get("Variance", INIT_PRIOR_VARIANCE)
                        due_date = start_entry.schedule_end_time + interval
                        active_intervals.append(
                            (
                                variance,
                                SchedulingDue(
                                    due_date=due_date, due_related_sub_name=end_name
                                ),
                            )
                        )

        if self._search_cache is not None:
            self._search_cache.active_interval_cache_misses += 1
            self._search_cache.active_intervals[cache_key] = active_intervals
        return active_intervals

    # ======================
    # Public method
    # ======================
    @time_logger
    def get_next_state(self, parent_state: SchedulerState) -> Optional[SchedulerState]:
        """
        Public method to retrieve the immediate next state (1-step ahead in time)
        from the given parent_state.

        Args:
            parent_state (SchedulerState): The current scheduling state.

        Returns:
            Optional[SchedulerState]: The next state after scheduling one subtask,
            or None if no feasible solution is found.
        """
        child_node = self._simulate_search(parent_state)
        if child_node is None:
            log.error("[get_next_state] No child_state found => No feasible solution.")
            return None

        new_state = self._extract_state(child_node)
        if new_state is None:
            log.error("[get_next_state] child_state found, but state is None.")
            return None

        log.debug(
            f"[get_next_state] => subtask={new_state.subtask.name}, "
            f"time={round(new_state.current_time,2)}"
        )
        return new_state

    # ======================
    # Core beam search
    # ======================
    def _simulate_search(self, init_state: SchedulerState) -> Optional[SimulationNode]:
        """
        Conducts a Layer-wise Beam Search up to self.simulation_depth.
        At each depth, we keep only the top `search_width` nodes (Layer-wise Pruning).

        Args:
            init_state (SchedulerState): The root state to start the simulation.

        Returns:
            Optional[SimulationNode]: The best goal node (lowest cost) among expansions.
        """
        self._begin_search_session()
        try:
            # 1. Initialize Beam
            init_node = SimulationNode(
                parent_node=None,
                heuristic_cost=0.0,
                depth=0,
                tie_breaker=next(self._counter),
                state=init_state,
                risk_level=0,
            )
            current_beam: List[SimulationNode] = [init_node]
            best_solutions: List[SimulationNode] = []

            log.debug(
                f"[_simulate_search] Starting Layer-wise Beam Search (Width={self.search_width}, Depth={self.simulation_depth})"
            )

            # 2. Layer-wise Loop
            for d in range(self.simulation_depth):
                if not current_beam:
                    log.debug(
                        f"[_simulate_search] Beam is empty at depth {d}. Stopping."
                    )
                    break

                next_beam_candidates: List[SimulationNode] = []

                # Expand all nodes in the current beam
                for curr_node in current_beam:
                    curr_state = curr_node.state

                    # (A) Check Termination (All tasks done?)
                    if not curr_state.remaining_subtasks:
                        log.debug(
                            f"  [Solution Found] Node depth {curr_node.depth} finished all tasks."
                        )
                        best_solutions.append(curr_node)
                        continue
                    log.debug(
                        f"\n=== [Beam Step] Depth {d} -> {d+1} | Beam Size: {len(current_beam)} | Current Time: {curr_state.current_time:.2f} ==="
                    )
                    # (B) Get Feasible Candidates
                    feasible_candidates, not_yet_candidates = (
                        self.constraint_handler.get_feasible_candidates(curr_node)
                    )

                    if not feasible_candidates and not not_yet_candidates:
                        # Dead end
                        continue

                    # (C) Expand
                    log.debug(
                        f"\n  • Completed : {[ce.subtask.name for ce in curr_state.completed_entries]}\n"
                        f"  • Remaining : {[r.name for r in curr_state.remaining_subtasks]}\n"
                        f"  • Feasible  : {[c.subtask.name for c in feasible_candidates]}\n"
                        f"  • Not Yet   : {[c.subtask.name for c in not_yet_candidates]}\n"
                        f"============================================================"
                    )
                    expanded_nodes = self._expand_candidates(
                        curr_node, feasible_candidates, not_yet_candidates
                    )

                    # (D) Collect Valid Expansions
                    # [Modified 251229] Risk 2 이상이라도, 대안이 없으면 채택하기 위해 별도로 수집합니다.
                    valid_expansions = []
                    high_risk_expansions = []

                    for nd in expanded_nodes:
                        if nd.risk_level < 2:
                            valid_expansions.append(nd)
                        else:
                            high_risk_expansions.append(nd)

                    if valid_expansions:
                        next_beam_candidates.extend(valid_expansions)
                    elif high_risk_expansions:
                        # 정상적인(Risk < 2) 후보가 하나도 없을 때만 Risk 높은 후보를 고려합니다.
                        log.warning(
                            f"[_simulate_search] Depth {d}: No valid expansions found. "
                            f"Fallback to {len(high_risk_expansions)} High-Risk candidates."
                        )
                        next_beam_candidates.extend(high_risk_expansions)

                # 3. Pruning (Layer-wise)
                if not next_beam_candidates:
                    log.debug(
                        f"[_simulate_search] No valid expansions at depth {d} (including high-risk)."
                    )
                    break

                # Sort by (Risk, Cost) to pick top K
                # Risk is primary, Cost (Heuristic) is secondary.
                next_beam_candidates.sort(
                    key=lambda nd: (nd.risk_level, nd.heuristic_cost)
                )

                # Keep top K
                current_beam = next_beam_candidates[: self.search_width]
                log.debug(
                    "============================================================"
                )
                log.debug(f"Layer-wise Beam Search: Depth {d} -> {d+1}")
                # Log the survivors
                log.debug(
                    f"  -> Pruning: Kept top {len(current_beam)} out of {len(next_beam_candidates)} candidates."
                )
                for i, node in enumerate(current_beam):
                    # Trace back path
                    path_names = []
                    curr = node
                    while curr:
                        if curr.state and curr.state.subtask:
                            path_names.append(curr.state.subtask.name)
                        curr = curr.parent_node
                    path_str = " -> ".join(reversed(path_names))

                    log.debug(
                        f"     [{i+1}] {path_str} (Risk={node.risk_level}, Cost={node.heuristic_cost:.2f})"
                    )
                log.debug(
                    "============================================================"
                )

            # 4. Final Collection
            # Add nodes that reached max depth but still have tasks remaining
            if current_beam:
                best_solutions.extend(current_beam)

            if not best_solutions:
                log.error("[_simulate_search] best_solutions empty => no feasible path")
                return None

            def get_depth1_cost(node: SimulationNode) -> float:
                """Backtrack to find the heuristic cost at Depth 1."""

                curr = node
                while curr.depth > 1:
                    if curr.parent_node:
                        curr = curr.parent_node
                    else:
                        break
                return curr.heuristic_cost

            depth1_cost_by_node = {
                id(node): get_depth1_cost(node) for node in best_solutions
            }

            # 5. Select Winner
            # [Modified] Sort by Risk Level first, then Leaf Node Cost (Make-span).
            # We prefer the path that finishes all tasks earliest (Lowest total cost).
            # Depth 1 Cost can be misleading if an action (like Wait) has low immediate cost
            # but delays the overall schedule.
            best_solutions.sort(
                key=lambda nd: (
                    nd.risk_level,
                    nd.heuristic_cost + depth1_cost_by_node[id(nd)],
                )
            )
            log.debug(
                f"best_solutions: {best_solutions[0].state.subtask.name}, {best_solutions[0].heuristic_cost} {depth1_cost_by_node[id(best_solutions[0])]}"
            )

            winner = best_solutions[0]

            # Trace back to find the immediate next action (Depth 1)
            immediate_node = winner
            while immediate_node.depth > 1:
                if immediate_node.parent_node:
                    immediate_node = immediate_node.parent_node
                else:
                    break

            log.debug(
                f"\n[_simulate_search] Final Decision: Path selected (Leaf: '{winner.state.subtask.name}').\n"
                f"  -> Immediate Action: '{immediate_node.state.subtask.name}'\n"
                f"  (Risk={winner.risk_level}, Depth1_Cost={depth1_cost_by_node[id(winner)]:.2f}, Leaf_Cost={winner.heuristic_cost:.2f}, Depth={winner.depth})"
            )
            return winner
        finally:
            self._end_search_session()

    def _extract_state(
        self, child_node: Optional[SimulationNode]
    ) -> Optional[SchedulerState]:
        """
        Traces from a terminal node (child_node) back to the root (init_state),
        and returns the **state at depth=1** in that path. This effectively
        picks the next immediate step in the best path found.

        Args:
            child_node (SimulationNode): The best solution node from the beam search.

        Returns:
            Optional[SchedulerState]: The state corresponding to the next step
            (depth=1). If only the root is found, returns the root state.
        """
        if child_node is None:
            log.error("[_extract_state] child_node is None")
            return None

        path = self._trace_solution_path(child_node)

        # If only the root (depth=0) is present
        if len(path) < 2:
            log.debug("[_extract_state] Only root node in path. No feasible next step.")
            return None

        # Return the state at the first step beyond root (depth=1)
        log.debug("[_extract_state] Returning state at depth=1 in the best path.")
        return path[1].state

    def _trace_solution_path(self, child_node: SimulationNode) -> List[SimulationNode]:
        """Trace a winner leaf back to the root and return the forward path."""

        path: List[SimulationNode] = []
        current_node: Optional[SimulationNode] = child_node
        while current_node:
            path.append(current_node)
            current_node = current_node.parent_node
        path.reverse()
        return path

    def _expand_candidates(
        self,
        curr_node: SimulationNode,
        feasible_candidates: List[Candidate],
        not_yet_candidates: List[Candidate],
    ) -> List[SimulationNode]:
        """
        Expands candidates based on a unified policy to prioritize critical tasks.

        Policy Hierarchy:
        1. Urgent Critical Tasks (Unified):
           - If any critical task is ready to start (within tolerance) or overdue,
             expand ONLY these tasks. This combines On-time, Closing, and Missed policies.
        2. Standard Expansion:
           - If no urgent tasks, expand all feasible tasks and valid 'WAIT' options.
        """
        expansions: List[SimulationNode] = []

        forced_monitor_candidate = self._get_forced_followup_monitoring_candidate(
            curr_node,
            feasible_candidates,
        )
        if forced_monitor_candidate is not None:
            log.debug(
                "Policy 0 (Committed Follow-up): Expanding only forced monitoring '%s' after committed wait '%s'.",
                forced_monitor_candidate.subtask.name,
                curr_node.state.subtask.name if curr_node.state.subtask else "N/A",
            )
            child_node = self._expand_single_subtask(
                curr_node,
                forced_monitor_candidate,
                not_yet_candidates,
                feasible_candidates,
            )
            return [child_node] if child_node else []

        # --- Policy 1 (Unified): Urgent Critical Tasks ---
        urgent_candidates = self._get_urgent_critical_candidates(
            curr_node, feasible_candidates, not_yet_candidates
        )

        if urgent_candidates:
            log.debug(
                f"Policy 1 (Urgent): Expanding {len(urgent_candidates)} urgent candidate(s)."
            )
            for candidate in urgent_candidates:
                # [Added 250202] Conflict-Avoidance Wait for Urgent Tasks
                # Check if immediate execution causes future conflicts.
                # Even for urgent tasks, if execution leads to a future deadline violation,
                # we should consider waiting (which might violate the current urgent interval,
                # but allows the scheduler to weigh the costs).
                conflict_delay, _ = self.cost_calculator.check_future_conflict(
                    curr_node, candidate
                )

                if conflict_delay > constants.EPSILON:
                    nav_time = self._estimate_candidate_navigation_duration(
                        curr_node, candidate
                    )
                    wait_node = self._expand_wait_wo_monitoring(
                        curr_node,
                        candidate,
                        not_yet_candidates,
                        nav_duration=nav_time,
                        feasible_candidates=feasible_candidates,
                        additional_delay=conflict_delay,
                    )

                    if wait_node:
                        log.debug(
                            f"[Conflict-Avoidance] Generated Wait Node for URGENT {candidate.subtask.name} "
                            f"(Delay: {conflict_delay:.2f}s) to avoid future conflict."
                        )
                        expansions.append(wait_node)

                child_node = self._expand_single_subtask(
                    curr_node, candidate, not_yet_candidates, feasible_candidates
                )
                if child_node:
                    expansions.append(child_node)

            if expansions:
                return expansions
            else:
                log.debug(
                    "All urgent candidates failed to expand. Falling back to standard expansion."
                )

        # --- Policy 2: Standard Expansion (other feasible + all waits) ---
        log.debug("Policy 2: No urgent criticals. Performing standard expansion.")
        for candidate in feasible_candidates:

            # 2. [Added 250130] Conflict-Avoidance Wait
            # Check if immediate execution causes future conflicts.
            # If so, generate an alternative 'Wait' node that delays execution just enough to avoid the conflict.
            conflict_delay, _ = self.cost_calculator.check_future_conflict(
                curr_node, candidate
            )

            if conflict_delay > constants.EPSILON:
                nav_time = self._estimate_candidate_navigation_duration(
                    curr_node, candidate
                )
                wait_node = self._expand_wait_wo_monitoring(
                    curr_node,
                    candidate,
                    not_yet_candidates,
                    nav_duration=nav_time,
                    feasible_candidates=feasible_candidates,
                    additional_delay=conflict_delay,
                )

                if wait_node:
                    log.debug(
                        f"[Conflict-Avoidance] Generated Wait Node for {candidate.subtask.name} "
                        f"(Delay: {conflict_delay:.2f}s) to avoid future conflict."
                    )
                    expansions.append(wait_node)

            # 1. Expand Action (Immediate Execution)
            child_node = self._expand_single_subtask(
                curr_node, candidate, not_yet_candidates, feasible_candidates
            )
            if child_node:
                expansions.append(child_node)

        if not_yet_candidates:
            blocked_frontier = self._get_blocked_candidate_frontier(not_yet_candidates)
            for blocked_candidate in blocked_frontier:
                log.debug(
                    "[_expand_candidates] blocked frontier candidate: %s",
                    blocked_candidate,
                )
                wait_nodes = self._expand_wait_options(
                    curr_node,
                    blocked_candidate,
                    not_yet_candidates,
                    feasible_candidates=feasible_candidates,
                )
                expansions.extend(wait_nodes)

        return expansions

    def _get_urgent_critical_candidates(
        self,
        curr_node: SimulationNode,
        feasible_candidates: List[Candidate],
        not_yet_candidates: List[Candidate],
    ) -> List[Candidate]:
        """
        Identifies critical candidates that are 'Urgent'.
        Urgent means the task has entered the same near-deadline window as the
        monitoring horizon. Exact interaction-start readiness is still checked
        separately so we do not dispatch a critical task too early.
        Also checks not_yet_candidates for urgent but blocked tasks, prioritizing their feasible predecessors.
        """

        urgent_list = []
        for candidate in feasible_candidates:
            if not candidate.is_critical:
                continue

            physical_earliest_start = (
                curr_node.state.current_time + candidate.estimated_first_nav_duration
            )

            if self._has_real_world_late_due_signal(curr_node, candidate.subtask.name):
                candidate.actual_interaction_start_time = physical_earliest_start
                log.debug(
                    "Found ROS due-now critical candidate '%s' from late observation "
                    "(Physical: %.2f).",
                    candidate.subtask.name,
                    physical_earliest_start,
                )
                urgent_list.append(candidate)
                continue

            if candidate.logical_interaction_start_time is None:
                continue

            log.debug(
                f"[_get_urgent_critical_candidates] diff: {candidate.logical_interaction_start_time - physical_earliest_start}"
            )

            # Check urgency inside the monitoring horizon. Actual dispatch still
            # needs to satisfy the interaction-start readiness grace.
            log.debug(
                f"[_get_urgent_critical_candidates] candidate.logical_interaction_start_time: {candidate.logical_interaction_start_time}, physical_earliest_start: {physical_earliest_start}"
            )

            if (candidate.logical_interaction_start_time - physical_earliest_start) <= (
                MONITORING_DURATION + EPSILON
            ):
                if not self._can_candidate_start_interaction_now(curr_node, candidate):
                    log.debug(
                        "Found horizon-urgent critical candidate '%s', but interaction is still early "
                        "(Physical: %.2f, Logical: %.2f, Grace: %.2f).",
                        candidate.subtask.name,
                        physical_earliest_start,
                        candidate.logical_interaction_start_time,
                        constants.RISK_GRACE_SECONDS,
                    )
                    continue
                slack_to_logical = (
                    candidate.logical_interaction_start_time - physical_earliest_start
                )
                if slack_to_logical > 0:
                    log.debug(
                        f"[_get_urgent_critical_candidates] grace-dispatch inside monitoring horizon: {slack_to_logical}"
                    )
                else:
                    log.debug(
                        f"[_get_urgent_critical_candidates] 늦어서 작업을 함: {slack_to_logical}"
                    )
                # Update actual interaction start time
                # We start as soon as physically possible (ASAP)
                candidate.actual_interaction_start_time = physical_earliest_start

                log.debug(
                    f"Found URGENT CRITICAL candidate: {candidate.subtask.name} "
                    f"(Physical: {physical_earliest_start:.2f}, Logical: {candidate.logical_interaction_start_time:.2f}, Horizon: {MONITORING_DURATION:.2f})"
                )
                urgent_list.append(candidate)

        # [Added 251215] Check blocked urgent tasks and prioritize predecessors
        feasible_map = {c.subtask.name: c for c in feasible_candidates}
        constraints = curr_node.state.constraints
        visited = set()

        def find_feasible_ancestor(target_name: str) -> None:
            """Recursively trace predecessors to find feasible ancestors."""
            if target_name in visited:
                return
            visited.add(target_name)

            if not constraints.has_node(target_name):
                return

            preds = list(constraints.predecessors(target_name))
            for pred_name in preds:
                if pred_name in feasible_map:
                    pred_cand = feasible_map[pred_name]
                    if pred_cand not in urgent_list:
                        log.debug(
                            f"Prioritizing feasible ancestor '{pred_name}' to unblock urgent task chain targeting '{candidate.subtask.name}'."
                        )
                        urgent_list.append(pred_cand)
                else:
                    # Recursive search for ancestors
                    find_feasible_ancestor(pred_name)

        for candidate in not_yet_candidates:
            # [TODO] candidate가 logical start time이 없으면 무시해야 함
            if (
                not candidate.is_critical
                or candidate.subtask.decomposed
                or candidate.logical_interaction_start_time is None
            ):
                continue

            physical_start = (
                curr_node.state.current_time
                + candidate.estimated_first_nav_duration
            )

            if self._has_real_world_late_due_signal(curr_node, candidate.subtask.name):
                log.debug(
                    "Found ROS due-now blocked candidate '%s' from late observation "
                    "(Physical: %.2f, Planner Logical: %.2f).",
                    candidate.subtask.name,
                    physical_start,
                    candidate.logical_interaction_start_time,
                )
                candidate.actual_interaction_start_time = physical_start
                urgent_list.append(candidate)
                continue

            logical_start = None
            if self.real_world_mode:
                logical_start = candidate.logical_interaction_start_time
            else:
                incoming_slots = self.constraint_handler.get_time_slots(
                    candidate.subtask.name, curr_node.state.constraints, "in"
                )
                critical_slots = [s for s in incoming_slots if s.is_critical]

                if critical_slots:
                    target_slot = max(critical_slots, key=lambda s: s.interval)
                    critical_start_sub_name = target_slot.related_subtask_name
                    critical_interval_duration = target_slot.interval

                    pred_entry = next(
                        (
                            ce
                            for ce in curr_node.state.completed_entries
                            if ce.subtask.name == critical_start_sub_name
                        ),
                        None,
                    )
                    if pred_entry:
                        logical_start = (
                            pred_entry.schedule_end_time + critical_interval_duration
                        )

            if logical_start is not None and (logical_start - physical_start) <= (
                MONITORING_DURATION + EPSILON
            ):
                # Urgent but blocked! Find feasible predecessors recursively.
                log.debug(
                    f"Found BLOCKED URGENT task: {candidate.subtask.name} "
                    f"(Physical: {physical_start:.2f}, Logical: {logical_start:.2f}, Horizon: {MONITORING_DURATION:.2f}). Tracing ancestors."
                )

                len_before = len(urgent_list)
                find_feasible_ancestor(candidate.subtask.name)

                if len(urgent_list) == len_before:
                    if self._can_candidate_start_interaction_now(
                        curr_node, candidate
                    ):
                        log.debug(
                            f"No feasible ancestors found for {candidate.subtask.name}. "
                            f"Adding the task itself as it is time-ready/urgent."
                        )
                        candidate.actual_interaction_start_time = physical_start
                        urgent_list.append(candidate)
                    else:
                        log.debug(
                            "Blocked urgent task '%s' is inside the monitoring horizon but still early "
                            "(Physical: %.2f, Logical: %.2f, Grace: %.2f). Leaving it to blocked staged-wait expansion.",
                            candidate.subtask.name,
                            physical_start,
                            logical_start,
                            constants.RISK_GRACE_SECONDS,
                        )

        return urgent_list

    def _extract_monitoring_target(self, candidate: Candidate) -> Optional[str]:
        return self._extract_first_actionable_object_id(candidate.subtask)

    @staticmethod
    def _extract_first_actionable_object_id(
        subtask: Optional[Subtask],
    ) -> Optional[str]:
        """Return the first non-WAIT object id mentioned by a subtask's actions."""

        if subtask is None or subtask.execution is None:
            return None

        primitive_actions = subtask.execution.primitive_actions or []
        for action in primitive_actions:
            parts = str(action).split()
            if len(parts) <= 1:
                continue
            if parts[0].upper() == "WAIT":
                continue
            return parts[1]
        return None

    def _get_monitoring_target_obj_for_subtask_name(
        self,
        curr_node: SimulationNode,
        subtask_name: str,
    ) -> Optional[str]:
        """Return the object id associated with the named critical-end subtask.

        Searches both remaining_subtasks and the current candidate subtask so
        that a task being waited on (already promoted out of remaining) is
        still resolvable.
        """

        current_subtask = curr_node.state.subtask
        candidates_to_search = list(curr_node.state.remaining_subtasks)
        if current_subtask is not None:
            candidates_to_search.append(current_subtask)

        return next(
            (
                target_obj_id
                for sub in candidates_to_search
                for target_obj_id in [
                    self._extract_first_actionable_object_id(sub)
                ]
                if sub.name == subtask_name
                and target_obj_id is not None
            ),
            None,
        )

    @staticmethod
    def _extract_monitoring_target_object_from_subtask(
        subtask: Optional[Subtask],
    ) -> Optional[str]:
        """Return the monitored object id recorded on a monitoring subtask."""

        if subtask is None or subtask.execution is None:
            return None

        objects = subtask.execution.objects
        if isinstance(objects, (list, tuple)) and objects:
            first_object = objects[0]
            if first_object is not None:
                return str(first_object)

        primitive_actions = subtask.execution.primitive_actions or []
        if not primitive_actions:
            return None

        return Scheduler._extract_first_actionable_object_id(subtask)

    @staticmethod
    def _extract_monitoring_target_name_from_subtask(
        subtask: Optional[Subtask],
    ) -> Optional[str]:
        """Return the monitored critical-end subtask name when encoded in the name."""

        if subtask is None:
            return None
        try:
            return extract_monitoring_target_name(subtask.name)
        except ValueError:
            return None

    def _just_monitored_target(
        self,
        curr_state: SchedulerState,
        *,
        monitoring_target_obj: Optional[str],
        monitoring_target_sub_name: Optional[str],
    ) -> bool:
        """Return whether the immediately previous committed step already monitored the target."""

        previous_subtask = curr_state.subtask
        if previous_subtask is None or previous_subtask.subtask_type != "Monitor":
            return False

        previous_target_obj = self._extract_monitoring_target_object_from_subtask(
            previous_subtask
        )
        if (
            monitoring_target_obj is not None
            and previous_target_obj is not None
            and previous_target_obj == monitoring_target_obj
        ):
            return True

        previous_target_name = self._extract_monitoring_target_name_from_subtask(
            previous_subtask
        )
        if (
            monitoring_target_sub_name is not None
            and previous_target_name is not None
            and previous_target_name == monitoring_target_sub_name
        ):
            return True

        return False

    @staticmethod
    def _get_forced_followup_monitoring_name(
        subtask: Optional[Subtask],
    ) -> Optional[str]:
        """Return the committed monitoring follow-up attached to a staging wait."""

        if subtask is None or subtask.subtask_type != "WAIT":
            return None
        followup_name = getattr(subtask, "_forced_followup_monitoring_name", None)
        if not followup_name:
            return None
        return str(followup_name)

    def _get_forced_followup_monitoring_candidate(
        self,
        curr_node: SimulationNode,
        feasible_candidates: List[Candidate],
    ) -> Optional[Candidate]:
        """Return the monitoring candidate committed by the immediately previous wait."""

        forced_name = self._get_forced_followup_monitoring_name(
            curr_node.state.subtask
        )
        if forced_name is None:
            return None

        for candidate in feasible_candidates:
            if candidate.subtask.name == forced_name:
                return candidate

        log.debug(
            "[_get_forced_followup_monitoring_candidate] Committed follow-up monitoring '%s' was not feasible in the next expansion. Falling back to normal policy.",
            forced_name,
        )
        return None

    @staticmethod
    def _build_critical_family_key(
        *,
        critical_start_sub_name: str,
        critical_end_sub_name: str,
    ) -> str:
        """Return a stable identifier for one critical interval family."""

        return f"{critical_start_sub_name}::{critical_end_sub_name}"

    def _replace_monitoring_residual_edge(
        self,
        graph,
        *,
        critical_start_sub_name: str,
        critical_end_sub_name: str,
        monitoring_sub_name: str,
        edge_info: dict,
    ) -> None:
        """Keep only the latest monitoring residual edge for one critical family."""

        family_key = self._build_critical_family_key(
            critical_start_sub_name=critical_start_sub_name,
            critical_end_sub_name=critical_end_sub_name,
        )
        stale_predecessors: list[str] = []

        for pred_name, _, data in graph.in_edges(critical_end_sub_name, data=True):
            if pred_name == monitoring_sub_name:
                continue
            info = data.get("info", {})
            if not info.get("IsMonitoringResidual", False):
                continue
            if info.get("CriticalFamilyKey") != family_key:
                continue
            stale_predecessors.append(pred_name)

        for pred_name in stale_predecessors:
            graph.remove_edge(pred_name, critical_end_sub_name)

        if stale_predecessors:
            log.debug(
                "[_replace_monitoring_residual_edge] Replaced stale monitoring residuals for '%s': %s",
                critical_end_sub_name,
                stale_predecessors,
            )

        updated_edge_info = {
            **edge_info,
            "IsMonitoringResidual": True,
            "CriticalFamilyKey": family_key,
        }
        if not graph.has_edge(monitoring_sub_name, critical_end_sub_name):
            graph.add_edge(
                monitoring_sub_name,
                critical_end_sub_name,
                info=updated_edge_info,
            )
        else:
            graph.edges[monitoring_sub_name, critical_end_sub_name]["info"].update(
                updated_edge_info
            )

    def _resolve_effective_monitoring_budget(self) -> int | None:
        """Return the active per-critical-interval monitoring cap."""

        if self.max_monitoring_per_critical_interval is None:
            return None
        if int(self.max_monitoring_per_critical_interval) < 0:
            return None
        return int(self.max_monitoring_per_critical_interval)

    def _count_monitoring_events_for_interval(
        self,
        curr_state: SchedulerState,
        *,
        critical_start_sub_end_time: float,
        critical_end_sub_name: str,
    ) -> int:
        """Return how many monitors have already executed for one interval."""

        monitor_count = 0
        for completed_entry in curr_state.completed_entries:
            if completed_entry.subtask.subtask_type != "Monitor":
                continue
            if completed_entry.schedule_start_time < critical_start_sub_end_time:
                continue
            try:
                monitored_target_name = extract_monitoring_target_name(
                    completed_entry.subtask.name
                )
            except ValueError:
                continue
            if monitored_target_name == critical_end_sub_name:
                monitor_count += 1
        return monitor_count

    def _monitoring_budget_reached(
        self,
        curr_state: SchedulerState,
        *,
        critical_start_sub_end_time: float,
        critical_end_sub_name: str,
    ) -> bool:
        """Return whether monitoring is exhausted for the current interval."""

        effective_budget = self._resolve_effective_monitoring_budget()
        if effective_budget is None:
            return False
        completed_monitors = self._count_monitoring_events_for_interval(
            curr_state,
            critical_start_sub_end_time=critical_start_sub_end_time,
            critical_end_sub_name=critical_end_sub_name,
        )
        return completed_monitors >= effective_budget

    def _estimate_candidate_navigation_duration(
        self,
        curr_node: SimulationNode,
        candidate: Candidate,
    ) -> float:
        """Estimate how much navigation remains before the candidate can interact."""

        primitive_actions = (
            candidate.subtask.execution.primitive_actions
            if candidate.subtask.execution
            else None
        )
        if not primitive_actions:
            return 0.0
        first_action = primitive_actions[0]
        if not first_action.startswith("NAVIGATE_TO"):
            return float(candidate.estimated_first_nav_duration or 0.0)
        target_obj_id = first_action.split()[1]
        nav_time = self._estimate_navigation_duration(
            curr_node,
            target_obj_id,
            candidate_name=candidate.subtask.name,
        )
        return 0.0 if nav_time is None else float(nav_time)

    def _estimate_candidate_monitoring_window_end(
        self,
        curr_node: SimulationNode,
        candidate: Candidate,
        *,
        expansion_kind: str,
    ) -> Optional[float]:
        """Estimate the latest time at which this expansion can host a monitor."""

        if expansion_kind == "wait":
            target_start_time = self._get_candidate_target_start_time(candidate)
            if target_start_time is None:
                return None
            return max(curr_node.state.current_time, float(target_start_time))

        primitive_actions = (
            candidate.subtask.execution.primitive_actions
            if candidate.subtask.execution
            else None
        )
        if not primitive_actions:
            return None

        action_info = self.action_handler.get_actions_info(curr_node, primitive_actions)
        if action_info is None or not action_info.success:
            log.debug(
                "[_estimate_candidate_monitoring_window_end] Could not simulate '%s' while checking monitoring relevance.",
                candidate.subtask.name,
            )
            return None
        return curr_node.state.current_time + float(action_info.cumulative_time)

    def _resolve_monitoring_obligation(
        self,
        curr_node: SimulationNode,
        *,
        variance_hint: float,
        due: SchedulingDue,
    ) -> Optional[MonitoringObligation]:
        """Resolve one cached active interval into a trigger-bearing obligation."""

        critical_end_sub_name = due.due_related_sub_name
        if critical_end_sub_name is None:
            return None

        monitoring_target_obj = self._get_monitoring_target_obj_for_subtask_name(
            curr_node, critical_end_sub_name
        )
        if monitoring_target_obj is None:
            log.debug(
                "[_resolve_monitoring_obligation] Could not determine monitoring target for '%s'.",
                critical_end_sub_name,
            )
            return None

        completed_entries_map = {
            ce.subtask.name: ce for ce in curr_node.state.completed_entries
        }
        incoming_edges = curr_node.state.constraints.in_edges(
            critical_end_sub_name, data=True
        )
        for critical_start_sub_name, _end_name, data in incoming_edges:
            info = data.get("info", {})
            if not info.get("IsCritical") or info.get("Interval", 0.0) <= EPSILON:
                continue

            start_entry = completed_entries_map.get(critical_start_sub_name)
            if start_entry is None or start_entry.subtask.subtask_type == "Monitor":
                continue

            interval = float(info.get("Interval", 0.0))
            due_date = start_entry.schedule_end_time + interval
            if abs(due_date - due.due_date) > EPSILON:
                continue

            variance = float(info.get("Variance", variance_hint))
            trigger_time = self._compute_monitoring_trigger_time(
                raw_object_name=monitoring_target_obj,
                critical_start_sub_end_time=start_entry.schedule_end_time,
                mean_duration=interval,
                variance=variance,
            )
            return MonitoringObligation(
                trigger_time=trigger_time,
                variance=variance,
                due=SchedulingDue(
                    due_date=due_date,
                    due_related_sub_name=critical_end_sub_name,
                ),
            )

        log.debug(
            "[_resolve_monitoring_obligation] Could not match an active critical edge for '%s' (due=%.2f).",
            critical_end_sub_name,
            due.due_date,
        )
        return None

    def _get_relevant_monitoring_obligations(
        self,
        curr_node: SimulationNode,
        candidate: Candidate,
        *,
        expansion_kind: str,
        restrict_wait_target: bool = True,
    ) -> List[MonitoringObligation]:
        """Return monitoring obligations whose trigger actually lands in this expansion."""

        if self._has_real_world_late_due_signal(curr_node, candidate.subtask.name):
            log.debug(
                "[_get_relevant_monitoring_obligations] Skipping monitoring obligations for '%s' "
                "because the latest ROS observation already marked it due/late.",
                candidate.subtask.name,
            )
            return []

        active_intervals = self._get_active_monitoring_intervals(curr_node)
        if not active_intervals:
            return []

        candidate_window_end = self._estimate_candidate_monitoring_window_end(
            curr_node,
            candidate,
            expansion_kind=expansion_kind,
        )
        if candidate_window_end is None:
            return []

        obligations: List[MonitoringObligation] = []
        for variance, due in active_intervals:
            if (
                expansion_kind == "wait"
                and restrict_wait_target
                and due.due_related_sub_name != candidate.subtask.name
            ):
                # Current wait expansion only knows how to split for the waited-on target.
                continue

            obligation = self._resolve_monitoring_obligation(
                curr_node,
                variance_hint=float(variance),
                due=due,
            )
            if obligation is None:
                continue
            if obligation.trigger_time <= candidate_window_end + EPSILON:
                obligations.append(obligation)

        obligations.sort(key=lambda obligation: obligation.trigger_time)
        return obligations

    def _get_candidate_target_start_time(
        self,
        candidate: Candidate,
    ) -> Optional[float]:
        """Return the earliest target interaction time for a candidate."""

        if candidate.actual_interaction_start_time is not None:
            return float(candidate.actual_interaction_start_time)
        if candidate.logical_interaction_start_time is not None:
            return float(candidate.logical_interaction_start_time)
        return None

    @staticmethod
    def _can_candidate_start_interaction_now(
        curr_node: SimulationNode,
        candidate: Candidate,
    ) -> bool:
        """Return whether the candidate can start interacting under planner grace."""

        if candidate.logical_interaction_start_time is None:
            return True
        physical_earliest_start = (
            curr_node.state.current_time + candidate.estimated_first_nav_duration
        )
        readiness_tolerance = constants.RISK_GRACE_SECONDS
        return physical_earliest_start >= (
            float(candidate.logical_interaction_start_time) - readiness_tolerance
        )

    def _get_blocked_candidate_frontier(
        self,
        not_yet_candidates: List[Candidate],
    ) -> List[Candidate]:
        """Return blocked candidates on the earliest timing frontier."""

        sorted_candidates = sorted(
            not_yet_candidates,
            key=lambda candidate: (
                (
                    self._get_candidate_target_start_time(candidate)
                    if self._get_candidate_target_start_time(candidate) is not None
                    else float("inf")
                ),
                candidate.subtask.name,
            ),
        )
        finite_candidates = [
            candidate
            for candidate in sorted_candidates
            if self._get_candidate_target_start_time(candidate) is not None
            and self._get_candidate_target_start_time(candidate) != float("inf")
        ]
        if not finite_candidates:
            return []

        earliest_target_time = self._get_candidate_target_start_time(
            finite_candidates[0]
        )
        if earliest_target_time is None:
            return []
        return [
            candidate
            for candidate in finite_candidates
            if abs(
                float(self._get_candidate_target_start_time(candidate))
                - earliest_target_time
            )
            <= EPSILON
        ]

    @staticmethod
    def _select_monitoring_obligation(
        obligations: List[MonitoringObligation],
    ) -> Optional[MonitoringObligation]:
        """Pick one obligation using the scheduler's trigger-first arbitration."""

        if not obligations:
            return None

        earliest_trigger_time = min(
            obligation.trigger_time for obligation in obligations
        )
        earliest_obligations = [
            obligation
            for obligation in obligations
            if abs(obligation.trigger_time - earliest_trigger_time) <= EPSILON
        ]
        return max(
            earliest_obligations,
            key=lambda obligation: (obligation.variance, -obligation.due.due_date),
        )

    @staticmethod
    def _select_wait_event(
        events: List[WaitExpansionEvent],
    ) -> Optional[WaitExpansionEvent]:
        """Select the next wait event using event-time-first arbitration."""

        if not events:
            return None

        return min(
            events,
            key=lambda event: (
                event.event_time,
                0 if event.obligation is not None else 1,
                event.raw_trigger_time,
                -event.variance,
                event.due_date,
                event.candidate.subtask.name,
            ),
        )

    @staticmethod
    def _wait_nodes_are_equivalent(
        first: Optional[SimulationNode],
        second: Optional[SimulationNode],
    ) -> bool:
        """Return whether two wait expansions represent the same committed step."""

        if first is None or second is None:
            return False

        return (
            first.state.subtask.name == second.state.subtask.name
            and abs(first.state.current_time - second.state.current_time) <= EPSILON
            and len(first.state.completed_entries)
            == len(second.state.completed_entries)
            and tuple(subtask.name for subtask in first.state.remaining_subtasks)
            == tuple(subtask.name for subtask in second.state.remaining_subtasks)
        )

    @staticmethod
    def _find_candidate_by_name(
        target_name: Optional[str],
        feasible_candidates: List[Candidate],
        not_yet_candidates: List[Candidate],
    ) -> Optional[Candidate]:
        """Return the exact candidate whose subtask name matches the target."""

        if target_name is None:
            return None

        for candidate in feasible_candidates:
            if candidate.subtask.name == target_name:
                return candidate

        for candidate in not_yet_candidates:
            if candidate.subtask.name == target_name:
                return candidate

        return None

    @staticmethod
    def _build_wait_navigation_action(candidate: Candidate) -> Optional[str]:
        """Best-effort navigation action used inside staged wait subtasks."""

        primitive_actions = (
            candidate.subtask.execution.primitive_actions
            if candidate.subtask.execution
            else None
        )
        if not primitive_actions:
            return None

        for action in primitive_actions:
            parts = action.split()
            if len(parts) <= 1:
                continue
            if parts[0].upper() == "WAIT":
                continue
            if parts[0] == "NAVIGATE_TO":
                return action
            return f"NAVIGATE_TO {parts[1]}"
        return None

    def _build_wait_primitive_actions(
        self,
        candidate: Candidate,
        *,
        nav_duration: float,
        idle_wait_duration: float,
    ) -> list[str]:
        """Return staged wait actions with optional early navigation."""

        primitive_actions: list[str] = []
        nav_action = None
        if nav_duration > EPSILON:
            nav_action = self._build_wait_navigation_action(candidate)
            if nav_action is not None:
                primitive_actions.append(nav_action)
            else:
                log.debug(
                    "[_build_wait_primitive_actions] Could not infer a navigation action for '%s'; "
                    "falling back to WAIT-only semantics for %.2fs.",
                    candidate.subtask.name,
                    nav_duration + idle_wait_duration,
                )

        effective_wait_duration = (
            idle_wait_duration
            if primitive_actions
            else nav_duration + idle_wait_duration
        )
        if effective_wait_duration > EPSILON or not primitive_actions:
            primitive_actions.append(f"WAIT {effective_wait_duration}")

        return primitive_actions

    def _simulate_wait_subtask(
        self,
        curr_node: SimulationNode,
        candidate: Candidate,
        *,
        idle_wait_duration: float,
        nav_duration: float,
    ) -> Optional[tuple[Subtask, CompletedEntry, SchedulerState]]:
        """Simulate one scheduler-generated wait step, optionally with early navigation."""

        primitive_actions = self._build_wait_primitive_actions(
            candidate,
            nav_duration=nav_duration,
            idle_wait_duration=idle_wait_duration,
        )
        action_info = self.action_handler.get_actions_info(curr_node, primitive_actions)
        if action_info is None or not action_info.success:
            log.warning(
                "Action simulation failed for staged wait of '%s'. Cannot expand wait branch.",
                candidate.subtask.name,
            )
            return None

        actual_duration = float(action_info.cumulative_time)
        wait_sub = Subtask(
            task_name=None,
            name=f"Wait for {candidate.subtask.name}",
            duration=Duration(
                interval=actual_duration,
                type="Controllable",
                total_time=actual_duration,
            ),
            repetition=1,
            subtask_type="WAIT",
            execution=Execution(objects=None, primitive_actions=primitive_actions),
            temporal_constraints=None,
        )

        curr_state = curr_node.state
        start_time = curr_state.current_time
        end_time = start_time + actual_duration
        completed_entry = CompletedEntry(
            subtask=wait_sub,
            schedule_start_time=start_time,
            schedule_end_time=end_time,
            schedule_nav_time=float(action_info.first_nav_duration or 0.0),
            execution_status=bool(action_info.success),
        )
        new_state = SchedulerState(
            subtask=wait_sub,
            completed_entries=curr_state.completed_entries + [completed_entry],
            remaining_subtasks=curr_state.remaining_subtasks,
            constraints=curr_state.constraints,
            current_time=end_time,
            scene_positions=action_info.scene_positions,
            held_object=(
                action_info.held_object
                if action_info.held_object is not None
                else curr_state.held_object
            ),
        )
        return wait_sub, completed_entry, new_state

    @staticmethod
    def _normalize_monitoring_object_name(
        raw_object_name: Optional[str],
    ) -> Optional[str]:
        """Normalize an object identifier to its object type.

        Args:
            raw_object_name: Raw object identifier or type string.

        Returns:
            Object type without instance suffix.
        """

        if not raw_object_name:
            return None
        return raw_object_name.split("|")[0]

    def _estimate_navigation_duration(
        self,
        curr_node: SimulationNode,
        target_obj_id: str,
        *,
        candidate_name: str,
    ) -> Optional[float]:
        """Estimate navigation time for a single target object.

        Args:
            curr_node: Current simulation node used as the starting state.
            target_obj_id: Target object identifier for navigation.
            candidate_name: Human-readable subtask name for diagnostics.

        Returns:
            Navigation duration when simulation succeeds, otherwise ``None``.
        """

        navigation_info = self.action_handler.get_actions_info(
            curr_node, [f"NAVIGATE_TO {target_obj_id}"]
        )
        if navigation_info is None or not navigation_info.success:
            log.warning(
                "Navigation simulation failed while expanding '%s' toward '%s'.",
                candidate_name,
                target_obj_id,
            )
            return None
        return navigation_info.action_duration

    def _compute_monitoring_trigger_time(
        self,
        *,
        raw_object_name: Optional[str],
        critical_start_sub_end_time: float,
        mean_duration: float,
        variance: float,
    ) -> float:
        """Delegate monitoring timing to the configured backend policy.

        Args:
            raw_object_name: Raw object identifier associated with the interval.
            critical_start_sub_end_time: End time of the critical start subtask.
            mean_duration: Current expected interval duration.
            variance: Current interval variance.

        Returns:
            Absolute trigger time for inserting monitoring.
        """

        object_name = self._normalize_monitoring_object_name(raw_object_name)
        trigger_cache_key = (
            self.monitoring_policy.method,
            object_name,
            critical_start_sub_end_time,
            mean_duration,
            variance,
        )
        if self._search_cache is not None:
            cached_trigger = self._search_cache.monitoring_triggers.get(
                trigger_cache_key
            )
            if cached_trigger is not None:
                self._search_cache.trigger_cache_hits += 1
                return cached_trigger

        trigger_time = self.monitoring_policy.compute_trigger_time(
            MonitoringTriggerContext(
                object_name=object_name,
                critical_start_end_time=critical_start_sub_end_time,
                mean_duration=mean_duration,
                variance=variance,
            )
        )
        log.debug(
            "Monitoring trigger (%s): object=%s, mean=%.2f, variance=%.2f -> %.2f",
            self.monitoring_policy.method,
            object_name,
            mean_duration,
            variance,
            trigger_time,
        )
        if self._search_cache is not None:
            self._search_cache.trigger_cache_misses += 1
            self._search_cache.monitoring_triggers[trigger_cache_key] = trigger_time
        return trigger_time

    @staticmethod
    def _compute_latest_safe_monitoring_start_time(
        *,
        target_interaction_start_time: float,
        post_monitor_buffer: float = 0.0,
    ) -> float:
        """Return the latest monitor-start time that still preserves the target start."""

        return (
            float(target_interaction_start_time)
            - MONITORING_DURATION
            - max(0.0, float(post_monitor_buffer))
        )

    @staticmethod
    def _resolve_original_critical_deadline(
        *,
        selected_obligation: Optional[MonitoringObligation] = None,
        critical_start_sub_end_time: Optional[float] = None,
        critical_interval_duration: Optional[float] = None,
    ) -> float:
        """Return the original critical deadline for a monitoring decision."""

        if selected_obligation is not None:
            return float(selected_obligation.due.due_date)
        if critical_start_sub_end_time is None or critical_interval_duration is None:
            raise ValueError(
                "Either selected_obligation or critical interval timing must be provided."
            )
        return float(critical_start_sub_end_time) + float(critical_interval_duration)

    @staticmethod
    def _monitoring_finishes_before_critical_deadline(
        *,
        monitor_start_time: float,
        critical_deadline: float,
    ) -> bool:
        """Return whether monitoring starting at ``monitor_start_time`` finishes before the deadline."""

        return (
            float(monitor_start_time) + MONITORING_DURATION
            <= float(critical_deadline) + EPSILON
        )

    @staticmethod
    def _immediate_monitoring_misses_critical_deadline(
        *,
        current_time: float,
        critical_deadline: float,
    ) -> bool:
        """Return whether starting monitoring now would finish after the deadline."""

        return not Scheduler._monitoring_finishes_before_critical_deadline(
            monitor_start_time=current_time,
            critical_deadline=critical_deadline,
        )

    # ==========================================================================
    #           SUBTASK EXPANSION: Single Subtask or Wait
    # ==========================================================================
    def _expand_single_subtask(
        self,
        curr_node: SimulationNode,
        candidate: Candidate,
        not_yet_candidates: List[Candidate],
        feasible_candidates: List[Candidate] = None,
    ) -> Optional[SimulationNode]:
        """
        Expands the given candidate subtask by deciding whether to split it
        into a monitoring subtask or not.

        Args:
            curr_node (SimulationNode): The current node in the search tree.
            candidate (Candidate): The subtask candidate to expand.

        Returns:
            Optional[SimulationNode]: The resulting child node if successful,
            otherwise None.
        """
        log.debug(
            f"[_expand_single_subtask] Checking expansion for subtask: {candidate.subtask.name}."
        )

        # 모니터링 필요?
        need_monitor, due_info = self._should_split_with_monitoring(
            curr_node,
            candidate,
            expansion_kind="action",
        )
        if need_monitor and constants.MONITORING_ENABLED:
            log.debug(
                f"[_expand_single_subtask] Subtask {candidate.subtask.name} requires monitoring-based splitting."
            )
            # 왜 덮어씌우지?
            candidate.scheduling_due = due_info
            return self._expand_subtask_with_monitoring(
                curr_node, candidate, not_yet_candidates, feasible_candidates
            )
        else:
            log.debug(
                f"[_expand_single_subtask] Subtask {candidate.subtask.name} will be executed without monitoring."
            )
            return self._expand_subtask_wo_monitoring(
                curr_node, candidate, not_yet_candidates, feasible_candidates
            )

    def _expand_single_wait(
        self,
        curr_node: SimulationNode,
        candidate: Candidate,
        not_yet_candidates: List[Candidate],
        feasible_candidates: List[Candidate] = None,
    ) -> Optional[SimulationNode]:
        """
        Expands the wait subtask by deciding whether to split it
        into a monitoring subtask or not.

        Args:
            curr_node (SimulationNode): The current node in the search tree.
            candidate (Candidate): The subtask candidate will be expand.

        Returns:
            Optional[SimulationNode]: The resulting child node if successful,
            otherwise None.
        """
        wait_nodes = self._expand_wait_options(
            curr_node,
            candidate,
            not_yet_candidates,
            feasible_candidates=feasible_candidates,
        )
        return wait_nodes[0] if wait_nodes else None

    def _expand_wait_event(
        self,
        curr_node: SimulationNode,
        original_candidate: Candidate,
        selected_event: Optional[WaitExpansionEvent],
        not_yet_candidates: List[Candidate],
        feasible_candidates: List[Candidate] = None,
        local_nav_time: float = 0.0,
        max_wait_duration: Optional[float] = None,
    ) -> Optional[SimulationNode]:
        """Expand one selected wait event into a concrete child node."""

        if selected_event is None:
            return self._expand_wait_wo_monitoring(
                curr_node,
                original_candidate,
                not_yet_candidates,
                nav_duration=local_nav_time,
                feasible_candidates=feasible_candidates,
                max_wait_duration=max_wait_duration,
            )

        if selected_event.obligation is not None and constants.MONITORING_ENABLED:
            monitor_candidate = replace(
                selected_event.candidate,
                scheduling_due=selected_event.obligation.due,
            )
            return self._expand_wait_with_monitoring(
                curr_node,
                monitor_candidate,
                not_yet_candidates,
                nav_duration=selected_event.nav_duration,
                feasible_candidates=feasible_candidates,
                selected_obligation=selected_event.obligation,
            )

        return self._expand_wait_wo_monitoring(
            curr_node,
            selected_event.candidate,
            not_yet_candidates,
            nav_duration=selected_event.nav_duration,
            feasible_candidates=feasible_candidates,
            max_wait_duration=max_wait_duration,
        )

    def _expand_wait_options(
        self,
        curr_node: SimulationNode,
        candidate: Candidate,
        not_yet_candidates: List[Candidate],
        feasible_candidates: List[Candidate] = None,
    ) -> List[SimulationNode]:
        """Expand the primary wait event and keep a local execute fallback when relevant."""

        if self._wait_is_disallowed_for_candidate(curr_node, candidate):
            log.debug(
                "[_expand_single_wait] Wait is disallowed for '%s'. Skipping wait expansion.",
                candidate.subtask.name,
            )
            return []

        local_nav_time = self._estimate_candidate_navigation_duration(
            curr_node, candidate
        )
        wait_events = self._collect_wait_events(
            curr_node,
            candidate,
            feasible_candidates or [],
            not_yet_candidates,
            local_nav_duration=local_nav_time,
        )
        selected_event = self._select_wait_event(wait_events)

        wait_nodes: List[SimulationNode] = []
        primary_wait_node = self._expand_wait_event(
            curr_node,
            candidate,
            selected_event,
            not_yet_candidates,
            feasible_candidates=feasible_candidates,
            local_nav_time=local_nav_time,
        )
        if primary_wait_node is not None:
            wait_nodes.append(primary_wait_node)

        if selected_event is None:
            return wait_nodes

        local_execute_event = next(
            (
                event
                for event in wait_events
                if event.kind == "local_execute"
                and event.candidate.subtask.name == candidate.subtask.name
            ),
            None,
        )
        should_add_local_execute_backup = (
            local_execute_event is not None and selected_event.kind != "local_execute"
        )
        if should_add_local_execute_backup:
            backup_max_wait_duration = None
            skip_local_execute_backup = False
            if selected_event.obligation is not None:
                backup_max_wait_duration = max(
                    0.0, selected_event.event_time - curr_node.state.current_time
                )
                if (
                    self.real_world_mode
                    and selected_event.kind == "local_monitor"
                    and primary_wait_node is not None
                ):
                    log.debug(
                        "[_expand_wait_options] Skipping local execute backup for '%s' "
                        "because the realized ROS local_monitor primary already commits the "
                        "same %.2fs checkpoint with forced follow-up.",
                        candidate.subtask.name,
                        backup_max_wait_duration,
                    )
                    skip_local_execute_backup = True
                if (
                    selected_event.kind == "local_monitor"
                    and primary_wait_node is not None
                    and not self.real_world_mode
                ):
                    log.debug(
                        "[_expand_wait_options] Retaining local execute backup for '%s' "
                        "because simulation mode allows the capped plain-wait alternative "
                        "beside the realized local_monitor primary (checkpoint %.2fs).",
                        candidate.subtask.name,
                        backup_max_wait_duration,
                    )
                if backup_max_wait_duration <= (
                    float(local_execute_event.nav_duration) + EPSILON
                ):
                    log.debug(
                        "[_expand_wait_options] Skipping local execute backup for '%s' "
                        "because the capped wait window %.2fs collapses to navigation-only "
                        "(nav_duration=%.2fs) after selecting %s.",
                        candidate.subtask.name,
                        backup_max_wait_duration,
                        float(local_execute_event.nav_duration),
                        selected_event.kind,
                    )
                    skip_local_execute_backup = True

            if not skip_local_execute_backup:
                local_execute_wait_node = self._expand_wait_event(
                    curr_node,
                    candidate,
                    local_execute_event,
                    not_yet_candidates,
                    feasible_candidates=feasible_candidates,
                    local_nav_time=local_nav_time,
                    max_wait_duration=backup_max_wait_duration,
                )
                if (
                    local_execute_wait_node is not None
                    and not self._wait_nodes_are_equivalent(
                        primary_wait_node,
                        local_execute_wait_node,
                    )
                ):
                    wait_nodes.append(local_execute_wait_node)

        return wait_nodes

    # ======================
    # Helper: 모니터링 필요한지
    # ======================
    def _monitoring_split_is_disallowed(
        self,
        curr_node: SimulationNode,
        candidate: Candidate,
    ) -> bool:
        """Return whether the candidate is categorically ineligible for monitoring."""

        if candidate.subtask.decomposed:
            log.debug(
                f"[_should_split_with_monitoring] Subtask {candidate.subtask.name} is already handled/decomposed. No split."
            )
            return True

        immediate_predecessor_name = self._get_immediate_critical_predecessor_name(
            curr_node, candidate
        )
        if immediate_predecessor_name is not None:
            log.debug(
                "[_should_split_with_monitoring] Subtask %s has an immediate critical predecessor (%s). "
                "Monitoring is disallowed.",
                candidate.subtask.name,
                immediate_predecessor_name,
            )
            return True

        return False

    def _get_immediate_critical_predecessor_name(
        self,
        curr_node: SimulationNode,
        candidate: Candidate,
    ) -> Optional[str]:
        """Return the immediate critical predecessor name when the candidate has a zero-interval in-edge."""

        in_slots = self.constraint_handler.get_time_slots(
            candidate.subtask.name, curr_node.state.constraints, direction="in"
        )
        for slot in in_slots:
            if slot.is_critical and slot.interval <= EPSILON:
                return slot.related_subtask_name
        return None

    def _wait_is_disallowed_for_candidate(
        self,
        curr_node: SimulationNode,
        candidate: Candidate,
    ) -> bool:
        """Return whether scheduler-generated wait branches are forbidden for the candidate."""

        immediate_predecessor_name = self._get_immediate_critical_predecessor_name(
            curr_node, candidate
        )
        if immediate_predecessor_name is None:
            return False

        log.debug(
            "[_wait_is_disallowed_for_candidate] Candidate %s has an immediate critical predecessor (%s). "
            "Scheduler-generated WAIT is disallowed.",
            candidate.subtask.name,
            immediate_predecessor_name,
        )
        return True

    def _should_split_with_monitoring(
        self,
        curr_node: SimulationNode,
        candidate: Candidate,
        *,
        expansion_kind: str = "action",
    ) -> tuple[bool, Optional[SchedulingDue]]:
        """
        Determines if a task should be split for monitoring based on three rules.
        Args:
            curr_node: The current simulation node.
            candidate: The candidate subtask to evaluate.

        Returns:
            True if the candidate should be split for monitoring, False otherwise.
        """
        if self._monitoring_split_is_disallowed(curr_node, candidate):
            return False, None

        # Rule 3: Split only if an active critical interval exists.
        relevant_obligations = self._get_relevant_monitoring_obligations(
            curr_node,
            candidate,
            expansion_kind=expansion_kind,
        )

        if not relevant_obligations:
            log.debug(
                "[_should_split_with_monitoring] No trigger-matured monitoring obligation intersects '%s' (%s expansion).",
                candidate.subtask.name,
                expansion_kind,
            )
            return False, None

        selected_obligation = self._select_monitoring_obligation(relevant_obligations)
        if selected_obligation is None:
            return False, None

        candidate.scheduling_due = selected_obligation.due
        log.debug(
            "[_should_split_with_monitoring] Selected monitoring obligation for '%s' (%s expansion): "
            "target='%s', trigger=%.2f, due=%.2f, contenders=%d.",
            candidate.subtask.name,
            expansion_kind,
            selected_obligation.due.due_related_sub_name,
            selected_obligation.trigger_time,
            selected_obligation.due.due_date,
            len(relevant_obligations),
        )

        return True, selected_obligation.due

    def _build_wait_monitor_event(
        self,
        curr_node: SimulationNode,
        candidate: Candidate,
        obligation: MonitoringObligation,
        *,
        kind: str,
        nav_duration: float,
    ) -> Optional[WaitExpansionEvent]:
        """Build one monitor-bearing wait event if it is still safe to realize."""

        target_start_time = self._get_candidate_target_start_time(candidate)
        if target_start_time is None:
            return None
        original_critical_deadline = self._resolve_original_critical_deadline(
            selected_obligation=obligation
        )

        if (
            curr_node.state.current_time + MONITORING_DURATION + nav_duration
            > target_start_time + EPSILON
        ):
            return None

        latest_safe_target_start_time = self._compute_latest_safe_monitoring_start_time(
            target_interaction_start_time=target_start_time
        )
        latest_safe_deadline_start_time = (
            self._compute_latest_safe_monitoring_start_time(
                target_interaction_start_time=original_critical_deadline
            )
        )
        latest_safe_start_time = min(
            latest_safe_target_start_time,
            latest_safe_deadline_start_time,
        )
        if kind == "local_monitor":
            event_time = max(
                curr_node.state.current_time,
                min(obligation.trigger_time, latest_safe_start_time),
            )
        else:
            event_time = max(curr_node.state.current_time, obligation.trigger_time)
            if event_time > latest_safe_start_time + EPSILON:
                return None
        if not self._monitoring_finishes_before_critical_deadline(
            monitor_start_time=event_time,
            critical_deadline=original_critical_deadline,
        ):
            return None

        return WaitExpansionEvent(
            kind=kind,
            candidate=candidate,
            event_time=event_time,
            nav_duration=nav_duration,
            raw_trigger_time=obligation.trigger_time,
            variance=obligation.variance,
            due_date=obligation.due.due_date,
            obligation=obligation,
        )

    def _collect_wait_events(
        self,
        curr_node: SimulationNode,
        candidate: Candidate,
        feasible_candidates: List[Candidate],
        not_yet_candidates: List[Candidate],
        *,
        local_nav_duration: float,
    ) -> List[WaitExpansionEvent]:
        """Collect the next meaningful wait events for one blocked-frontier candidate."""

        events: List[WaitExpansionEvent] = []
        target_start_time = self._get_candidate_target_start_time(candidate)
        if (
            target_start_time is not None
            and target_start_time > curr_node.state.current_time + EPSILON
        ):
            events.append(
                WaitExpansionEvent(
                    kind="local_execute",
                    candidate=candidate,
                    event_time=target_start_time,
                    nav_duration=local_nav_duration,
                    due_date=target_start_time,
                )
            )

        if not constants.MONITORING_ENABLED:
            return events

        if not self._monitoring_split_is_disallowed(curr_node, candidate):
            local_obligation = self._select_monitoring_obligation(
                self._get_relevant_monitoring_obligations(
                    curr_node,
                    candidate,
                    expansion_kind="wait",
                )
            )
            if local_obligation is not None:
                local_event = self._build_wait_monitor_event(
                    curr_node,
                    candidate,
                    local_obligation,
                    kind="local_monitor",
                    nav_duration=local_nav_duration,
                )
                if local_event is not None:
                    events.append(local_event)

        current_target_name = candidate.subtask.name
        global_obligations = self._get_relevant_monitoring_obligations(
            curr_node,
            candidate,
            expansion_kind="wait",
            restrict_wait_target=False,
        )
        for obligation in global_obligations:
            rescue_target_name = obligation.due.due_related_sub_name
            if rescue_target_name == current_target_name:
                continue

            rescue_candidate = self._find_candidate_by_name(
                rescue_target_name,
                feasible_candidates,
                not_yet_candidates,
            )
            if rescue_candidate is None:
                continue
            if self._monitoring_split_is_disallowed(curr_node, rescue_candidate):
                continue

            rescue_event = self._build_wait_monitor_event(
                curr_node,
                rescue_candidate,
                obligation,
                kind="rescue_monitor",
                nav_duration=0.0,
            )
            if rescue_event is not None:
                events.append(rescue_event)

        return events

    # -----------------------------------------------------
    # (A) 서브태스크 (no monitoring)
    # -----------------------------------------------------
    def _expand_subtask_wo_monitoring(
        self,
        curr_node: SimulationNode,
        candidate: Candidate,
        not_yet_candidates: List[Candidate],
        feasible_candidates: List[Candidate] = None,
    ) -> Optional[SimulationNode]:
        """
        Expands a non-monitoring subtask. The subtask is executed fully at once.
        Navigation (if any, as first_nav_duration) + Interaction are performed.
        """
        log.debug(f"[_expand_subtask_wo_monitoring] {candidate.subtask.name}")
        curr_state = curr_node.state

        curr_depth = curr_node.depth
        original_task_name = candidate.subtask.name

        planned_nav_start_time = curr_state.current_time

        sub_actions = candidate.subtask.execution.primitive_actions

        if not sub_actions:
            return None

        try:
            executed_action_info: Optional[ActionResult] = (
                self.action_handler.get_actions_info(curr_node, sub_actions)
            )
            if executed_action_info is None or not executed_action_info.success:
                log.warning(
                    f"Action simulation failed for {original_task_name}. Cannot expand."
                )
                return None
        except ValueError as e:
            log.error(f"Error during action simulation for {original_task_name}: {e}")
            return None

        total_subtask_duration_from_sim = executed_action_info.cumulative_time
        planned_subtask_completion_time = (
            planned_nav_start_time + total_subtask_duration_from_sim
        )

        copied_sub = copy.deepcopy(candidate.subtask)
        copied_sub.duration.total_time = total_subtask_duration_from_sim

        sim_success_status = (
            executed_action_info.success if executed_action_info else False
        )

        completed_entry = CompletedEntry(
            subtask=copied_sub,
            schedule_start_time=planned_nav_start_time,
            schedule_end_time=planned_subtask_completion_time,
            schedule_nav_time=executed_action_info.first_nav_duration,
            execution_status=sim_success_status,
        )
        new_completed = curr_state.completed_entries + [completed_entry]
        new_remain = [
            r for r in curr_state.remaining_subtasks if r.name != original_task_name
        ]

        new_scene_positions = executed_action_info.scene_positions
        new_held_obj = executed_action_info.held_object

        new_state = SchedulerState(
            subtask=copied_sub,
            completed_entries=new_completed,
            remaining_subtasks=new_remain,
            constraints=curr_state.constraints,
            current_time=planned_subtask_completion_time,
            scene_positions=new_scene_positions,
            held_object=new_held_obj,
        )

        # Global Risk Check을 위해 feasible_candidates도 포함하여 전달
        all_candidates = not_yet_candidates
        if feasible_candidates:
            all_candidates = feasible_candidates + not_yet_candidates

        step_risk, total_heuristic_cost = self.cost_calculator.calc_heuristic(
            curr_node, candidate, all_candidates
        )

        # [Modified] HeuristicManager returns h(n) (Remaining Work + Debt).
        # We add planned_subtask_completion_time (g(n)) here to get the total cost f(n) = g(n) + h(n).
        new_cost = planned_subtask_completion_time + total_heuristic_cost

        # Accumulate max risk level along the path
        new_risk = max(curr_node.risk_level, step_risk)

        log.info(
            f"[_expand_subtask_wo_monitoring] {candidate.subtask.name}: current_time({planned_subtask_completion_time:.2f}) = from_time({curr_state.current_time:.2f}) + total_subtask_duration({total_subtask_duration_from_sim:.2f})"
        )
        log.info(
            f"cost({new_cost:.2f}) = planned_subtask_completion_time({planned_subtask_completion_time:.2f}) + total_heuristic_cost({total_heuristic_cost:.2f})\n"
        )

        return SimulationNode(
            parent_node=curr_node,
            heuristic_cost=new_cost,
            depth=curr_depth + 1,
            tie_breaker=next(self._counter),
            state=new_state,
            risk_level=new_risk,
        )

    def _insert_monitoring_step(
        self,
        curr_node: SimulationNode,
        candidate: Candidate,
        monitoring_target_obj: Optional[str],
        predecessor_name: str,
        target_actual_start_time: Optional[float],
        not_yet_candidates: List[Candidate],
        *,
        critical_start_sub_name: Optional[str] = None,
        critical_start_sub_end_time: Optional[float] = None,
        critical_end_sub_name: Optional[str] = None,
        critical_interval_duration: Optional[float] = None,
        monitoring_target_sub_name: Optional[str] = None,
        is_critical_link: bool = True,
    ) -> Optional[SimulationNode]:
        """Execute monitoring immediately and update state/constraints for a follow-up `candidate`."""

        if not monitoring_target_obj:
            log.debug(
                "[_insert_monitoring] target_obj is None. Cannot insert monitoring step."
            )
            return None

        if target_actual_start_time is None:
            log.debug(
                "[_insert_monitoring] target_actual_start_time is None. Cannot insert monitoring step."
            )
            return None

        monitor_base_name = (
            monitoring_target_sub_name
            if monitoring_target_sub_name
            else candidate.subtask.name
        )
        monitor_sub = TaskUtil.create_monitoring_subtask(
            name=monitor_base_name, obj=monitoring_target_obj
        )
        monitor_sub.decomposed = True

        monitor_candidate = Candidate(
            subtask=monitor_sub,
            is_critical=True,
            actual_interaction_start_time=curr_node.state.current_time,
            logical_interaction_start_time=curr_node.state.current_time,
        )
        # 모니터링 작업을 수행하고, 그 결과를 모니터링 노드로 반환합니다.
        monitor_node = self._expand_subtask_wo_monitoring(
            curr_node, monitor_candidate, not_yet_candidates
        )
        if monitor_node is None:
            return None

        monitor_state = monitor_node.state
        monitor_start_time = curr_node.state.current_time
        monitor_finish_time = monitor_state.current_time

        final_remaining_subtasks = list(monitor_state.remaining_subtasks)
        # Ensure candidate is in remaining (if it wasn't already)
        if candidate.subtask.name not in [r.name for r in final_remaining_subtasks]:
            final_remaining_subtasks.append(copy.deepcopy(candidate.subtask))

        new_constraints = copy.deepcopy(monitor_state.constraints)
        if not new_constraints.has_node(monitor_sub.name):
            new_constraints.add_node(monitor_sub.name)
        if not new_constraints.has_node(candidate.subtask.name):
            new_constraints.add_node(candidate.subtask.name)

        pred_to_mon_edge = {"Interval": 0.0, "IsCritical": True}
        if not new_constraints.has_edge(predecessor_name, monitor_sub.name):
            new_constraints.add_edge(
                predecessor_name,
                monitor_sub.name,
                info=pred_to_mon_edge,
            )
        else:
            new_constraints.edges[predecessor_name, monitor_sub.name][
                "info"
            ] = pred_to_mon_edge

        remaining_slack = max(
            0.0, target_actual_start_time - monitor_state.current_time
        )
        mon_to_candidate_edge = {
            "Interval": remaining_slack,
            "IsCritical": is_critical_link,
        }

        if not new_constraints.has_edge(monitor_sub.name, candidate.subtask.name):
            new_constraints.add_edge(
                monitor_sub.name,
                candidate.subtask.name,
                info=mon_to_candidate_edge,
            )
        else:
            new_constraints.edges[monitor_sub.name, candidate.subtask.name][
                "info"
            ] = mon_to_candidate_edge

        updated_state = SchedulerState(
            subtask=monitor_state.subtask,
            completed_entries=monitor_state.completed_entries,
            remaining_subtasks=final_remaining_subtasks,
            constraints=new_constraints,
            current_time=monitor_state.current_time,
            scene_positions=monitor_state.scene_positions,
            held_object=monitor_state.held_object,
        )

        if (
            critical_start_sub_name
            and critical_end_sub_name
            and critical_start_sub_end_time is not None
            and critical_interval_duration is not None
        ):
            constraints_with_critical = copy.deepcopy(updated_state.constraints)

            for node_name in (
                critical_start_sub_name,
                critical_end_sub_name,
                monitor_sub.name,
            ):
                if not constraints_with_critical.has_node(node_name):
                    constraints_with_critical.add_node(node_name)

            interval_start_to_mon = max(
                0.0, monitor_start_time - critical_start_sub_end_time
            )
            edge_info_start = {"Interval": interval_start_to_mon, "IsCritical": True}
            if not constraints_with_critical.has_edge(
                critical_start_sub_name, monitor_sub.name
            ):
                constraints_with_critical.add_edge(
                    critical_start_sub_name, monitor_sub.name, info=edge_info_start
                )
            else:
                constraints_with_critical.edges[
                    critical_start_sub_name, monitor_sub.name
                ]["info"] = edge_info_start

            critical_deadline = self._resolve_original_critical_deadline(
                critical_start_sub_end_time=critical_start_sub_end_time,
                critical_interval_duration=critical_interval_duration,
            )

            # [Restored] '정확한 타이밍' 준수를 위해 Critical Deadline 기준으로 Interval을 재계산하여 적용합니다.
            # Fallback 상황(target_start=current)이라도 Critical Constraint가 있다면 그 시간을 지켜야 합니다.
            interval_mon_to_end = max(0.0, critical_deadline - monitor_finish_time)
            edge_info_end = {"Interval": interval_mon_to_end, "IsCritical": True}

            log.debug(
                f"[_insert_monitoring_step] Updating Edge '{monitor_sub.name}' -> '{critical_end_sub_name}'\n"
                f"  - CriticalDeadline: {critical_deadline:.2f} (StartEnd: {critical_start_sub_end_time:.2f} + Interval: {critical_interval_duration:.2f})\n"
                f"  - MonitorFinish: {monitor_finish_time:.2f}\n"
                f"  - 모니터링 끝난 시간 부터, 다음 critical subtask 시작 시간까지의 시간: {interval_mon_to_end:.2f} "
            )

            self._replace_monitoring_residual_edge(
                constraints_with_critical,
                critical_start_sub_name=critical_start_sub_name,
                critical_end_sub_name=critical_end_sub_name,
                monitoring_sub_name=monitor_sub.name,
                edge_info=edge_info_end,
            )

            updated_state = updated_state._replace(
                constraints=constraints_with_critical
            )

        return monitor_node._replace(state=updated_state)

    # -----------------------------------------------------
    # (B) 서브태스크 (with monitoring) - 정책 2 적용
    # -----------------------------------------------------
    def _expand_subtask_with_monitoring(
        self,
        curr_node: SimulationNode,
        candidate: Candidate,
        not_yet_candidates: List[Candidate],
        feasible_candidates: List[Candidate] = None,
    ) -> Optional[SimulationNode]:
        curr_state = curr_node.state
        original_task_name = candidate.subtask.name

        # Determine the critical interval that triggers this monitoring split
        scheduling_due = candidate.scheduling_due
        # 모니터링으로 쪼갤 때, critical end subtask를 가장 빠르게 deadline을 갖는 것으로 관찰하게 함. (분산 높은 것 보다...)
        if not (
            scheduling_due
            and scheduling_due.due_date != float("inf")
            and scheduling_due.due_related_sub_name
        ):
            log.debug(
                f"Candidate {original_task_name} has no valid scheduling_due. Fallback to non-monitoring."
            )
            return self._expand_subtask_wo_monitoring(
                curr_node, candidate, not_yet_candidates, feasible_candidates
            )
        critical_end_sub_name = scheduling_due.due_related_sub_name

        # Find the start of the critical interval
        incoming_constraints_to_crit_end = self.constraint_handler.get_time_slots(
            critical_end_sub_name, curr_state.constraints, "in"
        )
        critical_incoming_slots = [
            s for s in incoming_constraints_to_crit_end if s.is_critical
        ]
        if not critical_incoming_slots:
            log.debug(
                f"No incoming critical constraints for '{critical_end_sub_name}'. Fallback for {original_task_name}."
            )
            return self._expand_subtask_wo_monitoring(
                curr_node, candidate, not_yet_candidates, feasible_candidates
            )

        target_critical_slot = max(critical_incoming_slots, key=lambda s: s.interval)
        original_critical_interval_duration = target_critical_slot.interval
        critical_start_sub_name = target_critical_slot.related_subtask_name

        critical_start_completed_entry = next(
            (
                ce
                for ce in curr_state.completed_entries
                if ce.subtask.name == critical_start_sub_name
            ),
            None,
        )
        if not critical_start_completed_entry:
            log.error(
                f"CRITICAL LOGIC ERROR: start_subtask '{critical_start_sub_name}' not found. Fallback."
            )
            return self._expand_subtask_wo_monitoring(
                curr_node, candidate, not_yet_candidates, feasible_candidates
            )
        critical_start_sub_actual_end_time = (
            critical_start_completed_entry.schedule_end_time
        )

        if self._monitoring_budget_reached(
            curr_state,
            critical_start_sub_end_time=critical_start_sub_actual_end_time,
            critical_end_sub_name=critical_end_sub_name,
        ):
            log.debug(
                "Monitoring budget reached for '%s'. Falling back to non-monitoring expansion.",
                critical_end_sub_name,
            )
            return self._expand_subtask_wo_monitoring(
                curr_node, candidate, not_yet_candidates, feasible_candidates
            )

        monitoring_target_obj = next(
            (
                self._extract_first_actionable_object_id(remain_sub)
                for remain_sub in curr_node.state.remaining_subtasks
                if remain_sub.name == critical_end_sub_name
            ),
            None,
        )
        if not monitoring_target_obj:
            log.warning(
                f"Could not determine monitoring target for '{critical_end_sub_name}'. Fallback."
            )
            return self._expand_subtask_wo_monitoring(
                curr_node, candidate, not_yet_candidates, feasible_candidates
            )

        log.debug(
            f"CritStart='{critical_start_sub_name}' -> CritEnd='{critical_end_sub_name+original_task_name}' (ends {critical_start_sub_actual_end_time:.2f}: {critical_start_completed_entry.schedule_end_time}+{original_critical_interval_duration:.2f})",
        )

        # --- Refined Splitting Logic ---
        # Retrieve variance from the edge info
        edge_data = curr_state.constraints.get_edge_data(
            critical_start_sub_name, critical_end_sub_name
        )
        variance_val = INIT_PRIOR_VARIANCE
        if edge_data and "info" in edge_data:
            variance_val = edge_data["info"].get("Variance", INIT_PRIOR_VARIANCE)

        original_absolute_monitoring_trigger_time = (
            self._compute_monitoring_trigger_time(
                raw_object_name=monitoring_target_obj,
                critical_start_sub_end_time=critical_start_sub_actual_end_time,
                mean_duration=original_critical_interval_duration,
                variance=variance_val,
            )
        )
        original_critical_deadline = self._resolve_original_critical_deadline(
            critical_start_sub_end_time=critical_start_sub_actual_end_time,
            critical_interval_duration=original_critical_interval_duration,
        )

        full_candidate_action_info_check = self.action_handler.get_actions_info(
            curr_node, candidate.subtask.execution.primitive_actions
        )
        if not (
            full_candidate_action_info_check
            and full_candidate_action_info_check.success
        ):
            log.warning(
                f"Full action sim failed for candidate {original_task_name} during check.."
            )
            return self._expand_subtask_wo_monitoring(
                curr_node, candidate, not_yet_candidates, feasible_candidates
            )
        candidate_expected_completion_time_wo_split = (
            curr_state.current_time + full_candidate_action_info_check.cumulative_time
        )

        # Scenario 1: Task is "safe" and finishes before monitoring is needed.
        if (
            candidate_expected_completion_time_wo_split
            <= original_absolute_monitoring_trigger_time
        ):
            log.debug(
                f"Candidate {original_task_name} finishes before monitoring trigger ({original_absolute_monitoring_trigger_time:.2f}). Executing without split.\n"
            )
            return self._expand_subtask_wo_monitoring(
                curr_node, candidate, not_yet_candidates, feasible_candidates
            )

        # Scenario 3: Monitoring trigger falls during the task. Splitting is necessary.
        duration_for_early_sub_target = (
            original_absolute_monitoring_trigger_time - curr_state.current_time
        )

        pre_actions_log, post_actions_log, split_successful, pre_ends_holding_object = (
            self.action_handler.split_subtask_by_cutoff_time(
                curr_node,
                candidate.subtask.execution.primitive_actions,
                duration_for_early_sub_target,
            )
        )

        if not split_successful or pre_ends_holding_object:
            log.debug(
                f"Failed to split {original_task_name} with cutoff {duration_for_early_sub_target:.2f}. "
                f"Switching to Pre-Monitoring (Check-Before-Act) strategy as a fallback."
            )
            critical_deadline = self._resolve_original_critical_deadline(
                critical_start_sub_end_time=critical_start_sub_actual_end_time,
                critical_interval_duration=original_critical_interval_duration,
            )
            if self._immediate_monitoring_misses_critical_deadline(
                current_time=curr_state.current_time,
                critical_deadline=critical_deadline,
            ):
                log.debug(
                    "[_expand_subtask_with_monitoring] Immediate pre-monitor for '%s' would finish at %.2f "
                    "after critical deadline %.2f. Skipping monitoring fallback.",
                    original_task_name,
                    curr_state.current_time + MONITORING_DURATION,
                    critical_deadline,
                )
                return self._expand_subtask_wo_monitoring(
                    curr_node, candidate, not_yet_candidates, feasible_candidates
                )
            # [Safety Check 250130] Prevent redundant monitoring
            # If the immediately preceding task was ALREADY a monitoring action for the SAME object,
            # we should NOT insert another monitoring step. This prevents infinite monitoring loops.
            if self._just_monitored_target(
                curr_state,
                monitoring_target_obj=monitoring_target_obj,
                monitoring_target_sub_name=critical_end_sub_name,
            ):
                log.warning(
                    f"[_expand_subtask_with_monitoring] Redundant monitoring detected! "
                    f"Predecessor '{curr_node.state.subtask.name}' already monitored '{monitoring_target_obj}'. "
                    f"Skipping monitoring insertion and proceeding with original task."
                )
                return self._expand_subtask_wo_monitoring(
                    curr_node, candidate, not_yet_candidates, feasible_candidates
                )
            # [Fallback] 분할 실패 시, 작업을 시작하기 '전'에 미리 모니터링을 수행하는 경로를 탐색에 추가합니다.
            # 모니터링 시간만큼 작업 착수가 지연되지만, 불확실성을 해소할 수 있는 안전한 선택지입니다.
            return self._insert_monitoring_step(
                curr_node=curr_node,
                candidate=candidate,
                monitoring_target_obj=monitoring_target_obj,
                predecessor_name=curr_node.state.subtask.name,
                target_actual_start_time=curr_state.current_time,  # 즉시 수행
                not_yet_candidates=not_yet_candidates,
                critical_start_sub_name=critical_start_sub_name,
                critical_start_sub_end_time=critical_start_sub_actual_end_time,
                critical_end_sub_name=critical_end_sub_name,
                critical_interval_duration=original_critical_interval_duration,
                monitoring_target_sub_name=critical_end_sub_name,
                is_critical_link=False,
            )

        log.info(f"{pre_actions_log.total_time_used()},{pre_actions_log=}")
        log.info(f"{post_actions_log.total_time_used()},{post_actions_log=}")

        early_sub_actions = []
        actual_early_sub_duration = 0.0

        if (
            pre_actions_log
            and pre_actions_log.results
            and pre_actions_log.get_actions()
        ):
            early_sub_actions = pre_actions_log.get_actions()
            actual_early_sub_duration = pre_actions_log.total_time_used()
            log.debug(
                f"Split {original_task_name}: Initial early_actions ({len(early_sub_actions)}), actual_duration={actual_early_sub_duration:.2f}, target_duration={duration_for_early_sub_target:.2f}"
            )
        else:
            log.warning(
                f"Failed to get valid early_actions from split for {original_task_name} with cutoff {duration_for_early_sub_target:.2f}. "
                f"Executing the task without splitting as a fallback."
            )
            return self._expand_subtask_wo_monitoring(
                curr_node, candidate, not_yet_candidates, feasible_candidates
            )

        # --- Phase 3: early_sub 확장 및 실제 모니터링 시점 결정 ---
        early_sub_task = copy.deepcopy(candidate.subtask)
        early_sub_task.name = (
            f"EARLY_{original_task_name}"
            if len(post_actions_log.results) != 0
            else original_task_name
        )
        early_sub_task.execution.primitive_actions = early_sub_actions
        early_sub_task.decomposed = True

        early_candidate = Candidate(
            subtask=early_sub_task,
            scheduling_due=None,
            is_critical=candidate.is_critical,
            logical_interaction_start_time=candidate.logical_interaction_start_time,
            actual_interaction_start_time=candidate.actual_interaction_start_time,
            estimated_first_nav_duration=candidate.estimated_first_nav_duration,
        )

        log.info(
            f"  Expanding adjusted EARLY subtask: {early_sub_task.name} (actions: {len(early_sub_actions)}, initial_est_duration: {actual_early_sub_duration:.2f})"
        )
        node_after_early_sub = self._expand_subtask_wo_monitoring(
            curr_node, early_candidate, not_yet_candidates
        )

        if node_after_early_sub is None:
            log.warning(
                f"Expansion of EARLY subtask {early_sub_task.name} failed. "
                f"Executing the original task without splitting as a fallback."
            )
            return self._expand_subtask_wo_monitoring(
                curr_node, candidate, not_yet_candidates, feasible_candidates
            )

        actual_monitoring_trigger_time = node_after_early_sub.state.current_time

        log.info(
            f"  EARLY subtask {early_sub_task.name} expanded. Actual Monitoring Trigger Time: {actual_monitoring_trigger_time:.2f} "
            f"(Original trigger was: {original_absolute_monitoring_trigger_time:.2f})"
        )

        state_after_early_expansion = node_after_early_sub.state

        # --- Phase 4: mon_sub (주요 인터벌용) 및 remain_sub 생성 ---
        mon_sub_task_for_main_interval = TaskUtil.create_monitoring_subtask(
            name=f"{critical_end_sub_name}",
            obj=monitoring_target_obj,
        )
        log.debug(f"mon_sub_task_for_main_interval: {mon_sub_task_for_main_interval}")
        mon_sub_task_for_main_interval.decomposed = True

        remain_sub_actions = (
            post_actions_log.get_actions()
            if post_actions_log and post_actions_log.results
            else []
        )
        if remain_sub_actions:
            first_action = remain_sub_actions[0]
            if not first_action.startswith("NAVIGATE_TO"):
                # Try to infer a reasonable target object id from actions
                def _extract_target_id(actions: List[str]) -> Optional[str]:
                    for act in actions:
                        parts = act.split()
                        if len(parts) > 1:
                            return parts[1]
                    return None

                target_id = _extract_target_id(remain_sub_actions)
                if target_id and target_id in curr_state.scene_positions:
                    remain_sub_actions = [
                        f"NAVIGATE_TO {target_id}"
                    ] + remain_sub_actions

        if remain_sub_actions:
            remain_subtask = copy.deepcopy(candidate.subtask)
            remain_subtask.name = f"REMAIN_{original_task_name}"
            remain_subtask.execution.primitive_actions = remain_sub_actions
            remain_subtask.decomposed = True
            log.debug(
                f"Prepared REMAIN subtask: {remain_subtask.name} with {len(remain_sub_actions)} actions."
            )

        # --- Phase 5: 제약 조건 그래프 및 remaining_subtasks 업데이트 ---
        new_constraints_graph = copy.deepcopy(state_after_early_expansion.constraints)

        original_task_in_edges_data = []
        original_task_out_edges_data = []
        if new_constraints_graph.has_node(original_task_name):
            original_task_in_edges_data = list(
                new_constraints_graph.in_edges(original_task_name, data=True)
            )
            original_task_out_edges_data = list(
                new_constraints_graph.out_edges(original_task_name, data=True)
            )
            new_constraints_graph.remove_node(original_task_name)
            log.debug(f"Removed node '{original_task_name}' from constraints graph.")

        if not new_constraints_graph.has_node(early_sub_task.name):
            new_constraints_graph.add_node(early_sub_task.name)
            log.debug(f"Node for EARLY subtask '{early_sub_task.name}' added to graph.")
        if not new_constraints_graph.has_node(mon_sub_task_for_main_interval.name):
            new_constraints_graph.add_node(mon_sub_task_for_main_interval.name)
        if remain_subtask and not new_constraints_graph.has_node(remain_subtask.name):
            new_constraints_graph.add_node(remain_subtask.name)

        for pred_name, _, data in original_task_in_edges_data:
            if pred_name not in [
                early_sub_task.name,
                mon_sub_task_for_main_interval.name,
                (remain_subtask.name if remain_subtask else ""),
                original_task_name,
                # critical_start_sub_name,
            ]:
                edge_info = copy.deepcopy(data.get("info", {}))
                if not new_constraints_graph.has_edge(pred_name, early_sub_task.name):
                    new_constraints_graph.add_edge(
                        pred_name, early_sub_task.name, info=edge_info
                    )
                    log.debug(
                        f"Rerouted incoming constraint from '{pred_name}' to '{early_sub_task.name}'."
                    )

        source_for_outgoing_edges = (
            remain_subtask.name
            if remain_subtask
            else mon_sub_task_for_main_interval.name
        )
        for _, succ_name, data in original_task_out_edges_data:
            if succ_name not in [
                early_sub_task.name,
                mon_sub_task_for_main_interval.name,
                (remain_subtask.name if remain_subtask else ""),
                original_task_name,
                # critical_end_sub_name,
            ]:
                edge_info = copy.deepcopy(data.get("info", {}))
                if not new_constraints_graph.has_edge(
                    source_for_outgoing_edges, succ_name
                ):
                    new_constraints_graph.add_edge(
                        source_for_outgoing_edges, succ_name, info=edge_info
                    )
                    log.debug(
                        f"Rerouted outgoing constraint from '{source_for_outgoing_edges}' to '{succ_name}'."
                    )

        info_early_to_mon = {"Interval": 0.0, "IsCritical": True}
        if not new_constraints_graph.has_edge(
            early_sub_task.name, mon_sub_task_for_main_interval.name
        ):
            new_constraints_graph.add_edge(
                early_sub_task.name,
                mon_sub_task_for_main_interval.name,
                info=info_early_to_mon,
            )
            log.debug(
                f"Added internal constraint: '{early_sub_task.name}' -> '{mon_sub_task_for_main_interval.name}'."
            )

        if remain_subtask:
            info_mon_to_remain = {"Interval": 0.0, "IsCritical": False}
            if not new_constraints_graph.has_edge(
                mon_sub_task_for_main_interval.name, remain_subtask.name
            ):
                new_constraints_graph.add_edge(
                    mon_sub_task_for_main_interval.name,
                    remain_subtask.name,
                    info=info_mon_to_remain,
                )
                log.debug(
                    f"Added internal constraint: '{mon_sub_task_for_main_interval.name}' -> '{remain_subtask.name}'."
                )

        interval_crit_start_to_mon = (
            actual_monitoring_trigger_time - critical_start_sub_actual_end_time
        )
        info_crit_start_to_mon = {
            "Interval": max(0.0, interval_crit_start_to_mon),
            "IsCritical": True,
        }
        if not new_constraints_graph.has_edge(
            critical_start_sub_name, mon_sub_task_for_main_interval.name
        ):
            new_constraints_graph.add_edge(
                critical_start_sub_name,
                mon_sub_task_for_main_interval.name,
                info=info_crit_start_to_mon,
            )
        else:
            new_constraints_graph.edges[
                critical_start_sub_name, mon_sub_task_for_main_interval.name
            ]["info"].update(info_crit_start_to_mon)
        log.debug(
            f"Added/Updated main monitoring constraint: '{critical_start_sub_name}' -> '{mon_sub_task_for_main_interval.name}', Interval: {info_crit_start_to_mon['Interval']:.2f}."
        )

        mon_sub_expected_completion_time = (
            actual_monitoring_trigger_time + MONITORING_DURATION
        )

        interval_mon_to_crit_end = (
            original_critical_deadline - mon_sub_expected_completion_time
        )
        info_mon_to_crit_end = {
            "Interval": max(0.0, interval_mon_to_crit_end),
            "IsCritical": True,
        }

        # [DEBUG LOG] Check Interval Update in _expand_subtask_with_monitoring
        prev_interval_sub = "N/A"
        if new_constraints_graph.has_edge(
            mon_sub_task_for_main_interval.name, critical_end_sub_name
        ):
            prev_interval_sub = (
                new_constraints_graph.edges[
                    mon_sub_task_for_main_interval.name, critical_end_sub_name
                ]
                .get("info", {})
                .get("Interval", "N/A")
            )

        log.debug(
            f"[DEBUG _expand_subtask_with_monitoring] Updating Edge '{mon_sub_task_for_main_interval.name}' -> '{critical_end_sub_name}'\n"
            f"  - CritEndDeadline: {original_critical_deadline:.2f}\n"
            f"  - MonitorFinish: {mon_sub_expected_completion_time:.2f}\n"
            f"  - Calc Interval: {interval_mon_to_crit_end:.2f} (Prev: {prev_interval_sub})"
        )

        self._replace_monitoring_residual_edge(
            new_constraints_graph,
            critical_start_sub_name=critical_start_sub_name,
            critical_end_sub_name=critical_end_sub_name,
            monitoring_sub_name=mon_sub_task_for_main_interval.name,
            edge_info=info_mon_to_crit_end,
        )

        # Verify update
        check_interval_sub = new_constraints_graph.edges[
            mon_sub_task_for_main_interval.name, critical_end_sub_name
        ]["info"]["Interval"]
        log.debug(f"  -> Update Verified: {check_interval_sub:.2f}")
        log.debug(
            f"Added/Updated main monitoring constraint: '{mon_sub_task_for_main_interval.name}' -> '{critical_end_sub_name}', Interval: {info_mon_to_crit_end['Interval']:.2f}."
        )

        remaining_after_early_executed = list(
            state_after_early_expansion.remaining_subtasks
        )
        final_remaining_subtasks_list = [
            r for r in remaining_after_early_executed if r.name != original_task_name
        ]

        if mon_sub_task_for_main_interval.name not in {
            r.name for r in final_remaining_subtasks_list
        }:
            final_remaining_subtasks_list.append(mon_sub_task_for_main_interval)
        if remain_subtask and remain_subtask.name not in {
            r.name for r in final_remaining_subtasks_list
        }:
            final_remaining_subtasks_list.append(remain_subtask)

        log.debug(
            f"Updated remaining subtasks. Added mon: {mon_sub_task_for_main_interval.name}, remain: {remain_subtask.name if remain_subtask else 'None'}"
        )
        log.debug(
            f"  Final remaining: {[r.name for r in final_remaining_subtasks_list]}"
        )

        updated_final_state = SchedulerState(
            subtask=state_after_early_expansion.subtask,
            completed_entries=state_after_early_expansion.completed_entries,
            remaining_subtasks=final_remaining_subtasks_list,
            constraints=new_constraints_graph,
            current_time=actual_monitoring_trigger_time,
            scene_positions=state_after_early_expansion.scene_positions,
            held_object=state_after_early_expansion.held_object,
        )

        return node_after_early_sub._replace(state=updated_final_state)

    # -----------------------------------------------------
    # (C) Wait expansions
    # -----------------------------------------------------

    def _expand_wait_with_monitoring(
        self,
        curr_node: SimulationNode,
        candidate: Candidate,
        not_yet_candidates: List[Candidate],
        nav_duration: float = 0.0,
        feasible_candidates: List[Candidate] = None,
        selected_obligation: Optional[MonitoringObligation] = None,
    ) -> Optional[SimulationNode]:
        """
        Performs monitoring and, if needed, a follow-up wait until the
        candidate's actual_interaction_start_time.
        """
        # candidate는 wait for의 대상 subtask임.
        # candidate로 만드는 wait for candidate의 scheduling_due는 candidate의 logical_interaction_start_time 기준으로 계산됨.
        log.debug(
            f"[_expand_wait_with_monitoring] Waiting for {candidate.subtask.name}"
        )

        if self._wait_is_disallowed_for_candidate(curr_node, candidate):
            log.debug(
                "[_expand_wait_with_monitoring] Wait is disallowed for '%s'. Skipping wait-with-monitoring branch.",
                candidate.subtask.name,
            )
            return None

        curr_state = curr_node.state

        critical_end_sub_name = candidate.subtask.name
        if (
            candidate.scheduling_due is not None
            and candidate.scheduling_due.due_related_sub_name
        ):
            critical_end_sub_name = candidate.scheduling_due.due_related_sub_name

        # Find the start of the critical interval
        incoming_constraints_to_crit_end = self.constraint_handler.get_time_slots(
            critical_end_sub_name, curr_state.constraints, "in"
        )
        critical_incoming_slots = [
            s for s in incoming_constraints_to_crit_end if s.is_critical
        ]
        if not critical_incoming_slots:
            log.debug(
                f"No incoming critical constraints for '{critical_end_sub_name}'. Fallback for {candidate.subtask.name}."
            )
            return self._expand_subtask_wo_monitoring(
                curr_node, candidate, not_yet_candidates, feasible_candidates
            )

        target_critical_slot = max(critical_incoming_slots, key=lambda s: s.interval)
        original_critical_interval_duration = target_critical_slot.interval
        critical_start_sub_name = target_critical_slot.related_subtask_name

        critical_start_completed_entry = next(
            (
                ce
                for ce in curr_state.completed_entries
                if ce.subtask.name == critical_start_sub_name
            ),
            None,
        )
        if not critical_start_completed_entry:
            log.error(
                f"CRITICAL LOGIC ERROR: start_subtask '{critical_start_sub_name}' not found. Fallback."
            )
            return self._expand_subtask_wo_monitoring(
                curr_node, candidate, not_yet_candidates, feasible_candidates
            )
        critical_start_sub_actual_end_time = (
            critical_start_completed_entry.schedule_end_time
        )

        if self._monitoring_budget_reached(
            curr_state,
            critical_start_sub_end_time=critical_start_sub_actual_end_time,
            critical_end_sub_name=critical_end_sub_name,
        ):
            log.debug(
                "Monitoring budget reached for '%s'. Falling back to wait-without-monitoring.",
                critical_end_sub_name,
            )
            return self._expand_wait_wo_monitoring(
                curr_node,
                candidate,
                not_yet_candidates,
                nav_duration=nav_duration,
                feasible_candidates=feasible_candidates,
            )
        monitoring_target_obj = self._get_monitoring_target_obj_for_subtask_name(
            curr_node,
            critical_end_sub_name,
        )
        if monitoring_target_obj is None:
            return self._expand_wait_wo_monitoring(
                curr_node,
                candidate,
                not_yet_candidates,
                nav_duration=nav_duration,
                feasible_candidates=feasible_candidates,
            )

        if selected_obligation is None:
            edge_data = curr_state.constraints.get_edge_data(
                critical_start_sub_name, critical_end_sub_name
            )
            variance_val = INIT_PRIOR_VARIANCE
            if edge_data and "info" in edge_data:
                variance_val = edge_data["info"].get("Variance", INIT_PRIOR_VARIANCE)
            original_absolute_monitoring_trigger_time = (
                self._compute_monitoring_trigger_time(
                    raw_object_name=monitoring_target_obj,
                    critical_start_sub_end_time=critical_start_sub_actual_end_time,
                    mean_duration=original_critical_interval_duration,
                    variance=variance_val,
                )
            )
        else:
            original_absolute_monitoring_trigger_time = selected_obligation.trigger_time

        target_start_time = (
            self._get_candidate_target_start_time(candidate)
            or candidate.logical_interaction_start_time
        )
        if target_start_time is None:
            return self._expand_wait_wo_monitoring(
                curr_node,
                candidate,
                not_yet_candidates,
                nav_duration=nav_duration,
                feasible_candidates=feasible_candidates,
            )

        original_critical_deadline = self._resolve_original_critical_deadline(
            selected_obligation=selected_obligation,
            critical_start_sub_end_time=critical_start_sub_actual_end_time,
            critical_interval_duration=original_critical_interval_duration,
        )
        latest_safe_target_start_time = self._compute_latest_safe_monitoring_start_time(
            target_interaction_start_time=target_start_time,
        )
        latest_safe_deadline_start_time = (
            self._compute_latest_safe_monitoring_start_time(
                target_interaction_start_time=original_critical_deadline,
            )
        )
        latest_safe_monitoring_start_time = min(
            latest_safe_target_start_time,
            latest_safe_deadline_start_time,
        )
        effective_monitoring_start_time = min(
            original_absolute_monitoring_trigger_time,
            latest_safe_monitoring_start_time,
        )
        if effective_monitoring_start_time < original_absolute_monitoring_trigger_time:
            log.debug(
                "[_expand_wait_with_monitoring] Clamped monitor start for '%s' from %.2f to latest safe %.2f "
                "(target_start=%.2f, critical_deadline=%.2f).",
                candidate.subtask.name,
                original_absolute_monitoring_trigger_time,
                effective_monitoring_start_time,
                target_start_time,
                original_critical_deadline,
            )

        actual_monitoring_start_time = max(
            curr_state.current_time + nav_duration,
            effective_monitoring_start_time,
        )
        if not self._monitoring_finishes_before_critical_deadline(
            monitor_start_time=actual_monitoring_start_time,
            critical_deadline=original_critical_deadline,
        ):
            log.debug(
                "[_expand_wait_with_monitoring] Monitoring for '%s' would finish at %.2f after original critical deadline %.2f. "
                "Falling back to wait-without-monitoring.",
                candidate.subtask.name,
                actual_monitoring_start_time + MONITORING_DURATION,
                original_critical_deadline,
            )
            return self._expand_wait_wo_monitoring(
                curr_node,
                candidate,
                not_yet_candidates,
                nav_duration=nav_duration,
                feasible_candidates=feasible_candidates,
            )

        # If we are already beyond the latest feasible monitor point, this branch
        # can no longer monitor without sacrificing the critical end.
        if (
            curr_state.current_time + MONITORING_DURATION + nav_duration
            > target_start_time + EPSILON
        ):
            return self._expand_wait_wo_monitoring(
                curr_node,
                candidate,
                not_yet_candidates,
                nav_duration=nav_duration,
                feasible_candidates=feasible_candidates,
            )

        idle_wait_duration = max(
            0.0,
            effective_monitoring_start_time - curr_state.current_time - nav_duration,
        )
        total_wait_duration = nav_duration + idle_wait_duration

        if total_wait_duration <= EPSILON:
            if self._just_monitored_target(
                curr_state,
                monitoring_target_obj=monitoring_target_obj,
                monitoring_target_sub_name=critical_end_sub_name,
            ):
                log.debug(
                    "[_expand_wait_with_monitoring] Zero-duration wait for '%s' immediately after monitoring the same target. "
                    "Falling back to wait-without-monitoring.",
                    candidate.subtask.name,
                )
                return self._expand_wait_wo_monitoring(
                    curr_node,
                    candidate,
                    not_yet_candidates,
                    nav_duration=nav_duration,
                    feasible_candidates=feasible_candidates,
                )

            log.debug(
                "[_expand_wait_with_monitoring] Zero-duration wait for '%s'. "
                "Skipping synthetic WAIT node and inserting immediate monitoring.",
                candidate.subtask.name,
            )
            inserted_monitor_node = self._insert_monitoring_step(
                curr_node=curr_node,
                candidate=candidate,
                monitoring_target_obj=monitoring_target_obj,
                predecessor_name=(
                    curr_state.subtask.name
                    if curr_state.subtask is not None
                    else critical_start_sub_name
                ),
                target_actual_start_time=(
                    self._get_candidate_target_start_time(candidate)
                    or curr_state.current_time
                ),
                not_yet_candidates=not_yet_candidates,
                critical_start_sub_name=critical_start_sub_name,
                critical_start_sub_end_time=critical_start_sub_actual_end_time,
                critical_end_sub_name=critical_end_sub_name,
                critical_interval_duration=original_critical_interval_duration,
                monitoring_target_sub_name=critical_end_sub_name,
                is_critical_link=True,
            )
            if inserted_monitor_node is not None:
                return inserted_monitor_node
            log.debug(
                "[_expand_wait_with_monitoring] Immediate monitoring fallback for '%s' was rejected. "
                "Falling back to wait-without-monitoring.",
                candidate.subtask.name,
            )
            return self._expand_wait_wo_monitoring(
                curr_node,
                candidate,
                not_yet_candidates,
                nav_duration=nav_duration,
                feasible_candidates=feasible_candidates,
            )

        simulated_wait = self._simulate_wait_subtask(
            curr_node,
            candidate,
            idle_wait_duration=idle_wait_duration,
            nav_duration=nav_duration,
        )
        if simulated_wait is None:
            return None
        wait_sub, _completed_entry, wait_state = simulated_wait
        wait_end_time = wait_state.current_time

        # Global Risk Check을 위해 feasible_candidates도 포함하여 전달
        all_candidates = not_yet_candidates
        if feasible_candidates:
            all_candidates = feasible_candidates + not_yet_candidates

        monitoring_sub = TaskUtil.create_monitoring_subtask(
            name=f"{critical_end_sub_name}",
            obj=monitoring_target_obj,
        )
        monitoring_sub.decomposed = True
        setattr(wait_sub, "_forced_followup_monitoring_name", monitoring_sub.name)

        # --- Phase 5: 제약 조건 그래프 및 remaining_subtasks 업데이트 ---
        # 없던 wait subtask가 생긴 상황, 모니터링 task도 만들었음. 제약 조건으로 연결해야 함.
        # wait는 0,True로 monitoring과 연결짓고, candidate와 monitoring은 interval, True만큼 업데이트 시켜야 함
        new_constraints_graph = copy.deepcopy(wait_state.constraints)

        if not new_constraints_graph.has_node(wait_sub.name):
            new_constraints_graph.add_node(wait_sub.name)
            log.debug(f"Node for WAIT subtask '{wait_sub.name}' added to graph.")
        if not new_constraints_graph.has_node(monitoring_sub.name):
            new_constraints_graph.add_node(monitoring_sub.name)
            log.debug(
                f"Node for MONITORING subtask '{monitoring_sub.name}' added to graph."
            )

        edge_from_wait_to_mon = {"Interval": 0.0, "IsCritical": True}
        if not new_constraints_graph.has_edge(wait_sub.name, monitoring_sub.name):
            new_constraints_graph.add_edge(
                wait_sub.name,
                monitoring_sub.name,
                info=edge_from_wait_to_mon,
            )
            log.debug(
                f"Added wait to monitoring constraint: '{wait_sub.name}' -> '{monitoring_sub.name} ({edge_from_wait_to_mon['Interval']:.2f}, {edge_from_wait_to_mon['IsCritical']}).'."
            )

        interval_crit_start_to_mon = wait_end_time - critical_start_sub_actual_end_time
        info_crit_start_to_mon = {
            "Interval": max(0.0, interval_crit_start_to_mon),
            "IsCritical": True,
        }

        if not new_constraints_graph.has_edge(
            critical_start_sub_name, monitoring_sub.name
        ):
            new_constraints_graph.add_edge(
                critical_start_sub_name,
                monitoring_sub.name,
                info=info_crit_start_to_mon,
            )
        else:
            new_constraints_graph.edges[critical_start_sub_name, monitoring_sub.name][
                "info"
            ].update(info_crit_start_to_mon)
        log.debug(
            f"Added/Updated main monitoring constraint: '{critical_start_sub_name}' -> '{monitoring_sub.name}', Interval: {info_crit_start_to_mon['Interval']:.2f}."
        )

        critical_end_sub_logical_start_time = self._resolve_original_critical_deadline(
            selected_obligation=selected_obligation,
            critical_start_sub_end_time=critical_start_sub_actual_end_time,
            critical_interval_duration=original_critical_interval_duration,
        )
        monitoring_sub_expected_completion_time = wait_end_time + MONITORING_DURATION

        interval_mon_to_crit_end = (
            critical_end_sub_logical_start_time
            - monitoring_sub_expected_completion_time
        )
        info_mon_to_crit_end = {
            "Interval": max(0.0, interval_mon_to_crit_end),
            "IsCritical": True,
        }

        wait_candidate = Candidate(
            subtask=wait_sub,
            is_critical=candidate.is_critical,
            scheduling_due=candidate.scheduling_due,
        )

        # [DEBUG LOG] Check Interval Update in _expand_subtask_with_monitoring
        prev_interval_sub = "N/A"
        if new_constraints_graph.has_edge(monitoring_sub.name, critical_end_sub_name):
            prev_interval_sub = (
                new_constraints_graph.edges[monitoring_sub.name, critical_end_sub_name]
                .get("info", {})
                .get("Interval", "N/A")
            )

        log.debug(
            f"[DEBUG _expand_subtask_with_monitoring] Updating Edge '{monitoring_sub.name}' -> '{critical_end_sub_name}'\n"
            f"  - CritEndDeadline: {critical_end_sub_logical_start_time:.2f}\n"
            f"  - MonitorFinish: {monitoring_sub_expected_completion_time:.2f}\n"
            f"  - Mon -> CritEnd Interval: {interval_mon_to_crit_end:.2f} (Prev: {prev_interval_sub})"
        )

        self._replace_monitoring_residual_edge(
            new_constraints_graph,
            critical_start_sub_name=critical_start_sub_name,
            critical_end_sub_name=critical_end_sub_name,
            monitoring_sub_name=monitoring_sub.name,
            edge_info=info_mon_to_crit_end,
        )

        # Verify update
        check_interval_sub = new_constraints_graph.edges[
            monitoring_sub.name, critical_end_sub_name
        ]["info"]["Interval"]
        log.debug(f"  -> Update Verified: {check_interval_sub:.2f}")
        log.debug(
            f"Added/Updated main monitoring constraint: '{monitoring_sub.name}' -> '{critical_end_sub_name}', Interval: {info_mon_to_crit_end['Interval']:.2f}."
        )

        new_state = SchedulerState(
            subtask=wait_state.subtask,
            completed_entries=wait_state.completed_entries,
            remaining_subtasks=list(wait_state.remaining_subtasks) + [monitoring_sub],
            constraints=new_constraints_graph,
            current_time=wait_end_time,
            scene_positions=wait_state.scene_positions,
            held_object=wait_state.held_object,
        )

        temp_node = SimulationNode(
            parent_node=curr_node,
            heuristic_cost=0.0,
            depth=curr_node.depth + 1,
            tie_breaker=curr_node.tie_breaker,
            state=new_state,
            risk_level=curr_node.risk_level,
        )
        step_risk, total_heuristic_cost = self.cost_calculator.calc_heuristic(
            temp_node,
            wait_candidate,
            all_candidates,
            # Wait action creates delay. We must check if this delay hurts ANY feasible or not_yet task.
        )

        new_cost = wait_end_time + total_heuristic_cost

        # Accumulate max risk level
        new_risk = max(curr_node.risk_level, step_risk)

        log.info(
            f"[_expand_wait_with_monitoring] Waiting for{candidate.subtask.name}: end_time({wait_end_time:.2f}) = current_time({curr_state.current_time:.2f}) + wait_duration({total_wait_duration:.2f})"
        )
        log.info(
            f"cost({new_cost:.2f}) = end_time({wait_end_time:.2f}) + total_heuristic_cost({total_heuristic_cost:.2f})\n"
        )

        return SimulationNode(
            parent_node=curr_node,
            heuristic_cost=new_cost,
            depth=curr_node.depth + 1,
            tie_breaker=next(self._counter),
            state=new_state,
            risk_level=new_risk,
        )

    def _expand_wait_wo_monitoring(
        self,
        curr_node: SimulationNode,
        candidate: Candidate,
        not_yet_candidates: List[Candidate],
        nav_duration: float = 0.0,
        feasible_candidates: List[Candidate] = None,
        max_wait_duration: Optional[float] = None,
        additional_delay: float = 0.0,
    ) -> Optional[SimulationNode]:
        """
        Inserts a single "Wait" action until the candidate's actual_interaction_start_time.

        - If actual_interaction_start_time <= current_time, wait_duration becomes 0.
        - This wait is modeled as a Subtask with type="Wait".


        Args:
            curr_node (SimulationNode): Current node in the search tree.
            candidate (Candidate): The candidate subtask we're waiting for.
            nav_duration (float): Estimated navigation duration to the target object.
            additional_delay (float): Extra wait time to avoid conflicts (used for Conflict-Avoidance Wait).

        Returns:
            SimulationNode: The child node representing the new state after waiting.
            모니터링 시점까지 대기 혹은 Earliest start time의 candidate 시작 시점까지 대기
        """
        curr_state = curr_node.state
        depth = curr_node.depth

        if self._wait_is_disallowed_for_candidate(curr_node, candidate):
            log.debug(
                "[_expand_wait_wo_monitoring] Wait is disallowed for '%s'. Skipping wait branch.",
                candidate.subtask.name,
            )
            return None

        target_start_time = (
            candidate.actual_interaction_start_time
            if candidate.actual_interaction_start_time is not None
            else (
                candidate.logical_interaction_start_time
                if candidate.logical_interaction_start_time is not None
                else curr_state.current_time
            )
        )

        # [Added 250130] Support Conflict-Avoidance Wait
        # If additional_delay is provided, push the target start time further.
        if additional_delay > 0:
            target_start_time += additional_delay

        if max_wait_duration is not None:
            target_start_time = min(
                target_start_time, curr_state.current_time + max_wait_duration
            )

        # Calculate staged wait duration (move early, then idle if necessary).
        idle_wait_duration = max(
            0.0, target_start_time - curr_state.current_time - nav_duration
        )
        total_wait_duration = nav_duration + idle_wait_duration

        log.debug(
            f"[_expand_wait_wo_monitoring] Check for {candidate.subtask.name}:\n"
            f"  Current Time: {curr_state.current_time:.2f}\n"
            f"  Target Start (Est): {target_start_time:.2f}\n"
            f"  Nav Duration: {nav_duration:.2f}\n"
            f"  -> Calculated Idle Wait Duration: {idle_wait_duration:.2f}\n"
            f"  -> Calculated Staged Wait Duration: {total_wait_duration:.2f}"
        )

        if total_wait_duration <= EPSILON:
            log.debug(
                f"[_expand_wait_wo_monitoring] Total wait duration ({total_wait_duration:.2f}) is less than or equal to EPSILON. Skip waiting."
            )
            return None

        simulated_wait = self._simulate_wait_subtask(
            curr_node,
            candidate,
            idle_wait_duration=idle_wait_duration,
            nav_duration=nav_duration,
        )
        if simulated_wait is None:
            return None
        wait_sub, _completed_entry, new_state = simulated_wait
        end_time = new_state.current_time

        wait_candidate = Candidate(
            subtask=wait_sub,
            is_critical=candidate.is_critical,
            scheduling_due=candidate.scheduling_due,
        )

        # Global Risk Check을 위해 feasible_candidates도 포함하여 전달
        all_candidates = not_yet_candidates
        if feasible_candidates:
            all_candidates = feasible_candidates + not_yet_candidates

        temp_node = SimulationNode(
            parent_node=curr_node,
            heuristic_cost=0.0,
            depth=curr_node.depth + 1,
            tie_breaker=curr_node.tie_breaker,
            state=new_state,
            risk_level=curr_node.risk_level,
        )
        step_risk, total_heuristic_cost = self.cost_calculator.calc_heuristic(
            temp_node,
            wait_candidate,
            all_candidates,
            # Wait action creates delay. We must check if this delay hurts ANY feasible or not_yet task.
        )
        # [Modified] HeuristicManager returns h(n) (Remaining Work + Debt).
        # We add end_time (g(n)) here to get the total cost f(n) = g(n) + h(n).
        new_cost = end_time + total_heuristic_cost

        # Accumulate max risk level
        new_risk = max(curr_node.risk_level, step_risk)

        log.info(
            f"[_expand_wait_wo_monitoring] {candidate.subtask.name}: end_time({end_time:.2f}) = current_time({curr_state.current_time:.2f}) + wait_duration({total_wait_duration:.2f})"
        )
        log.info(
            f"cost({new_cost:.2f}) = end_time({end_time:.2f}) + total_heuristic_cost({total_heuristic_cost:.2f})\n"
        )

        return SimulationNode(
            parent_node=curr_node,
            heuristic_cost=new_cost,
            depth=depth + 1,
            tie_breaker=next(self._counter),
            state=new_state,
            risk_level=new_risk,
        )
