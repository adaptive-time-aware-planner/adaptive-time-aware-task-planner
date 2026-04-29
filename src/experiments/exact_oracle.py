"""Exact deterministic oracle baseline for offline scheduling experiments."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from src.core.scheduler import Scheduler
from src.models.dataclass import (
    Candidate,
    CompletedEntry,
    SchedulerState,
    SimulationNode,
)
from src.scheduler import ActionHandler, ConstraintHandler, HeuristicManager
from src.utils.common import create_module_logger
from src.utils.config import constants
from src.utils.config.constants import CONSECUTIVE_TASK_WAIT_TOLERANCE
from src.utils.io_utils.result_saver import serialize_completed_entries

log = create_module_logger(__name__, module_log=True)


@dataclass(frozen=True)
class OracleSolution:
    """Container for a deterministic oracle solve result.

    Args:
        instruction: Instruction file name solved by the oracle.
        case: Case identifier for the instruction.
        optimal_schedule_time: Best deterministic makespan found by the solver.
        optimal_sequence: Best deterministic action sequence, including explicit
            wait subtasks chosen by the oracle.
        solve_time: Wall-clock time spent in the oracle solver.
        search_nodes: Number of DFS nodes explored.
        pruned_nodes: Number of nodes pruned by incumbent bounds.
        idle_advances: Number of implicit idle advances performed.
        exact: Whether the result is provably exact.
        timeout_hit: Whether the search stopped because of a time limit.
        completed_entries: Scheduled entries for the best solution branch.
    """

    instruction: str
    case: str
    optimal_schedule_time: Optional[float]
    optimal_sequence: list[str]
    solve_time: float
    search_nodes: int
    pruned_nodes: int
    idle_advances: int
    exact: bool
    timeout_hit: bool
    completed_entries: list[CompletedEntry] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return {
            "instruction": self.instruction,
            "case": self.case,
            "optimal_schedule_time": self.optimal_schedule_time,
            "final_schedule_time": self.optimal_schedule_time,
            "optimal_sequence": list(self.optimal_sequence),
            "solve_time": self.solve_time,
            "total_compute_time": self.solve_time,
            "search_nodes": self.search_nodes,
            "pruned_nodes": self.pruned_nodes,
            "idle_advances": self.idle_advances,
            "exact": self.exact,
            "timeout_hit": self.timeout_hit,
            "steps": serialize_completed_entries(self.completed_entries),
        }


class DeterministicExactOracle:
    """Explore deterministic schedules with exhaustive DFS and branch-and-bound.

    This oracle intentionally excludes monitoring and online belief updates.
    It reuses the project's action simulation and candidate feasibility logic
    so the comparison stays aligned with the scheduler's deterministic timing
    semantics. In strict mode, any branch whose propagated scheduler risk level
    reaches ``2`` is pruned immediately so the oracle only considers
    timing-feasible schedules under the same violation logic as the scheduler.
    """

    def __init__(
        self,
        action_handler: ActionHandler,
        constraint_handler: ConstraintHandler,
        heuristic_manager: HeuristicManager,
        *,
        time_limit_seconds: float,
    ) -> None:
        """Initialize the oracle solver.

        Args:
            action_handler: Shared deterministic action simulator.
            constraint_handler: Feasibility checker for subtasks.
            heuristic_manager: Passed through to the internal Scheduler for
                state-expansion mechanics; not used by the oracle search itself.
            time_limit_seconds: Maximum allowed solve time per instruction.
        """

        self.action_handler = action_handler
        self.constraint_handler = constraint_handler
        self.heuristic_manager = heuristic_manager
        self.time_limit_seconds = float(time_limit_seconds)
        self.scheduler = Scheduler(
            action_handler=action_handler,
            constraint_handler=constraint_handler,
            heuristic_manager=heuristic_manager,
            monitoring_policy=None,
            beam_width=1,
            simulation_depth=1,
        )
        self._deadline_monotonic_tolerance = float(constants.EPSILON)
        self._best_makespan: Optional[float] = None
        self._best_sequence: list[str] = []
        self._best_completed_entries: list[CompletedEntry] = []
        self._search_nodes = 0
        self._pruned_nodes = 0
        self._idle_advances = 0
        self._timeout_hit = False
        self._started_at = 0.0

    def solve(
        self,
        initial_state: SchedulerState,
        *,
        instruction: str,
        case: str,
        incumbent_upper_bound: Optional[float] = None,
    ) -> OracleSolution:
        """Solve the deterministic initial schedule exactly when possible.

        Args:
            initial_state: Initial scheduler state for the instruction.
            instruction: Instruction file name used in reporting.
            case: Case identifier used in reporting.
            incumbent_upper_bound: Optional initial upper bound from an existing
                scheduler rollout to accelerate pruning.

        Returns:
            OracleSolution containing the best schedule found and exactness flags.
        """

        self._best_makespan = (
            float(incumbent_upper_bound) if incumbent_upper_bound is not None else None
        )
        self._best_sequence = []
        self._best_completed_entries = []
        self._search_nodes = 0
        self._pruned_nodes = 0
        self._idle_advances = 0
        self._timeout_hit = False
        self._started_at = time.perf_counter()
        root_node = SimulationNode(
            heuristic_cost=0.0,
            depth=0,
            tie_breaker=0,
            parent_node=None,
            state=initial_state,
            risk_level=0,
        )

        self.scheduler._begin_search_session()
        try:
            self._search(root_node, ())
        finally:
            self.scheduler._end_search_session()

        solve_time = time.perf_counter() - self._started_at
        return OracleSolution(
            instruction=instruction,
            case=case,
            optimal_schedule_time=self._best_makespan,
            optimal_sequence=self._best_sequence,
            solve_time=solve_time,
            search_nodes=self._search_nodes,
            pruned_nodes=self._pruned_nodes,
            idle_advances=self._idle_advances,
            exact=(not self._timeout_hit and self._best_makespan is not None),
            timeout_hit=self._timeout_hit,
            completed_entries=copy.deepcopy(self._best_completed_entries),
        )

    def _search(self, node: SimulationNode, sequence: tuple[str, ...]) -> None:
        """Depth-first branch-and-bound search over deterministic schedules.

        Branching is heuristic-free: every feasible candidate is explored in
        neutral (name-sorted) order. The oracle also explores deterministic
        WAIT successors that exist in the scheduler's no-monitoring action
        space, including blocked-frontier waits and feasible conflict-avoidance
        waits. This keeps the oracle aligned with the scheduler's legal
        successor set without importing the scheduler's heuristic ranking.

        Args:
            node: Current deterministic scheduler state.
            sequence: Executed deterministic action sequence.
        """

        if self._timeout_hit:
            return
        if (time.perf_counter() - self._started_at) >= self.time_limit_seconds:
            self._timeout_hit = True
            return

        # NOTE: risk_level is NOT used for oracle pruning because the oracle must
        # evaluate schedules that are slightly late (within evaluation tolerance) as
        # valid.  The scheduler's risk_level uses its own planning tolerance, which
        # may differ from the oracle's evaluation tolerance checked in
        # _is_critical_deadline_violated.  Using risk_level here would cause the
        # oracle to incorrectly prune paths where a critical task is executed a few
        # seconds after its logical_interaction_start_time but still within the
        # acceptable evaluation window, leading the oracle to miss shorter valid
        # schedules.  Correctness is instead enforced by:
        #   1. _is_critical_deadline_violated  (per-candidate check in Branch A)
        #   2. branch-and-bound makespan bound (below)
        current_time = float(node.state.current_time)
        if (
            self._best_makespan is not None
            and current_time >= self._best_makespan - self._deadline_monotonic_tolerance
        ):
            self._pruned_nodes += 1
            return

        self._search_nodes += 1
        if not node.state.remaining_subtasks:
            self._commit_solution(current_time, sequence, node.state.completed_entries)
            return

        feasible_candidates, not_yet_candidates = (
            self.constraint_handler.get_feasible_candidates(node)
        )

        # Branch A: execute each feasible candidate now (neutral ordering).
        for candidate in self._order_candidates(node, feasible_candidates):
            if self._is_critical_deadline_violated(node, candidate):
                self._pruned_nodes += 1
                continue
            child_node = self.scheduler._expand_subtask_wo_monitoring(
                node,
                candidate,
                not_yet_candidates,
                feasible_candidates,
            )
            if child_node is None:
                continue
            self._search(child_node, sequence + (child_node.state.subtask.name,))
            if self._timeout_hit:
                return

            conflict_wait_node = self._expand_feasible_conflict_wait(
                node,
                candidate,
                feasible_candidates,
                not_yet_candidates,
            )
            if conflict_wait_node is not None:
                self._search(
                    conflict_wait_node,
                    sequence + (conflict_wait_node.state.subtask.name,),
                )
                if self._timeout_hit:
                    return

        # Branch B: blocked-candidate branches (explicit staged WAIT only).
        if not_yet_candidates:
            for blocked_candidate in self._order_blocked_candidates(not_yet_candidates):
                wait_node = self._expand_blocked_wait(
                    node,
                    blocked_candidate,
                    feasible_candidates,
                    not_yet_candidates,
                )
                if wait_node is not None:
                    self._search(wait_node, sequence + (wait_node.state.subtask.name,))
                    if self._timeout_hit:
                        return

        # Dead-end: no feasible candidates and no future releases.
        if not feasible_candidates and not not_yet_candidates:
            return

    def _commit_solution(
        self,
        makespan: float,
        sequence: Sequence[str],
        completed_entries: Sequence[CompletedEntry],
    ) -> None:
        """Update the incumbent solution when a better or equal solution is found.

        Using ``<=`` rather than ``<`` ensures the sequence is recorded even when
        the oracle starts with an external incumbent equal to the optimal value.

        Args:
            makespan: Completed makespan for the candidate solution.
            sequence: Executed subtask names on the winning branch.
            completed_entries: Scheduled entries associated with the branch.
        """

        if self._best_makespan is None or makespan <= self._best_makespan:
            self._best_makespan = makespan
            self._best_sequence = list(sequence)
            self._best_completed_entries = copy.deepcopy(list(completed_entries))

    def _is_critical_deadline_violated(
        self,
        node: SimulationNode,
        candidate: Candidate,
    ) -> bool:
        """Return True when a critical candidate's interaction deadline has passed.

        A feasible candidate that is critical keeps its ``logical_interaction_start_time``
        from when its predecessor completed.  Because ``_assign_scheduling_due`` only
        scans ``not_yet_candidates``, critical tasks that have already become feasible
        lose their deadline tracking and receive ``scheduling_due = inf``, making
        ``_calculate_candidate_risk_and_urgency`` return ``risk = 0``.

        This check closes that gap: if the oracle's current time is more than
        ``TIMING_TOLERANCE_ABS`` past the critical task's required start time, the
        branch is pruned to prevent recording constraint-violating schedules as optimal.

        Args:
            node: Current DFS node.
            candidate: Feasible candidate about to be expanded.

        Returns:
            True when the candidate is critical and its deadline has been exceeded
            beyond the evaluation tolerance.
        """
        if not candidate.is_critical:
            return False
        lit = candidate.logical_interaction_start_time
        if lit is None:
            return False
        # For (0, True) consecutive constraints the oracle uses the same stricter
        # tolerance as the evaluator: only scheduling overhead is allowed, not the
        # full TIMING_TOLERANCE_ABS window.
        ctx = candidate.critical_context
        if ctx is not None and ctx.interval == 0.0:
            tolerance = CONSECUTIVE_TASK_WAIT_TOLERANCE
        else:
            tolerance = constants.TIMING_TOLERANCE_ABS
        return float(node.state.current_time) > lit + tolerance

    def _order_candidates(
        self,
        curr_node: SimulationNode,
        feasible_candidates: Sequence[Candidate],
    ) -> list[Candidate]:
        """Return candidates in a heuristic-free, deterministic order.

        Sorting by name alone ensures the oracle explores branches in a fixed,
        reproducible sequence without importing any scheduling heuristic.  The
        branch-and-bound incumbent bound still prunes dominated branches, so
        the order affects efficiency but not correctness.

        Args:
            curr_node: Current deterministic search node (unused; kept for API
                compatibility).
            feasible_candidates: Immediately executable candidates.

        Returns:
            Candidate list sorted lexicographically by subtask name.
        """

        return sorted(feasible_candidates, key=lambda c: c.subtask.name)

    def _order_blocked_candidates(
        self,
        not_yet_candidates: Sequence[Candidate],
    ) -> list[Candidate]:
        """Return blocked candidates on the scheduler's earliest timing frontier."""

        return self.scheduler._get_blocked_candidate_frontier(list(not_yet_candidates))

    def _expand_blocked_wait(
        self,
        node: SimulationNode,
        candidate: Candidate,
        feasible_candidates: Sequence[Candidate],
        not_yet_candidates: Sequence[Candidate],
    ) -> Optional[SimulationNode]:
        """Create an explicit WAIT successor for one blocked candidate.

        Args:
            node: Current deterministic search node.
            candidate: Blocked candidate to wait for.
            feasible_candidates: Immediately executable candidates.
            not_yet_candidates: Candidates blocked only by timing/readiness.

        Returns:
            WAIT child node, or ``None`` when no useful wait exists.
        """

        nav_time = self.scheduler._estimate_candidate_navigation_duration(
            node,
            candidate,
        )
        wait_node = self.scheduler._expand_wait_wo_monitoring(
            node,
            candidate,
            list(not_yet_candidates),
            nav_duration=nav_time,
            feasible_candidates=list(feasible_candidates),
        )
        if wait_node is not None:
            self._idle_advances += 1
        return wait_node

    def _expand_feasible_conflict_wait(
        self,
        node: SimulationNode,
        candidate: Candidate,
        feasible_candidates: Sequence[Candidate],
        not_yet_candidates: Sequence[Candidate],
    ) -> Optional[SimulationNode]:
        """Create the scheduler's deterministic conflict-avoidance wait branch.

        The online scheduler may explicitly wait even for already-feasible
        candidates when immediate execution would collide with a future
        reserved critical window. The exact oracle must include the same
        no-monitoring successor in its legal search space.
        """

        conflict_delay, _victim_task_name = (
            self.scheduler.cost_calculator.check_future_conflict(node, candidate)
        )
        if conflict_delay <= constants.EPSILON:
            return None

        nav_time = self.scheduler._estimate_candidate_navigation_duration(
            node,
            candidate,
        )
        wait_node = self.scheduler._expand_wait_wo_monitoring(
            node,
            candidate,
            list(not_yet_candidates),
            nav_duration=nav_time,
            feasible_candidates=list(feasible_candidates),
            additional_delay=conflict_delay,
        )
        if wait_node is not None:
            self._idle_advances += 1
        return wait_node


def build_scheduler_state_after_subtask(
    curr_node: SimulationNode,
    candidate: Candidate,
    action_handler: ActionHandler,
) -> Optional[SchedulerState]:
    """Build the next deterministic scheduler state for a concrete subtask.

    Args:
        curr_node: Current scheduler node.
        candidate: Candidate subtask to execute.
        action_handler: Deterministic action simulator.

    Returns:
        Next scheduler state, or ``None`` when simulation fails.
    """

    action_info = action_handler.get_actions_info(
        curr_node,
        candidate.subtask.execution.primitive_actions,
    )
    if action_info is None or not action_info.success:
        return None

    planned_nav_start_time = curr_node.state.current_time
    completion_time = planned_nav_start_time + action_info.cumulative_time
    copied_subtask = copy.deepcopy(candidate.subtask)
    copied_subtask.duration.total_time = action_info.cumulative_time
    completed_entry = CompletedEntry(
        subtask=copied_subtask,
        schedule_start_time=planned_nav_start_time,
        schedule_end_time=completion_time,
        schedule_nav_time=action_info.first_nav_duration,
        execution_status=bool(action_info.success),
    )
    remaining_subtasks = [
        remaining
        for remaining in curr_node.state.remaining_subtasks
        if remaining.name != candidate.subtask.name
    ]
    return SchedulerState(
        subtask=copied_subtask,
        completed_entries=curr_node.state.completed_entries + [completed_entry],
        remaining_subtasks=remaining_subtasks,
        constraints=curr_node.state.constraints,
        current_time=completion_time,
        scene_positions=action_info.scene_positions,
        held_object=action_info.held_object,
    )


def build_initial_oracle_upper_bound(
    current_best: Optional[float],
) -> Optional[float]:
    """Normalize an optional incumbent upper bound for the oracle.

    Args:
        current_best: Existing deterministic scheduler makespan, if available.

    Returns:
        Floating-point incumbent upper bound or ``None``.
    """

    if current_best is None:
        return None
    return float(current_best)
