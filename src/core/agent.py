from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from src.core.monitoring import (
    BeliefStore,
    BeliefUpdateContext,
    BeliefUpdater,
    GroundTruthStore,
    create_belief_updater,
)
from src.models.dataclass import SchedulerState
from src.utils.common import create_module_logger, extract_monitoring_target_name
from src.utils.config.constants import (
    CRITICAL_OBJECT_GROUND_TRUTH,
    MIN_VARIANCE,
)

if TYPE_CHECKING:
    from src.scheduler import ConstraintHandler

log = create_module_logger(module_name=__name__, module_log=True)


class Agent:
    """Update monitoring beliefs and propagate the result to constraints."""

    def __init__(
        self,
        constraint_handler: ConstraintHandler,
        bayesian_load: dict[str, dict[str, Any]],
        belief_updater: Optional[BeliefUpdater] = None,
        belief_store: Optional[BeliefStore] = None,
        ground_truth_store: Optional[GroundTruthStore] = None,
        real_world_mode: bool = False,
    ) -> None:
        """Initialize the runtime monitoring agent.

        Args:
            constraint_handler: Helper used to recover critical interval context.
            bayesian_load: Initial belief mapping created during task parsing.
            belief_updater: Optional backend-specific updater implementation.
            belief_store: Optional shared belief store instance.
            ground_truth_store: Optional runtime ground-truth sampler.
        """

        self.belief_store = belief_store or BeliefStore(bayesian_load)
        self.belief_updater = belief_updater or create_belief_updater(
            "bayesian",
            self.belief_store,
        )
        self.ground_truth_store = ground_truth_store or GroundTruthStore(
            CRITICAL_OBJECT_GROUND_TRUTH
        )
        self.real_world_mode = bool(real_world_mode)
        self.estimate_knowledge: Dict[str, Dict[str, Any]] = self.belief_store.as_dict()
        self.constraint_handler = constraint_handler

    @staticmethod
    def _build_critical_family_key(
        *,
        critical_start_sub_name: str,
        critical_end_sub_name: str,
    ) -> str:
        """Return a stable identifier for one critical interval family."""

        return f"{critical_start_sub_name}::{critical_end_sub_name}"

    def _update_knowledge_and_constraints(
        self,
        state: SchedulerState,
        monitoring_target_obj_name: str,
        posterior_mean: float,
        posterior_variance: float,
        critical_start_sub_name: str,
        monitoring_target_sub_name: str,
        critical_start_sub_end_time: float,
        elapsed_interval: float,
        late_due_observation: bool,
    ) -> None:
        """Persist posterior summaries and rewrite the active constraint edges.

        Args:
            state: Scheduler state after a monitor action.
            monitoring_target_obj_name: Object type whose belief changed.
            posterior_mean: Updated expected interval duration.
            posterior_variance: Updated posterior variance.
            critical_start_sub_name: Critical start subtask name.
            monitoring_target_sub_name: Critical end subtask name.
            critical_start_sub_end_time: End time of the critical start subtask.
        """

        self.estimate_knowledge = self.belief_store.as_dict()
        self.belief_store.persist()
        family_key = self._build_critical_family_key(
            critical_start_sub_name=critical_start_sub_name,
            critical_end_sub_name=monitoring_target_sub_name,
        )

        # 2) constraints 그래프 업데이트 (이 로직은 유지)
        #    - (critical_start_sub_name, monitoring_target_sub_name)에 posterior_mean 반영

        if not state.constraints.has_edge(
            critical_start_sub_name, monitoring_target_sub_name
        ):
            state.constraints.add_edge(
                critical_start_sub_name,
                monitoring_target_sub_name,
                info={},
            )
        edge_info = state.constraints.edges[
            critical_start_sub_name, monitoring_target_sub_name
        ].setdefault("info", {})
        edge_info["Interval"] = posterior_mean
        edge_info["Variance"] = posterior_variance
        edge_info["IsCritical"] = True
        edge_info["IsMonitoringResidual"] = False
        edge_info["CriticalFamilyKey"] = family_key
        edge_info["LateObservation"] = late_due_observation
        edge_info["ObservedElapsedInterval"] = elapsed_interval

        #    - (현재 모니터링 서브태스크, 모니터링 대상) 간 엣지에 잔여 구간 반영
        if self.real_world_mode:
            updated_interval = max(0.0, posterior_mean - elapsed_interval)
        else:
            updated_interval = max(
                0.0,
                critical_start_sub_end_time + posterior_mean - state.current_time,
            )

        if not state.constraints.has_edge(state.subtask.name, monitoring_target_sub_name):
            state.constraints.add_edge(
                state.subtask.name,
                monitoring_target_sub_name,
                info={},
            )
        edge_info = state.constraints.edges[
            state.subtask.name, monitoring_target_sub_name
        ].setdefault("info", {})
        edge_info["Interval"] = updated_interval
        edge_info["Variance"] = posterior_variance
        edge_info["IsCritical"] = True
        edge_info["IsMonitoringResidual"] = True
        edge_info["CriticalFamilyKey"] = family_key
        edge_info["LateObservation"] = late_due_observation
        edge_info["ObservedElapsedInterval"] = elapsed_interval

    def _get_prior_estimate(self, obj_name: str) -> Tuple[float, float]:
        """Return the prior summary for a monitored object.

        Args:
            obj_name: Object type key in the belief store.

        Returns:
            Prior mean and variance.
        """

        summary = self.belief_store.get_summary(obj_name)
        return max(0.0, summary.expected_duration), max(summary.variance, MIN_VARIANCE)

    def _extract_monitoring_object_name(self, state: SchedulerState) -> Optional[str]:
        """Extract the object type associated with a monitoring subtask.

        Args:
            state: Scheduler state containing the monitoring subtask.

        Returns:
            Object type string when available, otherwise ``None``.
        """

        execution = getattr(state.subtask, "execution", None)
        if execution is None:
            return None

        objects = getattr(execution, "objects", None)
        raw_object: Optional[str] = None
        if isinstance(objects, list) and objects:
            raw_object = objects[0]
        elif isinstance(objects, dict) and objects:
            raw_object = next(iter(objects.keys()))

        if not raw_object:
            return None
        return raw_object.split("|")[0]

    @staticmethod
    def _find_completed_entry(
        state: SchedulerState,
        subtask_name: str,
    ) -> Optional[Any]:
        """Return the completed entry for ``subtask_name`` when available."""

        return next(
            (entry for entry in state.completed_entries if entry.subtask.name == subtask_name),
            None,
        )

    def _resolve_elapsed_interval(
        self,
        state: SchedulerState,
        *,
        critical_start_sub_name: str,
        critical_start_sub_end_time: float,
    ) -> Tuple[float, bool]:
        """Return the elapsed interval used for posterior updates.

        In ROS mode, use actual elapsed wall-clock time recovered from completed
        entries. Otherwise preserve the planner-clock behavior.
        """

        if self.real_world_mode:
            start_entry = self._find_completed_entry(state, critical_start_sub_name)
            monitor_entry = self._find_completed_entry(state, state.subtask.name)
            if (
                start_entry is not None
                and monitor_entry is not None
                and math.isfinite(start_entry.sim_end_time)
                and math.isfinite(monitor_entry.sim_end_time)
            ):
                return max(0.0, monitor_entry.sim_end_time - start_entry.sim_end_time), True

        return max(0.0, state.current_time - critical_start_sub_end_time), False

    def update_monitoring_belief(
        self, state: SchedulerState
    ) -> Tuple[SchedulerState, Optional[Dict[str, Any]]]:
        """Update posterior belief after executing a monitoring subtask.

        Args:
            state: Scheduler state after the monitoring subtask completed.

        Returns:
            The unchanged state and a diagnostics payload for logging/results.
        """

        from src.utils.task.constraints_util import get_critical_start_info

        monitoring_target_sub_name = extract_monitoring_target_name(state.subtask.name)
        monitoring_target_obj_name = self._extract_monitoring_object_name(state)

        if monitoring_target_obj_name is None:
            log.warning("Monitoring target object not found. Skipping belief update.")
            return state, {
                "updated_subtask_name": "N/A",
                "error": "Monitoring target object not found.",
            }

        # 3-3. G.T.와 prior를 직접 조회합니다.
        gt_interval = self.ground_truth_store.get_interval(monitoring_target_obj_name)

        # TODO 4: (중요) G.T. 값이 없을 경우의 처리 - 계산 전에 미리 확인하여 TypeError 방지
        if gt_interval is None:
            log.warning(
                f"Ground truth not found for '{monitoring_target_obj_name}'. Skipping Bayesian update."
            )
            return state, {
                "updated_subtask_name": "N/A",
                "error": "Ground truth not found for this object.",
            }

        prior_mean, prior_variance = self._get_prior_estimate(
            monitoring_target_obj_name
        )

        critical_start_sub_name, critical_start_sub_end_time = get_critical_start_info(
            subtask_name=state.subtask.name,
            completed=state.completed_entries,
            constraints=state.constraints,
            constraint_handler=self.constraint_handler,
        )

        # 5) 베이지안 업데이트 계산
        # critical 제약이 시작 된 이후 경과된 separation interval
        critical_elapsed_interval, used_actual_elapsed = self._resolve_elapsed_interval(
            state,
            critical_start_sub_name=critical_start_sub_name,
            critical_start_sub_end_time=critical_start_sub_end_time,
        )

        update_result = self.belief_updater.update(
            BeliefUpdateContext(
                object_name=monitoring_target_obj_name,
                gt_interval=gt_interval,
                prior_mean=prior_mean,
                prior_variance=prior_variance,
                elapsed_interval=critical_elapsed_interval,
            )
        )

        # ZeroDivisionError 방지
        if prior_mean > 0:
            belief_diff = abs(update_result.posterior_mean - prior_mean) / prior_mean
        else:
            # prior_mean이 0일 경우, posterior_mean이 0이 아니면 무한대의 차이로 간주하여 항상 업데이트
            belief_diff = float("inf") if update_result.posterior_mean != 0 else 0.0

        log.info("%s_diff: %s", self.belief_updater.method, belief_diff)
        remaining_after_monitor = (
            update_result.posterior_mean - critical_elapsed_interval
        )
        late_due_observation = bool(
            self.real_world_mode and remaining_after_monitor <= 0.0
        )

        self._update_knowledge_and_constraints(
            state=state,
            monitoring_target_obj_name=monitoring_target_obj_name,
            posterior_mean=update_result.posterior_mean,
            posterior_variance=update_result.posterior_variance,
            critical_start_sub_name=critical_start_sub_name,
            monitoring_target_sub_name=monitoring_target_sub_name,
            critical_start_sub_end_time=critical_start_sub_end_time,
            elapsed_interval=critical_elapsed_interval,
            late_due_observation=late_due_observation,
        )

        monitored_subtask = {
            "updated_subtask_name": critical_start_sub_name,
            "critical_start_subtask_name": critical_start_sub_name,
            "critical_end_subtask_name": monitoring_target_sub_name,
            "critical_interval_key": (
                f"{critical_start_sub_name} -> {monitoring_target_sub_name}"
            ),
            "original_expected_time": prior_mean,
            "updated_expected_time": update_result.posterior_mean,
            "ground_truth_time": gt_interval,
            "ground_truth_distribution": self.ground_truth_store.distribution,
            "update_method": update_result.method,
            "elapsed_interval_used": critical_elapsed_interval,
            "uses_actual_elapsed_interval": used_actual_elapsed,
            "remaining_after_monitor": remaining_after_monitor,
            "late_due_observation": late_due_observation,
            **update_result.diagnostics,
        }

        return state, monitored_subtask

    def bayesian_estimate(
        self, state: SchedulerState
    ) -> Tuple[SchedulerState, Optional[Dict[str, Any]]]:
        """Backward-compatible wrapper for the legacy method name.

        Args:
            state: Scheduler state after a monitoring subtask completed.

        Returns:
            Result from the backend-agnostic monitoring update path.
        """

        return self.update_monitoring_belief(state)
