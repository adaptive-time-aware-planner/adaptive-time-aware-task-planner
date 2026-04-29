from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import networkx as nx

if TYPE_CHECKING:
    from src.core.scheduler import ConstraintHandler
    from src.models.dataclass import CompletedEntry

log = logging.getLogger(__name__)


def get_critical_start_info(
    subtask_name: str,
    completed: list[CompletedEntry],
    constraints: nx.DiGraph,
    constraint_handler: "ConstraintHandler",
) -> Tuple[Optional[str], Optional[float]]:
    """
    Calculate the earliest start time based on critical constraints ('After', critical=True).
    Finds the predecessor that dictates the latest start time among critical constraints.
    """

    slots = constraint_handler.get_time_slots(subtask_name, constraints, direction="in")
    critical_slots = [slot for slot in slots if slot.is_critical]

    if not critical_slots:
        for slot in slots:
            if slot.related_subtask_name.startswith("Monitoring for"):
                return get_critical_start_info(
                    slot.related_subtask_name,
                    completed,
                    constraints,
                    constraint_handler,
                )
            
        raise ValueError(f"No critical slots found for {subtask_name}")

    max_critical = max(critical_slots, key=lambda x: x.interval)
    critical_start_sub_name = max_critical.related_subtask_name

    # completed_subtasks에서 critical_start_sub_name의 end_time 찾기
    critical_start_sub_end_time = next(
        (
            ce.sim_end_time
            for ce in completed
            if ce.subtask.name == critical_start_sub_name
        ),
        None,
    )
    if critical_start_sub_end_time is None:
        raise ValueError(
            f"Critical start sub '{critical_start_sub_name}' end time not found in completed list."
        )
    return critical_start_sub_name, critical_start_sub_end_time
