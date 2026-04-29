from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class PrimitivePlaceContext:
    """Context captured before an invalid PLACE action is encountered."""

    action_type: str
    receptacle: str
    held_object: str


@dataclass(frozen=True)
class PrimitiveActionIssue:
    """A semantic violation in a primitive action sequence."""

    task_name: str
    subtask_name: str
    issue_type: str
    action_index: int
    action: str
    receptacle: str | None = None
    current_held_object: str | None = None
    prior_places: tuple[PrimitivePlaceContext, ...] = ()


def iter_task_subtasks(task_data: object) -> Iterable[tuple[str, dict]]:
    """Yield ``(task_name, subtask_dict)`` pairs from decomposed task JSON."""

    if not isinstance(task_data, list):
        return

    for task in task_data:
        if not isinstance(task, dict):
            continue
        task_name = str(task.get("Task", ""))
        subtasks = task.get("Subtasks")
        if not isinstance(subtasks, list):
            continue
        for subtask in subtasks:
            if isinstance(subtask, dict):
                yield task_name, subtask


def _parse_action(action: str) -> tuple[str, str]:
    parts = action.split(" ", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _object_type(object_ref: str) -> str:
    return object_ref.split("|", 1)[0]


def find_held_object_semantic_issues(task_data: object) -> list[PrimitiveActionIssue]:
    """Detect violations of the ActionHandler's held-object semantics.

    This mirrors the core preconditions in ``ActionHandler``:
    - ``GRASP`` fails if the agent is already holding something.
    - ``PLACE_INSIDE`` / ``PLACE_ON_TOP`` fail if the agent is holding nothing.
    """

    issues: list[PrimitiveActionIssue] = []

    for task_name, subtask in iter_task_subtasks(task_data):
        subtask_name = str(subtask.get("Name", ""))
        execution = subtask.get("Executions") or {}
        actions = execution.get("PrimitiveActions") or []
        if not isinstance(actions, Sequence):
            continue

        held_object: str | None = None
        prior_places: list[PrimitivePlaceContext] = []

        for action_index, raw_action in enumerate(actions):
            if not isinstance(raw_action, str):
                continue
            action_type, action_arg = _parse_action(raw_action)

            if action_type == "GRASP":
                if held_object is not None:
                    issues.append(
                        PrimitiveActionIssue(
                            task_name=task_name,
                            subtask_name=subtask_name,
                            issue_type="double_grasp",
                            action_index=action_index,
                            action=raw_action,
                            current_held_object=held_object,
                            prior_places=tuple(prior_places),
                        )
                    )
                held_object = action_arg
                continue

            if action_type in {"PLACE_INSIDE", "PLACE_ON_TOP"}:
                receptacle = _object_type(action_arg)
                if held_object is None:
                    issues.append(
                        PrimitiveActionIssue(
                            task_name=task_name,
                            subtask_name=subtask_name,
                            issue_type="place_without_hold",
                            action_index=action_index,
                            action=raw_action,
                            receptacle=receptacle,
                            prior_places=tuple(prior_places),
                        )
                    )
                    continue

                prior_places.append(
                    PrimitivePlaceContext(
                        action_type=action_type,
                        receptacle=receptacle,
                        held_object=_object_type(held_object),
                    )
                )
                held_object = None

    return issues


def find_first_task_sequence_issue(task_data: object) -> PrimitiveActionIssue | None:
    """Return the first held-object semantic issue across the full task sequence.

    Unlike :func:`find_held_object_semantic_issues`, this validator carries the
    held-object state across subtask boundaries in the order given by the
    decomposed task JSON. This better matches how the scheduler executes a full
    translated instruction and is the right predicate for "translation-valid"
    filtering in aggregate experiment tables.
    """

    held_object: str | None = None
    prior_places: list[PrimitivePlaceContext] = []

    for task_name, subtask in iter_task_subtasks(task_data):
        subtask_name = str(subtask.get("Name", ""))
        execution = subtask.get("Executions") or {}
        actions = execution.get("PrimitiveActions") or []
        if not isinstance(actions, Sequence):
            continue

        for action_index, raw_action in enumerate(actions):
            if not isinstance(raw_action, str):
                continue
            action_type, action_arg = _parse_action(raw_action)

            if action_type == "GRASP":
                if held_object is not None:
                    return PrimitiveActionIssue(
                        task_name=task_name,
                        subtask_name=subtask_name,
                        issue_type="double_grasp",
                        action_index=action_index,
                        action=raw_action,
                        current_held_object=held_object,
                        prior_places=tuple(prior_places),
                    )
                held_object = action_arg
                continue

            if action_type in {"PLACE_INSIDE", "PLACE_ON_TOP"}:
                receptacle = _object_type(action_arg)
                if held_object is None:
                    return PrimitiveActionIssue(
                        task_name=task_name,
                        subtask_name=subtask_name,
                        issue_type="place_without_hold",
                        action_index=action_index,
                        action=raw_action,
                        receptacle=receptacle,
                        prior_places=tuple(prior_places),
                    )

                prior_places.append(
                    PrimitivePlaceContext(
                        action_type=action_type,
                        receptacle=receptacle,
                        held_object=_object_type(held_object),
                    )
                )
                held_object = None

    return None


def classify_issue_group(issue: PrimitiveActionIssue) -> str:
    """Map a primitive-action issue to a human-readable failure group."""

    if issue.issue_type == "double_grasp":
        return "Double grasp while already holding an object"

    if issue.issue_type != "place_without_hold":
        return f"Unknown issue type: {issue.issue_type}"

    if issue.receptacle == "Sink":
        return "Pot placed into sink without first grasping pot"

    if issue.receptacle == "CoffeeMachine":
        return "Coffee machine prepared without grasping mug/cup"

    if issue.receptacle == "StoveBurner":
        if issue.prior_places:
            last_place = issue.prior_places[-1]
            if last_place.receptacle in {"Pan", "Pot"}:
                return (
                    "Container placed onto stove after ingredient placement, "
                    "without re-grasp"
                )
        return "StoveBurner placement without held object"

    return f"Other illegal place target: {issue.receptacle or '<unknown>'}"


def choose_primary_issue_group(issues: Sequence[PrimitiveActionIssue]) -> str | None:
    """Return one stable primary failure label for a file-level issue set."""

    if not issues:
        return None

    labels = {classify_issue_group(issue) for issue in issues}
    priority = [
        "Container placed onto stove after ingredient placement, without re-grasp",
        "Pot placed into sink without first grasping pot",
        "Coffee machine prepared without grasping mug/cup",
        "StoveBurner placement without held object",
        "Double grasp while already holding an object",
    ]

    for label in priority:
        if label in labels:
            return label

    return sorted(labels)[0]
