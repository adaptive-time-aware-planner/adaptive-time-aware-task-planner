from typing import Dict, List, Optional

import networkx as nx


class Duration:
    def __init__(self, type: str, interval: int, total_time: float = 0.0):
        self.type: str = type
        self.interval: int = interval
        self.total_time: Optional[float] = total_time

    def __repr__(self):
        return f"Duration(type={self.type}, interval={self.interval}, total_time={self.total_time})"

    @classmethod
    def from_dict(cls, data: Dict) -> "Duration":
        return cls(
            type=data["Type"],
            interval=float(data["Interval"]),
            total_time=None,  # 초기에는 total_time 없음
        )


class Execution:
    def __init__(self, objects: Dict[str, int], primitive_actions: List[str]):
        self.objects = objects
        self.primitive_actions = primitive_actions

    def __repr__(self):
        return f"Execution(objects={self.objects}, primitive_actions={self.primitive_actions})"

    @classmethod
    def from_dict(cls, data: Dict) -> "Execution":
        return cls(
            objects=data["Objects"],
            primitive_actions=data["PrimitiveActions"],
        )


class TemporalConstraint:
    def __init__(
        self,
        constraint_type: str,
        rel_subtask_name: str,
        interval: int,
        is_critical: bool,
    ):
        self.constraint_type = constraint_type
        self.rel_subtask_name = rel_subtask_name
        self.interval = interval
        self.is_critical = is_critical

    def __repr__(self):
        return (
            f"TemporalConstraint(type={self.constraint_type}, subtask={self.rel_subtask_name}, "
            f"interval={self.interval}, is_critical={self.is_critical})"
        )

    @classmethod
    def from_dict(cls, data: Dict) -> "TemporalConstraint":
        return cls(
            constraint_type=data["Type"],
            rel_subtask_name=data["Subtask"],
            interval=data["Interval"],
            is_critical=data["Urgency"],
        )


class Subtask:
    def __init__(
        self,
        task_name: str,
        name: str,
        repetition: int,
        subtask_type: str,
        execution: Execution,
        duration: Duration,
        temporal_constraints: Optional[List[TemporalConstraint]] = None,
        decomposed: bool = False,
        execution_status: bool = False,
    ):
        self.task_name = task_name
        self.name = name
        self.repetition = repetition
        self.subtask_type = subtask_type
        self.execution = execution
        self.duration = duration
        self.temporal_constraints = temporal_constraints or []
        self.decomposed = decomposed
        self.execution_status = execution_status

    def __repr__(self):
        return f"Subtask({self.name})"

    @classmethod
    def from_dict(cls, subtask_data: Dict, task_name: str) -> "Subtask":
        """Create a Subtask object from a dictionary.

        Args:
            subtask_data (Dict): dictionary containing subtask data
            task_name (str): task which include the subtask

        Returns:
            Subtask: Subtask object created from the dictionary
        """
        execution = Execution.from_dict(subtask_data["Executions"])
        duration = Duration.from_dict(subtask_data["Duration"])
        repetition = subtask_data.get("Repetition", 1)

        temporal_constraints_data = subtask_data.get("TemporalConstraints", [])
        temporal_constraints = [
            TemporalConstraint.from_dict(tc) for tc in temporal_constraints_data
        ]

        return cls(
            task_name=task_name,
            name=subtask_data["Name"],
            repetition=repetition,
            subtask_type=subtask_data["Type"],
            duration=duration,
            execution=execution,
            temporal_constraints=temporal_constraints,
        )

    def decompose(self) -> List["Subtask"]:
        """Decompose a subtask into multiple subtasks when repetition larger than 1."""
        if self.repetition > 1:
            return self._decompose_subtask()
        else:
            return [self]

    def _decompose_subtask(self) -> List["Subtask"]:
        """Implementation of Subtask Decomposition."""
        decomposed_subtasks = []
        # 실제 decomposed subtask 생성
        for part_index in range(self.repetition):
            decomposed_subtask = Subtask(
                task_name=self.task_name,
                name=f"{self.name}_part_{part_index + 1}",
                subtask_type=self.subtask_type,
                repetition=1,
                duration=Duration(
                    type=self.duration.type,
                    interval=self.duration.interval,
                ),
                execution=self.execution,
                temporal_constraints=self._get_temporal_constraints(part_index),
            )
            decomposed_subtasks.append(decomposed_subtask)

        return decomposed_subtasks

    def _get_temporal_constraints(self, part_index: int) -> List[TemporalConstraint]:
        if part_index == 0:
            return self.temporal_constraints
        else:
            return [
                TemporalConstraint(
                    constraint_type="After",
                    rel_subtask_name=f"{self.name}_part_{part_index}",
                    interval=0,
                    is_critical=False,
                )
            ]


class Task:
    def __init__(self, name: str, subtasks: List[Subtask]):
        self.name = name
        self.subtasks = subtasks

    def __repr__(self):
        return f"Task(name={self.name}, subtasks={self.subtasks})"

    def get_total_seq_duration(self) -> int:
        return sum(subtask.duration.interval for subtask in self.subtasks)

    @classmethod
    def from_dict(cls, task_data: Dict) -> "Task":
        task_name = task_data["Task"]
        subtasks = [
            Subtask.from_dict(subtask_data, task_name)
            for subtask_data in task_data["Subtasks"]
            if "impossible" not in subtask_data["Name"].lower()
        ]
        return cls(name=task_name, subtasks=subtasks)

    @classmethod
    def parse_instruction(cls, data: List[Dict]) -> List["Task"]:
        return [cls.from_dict(task_data) for task_data in data]

    def decompose_subtasks(self):
        decomposed_subtasks = []
        subtask_mapping = {}

        for subtask in self.subtasks:
            decomposed_parts = subtask.decompose()
            decomposed_subtasks.extend(decomposed_parts)
            # subtask가 decompose가 된 경우, mapping 정보를 담는 dictionary 생성
            subtask_mapping[subtask.name] = decomposed_parts

        self.subtasks = decomposed_subtasks

        self.update_constraints(subtask_mapping)
        return self

    def update_constraints(self, subtask_mapping: Dict[str, List[Subtask]]):
        """
        - 기존 Temporal Constraints:
            Subtask_B는 Subtask_A가 끝난 후에 시작해야 한다는 "After" 조건
        - 분해 후 Temporal Constraints:
            Subtask_A가 Subtask_A_1, Subtask_A_2, Subtask_A_3로 분해되었을 경우,
            이제 Subtask_B는 Subtask_A_3(마지막 파트)의 끝난 후에 시작해야 한다는 조건을 가지도록 수정
        """
        # Task에 종속된 subtask에 대하여 temporal constraints 업데이트
        for subtask in self.subtasks:
            updated_constraints = []
            # decompose된 subtask에 대하여
            for constraint in subtask.temporal_constraints:
                # decompose된 subtask가 기존 subtask의 temporal constraints에 포함되어 있는 경우
                if constraint.rel_subtask_name in subtask_mapping:
                    last_decomposed_part = subtask_mapping[constraint.rel_subtask_name][
                        -1
                    ]
                    updated_constraints.append(
                        TemporalConstraint(
                            constraint_type=constraint.constraint_type,
                            rel_subtask_name=last_decomposed_part.name,
                            interval=constraint.interval,
                            is_critical=constraint.is_critical,
                        )
                    )
                else:
                    updated_constraints.append(constraint)
            subtask.temporal_constraints = updated_constraints


class TaskGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(self, tasks: List[Task]):
        for task in tasks:
            for subtask in task.subtasks:
                # 노드 추가
                subtask_node = subtask.name
                self.graph.add_node(subtask_node, type=subtask.subtask_type)

                # 엣지 추가
                for constraint in subtask.temporal_constraints:
                    if constraint.rel_subtask_name:
                        edge_data = {
                            "info": {
                                "Type": constraint.constraint_type,
                                "Interval": constraint.interval,
                                "IsCritical": constraint.is_critical,
                            }
                        }
                        if constraint.constraint_type == "Before":
                            self.graph.add_edge(
                                subtask_node, constraint.rel_subtask_name, **edge_data
                            )
                        elif constraint.constraint_type == "After":
                            self.graph.add_edge(
                                constraint.rel_subtask_name, subtask_node, **edge_data
                            )
                    else:
                        raise ValueError("Constrained Node does not exist")

        return self.graph
