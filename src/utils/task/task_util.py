# utils/task/task_util.py
from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from dotenv import load_dotenv
from networkx import DiGraph

from src.models.dataclass import CompletedEntry, SchedulerState
from src.models.task import Duration, Execution, Subtask, Task, TemporalConstraint

# 내부 프로젝트 모듈
from src.utils.common import create_module_logger
from src.utils.config import constants
from src.utils.config.constants import (  # 분산 값도 상수로 사용하기 위해 추가
    GRASP_ACTION_DURATION,
    MONITORING_DURATION,
    PLACE_ACTION_DURATION,
    PRIMITIVE_ACTION_SET,
    SCENE_KNOWLEDGE_PATH,
    TOGGLE_ACTION_DURATION,
)
from src.utils.nlp.sentence_transformer import SentenceSimilarityModel

load_dotenv()
log = create_module_logger(__name__, module_log=True)


class TaskUtil:
    """
    Task(혹은 Subtask) 처리와 관련된 유틸리티 메서드들을 제공하는 클래스.
    """

    # Sentence Transformer 인스턴스(싱글톤) 로드
    _sentence_sim_model = SentenceSimilarityModel.get_instance()

    @staticmethod
    def _load_json_file(file_path: Path) -> dict:
        """
        주어진 file_path에서 JSON 데이터를 로드하여 반환한다.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _save_json_file(file_path: Path, data: dict) -> None:
        """
        주어진 file_path에 JSON 데이터를 저장한다.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def _load_object_ids(scene_file_name: str) -> Dict[str, List[str]]:
        """
        scene_name_physics.json  파일에서 object ID 정보를 로드한다.
        """
        room_type_dirs = list(SCENE_KNOWLEDGE_PATH.glob("*"))
        for room_type_dir in room_type_dirs:
            file_path = room_type_dir / "environment" / f"{scene_file_name}"
            if file_path.exists():
                return TaskUtil._load_json_file(file_path)
        # If the loop completes without finding the file, raise an error.
        raise FileNotFoundError(
            f"Physics file for scene '{scene_file_name}' not found in any subdirectories of {file_path}"
        )

    @staticmethod
    def tasks_to_subtasks(
        tasks: List[Task],
        mode: Literal["all", "name"] = "all",
    ) -> Union[List[Subtask], List[str]]:
        """
        주어진 Task 리스트에서 모든 Subtask(혹은 subtask name)만 추출하여 반환한다.

        :param tasks: List of Task
        :param mode:
          - "all": Subtask 객체 리스트
          - "name": Subtask의 name(str) 리스트
        :return: Subtask 리스트 or Subtask name 리스트
        """
        if mode == "all":
            return [subtask for task in tasks for subtask in task.subtasks]
        elif mode == "name":
            return [subtask.name for task in tasks for subtask in task.subtasks]
        else:
            raise ValueError("mode must be either 'all' or 'name'.")

    @staticmethod
    def adjust_subtasks_duration(subtasks: List[Subtask]) -> List[Subtask]:
        """
        Subtask들의 duration.interval 값을, 에이전트의 사전 지식을 기반으로 보정한다.
        (PRIMITIVE_ACTION_DURATION 등을 참조)

        :param subtasks: Subtask 리스트
        :return: 동일 Subtask 리스트(단, duration.interval만 업데이트)
        """

        def _get_action_duration(st: Subtask) -> float:
            total_duration = 0.0
            for primitive_action in st.execution.primitive_actions:
                action_name = primitive_action.split()[0]
                if action_name not in PRIMITIVE_ACTION_SET:
                    raise ValueError(
                        f"Action '{action_name}' not found in the primitive action set."
                    )
                # NAVIGATE_TO, MONITORING, WAIT 등은 지속시간 계산 제외
                if action_name not in {"NAVIGATE_TO", "MONITORING", "WAIT"}:
                    # Per-action duration mapping
                    duration_map = {
                        "GRASP": GRASP_ACTION_DURATION,
                        "PLACE_INSIDE": PLACE_ACTION_DURATION,
                        "PLACE_ON_TOP": PLACE_ACTION_DURATION,
                        "OPEN": TOGGLE_ACTION_DURATION,
                        "CLOSE": TOGGLE_ACTION_DURATION,
                        "TOGGLE_ON": TOGGLE_ACTION_DURATION,
                        "TOGGLE_OFF": TOGGLE_ACTION_DURATION,
                        "SLICE": TOGGLE_ACTION_DURATION,
                        "FILL": PLACE_ACTION_DURATION,
                    }
                    total_duration += duration_map.get(
                        action_name, TOGGLE_ACTION_DURATION
                    )
            return total_duration

        for subtask in subtasks:
            subtask.duration.interval = _get_action_duration(subtask)
        return subtasks

    @staticmethod
    def refine_primitive_actions(tasks: List[Task]) -> List[Task]:
        """
        1) 모든 subtask가 첫 액션으로 'NAVIGATE_TO <obj>'를 가지도록 보장
        2) PLACE 액션이 직전에 'NAVIGATE_TO <obj>'가 없는 경우, 자동으로 NAVIGATE_TO <obj> 추가
        3) Sink 처리: Sink인데 SinkBasin이 언급되지 않은 경우, 'PLACE_INSIDE <obj>|SinkBasin'로 교정
        """
        subtasks = TaskUtil.tasks_to_subtasks(tasks, mode="all")

        for st in subtasks:
            actions = st.execution.primitive_actions
            if not actions:
                continue

            # (1) 첫 액션이 'NAVIGATE_TO'인지 확인, 아니면 삽입
            first_parts = actions[0].split(" ", 1)
            if first_parts[0] != "NAVIGATE_TO" and len(first_parts) == 2:
                _, obj = first_parts
                actions.insert(0, f"NAVIGATE_TO {obj}")

            # (2), (3) PLACE 액션 보정 + SinkBasin 처리
            updated_actions = []
            for i, action in enumerate(actions):
                parts = action.split(" ", 1)
                if len(parts) != 2:
                    updated_actions.append(action)
                    continue

                base_action, to_obj = parts

                # PLACE 계열인데 직전 액션이 NAVIGATE_TO <same obj>가 아닌 경우
                if (
                    i > 0
                    and base_action.startswith("PLACE")
                    and not actions[i - 1].startswith(f"NAVIGATE_TO {to_obj}")
                ):
                    updated_actions.append(f"NAVIGATE_TO {to_obj}")

                # Sink → SinkBasin 교정
                if (
                    base_action.startswith("PLACE")
                    and "Sink" in to_obj
                    and "SinkBasin" not in to_obj
                ):
                    corrected_obj = f"{to_obj}|SinkBasin"
                    updated_actions.append(f"PLACE_INSIDE {corrected_obj}")
                else:
                    updated_actions.append(action)

            st.execution.primitive_actions = updated_actions

        return tasks

    @staticmethod
    def _parse_primitive_action(action: str) -> Tuple[str, str]:
        """Split one primitive action into ``(ACTION_TYPE, ARGUMENT)``."""

        parts = action.split(" ", 1)
        if len(parts) == 1:
            return parts[0].upper(), ""
        return parts[0].upper(), parts[1]

    @staticmethod
    def _object_type(object_ref: str) -> str:
        """Return the object category prefix from a full object id."""

        return object_ref.split("|", 1)[0]

    @classmethod
    def _infer_missing_hold_repair_target(
        cls,
        destination_obj: str,
        recent_places: List[Dict[str, str]],
    ) -> Optional[str]:
        """Infer which object should be re-grasped before an invalid PLACE.

        The current translation failures are highly templatic:
        - ingredient placed into ``Pan``/``Pot`` and then the container is moved
          to ``StoveBurner`` without re-grasping the container;
        - ingredient placed into ``Pot`` and then the pot is moved to ``Sink``
          without re-grasping the pot;
        - occasionally a ``Mug``/``Cup`` should be re-grasped before placing it
          on the coffee machine.
        """

        destination_type = cls._object_type(destination_obj)
        for context in reversed(recent_places):
            receptacle_type = context["receptacle_type"]
            if destination_type == "StoveBurner" and receptacle_type in {"Pan", "Pot"}:
                return context["receptacle_id"]
            if destination_type == "Sink" and receptacle_type == "Pot":
                return context["receptacle_id"]
            if destination_type == "CoffeeMachine" and context["held_object_type"] in {
                "Mug",
                "Cup",
            }:
                return context["held_object_id"]
        return None

    @classmethod
    def repair_held_object_semantics(cls, tasks: List[Task]) -> List[Task]:
        """Repair common held-object semantic omissions in primitive sequences.

        This pass performs a lightweight symbolic replay over the full
        instruction order and inserts the minimal re-grasp sequence needed for
        templatic translation failures:
        - insert ``GRASP <container>`` when we are already at the container;
        - otherwise insert ``NAVIGATE_TO <container>``, ``GRASP <container>``.

        The pass intentionally remains conservative: if no reliable repair
        target can be inferred from the prior place history, the original
        sequence is preserved.
        """

        subtasks = cls.tasks_to_subtasks(tasks, mode="all")
        held_object: Optional[str] = None
        current_anchor: Optional[str] = None
        recent_places: List[Dict[str, str]] = []

        for subtask in subtasks:
            actions = subtask.execution.primitive_actions
            if not actions:
                continue

            repaired_actions: List[str] = []
            action_anchor_meta: List[Tuple[Optional[str], Optional[str]]] = []
            local_anchor = current_anchor

            def append_action(action: str) -> None:
                nonlocal held_object, local_anchor

                before_anchor = local_anchor
                action_type, target_obj = cls._parse_primitive_action(action)

                if action_type == "NAVIGATE_TO" and target_obj:
                    local_anchor = target_obj
                elif action_type == "GRASP" and target_obj:
                    if held_object is None:
                        held_object = target_obj
                elif action_type in {"PLACE_INSIDE", "PLACE_ON_TOP"} and target_obj:
                    if held_object is not None:
                        recent_places.append(
                            {
                                "action_type": action_type,
                                "receptacle_id": target_obj,
                                "receptacle_type": cls._object_type(target_obj),
                                "held_object_id": held_object,
                                "held_object_type": cls._object_type(held_object),
                            }
                        )
                        held_object = None

                repaired_actions.append(action)
                action_anchor_meta.append((before_anchor, local_anchor))

            for action in actions:
                action_type, target_obj = cls._parse_primitive_action(action)

                if (
                    action_type in {"PLACE_INSIDE", "PLACE_ON_TOP"}
                    and target_obj
                    and held_object is None
                ):
                    repair_target = cls._infer_missing_hold_repair_target(
                        target_obj,
                        recent_places,
                    )
                    if repair_target:
                        popped_destination_nav: Optional[str] = None
                        if repaired_actions:
                            prev_type, prev_target = cls._parse_primitive_action(
                                repaired_actions[-1]
                            )
                            if (
                                prev_type == "NAVIGATE_TO"
                                and prev_target == target_obj
                            ):
                                popped_destination_nav = repaired_actions.pop()
                                before_anchor, _ = action_anchor_meta.pop()
                                local_anchor = before_anchor

                        inserted_actions: List[str] = []
                        if local_anchor != repair_target:
                            nav_action = f"NAVIGATE_TO {repair_target}"
                            append_action(nav_action)
                            inserted_actions.append(nav_action)

                        grasp_action = f"GRASP {repair_target}"
                        append_action(grasp_action)
                        inserted_actions.append(grasp_action)

                        if popped_destination_nav is not None:
                            append_action(popped_destination_nav)

                        log.debug(
                            "[repair_held_object_semantics] Inserted %s before '%s' in subtask '%s'.",
                            inserted_actions,
                            action,
                            subtask.name,
                        )

                append_action(action)

            subtask.execution.primitive_actions = repaired_actions
            current_anchor = local_anchor

        return tasks

    @classmethod
    def check_obj_id(cls, scene_name: str, tasks: List[Task]) -> List[Task]:
        """
        scene_name 은 physics_environment.json 으로 끝남
        Subtask의 primitive_actions에 사용된 obj_id가 유효한지 확인 후,
        유효하지 않다면 문장 유사도 기반으로 가장 가까운 후보로 교체한다.
        """
        from src.utils.io_utils.task_io import load_scene_positions

        # 1) Load all available object IDs and their categories for the scene
        object_categories = cls._load_object_ids(scene_name)
        scene_positions = load_scene_positions(scene_name)
        all_object_ids_in_scene = list(scene_positions.keys())

        # Create a reverse map from object type to a list of full object IDs in the scene
        object_map_in_scene = {}
        for category, object_types in object_categories.items():
            for obj_type in object_types:
                # Find all instances of this object type in the current scene
                matching_instances = [
                    full_id
                    for full_id in all_object_ids_in_scene
                    if full_id.startswith(obj_type)
                ]
                if matching_instances:
                    if category not in object_map_in_scene:
                        object_map_in_scene[category] = []
                    object_map_in_scene[category].extend(matching_instances)

        def transform_action(action: str) -> str:
            parts = action.split(" ", 1)
            if len(parts) < 2:
                return action  # e.g., "ACTION" only

            base_action, target_obj = parts

            # WAIT carries a numeric duration, not an object id to normalize.
            if base_action == "WAIT":
                return action

            # If the target object is already valid in the scene, no change needed
            if target_obj in all_object_ids_in_scene:
                return action

            # Determine the candidate list based on the action type
            if base_action == "NAVIGATE_TO":
                candidates = all_object_ids_in_scene
            elif base_action in ["PLACE_INSIDE", "PLACE_ON_TOP"]:
                candidates = object_map_in_scene.get("RECEPTACLE", [])
                split_obj = target_obj.split(" ", 1)
                if len(split_obj) > 1:
                    obj1, obj2 = split_obj
                    obj1_type = obj1.split("|")[0]
                    obj2_type = obj2.split("|")[0]

                    receptacle_types = object_categories.get("RECEPTACLE", [])
                    recepacles = set()
                    for receptacle_type in receptacle_types:
                        receptacle_type = receptacle_type.split("|")[0]
                        recepacles.add(receptacle_type)

                    is_obj1_receptacle = obj1_type in recepacles
                    is_obj2_receptacle = obj2_type in recepacles

                    if is_obj2_receptacle:
                        target_obj = obj2
                    elif is_obj1_receptacle:
                        target_obj = obj1
                    else:
                        raise ValueError(
                            f"For action '{action}', neither '{obj1}' nor '{obj2}' is a valid receptacle."
                        )
                else:
                    # When only one object is specified for a PLACE action, it's assumed to be the receptacle.
                    target_obj = target_obj
            else:
                # Fallback for other actions, get candidates of the same type
                target_obj_type = target_obj.split("|")[0]
                candidates = [
                    obj_id
                    for obj_id in all_object_ids_in_scene
                    if obj_id.startswith(target_obj_type)
                ]
                # If no candidates of the same type, use all objects as a last resort
                if not candidates:
                    candidates = all_object_ids_in_scene

            # If there are no candidates, we cannot correct the action
            if not candidates:
                log.warning(f"Could not find any candidates for action: {action}")
                return action

            # Find the most similar valid object and replace it
            matched = cls._sentence_sim_model.get_similar_ref(target_obj, candidates)
            log.debug(
                f"Correcting object in action '{action}': replaced '{target_obj}' with '{matched}'"
            )
            return f"{base_action} {matched}"

        # Correct actions for all subtasks
        subtasks = cls.tasks_to_subtasks(tasks, mode="all")
        for st in subtasks:
            st.execution.primitive_actions = [
                transform_action(a) for a in st.execution.primitive_actions
            ]

        return tasks

    @classmethod
    def build_tasks_and_constraints(
        cls,
        task_data: dict,
        scene_file_name: str,
        enable_decomposition: bool = True,
    ) -> Tuple[List[Subtask], DiGraph]:
        """
        1) JSON 형태의 raw task_data를 Task로 파싱
        2) Object ID 검사 + 액션 정제(check_obj_id, refine_primitive_actions)
        3) 필요 시 enable_decomposition=True → 서브태스크 분해
        4) critical/non-critical constraint의 interval 값을 규칙 기반으로 설정하고, belief 딕셔너리 생성
        5) 생성된 belief 딕셔너리를 파일(bayesian_estimate.json)에 저장 (디버깅용)
        6) 다시 액션 정제 + Subtask duration 보정
        7) TaskGraph 빌드

        :param task_data: JSON 형태(이미 파싱된 Dict)
        :param enable_decomposition: 서브태스크 분해 여부
        :return: (최종 Subtask 리스트, TaskGraph 객체)
        """
        from src.models.task import Task, TaskGraphBuilder

        # 재현 가능한 실험을 위해 시드 고정
        random.seed(42)

        # 1) 빈 belief 딕셔너리 생성 (파일 로드 제거)
        bayesian_load = {}

        # 2) Task 파싱, Object ID/액션 보정
        tasks = Task.parse_instruction(task_data)
        tasks = cls.check_obj_id(scene_file_name, tasks)
        tasks = cls.refine_primitive_actions(tasks)
        tasks = cls.repair_held_object_semantics(tasks)

        # 3) enable_decomposition 옵션 처리
        if enable_decomposition:
            tasks = [t.decompose_subtasks() for t in tasks]

        # 4) temporal constraint 처리 (규칙 기반 Interval 설정 포함)
        subtasks = cls.tasks_to_subtasks(tasks, mode="all")
        # 효율적인 조회를 위해 subtask 이름과 객체를 매핑
        subtask_map = {st.name: st for st in subtasks}

        for st in subtasks:
            current_obj_types = set()
            if st.execution and st.execution.objects:
                for obj_name in st.execution.objects.keys():
                    current_obj_types.add(obj_name.split("|")[0])

            # 각 temporal constraint 별로 순회하여 `tc` 변수 범위 문제 해결
            for tc in st.temporal_constraints:
                # 1. 제약 조건(tc)으로 연결된 두 subtask가 '공유하는' 객체를 찾음
                # NOTE: 공유 객체 체크 로직을 임시로 비활성화함.
                # 대신 현재 Subtask가 가진 객체를 기준으로 Interval을 업데이트함.
                # related_subtask = subtask_map.get(tc.rel_subtask_name)
                # related_obj_types = set()
                # if (
                #     related_subtask
                #     and related_subtask.execution
                #     and related_subtask.execution.objects
                # ):
                #     for obj_name in related_subtask.execution.objects.keys():
                #         related_obj_types.add(obj_name.split("|")[0])
                #
                # common_obj_types = current_obj_types.intersection(related_obj_types)

                # 공유 여부를 따지지 않고 현재 Subtask의 객체를 사용
                common_obj_types = current_obj_types

                # 2. 조건을 만족하는 '모든' 객체에 대해 Belief 생성
                if common_obj_types:
                    for obj_type in common_obj_types:
                        if obj_type in constants.CRITICAL_OBJECT_INTERVALS:
                            # Force urgency to True if it's a critical object
                            tc.is_critical = True
                            cls._update_constraint_belief(obj_type, tc, bayesian_load)
                        elif obj_type in constants.NON_CRITICAL_OBJECT_INTERVALS:
                            tc.is_critical = False
                            cls._update_constraint_belief(obj_type, tc, bayesian_load)

        # 3. (안전장치) LLM이 temporal_constraint를 생성하지 않았더라도, Belief를 가지도록 보장
        for st in subtasks:
            if st.execution and st.execution.objects:
                for obj_name in st.execution.objects.keys():
                    obj_type = obj_name.split("|")[0]
                    if (
                        obj_type in constants.CRITICAL_OBJECT_INTERVALS
                        and obj_type not in bayesian_load
                    ):
                        default_tc = TemporalConstraint(
                            constraint_type="default",
                            rel_subtask_name="",
                            interval=constants.INIT_PRIOR_MEAN,
                            is_critical=True,
                        )
                        cls._update_constraint_belief(
                            obj_type, default_tc, bayesian_load
                        )

        # 6) 액션 정제(재적용) + duration 조정
        tasks = cls.refine_primitive_actions(tasks)
        subtasks = cls.tasks_to_subtasks(tasks, mode="all")
        subtasks = cls.adjust_subtasks_duration(subtasks)

        # 7) TaskGraph 빌드
        task_graph_builder = TaskGraphBuilder()
        task_graph = task_graph_builder.build_graph(tasks)

        # 8) 그래프 엣지에 Variance 정보 주입 (Bayesian Update를 위해)
        for st in subtasks:
            current_obj_types = set()
            if st.execution and st.execution.objects:
                for obj_name in st.execution.objects.keys():
                    current_obj_types.add(obj_name.split("|")[0])

            for tc in st.temporal_constraints:
                related_subtask = subtask_map.get(tc.rel_subtask_name)
                if not related_subtask:
                    continue

                related_obj_types = set()
                if related_subtask.execution and related_subtask.execution.objects:
                    for obj_name in related_subtask.execution.objects.keys():
                        related_obj_types.add(obj_name.split("|")[0])

                common_obj_types = current_obj_types.intersection(related_obj_types)

                variance_val = constants.INIT_PRIOR_VARIANCE
                found_variance = False
                for obj_type in common_obj_types:
                    if obj_type in bayesian_load:
                        variance_val = bayesian_load[obj_type]["variance"]
                        found_variance = True
                        break

                if not found_variance:
                    # bayesian_load에 없어도 default tc 처리가 위에서 되었을 수 있음.
                    # 하지만 안전하게 기본값 사용 혹은 패스.
                    # 여기서는 IsCritical인 경우만 중요하므로 IsCritical 체크를 하면 좋겠지만
                    # 엣지에만 넣는 것이므로 무조건 넣어도 무방.
                    pass

                # 엣지 방향 확인 및 업데이트
                # TemporalConstraint의 방향성을 정확히 모르므로 양방향 체크
                if task_graph.has_edge(st.name, tc.rel_subtask_name):
                    if "info" not in task_graph.edges[st.name, tc.rel_subtask_name]:
                        task_graph.edges[st.name, tc.rel_subtask_name]["info"] = {}
                    task_graph.edges[st.name, tc.rel_subtask_name]["info"][
                        "Variance"
                    ] = variance_val

                if task_graph.has_edge(tc.rel_subtask_name, st.name):
                    if "info" not in task_graph.edges[tc.rel_subtask_name, st.name]:
                        task_graph.edges[tc.rel_subtask_name, st.name]["info"] = {}
                    task_graph.edges[tc.rel_subtask_name, st.name]["info"][
                        "Variance"
                    ] = variance_val

        return subtasks, task_graph, bayesian_load

    @classmethod
    def _update_constraint_belief(
        cls,
        obj_type: str,
        temporal_constraint: "TemporalConstraint",
        bayesian_load: dict,
    ) -> None:
        """
        obj_type을 Key로 사용하여 bayesian_load에 Belief를 생성/업데이트한다.
        agent.py와 호환되도록 {"expected_duration": ..., "variance": ...} 구조를 사용한다.
        """
        edge_interval_value = (
            constants.INIT_PRIOR_MEAN if temporal_constraint.interval != 0.0 else 0.0
        )
        object_prior_value = (
            constants.INIT_PRIOR_MEAN
            if temporal_constraint.is_critical
            else edge_interval_value
        )

        if obj_type not in bayesian_load:
            bayesian_load[obj_type] = {
                "expected_duration": object_prior_value,
                "variance": constants.INIT_PRIOR_VARIANCE,
            }
        elif (
            temporal_constraint.is_critical
            and bayesian_load[obj_type]["expected_duration"] <= 0.0
        ):
            bayesian_load[obj_type]["expected_duration"] = constants.INIT_PRIOR_MEAN

        # Edge interval은 tDAG의 시간 의미를 유지해야 하므로 0,True는 그대로 둡니다.
        temporal_constraint.interval = edge_interval_value

    @staticmethod
    def get_init_state(
        subtasks: List[Subtask],
        constraints: DiGraph,
        scene_poses: dict,
    ) -> SchedulerState:
        """
        Init Subtask 및 SchedulerState를 구성한다.
        """
        init_subtask = Subtask(
            task_name=None,
            name="Init",
            duration=Duration(interval=0, type="Init"),
            repetition=1,
            subtask_type="Init",
            execution=Execution(objects=[], primitive_actions=None),
            temporal_constraints=None,
        )
        init_completed = CompletedEntry(
            subtask=init_subtask,
        )

        init_state = SchedulerState(
            subtask=init_subtask,
            completed_entries=[init_completed],
            remaining_subtasks=subtasks,
            constraints=constraints,
            current_time=0,
            scene_positions=scene_poses,
            held_object=None,
        )
        return init_state

    @staticmethod
    def create_monitoring_subtask(name: str, obj: str = None) -> Subtask:
        """
        MONITORING 액션을 수행하는 Subtask를 생성해 반환한다.
        """
        base_name = name
        if base_name.startswith("Monitoring for "):
            base_name = base_name[len("Monitoring for ") :]
        monitoring_action = None if obj is None else [f"MONITORING {obj}"]
        monitoring_subtask = Subtask(
            task_name=None,
            name=f"Monitoring for {base_name}_{uuid.uuid4().hex[:8]}",
            duration=Duration(
                interval=MONITORING_DURATION,
                type="Monitor",
                total_time=MONITORING_DURATION,
            ),
            repetition=1,
            subtask_type="Monitor",
            execution=Execution(objects=[obj], primitive_actions=monitoring_action),
            temporal_constraints=None,
            decomposed=True,
        )
        return monitoring_subtask
