import copy
import logging
import math
from collections import deque
from typing import Dict, List, Optional, Tuple, TypeAlias

from ithor.utils.math_utils import adjust_if_unreachable
from src.models.dataclass import ActionResult, ActionSimulationLog, SimulationNode
from src.utils.common import create_module_logger
from src.utils.config.constants import (
    ACTION_SPLIT_PRECISION,
    EPSILON,
    GRASP_ACTION_DURATION,
    MONITORING_DURATION,
    NAV_STEP_DURATION,
    PLACE_ACTION_DURATION,
    REACHABLE_DISTANCE_THRESHOLD,
    REAL_NAV_DURATION,
    STATIC_ACTION_SET,
    TIMING_TOLERANCE_ABS,
    TOGGLE_ACTION_DURATION,
)

log = create_module_logger(__name__, module_log=True, level=logging.DEBUG)

Position: TypeAlias = Tuple[float, float, float]
NavGraph: TypeAlias = Dict[Position, List[Position]]  # 네비게이션 그래프 타입 정의
ActionCacheKey: TypeAlias = Tuple[
    Optional[str],
    Optional[str],
    Tuple[Tuple[str, object], ...],
    Tuple[str, ...],
    bool,
]


class ActionHandler:
    def __init__(self, nav_graph: NavGraph, real_world_mode: bool = False):
        """
        ActionHandler를 초기화합니다.

        Args:
            nav_graph: 네비게이션에 사용될 그래프. {position_tuple: [neighbor_position_tuples]} 형식.
            logger: The logger instance to use.
        """
        self.nav_graph = nav_graph
        self.real_world_mode = real_world_mode
        self._search_action_cache: Optional[
            Dict[ActionCacheKey, Optional[ActionResult]]
        ] = None
        self._search_cache_hits = 0
        self._search_cache_misses = 0

    def begin_search_session(
        self, cache: Dict[ActionCacheKey, Optional[ActionResult]]
    ) -> None:
        """Attach a search-scoped cache for action simulation reuse.

        Args:
            cache: Mutable cache owned by the scheduler for the current search call.
        """

        self._search_action_cache = cache
        self._search_cache_hits = 0
        self._search_cache_misses = 0

    def end_search_session(self) -> tuple[int, int]:
        """Detach the search-scoped action cache and return cache statistics.

        Returns:
            A tuple of `(cache_hits, cache_misses)` collected in the current search.
        """

        stats = (self._search_cache_hits, self._search_cache_misses)
        self._search_action_cache = None
        self._search_cache_hits = 0
        self._search_cache_misses = 0
        return stats

    def _build_action_cache_key(
        self, current_node: SimulationNode, actions: List[str]
    ) -> ActionCacheKey:
        """Build a deterministic cache key for an action simulation request.

        Args:
            current_node: Starting node for the simulated action sequence.
            actions: Primitive actions to simulate.

        Returns:
            Hashable key that captures the state elements affecting simulation.
        """

        state = current_node.state
        normalized_scene_positions = tuple(
            sorted(
                (
                    name,
                    tuple(value) if isinstance(value, (list, tuple)) else value,
                )
                for name, value in state.scene_positions.items()
            )
        )
        return (
            state.held_object,
            state.agent_location,
            normalized_scene_positions,
            tuple(actions),
            self.real_world_mode,
        )

    def get_actions_info(
        self, current_node: SimulationNode, actions: list[str]
    ) -> Optional[ActionResult]:
        """
        주어진 액션 시퀀스를 시뮬레이션하고 최종 ActionResult를 반환합니다.
        시뮬레이션을 위해 노드 상태의 복사본을 생성합니다.
        액션 목록이 비어 있거나 시뮬레이션에 실패하면 None을 반환합니다.

        Args:
            current_node: 시뮬레이션을 시작할 현재 노드 상태.
            actions: 실행할 원시 액션 문자열 목록.

        Returns:
            시뮬레이션 성공 시 마지막 액션의 결과(ActionResult), 실패 시 None.
        """
        if not actions:
            log.warning(
                "get_actions_info called with empty actions list. Returning None."
            )
            return None

        cache_key: Optional[ActionCacheKey] = None
        if self._search_action_cache is not None:
            cache_key = self._build_action_cache_key(current_node, actions)
            if cache_key in self._search_action_cache:
                self._search_cache_hits += 1
                return self._search_action_cache[cache_key]

        action_sim_info: Optional[ActionSimulationLog] = self._simulate_actions(
            current_node, actions
        )
        self._search_cache_misses += 1

        # action_sim_info가 None이거나, 결과가 비어있는 경우 처리
        if not action_sim_info or not action_sim_info.results:
            log.debug("Action simulation did not produce any results. Returning None.")
            if cache_key is not None and self._search_action_cache is not None:
                self._search_action_cache[cache_key] = None
            return None

        # 마지막 ActionResult 가져오기
        last_action_result: ActionResult = action_sim_info.results[-1]

        first_nav_duration = 0.0
        if action_sim_info.results:  # 결과가 하나라도 있는지 확인
            first_action_in_log = action_sim_info.results[0]
            if first_action_in_log.action_type.upper() == "NAVIGATE_TO":
                first_nav_duration = first_action_in_log.action_duration

        try:
            last_action_result.first_nav_duration = first_nav_duration
        except AttributeError:
            log.warning(
                "ActionResult dataclass does not have 'first_nav_duration' attribute. "
                "This information will not be directly added to the returned ActionResult object. "
                "Consider modifying the ActionResult dataclass or using the ActionSimulationLog object."
            )

        if cache_key is not None and self._search_action_cache is not None:
            self._search_action_cache[cache_key] = last_action_result
        return last_action_result

    # --------------------------------------------------------------------------
    # 액션 시뮬레이션 핵심 로직 (_simulate_actions 및 헬퍼 메서드)
    # --------------------------------------------------------------------------

    def _simulate_actions(
        self,
        initial_node: SimulationNode,
        primitive_actions: List[str],
    ) -> Optional[ActionSimulationLog]:
        """
        주어진 액션 시퀀스를 내부 모델을 사용하여 순차적으로 시뮬레이션합니다.
        시뮬레이션은 상태의 깊은 복사본을 사용하여 수행됩니다.
        액션 실행 중 실패가 발생하면 시뮬레이션을 중단하고 현재까지의 로그를 반환합니다.

        Args:
            initial_node: 시뮬레이션 시작 상태를 담은 노드.
            primitive_actions: 실행할 원시 액션 문자열 리스트.

        Returns:
            액션 시뮬레이션 로그(ActionSimulationLog). 시뮬레이션 시작 자체가 불가능하거나
            치명적인 오류 발생 시 None을 반환할 수 있습니다. (예: 초기 상태 오류)
        """
        if not initial_node or not initial_node.state:
            log.error(
                "Cannot simulate actions with invalid initial_node or state. Returning None."
            )
            return None

        action_log = ActionSimulationLog()
        # 시뮬레이션을 위해 상태 깊은 복사
        sim_state = copy.deepcopy(initial_node.state)
        current_scene_positions = sim_state.scene_positions
        current_held_object = sim_state.held_object
        new_held_object = sim_state.held_object
        current_cumulative_time = 0.0
        for i, action_str in enumerate(primitive_actions):
            # log.debug(
            #     f"--- Simulating action {i+1}/{len(primitive_actions)}: '{action_str}' ---"
            # )
            # log.debug(
            #     f"    State before: Time={current_cumulative_time:.2f}, Held={current_held_object}"
            # )

            # 액션 파싱
            tokens = action_str.split()
            if not tokens:
                log.warning(f"Empty action string encountered at index {i}. Skipping.")
                continue  # 다음 액션으로

            action_type = tokens[0].upper()
            target_obj_id: Optional[str] = tokens[1] if len(tokens) > 1 else None
            # NAVIGATE_TO의 경우 부분 시간 정보 추가 파싱
            partial_time_str: Optional[str] = (
                tokens[2] if len(tokens) == 3 and action_type == "NAVIGATE_TO" else None
            )

            action_duration = 0.0
            action_success = True

            agent_pos: Optional[Position] = (
                tuple(current_scene_positions.get("agent"))
                if "agent" in current_scene_positions
                else None
            )
            if agent_pos is None:
                raise ValueError("Agent position ('agent') missing in scene_positions.")

            # 액션 타입별 헬퍼 메서드 호출
            if action_type == "NAVIGATE_TO":
                action_duration, action_success, new_agent_pos = (
                    self._simulate_navigate(
                        agent_pos,
                        target_obj_id,
                        partial_time_str,
                        current_scene_positions,
                    )
                )
                if action_success and new_agent_pos is not None:
                    current_scene_positions["agent"] = tuple(new_agent_pos)

            elif action_type == "GRASP":
                action_duration, action_success, new_held_object = self._simulate_grasp(
                    agent_pos,
                    target_obj_id,
                    current_held_object,
                    current_scene_positions,
                )

            elif action_type in ["PLACE_INSIDE", "PLACE_ON_TOP"]:
                action_duration, action_success, new_held_object = self._simulate_place(
                    agent_pos,
                    target_obj_id,
                    current_held_object,
                    current_scene_positions,
                )

            elif action_type in STATIC_ACTION_SET:
                action_duration, action_success = self._simulate_interaction(
                    agent_pos, action_type, target_obj_id, current_scene_positions
                )

            elif action_type == "WAIT":
                action_duration, action_success = self._simulate_wait(target_obj_id)
            elif action_type == "MONITORING":
                action_duration, action_success = self._simulate_monitoring()
            else:
                log.warning(
                    f"Unhandled action type in internal simulation: {action_type}. Assuming default duration and failure."
                )
                action_duration = 0.0
                action_success = False

            # 시뮬레이션 상태 업데이트 (헬퍼 메서드에서 변경된 부분 반영)
            current_held_object = new_held_object

            # 경과 시간 및 누적 시간 업데이트
            current_cumulative_time += action_duration

            log_entry = ActionResult(
                action_full_name=action_str,
                action_type=action_type,
                cumulative_time=current_cumulative_time,
                action_duration=action_duration,
                scene_positions=copy.deepcopy(current_scene_positions),
                held_object=copy.deepcopy(current_held_object),
                success=action_success,
            )
            action_log.results.append(log_entry)
            # log.debug(
            #     f"    Action Result: Success={action_success}, Duration={action_duration:.2f}"
            # )
            # log.debug(
            #     f"    State after:  Time={current_cumulative_time:.2f}, Held={current_held_object}"
            # )
            # 액션 실패 시 시뮬레이션 중단
            if not action_success:
                log.debug(
                    f"Action '{action_str}' failed. Stopping simulation sequence."
                )
                break

        return action_log

    def _check_reachability(
        self,
        agent_pos: Position,
        target_pos: Position,
        action_name: str,
        target_id: Optional[str],
    ) -> bool:
        """주어진 위치에서 타겟 위치가 상호작용 가능한 거리 내에 있는지 확인합니다."""
        dist = math.dist(agent_pos, target_pos)
        if dist > REACHABLE_DISTANCE_THRESHOLD:
            log.debug(
                f"  {action_name} target '{target_id}' might be unreachable "
                f"(Distance: {dist:.2f} > {REACHABLE_DISTANCE_THRESHOLD:.2f}). Action FAILED."
            )
            return False
        return True

    def _simulate_navigate(
        self,
        agent_pos: Position,
        target_obj_id: Optional[str],
        partial_time_str: Optional[str],
        scene_positions: Dict[str, Position],
    ) -> Tuple[float, bool, Optional[Position]]:
        """
        NAVIGATE_TO 액션을 시뮬레이션합니다. 전체 경로 또는 부분 시간을 기반으로
        소요 시간, 성공 여부, 그리고 액션 후 에이전트의 최종 위치를 계산합니다.
        Args:
            agent_pos: 현재 에이전트 위치.
            target_obj_id: 목표 객체 ID.
            partial_time_str: 부분 이동 시간을 나타내는 문자열 (있는 경우).
            scene_positions: 현재 씬의 객체 위치 정보.
        Returns:
            Tuple[float, bool, Optional[Position]]:
            - 소요 시간 (float).
            - 성공 여부 (bool).
            - 액션 후 에이전트의 새로운 위치 (Optional[Position]). 부분 시간 이동 시
              정확한 최종 위치를 알 수 없으면 None일 수 있음.
        Raises:
            ValueError: 목표 객체를 찾을 수 없거나, 부분 시간 문자열이 잘못된 경우.
        """
        duration = 0.0
        success = False
        new_agent_pos: Optional[Position] = None  # 액션 완료 후의 최종 위치

        if self.real_world_mode:
            return REAL_NAV_DURATION, True, (0, 0, 0)

        # 1. 목표 유효성 검사 및 위치 가져오기
        if not target_obj_id or target_obj_id not in scene_positions:
            log.warning(
                "Navigation target '%s' not found in scene positions. Action FAILED.",
                target_obj_id,
            )
            return 0.0, False, agent_pos
        target_pos = tuple(scene_positions[target_obj_id])
        # 2. 경로 탐색 시도 (partial time 여부와 관계없이 일단 시도)
        navigate_path: Optional[List[Position]] = None

        # log.debug(
        #     f"  Finding path from {agent_pos} to {target_pos} for '{target_obj_id}'"
        # )
        try:
            navigate_path = self._find_shortest_path(agent_pos, target_pos)
        except ValueError:
            log.warning(
                "No navigation path found from %s to '%s'. Action FAILED.",
                agent_pos,
                target_obj_id,
            )
            return 0.0, False, agent_pos
        # 3. 부분 시간 이동 처리
        if partial_time_str:
            # log.debug(f"  Processing NAVIGATE_TO with partial time: {partial_time_str}")
            partial_duration = float(partial_time_str)
            duration = partial_duration  # 액션 소요 시간은 주어진 부분 시간

            # 이동할 스텝 수 계산 (올림/내림 정책 확인 필요, 여기선 내림 사용)
            steps_can_take = int(math.floor(partial_duration / NAV_STEP_DURATION))
            path_after_start = navigate_path[1:] if navigate_path else []
            # 경로 길이 내에서만 이동 가능
            actual_steps = min(steps_can_take, len(path_after_start))

            if path_after_start and actual_steps > 0:
                new_agent_pos = path_after_start[actual_steps - 1]
            else:
                new_agent_pos = agent_pos
            success = True  # 부분 시간 이동은 일단 성공으로 간주

        # 4. 전체 경로 이동 처리
        else:
            # log.debug("  Processing NAVIGATE_TO for full path.")
            # ithor의 action.py에서는 첫좌표를 제거하므로 여기서도 동일하게 제거
            if navigate_path:
                navigate_path.pop(0)
            path_steps = len(navigate_path)  # 실제 이동 스텝 수
            duration = round(path_steps * NAV_STEP_DURATION, 2)
            # 경로의 마지막 위치가 새로운 에이전트 위치 (경로가 비었으면 현재 위치)
            new_agent_pos = navigate_path[-1] if navigate_path else agent_pos
            success = True
            # log.debug(
            #     f"    Path found to {target_obj_id} with {path_steps} steps. Duration: {duration:.2f}s. Final pos: {new_agent_pos}"
            # )
        # 5. 결과 반환
        return duration, success, new_agent_pos

    def _simulate_grasp(
        self,
        agent_pos: Position,
        target_obj_id: Optional[str],
        current_held_object: Optional[str],
        scene_positions: Dict[str, Position],
    ) -> Tuple[float, bool, Optional[str]]:
        """GRASP 액션을 시뮬레이션합니다."""
        duration = 0.0
        success = False
        new_held_object = current_held_object

        if not target_obj_id or target_obj_id not in scene_positions:
            raise ValueError(
                f"Grasp target '{target_obj_id}' not found in scene positions."
            )
        if current_held_object:
            log.warning(
                f"Agent already holding '{current_held_object}'. Cannot grasp '{target_obj_id}'. Action FAILED."
            )
            success = False
        else:
            target_actual_pos = tuple(scene_positions[target_obj_id])
            if self._check_reachability(
                agent_pos, target_actual_pos, "Grasp", target_obj_id
            ):
                new_held_object = target_obj_id
                # Use per-action configured duration
                duration = GRASP_ACTION_DURATION
                success = True
                # log.debug(f"  Grasped '{target_obj_id}'.")
            else:
                success = False  # Unreachable

        return duration, success, new_held_object

    def _simulate_place(
        self,
        agent_pos: Position,
        receptacle_id: Optional[str],
        current_held_object: Optional[str],
        scene_positions: Dict[str, Position],
    ) -> Tuple[float, bool, Optional[str]]:
        """PLACE_INSIDE 또는 PLACE_ON_TOP 액션을 시뮬레이션합니다."""
        duration = 0.0
        success = False
        new_held_object = current_held_object

        if not current_held_object:
            log.warning("Agent not holding anything. Cannot place. Action FAILED.")
            success = False
        elif not receptacle_id or receptacle_id not in scene_positions:
            raise ValueError(
                f"Place target receptacle '{receptacle_id}' not found in scene positions."
            )
        else:
            receptacle_pos = tuple(scene_positions[receptacle_id])
            if self._check_reachability(
                agent_pos, receptacle_pos, "Place", receptacle_id
            ):
                # log.debug(f"  Placing '{current_held_object}' on/in '{receptacle_id}'.")
                # 객체 상태 업데이트 (시뮬레이션 모델에 따라 달라짐)
                # 여기서는 단순히 손을 비우는 것으로 처리
                if current_held_object in scene_positions:
                    scene_positions[current_held_object] = scene_positions[
                        receptacle_id
                    ]
                new_held_object = None
                # Use per-action configured duration
                duration = PLACE_ACTION_DURATION
                success = True
            else:
                success = False  # Unreachable

        return duration, success, new_held_object

    def _simulate_interaction(
        self,
        agent_pos: Position,
        action_type: str,
        target_obj_id: Optional[str],
        scene_positions: Dict[str, Position],
    ) -> Tuple[float, bool]:
        """OPEN, CLOSE, TOGGLE_ON/OFF, SLICE 등 일반적인 상호작용 액션을 시뮬레이션합니다."""
        duration = 0.0
        success = False

        if not target_obj_id or target_obj_id not in scene_positions:
            raise ValueError(
                f"{action_type} target '{target_obj_id}' not found in scene positions."
            )

        target_actual_pos = tuple(scene_positions[target_obj_id])
        if self._check_reachability(
            agent_pos, target_actual_pos, action_type, target_obj_id
        ):
            # Map static interactions to configured durations
            # OPEN, CLOSE, TOGGLE_ON, TOGGLE_OFF, SLICE, FILL → TOGGLE_ACTION_DURATION by default
            duration_map = {
                "OPEN": TOGGLE_ACTION_DURATION,
                "CLOSE": TOGGLE_ACTION_DURATION,
                "TOGGLE_ON": TOGGLE_ACTION_DURATION,
                "TOGGLE_OFF": TOGGLE_ACTION_DURATION,
                "SLICE": TOGGLE_ACTION_DURATION,
                "FILL": TOGGLE_ACTION_DURATION,
            }
            duration = duration_map.get(action_type, TOGGLE_ACTION_DURATION)

            success = True
            # log.debug(f"  Simulated {action_type} on '{target_obj_id}'.")
        else:
            success = False  # Unreachable

        return duration, success

    def _simulate_wait(self, wait_time_str: Optional[str]) -> Tuple[float, bool]:
        """WAIT 액션을 시뮬레이션합니다."""
        duration = 0.0
        success = False
        if wait_time_str is None:
            raise ValueError("WAIT action requires a duration argument.")
        try:
            wait_time = float(wait_time_str)
            if wait_time < 0:
                log.warning(f"Invalid negative wait time: {wait_time}. Using 0.")
                duration = 0.0
            else:
                duration = wait_time
            success = True
            # log.debug(f"  Simulated WAIT for {duration:.2f}s.")
        except (TypeError, ValueError):
            raise ValueError(f"Invalid WAIT duration: {wait_time_str}")

        return duration, success

    def _simulate_monitoring(self) -> Tuple[float, bool]:
        """MONITORING 액션을 시뮬레이션합니다."""
        duration = MONITORING_DURATION
        success = True
        # log.debug(f"  Simulated MONITORING for {duration:.2f}s.")
        return duration, success

    def _find_shortest_path(
        self, start_pos: Tuple[float, float, float], end_pos: Tuple[float, float, float]
    ) -> list[Tuple[float, float, float]]:
        """Return the minimum-step path between two navigation nodes.

        Args:
            start_pos: Starting position in the navigation graph.
            end_pos: Goal position in the navigation graph.

        Returns:
            Sequence of positions including the start and end nodes.

        Raises:
            ValueError: If no path exists between the two positions.
        """

        start_pos = adjust_if_unreachable(self.nav_graph, start_pos)
        end_pos = adjust_if_unreachable(self.nav_graph, end_pos)
        if start_pos == end_pos:
            return []

        queue = deque([[start_pos]])
        visited = {start_pos}

        while queue:
            path = queue.popleft()
            cur_pos = path[-1]
            if cur_pos == end_pos:
                return path
            for nxt in self.nav_graph.get(cur_pos, []):
                if nxt in visited:
                    continue
                visited.add(nxt)
                queue.append(path + [nxt])

        raise ValueError(f"No path found from {start_pos} to {end_pos}.")

    def split_subtask_by_cutoff_time(
        self,
        current_node: SimulationNode,
        primitive_actions: List[str],
        target_cutoff_time: float,
    ) -> Tuple[ActionSimulationLog, ActionSimulationLog, bool, bool]:
        """
        주어진 액션 시퀀스를 지정된 target_cutoff_time 기준으로 두 부분으로 나눕니다.
        1. 전체 시퀀스를 시뮬레이션합니다.
        2. target_cutoff_time 내에 완료되는 액션들로 초기 pre_log를 구성합니다.
        3. pre_log의 마지막 액션에서 객체를 들고 있다면(GRASP 이후),
           다음 PLACE 액션까지를 pre_log에 포함시키려 시도합니다.
        4. 단, 이 조정으로 인해 pre_log의 완료 시간이 target_cutoff_time을 기준으로
           TIMING_TOLERANCE 허용 범위(비율/절대)를 초과하지 않는 경우에만 조정을 확정합니다.
        5. 분할 성공 여부와 pre_log 완료 시 객체를 들고 있는지 여부를 함께 반환합니다.

        Args:
            current_node: 분할을 시작할 기준 노드.
            primitive_actions: 분할 대상 액션 시퀀스.
            target_cutoff_time: 분할 기준이 되는 목표 상대 시간.

        Returns:
            Tuple: (pre_actions_log, post_actions_log, split_successful, pre_ends_holding_object)
                   split_successful: 유의미한 분할이 이루어졌는지 여부.
                   pre_ends_holding_object: pre_actions_log 완료 시 객체를 들고 있는지 여부.
        """
        log.debug(
            f"Attempting to split actions with target_cutoff_time: {target_cutoff_time:.2f} "
            f"(Node time: {current_node.state.current_time:.2f})"
        )

        pre_log = ActionSimulationLog()
        post_log = ActionSimulationLog()
        split_successful = False
        pre_ends_holding_object = False

        # target_cutoff_time이 너무 작으면 분할 의미 없음 (상대 시간이므로 0보다 커야 함)
        if target_cutoff_time < EPSILON:
            log.debug(
                f"Target cutoff time {target_cutoff_time:.2f} is too small. No effective split will be performed."
            )
            # 전체 액션 시뮬레이션 결과를 pre_log로 간주 (분할 실패)
            full_sim_log = self._simulate_actions(current_node, primitive_actions)
            if full_sim_log and full_sim_log.results:
                pre_log.results = full_sim_log.results[:]
                last_action_res = pre_log.results[-1]
                pre_ends_holding_object = last_action_res.held_object is not None
            return pre_log, post_log, False, pre_ends_holding_object

        # 1. 전체 액션 시퀀스 시뮬레이션
        full_simulation_log = self._simulate_actions(current_node, primitive_actions)

        if not full_simulation_log or not full_simulation_log.results:
            log.error(
                "Full internal simulation failed or produced no results. Returning empty logs and split_failed."
            )
            return pre_log, post_log, False, False

        # 2. 초기 분할 지점 결정: target_cutoff_time을 넘지 않는 마지막 액션
        initial_split_index = -1
        for i, result in enumerate(full_simulation_log.results):
            if result.cumulative_time <= target_cutoff_time + EPSILON:
                initial_split_index = i
            else:
                break  # cutoff_time을 초과하는 첫 액션 앞에서 멈춤
        log.debug(
            f"Initial split index based on target_cutoff_time ({target_cutoff_time:.2f}): {initial_split_index}"
        )

        # 모든 액션이 cutoff 이후이거나, 액션이 아예 없는 경우
        if initial_split_index == -1:
            if full_simulation_log.results:  # 액션은 있지만 모두 cutoff 이후
                post_log.results = full_simulation_log.results[:]
                log.debug(
                    "All actions occur after target_cutoff_time. Pre-log is empty."
                )
                # post_log에만 액션이 있다면 유의미한 분할은 아님 (또는 성공으로 볼 수도 있음, 정책에 따라)
                return pre_log, post_log, False, False  # 여기서는 분할 실패로 간주
            else:  # 액션이 아예 없는 경우
                log.debug("No actions in primitive_actions. Both logs empty.")
                return pre_log, post_log, False, False

        # --- 초기 분할 상태 ---
        current_split_index = initial_split_index

        # 3. GRASP / PLACE 제약 조건 및 TIMING_TOLERANCE 기반 분할 지점 조정
        last_action_at_initial_split = full_simulation_log.results[initial_split_index]
        object_held_at_initial_split = last_action_at_initial_split.held_object

        if object_held_at_initial_split is not None:
            log.debug(
                f"Object '{object_held_at_initial_split}' is held at initial split index {initial_split_index} "
                f"(time: {last_action_at_initial_split.cumulative_time:.2f}). Checking for subsequent PLACE action."
            )

            found_place_action_index_in_full_log = -1
            for j, next_result in enumerate(
                full_simulation_log.results[initial_split_index + 1 :]
            ):
                if next_result.action_type.startswith("PLACE_"):
                    found_place_action_index_in_full_log = initial_split_index + 1 + j
                    break

            if found_place_action_index_in_full_log != -1:
                duration_if_place_included = full_simulation_log.results[
                    found_place_action_index_in_full_log
                ].cumulative_time

                # TIMING_TOLERANCE 검사: 비율과 절대 허용폭을 함께 고려
                allowable_deviation = ACTION_SPLIT_PRECISION

                if (
                    abs(duration_if_place_included - target_cutoff_time)
                    <= allowable_deviation
                ):
                    log.debug(
                        f"Found subsequent PLACE action. Including it in pre_log. "
                        f"New pre_log duration: {duration_if_place_included:.2f} (target_cutoff: {target_cutoff_time:.2f}, "
                        f"deviation: {abs(duration_if_place_included - target_cutoff_time):.2f} <= allowable: {allowable_deviation:.2f})."
                    )
                    current_split_index = found_place_action_index_in_full_log
                    pre_ends_holding_object = False  # PLACE로 끝났으므로
                else:
                    log.debug(
                        f"Found subsequent PLACE action, but including it would make pre_log duration ({duration_if_place_included:.2f}) "
                        f"exceed target_cutoff_time ({target_cutoff_time:.2f}) beyond the timing tolerance window (allowable deviation: {allowable_deviation:.2f}). "
                        f"Splitting after GRASP at index {initial_split_index}."
                    )
                    # GRASP/PLACE 묶기 포기, 초기 분할 지점 유지.
                    pre_ends_holding_object = True  # 초기 분할 지점에서 들고 있었음
            else:  # PLACE 액션이 아예 없는 경우
                log.debug(
                    f"Object '{object_held_at_initial_split}' was held at initial split index {initial_split_index}, "
                    f"but no subsequent PLACE action was found. EARLY_ subtask will end holding the object."
                )
                pre_ends_holding_object = True
        else:  # 초기 분할 지점에서 객체를 들고 있지 않은 경우
            pre_ends_holding_object = False
            log.debug(
                f"No object held at initial split index {initial_split_index}. No GRASP/PLACE adjustment needed."
            )

        # 4. 최종 로그 생성
        pre_log.results = full_simulation_log.results[: current_split_index + 1]
        post_log.results = full_simulation_log.results[current_split_index + 1 :]

        # 5. 분할 성공 여부 판단
        # pre_log와 post_log 모두에 액션이 있고, pre_log에 네비게이션 외의 의미 있는 액션이 있어야
        # 유의미한 분할로 간주
        if pre_log.results and post_log.results:
            has_meaningful_action = any(
                not result.action_type.startswith("NAVIGATE_TO")
                for result in pre_log.results
            )

            if has_meaningful_action:
                split_successful = True
                # 추가적으로, pre_log의 최종 완료 시간이 target_cutoff_time 대비 허용 오차 창을 만족하는지 확인
                final_pre_log_duration = pre_log.results[-1].cumulative_time
                allowable_deviation = ACTION_SPLIT_PRECISION
                if (
                    abs(final_pre_log_duration - target_cutoff_time)
                    > allowable_deviation + EPSILON
                ):  # EPSILON 추가는 부동소수점 오차 감안
                    log.warning(
                        f"Final pre_log duration {final_pre_log_duration:.2f} significantly deviates from target_cutoff_time {target_cutoff_time:.2f} "
                        f"(allowable deviation: {allowable_deviation:.2f}). This split might be suboptimal."
                    )
                    # 이 경우, split_successful을 False로 설정하여 상위에서 fallback하도록 할 수도 있음.
                    split_successful = False  # 정책에 따라 주석 해제 또는 변경
            else:
                split_successful = False
                log.debug(
                    "Split deemed unsuccessful as pre_log only contains navigation actions."
                )
        else:
            split_successful = (
                False  # pre 또는 post 중 하나라도 비어있으면 유의미한 분할 아님
            )
            if not pre_log.results:
                log.debug("No actions in pre_log after split attempt.")
            if not post_log.results:
                log.debug(
                    "No actions in post_log after split attempt. All actions in pre_log."
                )

        log.debug(
            f"Final split index: {current_split_index}. "
            f"Pre-log ({len(pre_log.results)} actions, ends holding: {pre_ends_holding_object}). "
            f"Post-log ({len(post_log.results)} actions). Split successful: {split_successful}"
        )

        return pre_log, post_log, split_successful, pre_ends_holding_object
