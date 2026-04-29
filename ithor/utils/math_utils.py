# math_utils.py
import json
import math
from typing import Dict, Set, Tuple

from src.utils.common import create_module_logger
from src.utils.config.constants import GRID_SIZE, SCENE_KNOWLEDGE_PATH

log = create_module_logger(module_name=__name__, module_log=True)


def load_navigation_graph(
    controller,
) -> Dict[Tuple[float, float, float], Set[Tuple[float, float, float]]]:
    """
    AI2THOR에서 한 번만 호출해서 reachable positions를 받아,
    '대각 이동 없이' 한 축만 차이 나는 위치들끼리 연결한 그래프를 만든다.
    """
    positions_data = controller.step("GetReachablePositions").metadata["actionReturn"]
    positions = [quantize_position((p["x"], p["y"], p["z"])) for p in positions_data]

    neighbors = {}
    for pos in positions:
        neighbors[pos] = set()

    for pos in positions:
        for other in positions:
            if pos == other:
                continue
            dx = abs(pos[0] - other[0])
            dy = abs(pos[1] - other[1])
            dz = abs(pos[2] - other[2])
            # 원본처럼 sum(dx,dy,dz)가 GRID_SIZE와 거의 같으면 neighbor
            if abs((dx + dy + dz) - GRID_SIZE) < 1e-5:
                neighbors[pos].add(other)

    return neighbors


def adjust_if_unreachable(
    nav_graph: Dict[Tuple[float, float, float], Set[Tuple[float, float, float]]],
    pos: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    qpos = quantize_position(pos)
    while qpos not in nav_graph:
        qpos = _adjust_to_reachable(nav_graph, qpos)

    return qpos


def _adjust_to_reachable(nav_graph, pos):
    # unreachable이면 nav_graph 중 가장 가까운 위치로 보정
    pos = quantize_position(pos)
    if pos in nav_graph:
        return pos
    # 없으면 전체 노드 중 최단거리
    all_reachable = list(nav_graph.keys())
    return closest_position(pos, all_reachable)


def closest_grid_point(pos):
    """Helper function to find the closest grid point to a given position."""
    return tuple(round(coord, 1) for coord in pos)


def closest_position(object_position, reachable_positions):
    """Find the closest reachable position to the given object."""
    min_distance = float("inf")
    closest = reachable_positions[0]
    for pos in reachable_positions:
        dist = euclidean_distance(object_position, pos)
        if dist < min_distance:
            min_distance = dist
            closest = pos
    return closest


def quantize_position(pos):
    return tuple(round(coord / GRID_SIZE) * GRID_SIZE for coord in pos)


def euclidean_distance(pointA, pointB):
    """Compute the Euclidean distance between two 3D points."""
    return (
        (pointA[0] - pointB[0]) ** 2
        + (pointA[1] - pointB[1]) ** 2
        + (pointA[2] - pointB[2]) ** 2
    ) ** 0.5


def calculate_rotation_angle(agent_position, object_position):
    agent_x = agent_position[0]
    agent_z = agent_position[2]
    obj_x = object_position[0]
    obj_z = object_position[2]

    a = euclidean_distance([agent_x, agent_z], [obj_x, obj_z])
    b = euclidean_distance([agent_x, agent_z], [agent_x - 2, agent_z])
    c = euclidean_distance([obj_x, obj_z], [agent_x - 2, agent_z])

    gamma = math.degrees(math.acos((a**2 + b**2 - c**2) / (2 * a * b)))
    return gamma
