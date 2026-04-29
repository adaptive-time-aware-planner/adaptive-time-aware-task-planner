### simulation/extractor.py
from typing import Dict, List, Tuple

from ai2thor.controller import Controller

from ithor.handlers.navigation_handler import NavigationHandler


def extract_object_positions(
    controller: Controller,
) -> Dict[str, Tuple[float, float, float]]:
    controller.step("Pass")
    metadata = controller.last_event.metadata

    positions = {
        "agent": (
            metadata["agent"]["position"]["x"],
            metadata["agent"]["position"]["y"],
            metadata["agent"]["position"]["z"],
        )
    }

    for obj in metadata["objects"]:
        pos = obj["position"]
        positions[obj["objectId"]] = (pos["x"], pos["y"], pos["z"])

    return positions


def extract_environment(controller: Controller) -> Tuple[dict, List[str]]:
    controller.step("Pass")
    metadata = controller.last_event.metadata

    categories = {
        "openable": [],
        "toggleable": [],
        "pickupable": [],
        "sliceable": [],
        "receptacle": [],
    }
    for obj in metadata["objects"]:
        for key in categories:
            if obj.get(key, False):
                categories[key].append(obj["objectType"])

    for key in categories:
        categories[key] = list(set(categories[key]))

    env = {
        "OPEN": [],
        "CLOSE": [],
        "TOGGLE_ON": [],
        "TOGGLE_OFF": [],
        "GRASP": [],
        "SLICE": [],
        "RECEPTACLE": [],
    }
    object_ids = []

    mapping = {
        "openable": ["OPEN", "CLOSE"],
        "toggleable": ["TOGGLE_ON", "TOGGLE_OFF"],
        "pickupable": ["GRASP"],
        "sliceable": ["SLICE"],
        "receptacle": ["RECEPTACLE"],
    }

    for attr, actions in mapping.items():
        for obj_type in categories[attr]:
            ids = controller.step(
                action="ObjectTypeToObjectIds", objectType=obj_type
            ).metadata["actionReturn"]
            for action_type in actions:
                env[action_type].extend(ids)
            object_ids.extend(ids)

    return env, list(set(object_ids))


def extract_navigation_time(controller: Controller, object_ids: List[str]) -> dict:
    move_time = {"agent": {}}
    nav_handler = NavigationHandler(controller)

    agent_pos = nav_handler.get_agent_position()
    for obj in object_ids:
        obj_pos = nav_handler.get_object_position(obj)
        path = nav_handler.find_shortest_path(agent_pos, obj_pos)
        move_time["agent"][obj] = round(len(path) * 0.1, 2)

    for obj1 in object_ids:
        move_time[obj1] = {}
        for obj2 in object_ids:
            pos1 = nav_handler.get_object_position(obj1)
            pos2 = nav_handler.get_object_position(obj2)
            path = nav_handler.find_shortest_path(pos1, pos2)
            move_time[obj1][obj2] = round(len(path) * 0.1, 2)

    return move_time
