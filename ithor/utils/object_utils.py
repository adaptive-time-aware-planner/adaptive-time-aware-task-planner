from .math_utils import calculate_rotation_angle, closest_position


# object_utils.py
def detect_manipulable_objs(controller):
    manipulable_objects = set(
        controller.last_event.metadata["arm"]["pickupableObjects"]
    )
    held_objects = set(controller.last_event.metadata["arm"]["heldObjects"])
    return manipulable_objects - held_objects if manipulable_objects else None


def get_object_position(controller, object_id):
    """
    Retrieves the position of the object with the given ID.
    Raises an error if the object is not found.
    """
    for obj in controller.last_event.metadata["objects"]:
        if obj["objectId"] == object_id:
            return (
                obj["position"]["x"],
                obj["position"]["y"],
                obj["position"]["z"],
            )
    raise ValueError(f"Object with ID {object_id} not found.")


def get_agent_position(controller):
    """
    Retrieves the agent's current position.
    """
    agent_position = controller.last_event.metadata["agent"]["position"]
    return (agent_position["x"], agent_position["y"], agent_position["z"])


def obj_in_scene(controller, object_type):
    """현재 scene에 object type과 일치하는 object가 있는지 확인"""
    types_in_scene = sorted(
        [
            obj["objectType"]
            for obj in controller.last_event.metadata["objects"]
            if obj["visible"] and obj["isInteractable"]
        ]
    )
    assert object_type in types_in_scene, "Object not in scene"
    return next(
        obj
        for obj in controller.last_event.metadata["objects"]
        if obj["objectType"] == object_type
    )


def rotate_to_object(controller, object_type):
    """
    Rotates the agent to face the specified object type.
    """
    obj = obj_in_scene(controller, object_type)
    obj_position = obj["position"]
    agent_position = controller.last_event.metadata["agent"]["position"]

    rotation_angle = calculate_rotation_angle(agent_position, obj_position)
    controller.step(action="RotateRight", degrees=rotation_angle)


# def teleport_to_object(controller, object_type):
#     """
#     Teleports the agent close to the specified object type and rotates to face it.
#     """
#     obj = obj_in_scene(controller, object_type)
#     reachable_positions = controller.step(action="GetReachablePositions").metadata[
#         "actionReturn"
#     ]
#     closest_pos = closest_position(obj["position"], reachable_positions)
#     controller.step(action="Teleport", position=closest_pos)
#     rotate_to_object(controller, object_type)
#     return obj["objectId"]
