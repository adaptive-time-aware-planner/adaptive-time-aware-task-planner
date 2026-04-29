import random

from ithor.concepts.actions import *
from ithor.utils.object_utils import detect_manipulable_objs
from src.utils.config.constants import OBJECT_INTERESTS


class InteractionHandler:
    def __init__(self, controller):
        self.controller = controller

    def random_interact(self, object_infos: dict):
        # agent와 가장 가까운 object 선택
        # Predefined된 가능한 Interaction내에서 선택
        obj_id, obj_info = random.choice(list(object_infos.items()))
        obj_position = obj_info.get("position")
        available_interactions = list(
            set(obj_info.get("interactions"))
            & set(OBJECT_INTERESTS["object_interactions"])
        )

        obj_state = obj_info.get("states")

        if len(available_interactions) == 1:
            selected_action = available_interactions[0]
        elif len(available_interactions) >= 1:
            selected_action = random.choice(available_interactions)
        else:
            raise ValueError("No interactions available for this object.")

        self.execute(selected_action, obj_id, obj_state)

    def execute(self, interaction: str, obj_id: str, obj_state=None):
        if interaction == "manipulable":
            if self.controller.last_event.metadata["arm"]["heldObjects"]:
                drop_object(self.controller)
            else:
                pick_up_with_arm(self.controller)
        elif interaction == "pickupable":
            if "isPickedUp" in obj_state:
                drop(self.controller, obj_id)
            else:
                pick_up(self.controller, obj_id)
        elif interaction == "toggleable":
            if "isToggled" in obj_state:
                toggle_off(self.controller, obj_id)
            else:
                toggle_on(self.controller, obj_id)
        elif interaction == "sliceable":
            slice(self.controller, obj_id)
        elif interaction == "openable":
            if "isOpen" in obj_state:
                close(self.controller, obj_id)
            else:
                open(self.controller, obj_id)
        # elif action == "receptacle":
        #     # Attempt to put the held object into/on the target receptacle
        #     put(self.controller, obj_id)
        else:
            raise ValueError(f"Unknown action '{interaction}' for object '{obj_id}'.")

    def detect_objects(self):
        """
        Detects interactable and visible objects in the scene and returns their information.
        """
        object_infos = {}
        objects = [
            obj
            for obj in self.controller.last_event.metadata["objects"]
            if obj["visible"] and obj["isInteractable"]
        ]

        for obj in objects:
            obj_id = obj["objectId"]
            obj_interactions = [
                interaction
                for interaction in OBJECT_INTERESTS["object_interactions"]
                if obj.get(interaction)
            ]

            if detect_manipulable_objs(self.controller):
                obj_interactions.append("manipulable")

            obj_states = [
                state for state in OBJECT_INTERESTS["object_states"] if obj.get(state)
            ]

            if obj_interactions or obj_states:
                object_infos[obj_id] = {
                    "type": obj["objectType"],
                    "position": obj["position"],
                    "interactions": obj_interactions,
                    "states": obj_states,
                }
                print(f"Object ID: {obj_id}")
                print(f"Object interactions: {obj_interactions}")
                print(f"Object states: {obj_states}\n")

        return object_infos
