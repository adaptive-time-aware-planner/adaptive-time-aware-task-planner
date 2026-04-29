# camera_views.py
import numpy as np

from src.utils.config.constants import CAMERA_DISTANCE_BEHIND, CAMERA_HEIGHT_ABOVE


class CameraView:
    def __init__(self, controller):
        self.controller = controller

    def set_view(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def update_view(self):
        pass  # 기본 구현

    def get_frame(self):
        raise NotImplementedError("Subclasses should implement this method.")


class EgocentricView(CameraView):
    def set_view(self):
        self.controller.step(
            action="UpdateThirdPartyCamera",
            thirdPartyCameraId=0,
            position={"x": 0, "y": -100, "z": 0},
        )

    def get_frame(self):
        return self.controller.last_event.frame


class ThirdPersonView(CameraView):
    def calculate_camera_pos_rot(self):
        agent_position = self.controller.last_event.metadata["agent"]["position"]
        agent_rotation = self.controller.last_event.metadata["agent"]["rotation"]

        theta = np.deg2rad(agent_rotation["y"])
        offset_x = -CAMERA_DISTANCE_BEHIND * np.sin(theta)
        offset_z = -CAMERA_DISTANCE_BEHIND * np.cos(theta)

        position = {
            "x": agent_position["x"] + offset_x,
            "y": agent_position["y"] + CAMERA_HEIGHT_ABOVE,
            "z": agent_position["z"] + offset_z,
        }

        rotation = {
            "x": 20,
            "y": agent_rotation["y"],
            "z": 0,
        }
        return position, rotation

    def set_view(self):
        position, rotation = self.calculate_camera_pos_rot()
        self.controller.step(
            action="AddThirdPartyCamera",
            position=position,
            rotation=rotation,
            fieldOfView=60,
        )

    def update_view(self):
        position, rotation = self.calculate_camera_pos_rot()
        self.controller.step(
            action="UpdateThirdPartyCamera",
            thirdPartyCameraId=0,
            position=position,
            rotation=rotation,
            fieldOfView=60,
        )

    def get_frame(self):
        return self.controller.last_event.third_party_camera_frames[0]


class TopDownView(CameraView):
    def calculate_camera_pos_rot(self):
        agent_position = self.controller.last_event.metadata["agent"]["position"]
        position = {
            "x": agent_position["x"],
            "y": agent_position["y"] + CAMERA_HEIGHT_ABOVE,
            "z": agent_position["z"],
        }
        rotation = {
            "x": 90,
            "y": 0,
            "z": 0,
        }
        return position, rotation

    def set_view(self):
        position, rotation = self.calculate_camera_pos_rot()
        self.controller.step(
            action="AddThirdPartyCamera",
            position=position,
            rotation=rotation,
            fieldOfView=90,
        )

    def update_view(self):
        position, rotation = self.calculate_camera_pos_rot()
        self.controller.step(
            action="UpdateThirdPartyCamera",
            position=position,
            rotation=rotation,
            fieldOfView=90,
        )

    def get_frame(self):
        return self.controller.last_event.third_party_camera_frames[0]
