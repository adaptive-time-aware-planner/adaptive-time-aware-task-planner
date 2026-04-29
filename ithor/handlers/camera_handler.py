# camera_handler.py
from ithor.concepts.camera_views import EgocentricView, ThirdPersonView, TopDownView


class CameraHandler:
    def __init__(self, controller):
        self.controller = controller
        self.camera_modes = ["egocentric", "third_person", "top_down"]
        self.camera_mode = "egocentric"
        self.current_view = EgocentricView(self.controller)
        self.current_view.set_view()

    def set_view(self, mode):
        if mode == "egocentric":
            self.current_view = EgocentricView(self.controller)
        elif mode == "third_person":
            self.current_view = ThirdPersonView(self.controller)
        elif mode == "top_down":
            self.current_view = TopDownView(self.controller)
        else:
            raise ValueError(f"Unknown camera mode: {mode}")
        self.current_view.set_view()
        self.camera_mode = mode

    def toggle_view(self):
        current_index = self.camera_modes.index(self.camera_mode)
        next_index = (current_index + 1) % len(self.camera_modes)
        next_mode = self.camera_modes[next_index]
        self.set_view(next_mode)

    def update_view(self):
        self.current_view.update_view()

    def get_current_frame(self):
        return self.current_view.get_frame()
