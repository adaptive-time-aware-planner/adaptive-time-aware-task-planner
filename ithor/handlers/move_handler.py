from src.utils.config.constants import *


class MoveHandler:
    def __init__(self, controller, camera_handler):
        self.controller = controller
        self.camera_handler = camera_handler
        self.move_step = MOVE_STEP
        self.rotate_step = ROTATE_STEP
        self.run_speed_multiplier = RUN_SPEED_MULTIPLIER

    def move_agent(self, move_action, run_mode=False):
        speed = (
            self.move_step * self.run_speed_multiplier if run_mode else self.move_step
        )
        self.controller.step(action=move_action, moveMagnitude=speed)

        self.camera_handler.update_view()

    def rotate_agent(self, rotate_action):
        self.controller.step(action=rotate_action, degrees=self.rotate_step)

        self.camera_handler.update_view()
