class ArmHandler:
    def __init__(self, controller):
        self.controller = controller
        self.arm_position = {"x": 0, "y": 0, "z": 0}  # 초기값 설정

    def move_arm(
        self,
        delta_x=0,
        delta_y=0,
        delta_z=0,
        coordinate_space="wrist",
        restrict_movement=False,
        speed=1,
        return_to_start=True,
        fixed_delta_time=0.02,
    ):
        """
        팔을 현재 위치에서 상대적으로 이동시키는 메소드.

        :param delta_x: 이동할 x 변화량
        :param delta_y: 이동할 y 변화량
        :param delta_z: 이동할 z 변화량
        :param coordinate_space: 좌표계를 지정 ("wrist", "armBase", "world")
        :param restrict_movement: 팔이 제한된 범위 내에서만 이동하도록 설정
        :param speed: 팔의 움직임 속도 (미터/초)
        :param return_to_start: 충돌 시 시작 위치로 복귀할지 여부
        :param fixed_delta_time: 시뮬레이션에서 물리 단계 시간 간격
        """
        try:
            # 현재 팔 위치에서 변화량을 더하여 새로운 목표 위치 계산
            target_position = {"x": delta_x, "y": delta_y, "z": delta_z}

            event = self.controller.step(
                action="MoveArm",
                position=target_position,
                coordinateSpace=coordinate_space,
                restrictMovement=restrict_movement,
                speed=speed,
                returnToStart=return_to_start,
                fixedDeltaTime=fixed_delta_time,
            )

        except Exception as e:
            print(f"MoveArm 액션 중 에러 발생: {str(e)}")

    def move_arm_base(
        self, delta_y, speed=1, return_to_start=True, fixed_delta_time=0.02
    ):
        """
        팔의 기반 높이를 상대적으로 조정하는 메소드.

        :param delta_y: 이동할 y 변화량 (0과 1 사이의 값)
        :param speed: 팔의 기반 이동 속도 (미터/초)
        :param return_to_start: 충돌 시 시작 위치로 복귀할지 여부
        :param fixed_delta_time: 시뮬레이션에서 물리 단계 시간 간격
        """
        try:
            # 현재 팔 기반의 y 위치 가져오기
            arm_base_current_y = self.controller.last_event.metadata["arm"]["joints"][
                0
            ]["rootRelativePosition"]["y"]

            # 새로운 목표 y 위치 계산
            target_y = arm_base_current_y + delta_y
            # y 값은 0과 1 사이로 클램핑
            target_y = max(0.0, min(1.0, target_y))

            self.controller.step(
                action="MoveArmBase",
                y=target_y,
                speed=speed,
                returnToStart=return_to_start,
                fixedDeltaTime=fixed_delta_time,
            )

        except Exception as e:
            print(f"MoveArmBase 액션 중 에러 발생: {str(e)}")
