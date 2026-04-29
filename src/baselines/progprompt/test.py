import argparse
import json
import os
import os.path as osp
import random
import sys
import time

import numpy as np
import openai
import prior
from ai2thor.controller import Controller
from ithor.utils.constants import *
from util.utils_execute import *

from ithor.handlers.action import Action

# setting.json 에 ai2thor 위치 환경변수 추가한 상태로 해야함.
from ithor.handlers.arm_handler import ArmHandler
from ithor.handlers.camera_handler import CameraHandler
from ithor.handlers.interaction_handler import InteractionHandler
from ithor.handlers.move_handler import MoveHandler
from ithor.handlers.navigation_handler import NavigationHandler
from ithor.utils.file_utils import *


def initialize_controller():
    # dataset = prior.load_dataset("procthor-10k")
    # house = dataset["train"][0]
    controller = Controller(
        agentMode="default",  # "default", "locobot", "drone", or "arm",
        massThreshold=0.04,  # 물리 엔진에서 물체를 움직이는 최소 질량
        scene="FloorPlan1_physics",  # Scene 이름
        gridSize=GRID_SIZE,  # Move Actions의 Mean
        movementGaussianSigma=0.005,  # Move Actions의 Sigma
        renderDepthImage=False,  # Depth Image 렌더링 여부 (오랜 시간 소요)
        renderInstanceSegmentation=False,  # Instance Segmentation 렌더링 여부 (오랜 시간 소요)
        width=SCREEN_WIDTH,
        height=SCREEN_HEIGHT,
        renderThirdPartyCameras=False,
        fieldOfView=60,
    )
    log_file = open("multi_scene_test.txt", "w", buffering=1)
    camera_handler = CameraHandler(controller)
    navi = NavigationHandler(controller, camera_handler)
    act = Action(controller, camera_handler, log_file)

    return controller, camera_handler, navi, act


def get_parent_receptacle(controller, object_id: str):
    # 해당 object의 부모 receptacle을 찾는 로직 구현
    object_metadata = controller.last_event.metadata["objects"]

    # 예시로 object의 metadata에서 parent receptacle을 가져오는 코드 작성
    # 실제로는 controller의 메타데이터나 객체 속성에 따라 다를 수 있음
    for obj in object_metadata:
        if obj["objectId"] == object_id:
            if "parentReceptacles" in obj:
                parent_receptacle_ids = obj["parentReceptacles"]
                break
    for rec in parent_receptacle_ids:
        for obj in object_metadata:
            if obj["objectId"] == rec and obj["visible"]:
                parent_receptacle_id = rec
                break

    return parent_receptacle_id


def openn(self, controller, object_id: str):
    # 일단 두 발자국 물러나기
    for i in range(2):
        controller.step(action="MoveBack", moveMagnitude=None)
        controller.step(action="Pass")
    camera_handler.update_view()
    time.sleep(0.1)

    # 열기
    controller.step(
        action="OpenObject", objectId=object_id, openness=1, forceAction=False
    )

    controller.step(action="Pass")
    camera_handler.update_view()
    time.sleep(0.3)


def closee(self, controller, object_id: str):
    controller.step(action="CloseObject", objectId=object_id, forceAction=False)
    controller.step(action="Pass")
    camera_handler.update_view()
    time.sleep(0.3)


if __name__ == "__main__":

    controller, camera_handler, navi, act = initialize_controller()
    controller.step("MoveBack")
    controller.step("Pass")

    scene_name = controller.step("Pass").metadata["sceneName"]
    objs = controller.step("Pass").metadata["objects"]
    openable = []
    toggleable = []
    pickupable = []
    receptacle = []

    for obj in objs:
        if obj["openable"]:
            openable.append(obj["objectType"])
        if obj["toggleable"]:
            toggleable.append(obj["objectType"])
        if obj["pickupable"]:
            pickupable.append(obj["objectType"])
        if obj["receptacle"]:
            receptacle.append(obj["objectType"])

    openable = list(set(openable))
    toggleable = list(set(toggleable))
    pickupable = list(set(pickupable))
    receptacle = list(set(receptacle))

    env = {
        scene_name: {
            "OPEN": openable,
            "CLOSE": openable,
            "TOGGLE_ON": toggleable,
            "TOGGLE_OFF": toggleable,
            "GRASP": pickupable,
            "RECEPTACLE": receptacle,
        }
    }

    with open("metadata.json", "w") as json_file:
        json.dump(env, json_file, indent=4)
    with open("metadata.txt", "w", buffering=1) as txt_file:
        txt_file.write(str(env))

    egg = "Egg|-02.04|+00.81|+01.24"

    result = controller.step(
        action="PickupObject",
        objectId=egg,
        forceAction=True,
        manualInteract=False,
    )
    print(f"결과= {result.metadata['lastActionSuccess']}")
    # 물체를 집은 후의 결과 처리
    if not result.metadata["lastActionSuccess"]:
        # 물체를 집지 못한 경우, parent receptacle을 열고 다시 시도
        receptacle_id = get_parent_receptacle(controller, "Egg|-02.04|+00.81|+01.24")
        print(f"{receptacle_id=}")

        if receptacle_id:
            # parent receptacle을 열기
            openn(controller, receptacle_id)
            time.sleep(0.5)  # 잠시 대기하여 열리도록 시간 확보

            # 물체를 다시 집기 시도
            print("일")
            result = controller.step(
                action="PickupObject",
                objectId=egg,
                forceAction=True,
                manualInteract=False,
            )
            print("이")
            print(result.metadata["lastActionSuccess"])
            print(result.metadata["errorMessage"])
            if result.metadata["lastActionSuccess"]:
                print("삼")
                # 물체를 성공적으로 집었으면 receptacle을 다시 닫기
                closee(controller, receptacle_id)

