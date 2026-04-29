import json
from pathlib import Path

from src.utils.config import SCENE_KNOWLEDGE_PATH


def save_the_agent_knowledge(scene_name: str, agent_knowledge: dict):

    # 저장할 디렉토리 경로

    SCENE_KNOWLEDGE_PATH.mkdir(parents=True, exist_ok=True)  # 경로가 없으면 생성

    # 파일 저장 경로
    file_path = SCENE_KNOWLEDGE_PATH / f"{scene_name}.json"

    # 파일 저장
    with file_path.open("w") as f:
        json.dump(agent_knowledge, f)


def load_agent_knowledge(scene_name: str):
    # 파일이 저장된 디렉토리 경로
    file_path = Path.cwd().joinpath(SCENE_KNOWLEDGE_PATH) / f"{scene_name}.json"
    try:
        with file_path.open("r") as f:
            agent_knowledge = json.load(f)
    except FileNotFoundError:
        agent_knowledge = {}

    return agent_knowledge


# from ai2thor.controller import Controller
#     c = Controller()
#     event = c.step('GetReachablePositions')
#     positions = event.metadata['reachablePositions']
#     for pos in positions:
#         for rotation in (0, 90, 180, 270):
#             c.step('TeleportFull', rotation=dict(x=0.0, y=rotation, z=0.0), **pos)
