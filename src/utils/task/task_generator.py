# utils/task/task_generator.py
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
from dotenv import load_dotenv

from src.utils.common import create_module_logger
from src.utils.config.constants import (
    ENV_PLACEHOLDER,
    PROMPT_PATH,
    SCENE_KNOWLEDGE_PATH,
    TASK_PATH,
    TOP_K,
)

# from src.utils.nlp.few_shot_retriever import FewShotRetriever
from src.utils.task.task_cache import check_cache, get_cache_key, store_cache

# 내부 모듈
from src.utils.task.task_validators import validate_output_format

# .env 파일 로딩 (OPENAI_API_KEY 등)
load_dotenv()

logger = create_module_logger(__name__)


def initialize_openai() -> openai.OpenAI:
    """
    OpenAI API 클라이언트를 초기화한다.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables.")
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables.")
    return openai.OpenAI(api_key=openai_api_key)


class TaskGenerator:
    """
    태스크 생성 전반에 관한 로직을 관리하는 클래스.
    - OpenAI Client
    - Prompt 구성
    - 결과 검증/재생성
    - 캐싱
    등
    """

    def __init__(
        self,
        is_rag: bool = False,
        is_test: bool = False,
        use_cache: bool = True,
        prompt_file_override: Optional[str] = None,
    ):
        self.client = initialize_openai()
        self.is_rag = is_rag
        self.is_test = is_test
        self.use_cache = use_cache
        self.prompt_file_override = prompt_file_override
        # 추가적으로 필요한 상태(예: knowledge 등) 여기서 관리 가능

    def load_file(self, file_path: Path, file_type: str) -> Any:
        """
        지정된 파일 경로에서 데이터를 로딩한다.
        """
        if not file_path.exists():
            raise FileNotFoundError(
                f"{file_type.capitalize()} file not found: {file_path}"
            )

        with open(file_path, "r", encoding="utf-8") as f:
            if file_type == "json":
                return json.load(f)
            else:
                return f.read()

    def generate_task(
        self, user_input: str, scene_name: Optional[str] = "FloorPlan1"
    ) -> str:
        """
        유저 입력 + Knowledge Base + Prompt를 조합하여 태스크를 생성한다.
        결과를 파일에 저장하고, 파일명을 반환한다.

        Args:
            user_input: The user's input instruction string.
            environment_file_name: Optional name of the JSON file in ENVIRONMENT_PATH
                                    containing environment details.
        """

        user_input = user_input.strip()
        if not user_input:
            raise ValueError("User input cannot be empty.")
        scene_number = int(scene_name.replace("FloorPlan", ""))
        if scene_number > 400:
            scene_type = "bathroom"
        elif scene_number > 300:
            scene_type = "real_world"
        else:
            scene_type = "kitchen"

        # Prompt(예제) 로드
        if self.prompt_file_override:
            prompt_path = Path(PROMPT_PATH) / self.prompt_file_override
        elif scene_type in {"bathroom", "kitchen"}:
            prompt_path = Path(PROMPT_PATH) / "e2e_generator_ver12_kitchen.txt"
        else:
            prompt_path = Path(PROMPT_PATH) / "e2e_generator_ver13_real.txt"

        examples_prompt = self.load_file(prompt_path, "txt")

        # # RAG 모드 활성화 시, FewShotRetriever 활용
        # if self.is_rag:
        #     rag_system = FewShotRetriever()
        #     retrieved_few_shot_prompts = rag_system.generate_few_shot_prompts(
        #         user_input, top_k=TOP_K
        #     )
        #     examples_prompt = examples_prompt.replace(
        #         "<Example>", retrieved_few_shot_prompts
        #     )

        # 환경 정보 로드 및 주입 (environment_file_name이 제공된 경우)
        environment_info_str = ""

        try:
            env_file_path = (
                Path(SCENE_KNOWLEDGE_PATH)
                / scene_type
                / "environment"
                / f"{scene_name}_physics_environment.json"
            )
            env_data = self.load_file(env_file_path, "json")

            # 전체 환경 데이터를 JSON 문자열로 변환
            environment_info_str = json.dumps(env_data, indent=2, ensure_ascii=False)

        except FileNotFoundError:
            logger.warning(
                f"Environment file not found: {scene_name}. Proceeding without environment info."
            )
        except ValueError as e:  # Catches JSONDecodeError from load_file
            logger.warning(
                f"Error loading environment file {scene_name}: {e}. Proceeding without environment info."
            )
        except Exception as e:
            logger.error(
                f"Unexpected error processing environment file {scene_name}: {e}"
            )
            # Decide if you want to raise here or proceed without env info
            # raise

        # 프롬프트 템플릿에 환경 정보 삽입 (또는 플레이스홀더 제거)
        if ENV_PLACEHOLDER in examples_prompt:
            examples_prompt = examples_prompt.replace(
                ENV_PLACEHOLDER, environment_info_str
            )
        elif environment_info_str:  # Placeholder not found, but env info exists
            logger.warning(
                f"Placeholder '{ENV_PLACEHOLDER}' not found in prompt template, but environment info was loaded."
            )

        # Knowledge Base 로드
        # knowledge = self.load_file(Path(SCENE_KNOWLEDGE_PATH) / ESTIMATE_FILE_NAME, "json")

        # 최종 OpenAI 프롬프트
        full_prompt = [
            {"role": "system", "content": examples_prompt},
            {"role": "user", "content": f"# [Input]\n\n{user_input}"},
        ]

        # 캐싱 체크
        prompt_key = get_cache_key(full_prompt)
        cached_result = check_cache(prompt_key) if self.use_cache else None
        if cached_result:
            logger.info("Using cached result.")
            output = cached_result
        else:
            # OpenAI 호출
            output = self._call_openai(full_prompt)
            if output and validate_output_format(output):
                if self.use_cache:
                    store_cache(prompt_key, output)
            else:
                raise ValueError("Task generation failed. (Invalid Format)")

        # 파일로 저장
        # time_stamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        # sanitized_input = re.sub(r"[^\w\-_\.]+", "_", user_input[:30])
        # output_file_name = f"{time_stamp}_{sanitized_input}.json"
        output_file_name = f"{user_input}.json"
        output_file_path = Path(TASK_PATH) / output_file_name

        # self.save_to_file(output, output_file_path)
        if self.is_test:
            return output
        return output_file_name

    def _call_openai(
        self, full_prompt: List[Dict[str, str]]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        OpenAI API를 호출하여 태스크를 생성한다. (재시도 로직 포함)
        """
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=full_prompt,
                    temperature=0,
                )
                output_content = response.choices[0].message.content.strip()

                # JSON 코드블록 제거
                if "```json" in output_content:
                    output_content = output_content.strip("```json").strip("```")

                output = json.loads(output_content)
                if validate_output_format(output):
                    # validate_output_format이 output을 수정했으므로 수정된 버전을 반환
                    return output

            except json.JSONDecodeError as e:
                logger.error(f"[Attempt {attempt}] JSON decoding failed: {e}")
            except Exception as e:
                logger.error(f"[Attempt {attempt}] Unexpected error: {e}")

        logger.error("Failed to generate valid output after retries.")
        return None

    def save_to_file(self, data: Any, file_path: Path) -> None:
        """
        JSON 파일로 결과를 저장한다.
        """
        os.makedirs(file_path.parent, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Output saved to {file_path}.")


if __name__ == "__main__":
    task_generator = TaskGenerator()
    task_generator.generate_task(
        "boil_water_with_kettle and heat_the_potato_using_microwave and put_apple_and_lettuce_in_fridge and put_the_wine_bottle_inside_a_cabinet and put_the_creditcard_on_the_countertop"
    )
