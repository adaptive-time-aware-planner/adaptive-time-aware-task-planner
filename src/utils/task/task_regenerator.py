# utils/task/regenerators.py
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def regenerate_invalid_subtasks(
    client, invalid_subtasks: List[Dict[str, Any]], knowledge: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    유효하지 않은 서브태스크들을 OpenAI를 통해 재생성한다.
    """
    regenerated_subtasks = []
    valid_actions_str = ", ".join(list(knowledge.get("Valid_actions", {}).keys()))

    for subtask in invalid_subtasks:
        invalid_actions = subtask.get("InvalidActions", [])
        regenerate_prompt = [
            {
                "role": "user",
                "content": (
                    f"Invalid actions: {', '.join(invalid_actions)}.\n"
                    f"Valid actions: {valid_actions_str}.\n"
                    f"Please correct the subtask:\n"
                    f"{json.dumps(subtask, indent=4, ensure_ascii=False)}"
                ),
            }
        ]
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=regenerate_prompt,
                temperature=0,
            )
            regenerated_content = response.choices[0].message.content.strip()
            # JSON 코드블록 제거 로직
            regenerated_content = regenerated_content.strip("```json").strip("```")

            new_subtask = json.loads(regenerated_content)
            regenerated_subtasks.append(new_subtask)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
        except ValueError as e:
            logger.error(f"Value error during subtask regeneration: {e}")
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Network error during API call: {e}")
        except AttributeError as e:
            logger.error(f"Attribute error in response handling: {e}")

    return regenerated_subtasks


def regenerate_subtasks_parallel(
    client, invalid_subtasks: List[Dict[str, Any]], knowledge: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    유효하지 않은 서브태스크들을 멀티 스레딩 방식으로 병렬 재생성한다.
    """
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for subtask in invalid_subtasks:
            futures.append(
                executor.submit(
                    regenerate_invalid_subtasks, client, [subtask], knowledge
                )
            )

        for future in futures:
            results.extend(future.result())

    return results
