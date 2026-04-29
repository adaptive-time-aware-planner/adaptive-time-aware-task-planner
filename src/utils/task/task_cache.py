# utils/task/cache.py
import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# 전역/클래스 수준 등으로 관리할 수 있음
task_cache: Dict[str, List[Dict[str, Any]]] = {}


def get_cache_key(prompt: Any) -> str:
    """
    프롬프트로부터 캐싱용 key를 생성한다.
    """
    return json.dumps(prompt, sort_keys=True)


def check_cache(key: str) -> Any:
    """
    캐시에 key가 있으면 반환, 없으면 None
    """
    return task_cache.get(key)


def store_cache(key: str, value: Any) -> None:
    """
    결과를 캐시에 저장한다.
    """
    task_cache[key] = value
    logger.info(f"Cache stored. key={key[:60]}...")  # 필요시 앞부분만 로깅
