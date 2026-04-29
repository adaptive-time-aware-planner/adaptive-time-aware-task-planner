import signal
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar("T", bound=Callable[..., Any])


def retry(retries: int = 3, delay: int = 1) -> Callable[[T], T]:
    """
    재시도 데코레이터.
    지정한 예외가 발생하면 주어진 횟수만큼 재시도합니다.

    Args:
        retries (int): 재시도 횟수. 기본값 3.
        delay (int): 재시도 간 간격(초). 기본값 1초.

    Returns:
        Callable: 원래 함수에 데코레이터 적용된 함수
    """

    def decorator(func: T) -> T:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except (RuntimeError, ValueError, TypeError) as e:
                    print(f"[Retry {attempt + 1}/{retries}] Failed: {e}")
                    time.sleep(delay)
            raise RuntimeError(
                f"Function '{func.__name__}' failed after {retries} retries."
            )

        return wrapper  # type: ignore

    return decorator


class Timeout:
    """
    함수 실행 시간을 제한하는 컨텍스트 매니저.
    지정한 시간 내에 종료되지 않으면 TimeoutError 발생.

    Usage:
        with Timeout(seconds=3):
            do_something()

    Args:
        seconds (int): 제한 시간 (초). 기본값 1초.
        error_message (str): 타임아웃 발생 시 메시지.
    """

    def __init__(self, seconds: int = 1, error_message: str = "Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)


@contextmanager
def timeout_context(seconds: int = 1, error_message: str = "Timeout"):
    """
    timeout용 컨텍스트 매니저 (함수 형태).
    Timeout 클래스의 함수형 대안.

    Usage:
        with timeout_context(seconds=2):
            do_something()

    Args:
        seconds (int): 제한 시간 (초). 기본값 1초.
        error_message (str): 에러 메시지. 기본값 'Timeout'
    """

    def handler(signum, frame):
        raise TimeoutError(error_message)

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def time_logger(func):
    """시간초 기록 데코레이터.
    지정한 함수의 실행 시간을 측정합니다.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed_time = end - start
        return result, elapsed_time

    return wrapper
