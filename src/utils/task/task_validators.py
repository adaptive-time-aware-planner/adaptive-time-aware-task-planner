# utils/task/validators.py

import logging
from typing import Any

logger = logging.getLogger(__name__)


def validate_output_format(output: Any) -> bool:
    """
    생성된 Task 출력 포맷의 유효성을 검증한다.
    [
      {
        "Task": "...",
        "Subtasks": [
          {
            "Name": "...",
            "Repetition": "...",
            "Type": "...",
            "Executions": {
              "Objects": [...],
              "PrimitiveActions": [...]
            },
            "Duration": "...",
            "TemporalConstraints": "..."
          },
          ...
        ]
      },
      ...
    ]

    Returns:
        bool: 포맷이 유효하면 True, 아니면 False
    """
    if not isinstance(output, list):
        return False

    for task in output:
        if not isinstance(task, dict) or not {"Task", "Subtasks"}.issubset(task):
            return False

        for subtask in task.get("Subtasks", []):
            required_keys = {
                "Name",
                "Repetition",
                "Type",
                "Executions",
                "Duration",
                "TemporalConstraints",
            }
            if not isinstance(subtask, dict) or not required_keys.issubset(subtask):
                return False

            executions = subtask.get("Executions", {})
            if not isinstance(executions, dict) or not {
                "Objects",
                "PrimitiveActions",
            }.issubset(executions):
                return False

    return True
