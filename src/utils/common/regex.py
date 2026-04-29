import re


def extract_monitoring_target_name(obj_name: str) -> str:
    """
    'Monitoring for (.*?)_' 패턴에서 실제 모니터링 대상 subtask 이름을 추출.
    예: "Monitoring for Make_Coffee_sub_1" -> "Make_Coffee_sub"
    """
    m = re.search(r"Monitoring for (.*?)_", obj_name)
    if not m:
        raise ValueError(f"Invalid monitoring subtask name: {obj_name}")
    return m.group(1)
