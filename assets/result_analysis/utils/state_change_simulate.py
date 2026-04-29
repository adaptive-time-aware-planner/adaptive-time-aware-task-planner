#state chage.json을 읽고 변화를 누적하는 파일
#매 스텝마다 체크하고자하는 상태에 도달했는지 알아야함
#체크하고자 하는 task list
#각 task 별로 체크하고자 하는 상태 list
#매 스텝마다 상태list가 달성 되었는가 확인.
# "state_change"  object 별로 누적. object는 유일한 name을 가짐.
#unified_analysis.py에서 파싱된 parsed_tasks를 input으로 받아.
#미리정의된 조건들 전체가 있으면 그중에 봐야할게 이 parsed_tasks로 결정됨.
#각 parsed_task 별로 GCR 달성 여부를 체크
#각 parsed_task 중 TSR달성 여부를 체크
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence



def accumulate_state_changes(
    events: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Dict[str, Any]]]:
    """스텝별 객체 상태 변화를 누적하여 스냅샷을 생성합니다.

    시간 순서의 이벤트 목록을 순회하면서 객체 속성을 누적합니다.
    각 스텝마다 object_name -> 최신 속성 맵의 스냅샷을 반환합니다.

    Args:
        events: JSON에서 파싱된 스텝 딕셔너리 시퀀스.

    Returns:
        스냅샷 리스트. 각 스냅샷은 object_name을 키로 하고
        누적 속성 딕셔너리를 값으로 갖는 딕셔너리입니다.
    """
    cumulative_state: Dict[str, Dict[str, Any]] = {}
    snapshots: List[Dict[str, Dict[str, Any]]] = []

    for event in events:
        changes = event.get("state_change")
        for change in changes:
            object_name = change.get("object_name")
            properties = change.get("property", {})
            # 객체별 최신 속성으로 병합/갱신
            object_state: MutableMapping[str, Any] = cumulative_state.setdefault(object_name, {})
            object_state.update(properties)
        # 이 스텝의 누적 상태 스냅샷 저장(불변 복사본)
        snapshots.append({name: dict(props) for name, props in cumulative_state.items()})
    return snapshots


def load_events_from_file(json_path: Path) -> List[Dict[str, Any]]:
    """JSON 파일에서 이벤트 목록을 로드합니다.

    Args:
        json_path: 이벤트 리스트를 담은 JSON 파일 경로.

    Returns:
        이벤트 딕셔너리의 리스트.
    """
    with json_path.open("r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)
    return data


if __name__ == "__main__":
    # 현재 스크립트와 같은 디렉토리의 더미 데이터를 기본 입력으로 사용
    script_dir = Path(__file__).resolve().parent
    default_json = script_dir / "test_data/trajectory_log.json"
    events_data = load_events_from_file(default_json)
    snapshots_per_step = accumulate_state_changes(events_data)
    # 데모: 각 스텝의 누적 상태를 출력
    for idx, snapshot in enumerate(snapshots_per_step):
        print(f"[step {idx}] {json.dumps(snapshot, ensure_ascii=False)}\n")
