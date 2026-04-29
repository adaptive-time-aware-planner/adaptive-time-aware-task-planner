import json
from pathlib import Path

# 프로젝트 루트 경로를 설정합니다.
# 이 스크립트가 'scripts' 디렉터리 안에 있다고 가정합니다.
try:
    ROOT_PATH = Path(__file__).resolve().parents[1]
except NameError:
    # 대화형 환경(예: Jupyter)에서 실행될 경우를 대비한 대체 경로입니다.
    ROOT_PATH = Path.cwd()

ASSETS_PATH = ROOT_PATH / "assets"
TASKS_PATH = ASSETS_PATH / "tasks"


def analyze_temporal_constraints():
    """
    assets/tasks 내의 모든 JSON 파일을 분석하여 시간 제약조건과 관련된 객체를 추출합니다.
    """
    # 'After' 제약조건 객체 유형을 Urgency 값에 따라 분리하여 저장
    after_constraint_objects_critical = set()
    after_constraint_objects_non_critical = set()
    # 'Before' 제약조건 객체 유형을 Urgency 값에 따라 분리하여 저장
    before_constraint_objects_critical = set()
    before_constraint_objects_non_critical = set()

    # 1. asset/tasks 내 'FloorPlan'으로 시작하는 폴더에서 모든 json 파일을 재귀적으로 탐색합니다.
    floorplan_dirs = [
        d for d in TASKS_PATH.iterdir() if d.is_dir() and d.name.startswith("FloorPlan")
    ]

    task_files = []
    for dir_path in floorplan_dirs:
        task_files.extend(sorted(dir_path.rglob("*.json")))

    if not task_files:
        print(
            f"Error: No task files found in FloorPlan* directories under {TASKS_PATH}"
        )
        return

    print(f"Found {len(task_files)} task files to analyze...")

    for file_path in task_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tasks_data = json.load(f)

            for task in tasks_data:
                for subtask in task.get("Subtasks", []):
                    # 2. TemporalConstraints 필드를 확인합니다.
                    for constraint in subtask.get("TemporalConstraints", []):
                        constraint_type = constraint.get("Type")
                        is_urgent = constraint.get(
                            "Urgency", False
                        )  # Urgency 필드 값 확인

                        # 'After' 타입인 경우 객체를 추출합니다.
                        if constraint_type == "After":
                            executions = subtask.get("Executions", {})
                            objects = executions.get("Objects")
                            if objects:
                                for obj_name in objects.keys():
                                    obj_type = obj_name.split("|")[0]
                                    if is_urgent:
                                        after_constraint_objects_critical.add(obj_type)
                                    else:
                                        after_constraint_objects_non_critical.add(
                                            obj_type
                                        )

                        # 'Before' 타입인 경우 객체를 추출합니다.
                        elif constraint_type == "Before":
                            related_subtask_title = constraint.get("Subtask")
                            # 해당 subtask의 Name과 related_subtask_title이 같은지 확인 (next함수로 찾기)
                            related_subtask = next(
                                (
                                    s
                                    for s in task.get("Subtasks", [])
                                    if s.get("Name") == related_subtask_title
                                ),
                                None,
                            )
                            if related_subtask:
                                # 해당 subtask의 Executions.Objects 필드에서 객체를 추출합니다.
                                executions = related_subtask.get("Executions", {})
                                objects = executions.get("Objects")
                                if objects:
                                    for obj_name in objects.keys():
                                        obj_type = obj_name.split("|")[0]
                                        if is_urgent:
                                            before_constraint_objects_critical.add(
                                                obj_type
                                            )
                                        else:
                                            before_constraint_objects_non_critical.add(
                                                obj_type
                                            )
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    # 최종 결과를 출력합니다.
    print("\n--- Analysis Complete ---")

    print("\n[Objects involved in 'After' Temporal Constraints (Urgency: True)]")
    if after_constraint_objects_critical:
        for obj in sorted(list(after_constraint_objects_critical)):
            print(f"- {obj}")
    else:
        print("No critical objects found for 'After' constraints.")

    print("\n[Objects involved in 'After' Temporal Constraints (Urgency: False)]")
    if after_constraint_objects_non_critical:
        for obj in sorted(list(after_constraint_objects_non_critical)):
            print(f"- {obj}")
    else:
        print("No non-critical objects found for 'After' constraints.")

    print("\n[Objects involved in 'Before' Temporal Constraints (Urgency: True)]")
    if before_constraint_objects_critical:
        for obj in sorted(list(before_constraint_objects_critical)):
            print(f"- {obj}")
    else:
        print("No critical objects found for 'Before' constraints.")

    print("\n[Objects involved in 'Before' Temporal Constraints (Urgency: False)]")
    if before_constraint_objects_non_critical:
        for obj in sorted(list(before_constraint_objects_non_critical)):
            print(f"- {obj}")
    else:
        print("No non-critical objects found for 'Before' constraints.")


if __name__ == "__main__":
    analyze_temporal_constraints()
