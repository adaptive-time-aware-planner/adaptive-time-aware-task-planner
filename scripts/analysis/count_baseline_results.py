import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def count_results(base_path: str = "assets/results") -> None:
    """
    assets/results 경로 하위의 states* 폴더 구조를 탐색하여
    base_line_name 별 결과 개수를 셉니다.

    Target Structure:
    assets/results/states<prior>/<case>/<instruction>/<scene>/<base_line_name>
    """
    root_dir = Path(base_path)

    if not root_dir.exists():
        print(f"Error: 경로를 찾을 수 없습니다: {root_dir}")
        return

    # 결과를 저장할 딕셔너리
    # {base_line_name: {total: int, details: {states60: int, states80: int, ...}}}
    summary: Dict[str, Dict] = defaultdict(
        lambda: {"total": 0, "details": defaultdict(int)}
    )

    # 1. states로 시작하는 폴더 탐색 (Level 1)
    # glob 패턴: states* / * / * / * / *
    # (states / case / instruction / scene / baseline)

    print(f"Searching in: {root_dir.absolute()}")
    print("-" * 50)

    # states 폴더 찾기
    state_dirs = [
        d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("states")
    ]
    state_dirs.sort()

    for state_dir in state_dirs:
        # 진행 상황 표시
        # print(f"Scanning {state_dir.name}...")

        # Level 2: Case (e.g., tasks_3_constraints_2)
        for case_dir in state_dir.iterdir():
            if not case_dir.is_dir():
                continue

            # Level 3: Instruction (e.g., 08_heat_the_potato...)
            for instr_dir in case_dir.iterdir():
                if not instr_dir.is_dir():
                    continue

                # Level 4: Scene (e.g., FloorPlan18)
                for scene_dir in instr_dir.iterdir():
                    if not scene_dir.is_dir():
                        continue

                    # Level 5: Baseline Name (This is what we count)
                    for baseline_dir in scene_dir.iterdir():
                        if not baseline_dir.is_dir():
                            continue

                        # Check if end_state.json exists
                        if not (baseline_dir / "end_state.json").exists():
                            continue

                        baseline_name = baseline_dir.name

                        # 결과 집계
                        summary[baseline_name]["total"] += 1
                        summary[baseline_name]["details"][state_dir.name] += 1

    print("-" * 50)
    print("집계 결과 (Base Line Name 별 개수):")
    print("-" * 50)

    # 결과 출력 (Total 기준 내림차순 정렬)
    sorted_stats = sorted(summary.items(), key=lambda x: x[1]["total"], reverse=True)

    if not sorted_stats:
        print("해당 구조의 결과를 찾지 못했습니다.")
        return

    for baseline, stats in sorted_stats:
        print(f"[{baseline}]")
        print(f"  Total: {stats['total']}")
        # details 키를 정렬하여 출력
        details_str = ", ".join(
            [f"{k}: {v}" for k, v in sorted(stats["details"].items())]
        )
        print(f"  Details: {{{details_str}}}")
        print("")


if __name__ == "__main__":
    # 프로젝트 루트에서 실행한다고 가정하고 경로 설정
    # 현재 파일 위치: scripts/count_baseline_results.py
    # 목표 경로: assets/results

    current_path = Path(__file__).resolve()
    # If script is in scripts/, parent is project root
    project_root = current_path.parent.parent
    target_path = project_root / "assets" / "results"

    count_results(str(target_path))
