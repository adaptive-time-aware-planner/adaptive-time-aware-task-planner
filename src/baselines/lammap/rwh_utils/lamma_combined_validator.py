# utils/lamma_combined_validator.py
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import re


@dataclass
class ParsedAction:
    t: float
    name: str
    args: List[str]
    raw: str


@dataclass
class ValidationReport:
    task: str
    folder: str
    success: bool
    goal_satisfied: bool
    score: float
    total_actions: int
    executable_actions: int
    matched_expected_steps: int
    expected_steps: int
    violations: List[str] = field(default_factory=list)
    final_locations: Dict[str, str] = field(default_factory=dict)
    final_contains: Dict[str, List[str]] = field(default_factory=dict)
    final_states: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


PLAN_LINE_RE = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]+)?)\s*:\s*\(([^)]+)\)\s*\[([0-9]+(?:\.[0-9]+)?)\]\s*$"
)

CONTAINERS = {
    "fridge", "cabinet", "drawer", "microwave",
}


def normalize_name(x: str) -> str:
    s = str(x).strip()
    s = s.replace("_", "")
    s = s.replace(" ", "")
    s = s.lower()
    return s


def canonical_action_name(name: str) -> str:
    n = normalize_name(name)
    if n in {"pickupobject", "pickup"}:
        return "pickup"
    if n in {"putobject", "put"}:
        return "put"
    if n in {"openobject", "open"}:
        return "open"
    if n in {"closeobject", "close"}:
        return "close"
    if n in {"toggleobjecton", "toggleon", "switchonobject", "switchon"}:
        return "toggleon"
    if n in {"toggleobjectoff", "toggleoff", "switchoffobject", "switchoff"}:
        return "toggleoff"
    if n in {"gotoobject", "goto"}:
        return "goto"
    return n


def parse_combined_plan(plan_path: str | Path) -> List[ParsedAction]:
    plan_path = Path(plan_path)
    if not plan_path.exists():
        raise FileNotFoundError(f"combined plan not found: {plan_path}")

    actions: List[ParsedAction] = []
    for line in plan_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        m = PLAN_LINE_RE.match(line)
        if not m:
            continue

        t_str, inner, _dur = m.groups()
        toks = inner.strip().split()
        if not toks:
            continue

        name = toks[0].strip()
        args = [a.strip() for a in toks[1:]]

        actions.append(
            ParsedAction(
                t=float(t_str),
                name=name,
                args=args,
                raw=line,
            )
        )
    return actions


class SymbolicState:
    def __init__(self):
        self.obj_loc: Dict[str, str] = {}
        self.contains: Dict[str, List[str]] = {}
        self.state: Dict[str, str] = {}
        self.robot_holding: Dict[str, Optional[str]] = {}

    def snapshot(self) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, str]]:
        return (
            dict(self.obj_loc),
            {k: list(v) for k, v in self.contains.items()},
            dict(self.state),
        )


def build_initial_state(task_spec: Dict[str, Any], robots: Optional[List[str]] = None) -> SymbolicState:
    st = SymbolicState()
    robots = robots or ["robot1", "robot2"]

    for r in robots:
        st.robot_holding[normalize_name(r)] = None

    init_states = task_spec.get("init_states", {}) or {}
    for obj, loc in init_states.items():
        obj_n = normalize_name(obj)
        loc_n = normalize_name(loc)
        st.obj_loc[obj_n] = loc_n
        st.contains.setdefault(loc_n, [])
        if obj_n not in st.contains[loc_n]:
            st.contains[loc_n].append(obj_n)

    for ent in task_spec.get("object_states", []) or []:
        name = ent.get("name")
        if not isinstance(name, str):
            continue
        name_n = normalize_name(name)

        if name_n in CONTAINERS and name_n not in st.state:
            st.state[name_n] = "CLOSED"

        for obj in ent.get("contains", []) or []:
            obj_n = normalize_name(obj)
            st.obj_loc[obj_n] = name_n
            st.contains.setdefault(name_n, [])
            if obj_n not in st.contains[name_n]:
                st.contains[name_n].append(obj_n)

    return st


def normalize_expected_step(step: str) -> Optional[Tuple[str, List[str]]]:
    s = step.strip().lower()

    m = re.match(r"open\s+(.+)$", s)
    if m:
        return ("open", [normalize_name(m.group(1))])

    m = re.match(r"close\s+(.+)$", s)
    if m:
        return ("close", [normalize_name(m.group(1))])

    m = re.match(r"(?:toggleobjecton|toggleon|switchon)\s+(.+)$", s)
    if m:
        return ("toggleon", [normalize_name(m.group(1))])

    m = re.match(r"(?:toggleobjectoff|toggleoff|switchoff)\s+(.+)$", s)
    if m:
        return ("toggleoff", [normalize_name(m.group(1))])

    m = re.match(r"grab\s+(.+)$", s)
    if m:
        return ("pickup", [normalize_name(m.group(1))])

    m = re.match(r"pickup\s+(.+)$", s)
    if m:
        return ("pickup", [normalize_name(m.group(1))])

    m = re.match(r"place\s+(.+?)\s+on\s+(.+)$", s)
    if m:
        return ("put", [normalize_name(m.group(1)), normalize_name(m.group(2))])

    m = re.match(r"place\s+(.+?)\s+in\s+(.+)$", s)
    if m:
        return ("put", [normalize_name(m.group(1)), normalize_name(m.group(2))])

    m = re.match(r"put\s+(.+?)\s+on\s+(.+)$", s)
    if m:
        return ("put", [normalize_name(m.group(1)), normalize_name(m.group(2))])

    m = re.match(r"put\s+(.+?)\s+in\s+(.+)$", s)
    if m:
        return ("put", [normalize_name(m.group(1)), normalize_name(m.group(2))])

    return None


def action_matches_expected(act: ParsedAction, expected: Tuple[str, List[str]]) -> bool:
    exp_name, exp_args = expected
    act_name = canonical_action_name(act.name)
    args_n = [normalize_name(a) for a in act.args]

    if act_name != exp_name:
        return False

    if exp_name == "pickup":
        return len(args_n) >= 2 and args_n[1] == exp_args[0]

    if exp_name == "put":
        return len(args_n) >= 3 and args_n[1] == exp_args[0] and args_n[2] == exp_args[1]

    if exp_name in {"open", "close", "toggleon", "toggleoff"}:
        return len(args_n) >= 2 and args_n[1] == exp_args[0]

    return False


def _remove_from_all_contains(st: SymbolicState, obj_n: str):
    for _, items in st.contains.items():
        if obj_n in items:
            items.remove(obj_n)


def execute_action(st: SymbolicState, act: ParsedAction) -> Tuple[bool, Optional[str]]:
    name = canonical_action_name(act.name)
    args_n = [normalize_name(a) for a in act.args]

    if name == "goto":
        return True, None

    if name == "open":
        if len(args_n) < 2:
            return False, "bad_args:open"
        target = args_n[1]
        st.state[target] = "OPEN"
        return True, None

    if name == "close":
        if len(args_n) < 2:
            return False, "bad_args:close"
        target = args_n[1]
        st.state[target] = "CLOSED"
        return True, None

    if name == "toggleon":
        if len(args_n) < 2:
            return False, "bad_args:toggleon"

        target = args_n[1]

        # microwave는 닫힌 상태에서만 켤 수 있다고 가정
        if target == "microwave" and st.state.get(target, "CLOSED") != "CLOSED":
            return False, "toggleon_with_open_microwave"

        st.state[target] = "ON"

        if target == "microwave":
            inside = st.contains.get("microwave", [])
            for obj in inside:
                st.state[obj] = "HOT"
                for child in st.contains.get(obj, []):
                    st.state[child] = "HOT"

        return True, None

    if name == "toggleoff":
        if len(args_n) < 2:
            return False, "bad_args:toggleoff"
        target = args_n[1]
        st.state[target] = "OFF"
        return True, None

    if name == "pickup":
        if len(args_n) < 2:
            return False, "bad_args:pickup"

        robot = args_n[0]
        obj = args_n[1]
        src = args_n[2] if len(args_n) >= 3 else st.obj_loc.get(obj)

        if robot not in st.robot_holding:
            st.robot_holding[robot] = None

        if st.robot_holding[robot] is not None:
            return False, f"hand_not_empty:{robot}"

        cur = st.obj_loc.get(obj)
        if cur is None:
            return False, f"pickup_unknown_object:{obj}"

        if src is not None and cur != src:
            return False, f"pickup_wrong_source:{obj}@{src},current={cur}"

        actual_src = cur
        if actual_src in CONTAINERS and st.state.get(actual_src, "CLOSED") != "OPEN":
            return False, f"pickup_from_closed_container:{obj}@{actual_src}"

        _remove_from_all_contains(st, obj)
        st.obj_loc[obj] = f"holding:{robot}"
        st.robot_holding[robot] = obj
        return True, None

    if name == "put":
        if len(args_n) < 3:
            return False, "bad_args:put"

        robot, obj, dst = args_n[0], args_n[1], args_n[2]

        if st.robot_holding.get(robot) != obj:
            return False, f"put_without_holding:{robot}:{obj}"

        if dst in CONTAINERS and st.state.get(dst, "CLOSED") != "OPEN":
            return False, f"put_into_closed_container:{obj}@{dst}"

        st.robot_holding[robot] = None
        st.obj_loc[obj] = dst
        st.contains.setdefault(dst, [])
        if obj not in st.contains[dst]:
            st.contains[dst].append(obj)
        return True, None

    return False, f"unsupported_action:{act.name}"


def check_goal_satisfied(st: SymbolicState, task_spec: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    ok = True

    for ent in task_spec.get("object_states", []) or []:
        name_raw = ent.get("name")
        if not isinstance(name_raw, str):
            continue

        name = normalize_name(name_raw)
        desired_state = ent.get("state")
        desired_contains = [normalize_name(x) for x in (ent.get("contains") or [])]

        if desired_state is not None:
            cur_state = st.state.get(name)
            expected_state = str(desired_state).upper()
            if cur_state != expected_state:
                ok = False
                reasons.append(
                    f"state_mismatch:{name}:expected={expected_state},current={cur_state}"
                )

        if desired_contains:
            cur_contains = set(st.contains.get(name, []))
            if not set(desired_contains).issubset(cur_contains):
                ok = False
                reasons.append(
                    f"contains_mismatch:{name}:expected_contains={desired_contains},current={sorted(cur_contains)}"
                )

    return ok, reasons


class LaMMACombinedPlanValidator:
    def __init__(self, dataset_specs: List[Dict[str, Any]]):
        self.dataset_specs = dataset_specs

    def find_task_spec_by_index(self, idx_1based: int) -> Dict[str, Any]:
        if idx_1based < 1 or idx_1based > len(self.dataset_specs):
            raise IndexError(f"task index out of range: {idx_1based}")
        return self.dataset_specs[idx_1based - 1]

    def validate_plan_file(
        self,
        folder_path: str | Path,
        task_spec: Dict[str, Any],
        robots: Optional[List[str]] = None,
    ) -> ValidationReport:
        folder_path = Path(folder_path)
        plan_path = folder_path / "combined_plan.py"

        actions = parse_combined_plan(plan_path)
        st = build_initial_state(task_spec, robots=robots)

        execution_violations: List[str] = []
        executable = 0

        expected_steps_raw = task_spec.get("Plan", []) or []
        expected_steps = [normalize_expected_step(x) for x in expected_steps_raw]
        expected_steps = [x for x in expected_steps if x is not None]

        remaining_expected = list(expected_steps)
        matched_expected_steps = 0

        for act in actions:
            ok, reason = execute_action(st, act)
            if ok:
                executable += 1
            else:
                execution_violations.append(f"{act.raw} -> {reason}")

            if canonical_action_name(act.name) == "goto":
                continue

            matched_pos = None
            for j, exp in enumerate(remaining_expected):
                if action_matches_expected(act, exp):
                    matched_pos = j
                    break

            if matched_pos is not None:
                matched_expected_steps += 1
                remaining_expected.pop(matched_pos)
            else:
                execution_violations.append(f"unexpected_step: got={act.raw}")

        goal_ok, goal_reasons = check_goal_satisfied(st, task_spec)
        violations = execution_violations + goal_reasons

        success = (len(execution_violations) == 0) and goal_ok

        exec_ratio = (executable / len(actions)) if actions else 0.0
        step_ratio = (matched_expected_steps / len(expected_steps)) if expected_steps else 0.0
        goal_ratio = 1.0 if goal_ok else 0.0
        score = 0.5 * exec_ratio + 0.3 * step_ratio + 0.2 * goal_ratio

        final_loc, final_contains, final_state = st.snapshot()

        task_name = (
            task_spec.get("task")
            or task_spec.get("instruction")
            or task_spec.get("name")
            or "unknown_task"
        )

        return ValidationReport(
            task=task_name,
            folder=str(folder_path.name),
            success=success,
            goal_satisfied=goal_ok,
            score=round(score, 4),
            total_actions=len(actions),
            executable_actions=executable,
            matched_expected_steps=matched_expected_steps,
            expected_steps=len(expected_steps),
            violations=violations,
            final_locations=final_loc,
            final_contains=final_contains,
            final_states=final_state,
        )


def extract_index_from_folder_name(folder_name: str) -> Optional[int]:
    m = re.match(r"^\[(\d+)\]", folder_name.strip())
    if not m:
        return None
    return int(m.group(1))


def load_dataset_specs(json_path: str | Path) -> List[Dict[str, Any]]:
    json_path = Path(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("dataset spec json must be a list")
    return data