from pathlib import Path

MONITORING_ENABLED = True

# ========== 경로 설정 ==========
ROOT_PATH = Path(__file__).resolve().parents[3]
ASSETS_PATH = ROOT_PATH / "assets"
SRC_PATH = ROOT_PATH / "src"
LOG_PATH = ROOT_PATH / "logs"
SCRIPTS_PATH = ROOT_PATH / "scripts"

AGENT_KNOWLEDGE_PATH = ASSETS_PATH / "agent_knowledge"
SCENE_KNOWLEDGE_PATH = ASSETS_PATH / "scene_knowledge"
BATHROOM_PATH = SCENE_KNOWLEDGE_PATH / "bathroom"
BEDROOM_PATH = SCENE_KNOWLEDGE_PATH / "bedroom"
KITCHEN_PATH = SCENE_KNOWLEDGE_PATH / "kitchen"
LIVING_ROOM_PATH = SCENE_KNOWLEDGE_PATH / "living_room"
BAYESIAN_PATH = SCENE_KNOWLEDGE_PATH / "bayesian"

PROMPT_PATH = ASSETS_PATH / "prompts"
TASK_PATH = (
    ASSETS_PATH / "tasks" / "sampled_10_instruction_set_for_final_experiment_set_b"
)
RESULT_PATH = ASSETS_PATH / "results"
NAV_GRAPH_CACHE_PATH = ASSETS_PATH / "cache" / "ai2thor_nav_graphs"

# ========== 시뮬레이션 관련 ==========
DEFAULT_SCENE_NAME = "FloorPlan1"
ROOM_TYPE = ["bathroom", "bedroom", "kitchen", "living_room"]
PRIMITIVE_ACTION_SET = {
    "NAVIGATE_TO",
    "GRASP",
    "PLACE_INSIDE",
    "PLACE_ON_TOP",
    "OPEN",
    "CLOSE",
    "TOGGLE_ON",
    "TOGGLE_OFF",
    "SLICE",
    "MONITORING",
    "WAIT",
    "FILL",
}
STATIC_ACTION_SET = {
    "GRASP",
    "PLACE_INSIDE",
    "PLACE_ON_TOP",
    "OPEN",
    "CLOSE",
    "TOGGLE_ON",
    "TOGGLE_OFF",
    "SLICE",
    "FILL",
    "NAVIGATE_TO",
}
DYNAMIC_ACTION_SET = {
    "WAIT",
    "MONITORING",
}
# PRIMITIVE_ACTION_DURATION = 15.0
MONITORING_DURATION = 5.90
NAV_STEP_DURATION = 0.08
REAL_NAV_DURATION = 2.1
TOGGLE_ACTION_DURATION = 5.45
GRASP_ACTION_DURATION = 4.51
PLACE_ACTION_DURATION = 4.75

REACHABLE_DISTANCE_THRESHOLD = 50.0

# Heuristic weights
ALPHA_HEURISTIC = 1.0  # Navigation cost
BETA_HEURISTIC = 1.0  # Urgency (slack) cost
GAMMA_HEURISTIC = 1  # Remaining work cost

# Wait action control
WAIT_TIME_UPPER_BOUND = 200.0  # 'wait' 액션의 최대 허용 시간 (초)

# ========== 베이지안 ==========
# BAYESIAN_CRITERIA = 0.7  # Legacy fixed ratio
BAYESIAN_THRESHOLD_PROBABILITY = 0.1  # New threshold probability (eta)

GT_INTERVAL = 100.0
INIT_PRIOR_MEAN = 100.0
INIT_PRIOR_VARIANCE = 1600.0


# 지우지 마시오. 실험 돌릴동안만 쓰자구요 ㅎㅎㅎ
def set_init_prior_mean(value: float) -> None:
    """전역 INIT_PRIOR_MEAN 값을 설정합니다.

    Args:
        value (float): 새로 설정할 INIT_PRIOR_MEAN 값.
    """
    global INIT_PRIOR_MEAN
    INIT_PRIOR_MEAN = value


def set_monitoring_enabled(value: bool) -> None:
    """Sets the global MONITORING_ENABLED value."""
    global MONITORING_ENABLED
    MONITORING_ENABLED = value


# 지우지 마시오. 실험 돌릴동안만 쓰자구요 ㅎㅎㅎ
def set_init_prior_variance(value: float) -> None:
    """전역 INIT_PRIOR_VARIANCE 값을 설정합니다.

    Args:
        value (float): 새로 설정할 INIT_PRIOR_VARIANCE 값.
    """
    global INIT_PRIOR_VARIANCE
    INIT_PRIOR_VARIANCE = value


def set_alpha_heuristic(value: float) -> None:
    """Sets the global ALPHA_HEURISTIC value."""
    global ALPHA_HEURISTIC
    ALPHA_HEURISTIC = value


def set_beta_heuristic(value: float) -> None:
    """Sets the global BETA_HEURISTIC value."""
    global BETA_HEURISTIC
    BETA_HEURISTIC = value


def set_gamma_heuristic(value: float) -> None:
    """Sets the global GAMMA_HEURISTIC value."""
    global GAMMA_HEURISTIC
    GAMMA_HEURISTIC = value


def set_factor_alpha(value: float) -> None:
    """Sets the global FACTOR_ALPHA value."""
    global FACTOR_ALPHA
    FACTOR_ALPHA = value


def set_bayesian_threshold_probability(value: float) -> None:
    """Set global Bayesian trigger probability threshold (eta)."""
    global BAYESIAN_THRESHOLD_PROBABILITY
    BAYESIAN_THRESHOLD_PROBABILITY = value


def set_beam_width(value: int) -> None:
    """Sets the global BEAM_WIDTH value."""
    global BEAM_WIDTH
    BEAM_WIDTH = value


def set_simulation_depth(value: int) -> None:
    """Sets the global SIMULATION_DEPTH value."""
    global SIMULATION_DEPTH
    SIMULATION_DEPTH = value


def set_task_path(value: Path) -> None:
    """Sets the global TASK_PATH value dynamically."""
    global TASK_PATH
    TASK_PATH = value


FACTOR_ALPHA = 0.001
SIMILARITY_THRESHOLD = 0.7
MIN_VARIANCE = 1e-6

# Timing tolerance can be interpreted both as a ratio and an absolute cap.
# The ratio (30%) mirrors the previous behaviour, while the absolute value
# allows capping the tolerance window for large intervals.
TIMING_TOLERANCE_DEFAULT = 100.0
TIMING_TOLERANCE_RATIO = 0.3
TIMING_TOLERANCE_ABS = 12.5  # 결과 이상하면, 15로 rollback 할 것
MONITORING_SPLIT_TOLERANCE_ABS = 15.0
TSR_EVAL_TOLERANCE_ABS = 12.5

# Evaluation-only tolerance for (0, True) consecutive constraints.
# (0, True) means the successor navigation must start immediately after the
# predecessor finishes; nav time itself is physically inevitable and not a
# "wait".  This value captures only scheduling-cycle precision overhead.
CONSECUTIVE_TASK_WAIT_TOLERANCE = 2.0

# Scheduler-internal thresholds — conceptually distinct from evaluation tolerances.
# Values intentionally match the original TIMING_TOLERANCE_ABS-derived expressions
# so scheduler behaviour is unchanged; names make the distinction explicit.
CONSTRAINT_MERGE_THRESHOLD = 6.25  # TODO: calibrate from physical measurements
ACTION_SPLIT_PRECISION = 4.0  # TODO: calibrate from physical measurements
# A small planner-side grace keeps dispatch from oscillating on tiny timing noise
# near the interaction boundary.
RISK_GRACE_SECONDS = 4.5
# ========== 스케줄러 설정 ==========
SIMULATION_DEPTH = 3
BEAM_WIDTH = 3
EPSILON = 1
LARGE_NUMBER = 1e4
TOP_K = 1


# ========== ANSI 로그 색상 ==========
LOG_ROUND = 3
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
BLUE = "\x1b[34m"
MAGENTA = "\x1b[35m"
CYAN = "\x1b[36m"
WHITE = "\x1b[37m"
RESET = "\x1b[0m"

# ========== AI2-THOR 환경 상수 ==========
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

GRID_SIZE = 0.125
SMOOTH_LEVEL = 30

MOVE_STEP = 0.1
ROTATE_STEP = 2.5
RUN_SPEED_MULTIPLIER = 2

CAMERA_DISTANCE_BEHIND = 0.75
CAMERA_HEIGHT_ABOVE = 1.5

ARM_MOVE_STEP = 0.05
HAND_RADIUS = [0.1, 0.3, 0.5]

OBJECT_INTERESTS = {
    "object_interactions": [
        "toggleable",
        "breakable",
        "dirtyable",
        "cookable",
        "sliceable",
        "openable",
        "pickupable",
        "moveable",
        "canFillWithLiquid",
        "canBeUsedUp",
    ],
    "object_states": [
        "isToggled",
        "isBroken",
        "isDirty",
        "isCooked",
        "isSliced",
        "isOpen",
        "isPickedUp",
        "isFilledWithLiquid",
        "isUsedUp",
    ],
}

# ========== 환경 상수 ==========
ENV_PLACEHOLDER = "<ENVIRONMENT_INFO>"

# ========== 시간 제약 규칙 ==========
# 시간 제약이 중요한(critical) 객체 유형과 기본 Interval(초)
# 'After' 제약, Urgency: True 목록 기반
CRITICAL_OBJECT_INTERVALS = {
    "StoveBurner": INIT_PRIOR_MEAN,
    "Microwave": INIT_PRIOR_MEAN,
    "Faucet": INIT_PRIOR_MEAN,
    "Kettle": INIT_PRIOR_MEAN,
    "Laundry_Machine": INIT_PRIOR_MEAN,
    "ShowerHead": INIT_PRIOR_MEAN,
    "StoveKnob": INIT_PRIOR_MEAN,  # StoveBurner와 연관
    "Stove": INIT_PRIOR_MEAN,  # StoveBurner와 연관
    "Toilet": INIT_PRIOR_MEAN,
    "ScrubBrush": INIT_PRIOR_MEAN,
    "POT": INIT_PRIOR_MEAN,
    "Egg": INIT_PRIOR_MEAN,
    "CounterTop": INIT_PRIOR_MEAN,
    "CoffeeMachine": INIT_PRIOR_MEAN,
}

# Non-critical이지만 일관된 시간을 적용하고 싶은 객체
# 'After' 제약, Urgency: False 목록 기반
NON_CRITICAL_OBJECT_INTERVALS = {}

CRITICAL_OBJECT_GROUND_TRUTH = {
    "StoveBurner": GT_INTERVAL,
    "Microwave": GT_INTERVAL,
    "Faucet": GT_INTERVAL,
    "Kettle": GT_INTERVAL,
    "Laundry_Machine": GT_INTERVAL,
    "ShowerHead": GT_INTERVAL,
    "StoveKnob": GT_INTERVAL,
    "Stove": GT_INTERVAL,
    "Toilet": GT_INTERVAL,
    "ScrubBrush": GT_INTERVAL,
    "POT": GT_INTERVAL,
    "Egg": GT_INTERVAL,
    "CounterTop": GT_INTERVAL,
    "CoffeeMachine": GT_INTERVAL,
    "Cup": INIT_PRIOR_MEAN,
    "Mug": INIT_PRIOR_MEAN,
    "Plate": INIT_PRIOR_MEAN,
    "Bread": INIT_PRIOR_MEAN,
}
