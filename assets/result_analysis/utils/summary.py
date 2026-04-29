from __future__ import annotations

import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def aggregate_summary(
    trials: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]]]:
    """Aggregate trials into approach x difficulty x init x init_* summary.

    Returns:
        Structure: {approach: {difficulty: {init: {init_60/100/140: {metric: value}}}}}
    """

    by_key: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for t in trials:
        key = (str(t["approach"]), str(t["difficulty"]), str(t["states"]))
        by_key.setdefault(key, []).append(t)

    final: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]]] = {}
    for (approach, difficulty, states), items in by_key.items():
        # Map states to init_* format and filter out states80 and states120
        states_mapping = {
            "states60": "init_60",
            "states70": "init_70",
            "states80": "init_80",
            "states90": "init_90",
            "states100": "init_100",
            "states110": "init_110",
            "states120": "init_120",
            "states130": "init_130",
            "states140": "init_140",
        }

        if states not in states_mapping:
            # Skip states80 and states120
            continue

        init_key = states_mapping[states]

        total = len(items)
        if total == 0:
            continue
        # SR: trial_metrics에서 넘어온 sr(0/1)을 그대로 집계
        sr_values = [int(it.get("sr", 0)) for it in items]
        if sr_values:
            sr = statistics.mean(sr_values) * 100.0
            sr_std = statistics.stdev(sr_values) * 100.0 if len(sr_values) > 1 else 0.0
        else:
            sr, sr_std = 0.0, 0.0

        gcr_values = [float(it.get("instruction_gcr", 0.0)) for it in items]
        if gcr_values:
            gcr = statistics.mean(gcr_values) * 100.0
            gcr_std = (
                statistics.stdev(gcr_values) * 100.0 if len(gcr_values) > 1 else 0.0
            )
        else:
            gcr, gcr_std = 0.0, 0.0

        # TSR: average over numeric-only, ignore None
        tsr_samples: List[float] = []
        for it in items:
            v = it.get("tsr", None)
            if v is None:
                continue
            try:
                tsr_samples.append(float(v))
            except Exception:
                continue

        if tsr_samples:
            tsr = statistics.mean(tsr_samples) * 100.0
            tsr_std = (
                statistics.stdev(tsr_samples) * 100.0 if len(tsr_samples) > 1 else 0.0
            )
        else:
            tsr, tsr_std = 0.0, 0.0

        makespan_values = [float(it.get("makespan", 0.0)) for it in items]
        if makespan_values:
            makespan = statistics.mean(makespan_values)
            makespan_std = (
                statistics.stdev(makespan_values) if len(makespan_values) > 1 else 0.0
            )
        else:
            makespan, makespan_std = 0.0, 0.0

        makespan_sr_1_values = [
            float(it.get("makespan", 0.0)) for it in items if int(it.get("sr", 0)) == 1
        ]
        if makespan_sr_1_values:
            makespan_sr_1 = statistics.mean(makespan_sr_1_values)
            makespan_sr_1_std = (
                statistics.stdev(makespan_sr_1_values)
                if len(makespan_sr_1_values) > 1
                else 0.0
            )
        else:
            makespan_sr_1, makespan_sr_1_std = 0.0, 0.0

        # Structure: {approach: {difficulty: {init: {init_60/100/140: {metric: value}}}}}
        final.setdefault(approach, {}).setdefault(difficulty, {}).setdefault(
            "init", {}
        )[init_key] = {
            "sr": sr,
            "sr_std": sr_std,
            "gcr": gcr,
            "gcr_std": gcr_std,
            "tsr": tsr,
            "tsr_std": tsr_std,
            "makespan": makespan,
            "makespan_std": makespan_std,
            "makespan_sr_1": makespan_sr_1,
            "makespan_sr_1_std": makespan_sr_1_std,
        }
    return final
