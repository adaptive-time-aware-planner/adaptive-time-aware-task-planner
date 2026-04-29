"""Render eta-sensitivity LaTeX tables from a single offline analysis summary.

Outputs two files:
- eta_sensitivity_overall.tex  : weighted-average across cases per (prior, eta)
- eta_sensitivity_by_case.tex  : per (prior-regime, eta, case) breakdown

Usage::

    python -m assets.result_analysis.eta_sensitivity_tables
    python -m assets.result_analysis.eta_sensitivity_tables \\
        --analysis-root path/to/analysis/eta_sensitivity \\
        --batch-root path/to/offline_batch_eta_sensitivity \\
        --out-dir path/to/latex_tables

Summaries are read from ``<analysis-root>/<task_folder>/offline_analysis_summary.json``
(by default all subfolders that contain that file). Pass ``--summary`` only for a
single merged summary path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from functools import lru_cache
from pathlib import Path
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assets.result_analysis.utils.multi_source import (
    load_json_objects,
    mean_std,
    resolve_source_paths,
)
from src.utils.config.constants import ASSETS_PATH


def _load_primitive_semantics():
    """Load lightweight primitive-action helpers without importing task package."""

    checker_path = (
        PROJECT_ROOT / "src" / "utils" / "task" / "primitive_action_semantics.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_primitive_action_semantics",
        checker_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load primitive-action checker: {checker_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_primitive_semantics = _load_primitive_semantics()
find_first_task_sequence_issue = _primitive_semantics.find_first_task_sequence_issue

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_ANALYSIS_ROOT = (
    "assets/results/offline_exp_result/analysis/eta_sensitivity"
)
DEFAULT_BATCH_ROOT = (
    "assets/results/offline_exp_result/offline_batch_eta_sensitivity"
)
DEFAULT_OUT_DIR = (
    "assets/results/offline_exp_result/analysis/eta_sensitivity/latex_tables"
)

INIT_PRIOR_ORDER: list[str] = [
    "UNDER_ESTIMATE",
    "CORRECT_ESTIMATE",
    "OVER_ESTIMATE",
]
INIT_PRIOR_LABEL: dict[str, str] = {
    "CORRECT_ESTIMATE": "Correct",
    "OVER_ESTIMATE": "Over",
    "UNDER_ESTIMATE": "Under",
}

BY_CASE_PRIOR_REGIMES: list[tuple[str, list[str]]] = [
    ("Correct", ["CORRECT_ESTIMATE"]),
    ("Incorrect", ["UNDER_ESTIMATE", "OVER_ESTIMATE"]),
]

ETA_ORDER: list[str] = ["0.01", "0.1", "0.9"]

CASES: list[tuple[str, str]] = [
    ("tasks_2_constraints_1", "T2C1"),
    ("tasks_2_constraints_2", "T2C2"),
    ("tasks_3_constraints_1", "T3C1"),
    ("tasks_3_constraints_2", "T3C2"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setting_key(init_prior: str, eta: str) -> str:
    return f"{init_prior}__bayesian__DEFAULT__w10_d10__eta{eta}"


def _eta_token(value: object) -> str:
    return f"{float(value):g}"


def _tex(
    mean: float | None,
    std: float | None,
    nd: int = 2,
    *,
    scale: float = 1.0,
) -> str:
    if mean is None:
        return "--"
    mean *= scale
    std_value = 0.0 if std is None else std
    std_value *= scale
    s = f"{mean:.{nd}f} \\pm {std_value:.{nd}f}"
    return f"${s}$"


def _rank_format(
    values: list[tuple[float | None, float | None]],
    nd: int = 2,
    *,
    higher_is_better: bool,
    scale: float = 1.0,
) -> list[str]:
    """Return formatted strings for each value."""
    eps = 1e-9
    sign = -1.0 if higher_is_better else 1.0
    present_values = [v[0] for v in values if v[0] is not None]
    sorted_unique = sorted(set(sign * v for v in present_values))
    rank0_val = sorted_unique[0] if len(sorted_unique) > 0 else None
    rank1_val = sorted_unique[1] if len(sorted_unique) > 1 else None

    result = []
    for mean, std in values:
        if mean is None:
            result.append("--")
            continue
        sv = sign * mean
        if rank0_val is not None and abs(sv - rank0_val) < eps:
            result.append(_tex(mean, std, nd=nd, scale=scale))
        elif rank1_val is not None and abs(sv - rank1_val) < eps:
            result.append(_tex(mean, std, nd=nd, scale=scale))
        else:
            result.append(_tex(mean, std, nd=nd, scale=scale))
    return result


def _weighted(case_metrics: dict[str, dict[str, Any]], field: str) -> float:
    """Aggregate ``field`` across cases.

    When every case provides a positive ``n_instructions``, uses an
    instruction-count-weighted mean. Otherwise falls back to an unweighted
    mean of case-level values (for legacy summaries that omit ``n_instructions``).
    """

    if not case_metrics:
        raise ValueError(f"No cases for field '{field}'.")

    weighted: list[tuple[int, float]] = []
    for m in case_metrics.values():
        if field not in m:
            raise KeyError(field)
        n_raw = m.get("n_instructions")
        if n_raw is None:
            weighted = []
            break
        n = int(n_raw)
        if n <= 0:
            weighted = []
            break
        weighted.append((n, float(m[field])))

    if weighted and len(weighted) == len(case_metrics):
        denom = sum(n for n, _ in weighted)
        if denom == 0:
            raise ValueError(f"No instructions for field '{field}'.")
        return sum(n * v for n, v in weighted) / denom

    values = [float(m[field]) for m in case_metrics.values()]
    return sum(values) / len(values)


def _weighted_optional(
    case_metrics: dict[str, dict[str, float | int | None]],
    field: str,
    *,
    weight_field: str = "n_instructions",
) -> float | None:
    total_n, total_w = 0, 0.0
    for m in case_metrics.values():
        value = m.get(field)
        if value is None:
            continue
        weight = int(m.get(weight_field, 0) or 0)
        if weight <= 0:
            continue
        total_w += weight * float(value)
        total_n += weight
    if total_n == 0:
        return None
    return total_w / total_n


# ---------------------------------------------------------------------------
# Monitor count loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _n_uncontrollable_targets(
    task_folder_name: str,
    case_name: str,
    scene_name: str,
    instruction_name: str,
) -> int:
    """Count critical-interval endpoints used for Mon./Unc. normalization."""

    from src.utils.config.constants import EPSILON
    from src.utils.task.task_util import TaskUtil

    instruction_stem = Path(str(instruction_name)).stem
    task_path = (
        ASSETS_PATH
        / "tasks"
        / task_folder_name
        / case_name
        / scene_name
        / f"{instruction_stem}.json"
    )
    if not task_path.is_file():
        raise FileNotFoundError(f"Task file not found: {task_path}")
    task_data = json.loads(task_path.read_text(encoding="utf-8"))
    _, constraints, _ = TaskUtil.build_tasks_and_constraints(
        task_data,
        scene_file_name=f"{scene_name}_physics_environment.json",
    )
    return len({
        end
        for _, end, data in constraints.edges(data=True)
        if data.get("info", {}).get("IsCritical")
        and float(data.get("info", {}).get("Interval", 0.0)) > EPSILON
    })


@lru_cache(maxsize=None)
def _is_translation_valid(
    task_folder_name: str,
    case_name: str,
    scene_name: str,
    instruction_name: str,
) -> bool:
    instruction_stem = Path(str(instruction_name)).stem
    task_path = (
        ASSETS_PATH
        / "tasks"
        / task_folder_name
        / case_name
        / scene_name
        / f"{instruction_stem}.json"
    )
    if not task_path.is_file():
        return False

    task_data = json.loads(task_path.read_text(encoding="utf-8"))
    return find_first_task_sequence_issue(task_data) is None


def load_monitor_summary(
    batch_root: Path,
    *,
    task_folder_name: str,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return avg monitor counts per (setting_key, case_name)."""

    buckets: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for path in batch_root.rglob("*.json"):
        if "_batch_summary" in path.parts:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload.get("meta_data", {})
        if not isinstance(meta, dict):
            continue
        if meta.get("suite_name") != "eta_sensitivity":
            continue
        if meta.get("ablation_config") != "DEFAULT":
            continue
        if int(meta.get("beam_width", -1)) != 10 or int(meta.get("beam_depth", -1)) != 10:
            continue

        init_prior = meta.get("init_prior_config")
        eta = meta.get("eta")
        case_name = payload.get("case")
        scene_name = payload.get("scene_name")
        instruction_name = payload.get("instruction")
        monitor_count = payload.get("actual_monitor_count_total")
        interval_counts = payload.get("actual_monitor_count_by_interval")

        if any(
            v is None
            for v in [
                init_prior,
                eta,
                case_name,
                scene_name,
                instruction_name,
                monitor_count,
            ]
        ):
            continue
        if not isinstance(interval_counts, dict):
            continue
        if not _is_translation_valid(
            task_folder_name=task_folder_name,
            case_name=str(case_name),
            scene_name=str(scene_name),
            instruction_name=str(instruction_name),
        ):
            continue

        n_unc = len(interval_counts)
        per_unc = float(monitor_count) / n_unc if n_unc > 0 else 0.0
        key = _setting_key(str(init_prior), _eta_token(eta))
        buckets.setdefault(key, {}).setdefault(str(case_name), []).append(
            (float(monitor_count), per_unc)
        )

    return {
        key: {
            case: {
                "avg_monitors": sum(c for c, _ in vals) / len(vals),
                "avg_monitors_per_unc": sum(p for _, p in vals) / len(vals),
            }
            for case, vals in cases.items()
        }
        for key, cases in buckets.items()
    }


def load_valid_gap_summary(
    raw_path: Path,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Return clipped oracle gap averaged over TCSR-valid instructions only."""

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    buckets: dict[str, dict[str, list[float]]] = {}

    for scene_data in raw.values():
        if not isinstance(scene_data, dict):
            continue
        for case_name, instructions in scene_data.items():
            if not isinstance(instructions, dict):
                continue
            for entry in instructions.values():
                if not isinstance(entry, dict):
                    continue
                if not entry.get("translation_valid", True):
                    continue
                for setting_key, metrics in entry.items():
                    if setting_key in {
                        "oracle",
                        "oracle_valid",
                        "translation_valid",
                        "translation_issue_group",
                    }:
                        continue
                    if not isinstance(metrics, dict):
                        continue
                    tsr = metrics.get("tsr")
                    gap = metrics.get("makespan_gap")
                    if tsr is None or gap is None:
                        continue
                    if float(tsr) < 1.0 - 1e-9:
                        continue
                    buckets.setdefault(setting_key, {}).setdefault(str(case_name), []).append(
                        max(0.0, float(gap))
                    )

    return {
        key: {
            case: {
                "gap_valid_plus": sum(vals) / len(vals),
                "n_valid_instructions": len(vals),
            }
            for case, vals in cases.items()
        }
        for key, cases in buckets.items()
    }


def load_instruction_feasibility_summary(
    raw_path: Path,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Return instruction-level temporal feasibility per setting and case.

    An instruction is feasible iff all temporal constraints in the generated
    offline schedule are satisfied, i.e., its saved schedule-level TSR is 1.0.
    """

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    buckets: dict[str, dict[str, list[float]]] = {}

    for scene_data in raw.values():
        if not isinstance(scene_data, dict):
            continue
        for case_name, instructions in scene_data.items():
            if not isinstance(instructions, dict):
                continue
            for entry in instructions.values():
                if not isinstance(entry, dict):
                    continue
                if not entry.get("translation_valid", True):
                    continue
                for setting_key, metrics in entry.items():
                    if setting_key in {
                        "oracle",
                        "oracle_valid",
                        "translation_valid",
                        "translation_issue_group",
                    }:
                        continue
                    if not isinstance(metrics, dict):
                        continue
                    tsr = metrics.get("tsr")
                    if tsr is None:
                        continue
                    feasible = 1.0 if float(tsr) >= 1.0 - 1e-9 else 0.0
                    buckets.setdefault(setting_key, {}).setdefault(str(case_name), []).append(
                        feasible
                    )

    return {
        key: {
            case: {
                "instr_feas": 100.0 * sum(vals) / len(vals),
                "n_instructions": len(vals),
            }
            for case, vals in cases.items()
        }
        for key, cases in buckets.items()
    }


def _has_setting(summary_sources: list[dict[str, Any]], key: str) -> bool:
    return any(key in summary for summary in summary_sources)


def _collect_case_metric_stats(
    summary_sources: list[dict[str, Any]],
    *,
    setting_key: str,
    case_name: str,
    field: str,
) -> tuple[float | None, float | None]:
    values: list[float] = []
    for summary in summary_sources:
        metrics = summary.get(setting_key, {}).get(case_name, {})
        if not isinstance(metrics, dict):
            continue
        value = metrics.get(field)
        if value is None:
            continue
        values.append(float(value))
    return mean_std(values)


def _collect_case_metric_stats_for_priors(
    summary_sources: list[dict[str, Any]],
    *,
    priors: list[str],
    eta: str,
    case_name: str,
    field: str,
) -> tuple[float | None, float | None]:
    values: list[float] = []
    for summary in summary_sources:
        for prior in priors:
            metrics = summary.get(_setting_key(prior, eta), {}).get(case_name, {})
            if not isinstance(metrics, dict):
                continue
            value = metrics.get(field)
            if value is None:
                continue
            values.append(float(value))
    return mean_std(values)


def _collect_case_instruction_feasibility_for_priors(
    instr_feas_sources: list[dict[str, dict[str, dict[str, float | int]]]],
    *,
    priors: list[str],
    eta: str,
    case_name: str,
) -> tuple[float | None, float | None]:
    values: list[float] = []
    for instr_feas in instr_feas_sources:
        for prior in priors:
            case_info = instr_feas.get(_setting_key(prior, eta), {}).get(case_name, {})
            if not isinstance(case_info, dict):
                continue
            value = case_info.get("instr_feas")
            if value is None:
                continue
            values.append(float(value))
    return mean_std(values)


def _collect_gap_case_stats(
    valid_gap_sources: list[dict[str, dict[str, dict[str, float | int]]]],
    *,
    setting_key: str,
    case_name: str,
) -> tuple[float | None, float | None]:
    values: list[float] = []
    for valid_gaps in valid_gap_sources:
        gap_info = valid_gaps.get(setting_key, {}).get(case_name, {})
        if not isinstance(gap_info, dict):
            continue
        gap_value = gap_info.get("gap_valid_plus")
        if gap_value is None:
            continue
        values.append(float(gap_value))
    return mean_std(values)


def _collect_gap_case_stats_for_priors(
    valid_gap_sources: list[dict[str, dict[str, dict[str, float | int]]]],
    *,
    priors: list[str],
    eta: str,
    case_name: str,
) -> tuple[float | None, float | None]:
    values: list[float] = []
    for valid_gaps in valid_gap_sources:
        for prior in priors:
            gap_info = valid_gaps.get(_setting_key(prior, eta), {}).get(case_name, {})
            if not isinstance(gap_info, dict):
                continue
            gap_value = gap_info.get("gap_valid_plus")
            if gap_value is None:
                continue
            values.append(float(gap_value))
    return mean_std(values)


def _collect_monitor_case_stats(
    monitor_sources: list[dict[str, dict[str, dict[str, float]]]],
    *,
    setting_key: str,
    case_name: str,
    field: str,
) -> tuple[float | None, float | None]:
    values: list[float] = []
    for monitors in monitor_sources:
        monitor_info = monitors.get(setting_key, {}).get(case_name, {})
        if not isinstance(monitor_info, dict):
            continue
        value = monitor_info.get(field)
        if value is None:
            continue
        values.append(float(value))
    return mean_std(values)


def _collect_monitor_case_stats_for_priors(
    monitor_sources: list[dict[str, dict[str, dict[str, float]]]],
    *,
    priors: list[str],
    eta: str,
    case_name: str,
    field: str,
) -> tuple[float | None, float | None]:
    values: list[float] = []
    for monitors in monitor_sources:
        for prior in priors:
            monitor_info = monitors.get(_setting_key(prior, eta), {}).get(case_name, {})
            if not isinstance(monitor_info, dict):
                continue
            value = monitor_info.get(field)
            if value is None:
                continue
            values.append(float(value))
    return mean_std(values)


def _overall_setting_stats(
    summary: dict[str, Any],
    monitors: dict[str, dict[str, dict[str, float]]],
    valid_gaps: dict[str, dict[str, dict[str, float | int]]],
    *,
    prior: str,
    eta: str,
) -> dict[str, float | None]:
    key = _setting_key(prior, eta)
    case_data = summary.get(key, {})
    if not isinstance(case_data, dict) or not case_data:
        return {}
    mon_data = monitors.get(key, {})
    gap_data = valid_gaps.get(key, {})

    avg_mon = (
        _weighted(
            {
                c: (
                    {
                        "avg_monitors": mon_data[c]["avg_monitors"],
                        "n_instructions": case_data[c]["n_instructions"],
                    }
                    if "n_instructions" in case_data[c]
                    else {"avg_monitors": mon_data[c]["avg_monitors"]}
                )
                for c in case_data
                if c in mon_data
            },
            "avg_monitors",
        )
        if any(c in mon_data for c in case_data)
        else 0.0
    )
    avg_per_unc = (
        _weighted(
            {
                c: (
                    {
                        "avg_monitors_per_unc": mon_data[c]["avg_monitors_per_unc"],
                        "n_instructions": case_data[c]["n_instructions"],
                    }
                    if "n_instructions" in case_data[c]
                    else {"avg_monitors_per_unc": mon_data[c]["avg_monitors_per_unc"]}
                )
                for c in case_data
                if c in mon_data
            },
            "avg_monitors_per_unc",
        )
        if any(c in mon_data for c in case_data)
        else 0.0
    )
    gap = _weighted_optional(
        {
            c: {
                "gap_valid_plus": gap_data[c]["gap_valid_plus"],
                "n_valid_instructions": gap_data[c]["n_valid_instructions"],
            }
            for c in case_data
            if c in gap_data
        },
        "gap_valid_plus",
        weight_field="n_valid_instructions",
    )

    return {
        "tsr": _weighted(case_data, "tsr"),
        "gap": gap,
        "avg_per_unc": avg_per_unc,
        "avg_mon": avg_mon,
    }


def _collect_overall_stats(
    summary_sources: list[dict[str, Any]],
    monitor_sources: list[dict[str, dict[str, dict[str, float]]]],
    valid_gap_sources: list[dict[str, dict[str, dict[str, float | int]]]],
    *,
    prior: str,
    eta: str,
) -> dict[str, tuple[float | None, float | None]]:
    tsr_values: list[float] = []
    gap_values: list[float] = []
    avg_per_unc_values: list[float] = []
    avg_mon_values: list[float] = []

    for summary, monitors, valid_gaps in zip(
        summary_sources,
        monitor_sources,
        valid_gap_sources,
    ):
        metrics = _overall_setting_stats(
            summary,
            monitors,
            valid_gaps,
            prior=prior,
            eta=eta,
        )
        if not metrics:
            continue
        if metrics.get("tsr") is not None:
            tsr_values.append(float(metrics["tsr"]))
        if metrics.get("gap") is not None:
            gap_values.append(float(metrics["gap"]))
        if metrics.get("avg_per_unc") is not None:
            avg_per_unc_values.append(float(metrics["avg_per_unc"]))
        if metrics.get("avg_mon") is not None:
            avg_mon_values.append(float(metrics["avg_mon"]))

    return {
        "tsr": mean_std(tsr_values),
        "gap": mean_std(gap_values),
        "avg_per_unc": mean_std(avg_per_unc_values),
        "avg_mon": mean_std(avg_mon_values),
    }


def _collect_overall_stats_for_priors(
    summary_sources: list[dict[str, Any]],
    monitor_sources: list[dict[str, dict[str, dict[str, float]]]],
    valid_gap_sources: list[dict[str, dict[str, dict[str, float | int]]]],
    *,
    priors: list[str],
    eta: str,
) -> dict[str, tuple[float | None, float | None]]:
    tsr_values: list[float] = []
    gap_values: list[float] = []
    avg_per_unc_values: list[float] = []
    avg_mon_values: list[float] = []

    for summary, monitors, valid_gaps in zip(
        summary_sources,
        monitor_sources,
        valid_gap_sources,
    ):
        for prior in priors:
            metrics = _overall_setting_stats(
                summary,
                monitors,
                valid_gaps,
                prior=prior,
                eta=eta,
            )
            if not metrics:
                continue
            if metrics.get("tsr") is not None:
                tsr_values.append(float(metrics["tsr"]))
            if metrics.get("gap") is not None:
                gap_values.append(float(metrics["gap"]))
            if metrics.get("avg_per_unc") is not None:
                avg_per_unc_values.append(float(metrics["avg_per_unc"]))
            if metrics.get("avg_mon") is not None:
                avg_mon_values.append(float(metrics["avg_mon"]))

    return {
        "tsr": mean_std(tsr_values),
        "gap": mean_std(gap_values),
        "avg_per_unc": mean_std(avg_per_unc_values),
        "avg_mon": mean_std(avg_mon_values),
    }


def _collect_overall_instruction_feasibility_for_priors(
    instr_feas_sources: list[dict[str, dict[str, dict[str, float | int]]]],
    *,
    priors: list[str],
    eta: str,
) -> tuple[float | None, float | None]:
    values: list[float] = []
    for instr_feas in instr_feas_sources:
        for prior in priors:
            case_data = instr_feas.get(_setting_key(prior, eta), {})
            if not isinstance(case_data, dict) or not case_data:
                continue
            values.append(_weighted(case_data, "instr_feas"))
    return mean_std(values)


# ---------------------------------------------------------------------------
# Overall table
# ---------------------------------------------------------------------------

def _overall_tabular(
    summary_sources: list[dict[str, Any]],
    monitor_sources: list[dict[str, dict[str, dict[str, float]]]],
    valid_gap_sources: list[dict[str, dict[str, dict[str, float | int]]]],
    instr_feas_sources: list[dict[str, dict[str, dict[str, float | int]]]],
) -> str:
    lines: list[str] = []
    lines.append(r"\begin{tabular}{@{}llrrr@{}}" "\n")
    lines.append(r"\toprule" "\n")
    lines.append(
        r"\textbf{Prior Regime} & \textbf{$\eta$} & "
        r"\textbf{Instr. Feas. (\%)} ($\uparrow$) & "
        r"\textbf{Gap$^{+}$ (s)} ($\downarrow$) & "
        r"\textbf{Mon./Int. (count)} ($\downarrow$) \\" "\n"
    )
    lines.append(r"\midrule" "\n")

    present_regimes = [
        (label, priors)
        for label, priors in BY_CASE_PRIOR_REGIMES
        if any(
            _has_setting(summary_sources, _setting_key(prior, eta))
            for prior in priors
            for eta in ETA_ORDER
        )
    ]

    for regime_idx, (regime_label, priors) in enumerate(present_regimes):
        present_etas = [
            eta
            for eta in ETA_ORDER
            if any(
                _has_setting(summary_sources, _setting_key(prior, eta))
                for prior in priors
            )
        ]
        for eta_idx, eta in enumerate(present_etas):
            overall_stats = _collect_overall_stats_for_priors(
                summary_sources,
                monitor_sources,
                valid_gap_sources,
                priors=priors,
                eta=eta,
            )
            instr_feas_mean, instr_feas_std = (
                _collect_overall_instruction_feasibility_for_priors(
                    instr_feas_sources,
                    priors=priors,
                    eta=eta,
                )
            )
            gap_mean, gap_std = overall_stats["gap"]
            avg_per_unc_mean, avg_per_unc_std = overall_stats["avg_per_unc"]

            prior_cell = (
                rf"\multirow{{{len(present_etas)}}}{{*}}{{\textbf{{{regime_label}}}}}"
                if eta_idx == 0 else ""
            )
            lines.append(
                f"{prior_cell} & ${eta}$ & "
                f"{_tex(instr_feas_mean, instr_feas_std, nd=1)} & "
                f"{_tex(gap_mean, gap_std, nd=1)} & "
                f"{_tex(avg_per_unc_mean, avg_per_unc_std, nd=2)} \\\\\n"
            )
        if regime_idx != len(present_regimes) - 1:
            lines.append(r"\midrule" "\n")

    lines.append(r"\bottomrule" "\n")
    lines.append(r"\end{tabular}" "\n")
    return "".join(lines)


def build_overall_tex(
    summary_sources: list[dict[str, Any]],
    monitor_sources: list[dict[str, dict[str, dict[str, float]]]],
    valid_gap_sources: list[dict[str, dict[str, dict[str, float | int]]]],
    instr_feas_sources: list[dict[str, dict[str, dict[str, float | int]]]],
) -> str:
    tabular = _overall_tabular(
        summary_sources,
        monitor_sources,
        valid_gap_sources,
        instr_feas_sources,
    )
    return (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        r"{\small" "\n"
        r"\setlength{\tabcolsep}{4.5pt}" "\n"
        r"\renewcommand{\arraystretch}{1.05}" "\n"
        + tabular
        + "}\n"
        r"\caption{Offline schedule-level sensitivity analysis for selecting the "
        r"risk tolerance $\eta$. The Incorrect regime aggregates under- and "
        r"over-estimated initial temporal priors. Instr. Feas. denotes the "
        r"percentage of generated offline schedules in which all temporal "
        r"constraints in an instruction are satisfied; it does not evaluate "
        r"AI2-THOR goal-condition completion. Results use the DEFAULT "
        r"planner ($W{=}D{=}10$) and are reported as mean $\pm$ standard deviation.}"
        "\n"
        r"\label{tab:eta_sensitivity_overall}"
        "\n"
        r"\end{table}"
        "\n"
    )


# ---------------------------------------------------------------------------
# By-case table  (rows: prior regime x case; eta values as sub-columns per metric)
# ---------------------------------------------------------------------------

def _by_case_tabular(
    summary_sources: list[dict[str, Any]],
    monitor_sources: list[dict[str, dict[str, dict[str, float]]]],
    valid_gap_sources: list[dict[str, dict[str, dict[str, float | int]]]],
    instr_feas_sources: list[dict[str, dict[str, dict[str, float | int]]]],
) -> str:
    """Render rows = (prior regime, case) and eta sub-columns per metric."""

    present_regimes = [
        (label, priors)
        for label, priors in BY_CASE_PRIOR_REGIMES
        if any(
            _has_setting(summary_sources, _setting_key(prior, eta))
            for prior in priors
            for eta in ETA_ORDER
        )
    ]
    present_etas = [eta for eta in ETA_ORDER if any(
        _has_setting(summary_sources, _setting_key(p, eta)) for p in INIT_PRIOR_ORDER
    )]
    n_eta = len(present_etas)

    col_spec = "ll|" + "|".join("r" * n_eta for _ in range(3))
    lines: list[str] = []
    lines.append(rf"\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}" "\n")
    lines.append(r"\toprule" "\n")

    # header row 1: metric group labels
    lines.append(
        r"\multirow{2}{*}{\textbf{Prior Regime}} & "
        r"\multirow{2}{*}{\textbf{Case}} & "
        rf"\multicolumn{{{n_eta}}}{{c}}{{\textbf{{Instr. Feas. (\%)}} ($\uparrow$)}} & "
        rf"\multicolumn{{{n_eta}}}{{c}}{{\textbf{{Gap$^{{+}}$ (s)}} ($\downarrow$)}} & "
        rf"\multicolumn{{{n_eta}}}{{c}}{{\textbf{{Mon./Int. (count)}}}} \\" "\n"
    )
    # cmidrule positions
    def _cmidrule(start: int, end: int) -> str:
        return rf"\cmidrule(lr){{{start}-{end}}}"
    c1, c2 = 3, 2 + n_eta
    c3, c4 = c2 + 1, c2 + n_eta
    c5, c6 = c4 + 1, c4 + n_eta
    lines.append(f"{_cmidrule(c1,c2)}{_cmidrule(c3,c4)}{_cmidrule(c5,c6)}\n")

    # header row 2: eta values repeated per metric group
    eta_cols = " & ".join(rf"${e}$" for e in present_etas)
    lines.append(f"& & {eta_cols} & {eta_cols} & {eta_cols} \\\\\n")
    lines.append(r"\midrule" "\n")

    for regime_idx, (regime_label, priors) in enumerate(present_regimes):
        for case_idx, (case_name, case_label) in enumerate(CASES):
            prior_cell = (
                rf"\multirow{{{len(CASES)}}}{{*}}{{\textbf{{{regime_label}}}}}"
                if case_idx == 0 else ""
            )
            instr_feas_vals: list[tuple[float | None, float | None]] = []
            gap_vals: list[tuple[float | None, float | None]] = []
            unc_vals: list[tuple[float | None, float | None]] = []
            for eta in present_etas:
                instr_feas_vals.append(
                    _collect_case_instruction_feasibility_for_priors(
                        instr_feas_sources,
                        priors=priors,
                        eta=eta,
                        case_name=case_name,
                    )
                )
                gap_vals.append(
                    _collect_gap_case_stats_for_priors(
                        valid_gap_sources,
                        priors=priors,
                        eta=eta,
                        case_name=case_name,
                    )
                )
                unc_vals.append(
                    _collect_monitor_case_stats_for_priors(
                        monitor_sources,
                        priors=priors,
                        eta=eta,
                        case_name=case_name,
                        field="avg_monitors_per_unc",
                    )
                )

            instr_feas_cells = _rank_format(
                instr_feas_vals,
                nd=1,
                higher_is_better=True,
            )
            gap_cells = _rank_format(gap_vals, nd=1, higher_is_better=False)
            unc_cells = [_tex(mean, std, nd=2) for mean, std in unc_vals]

            lines.append(
                f"{prior_cell} & \\textbf{{{case_label}}} & "
                f"{' & '.join(instr_feas_cells)} & "
                f"{' & '.join(gap_cells)} & "
                f"{' & '.join(unc_cells)} \\\\\n"
            )

        if regime_idx != len(present_regimes) - 1:
            lines.append(r"\midrule" "\n")

    lines.append(r"\bottomrule" "\n")
    lines.append(r"\end{tabular}" "\n")
    return "".join(lines)


def build_by_case_tex(
    summary_sources: list[dict[str, Any]],
    monitor_sources: list[dict[str, dict[str, dict[str, float]]]],
    valid_gap_sources: list[dict[str, dict[str, dict[str, float | int]]]],
    instr_feas_sources: list[dict[str, dict[str, dict[str, float | int]]]],
) -> str:
    tabular = _by_case_tabular(
        summary_sources,
        monitor_sources,
        valid_gap_sources,
        instr_feas_sources,
    )
    return (
        r"\begin{table*}[t]" "\n"
        r"\centering" "\n"
        r"{\small" "\n"
        r"\setlength{\tabcolsep}{4.2pt}" "\n"
        r"\renewcommand{\arraystretch}{1.05}" "\n"
        r"\resizebox{\linewidth}{!}{%" "\n"
        + tabular
        + "}\n"
        "}\n"
        r"\caption{Offline schedule-level eta-sensitivity results by prior regime and task-complexity "
        r"case. The Incorrect regime aggregates under- and over-estimated "
        r"initial temporal priors. Each metric reports values for "
        r"$\eta\in\{0.01,0.1,0.9\}$ using the DEFAULT planner ($W{=}D{=}10$), "
        r"where Instr. Feas. denotes the percentage of generated offline schedules "
        r"in which all temporal constraints in an instruction are satisfied, "
        r"reported as mean $\pm$ standard deviation over decomposed instruction "
        r"sets.}"
        "\n"
        r"\label{tab:eta_sensitivity_by_case}"
        "\n"
        r"\end{table*}"
        "\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help=(
            "Optional path to a single offline_analysis_summary.json. "
            "If omitted, summaries are loaded from --analysis-root (per task folder)."
        ),
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_ANALYSIS_ROOT,
        help=(
            "Directory containing one subdirectory per task_folder with "
            "offline_analysis_summary.json (matches offline_comparison output layout)."
        ),
    )
    parser.add_argument(
        "--task-folders",
        nargs="+",
        default=None,
        help="Task folder names to aggregate under --analysis-root.",
    )
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_BATCH_ROOT,
        help="Batch results root directory (for monitor counts).",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="Path to offline_comparison_raw.json. Defaults to a sibling of --summary.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_OUT_DIR,
    )
    args = parser.parse_args()

    if args.summary is not None:
        summary_paths = resolve_source_paths(
            single_path=args.summary,
            analysis_root=None,
            task_folders=None,
            filename="offline_analysis_summary.json",
        )
    else:
        summary_paths = resolve_source_paths(
            single_path=None,
            analysis_root=args.analysis_root,
            task_folders=args.task_folders,
            filename="offline_analysis_summary.json",
        )

    if args.raw is not None:
        raw_paths = resolve_source_paths(
            single_path=args.raw,
            analysis_root=None,
            task_folders=None,
            filename="offline_comparison_raw.json",
        )
    elif args.summary is not None:
        raw_paths = resolve_source_paths(
            single_path=args.summary.with_name("offline_comparison_raw.json"),
            analysis_root=None,
            task_folders=None,
            filename="offline_comparison_raw.json",
        )
    else:
        raw_paths = resolve_source_paths(
            single_path=None,
            analysis_root=args.analysis_root,
            task_folders=args.task_folders,
            filename="offline_comparison_raw.json",
        )
    summary_sources = load_json_objects(summary_paths)
    task_folder_names = [path.parent.name for path in summary_paths]
    monitor_sources = []
    for task_folder_name in task_folder_names:
        folder_batch_root = args.batch_root / task_folder_name
        monitor_root = folder_batch_root if folder_batch_root.exists() else args.batch_root
        monitor_sources.append(
            load_monitor_summary(
                monitor_root,
                task_folder_name=task_folder_name,
            )
        )
    valid_gap_sources = [load_valid_gap_summary(path) for path in raw_paths]
    instr_feas_sources = [
        load_instruction_feasibility_summary(path)
        for path in raw_paths
    ]

    # warn about any missing keys (OVER_ESTIMATE may not exist yet)
    for prior in INIT_PRIOR_ORDER:
        for eta in ETA_ORDER:
            key = _setting_key(prior, eta)
            if not _has_setting(summary_sources, key):
                print(f"[warn] missing in summary, will be skipped: {key}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overall_path = args.out_dir / "eta_sensitivity_overall.tex"
    by_case_path = args.out_dir / "eta_sensitivity_by_case.tex"

    overall_path.write_text(
        build_overall_tex(
            summary_sources,
            monitor_sources,
            valid_gap_sources,
            instr_feas_sources,
        ),
        encoding="utf-8",
    )
    by_case_path.write_text(
        build_by_case_tex(
            summary_sources,
            monitor_sources,
            valid_gap_sources,
            instr_feas_sources,
        ),
        encoding="utf-8",
    )
    print(overall_path)
    print(by_case_path)


if __name__ == "__main__":
    main()
