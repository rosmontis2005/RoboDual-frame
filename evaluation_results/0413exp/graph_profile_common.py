#!/usr/bin/env python3
"""Shared helpers for plotting specialist profiling JSONL files."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_JSONL = Path(__file__).with_name("specialist_profile_rank0.jsonl")

COLOR_SUCCESS = "#2A9D8F"
COLOR_FAILURE = "#E76F51"
COLOR_NEUTRAL = "#457B9D"
COLOR_ACCENT = "#F2B705"
COLOR_PURPLE = "#7A5195"
COLOR_GRAY = "#6B7280"
BG = "#F8FAFC"
GRID = "#D7DEE8"


def parse_graph_args(description: str, default_output_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "jsonl",
        nargs="?",
        default=DEFAULT_JSONL,
        type=Path,
        help="Path to specialist_profile_rank0.jsonl",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=Path(__file__).with_name(default_output_name),
        type=Path,
        help="Output image path",
    )
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": BG,
            "axes.edgecolor": "#9AA5B1",
            "axes.labelcolor": "#1F2937",
            "axes.titlecolor": "#111827",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.75,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#CBD5E1",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 180,
        }
    )


def is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def load_profile_data(jsonl_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load step rows and subtask-end rows, then attach final outcome to steps."""
    jsonl_path = jsonl_path.expanduser().resolve()
    if not jsonl_path.exists():
        raise SystemExit(f"Input file not found: {jsonl_path}")

    step_rows: list[dict[str, Any]] = []
    end_rows: list[dict[str, Any]] = []
    outcomes: dict[tuple[int, int, str], dict[str, Any]] = {}

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            event = row.get("event")
            task = row.get("task")
            sequence = row.get("sequence")
            subtask_i = row.get("subtask_i")
            if not isinstance(task, str) or not isinstance(sequence, int) or not isinstance(subtask_i, int):
                continue
            key = (sequence, subtask_i, task)

            if event == "step":
                profile = row.get("profile") or {}
                flat: dict[str, Any] = {
                    "line_no": line_no,
                    "key": key,
                    "sequence": sequence,
                    "subtask_i": subtask_i,
                    "task": task,
                    "step": row.get("step"),
                    "ep_len": row.get("ep_len"),
                    "model_s": row.get("model_s"),
                    "env_s": row.get("env_s"),
                    "oracle_s": row.get("oracle_s"),
                    "step_success": row.get("step_success"),
                    "terminal_step": row.get("terminal_step"),
                }
                if isinstance(profile, dict):
                    flat.update(profile)
                step_rows.append(flat)
            elif event == "subtask_end":
                end_row = {
                    "key": key,
                    "sequence": sequence,
                    "subtask_i": subtask_i,
                    "task": task,
                    "task_success": bool(row.get("task_success")),
                    "steps": row.get("steps"),
                }
                end_rows.append(end_row)
                outcomes[key] = end_row

    for row in step_rows:
        outcome = outcomes.get(row["key"])
        row["task_success"] = None if outcome is None else outcome["task_success"]
        row["subtask_steps"] = None if outcome is None else outcome["steps"]

    return step_rows, end_rows


def values(
    rows: list[dict[str, Any]],
    key: str,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> np.ndarray:
    out: list[float] = []
    for row in rows:
        if predicate is not None and not predicate(row):
            continue
        value = row.get(key)
        if is_number(value):
            out.append(float(value))
    return np.asarray(out, dtype=float)


def outcome_values(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    success = values(rows, key, lambda row: row.get("task_success") is True)
    failure = values(rows, key, lambda row: row.get("task_success") is False)
    return success, failure


def positive_for_log(arr: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    if arr.size == 0:
        return arr
    return np.maximum(arr, eps)


def clipped(arr: np.ndarray, q: float = 99.0) -> np.ndarray:
    if arr.size == 0:
        return arr
    return np.clip(arr, None, np.percentile(arr, q))


def pct(num: float, denom: float) -> float:
    return 100.0 * num / denom if denom else 0.0


def annotate_panel(ax: plt.Axes, text: str, loc: str = "upper right") -> None:
    anchor = {
        "upper right": (0.98, 0.96, "right", "top"),
        "upper left": (0.02, 0.96, "left", "top"),
        "lower right": (0.98, 0.04, "right", "bottom"),
        "lower left": (0.02, 0.04, "left", "bottom"),
    }[loc]
    x, y, ha, va = anchor
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=9,
        color="#1F2937",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.92},
    )


def save_figure(fig: plt.Figure, output: Path) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"Saved {output}")


def task_summary(end_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"trials": 0, "successes": 0, "steps": []})
    for row in end_rows:
        item = grouped[row["task"]]
        item["trials"] += 1
        item["successes"] += int(row.get("task_success") is True)
        if is_number(row.get("steps")):
            item["steps"].append(float(row["steps"]))

    result = []
    for task, item in grouped.items():
        trials = int(item["trials"])
        successes = int(item["successes"])
        steps = item["steps"]
        result.append(
            {
                "task": task,
                "trials": trials,
                "successes": successes,
                "success_rate": pct(successes, trials),
                "mean_steps": float(np.mean(steps)) if steps else float("nan"),
            }
        )
    return result


def quantile_by_step_since_slow(
    rows: list[dict[str, Any]],
    metric: str,
    quantile: float,
    success: bool | None = None,
) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for step_since in range(8):
        arr = values(
            rows,
            metric,
            lambda row, step_since=step_since: row.get("step_since_slow") == step_since
            and (success is None or row.get("task_success") is success),
        )
        if arr.size:
            xs.append(step_since)
            ys.append(float(np.percentile(arr, quantile)))
    return xs, ys


def count_by(rows: list[dict[str, Any]], key: str) -> Counter[Any]:
    counter: Counter[Any] = Counter()
    for row in rows:
        counter[row.get(key)] += 1
    return counter
