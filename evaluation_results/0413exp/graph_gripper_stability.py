#!/usr/bin/env python3
"""Plot gripper flip and gripper jerk stability metrics."""

from __future__ import annotations

from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np

from graph_profile_common import (
    COLOR_ACCENT,
    COLOR_FAILURE,
    COLOR_NEUTRAL,
    COLOR_SUCCESS,
    annotate_panel,
    apply_style,
    load_profile_data,
    parse_graph_args,
    save_figure,
    task_summary,
    values,
)


def _flip_bucket(value: int) -> str:
    if value <= 0:
        return "0 flips"
    if value == 1:
        return "1 flip"
    return "2+ flips"


def main() -> None:
    args = parse_graph_args(__doc__ or "", "graph_gripper_stability.png")
    apply_style()
    step_rows, end_rows = load_profile_data(args.jsonl)
    summary = {item["task"]: item for item in task_summary(end_rows)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    buckets = ["0 flips", "1 flip", "2+ flips"]
    for outcome, color, offset, label in [
        (True, COLOR_SUCCESS, -0.18, "Success"),
        (False, COLOR_FAILURE, 0.18, "Failure"),
    ]:
        rows = [row for row in step_rows if row.get("task_success") is outcome]
        counter = Counter(_flip_bucket(int(row.get("gripper_flip_count", 0))) for row in rows)
        total = sum(counter.values())
        pct_values = [100.0 * counter[bucket] / total if total else 0.0 for bucket in buckets]
        x = np.arange(len(buckets)) + offset
        axes[0, 0].bar(x, pct_values, width=0.34, color=color, label=label, edgecolor="white")
    axes[0, 0].set_xticks(np.arange(len(buckets)))
    axes[0, 0].set_xticklabels(buckets)
    axes[0, 0].set_ylabel("Step share (%)")
    axes[0, 0].set_title("Recent Gripper Flip Count by Outcome")
    axes[0, 0].legend()

    jerk_success = values(step_rows, "gripper_jerk", lambda row: row.get("task_success") is True)
    jerk_failure = values(step_rows, "gripper_jerk", lambda row: row.get("task_success") is False)
    bins = np.arange(-4.5, 5.5, 1.0)
    axes[0, 1].hist(jerk_success, bins=bins, color=COLOR_SUCCESS, alpha=0.72, label="Success")
    axes[0, 1].hist(jerk_failure, bins=bins, color=COLOR_FAILURE, alpha=0.72, label="Failure")
    axes[0, 1].set_xlabel("gripper_jerk")
    axes[0, 1].set_ylabel("Step count")
    axes[0, 1].set_title("Discrete Gripper Jerk")
    axes[0, 1].legend()

    task_stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"flip": [], "jerk": []})
    for row in step_rows:
        if row.get("gripper_flip_count") is not None:
            task_stats[row["task"]]["flip"].append(float(row["gripper_flip_count"]))
        if row.get("gripper_jerk") is not None:
            task_stats[row["task"]]["jerk"].append(abs(float(row["gripper_jerk"])))

    plot_rows = []
    for task, stat in task_stats.items():
        if task not in summary or not stat["flip"]:
            continue
        plot_rows.append(
            {
                "task": task,
                "mean_flip": float(np.mean(stat["flip"])),
                "mean_abs_jerk": float(np.mean(stat["jerk"])) if stat["jerk"] else 0.0,
                "success_rate": summary[task]["success_rate"],
                "trials": summary[task]["trials"],
            }
        )
    plot_rows = sorted(plot_rows, key=lambda item: item["mean_flip"], reverse=True)[:16]
    y = np.arange(len(plot_rows))
    axes[1, 0].barh(y, [item["mean_flip"] for item in plot_rows], color=COLOR_NEUTRAL, edgecolor="white")
    axes[1, 0].set_yticks(y)
    axes[1, 0].set_yticklabels([item["task"] for item in plot_rows])
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Mean gripper_flip_count")
    axes[1, 0].set_title("Tasks with Most Gripper Flips")

    scatter_x = [item["mean_flip"] for item in plot_rows]
    scatter_y = [item["success_rate"] for item in plot_rows]
    sizes = [45 + item["trials"] * 8 for item in plot_rows]
    colors = [COLOR_FAILURE if item["success_rate"] < 70 else COLOR_ACCENT if item["success_rate"] < 90 else COLOR_SUCCESS for item in plot_rows]
    axes[1, 1].scatter(scatter_x, scatter_y, s=sizes, c=colors, edgecolor="#111827", alpha=0.82)
    for item in plot_rows:
        axes[1, 1].text(item["mean_flip"] + 0.01, item["success_rate"], item["task"], fontsize=8, va="center")
    axes[1, 1].set_xlabel("Mean gripper_flip_count")
    axes[1, 1].set_ylabel("Task success rate (%)")
    axes[1, 1].set_ylim(-5, 105)
    axes[1, 1].set_title("Gripper Instability vs Task Outcome")
    annotate_panel(
        axes[1, 1],
        "Bubble size = number of subtasks\nColor = success-rate band",
        "lower left",
    )

    fig.suptitle("Specialist Profiling: Gripper Stability", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, args.output)


if __name__ == "__main__":
    main()
