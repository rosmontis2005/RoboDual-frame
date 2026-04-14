#!/usr/bin/env python3
"""Plot temporal aggregation delta and motion jerk metrics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from graph_profile_common import (
    COLOR_ACCENT,
    COLOR_FAILURE,
    COLOR_PURPLE,
    COLOR_SUCCESS,
    annotate_panel,
    apply_style,
    clipped,
    load_profile_data,
    outcome_values,
    parse_graph_args,
    quantile_by_step_since_slow,
    save_figure,
    values,
)


def main() -> None:
    args = parse_graph_args(__doc__ or "", "graph_aggregation_jerk.png")
    apply_style()
    step_rows, _ = load_profile_data(args.jsonl)

    agg_success, agg_failure = outcome_values(step_rows, "aggregation_delta_ee6")
    jerk_success, jerk_failure = outcome_values(step_rows, "jerk_l2_ee6")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    bp = axes[0, 0].boxplot(
        [agg_success, agg_failure],
        patch_artist=True,
        showfliers=False,
        tick_labels=["Success", "Failure"],
    )
    for patch, color in zip(bp["boxes"], [COLOR_SUCCESS, COLOR_FAILURE]):
        patch.set(facecolor=color, alpha=0.72, edgecolor="#111827")
    axes[0, 0].set_ylabel("aggregation_delta_ee6")
    axes[0, 0].set_title("Temporal Aggregation Delta by Outcome")
    axes[0, 0].axhline(np.percentile(values(step_rows, "aggregation_delta_ee6"), 95), color=COLOR_ACCENT, linestyle="--", label="Global p95")
    axes[0, 0].legend()

    bp = axes[0, 1].boxplot(
        [jerk_success, jerk_failure],
        patch_artist=True,
        showfliers=False,
        tick_labels=["Success", "Failure"],
    )
    for patch, color in zip(bp["boxes"], [COLOR_SUCCESS, COLOR_FAILURE]):
        patch.set(facecolor=color, alpha=0.72, edgecolor="#111827")
    axes[0, 1].set_ylabel("jerk_l2_ee6")
    axes[0, 1].set_title("Motion Jerk by Outcome")
    axes[0, 1].axhline(np.percentile(values(step_rows, "jerk_l2_ee6"), 95), color=COLOR_PURPLE, linestyle="--", label="Global p95")
    axes[0, 1].legend()

    for success, color, label in [
        (True, COLOR_SUCCESS, "Successful subtasks"),
        (False, COLOR_FAILURE, "Failed subtasks"),
    ]:
        xs, ys = quantile_by_step_since_slow(step_rows, "aggregation_delta_ee6", 90, success)
        axes[1, 0].plot(xs, ys, marker="o", color=color, linewidth=2, label=f"{label} p90")
    axes[1, 0].set_xticks(range(8))
    axes[1, 0].set_xlabel("Steps since last slow-system call")
    axes[1, 0].set_ylabel("aggregation_delta_ee6 p90")
    axes[1, 0].set_title("Aggregation Delta vs Slow-call Age")
    axes[1, 0].legend()

    success_rows = [row for row in step_rows if row.get("task_success") is True and row.get("jerk_l2_ee6") is not None]
    failure_rows = [row for row in step_rows if row.get("task_success") is False and row.get("jerk_l2_ee6") is not None]
    rng = np.random.default_rng(13)
    for rows, color, label, marker in [
        (success_rows, COLOR_SUCCESS, "Success", "."),
        (failure_rows, COLOR_FAILURE, "Failure", "x"),
    ]:
        if len(rows) > 9000:
            idx = rng.choice(len(rows), size=9000, replace=False)
            rows = [rows[int(i)] for i in idx]
        x = clipped(values(rows, "aggregation_delta_ee6"), 99.5)
        y = clipped(values(rows, "jerk_l2_ee6"), 99.5)
        axes[1, 1].scatter(x, y, s=10, alpha=0.22, color=color, label=label, marker=marker, linewidths=0.4)
    axes[1, 1].set_xlabel("aggregation_delta_ee6, clipped at p99.5")
    axes[1, 1].set_ylabel("jerk_l2_ee6, clipped at p99.5")
    axes[1, 1].set_title("Aggregation Conflict vs Motion Jerk")
    axes[1, 1].legend()
    annotate_panel(
        axes[1, 1],
        "High-high region:\nnew prediction conflicts with history\nand final motion changes sharply.",
        "upper right",
    )

    fig.suptitle("Specialist Profiling: Aggregation and Smoothness", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, args.output)


if __name__ == "__main__":
    main()
