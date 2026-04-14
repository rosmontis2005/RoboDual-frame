#!/usr/bin/env python3
"""Plot specialist-vs-reference action alignment metrics."""

from __future__ import annotations

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
    outcome_values,
    parse_graph_args,
    positive_for_log,
    quantile_by_step_since_slow,
    save_figure,
)


def _plot_quantile(ax: plt.Axes, rows: list[dict], metric: str, title: str, ylabel: str) -> None:
    for success, color, label in [
        (True, COLOR_SUCCESS, "Successful subtasks"),
        (False, COLOR_FAILURE, "Failed subtasks"),
    ]:
        xs50, ys50 = quantile_by_step_since_slow(rows, metric, 50, success)
        xs90, ys90 = quantile_by_step_since_slow(rows, metric, 90, success)
        ax.plot(xs50, ys50, marker="o", color=color, linewidth=2.0, label=f"{label} p50")
        ax.plot(xs90, ys90, marker="s", color=color, linewidth=1.6, linestyle="--", label=f"{label} p90")
    ax.set_title(title)
    ax.set_xlabel("Steps since last slow-system call")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(8))
    ax.legend(fontsize=8)


def main() -> None:
    args = parse_graph_args(__doc__ or "", "graph_dp_ref_alignment.png")
    apply_style()
    step_rows, _ = load_profile_data(args.jsonl)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _plot_quantile(
        axes[0, 0],
        step_rows,
        "dp_ref_l2_ee6",
        "EE6 Reference Deviation vs Slow-call Age",
        "RMS delta on x/y/z/roll/pitch/yaw",
    )
    _plot_quantile(
        axes[0, 1],
        step_rows,
        "dp_ref_l2_gripper",
        "Gripper Reference Deviation vs Slow-call Age",
        "RMS delta on gripper",
    )

    ee6_success, ee6_failure = outcome_values(step_rows, "dp_ref_l2_ee6")
    grip_success, grip_failure = outcome_values(step_rows, "dp_ref_l2_gripper")
    bp = axes[1, 0].boxplot(
        [positive_for_log(ee6_success), positive_for_log(ee6_failure)],
        patch_artist=True,
        showfliers=False,
        tick_labels=["Success", "Failure"],
    )
    for patch, color in zip(bp["boxes"], [COLOR_SUCCESS, COLOR_FAILURE]):
        patch.set(facecolor=color, alpha=0.72, edgecolor="#111827")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("dp_ref_l2_ee6 (log scale)")
    axes[1, 0].set_title("Outcome Split: EE6 Deviation")
    axes[1, 0].axhline(np.percentile(ee6_success, 95), color=COLOR_NEUTRAL, linestyle="--", label="Success p95")
    axes[1, 0].legend()

    bp = axes[1, 1].boxplot(
        [positive_for_log(grip_success), positive_for_log(grip_failure)],
        patch_artist=True,
        showfliers=False,
        tick_labels=["Success", "Failure"],
    )
    for patch, color in zip(bp["boxes"], [COLOR_SUCCESS, COLOR_FAILURE]):
        patch.set(facecolor=color, alpha=0.72, edgecolor="#111827")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("dp_ref_l2_gripper (log scale)")
    axes[1, 1].set_title("Outcome Split: Gripper Deviation")
    axes[1, 1].axhline(np.percentile(grip_success, 95), color=COLOR_ACCENT, linestyle="--", label="Success p95")
    axes[1, 1].legend()

    annotate_panel(
        axes[1, 0],
        "Interpretation:\nlarge deviation means the specialist\nmoves away from the generalist reference.",
        "upper left",
    )
    fig.suptitle("Specialist Profiling: DP Action vs Generalist Reference", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, args.output)


if __name__ == "__main__":
    main()
