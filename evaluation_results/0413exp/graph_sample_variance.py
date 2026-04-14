#!/usr/bin/env python3
"""Plot stochastic sample variance from repeated specialist predictions."""

from __future__ import annotations

from collections import Counter

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
    save_figure,
    values,
)


def main() -> None:
    args = parse_graph_args(__doc__ or "", "graph_sample_variance.png")
    apply_style()
    step_rows, _ = load_profile_data(args.jsonl)
    sampled_rows = [row for row in step_rows if row.get("sample_var") is not None]

    ee6_success, ee6_failure = outcome_values(sampled_rows, "sample_var_ee6")
    grip_success, grip_failure = outcome_values(sampled_rows, "sample_var_gripper")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    bp = axes[0, 0].boxplot(
        [positive_for_log(ee6_success), positive_for_log(ee6_failure)],
        patch_artist=True,
        showfliers=False,
        tick_labels=["Success", "Failure"],
    )
    for patch, color in zip(bp["boxes"], [COLOR_SUCCESS, COLOR_FAILURE]):
        patch.set(facecolor=color, alpha=0.72, edgecolor="#111827")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("sample_var_ee6 (log scale)")
    axes[0, 0].set_title("EE6 Sampling Uncertainty by Outcome")
    axes[0, 0].axhline(np.percentile(positive_for_log(ee6_success), 95), color=COLOR_ACCENT, linestyle="--", label="Success p95")
    axes[0, 0].legend()

    bp = axes[0, 1].boxplot(
        [positive_for_log(grip_success), positive_for_log(grip_failure)],
        patch_artist=True,
        showfliers=False,
        tick_labels=["Success", "Failure"],
    )
    for patch, color in zip(bp["boxes"], [COLOR_SUCCESS, COLOR_FAILURE]):
        patch.set(facecolor=color, alpha=0.72, edgecolor="#111827")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("sample_var_gripper (log scale)")
    axes[0, 1].set_title("Gripper Sampling Uncertainty by Outcome")
    axes[0, 1].axhline(np.percentile(positive_for_log(grip_success), 95), color=COLOR_NEUTRAL, linestyle="--", label="Success p95")
    axes[0, 1].legend()

    x_success = positive_for_log(values(sampled_rows, "sample_var_ee6", lambda row: row.get("task_success") is True))
    y_success = positive_for_log(values(sampled_rows, "sample_var_gripper", lambda row: row.get("task_success") is True))
    x_failure = positive_for_log(values(sampled_rows, "sample_var_ee6", lambda row: row.get("task_success") is False))
    y_failure = positive_for_log(values(sampled_rows, "sample_var_gripper", lambda row: row.get("task_success") is False))
    axes[1, 0].scatter(x_success, y_success, s=14, alpha=0.35, color=COLOR_SUCCESS, label="Success")
    axes[1, 0].scatter(x_failure, y_failure, s=18, alpha=0.55, color=COLOR_FAILURE, marker="x", label="Failure")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("sample_var_ee6")
    axes[1, 0].set_ylabel("sample_var_gripper")
    axes[1, 0].set_title("EE6 vs Gripper Uncertainty")
    axes[1, 0].legend()

    step_counter = Counter(row.get("step_since_slow") for row in sampled_rows)
    xs = sorted(key for key in step_counter if isinstance(key, int))
    ys = [step_counter[key] for key in xs]
    axes[1, 1].bar(xs, ys, color=COLOR_NEUTRAL, edgecolor="white")
    axes[1, 1].set_xticks(range(8))
    axes[1, 1].set_xlabel("Steps since last slow-system call")
    axes[1, 1].set_ylabel("Sampled step count")
    axes[1, 1].set_title("Where Sample Variance Was Measured")
    annotate_panel(
        axes[1, 1],
        f"Sampled rows: {len(sampled_rows):,}\nTotal step rows: {len(step_rows):,}\nK=3 only on scheduled probe steps",
        "upper right",
    )

    fig.suptitle("Specialist Profiling: Sampling Uncertainty", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, args.output)


if __name__ == "__main__":
    main()
