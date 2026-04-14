#!/usr/bin/env python3
"""Plot subtask success rate and mean completion steps."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from graph_profile_common import (
    COLOR_ACCENT,
    COLOR_FAILURE,
    COLOR_SUCCESS,
    apply_style,
    load_profile_data,
    parse_graph_args,
    save_figure,
    task_summary,
)


def main() -> None:
    args = parse_graph_args(__doc__ or "", "graph_task_outcomes.png")
    apply_style()
    _, end_rows = load_profile_data(args.jsonl)
    summary = sorted(task_summary(end_rows), key=lambda item: (item["success_rate"], item["mean_steps"]))

    tasks = [item["task"] for item in summary]
    rates = np.asarray([item["success_rate"] for item in summary])
    mean_steps = np.asarray([item["mean_steps"] for item in summary])
    labels = [f"{item['successes']}/{item['trials']}" for item in summary]

    fig, axes = plt.subplots(1, 2, figsize=(15, 11), gridspec_kw={"width_ratios": [1.35, 1.0]})
    y = np.arange(len(tasks))
    colors = [COLOR_FAILURE if rate < 70 else COLOR_ACCENT if rate < 90 else COLOR_SUCCESS for rate in rates]

    axes[0].barh(y, rates, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(tasks)
    axes[0].set_xlim(0, 105)
    axes[0].set_xlabel("Success rate (%)")
    axes[0].set_title("Task Success Rate")
    axes[0].axvline(82.15, color="#1F2937", linestyle="--", linewidth=1.2, label="Overall 82.15%")
    axes[0].legend(loc="lower right")
    for yi, rate, label in zip(y, rates, labels):
        axes[0].text(min(rate + 2.0, 101.5), yi, label, va="center", ha="left", fontsize=8, color="#111827")

    axes[1].barh(y, mean_steps, color="#457B9D", edgecolor="white", linewidth=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlim(0, 250)
    axes[1].set_xlabel("Mean end steps")
    axes[1].set_title("Completion Cost by Task")
    axes[1].axvline(240, color=COLOR_FAILURE, linestyle="--", linewidth=1.2, label="Timeout")
    axes[1].legend(loc="lower right")
    for yi, steps in zip(y, mean_steps):
        axes[1].text(min(steps + 4.0, 242.0), yi, f"{steps:.0f}", va="center", ha="left", fontsize=8)

    fig.suptitle("Specialist Profiling: Outcome by Task", fontsize=17, fontweight="bold")
    fig.text(
        0.5,
        0.012,
        "Tasks sorted from weakest to strongest. Low success rate with high mean steps indicates likely timeout-driven failure.",
        ha="center",
        fontsize=10,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    save_figure(fig, args.output)


if __name__ == "__main__":
    main()
