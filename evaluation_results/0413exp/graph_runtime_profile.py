#!/usr/bin/env python3
"""Plot runtime split between fast and slow system steps."""

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
    clipped,
    load_profile_data,
    parse_graph_args,
    pct,
    save_figure,
    values,
)


def main() -> None:
    args = parse_graph_args(__doc__ or "", "graph_runtime_profile.png")
    apply_style()
    step_rows, _ = load_profile_data(args.jsonl)

    fast_rows = [row for row in step_rows if row.get("slow_system") is False]
    slow_rows = [row for row in step_rows if row.get("slow_system") is True]
    fast_total = values(fast_rows, "model_s")
    slow_total = values(slow_rows, "model_s")
    slow_share = pct(len(slow_rows), len(step_rows))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    bp = axes[0, 0].boxplot(
        [clipped(fast_total), clipped(slow_total)],
        patch_artist=True,
        showfliers=False,
        tick_labels=["Fast-only steps", "Slow-call steps"],
    )
    for patch, color in zip(bp["boxes"], [COLOR_SUCCESS, COLOR_FAILURE]):
        patch.set(facecolor=color, alpha=0.7, edgecolor="#111827")
    axes[0, 0].set_ylabel("model_s, clipped at p99 (s)")
    axes[0, 0].set_title("Per-step Runtime Distribution")
    annotate_panel(
        axes[0, 0],
        f"Slow-call steps: {len(slow_rows):,} / {len(step_rows):,}\nShare: {slow_share:.1f}%",
        "upper left",
    )

    components = ["preprocess_s", "fast_system_s", "slow_system_s", "sample_var_s", "to_numpy_s"]
    fast_medians = [np.median(values(fast_rows, key)) if values(fast_rows, key).size else 0.0 for key in components]
    slow_medians = [np.median(values(slow_rows, key)) if values(slow_rows, key).size else 0.0 for key in components]
    x = np.arange(len(components))
    width = 0.38
    axes[0, 1].bar(x - width / 2, fast_medians, width, label="Fast-only median", color=COLOR_SUCCESS)
    axes[0, 1].bar(x + width / 2, slow_medians, width, label="Slow-call median", color=COLOR_FAILURE)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(components, rotation=25, ha="right")
    axes[0, 1].set_ylabel("Seconds")
    axes[0, 1].set_title("Median Runtime Components")
    axes[0, 1].legend(loc="upper left")

    axes[1, 0].hist(fast_total, bins=60, color=COLOR_SUCCESS, alpha=0.78, label="Fast-only", range=(0, 0.7))
    axes[1, 0].set_xlabel("model_s (s)")
    axes[1, 0].set_ylabel("Step count")
    axes[1, 0].set_title("Fast Step Runtime Shape")
    axes[1, 0].legend()
    annotate_panel(
        axes[1, 0],
        f"p50={np.percentile(fast_total, 50):.3f}s\np95={np.percentile(fast_total, 95):.3f}s",
        "upper right",
    )

    axes[1, 1].hist(clipped(slow_total, 99.5), bins=50, color=COLOR_FAILURE, alpha=0.78, label="Slow-call")
    axes[1, 1].axvline(np.percentile(slow_total, 50), color=COLOR_NEUTRAL, linestyle="--", label="p50")
    axes[1, 1].axvline(np.percentile(slow_total, 95), color=COLOR_ACCENT, linestyle="--", label="p95")
    axes[1, 1].set_xlabel("model_s (s), clipped at p99.5")
    axes[1, 1].set_ylabel("Step count")
    axes[1, 1].set_title("Slow-call Runtime Shape")
    axes[1, 1].legend()
    annotate_panel(
        axes[1, 1],
        f"p50={np.percentile(slow_total, 50):.3f}s\np95={np.percentile(slow_total, 95):.3f}s\nmax={np.max(slow_total):.3f}s",
        "upper right",
    )

    fig.suptitle("Specialist Profiling: Runtime Bottleneck", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, args.output)


if __name__ == "__main__":
    main()
