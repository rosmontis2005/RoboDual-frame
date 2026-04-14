#!/usr/bin/env python3
"""Summarize specialist profile JSONL logs.

The input file is expected to be a JSONL stream where each line is one event.
This script keeps the analysis dependency-free so it can run directly in the
experiment folder:

    python specialist_profile_conclude.py
    python specialist_profile_conclude.py /path/to/specialist_profile_rank0.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ACTION_DIMS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def pct(part: int | float, total: int | float) -> str:
    if not total:
        return "n/a"
    return f"{100.0 * float(part) / float(total):.2f}%"


def fmt(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.6g}"
    return str(value)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


class NumericSeries:
    def __init__(self) -> None:
        self.values: list[float] = []

    def add(self, value: Any) -> None:
        if is_number(value):
            self.values.append(float(value))

    @property
    def n(self) -> int:
        return len(self.values)

    def mean(self) -> float:
        return sum(self.values) / self.n if self.values else float("nan")

    def std(self) -> float:
        if self.n < 2:
            return 0.0 if self.n == 1 else float("nan")
        mean = self.mean()
        return math.sqrt(sum((x - mean) ** 2 for x in self.values) / (self.n - 1))

    def line(self) -> str:
        if not self.values:
            return "count=0"
        return (
            f"count={self.n}, mean={fmt(self.mean())}, std={fmt(self.std())}, "
            f"min={fmt(min(self.values))}, p50={fmt(percentile(self.values, 0.50))}, "
            f"p90={fmt(percentile(self.values, 0.90))}, p95={fmt(percentile(self.values, 0.95))}, "
            f"p99={fmt(percentile(self.values, 0.99))}, max={fmt(max(self.values))}"
        )


class FieldInfo:
    def __init__(self) -> None:
        self.present = 0
        self.null = 0
        self.types: Counter[str] = Counter()
        self.categories: Counter[Any] = Counter()

    def add(self, value: Any) -> None:
        self.present += 1
        if value is None:
            self.null += 1
        self.types[type(value).__name__] += 1
        if value is None or isinstance(value, (str, bool)):
            self.categories[value] += 1


class Analyzer:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.total_rows = 0
        self.bad_rows = 0
        self.step_rows = 0
        self.subtask_end_rows = 0

        self.events: Counter[str] = Counter()
        self.sequences: set[int] = set()
        self.ranks: Counter[Any] = Counter()
        self.top_fields: defaultdict[str, FieldInfo] = defaultdict(FieldInfo)
        self.profile_fields: defaultdict[str, FieldInfo] = defaultdict(FieldInfo)
        self.scalar_stats: defaultdict[str, NumericSeries] = defaultdict(NumericSeries)
        self.group_stats: defaultdict[tuple[str, str], NumericSeries] = defaultdict(NumericSeries)
        self.list_lengths: defaultdict[str, Counter[int]] = defaultdict(Counter)
        self.nested_lengths: defaultdict[str, Counter[int]] = defaultdict(Counter)
        self.vector_stats: defaultdict[str, defaultdict[int, NumericSeries]] = defaultdict(
            lambda: defaultdict(NumericSeries)
        )

        self.task_step_counts: Counter[str] = Counter()
        self.task_end_counts: Counter[str] = Counter()
        self.task_success_counts: Counter[str] = Counter()
        self.subtask_step_counts: Counter[tuple[int, int, str]] = Counter()
        self.subtask_end_steps: dict[tuple[int, int, str], int] = {}
        self.sample_k: Counter[Any] = Counter()
        self.slow_system: Counter[Any] = Counter()
        self.runtime_dtype: Counter[Any] = Counter()
        self.terminal_steps = 0
        self.step_successes = 0
        self.slowest_model_steps: list[tuple[float, dict[str, Any]]] = []

    def analyze(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    self.bad_rows += 1
                    continue
                self.total_rows += 1
                self._add_row(row, lineno)

    def _add_row(self, row: dict[str, Any], lineno: int) -> None:
        event = row.get("event")
        self.events[str(event)] += 1
        if event == "step":
            self.step_rows += 1
        elif event == "subtask_end":
            self.subtask_end_rows += 1

        sequence = row.get("sequence")
        subtask_i = row.get("subtask_i")
        task = row.get("task")
        rank = row.get("rank")
        if isinstance(sequence, int):
            self.sequences.add(sequence)
        self.ranks[rank] += 1

        for key, value in row.items():
            self.top_fields[key].add(value)
            path = f"top.{key}"
            self._add_value_stats(path, value)

        if event == "step":
            if isinstance(task, str):
                self.task_step_counts[task] += 1
            if isinstance(sequence, int) and isinstance(subtask_i, int) and isinstance(task, str):
                self.subtask_step_counts[(sequence, subtask_i, task)] += 1
            if row.get("terminal_step") is True:
                self.terminal_steps += 1
            if row.get("step_success") is True:
                self.step_successes += 1
            if is_number(row.get("model_s")):
                self.slowest_model_steps.append(
                    (
                        float(row["model_s"]),
                        {
                            "line": lineno,
                            "sequence": sequence,
                            "subtask_i": subtask_i,
                            "step": row.get("step"),
                            "task": task,
                            "slow_system": None,
                        },
                    )
                )

        if event == "subtask_end":
            if isinstance(task, str):
                self.task_end_counts[task] += 1
                if row.get("task_success") is True:
                    self.task_success_counts[task] += 1
            if isinstance(sequence, int) and isinstance(subtask_i, int) and isinstance(task, str):
                steps = row.get("steps")
                if isinstance(steps, int):
                    self.subtask_end_steps[(sequence, subtask_i, task)] = steps

        profile = row.get("profile")
        if isinstance(profile, dict):
            slow_system = profile.get("slow_system")
            self.slow_system[slow_system] += 1
            self.sample_k[profile.get("sample_k")] += 1
            self.runtime_dtype[profile.get("runtime_dtype")] += 1
            if self.slowest_model_steps:
                self.slowest_model_steps[-1][1]["slow_system"] = slow_system
            for key, value in profile.items():
                self.profile_fields[key].add(value)
                path = f"profile.{key}"
                self._add_value_stats(path, value)
                self._add_group_stats(key, value, slow_system)
        else:
            self.slow_system[None] += 1
            self.sample_k[None] += 1

    def _add_value_stats(self, path: str, value: Any) -> None:
        if is_number(value):
            self.scalar_stats[path].add(value)
        elif isinstance(value, list):
            self._add_list_stats(path, value)

    def _add_group_stats(self, key: str, value: Any, slow_system: Any) -> None:
        if is_number(value):
            group = f"slow_system={slow_system}"
            self.group_stats[(group, f"profile.{key}")].add(value)

    def _add_list_stats(self, path: str, value: list[Any]) -> None:
        self.list_lengths[path][len(value)] += 1
        if all(is_number(item) for item in value):
            for idx, item in enumerate(value):
                self.vector_stats[path][idx].add(item)
            return
        if all(isinstance(item, list) for item in value):
            for inner in value:
                self.nested_lengths[path][len(inner)] += 1
                if all(is_number(item) for item in inner):
                    for idx, item in enumerate(inner):
                        self.vector_stats[f"{path}[]"][idx].add(item)

    def print_report(self) -> None:
        print_header("File")
        size_mb = self.path.stat().st_size / (1024 * 1024)
        print(f"path: {self.path}")
        print(f"size_mb: {size_mb:.2f}")
        print(f"json_rows: {self.total_rows}")
        print(f"bad_json_rows: {self.bad_rows}")
        print(f"events: {format_counter(self.events)}")
        print(f"ranks: {format_counter(self.ranks)}")
        if self.sequences:
            print(
                f"sequences: count={len(self.sequences)}, "
                f"min={min(self.sequences)}, max={max(self.sequences)}"
            )

        print_header("Data Organization")
        print(
            "Each line is one event. Step rows include timing fields plus a profile object; "
            "subtask_end rows include steps/task_success but no profile object."
        )
        print(f"step_rows: {self.step_rows} ({pct(self.step_rows, self.total_rows)})")
        print(
            f"subtask_end_rows: {self.subtask_end_rows} "
            f"({pct(self.subtask_end_rows, self.total_rows)})"
        )
        print(f"unique_subtasks_from_steps: {len(self.subtask_step_counts)}")
        print(f"subtask_end_records: {len(self.subtask_end_steps)}")
        print(f"terminal_step_true: {self.terminal_steps} ({pct(self.terminal_steps, self.step_rows)})")
        print(f"step_success_true: {self.step_successes} ({pct(self.step_successes, self.step_rows)})")
        print(f"slow_system: {format_counter(self.slow_system)}")
        print(f"sample_k: {format_counter(self.sample_k)}")
        print(f"runtime_dtype: {format_counter(self.runtime_dtype)}")

        print_header("Top-Level Fields")
        self._print_field_table(self.top_fields, self.total_rows)

        print_header("Profile Fields")
        self._print_field_table(self.profile_fields, self.step_rows)

        print_header("Scalar Numeric Distributions")
        for path in sorted(self.scalar_stats):
            print(f"{path}: {self.scalar_stats[path].line()}")

        print_header("List And Vector Distributions")
        for path in sorted(self.list_lengths):
            print(f"{path}: list_length={format_counter(self.list_lengths[path])}")
            if path in self.nested_lengths:
                print(f"{path}: nested_length={format_counter(self.nested_lengths[path])}")
            vector_key = f"{path}[]" if f"{path}[]" in self.vector_stats else path
            if vector_key in self.vector_stats:
                for idx in sorted(self.vector_stats[vector_key]):
                    dim_name = ACTION_DIMS[idx] if idx < len(ACTION_DIMS) else str(idx)
                    print(
                        f"  {vector_key}[{idx}:{dim_name}]: "
                        f"{self.vector_stats[vector_key][idx].line()}"
                    )

        print_header("Runtime By Slow/Fast System")
        interesting = {
            "profile.total_s",
            "profile.fast_system_s",
            "profile.slow_system_s",
            "profile.preprocess_s",
            "profile.sample_var_s",
            "profile.to_numpy_s",
        }
        for group, path in sorted(self.group_stats):
            if path in interesting:
                print(f"{group} {path}: {self.group_stats[(group, path)].line()}")

        print_header("Task Outcomes")
        total_ends = sum(self.task_end_counts.values())
        total_success = sum(self.task_success_counts.values())
        print(f"subtask_end_total: {total_ends}")
        print(f"subtask_success_total: {total_success} ({pct(total_success, total_ends)})")
        print("task, step_rows, subtask_ends, successes, success_rate, mean_end_steps")
        for task in sorted(set(self.task_step_counts) | set(self.task_end_counts)):
            end_count = self.task_end_counts[task]
            success = self.task_success_counts[task]
            end_steps = [
                steps
                for (sequence, subtask_i, name), steps in self.subtask_end_steps.items()
                if name == task
            ]
            mean_steps = sum(end_steps) / len(end_steps) if end_steps else float("nan")
            print(
                f"{task}, {self.task_step_counts[task]}, {end_count}, {success}, "
                f"{pct(success, end_count)}, {fmt(mean_steps)}"
            )

        print_header("Slowest Model Steps")
        for value, meta in sorted(
            self.slowest_model_steps, key=lambda item: item[0], reverse=True
        )[:10]:
            print(
                f"model_s={fmt(value)}, line={meta['line']}, sequence={meta['sequence']}, "
                f"subtask_i={meta['subtask_i']}, step={meta['step']}, task={meta['task']}, "
                f"slow_system={meta['slow_system']}"
            )

    def _print_field_table(self, fields: dict[str, FieldInfo], denominator: int) -> None:
        print("field, present, present_rate, null, null_rate_within_present, types, categories")
        for key in sorted(fields):
            info = fields[key]
            categories = ""
            if info.categories:
                categories = format_counter(info.categories, limit=8)
            print(
                f"{key}, {info.present}, {pct(info.present, denominator)}, "
                f"{info.null}, {pct(info.null, info.present)}, "
                f"{format_counter(info.types)}, {categories}"
            )


def format_counter(counter: Counter[Any], limit: int | None = None) -> str:
    items = counter.most_common(limit)
    body = ", ".join(f"{repr(key)}:{value}" for key, value in items)
    if limit is not None and len(counter) > limit:
        body += f", ...(+{len(counter) - limit})"
    return "{" + body + "}"


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jsonl",
        nargs="?",
        default=Path(__file__).with_name("specialist_profile_rank0.jsonl"),
        type=Path,
        help="Path to specialist_profile_rank0.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.jsonl.expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")
    analyzer = Analyzer(path)
    analyzer.analyze()
    analyzer.print_report()


if __name__ == "__main__":
    main()
