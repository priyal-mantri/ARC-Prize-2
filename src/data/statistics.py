"""
Statistical analysis tools for ARC-AGI-2 datasets.

Extracts grid geometry distributions, training pair frequencies, test input distributions,
color usage metrics, and multi-test task breakdowns.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class DatasetStatistics:
    """Encapsulates descriptive and structural statistics of an ARC dataset."""
    name: str
    total_tasks: int = 0
    total_train_examples: int = 0
    total_test_inputs: int = 0
    total_solutions: int = 0

    # Task breakdown
    train_examples_per_task: Dict[int, int] = field(default_factory=dict)
    test_inputs_per_task: Dict[int, int] = field(default_factory=dict)
    multi_test_task_ids: List[str] = field(default_factory=list)

    # Grid geometry
    dim_same_pairs: int = 0
    dim_diff_pairs: int = 0
    square_grids: int = 0
    non_square_grids: int = 0
    input_height_min: int = 0
    input_height_max: int = 0
    input_width_min: int = 0
    input_width_max: int = 0
    output_height_min: int = 0
    output_height_max: int = 0
    output_width_min: int = 0
    output_width_max: int = 0
    top_grid_dimensions: List[Tuple[Tuple[int, int], int]] = field(default_factory=list)

    # Colors
    color_cell_counts: Dict[int, int] = field(default_factory=dict)
    color_cell_percentages: Dict[int, float] = field(default_factory=dict)
    unique_colors_per_grid: Dict[int, int] = field(default_factory=dict)

    @property
    def avg_train_examples(self) -> float:
        return self.total_train_examples / self.total_tasks if self.total_tasks > 0 else 0.0

    @property
    def avg_test_inputs(self) -> float:
        return self.total_test_inputs / self.total_tasks if self.total_tasks > 0 else 0.0

    @property
    def dim_same_percentage(self) -> float:
        total_pairs = self.dim_same_pairs + self.dim_diff_pairs
        return (self.dim_same_pairs / total_pairs * 100.0) if total_pairs > 0 else 0.0

    @property
    def square_grid_percentage(self) -> float:
        total = self.square_grids + self.non_square_grids
        return (self.square_grids / total * 100.0) if total > 0 else 0.0

    def to_markdown_summary(self) -> str:
        """Formats the dataset statistics into a detailed Markdown report table."""
        lines = [
            f"### Dataset Audit Statistics: {self.name}",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Total Tasks** | `{self.total_tasks:,}` |",
            f"| **Total Training Examples** | `{self.total_train_examples:,}` (mean: `{self.avg_train_examples:.2f}`/task) |",
            f"| **Total Test Inputs** | `{self.total_test_inputs:,}` (mean: `{self.avg_test_inputs:.2f}`/task) |",
            f"| **Total Available Solutions** | `{self.total_solutions:,}` |",
            f"| **Multi-Test Tasks (>1 test)** | `{len(self.multi_test_task_ids):,}` tasks |",
            f"| **Dimension Preservation** | `{self.dim_same_pairs:,}` same dim (`{self.dim_same_percentage:.1f}%`), `{self.dim_diff_pairs:,}` resized |",
            f"| **Square Grids** | `{self.square_grids:,}` (`{self.square_grid_percentage:.1f}%`), `{self.non_square_grids:,}` non-square |",
            f"| **Input Dimensions (H × W)** | `[{self.input_height_min}, {self.input_height_max}] × [{self.input_width_min}, {self.input_width_max}]` |",
            f"| **Output Dimensions (H × W)** | `[{self.output_height_min}, {self.output_height_max}] × [{self.output_width_min}, {self.output_width_max}]` |",
            "",
            "#### Training Examples per Task Distribution",
            "| Number of Examples | Task Count | Percentage |",
            "| :---: | :---: | :---: |",
        ]
        for k in sorted(self.train_examples_per_task.keys()):
            cnt = self.train_examples_per_task[k]
            pct = cnt / self.total_tasks * 100.0 if self.total_tasks > 0 else 0.0
            lines.append(f"| {k} | {cnt:,} | {pct:.1f}% |")

        lines.extend([
            "",
            "#### Test Inputs per Task Distribution",
            "| Number of Test Inputs | Task Count | Percentage |",
            "| :---: | :---: | :---: |",
        ])
        for k in sorted(self.test_inputs_per_task.keys()):
            cnt = self.test_inputs_per_task[k]
            pct = cnt / self.total_tasks * 100.0 if self.total_tasks > 0 else 0.0
            lines.append(f"| {k} | {cnt:,} | {pct:.1f}% |")

        lines.extend([
            "",
            "#### Color Frequency Breakdown (Cells)",
            "| Color ID | Color Name | Cell Count | Percentage |",
            "| :---: | :--- | :---: | :---: |",
        ])
        color_names = [
            "0: Black", "1: Blue", "2: Red", "3: Green", "4: Yellow",
            "5: Grey", "6: Magenta", "7: Orange", "8: Teal", "9: Maroon"
        ]
        for c in range(10):
            cnt = self.color_cell_counts.get(c, 0)
            pct = self.color_cell_percentages.get(c, 0.0)
            lines.append(f"| {c} | {color_names[c]} | {cnt:,} | {pct:.2f}% |")

        return "\n".join(lines)


def compute_dataset_statistics(
    challenges: Dict[str, Any],
    solutions: Optional[Dict[str, Any]] = None,
    name: str = "ARC Dataset",
) -> DatasetStatistics:
    """
    Computes comprehensive structural and color statistics over a dataset.

    Args:
        challenges: Dict mapping task_id to task dict.
        solutions: Optional dict mapping task_id to list of solution grids.
        name: Human-readable name for reporting.

    Returns:
        DatasetStatistics object.
    """
    stats = DatasetStatistics(name=name, total_tasks=len(challenges))

    train_ex_counts: Counter = Counter()
    test_in_counts: Counter = Counter()
    color_counts: Counter = Counter()
    unique_colors_counter: Counter = Counter()
    dim_counter: Counter = Counter()

    in_hs, in_ws = [], []
    out_hs, out_ws = [], []

    for task_id, task in challenges.items():
        train_pairs = task.get("train", [])
        test_inputs = task.get("test", [])

        num_train = len(train_pairs)
        num_test = len(test_inputs)

        stats.total_train_examples += num_train
        stats.total_test_inputs += num_test
        train_ex_counts[num_train] += 1
        test_in_counts[num_test] += 1

        if num_test > 1:
            stats.multi_test_task_ids.append(task_id)

        # Inspect train pairs
        for pair in train_pairs:
            inp = pair.get("input", [])
            out = pair.get("output", [])

            if inp and len(inp) > 0 and len(inp[0]) > 0:
                h_in, w_in = len(inp), len(inp[0])
                in_hs.append(h_in)
                in_ws.append(w_in)
                dim_counter[(h_in, w_in)] += 1
                if h_in == w_in:
                    stats.square_grids += 1
                else:
                    stats.non_square_grids += 1

                colors_in = set()
                for r in inp:
                    for c in r:
                        color_counts[c] += 1
                        colors_in.add(c)
                unique_colors_counter[len(colors_in)] += 1

            if out and len(out) > 0 and len(out[0]) > 0:
                h_out, w_out = len(out), len(out[0])
                out_hs.append(h_out)
                out_ws.append(w_out)
                dim_counter[(h_out, w_out)] += 1
                if h_out == w_out:
                    stats.square_grids += 1
                else:
                    stats.non_square_grids += 1

                colors_out = set()
                for r in out:
                    for c in r:
                        color_counts[c] += 1
                        colors_out.add(c)
                unique_colors_counter[len(colors_out)] += 1

            if inp and out and len(inp) > 0 and len(out) > 0:
                if (len(inp), len(inp[0])) == (len(out), len(out[0])):
                    stats.dim_same_pairs += 1
                else:
                    stats.dim_diff_pairs += 1

        # Inspect test inputs
        for item in test_inputs:
            inp = item.get("input", [])
            if inp and len(inp) > 0 and len(inp[0]) > 0:
                h, w = len(inp), len(inp[0])
                in_hs.append(h)
                in_ws.append(w)
                dim_counter[(h, w)] += 1
                if h == w:
                    stats.square_grids += 1
                else:
                    stats.non_square_grids += 1

                colors_in = set()
                for r in inp:
                    for c in r:
                        color_counts[c] += 1
                        colors_in.add(c)
                unique_colors_counter[len(colors_in)] += 1

        # Inspect solutions if provided
        if solutions and task_id in solutions:
            sol_list = solutions[task_id]
            stats.total_solutions += len(sol_list)
            for sol_grid in sol_list:
                if sol_grid and len(sol_grid) > 0 and len(sol_grid[0]) > 0:
                    h, w = len(sol_grid), len(sol_grid[0])
                    out_hs.append(h)
                    out_ws.append(w)
                    dim_counter[(h, w)] += 1
                    if h == w:
                        stats.square_grids += 1
                    else:
                        stats.non_square_grids += 1

    stats.train_examples_per_task = dict(sorted(train_ex_counts.items()))
    stats.test_inputs_per_task = dict(sorted(test_in_counts.items()))
    stats.color_cell_counts = {c: color_counts[c] for c in range(10)}

    total_cells = sum(color_counts.values())
    if total_cells > 0:
        stats.color_cell_percentages = {
            c: (color_counts[c] / total_cells * 100.0) for c in range(10)
        }
    else:
        stats.color_cell_percentages = {c: 0.0 for c in range(10)}

    stats.unique_colors_per_grid = dict(sorted(unique_colors_counter.items()))
    stats.top_grid_dimensions = dim_counter.most_common(10)

    if in_hs and in_ws:
        stats.input_height_min, stats.input_height_max = min(in_hs), max(in_hs)
        stats.input_width_min, stats.input_width_max = min(in_ws), max(in_ws)
    if out_hs and out_ws:
        stats.output_height_min, stats.output_height_max = min(out_hs), max(out_hs)
        stats.output_width_min, stats.output_width_max = min(out_ws), max(out_ws)

    return stats


def check_task_overlaps(datasets: Dict[str, Set[str]]) -> Dict[Tuple[str, str], Set[str]]:
    """
    Checks pairwise task ID overlap across all provided dataset key sets.

    Args:
        datasets: Dict mapping dataset label to set of task IDs.

    Returns:
        Dict mapping (label_a, label_b) to the set of overlapping task IDs.
    """
    overlaps = {}
    names = list(datasets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = datasets[a].intersection(datasets[b])
            overlaps[(a, b)] = overlap
    return overlaps
