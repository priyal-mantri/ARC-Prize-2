"""Reproducible Benchmarking and Experiment Tracking Harness.

Evaluates solvers on specified ARC dataset splits with strict separation between
solver-facing task objects and held-out evaluation ground truth.
Reports task-level accuracy, test-input-level accuracy, stage breakdowns, and timing.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.data.dataset_loader import ArcTask
from src.solvers.integrated_solver import IntegratedArcSolver, TaskSolution

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkTaskResult:
    """Individual task evaluation record."""
    task_id: str
    solved: bool
    num_test_inputs: int
    test_inputs_correct: int
    execution_time_sec: float
    solver_stage: str
    best_program_name: Optional[str]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Aggregated benchmark evaluation summary."""
    split_name: str
    timestamp: str
    git_revision: str
    total_tasks_attempted: int
    tasks_solved: int
    task_level_accuracy: float
    total_test_inputs: int
    test_inputs_correct: int
    test_input_level_accuracy: float
    mean_execution_time_sec: float
    total_duration_sec: float
    stage_breakdown: Dict[str, int] = field(default_factory=dict)
    task_results: List[BenchmarkTaskResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown_summary(self) -> str:
        """Render a clean Markdown summary table."""
        lines = [
            f"# Benchmark Evaluation: {self.split_name.upper()} Split",
            f"- **Timestamp**: {self.timestamp}",
            f"- **Git Revision**: `{self.git_revision}`",
            f"- **Tasks Attempted**: {self.total_tasks_attempted}",
            f"- **Tasks Solved**: {self.tasks_solved} ({self.task_level_accuracy:.2%})",
            f"- **Test-Input Accuracy**: {self.test_inputs_correct} / {self.total_test_inputs} ({self.test_input_level_accuracy:.2%})",
            f"- **Total Duration**: {self.total_duration_sec:.2f}s (mean: {self.mean_execution_time_sec:.3f}s/task)",
            "",
            "## Solver Stage Breakdown",
            "| Stage | Solved Tasks | Percentage |",
            "|---|---|---|",
        ]
        for stage, count in sorted(self.stage_breakdown.items(), key=lambda x: x[1], reverse=True):
            pct = (count / self.tasks_solved) * 100 if self.tasks_solved > 0 else 0.0
            lines.append(f"| `{stage}` | {count} | {pct:.1f}% |")

        return "\n".join(lines)


class BenchmarkHarness:
    """Experiment runner and accuracy auditor for ARC tasks."""

    def __init__(self, solver: Optional[IntegratedArcSolver] = None):
        self.solver = solver or IntegratedArcSolver()

    def run_benchmark(
        self,
        tasks: List[ArcTask],
        solutions: Optional[Dict[str, List[np.ndarray]]] = None,
        split_name: str = "custom",
        max_tasks: Optional[int] = None,
        save_dir: Optional[Path] = None,
    ) -> BenchmarkReport:
        """Execute benchmark evaluation across a set of tasks.

        Args:
            tasks: Target ArcTask instances (contains only demonstrations and test inputs).
            solutions: Explicit ground-truth solutions dictionary {task_id: [test_1_np, ...]}.
            split_name: Name of the split (e.g. 'training_subset', 'evaluation').
            max_tasks: Optional limit for fast runs.
            save_dir: Optional directory to persist report JSON and Markdown.

        Returns:
            BenchmarkReport containing comprehensive metrics.
        """
        targets = tasks[:max_tasks] if max_tasks else tasks
        logger.info("Executing benchmark on split '%s' with %d tasks...", split_name, len(targets))

        git_rev = self._get_git_revision()
        start_bench_time = time.time()

        task_results: List[BenchmarkTaskResult] = []
        stage_counts: Dict[str, int] = {}
        total_test_inputs = 0
        total_test_correct = 0
        tasks_solved = 0

        for idx, task in enumerate(targets):
            solution = self.solver.solve(task)
            num_tests = len(task.test_inputs)
            total_test_inputs += num_tests

            # Check correctness against explicit solutions if provided
            test_correct_count = 0
            if solutions and task.task_id in solutions:
                ground_truths = solutions[task.task_id]
                for t_idx, pred in enumerate(solution.test_predictions):
                    if t_idx < len(ground_truths):
                        target_np = ground_truths[t_idx]
                        att1_match = np.array_equal(pred.attempt_1.to_numpy(), target_np)
                        att2_match = np.array_equal(pred.attempt_2.to_numpy(), target_np)
                        if att1_match or att2_match:
                            test_correct_count += 1
            else:
                # If ground truth not provided, task is considered solved if solver verified demonstrations
                test_correct_count = num_tests if solution.solved else 0

            # Under ARC rules, a task is solved only if ALL its test inputs are solved!
            is_task_solved = (test_correct_count == num_tests) and num_tests > 0

            if is_task_solved:
                tasks_solved += 1
                stage = solution.solver_pipeline_stage
                stage_counts[stage] = stage_counts.get(stage, 0) + 1

            total_test_correct += test_correct_count

            task_results.append(
                BenchmarkTaskResult(
                    task_id=task.task_id,
                    solved=is_task_solved,
                    num_test_inputs=num_tests,
                    test_inputs_correct=test_correct_count,
                    execution_time_sec=solution.execution_time_sec,
                    solver_stage=solution.solver_pipeline_stage,
                    best_program_name=solution.best_program.name if solution.best_program else None,
                    diagnostics=solution.diagnostics,
                )
            )

            if (idx + 1) % 20 == 0 or (idx + 1) == len(targets):
                logger.info(
                    "Benchmark [%s] %d / %d processed. Solved: %d (%.2f%%)",
                    split_name, idx + 1, len(targets), tasks_solved, (tasks_solved / (idx + 1)) * 100
                )

        total_duration = time.time() - start_bench_time
        mean_time = total_duration / len(targets) if targets else 0.0
        task_acc = (tasks_solved / len(targets)) if targets else 0.0
        test_acc = (total_test_correct / total_test_inputs) if total_test_inputs > 0 else 0.0

        report = BenchmarkReport(
            split_name=split_name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            git_revision=git_rev,
            total_tasks_attempted=len(targets),
            tasks_solved=tasks_solved,
            task_level_accuracy=task_acc,
            total_test_inputs=total_test_inputs,
            test_inputs_correct=total_test_correct,
            test_input_level_accuracy=test_acc,
            mean_execution_time_sec=mean_time,
            total_duration_sec=total_duration,
            stage_breakdown=stage_counts,
            task_results=task_results,
        )

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            json_path = save_dir / f"benchmark_{split_name}.json"
            md_path = save_dir / f"benchmark_{split_name}.md"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(report.to_markdown_summary())
            logger.info("Saved benchmark reports to %s and %s", json_path, md_path)

        return report

    def _get_git_revision(self) -> str:
        """Retrieve current git revision hash safely."""
        try:
            rev = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            return rev
        except Exception:
            return "unknown_revision"
