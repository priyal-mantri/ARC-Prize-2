"""Kaggle ARC-AGI-2 Competition Submission Adapter.

Formats solver test predictions into the exact required Kaggle submission JSON schema:
{
    "<task_id>": [
        {
            "attempt_1": [[...], ...],
            "attempt_2": [[...], ...]
        },
        ... (one entry per test input)
    ]
}

Validates schema constraints, dimensions, color ranges, and test count coverage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.data.dataset_loader import ArcTask
from src.solvers.integrated_solver import IntegratedArcSolver, TaskSolution

logger = logging.getLogger(__name__)


class SubmissionAdapter:
    """Formats and validates official Kaggle competition submission files."""

    def __init__(self, solver: Optional[IntegratedArcSolver] = None):
        self.solver = solver or IntegratedArcSolver(search_timeout_seconds=2.0)

    def generate_submission(
        self,
        test_tasks: List[ArcTask],
        output_path: Path,
        max_tasks: Optional[int] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Generate, validate, and write Kaggle submission JSON.

        Args:
            test_tasks: List of ArcTask objects to evaluate.
            output_path: Filepath where submission.json will be saved.
            max_tasks: Optional limit for dry-run testing.
            dry_run: If True, uses rapid baseline solver only.

        Returns:
            Dictionary mapping task_id to submission predictions.
        """
        targets = test_tasks[:max_tasks] if max_tasks else test_tasks
        logger.info("Generating submission for %d tasks (dry_run=%s)...", len(targets), dry_run)

        submission_dict: Dict[str, List[Dict[str, List[List[int]]]]] = {}
        solved_count = 0

        # Adjust timeout if dry_run
        if dry_run:
            self.solver.search_timeout = 0.5

        for idx, task in enumerate(targets):
            solution: TaskSolution = self.solver.solve(task)
            if solution.solved:
                solved_count += 1

            task_attempts: List[Dict[str, List[List[int]]]] = []
            for pred in solution.test_predictions:
                task_attempts.append({
                    "attempt_1": pred.attempt_1.to_list(),
                    "attempt_2": pred.attempt_2.to_list(),
                })

            submission_dict[task.task_id] = task_attempts

            if (idx + 1) % 50 == 0 or (idx + 1) == len(targets):
                logger.info("Processed %d / %d tasks for submission", idx + 1, len(targets))

        # Validate complete schema
        self.validate_submission_schema(submission_dict, targets)

        # Write to disk
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(submission_dict, f)

        logger.info(
            "Saved valid submission to %s (tasks: %d, solved: %d)",
            output_path, len(submission_dict), solved_count
        )
        return submission_dict

    @staticmethod
    def validate_submission_schema(
        submission: Dict[str, Any],
        expected_tasks: List[ArcTask],
    ) -> None:
        """Strictly validate submission formatting against competition specification.

        Raises:
            ValueError: If any structural, dimension, or typing error is detected.
        """
        if not isinstance(submission, dict):
            raise ValueError(f"Submission root must be a dict, got {type(submission)}")

        expected_ids = {t.task_id for t in expected_tasks}
        submitted_ids = set(submission.keys())

        missing = expected_ids - submitted_ids
        if missing:
            raise ValueError(f"Submission missing {len(missing)} required task IDs: {list(missing)[:5]}")

        for task in expected_tasks:
            t_id = task.task_id
            entries = submission[t_id]
            if not isinstance(entries, list):
                raise ValueError(f"Task {t_id} submission must be a list, got {type(entries)}")

            expected_test_count = len(task.test_inputs)
            if len(entries) != expected_test_count:
                raise ValueError(
                    f"Task {t_id} expected {expected_test_count} test predictions, got {len(entries)}"
                )

            for i, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    raise ValueError(f"Task {t_id} entry {i} must be a dict")
                if "attempt_1" not in entry or "attempt_2" not in entry:
                    raise ValueError(f"Task {t_id} entry {i} missing attempt_1 or attempt_2")

                for att_key in ["attempt_1", "attempt_2"]:
                    grid = entry[att_key]
                    if not isinstance(grid, list):
                        raise ValueError(f"Task {t_id} {att_key} must be a list of lists")
                    if len(grid) == 0:
                        raise ValueError(f"Task {t_id} {att_key} grid height is 0")
                    first_row_len = len(grid[0])
                    if first_row_len == 0:
                        raise ValueError(f"Task {t_id} {att_key} grid width is 0")
                    for r, row in enumerate(grid):
                        if not isinstance(row, list) or len(row) != first_row_len:
                            raise ValueError(f"Task {t_id} {att_key} row {r} has inconsistent width")
                        for c, val in enumerate(row):
                            if not isinstance(val, int) or val < 0 or val > 9:
                                raise ValueError(
                                    f"Task {t_id} {att_key} invalid pixel ({r},{c}) value {val}: must be int in 0..9"
                                )
