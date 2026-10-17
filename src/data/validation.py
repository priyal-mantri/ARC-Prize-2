"""
Validation utilities for ARC-AGI-2 grids, tasks, and datasets.

Enforces ARC-AGI domain constraints:
1. Grids must be 2D rectangular matrices.
2. Grid dimensions must be within [1, 30] in height and width.
3. Grid cell values must be integers in range [0, 9] (10 canonical colors).
4. Tasks must have valid 'train' pairs and 'test' inputs.
5. Solutions, where present, must match test input counts and have valid grids.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class ValidationResult:
    """Stores the aggregated results of a dataset or task validation run."""
    dataset_name: str
    total_tasks: int = 0
    valid_tasks: int = 0
    malformed_tasks: int = 0
    total_grids_checked: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Returns True if there are zero errors in the validation result."""
        return len(self.errors) == 0

    def summary(self) -> str:
        """Generates a human-readable summary of validation results."""
        status = "PASSED (Clean)" if self.is_valid else f"FAILED ({len(self.errors)} errors)"
        lines = [
            f"=== Validation Summary: {self.dataset_name} ===",
            f"Status: {status}",
            f"Total Tasks Inspected: {self.total_tasks}",
            f"Valid Tasks: {self.valid_tasks}",
            f"Malformed Tasks: {self.malformed_tasks}",
            f"Total Grids Checked: {self.total_grids_checked}",
            f"Total Errors: {len(self.errors)}",
            f"Total Warnings: {len(self.warnings)}",
        ]
        if self.errors:
            lines.append("Sample Errors (up to 5):")
            for err in self.errors[:5]:
                lines.append(f"  - {err}")
        if self.warnings:
            lines.append("Sample Warnings (up to 5):")
            for warn in self.warnings[:5]:
                lines.append(f"  - {warn}")
        return "\n".join(lines)


def validate_grid(
    grid: Any,
    min_dim: int = 1,
    max_dim: int = 30,
    valid_colors: Optional[range] = None,
) -> Tuple[bool, List[str]]:
    """
    Validates that a grid conforms to the ARC specification.

    Args:
        grid: List of lists or 2D numpy array representing an ARC grid.
        min_dim: Minimum allowed dimension (default: 1).
        max_dim: Maximum allowed dimension (default: 30).
        valid_colors: Range or set of allowed integer colors (default: 0-9).

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    errors: List[str] = []
    if valid_colors is None:
        valid_colors = range(10)

    if grid is None:
        return False, ["Grid is None."]

    # Convert numpy array to list if needed
    if isinstance(grid, np.ndarray):
        if grid.ndim != 2:
            return False, [f"Grid must be 2D, got {grid.ndim}D array with shape {grid.shape}."]
        h, w = grid.shape
        if h < min_dim or h > max_dim:
            errors.append(f"Grid height {h} outside allowed range [{min_dim}, {max_dim}].")
        if w < min_dim or w > max_dim:
            errors.append(f"Grid width {w} outside allowed range [{min_dim}, {max_dim}].")
        
        # Check integer dtype
        if not np.issubdtype(grid.dtype, np.integer):
            errors.append(f"Grid dtype must be integer, got {grid.dtype}.")
        else:
            invalid_vals = set(grid.flatten()) - set(valid_colors)
            if invalid_vals:
                errors.append(f"Grid contains invalid color integers: {sorted(list(invalid_vals))}.")
        return len(errors) == 0, errors

    # Check list of lists representation
    if not isinstance(grid, list):
        return False, [f"Grid must be a list of lists or 2D numpy array, got {type(grid).__name__}."]

    h = len(grid)
    if h < min_dim or h > max_dim:
        errors.append(f"Grid height {h} outside allowed range [{min_dim}, {max_dim}].")
        if h == 0:
            return False, errors

    if not isinstance(grid[0], list):
        return False, [f"First row of grid is not a list (got {type(grid[0]).__name__})."]

    w = len(grid[0])
    if w < min_dim or w > max_dim:
        errors.append(f"Grid width {w} outside allowed range [{min_dim}, {max_dim}].")

    for r_idx, row in enumerate(grid):
        if not isinstance(row, list):
            errors.append(f"Row {r_idx} is not a list (got {type(row).__name__}).")
            continue
        if len(row) != w:
            errors.append(f"Row {r_idx} has width {len(row)}, expected {w} (non-rectangular grid).")
        for c_idx, cell in enumerate(row):
            if not isinstance(cell, int) or isinstance(cell, bool):
                errors.append(f"Cell at ({r_idx}, {c_idx}) is not an integer (got {type(cell).__name__}: {cell}).")
            elif cell not in valid_colors:
                errors.append(f"Cell at ({r_idx}, {c_idx}) has invalid color value {cell} (expected 0-9).")

    return len(errors) == 0, errors


def validate_task(
    task_id: str,
    task_dict: Any,
    solutions_list: Optional[List[Any]] = None,
) -> Tuple[bool, List[str], List[str], int]:
    """
    Validates the structure of a single ARC task dictionary and its optional solutions.

    Args:
        task_id: Unique task identifier string.
        task_dict: Dictionary containing 'train' and 'test' keys.
        solutions_list: Optional list of solution grids for the test inputs.

    Returns:
        Tuple of (is_valid, errors, warnings, total_grids_checked).
    """
    errors: List[str] = []
    warnings: List[str] = []
    grids_checked = 0

    if not isinstance(task_dict, dict):
        return False, [f"Task '{task_id}' data is not a dictionary (got {type(task_dict).__name__})."], warnings, 0

    if "train" not in task_dict:
        errors.append(f"Task '{task_id}' is missing required key 'train'.")
    if "test" not in task_dict:
        errors.append(f"Task '{task_id}' is missing required key 'test'.")

    if errors:
        return False, errors, warnings, 0

    train_pairs = task_dict["train"]
    if not isinstance(train_pairs, list) or len(train_pairs) == 0:
        errors.append(f"Task '{task_id}': 'train' must be a non-empty list of example pairs.")
    else:
        for idx, pair in enumerate(train_pairs):
            if not isinstance(pair, dict):
                errors.append(f"Task '{task_id}': train example {idx} is not a dict.")
                continue
            if "input" not in pair:
                errors.append(f"Task '{task_id}': train example {idx} missing 'input'.")
            else:
                ok, g_errs = validate_grid(pair["input"])
                grids_checked += 1
                if not ok:
                    errors.extend([f"Task '{task_id}' train[{idx}] input: {e}" for e in g_errs])

            if "output" not in pair:
                errors.append(f"Task '{task_id}': train example {idx} missing 'output'.")
            else:
                ok, g_errs = validate_grid(pair["output"])
                grids_checked += 1
                if not ok:
                    errors.extend([f"Task '{task_id}' train[{idx}] output: {e}" for e in g_errs])

    test_inputs = task_dict["test"]
    if not isinstance(test_inputs, list) or len(test_inputs) == 0:
        errors.append(f"Task '{task_id}': 'test' must be a non-empty list of test inputs.")
    else:
        for idx, item in enumerate(test_inputs):
            if not isinstance(item, dict):
                errors.append(f"Task '{task_id}': test item {idx} is not a dict.")
                continue
            if "input" not in item:
                errors.append(f"Task '{task_id}': test item {idx} missing 'input'.")
            else:
                ok, g_errs = validate_grid(item["input"])
                grids_checked += 1
                if not ok:
                    errors.extend([f"Task '{task_id}' test[{idx}] input: {e}" for e in g_errs])

            # In challenge files, test items should NOT contain ground truth outputs
            if "output" in item:
                warnings.append(
                    f"Task '{task_id}': test item {idx} contains 'output' in challenge dictionary (potential leak)."
                )
                ok, g_errs = validate_grid(item["output"])
                grids_checked += 1
                if not ok:
                    errors.extend([f"Task '{task_id}' test[{idx}] output: {e}" for e in g_errs])

    if solutions_list is not None:
        if not isinstance(solutions_list, list):
            errors.append(f"Task '{task_id}': solution must be a list of grids, got {type(solutions_list).__name__}.")
        else:
            expected_count = len(test_inputs) if isinstance(test_inputs, list) else 0
            if len(solutions_list) != expected_count:
                errors.append(
                    f"Task '{task_id}': solution count ({len(solutions_list)}) does not match test input count ({expected_count})."
                )
            for idx, sol_grid in enumerate(solutions_list):
                ok, g_errs = validate_grid(sol_grid)
                grids_checked += 1
                if not ok:
                    errors.extend([f"Task '{task_id}' solution[{idx}]: {e}" for e in g_errs])

    return len(errors) == 0, errors, warnings, grids_checked


def validate_dataset(
    challenges: Dict[str, Any],
    solutions: Optional[Dict[str, Any]] = None,
    dataset_name: str = "ARC Dataset",
) -> ValidationResult:
    """
    Performs full structural validation across an entire ARC challenges dictionary.

    Args:
        challenges: Dict mapping task_id to task dictionary.
        solutions: Optional dict mapping task_id to list of solution grids.
        dataset_name: Identifier label for reporting.

    Returns:
        ValidationResult with aggregated counts, status, errors, and warnings.
    """
    result = ValidationResult(dataset_name=dataset_name)
    result.total_tasks = len(challenges)

    # Check for solutions consistency if provided
    if solutions is not None:
        sol_keys = set(solutions.keys())
        chal_keys = set(challenges.keys())
        missing_sols = chal_keys - sol_keys
        extra_sols = sol_keys - chal_keys
        if missing_sols:
            result.errors.append(
                f"{len(missing_sols)} tasks in challenges lack corresponding entries in solutions dictionary."
            )
        if extra_sols:
            result.warnings.append(
                f"{len(extra_sols)} tasks in solutions dictionary do not exist in challenges."
            )

    for task_id, task_data in challenges.items():
        sol_list = solutions.get(task_id) if solutions is not None else None
        is_valid, errors, warnings, grids_checked = validate_task(task_id, task_data, sol_list)
        result.total_grids_checked += grids_checked

        if is_valid:
            result.valid_tasks += 1
        else:
            result.malformed_tasks += 1
            result.errors.extend(errors)

        if warnings:
            result.warnings.extend(warnings)

    return result
