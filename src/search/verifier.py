"""Verification and Failure Analysis Engine for ARC-AGI-2 candidate programs.

Evaluates synthesized candidate programs against demonstration pairs, asserting
100% exact pixel matches while collecting granular error diagnostics when imperfect.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.ndimage

from src.representation.grid import ArcGrid
from src.dsl.program import DSLProgram


@dataclass
class DemonstrationDiagnostic:
    """Detailed error diagnostics for a single demonstration pair."""
    pair_index: int
    shape_match: bool
    predicted_shape: Tuple[int, int]
    target_shape: Tuple[int, int]
    pixel_accuracy: float  # 0.0 to 1.0
    mismatched_pixels_count: int
    colors_missing: List[int]
    colors_surplus: List[int]
    object_count_delta: int
    quadrant_error_counts: Dict[str, int]
    predicted_output: Optional[ArcGrid] = None
    error_message: Optional[str] = None


@dataclass
class VerificationResult:
    """Aggregate verification results across all demonstration pairs for a candidate program."""
    program_name: str
    is_fully_solved: bool  # 100% exact pixel match on EVERY demonstration pair
    mean_pixel_accuracy: float  # 0.0 to 1.0
    demonstration_diagnostics: List[DemonstrationDiagnostic] = field(default_factory=list)
    execution_time_sec: float = 0.0
    error_message: Optional[str] = None

    @property
    def is_solution(self) -> bool:
        return self.is_fully_solved


@dataclass
class TaskDemonstration:
    """A single input/output training demonstration pair."""
    input_grid: ArcGrid
    output_grid: ArcGrid


# Legacy / backwards compatibility aliases
FailureDiagnostic = DemonstrationDiagnostic
CandidateVerificationResult = VerificationResult


class DemonstrationVerifier:
    """Evaluates candidate DSL programs or callable functions against demonstration pairs."""

    def __init__(self, timeout_seconds: float = 0.2):
        self.timeout_seconds = timeout_seconds

    def verify(
        self,
        program: DSLProgram,
        demonstrations: List[TaskDemonstration],
    ) -> VerificationResult:
        """Execute a DSLProgram on a list of TaskDemonstrations and evaluate correctness."""
        start_t = time.perf_counter()
        diagnostics: List[DemonstrationDiagnostic] = []
        all_perfect = True
        err_msg: Optional[str] = None
        accuracies: List[float] = []

        for idx, demo in enumerate(demonstrations):
            inp_grid = demo.input_grid
            tgt_grid = demo.output_grid
            tgt_arr = tgt_grid.to_numpy()

            try:
                pred_grid = program.execute(inp_grid)
                if pred_grid is None:
                    raise RuntimeError("Program execution returned None")
                pred_arr = pred_grid.to_numpy()
            except Exception as e:
                err_msg = f"Runtime error on demo {idx}: {type(e).__name__}: {str(e)}"
                all_perfect = False
                accuracies.append(0.0)
                diagnostics.append(
                    DemonstrationDiagnostic(
                        pair_index=idx,
                        shape_match=False,
                        predicted_shape=(0, 0),
                        target_shape=tgt_grid.shape,
                        pixel_accuracy=0.0,
                        mismatched_pixels_count=tgt_grid.size,
                        colors_missing=sorted(list(tgt_grid.unique_colors)),
                        colors_surplus=[],
                        object_count_delta=-1,
                        quadrant_error_counts={},
                        predicted_output=None,
                        error_message=str(e),
                    )
                )
                continue

            # Compare shape
            shape_ok = bool(pred_arr.shape == tgt_arr.shape)
            if not shape_ok:
                all_perfect = False
                accuracies.append(0.0)
                diagnostics.append(
                    DemonstrationDiagnostic(
                        pair_index=idx,
                        shape_match=False,
                        predicted_shape=pred_grid.shape,
                        target_shape=tgt_grid.shape,
                        pixel_accuracy=0.0,
                        mismatched_pixels_count=tgt_grid.size,
                        colors_missing=sorted(list(set(tgt_grid.unique_colors) - set(pred_grid.unique_colors))),
                        colors_surplus=sorted(list(set(pred_grid.unique_colors) - set(tgt_grid.unique_colors))),
                        object_count_delta=0,
                        quadrant_error_counts={},
                        predicted_output=pred_grid,
                        error_message="Shape mismatch",
                    )
                )
                continue

            # Exact pixel match comparison
            match_mask = (pred_arr == tgt_arr)
            mismatch_count = int(np.sum(~match_mask))
            acc = float(np.mean(match_mask))  # in 0.0 .. 1.0
            accuracies.append(acc)

            if mismatch_count > 0:
                all_perfect = False

            # Diagnostic metrics
            missing_c = sorted(list(set(tgt_grid.unique_colors) - set(pred_grid.unique_colors)))
            surplus_c = sorted(list(set(pred_grid.unique_colors) - set(tgt_grid.unique_colors)))

            # Quadrant errors
            h, w = tgt_arr.shape
            mid_r, mid_c = h // 2, w // 2
            err_grid = ~match_mask
            quadrants = {
                "top_left": int(np.sum(err_grid[:mid_r, :mid_c])),
                "top_right": int(np.sum(err_grid[:mid_r, mid_c:])),
                "bottom_left": int(np.sum(err_grid[mid_r:, :mid_c])),
                "bottom_right": int(np.sum(err_grid[mid_r:, mid_c:])),
            }

            # Object count delta
            _, num_p = scipy.ndimage.label(pred_arr > 0)
            _, num_t = scipy.ndimage.label(tgt_arr > 0)

            diagnostics.append(
                DemonstrationDiagnostic(
                    pair_index=idx,
                    shape_match=True,
                    predicted_shape=pred_grid.shape,
                    target_shape=tgt_grid.shape,
                    pixel_accuracy=acc,
                    mismatched_pixels_count=mismatch_count,
                    colors_missing=missing_c,
                    colors_surplus=surplus_c,
                    object_count_delta=int(num_p - num_t),
                    quadrant_error_counts=quadrants,
                    predicted_output=pred_grid,
                )
            )

        elapsed = time.perf_counter() - start_t
        mean_acc = float(np.mean(accuracies)) if accuracies else 0.0

        return VerificationResult(
            program_name=program.name,
            is_fully_solved=all_perfect,
            mean_pixel_accuracy=mean_acc,
            demonstration_diagnostics=diagnostics,
            execution_time_sec=elapsed,
            error_message=err_msg,
        )

    def verify_candidate(
        self,
        candidate_func: Callable[[np.ndarray], np.ndarray],
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        candidate_id: str = "candidate",
    ) -> VerificationResult:
        """Legacy helper for verifying raw numpy callable against pairs."""
        from src.dsl.program import LambdaProgram
        prog = LambdaProgram(candidate_id, candidate_func)
        demos = [
            TaskDemonstration(ArcGrid(np_grid=p[0]), ArcGrid(np_grid=p[1]))
            for p in train_pairs
        ]
        return self.verify(prog, demos)
