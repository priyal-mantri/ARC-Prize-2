"""Hunter Symbolic Baseline Solver.

A fast, deterministic pattern-matching baseline for common ARC transformations:
geometric symmetries, color mappings, crops, gravity, and tiling.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.data.dataset_loader import ArcTask
from src.representation.grid import ArcGrid
from src.dsl.primitives import PRIMITIVE_REGISTRY
from src.dsl.program import DSLProgram, PrimitiveCall
from src.search.verifier import DemonstrationVerifier, TaskDemonstration, VerificationResult

logger = logging.getLogger(__name__)


class HunterBaselineSolver:
    """Fast symbolic baseline testing atomic geometric, color, and structural templates."""

    def __init__(self, verifier: Optional[DemonstrationVerifier] = None):
        self.verifier = verifier or DemonstrationVerifier(timeout_seconds=0.05)

    def solve(self, task: ArcTask) -> Optional[DSLProgram]:
        """Attempt to solve the task using rapid pattern matching rules.

        Args:
            task: ARC task to solve.

        Returns:
            DSLProgram if a fully verified pattern explains all demonstrations, else None.
        """
        demonstrations = [
            TaskDemonstration(
                input_grid=ArcGrid(np_grid=ex.input_grid),
                output_grid=ArcGrid(np_grid=ex.output_grid),
            )
            for ex in task.train_examples
        ]
        if not demonstrations:
            return None

        # 1. Identity Check
        if all(np.array_equal(d.input_grid.to_numpy(), d.output_grid.to_numpy()) for d in demonstrations):
            return DSLProgram(name="identity", calls=[PrimitiveCall("identity")], origin="hunter_baseline")

        # 2. Geometric Rigid Symmetries
        rigid_ops = [
            "rot90", "rot180", "rot270",
            "flip_horizontal", "flip_vertical",
            "transpose", "reflect_main_diagonal", "reflect_anti_diagonal"
        ]
        for op in rigid_ops:
            prog = DSLProgram(name=op, calls=[PrimitiveCall(op)], origin="hunter_baseline")
            res = self.verifier.verify(prog, demonstrations)
            if res.is_fully_solved:
                return prog

        # 3. Direct Bounding Box Cropping
        crop_ops = ["crop_to_content", "crop_to_largest_object"]
        for op in crop_ops:
            prog = DSLProgram(name=op, calls=[PrimitiveCall(op)], origin="hunter_baseline")
            res = self.verifier.verify(prog, demonstrations)
            if res.is_fully_solved:
                return prog

        # 4. Pure Color Substitution
        # Check if same shape on all demos and pure 1-to-1 color permutation
        same_shapes = all(d.input_grid.shape == d.output_grid.shape for d in demonstrations)
        if same_shapes:
            color_map: Dict[int, int] = {}
            valid_map = True
            for d in demonstrations:
                in_arr = d.input_grid.to_numpy()
                out_arr = d.output_grid.to_numpy()
                for c_in in np.unique(in_arr):
                    c_out_candidates = np.unique(out_arr[in_arr == c_in])
                    if len(c_out_candidates) != 1:
                        valid_map = False
                        break
                    c_out = int(c_out_candidates[0])
                    if c_in in color_map and color_map[c_in] != c_out:
                        valid_map = False
                        break
                    color_map[c_in] = c_out
                if not valid_map:
                    break

            if valid_map and any(k != v for k, v in color_map.items()):
                calls = [
                    PrimitiveCall("replace_color", {"old_color": k, "new_color": v})
                    for k, v in color_map.items() if k != v
                ]
                prog = DSLProgram(name="hunter_color_map", calls=calls, origin="hunter_baseline")
                res = self.verifier.verify(prog, demonstrations)
                if res.is_fully_solved:
                    return prog

        # 5. Gravity Rules
        gravity_ops = ["gravity_down", "gravity_up", "gravity_left", "gravity_right"]
        for op in gravity_ops:
            prog = DSLProgram(name=op, calls=[PrimitiveCall(op)], origin="hunter_baseline")
            res = self.verifier.verify(prog, demonstrations)
            if res.is_fully_solved:
                return prog

        # 6. Symmetrical Completion Rules
        symmetry_ops = ["complete_symmetry_horizontal", "complete_symmetry_vertical"]
        for op in symmetry_ops:
            prog = DSLProgram(name=op, calls=[PrimitiveCall(op)], origin="hunter_baseline")
            res = self.verifier.verify(prog, demonstrations)
            if res.is_fully_solved:
                return prog

        # 7. Constant 1x1 Counting
        if all(d.output_grid.shape == (1, 1) for d in demonstrations):
            # Check if output is number of objects or unique colors
            for d in demonstrations:
                out_val = int(d.output_grid.to_numpy()[0, 0])
                num_colors = len(d.input_grid.unique_colors) - (1 if 0 in d.input_grid.unique_colors else 0)
                # Check object count
            # More complex counting can be found via beam search
            pass

        return None
