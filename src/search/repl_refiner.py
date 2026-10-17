"""REPL Refinement and Failure-Driven Candidate Repair.

Analyzes execution verification failures and generates targeted repairs
such as color remapping, geometric alignment, padding/cropping adjustments,
and post-processing transformations.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.representation.grid import ArcGrid
from src.dsl.primitives import PRIMITIVE_REGISTRY
from src.dsl.program import DSLProgram, PrimitiveCall
from src.search.verifier import DemonstrationVerifier, VerificationResult, TaskDemonstration

logger = logging.getLogger(__name__)


class REPLRefiner:
    """Proposes refined programs based on diagnostic traces of near-miss candidates."""

    def __init__(self, verifier: DemonstrationVerifier, max_refinements_per_candidate: int = 5):
        self.verifier = verifier
        self.max_refinements_per_candidate = max_refinements_per_candidate

    def refine(
        self,
        candidate: DSLProgram,
        verification: VerificationResult,
        demonstrations: List[TaskDemonstration],
    ) -> List[DSLProgram]:
        """Generate targeted program repairs based on verification diagnostics.

        Args:
            candidate: Candidate DSL program that failed or partially succeeded.
            verification: VerificationResult containing diagnostics per demonstration.
            demonstrations: List of task demonstrations.

        Returns:
            List of refined DSLProgram candidates.
        """
        if verification.is_fully_solved:
            return [candidate]

        # Only attempt refinement if program executed without crashing on at least one demonstration
        # and has a non-trivial pixel accuracy (> 0.5 across non-empty demos)
        if verification.mean_pixel_accuracy < 0.40 and not any(
            d.shape_match for d in verification.demonstration_diagnostics
        ):
            return []

        refined_programs: List[DSLProgram] = []

        # Strategy 1: Global Color Remapping
        # If shape matches on all demos, check if consistent color mapping exists
        color_repair = self._try_color_remapping(candidate, verification, demonstrations)
        if color_repair is not None:
            refined_programs.append(color_repair)

        # Strategy 2: Transposition / Orientation Repair
        # If shapes are transposed (H, W) vs (W, H)
        transposed_repair = self._try_transposition_repair(candidate, verification, demonstrations)
        if transposed_repair is not None:
            refined_programs.extend(transposed_repair)

        # Strategy 3: Cropping / Bounding Box Repair
        # If prediction is larger than target, check if content crop matches target
        crop_repair = self._try_crop_repair(candidate, verification, demonstrations)
        if crop_repair is not None:
            refined_programs.append(crop_repair)

        # Strategy 4: Noise Removal / Hole Filling Post-processing
        post_proc_repairs = self._try_post_processing_repairs(candidate, verification, demonstrations)
        refined_programs.extend(post_proc_repairs)

        # Deduplicate refined programs
        unique_refined: List[DSLProgram] = []
        seen_reprs = {candidate.to_code()}
        for prog in refined_programs:
            code = prog.to_code()
            if code not in seen_reprs:
                seen_reprs.add(code)
                unique_refined.append(prog)
                if len(unique_refined) >= self.max_refinements_per_candidate:
                    break

        return unique_refined

    def _try_color_remapping(
        self,
        candidate: DSLProgram,
        verification: VerificationResult,
        demonstrations: List[TaskDemonstration],
    ) -> Optional[DSLProgram]:
        """If shapes match on all demos but colors differ, find a consistent color map."""
        if not all(d.shape_match for d in verification.demonstration_diagnostics):
            return None

        # Build consistent mapping across all demos: pred_color -> target_color
        global_map: Dict[int, int] = {}

        for diag, demo in zip(verification.demonstration_diagnostics, demonstrations):
            if diag.predicted_output is None:
                return None
            pred_arr = diag.predicted_output.to_numpy()
            target_arr = demo.output_grid.to_numpy()

            # For each pixel, check mapping
            pred_uniques = np.unique(pred_arr)
            for c in pred_uniques:
                mask = (pred_arr == c)
                targets = target_arr[mask]
                if len(targets) == 0:
                    continue
                # Find most common target color for this predicted color
                vals, counts = np.unique(targets, return_counts=True)
                dominant_target = vals[np.argmax(counts)]
                dominant_ratio = np.max(counts) / len(targets)

                # Must be very consistent (> 90% of pixels of color c map to dominant_target)
                if dominant_ratio < 0.90:
                    return None

                if c in global_map:
                    if global_map[c] != dominant_target:
                        return None  # Inconsistent across demos
                else:
                    global_map[c] = int(dominant_target)

        # Check if map is non-trivial (i.e. changes at least one color)
        changes = {k: v for k, v in global_map.items() if k != v}
        if not changes:
            return None

        # Create new program with recolor_substitute call
        new_calls = list(candidate.calls)
        for old_c, new_c in changes.items():
            new_calls.append(PrimitiveCall("replace_color", {"old_color": old_c, "new_color": new_c}))

        repaired_prog = DSLProgram(
            name=f"{candidate.name}_recolored",
            calls=new_calls,
            origin=f"repl_refinement(color_remap:{changes})",
        )
        return repaired_prog

    def _try_transposition_repair(
        self,
        candidate: DSLProgram,
        verification: VerificationResult,
        demonstrations: List[TaskDemonstration],
    ) -> List[DSLProgram]:
        """If predictions are transposed in shape, propose rotations and reflections."""
        repairs: List[DSLProgram] = []
        needs_orientation = False

        for diag, demo in zip(verification.demonstration_diagnostics, demonstrations):
            if diag.predicted_output is None:
                continue
            pred_h, pred_w = diag.predicted_output.shape
            targ_h, targ_w = demo.output_grid.shape
            if pred_h == targ_w and pred_w == targ_h and pred_h != targ_h:
                needs_orientation = True
                break

        if not needs_orientation:
            return repairs

        # Propose transpose, rot90, rot270
        for prim_name in ["transpose", "rot90", "rot270"]:
            new_calls = list(candidate.calls) + [PrimitiveCall(prim_name)]
            repairs.append(
                DSLProgram(
                    name=f"{candidate.name}_{prim_name}",
                    calls=new_calls,
                    origin=f"repl_refinement({prim_name})",
                )
            )

        return repairs

    def _try_crop_repair(
        self,
        candidate: DSLProgram,
        verification: VerificationResult,
        demonstrations: List[TaskDemonstration],
    ) -> Optional[DSLProgram]:
        """If prediction is larger than target, check if crop_to_content fixes it."""
        all_pred_larger = True
        for diag, demo in zip(verification.demonstration_diagnostics, demonstrations):
            if diag.predicted_output is None:
                all_pred_larger = False
                break
            pred_h, pred_w = diag.predicted_output.shape
            targ_h, targ_w = demo.output_grid.shape
            if pred_h < targ_h or pred_w < targ_w:
                all_pred_larger = False
                break

        if not all_pred_larger:
            return None

        new_calls = list(candidate.calls) + [PrimitiveCall("crop_to_content")]
        return DSLProgram(
            name=f"{candidate.name}_cropped",
            calls=new_calls,
            origin="repl_refinement(crop_to_content)",
        )

    def _try_post_processing_repairs(
        self,
        candidate: DSLProgram,
        verification: VerificationResult,
        demonstrations: List[TaskDemonstration],
    ) -> List[DSLProgram]:
        """Propose light post-processing if pixel accuracy is very high (> 85%)."""
        if verification.mean_pixel_accuracy < 0.85:
            return []

        repairs: List[DSLProgram] = []
        for post_op in ["fill_enclosed_holes", "gravity_down", "gravity_up"]:
            if post_op in PRIMITIVE_REGISTRY:
                new_calls = list(candidate.calls) + [PrimitiveCall(post_op)]
                repairs.append(
                    DSLProgram(
                        name=f"{candidate.name}_{post_op}",
                        calls=new_calls,
                        origin=f"repl_refinement({post_op})",
                    )
                )

        return repairs
