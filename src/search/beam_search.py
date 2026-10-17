"""Compositional Enumerative and Beam Search over ARC DSL Primitives.

Implements constrained program synthesis with early pruning, beam expansion,
multi-demonstration verification, and REPL refinement.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.data.dataset_loader import ArcTask
from src.representation.grid import ArcGrid
from src.dsl.primitives import PRIMITIVE_REGISTRY, get_primitives_by_category
from src.dsl.program import DSLProgram, PrimitiveCall
from src.search.constraint_filter import ConstraintFilter
from src.search.verifier import DemonstrationVerifier, VerificationResult, TaskDemonstration
from src.search.repl_refiner import REPLRefiner

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Outcome of a DSL beam search on a task."""
    task_id: str
    solved: bool
    best_program: Optional[DSLProgram]
    candidate_programs: List[DSLProgram] = field(default_factory=list)
    verification_results: List[VerificationResult] = field(default_factory=list)
    candidates_generated: int = 0
    candidates_pruned: int = 0
    candidates_executed: int = 0
    search_duration_sec: float = 0.0
    depth_reached: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class DSLBeamSearch:
    """Constrained enumerative and beam search synthesizer for ARC tasks."""

    def __init__(
        self,
        verifier: Optional[DemonstrationVerifier] = None,
        beam_width: int = 16,
        max_depth: int = 2,
        timeout_seconds: float = 4.0,
        enable_repl_refinement: bool = True,
    ):
        self.verifier = verifier or DemonstrationVerifier(timeout_seconds=0.1)
        self.refiner = REPLRefiner(self.verifier)
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.timeout_seconds = timeout_seconds
        self.enable_repl_refinement = enable_repl_refinement

    def search(self, task: ArcTask, timeout: Optional[float] = None) -> SearchResult:
        """Execute constrained DSL search to find programs explaining all training demonstrations.

        Args:
            task: The ArcTask to solve.
            timeout: Optional override for search timeout.

        Returns:
            SearchResult detailing search outcome, best program, and stats.
        """
        start_time = time.time()
        deadline = start_time + (timeout if timeout is not None else self.timeout_seconds)

        # Convert task train examples to TaskDemonstrations
        demonstrations = [
            TaskDemonstration(
                input_grid=ArcGrid(np_grid=ex.input_grid),
                output_grid=ArcGrid(np_grid=ex.output_grid),
            )
            for ex in task.train_examples
        ]

        if not demonstrations:
            return SearchResult(
                task_id=task.task_id,
                solved=False,
                best_program=None,
                search_duration_sec=0.0,
            )

        # Build constraint filter from demonstrations
        constraints = ConstraintFilter.from_demonstrations(demonstrations)

        candidates_generated = 0
        candidates_pruned = 0
        candidates_executed = 0

        # Step 0: Extract task-specific parameters (colors, scale factors)
        in_colors: Set[int] = set()
        out_colors: Set[int] = set()
        for demo in demonstrations:
            in_colors.update(demo.input_grid.unique_colors)
            out_colors.update(demo.output_grid.unique_colors)

        all_colors = in_colors.union(out_colors)

        # Generate Depth 1 candidate programs
        depth1_programs: List[DSLProgram] = []

        # 1. Zero-parameter primitives
        for name in PRIMITIVE_REGISTRY.keys():
            if not constraints.is_primitive_admissible(name):
                candidates_pruned += 1
                continue

            # Check if primitive requires parameters
            if name in {
                "replace_color", "crop", "pad", "scale_up", "tile_grid",
                "keep_only_color", "remove_color", "keep_largest_color", "keep_smallest_color"
            }:
                continue

            prog = DSLProgram(name=name, calls=[PrimitiveCall(name)], origin="enumerative_d1")
            depth1_programs.append(prog)
            candidates_generated += 1

        # 2. Parameterized primitives tailored to colors in task
        # replace_color(c1, c2)
        if constraints.is_primitive_admissible("replace_color"):
            for c_in in in_colors:
                for c_out in out_colors:
                    if c_in != c_out:
                        prog = DSLProgram(
                            name=f"replace_{c_in}_with_{c_out}",
                            calls=[PrimitiveCall("replace_color", {"old_color": c_in, "new_color": c_out})],
                            origin="enumerative_color_d1",
                        )
                        depth1_programs.append(prog)
                        candidates_generated += 1

        # keep_only_color(c)
        if constraints.is_primitive_admissible("keep_only_color"):
            for c in in_colors:
                prog = DSLProgram(
                    name=f"keep_only_{c}",
                    calls=[PrimitiveCall("keep_only_color", {"color": c})],
                    origin="enumerative_color_d1",
                )
                depth1_programs.append(prog)
                candidates_generated += 1

        # scale_up(factor)
        if constraints.is_primitive_admissible("scale_up"):
            for factor in [2, 3]:
                prog = DSLProgram(
                    name=f"scale_up_{factor}",
                    calls=[PrimitiveCall("scale_up", {"factor": factor})],
                    origin="enumerative_scale_d1",
                )
                depth1_programs.append(prog)
                candidates_generated += 1

        # tile_grid(times_h, times_w)
        if constraints.is_primitive_admissible("tile_grid"):
            for th in [1, 2, 3]:
                for tw in [1, 2, 3]:
                    if th == 1 and tw == 1:
                        continue
                    prog = DSLProgram(
                        name=f"tile_{th}x{tw}",
                        calls=[PrimitiveCall("tile_grid", {"times_h": th, "times_w": tw})],
                        origin="enumerative_tile_d1",
                    )
                    depth1_programs.append(prog)
                    candidates_generated += 1

        # Evaluate Depth 1
        evaluated_candidates: List[Tuple[DSLProgram, VerificationResult]] = []

        for prog in depth1_programs:
            if time.time() > deadline:
                break
            v_res = self.verifier.verify(prog, demonstrations)
            candidates_executed += 1

            if v_res.is_fully_solved:
                # Instant solve at Depth 1!
                elapsed = time.time() - start_time
                return SearchResult(
                    task_id=task.task_id,
                    solved=True,
                    best_program=prog,
                    candidate_programs=[prog],
                    verification_results=[v_res],
                    candidates_generated=candidates_generated,
                    candidates_pruned=candidates_pruned,
                    candidates_executed=candidates_executed,
                    search_duration_sec=elapsed,
                    depth_reached=1,
                    diagnostics={"instant_solve_d1": prog.name},
                )

            # Store for beam selection
            evaluated_candidates.append((prog, v_res))

        # Check REPL Refinement on Depth 1 promising candidates
        if self.enable_repl_refinement and time.time() < deadline:
            for prog, v_res in evaluated_candidates[:8]:
                if time.time() > deadline:
                    break
                refined = self.refiner.refine(prog, v_res, demonstrations)
                for ref_prog in refined:
                    candidates_generated += 1
                    ref_vres = self.verifier.verify(ref_prog, demonstrations)
                    candidates_executed += 1
                    if ref_vres.is_fully_solved:
                        elapsed = time.time() - start_time
                        return SearchResult(
                            task_id=task.task_id,
                            solved=True,
                            best_program=ref_prog,
                            candidate_programs=[ref_prog],
                            verification_results=[ref_vres],
                            candidates_generated=candidates_generated,
                            candidates_pruned=candidates_pruned,
                            candidates_executed=candidates_executed,
                            search_duration_sec=elapsed,
                            depth_reached=1,
                            diagnostics={"solved_via_repl_d1": ref_prog.name},
                        )
                    evaluated_candidates.append((ref_prog, ref_vres))

        # Sort evaluated candidates to form Beam
        # Score formula: shape_matches count * 2.0 + mean_pixel_accuracy - complexity * 0.05
        def score_candidate(pair: Tuple[DSLProgram, VerificationResult]) -> float:
            p, v = pair
            shape_score = sum(1.0 for d in v.demonstration_diagnostics if d.shape_match) / len(demonstrations)
            return shape_score * 2.0 + v.mean_pixel_accuracy - (p.complexity * 0.05)

        evaluated_candidates.sort(key=score_candidate, reverse=True)
        beam = evaluated_candidates[: self.beam_width]

        # Step 2: Depth 2 Expansion
        if self.max_depth >= 2 and time.time() < deadline:
            depth2_candidates: List[DSLProgram] = []

            # Primitives for depth 2 second step (filtering, geometry, cleanup)
            step2_prims = [
                "rot90", "rot180", "rot270", "flip_horizontal", "flip_vertical",
                "transpose", "crop_to_content", "fill_enclosed_holes",
                "gravity_down", "gravity_up", "gravity_left", "gravity_right",
                "outline_objects", "hollow_objects", "keep_largest_object",
                "keep_smallest_object", "complete_symmetry_horizontal",
                "complete_symmetry_vertical"
            ]

            # Add color replacements
            color_calls = []
            for c_in in in_colors:
                for c_out in out_colors:
                    if c_in != c_out:
                        color_calls.append(PrimitiveCall("replace_color", {"old_color": c_in, "new_color": c_out}))

            for base_prog, base_vres in beam:
                if time.time() > deadline:
                    break

                # Append step2 primitive
                for p_name in step2_prims:
                    if not constraints.is_primitive_admissible(p_name):
                        continue
                    new_prog = DSLProgram(
                        name=f"{base_prog.name}+{p_name}",
                        calls=list(base_prog.calls) + [PrimitiveCall(p_name)],
                        origin="beam_d2",
                    )
                    depth2_candidates.append(new_prog)
                    candidates_generated += 1

                for c_call in color_calls[:4]:
                    new_prog = DSLProgram(
                        name=f"{base_prog.name}+{c_call.name}_{c_call.params.get('old_color')}_{c_call.params.get('new_color')}",
                        calls=list(base_prog.calls) + [c_call],
                        origin="beam_d2_color",
                    )
                    depth2_candidates.append(new_prog)
                    candidates_generated += 1

            # Evaluate Depth 2 candidates
            for d2_prog in depth2_candidates:
                if time.time() > deadline:
                    break
                d2_vres = self.verifier.verify(d2_prog, demonstrations)
                candidates_executed += 1

                if d2_vres.is_fully_solved:
                    elapsed = time.time() - start_time
                    return SearchResult(
                        task_id=task.task_id,
                        solved=True,
                        best_program=d2_prog,
                        candidate_programs=[d2_prog],
                        verification_results=[d2_vres],
                        candidates_generated=candidates_generated,
                        candidates_pruned=candidates_pruned,
                        candidates_executed=candidates_executed,
                        search_duration_sec=elapsed,
                        depth_reached=2,
                        diagnostics={"instant_solve_d2": d2_prog.name},
                    )

                evaluated_candidates.append((d2_prog, d2_vres))

            # REPL Refinement on Depth 2 candidates
            if self.enable_repl_refinement and time.time() < deadline:
                evaluated_candidates.sort(key=score_candidate, reverse=True)
                for prog, v_res in evaluated_candidates[:6]:
                    if time.time() > deadline:
                        break
                    refined = self.refiner.refine(prog, v_res, demonstrations)
                    for ref_prog in refined:
                        candidates_generated += 1
                        ref_vres = self.verifier.verify(ref_prog, demonstrations)
                        candidates_executed += 1
                        if ref_vres.is_fully_solved:
                            elapsed = time.time() - start_time
                            return SearchResult(
                                task_id=task.task_id,
                                solved=True,
                                best_program=ref_prog,
                                candidate_programs=[ref_prog],
                                verification_results=[ref_vres],
                                candidates_generated=candidates_generated,
                                candidates_pruned=candidates_pruned,
                                candidates_executed=candidates_executed,
                                search_duration_sec=elapsed,
                                depth_reached=2,
                                diagnostics={"solved_via_repl_d2": ref_prog.name},
                            )
                        evaluated_candidates.append((ref_prog, ref_vres))

        # Final ranking of all surviving candidates
        evaluated_candidates.sort(key=score_candidate, reverse=True)
        top_candidates = evaluated_candidates[:5]

        best_prog = top_candidates[0][0] if top_candidates else None
        best_vres = top_candidates[0][1] if top_candidates else None
        is_solved = best_vres.is_fully_solved if best_vres else False

        elapsed = time.time() - start_time
        return SearchResult(
            task_id=task.task_id,
            solved=is_solved,
            best_program=best_prog,
            candidate_programs=[p for p, _ in top_candidates],
            verification_results=[v for _, v in top_candidates],
            candidates_generated=candidates_generated,
            candidates_pruned=candidates_pruned,
            candidates_executed=candidates_executed,
            search_duration_sec=elapsed,
            depth_reached=2 if self.max_depth >= 2 else 1,
            diagnostics={
                "top_score": score_candidate(top_candidates[0]) if top_candidates else 0.0,
                "top_accuracy": best_vres.mean_pixel_accuracy if best_vres else 0.0,
            },
        )
