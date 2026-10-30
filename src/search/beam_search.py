"""Compositional Enumerative and Beam Search over ARC DSL Primitives.

Implements constrained program synthesis with early pruning, beam expansion,
multi-demonstration verification, and REPL refinement.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.data.dataset_loader import ArcTask
from src.representation.grid import ArcGrid
from src.dsl.primitives import PRIMITIVE_REGISTRY, PRIMITIVE_SPECS, get_primitives_by_category
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
        max_candidates: int = 40,
    ):
        self.verifier = verifier or DemonstrationVerifier(timeout_seconds=0.1)
        self.refiner = REPLRefiner(self.verifier)
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.timeout_seconds = timeout_seconds
        self.enable_repl_refinement = enable_repl_refinement
        self.max_candidates = max_candidates

    def search(
        self,
        task: ArcTask,
        timeout: Optional[float] = None,
        max_candidates: Optional[int] = None,
        scene_graph_facts: Optional[Dict[str, Any]] = None,
        seed_programs: Optional[List[DSLProgram]] = None,
    ) -> SearchResult:
        """Execute constrained DSL search to find programs explaining all training demonstrations.

        Args:
            task: The ArcTask to solve.
            timeout: Optional override for search timeout.
            max_candidates: Optional override for candidate proposal budget.
            scene_graph_facts: Optional structural facts from scene graph reasoning.

        Returns:
            SearchResult detailing search outcome, best program, and stats.
        """
        start_time = time.time()
        deadline = start_time + (timeout if timeout is not None else self.timeout_seconds)
        budget = max_candidates if max_candidates is not None else self.max_candidates

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
        task_has_changes = False

        for demo in demonstrations:
            in_colors.update(demo.input_grid.unique_colors)
            out_colors.update(demo.output_grid.unique_colors)
            if not np.array_equal(demo.input_grid.to_numpy(), demo.output_grid.to_numpy()):
                task_has_changes = True

        new_colors = out_colors - in_colors
        eliminated_colors = in_colors - out_colors

        # Generate Depth 1 candidate programs using Schema-Driven Registry
        depth1_programs: List[DSLProgram] = []
        seen_signatures: Set[str] = set()

        def add_candidate(prog: DSLProgram) -> bool:
            nonlocal candidates_generated
            if budget is not None and len(depth1_programs) >= budget:
                return False
            sig = prog.to_code()
            if sig in seen_signatures:
                return False
            seen_signatures.add(sig)
            depth1_programs.append(prog)
            candidates_generated += 1
            return True

        # Scene graph structural hints
        has_holes = False
        if scene_graph_facts and scene_graph_facts.get("has_holes"):
            has_holes = True

        # Ingest external seed programs (retrieved memory skills and/or Qwen proposals)
        if seed_programs:
            for sp in seed_programs:
                add_candidate(sp)

        # High-priority parameterized candidates guided by scene graph & invariants:
        # 1. Enclosed region filling if new colors exist or scene graph indicates holes
        if new_colors or has_holes:
            fill_candidates = list(new_colors) if new_colors else [c for c in out_colors if c != 0]
            for fc in fill_candidates:
                add_candidate(DSLProgram(
                    name=f"fill_enclosed_{fc}",
                    calls=[PrimitiveCall("fill_enclosed", {"fill_color": fc})],
                    origin="scene_graph_topology_hint",
                ))
            if len(new_colors) > 1 or constraints.is_primitive_admissible("fill_enclosed_by_size"):
                add_candidate(DSLProgram(
                    name="fill_enclosed_by_size",
                    calls=[PrimitiveCall("fill_enclosed_by_size")],
                    origin="scene_graph_topology_size_hint",
                ))

        # 1b. Probe-conditioned recoloring if an auxiliary input color was eliminated
        if eliminated_colors and constraints.is_primitive_admissible("recolor_by_probe_key"):
            for elim_c in eliminated_colors:
                for main_c in (in_colors - eliminated_colors - {0}):
                    add_candidate(DSLProgram(
                        name=f"recolor_by_probe_{elim_c}_{main_c}",
                        calls=[PrimitiveCall("recolor_by_probe_key", {"probe_color": elim_c, "main_color": main_c})],
                        origin="relational_probe_key_hint",
                    ))

        # 2. Schema-Driven Zero-parameter primitives, prioritized by dimensional behavior
        compatible_zero_param = []
        other_zero_param = []
        for name, spec in PRIMITIVE_SPECS.items():
            if not constraints.is_primitive_admissible(name):
                candidates_pruned += 1
                continue

            if spec.is_zero_param:
                if constraints.task_constraints and constraints.task_constraints.dimensions_preserved:
                    if spec.preserves_dimensions:
                        compatible_zero_param.append(name)
                    else:
                        other_zero_param.append(name)
                else:
                    if not spec.preserves_dimensions:
                        compatible_zero_param.append(name)
                    else:
                        other_zero_param.append(name)

        for name in compatible_zero_param + other_zero_param:
            add_candidate(DSLProgram(
                name=name,
                calls=[PrimitiveCall(name)],
                origin="enumerative_zero_param_d1",
            ))

        # 3. Principled Parameterized Primitives tailored to demonstrations
        # Color replacement
        if constraints.is_primitive_admissible("replace_color") or constraints.is_primitive_admissible("change_color"):
            for c_in in in_colors:
                for c_out in out_colors:
                    if c_in != c_out:
                        add_candidate(DSLProgram(
                            name=f"replace_{c_in}_with_{c_out}",
                            calls=[PrimitiveCall("replace_color", {"old_color": c_in, "new_color": c_out})],
                            origin="enumerative_color_d1",
                        ))

        # Keep only color
        if constraints.is_primitive_admissible("keep_only_color"):
            for c in in_colors:
                if c != 0:
                    add_candidate(DSLProgram(
                        name=f"keep_only_{c}",
                        calls=[PrimitiveCall("keep_only_color", {"target_color": c})],
                        origin="enumerative_color_d1",
                    ))

        # Scaling and Tiling
        if constraints.is_primitive_admissible("scale_grid") or constraints.is_primitive_admissible("scale_up"):
            for factor in [2, 3]:
                add_candidate(DSLProgram(
                    name=f"scale_grid_{factor}",
                    calls=[PrimitiveCall("scale_grid", {"multiplier": factor})],
                    origin="enumerative_scale_d1",
                ))

        if constraints.is_primitive_admissible("tile_grid"):
            for th in [2, 3]:
                add_candidate(DSLProgram(
                    name=f"tile_grid_{th}",
                    calls=[PrimitiveCall("tile_grid", {"multiplier": th})],
                    origin="enumerative_tile_d1",
                ))

        # Alternating horizontal tiling
        if constraints.is_primitive_admissible("tile_alternating_h"):
            add_candidate(DSLProgram(
                name="tile_alternating_h_3x3",
                calls=[PrimitiveCall("tile_alternating_h", {"rep_h": 3, "rep_w": 3})],
                origin="enumerative_tile_d1",
            ))

        # Crop to color bounding box
        if constraints.is_primitive_admissible("extract_color_bounding_box"):
            for c in in_colors:
                if c != 0:
                    add_candidate(DSLProgram(
                        name=f"bbox_{c}",
                        calls=[PrimitiveCall("extract_color_bounding_box", {"target_color": c})],
                        origin="enumerative_crop_d1",
                    ))

        # Respect candidate proposal budget
        if budget and len(depth1_programs) > budget:
            depth1_programs = depth1_programs[:budget]

        # Evaluate Depth 1 candidates
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
                if time.time() > deadline or (budget is not None and candidates_generated >= budget):
                    break
                refined = self.refiner.refine(prog, v_res, demonstrations)
                for ref_prog in refined:
                    if budget is not None and candidates_generated >= budget:
                        break
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
        # Score formula: exact matches + shape matches + pixel accuracy - complexity
        def score_candidate(pair: Tuple[DSLProgram, VerificationResult]) -> float:
            p, v = pair
            if v.is_fully_solved:
                return 100.0 - (p.complexity * 0.01)

            # Heavily penalize identity when demonstrations show changes
            if task_has_changes and (
                p.name == "identity" or (len(p.calls) == 1 and p.calls[0].primitive_name == "identity")
            ):
                return -10.0

            n_demos = max(1, len(demonstrations))
            shape_score = sum(1.0 for d in v.demonstration_diagnostics if d.shape_match) / n_demos
            exact_score = sum(1.0 for d in v.demonstration_diagnostics if d.shape_match and d.pixel_accuracy == 1.0) / n_demos

            if shape_score == 0:
                return -5.0 - (p.complexity * 0.05)

            return (
                (exact_score * 10.0) +
                (shape_score * 2.0) +
                (v.mean_pixel_accuracy * 3.0) -
                (p.complexity * 0.05)
            )

        evaluated_candidates.sort(key=score_candidate, reverse=True)
        beam = evaluated_candidates[: self.beam_width]

        # Step 2: Depth 2 Expansion
        if self.max_depth >= 2 and time.time() < deadline:
            depth2_candidates: List[DSLProgram] = []

            # Zero-parameter primitives for depth 2 second step (filtering, geometry, cleanup)
            step2_prims = [
                "rot90", "rot180", "rot270", "flip_horizontal", "flip_vertical",
                "transpose", "crop_to_content", "crop_to_largest_object",
                "gravity_down", "gravity_up", "gravity_left", "gravity_right",
                "outline_objects", "hollow_out", "keep_largest_object",
                "keep_smallest_object", "complete_symmetry_horizontal",
                "complete_symmetry_vertical", "invert_colors", "remove_noise"
            ]

            # Add color replacements
            color_calls = []
            for c_in in in_colors:
                for c_out in out_colors:
                    if c_in != c_out:
                        color_calls.append(PrimitiveCall("replace_color", {"old_color": c_in, "new_color": c_out}))

            # Add fill enclosed if new colors exist
            if new_colors:
                for fc in new_colors:
                    color_calls.append(PrimitiveCall("fill_enclosed", {"fill_color": fc}))

            for base_prog, base_vres in beam:
                if time.time() > deadline or candidates_generated >= budget:
                    break

                # Append step2 primitive
                for p_name in step2_prims:
                    if candidates_generated >= budget:
                        break
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
                    if candidates_generated >= budget:
                        break
                    new_prog = DSLProgram(
                        name=f"{base_prog.name}+{c_call.name}_{c_call.params.get('old_color', '')}_{c_call.params.get('new_color', c_call.params.get('fill_color', ''))}",
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
                    if time.time() > deadline or (budget is not None and candidates_generated >= budget):
                        break
                    refined = self.refiner.refine(prog, v_res, demonstrations)
                    for ref_prog in refined:
                        if budget is not None and candidates_generated >= budget:
                            break
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

        # Step 3: Depth 3 Expansion (Bounded Composition)
        max_depth_reached = 2 if self.max_depth >= 2 else 1
        if self.max_depth >= 3 and time.time() < deadline:
            max_depth_reached = 3
            depth3_candidates: List[DSLProgram] = []
            evaluated_candidates.sort(key=score_candidate, reverse=True)
            beam_d3 = evaluated_candidates[: min(8, self.beam_width)]

            step3_prims = [
                "rot90", "rot180", "rot270", "flip_horizontal", "flip_vertical",
                "transpose", "crop_to_content", "keep_largest_object",
                "keep_smallest_object", "remove_noise", "invert_colors",
                "hollow_out", "complete_symmetry_horizontal", "complete_symmetry_vertical"
            ]

            for base_prog, base_vres in beam_d3:
                if time.time() > deadline or candidates_generated >= budget:
                    break

                # Intermediate validation: execute base program on first demonstration
                try:
                    inter_grid = base_prog.execute(demonstrations[0].input_grid)
                    if inter_grid is None or inter_grid.size == 0 or inter_grid.shape[0] > 30 or inter_grid.shape[1] > 30:
                        candidates_pruned += 1
                        continue
                except Exception:
                    candidates_pruned += 1
                    continue

                for p_name in step3_prims:
                    if candidates_generated >= budget:
                        break
                    if not constraints.is_primitive_admissible(p_name):
                        continue
                    new_prog = DSLProgram(
                        name=f"{base_prog.name}+{p_name}",
                        calls=list(base_prog.calls) + [PrimitiveCall(p_name)],
                        origin="beam_d3_composed",
                    )
                    depth3_candidates.append(new_prog)
                    candidates_generated += 1

                for c_call in color_calls[:2]:
                    if candidates_generated >= budget:
                        break
                    new_prog = DSLProgram(
                        name=f"{base_prog.name}+{c_call.name}",
                        calls=list(base_prog.calls) + [c_call],
                        origin="beam_d3_color",
                    )
                    depth3_candidates.append(new_prog)
                    candidates_generated += 1

            # Evaluate Depth 3 candidates
            for d3_prog in depth3_candidates:
                if time.time() > deadline:
                    break
                d3_vres = self.verifier.verify(d3_prog, demonstrations)
                candidates_executed += 1

                if d3_vres.is_fully_solved:
                    elapsed = time.time() - start_time
                    return SearchResult(
                        task_id=task.task_id,
                        solved=True,
                        best_program=d3_prog,
                        candidate_programs=[d3_prog],
                        verification_results=[d3_vres],
                        candidates_generated=candidates_generated,
                        candidates_pruned=candidates_pruned,
                        candidates_executed=candidates_executed,
                        search_duration_sec=elapsed,
                        depth_reached=3,
                        diagnostics={"instant_solve_d3": d3_prog.name},
                    )

                evaluated_candidates.append((d3_prog, d3_vres))

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
            depth_reached=max_depth_reached,
            diagnostics={
                "top_score": score_candidate(top_candidates[0]) if top_candidates else 0.0,
                "top_accuracy": best_vres.mean_pixel_accuracy if best_vres else 0.0,
            },
        )
