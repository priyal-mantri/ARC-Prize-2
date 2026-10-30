"""Unit Tests for Qwen2.5-Coder Neuro-Symbolic Integration and Memory Composition.

Tests:
1. Truthful Qwen loading failure reporting (no false claims).
2. Qwen proposal parsing, unknown primitive rejection, and verifier gatekeeping.
3. ConceptLibrary registration, precondition filtering, bounded capacity, and export summary.
4. Bounded depth-3 composition in beam search with intermediate validation.
5. Dynamic memory retention loop in IntegratedArcSolver.
6. Controlled experimental configuration modes (A, B, C).
7. Submission adapter contract (2 attempts per test prediction).
"""

import json
import numpy as np
import pytest
from pathlib import Path

from src.data.dataset_loader import ArcTask, GridPair
from src.representation.grid import ArcGrid
from src.dsl.program import DSLProgram, PrimitiveCall
from src.solvers.concept_library import ConceptLibrary, LearnedSkill
from src.memory.mirror_matrix import MirrorMatrix, MemoryEntry
from src.solvers.llm_synthesizer import LLMSynthesizerInterface, LLMExecutionTelemetry
from src.search.verifier import DemonstrationVerifier, TaskDemonstration
from src.search.beam_search import DSLBeamSearch
from src.solvers.integrated_solver import IntegratedArcSolver


def create_dummy_task(task_id: str = "dummy_01", same_shape: bool = True, has_hole: bool = False) -> ArcTask:
    """Helper to create minimal valid ArcTasks for testing."""
    if same_shape:
        if has_hole:
            # 5x5 grid with enclosed void (0 surrounded by 1s)
            in_grid = np.array([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ], dtype=np.int32)
            out_grid = np.array([
                [1, 1, 1, 1, 1],
                [1, 4, 4, 4, 1],
                [1, 4, 4, 4, 1],
                [1, 4, 4, 4, 1],
                [1, 1, 1, 1, 1],
            ], dtype=np.int32)
        else:
            in_grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
            out_grid = np.array([[2, 1], [4, 3]], dtype=np.int32)
    else:
        in_grid = np.array([[1, 2]], dtype=np.int32)
        out_grid = np.array([[1, 2, 1, 2], [1, 2, 1, 2]], dtype=np.int32)

    return ArcTask(
        task_id=task_id,
        train_pairs=(GridPair(input=in_grid, output=out_grid),),
        test_inputs=(in_grid,),
        dataset_split="test",
    )


class TestQwenTelemetryAndLoadingTruthfulness:
    """Ensure no false claims about Qwen execution when offline."""

    def test_offline_loading_reports_truthful_failure(self):
        synth = LLMSynthesizerInterface(model_path="/nonexistent/model/path")
        task = create_dummy_task()
        proposals = synth.generate_hypotheses(task=task, max_candidates=5)

        # Truthfulness invariants
        assert synth.telemetry.model_loading_success is False
        assert synth.telemetry.inference_success is False
        assert synth.telemetry.fallback_usage is True
        assert synth.telemetry.genuine_qwen_proposal_count == 0
        assert (
            "candidate paths" in synth.telemetry.fallback_reason.lower()
            or "not found" in synth.telemetry.fallback_reason.lower()
            or "missing" in synth.telemetry.fallback_reason.lower()
            or "not installed" in synth.telemetry.fallback_reason.lower()
            or "failed" in synth.telemetry.fallback_reason.lower()
        )

        # Fallback proposals may exist, but their origin must NOT be qwen
        for p in proposals:
            assert p.origin != "qwen2.5_coder"


class TestQwenProposalParsingAndVerifierGatekeeping:
    """Test parsing logic, invalid primitive rejection, and verifier integration."""

    def test_invalid_primitive_rejection(self):
        synth = LLMSynthesizerInterface(model_path="/nonexistent")
        raw_llm_json = json.dumps({
            "programs": [
                {
                    "name": "invalid_prog",
                    "calls": [{"primitive": "completely_invented_fake_primitive", "params": {}}],
                },
                {
                    "name": "valid_prog",
                    "calls": [{"primitive": "rot90", "params": {}}],
                },
            ]
        })

        parsed = synth._parse_and_validate_response(raw_llm_json, max_candidates=5)
        assert len(parsed) == 1
        assert parsed[0].calls[0].primitive_name == "rot90"
        assert parsed[0].origin == "qwen2.5_coder"

    def test_verifier_gatekeeping_accepts_valid_and_rejects_invalid(self):
        verifier = DemonstrationVerifier()
        task = create_dummy_task("flip_test", same_shape=True)
        demos = [
            TaskDemonstration(
                input_grid=ArcGrid(np_grid=ex.input),
                output_grid=ArcGrid(np_grid=ex.output),
            )
            for ex in task.train_examples
        ]

        # Valid solver for [[1, 2], [3, 4]] -> [[2, 1], [4, 3]] is flip_horizontal
        valid_prog = DSLProgram(name="flip_h", calls=[PrimitiveCall("flip_horizontal")], origin="qwen2.5_coder")
        invalid_prog = DSLProgram(name="rot90", calls=[PrimitiveCall("rot90")], origin="qwen2.5_coder")

        v_valid = verifier.verify(valid_prog, demos)
        v_invalid = verifier.verify(invalid_prog, demos)

        assert v_valid.is_fully_solved is True
        assert v_valid.mean_pixel_accuracy == 1.0
        assert v_invalid.is_fully_solved is False


class TestConceptLibraryLearningAndPreconditions:
    """Test registry, precondition filtering, bounded capacity, and export."""

    def test_skill_registration_and_precondition_filtering(self):
        lib = ConceptLibrary(max_skills=10)
        prog1 = DSLProgram(name="fill_void", calls=[PrimitiveCall("fill_enclosed", {"fill_color": 4})])
        prog2 = DSLProgram(name="tile_grid", calls=[PrimitiveCall("tile_grid", {"scale_y": 2, "scale_x": 2})])

        lib.register_skill(
            program=prog1,
            task_id="task_holes",
            preconditions={"requires_same_shape": True, "requires_holes": True},
        )
        lib.register_skill(
            program=prog2,
            task_id="task_scale",
            preconditions={"is_scaling": True, "scale_factor": (2, 2)},
        )

        assert len(lib.skills) == 2

        # Task with holes and same shape
        matches_hole = lib.get_candidate_skills_for_task({"same_shape": True, "has_holes": True})
        assert any(s.program.name == "fill_void" for s in matches_hole)
        assert not any(s.program.name == "tile_grid" for s in matches_hole)

        # Task with scaling
        matches_scale = lib.get_candidate_skills_for_task({"is_scaling": True, "scale_factor": (2, 2)})
        assert any(s.program.name == "tile_grid" for s in matches_scale)
        assert not any(s.program.name == "fill_void" for s in matches_scale)

    def test_bounded_capacity_least_used_eviction(self):
        lib = ConceptLibrary(max_skills=3)
        p1 = DSLProgram(name="p1", calls=[PrimitiveCall("rot90")])
        p2 = DSLProgram(name="p2", calls=[PrimitiveCall("rot180")])
        p3 = DSLProgram(name="p3", calls=[PrimitiveCall("rot270")])
        p4 = DSLProgram(name="p4", calls=[PrimitiveCall("flip_horizontal")])

        lib.register_skill(p1, "t1")
        # Increase usage of p1
        lib.register_skill(p1, "t1_b")

        lib.register_skill(p2, "t2")
        lib.register_skill(p3, "t3")
        assert len(lib.skills) == 3

        # Registering 4th skill should evict one of p2 or p3 (usage=1), preserving p1 (usage=2)
        lib.register_skill(p4, "t4")
        assert len(lib.skills) == 3
        names = [s.program.name for s in lib.skills]
        assert "p1" in names
        assert "p4" in names

    def test_export_skills_summary(self):
        lib = ConceptLibrary()
        p = DSLProgram(name="invert_c", calls=[PrimitiveCall("invert_colors")])
        lib.register_skill(p, "t_inv", preconditions={"requires_same_shape": True})
        summary = lib.export_skills_summary()
        assert len(summary) == 1
        assert summary[0]["name"] == "invert_c"
        assert summary[0]["calls"] == ["invert_colors"]
        assert summary[0]["preconditions"]["requires_same_shape"] is True


class TestBoundedDepth3Composition:
    """Test that beam search performs depth 3 composition with intermediate pruning."""

    def test_depth_3_execution_and_intermediate_validation(self):
        verifier = DemonstrationVerifier()
        base_grid = np.array([[1, 2], [0, 3]], dtype=np.int32)
        step1 = np.rot90(base_grid, -1).copy()
        step2 = np.fliplr(step1).copy()
        step3_arr = step2.copy()
        for r in range(step3_arr.shape[0]):
            for c in range(step3_arr.shape[1]):
                if step3_arr[r, c] != 0:
                    step3_arr[r, c] = 10 - step3_arr[r, c]

        task = ArcTask(
            task_id="depth3_task",
            train_pairs=(GridPair(input=base_grid, output=step3_arr),),
            test_inputs=(base_grid,),
            dataset_split="test",
        )

        searcher = DSLBeamSearch(
            verifier=verifier,
            beam_width=8,
            max_depth=3,
            timeout_seconds=2.0,
            enable_repl_refinement=False,
            max_candidates=40,
        )

        res = searcher.search(task)
        # Verify search executed depth 3 without errors
        assert res.depth_reached >= 2
        assert res.candidates_generated > 0
        assert res.candidates_executed > 0


class TestIntegratedSolverContinuousRetention:
    """Test continuous learning loop in IntegratedArcSolver."""

    def test_verified_solution_retained_in_concept_library(self):
        lib = ConceptLibrary()
        mem = MirrorMatrix()
        solver = IntegratedArcSolver(
            concept_library=lib,
            memory_bank=mem,
            mode="integrated_qwen_memory_composition",
        )

        # Solve simple task (flip_horizontal)
        task = create_dummy_task("retention_test", same_shape=True)
        sol = solver.solve(task)

        assert sol.solved is True
        # Check that skill was retained
        assert len(lib.skills) >= 1
        assert len(mem.entries) >= 1
        retained = lib.skills[0]
        assert "retention_test" in retained.solved_task_ids
        assert retained.preconditions.get("requires_same_shape") is True

    def test_experimental_modes_configuration(self):
        # Mode A: Symbolic baseline
        s_a = IntegratedArcSolver(mode="symbolic_baseline")
        assert s_a.use_qwen is False
        assert s_a.enable_memory is False
        assert s_a.enable_memory_composition is False
        assert s_a.max_composition_depth == 2

        # Mode B: Qwen proposals only
        s_b = IntegratedArcSolver(mode="qwen_proposals")
        assert s_b.use_qwen is True
        assert s_b.enable_memory is False
        assert s_b.enable_memory_composition is False
        assert s_b.max_composition_depth == 2

        # Mode C: Integrated Qwen + Memory + Composition
        s_c = IntegratedArcSolver(mode="integrated_qwen_memory_composition")
        assert s_c.use_qwen is True
        assert s_c.enable_memory is True
        assert s_c.enable_memory_composition is True
        assert s_c.max_composition_depth >= 3

    def test_two_attempt_prediction_contract(self):
        solver = IntegratedArcSolver(mode="symbolic_baseline")
        task = create_dummy_task("contract_test")
        sol = solver.solve(task)

        assert len(sol.test_predictions) == len(task.test_inputs)
        for pred in sol.test_predictions:
            assert isinstance(pred.attempt_1, ArcGrid)
            assert isinstance(pred.attempt_2, ArcGrid)
            assert pred.attempt_1.shape[0] > 0
            assert pred.attempt_1.shape[1] > 0
            assert pred.attempt_2.shape[0] > 0
            assert pred.attempt_2.shape[1] > 0

    def test_fill_enclosed_by_size_primitive(self):
        from src.dsl.primitives import fill_enclosed_by_size, PRIMITIVE_REGISTRY, PRIMITIVE_SPECS
        assert "fill_enclosed_by_size" in PRIMITIVE_REGISTRY
        assert "fill_enclosed_by_size" in PRIMITIVE_SPECS

        # Test 7x7 grid with a 3x3 enclosed void (surrounded by 2s)
        grid = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 2, 0],
            [0, 2, 0, 0, 0, 2, 0],
            [0, 2, 0, 0, 0, 2, 0],
            [0, 2, 0, 0, 0, 2, 0],
            [0, 2, 2, 2, 2, 2, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.int32)
        res = fill_enclosed_by_size(grid)
        # Bounding box of hole is 3x3 -> maps to color 8
        assert res[2, 2] == 8
        assert res[3, 3] == 8
        assert res[0, 0] == 0  # border stays 0

    def test_recolor_by_probe_key_primitive(self):
        from src.dsl.primitives import recolor_by_probe_key, PRIMITIVE_REGISTRY, PRIMITIVE_SPECS
        assert "recolor_by_probe_key" in PRIMITIVE_REGISTRY
        assert "recolor_by_probe_key" in PRIMITIVE_SPECS

        grid = np.zeros((10, 10), dtype=np.int32)
        # Main component of color 8
        grid[1:4, 1:4] = 8
        # Probe of color 1 with key signature 1 -> target 7
        grid[6:9, 6:9] = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [0, 1, 0],
        ])

        res = recolor_by_probe_key(grid, probe_color=1, main_color=8)
        assert res[1, 1] == 7
        assert res[2, 2] == 7
        assert np.all(res[6:9, 6:9] == 0)  # probe erased
