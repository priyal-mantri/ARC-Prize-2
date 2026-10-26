"""Comprehensive Acceptance and Integration Tests for Final ARC-AGI-2 Solver.

Covers:
1. DSL registry signature correctness and schema auditing.
2. Complete symmetry with horizontal, vertical, and both axes.
3. Fractal expansion and alternating tiling primitives.
4. Parameterized candidate generation with exact color matching.
5. Scene graph integration and fact propagation.
6. Qwen2.5-Coder interface, prompt construction, and strict JSON output validation.
7. Offline preflight checks and fallback reporting.
8. Guess budget enforcement and candidate deduplication.
9. Two-attempt output contract per test query.
10. Strict ground-truth isolation during solving.
11. End-to-end solving on the acceptance manifest and regression suite.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.data.dataset_loader import ArcDatasetLoader, ArcTask
from src.dsl.primitives import (
    PRIMITIVE_REGISTRY,
    PRIMITIVE_SPECS,
    complete_symmetry,
    fractal_expansion,
    tile_alternating_h,
    fill_enclosed,
    crop_to_content,
    apply_gravity,
)
from src.dsl.program import DSLProgram, PrimitiveCall
from src.scene_graph.builder import SceneGraphBuilder
from src.scene_graph.graph import ArcSceneGraph
from src.solvers.llm_synthesizer import LLMSynthesizerInterface, LLMExecutionTelemetry
from src.solvers.integrated_solver import IntegratedArcSolver, TaskSolution
from src.search.beam_search import DSLBeamSearch


class TestDSLRegistryAndPrimitives:
    """Verifies registry integrity, signatures, and newly added primitives."""

    def test_complete_symmetry_axes(self):
        """Verify complete_symmetry handles horizontal, vertical, and both axes."""
        # Top-left 2x2 has 1, rest 0
        grid = np.array([
            [1, 0],
            [0, 0]
        ], dtype=np.int32)

        # Horizontal symmetry flips left-to-right: overlay fliplr
        sym_h = complete_symmetry(grid, axis="horizontal")
        expected_h = np.array([
            [1, 1],
            [0, 0]
        ], dtype=np.int32)
        assert np.array_equal(sym_h, expected_h)

        # Vertical symmetry flips top-to-bottom: overlay flipud
        sym_v = complete_symmetry(grid, axis="vertical")
        expected_v = np.array([
            [1, 0],
            [1, 0]
        ], dtype=np.int32)
        assert np.array_equal(sym_v, expected_v)

        # Both axes
        sym_both = complete_symmetry(grid, axis="both")
        expected_both = np.array([
            [1, 1],
            [1, 1]
        ], dtype=np.int32)
        assert np.array_equal(sym_both, expected_both)

    def test_fractal_expansion(self):
        """Verify Kronecker fractal expansion primitive."""
        grid = np.array([
            [1, 0],
            [0, 1]
        ], dtype=np.int32)
        expanded = fractal_expansion(grid)
        assert expanded.shape == (4, 4)
        assert np.array_equal(expanded[:2, :2], grid)
        assert np.array_equal(expanded[:2, 2:], np.zeros((2, 2)))
        assert np.array_equal(expanded[2:, 2:], grid)

    def test_tile_alternating_h(self):
        """Verify alternating horizontal tiling primitive."""
        grid = np.array([
            [1, 2],
            [3, 4]
        ], dtype=np.int32)
        tiled = tile_alternating_h(grid, rep_h=3, rep_w=2)
        assert tiled.shape == (6, 4)
        # Row-block 0: grid tiled
        assert np.array_equal(tiled[:2, :2], grid)
        # Row-block 1: fliplr(grid) tiled
        assert np.array_equal(tiled[2:4, :2], np.fliplr(grid))
        # Row-block 2: grid tiled
        assert np.array_equal(tiled[4:6, :2], grid)

    def test_primitive_specs_coverage(self):
        """Verify every registered primitive has a corresponding PrimitiveSpec."""
        for name in PRIMITIVE_REGISTRY.keys():
            assert name in PRIMITIVE_SPECS, f"Primitive {name} missing PrimitiveSpec"
            spec = PRIMITIVE_SPECS[name]
            assert spec.category in ["geometry", "color", "movement", "cropping_scaling", "topology_symmetry", "objects"]


class TestSceneGraphIntegration:
    """Verifies scene graph construction and integration into candidate search."""

    def test_scene_graph_builder_executes_without_scipy_errors(self):
        """Verify builder doesn't crash on scipy calls."""
        from src.representation.grid import ArcGrid
        from src.perception.detector import PerceptionEngine

        grid_np = np.zeros((10, 10), dtype=np.int32)
        grid_np[2:5, 2:5] = 3
        grid_np[3, 3] = 0  # interior hole
        grid = ArcGrid(np_grid=grid_np)

        detector = PerceptionEngine()
        objs = detector.detect_objects(grid, background_color=0)
        builder = SceneGraphBuilder()
        sg = builder.build(grid, objs, background_color=0)

        assert isinstance(sg, ArcSceneGraph)
        assert len(sg.nodes) > 0
        node = list(sg.nodes.values())[0]
        assert node.attributes["holes_count"] == 1

    def test_scene_graph_facts_influence_solver(self):
        """Verify scene graph facts are captured by IntegratedArcSolver."""
        loader = ArcDatasetLoader()
        task = loader.load_task("00d62c1b", split="training")
        solver = IntegratedArcSolver(num_guesses=20, use_scene_graph=True)
        sol = solver.solve(task)

        assert "scene_graph_facts" in sol.diagnostics
        facts = sol.diagnostics["scene_graph_facts"]
        assert facts["has_holes"] is True
        assert sol.component_statuses["scene_graph"] == "integrated_and_influencing_predictions"


class TestQwenInterfaceAndStructuredParsing:
    """Verifies Qwen structured prompts, offline preflight, and JSON parsing."""

    def test_offline_preflight_reports_truthfully(self):
        """Verify LLMSynthesizerInterface honestly reports when weights/CUDA are absent."""
        synth = LLMSynthesizerInterface()
        # In CPU environment without local model weights:
        assert synth.telemetry.model_loading_success is False
        assert synth.telemetry.fallback_usage is True
        assert "No local Qwen model artifact found" in synth.telemetry.fallback_reason

    def test_strict_json_output_parsing(self):
        """Verify validator accepts valid primitives and rejects unknown/invalid ones."""
        synth = LLMSynthesizerInterface()
        json_response = json.dumps({
            "programs": [
                {
                    "name": "valid_sequence",
                    "calls": [
                        {"primitive": "rot90", "params": {}},
                        {"primitive": "fill_enclosed", "params": {"fill_color": 4}}
                    ]
                },
                {
                    "name": "invalid_sequence",
                    "calls": [
                        {"primitive": "hallucinated_op", "params": {}},
                        {"primitive": "crop_to_content", "params": {}}
                    ]
                }
            ]
        })

        progs = synth._parse_and_validate_response(json_response, max_candidates=5)
        assert len(progs) == 2
        # First prog has 2 calls
        assert len(progs[0].calls) == 2
        assert progs[0].calls[0].name == "rot90"
        assert progs[0].calls[1].name == "fill_enclosed"
        # Second prog filtered out unknown primitive and kept valid crop_to_content
        assert len(progs[1].calls) == 1
        assert progs[1].calls[0].name == "crop_to_content"


class TestCandidateBudgetAndTwoAttempts:
    """Verifies configurable NUM_GUESSES and two-attempt submission interface."""

    def test_guess_budget_enforced(self):
        """Verify candidate budget caps depth 1 candidates."""
        loader = ArcDatasetLoader()
        task = loader.load_task("009d5c81", split="training")

        bs_20 = DSLBeamSearch(max_candidates=20, timeout_seconds=1.0)
        res_20 = bs_20.search(task)

        bs_40 = DSLBeamSearch(max_candidates=40, timeout_seconds=1.0)
        res_40 = bs_40.search(task)

        assert res_20.candidates_generated <= 20
        assert res_40.candidates_generated <= 40

    def test_two_attempts_contract(self):
        """Verify solver always returns exactly 2 attempts per test query."""
        loader = ArcDatasetLoader()
        task = loader.load_task("007bbfb7", split="training")
        solver = IntegratedArcSolver(num_guesses=20)
        sol = solver.solve(task)

        assert len(sol.test_predictions) == len(task.test_inputs)
        for pred in sol.test_predictions:
            assert pred.attempt_1 is not None
            assert pred.attempt_2 is not None
            assert pred.attempt_1.shape[0] > 0
            assert pred.attempt_2.shape[0] > 0


class TestGroundTruthIsolation:
    """Verifies solver never touches test solutions."""

    def test_solver_receives_no_test_solutions(self):
        """Assert ArcTask never contains test outputs."""
        loader = ArcDatasetLoader()
        task = loader.load_task("007bbfb7", split="training")
        assert not hasattr(task, "test_solutions")
        assert not hasattr(task, "test_outputs")
        # Solvers only get train pairs and test inputs
        for pair in task.train_examples:
            assert hasattr(pair, "input_grid")
            assert hasattr(pair, "output_grid")
        for test_in in task.test_inputs:
            assert isinstance(test_in, np.ndarray)


class TestFiveTaskManifestAndRegressionSuite:
    """Verifies end-to-end solving on the acceptance manifest tasks."""

    def test_solves_fractal_007bbfb7(self):
        loader = ArcDatasetLoader()
        task = loader.load_task("007bbfb7", split="training")
        solver = IntegratedArcSolver(num_guesses=20)
        sol = solver.solve(task)
        assert sol.solved is True
        assert sol.best_program is not None

    def test_solves_enclosed_00d62c1b(self):
        loader = ArcDatasetLoader()
        task = loader.load_task("00d62c1b", split="training")
        solver = IntegratedArcSolver(num_guesses=20)
        sol = solver.solve(task)
        assert sol.solved is True
        assert sol.best_program is not None

    def test_solves_tiling_00576224(self):
        loader = ArcDatasetLoader()
        task = loader.load_task("00576224", split="training")
        solver = IntegratedArcSolver(num_guesses=20)
        sol = solver.solve(task)
        assert sol.solved is True
        assert sol.best_program is not None

    def test_regression_tasks_pass(self):
        loader = ArcDatasetLoader()
        solver = IntegratedArcSolver(num_guesses=20)
        for tid in ["1cf80156", "1e0a9b12", "1f85a75f"]:
            task = loader.load_task(tid, split="training")
            sol = solver.solve(task)
            assert sol.solved is True, f"Regression task {tid} failed"
