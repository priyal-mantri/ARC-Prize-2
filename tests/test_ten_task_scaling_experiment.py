"""Unit tests for the 10-Task Scaling Experiment and Protocol.

Tests manifest integrity, candidate budget propagation and enforcement,
evaluation isolation, determinism, Qwen telemetry reporting, timing fields,
and baseline artifact preservation.
"""

import json
import hashlib
from pathlib import Path
import numpy as np
import pytest

from src.data.dataset_loader import ArcDatasetLoader, ArcTask
from src.solvers.integrated_solver import IntegratedArcSolver
from src.solvers.llm_synthesizer import LLMSynthesizerInterface

REPO_ROOT = Path(__file__).resolve().parent.parent

# Checksums recorded at Phase 1 start
BASELINE_CHECKSUMS = {
    "data/five_task_manifest.json": "e38bbd5de244e1f73c9c6baacdbcc43fe765993e130c41776c0f0a6429bee201",
    "reports/five_task_experiment_results.json": "d54b879c283aea532aa601547829af636e41064c44d9e7901f6dbd2d2cbb17ef",
    "reports/final_integrated_architecture_report.md": "52453e9fd56f17a82c03893818ce14c552524796f9a4c531a77773ac9ce55016",
}


def test_ten_task_manifest_uniqueness_and_completeness():
    manifest_path = REPO_ROOT / "data" / "ten_task_manifest.json"
    assert manifest_path.exists(), "Manifest file missing"
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = data.get("tasks", [])
    assert len(tasks) == 10, f"Expected exactly 10 tasks, found {len(tasks)}"

    task_ids = [t["task_id"] for t in tasks]
    assert len(set(task_ids)) == 10, f"Duplicate task IDs found: {task_ids}"

    # Verify the 5 baseline tasks are retained
    baseline_ids = {"007bbfb7", "00d62c1b", "00576224", "009d5c81", "00dbd492"}
    assert baseline_ids.issubset(set(task_ids)), "Baseline tasks missing from 10-task manifest"

    # Verify all 10 tasks exist in raw training dataset
    raw_path = REPO_ROOT / "data" / "raw" / "arc-agi_training_challenges.json"
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_challenges = json.load(f)
    for tid in task_ids:
        assert tid in raw_challenges, f"Task {tid} not found in raw training dataset"


def test_candidate_budgets_reach_search_implementation():
    for budget in [20, 40, 60]:
        solver = IntegratedArcSolver(num_guesses=budget)
        assert solver.num_guesses == budget, f"Solver num_guesses mismatch: {solver.num_guesses} vs {budget}"
        assert solver.beam_search.max_candidates == budget, f"BeamSearch max_candidates mismatch: {solver.beam_search.max_candidates} vs {budget}"


def test_candidate_limits_strictly_enforced():
    loader = ArcDatasetLoader()
    task = loader.load_task("007bbfb7")
    for budget in [20, 40]:
        solver = IntegratedArcSolver(num_guesses=budget)
        sol = solver.solve(task)
        # Unique candidates ranked must never exceed the budget
        assert len(sol.candidates_ranked) <= budget, f"Ranked candidates {len(sol.candidates_ranked)} exceeded budget {budget}"


def test_strict_evaluation_isolation_contract():
    loader = ArcDatasetLoader()
    task = loader.load_task("007bbfb7")

    # ArcTask must NOT have solutions attribute or test outputs
    assert not hasattr(task, "test_solutions"), "Data leakage: ArcTask contains test_solutions"
    assert not hasattr(task, "solutions"), "Data leakage: ArcTask contains solutions"
    assert isinstance(task.test_inputs, tuple)
    for t_in in task.test_inputs:
        assert isinstance(t_in, np.ndarray)


def test_reproducibility_across_runs():
    loader = ArcDatasetLoader()
    task = loader.load_task("007bbfb7")
    solver1 = IntegratedArcSolver(num_guesses=20)
    sol1 = solver1.solve(task)

    solver2 = IntegratedArcSolver(num_guesses=20)
    sol2 = solver2.solve(task)

    assert sol1.solved == sol2.solved == True
    assert sol1.best_program is not None and sol2.best_program is not None
    assert sol1.best_program.to_code() == sol2.best_program.to_code()
    assert np.array_equal(sol1.test_predictions[0].attempt_1.to_numpy(), sol2.test_predictions[0].attempt_1.to_numpy())


def test_qwen_status_truthfully_reported():
    llm = LLMSynthesizerInterface()
    telemetry = llm.telemetry
    # In local environment without weights, model_loading_success must be False
    assert telemetry.model_loading_success is False, "Qwen falsely reported as loaded"
    assert telemetry.fallback_usage is True, "Fallback usage must be True when model is unavailable"
    assert "No local Qwen model artifact found" in telemetry.fallback_reason


def test_timing_fields_populated_with_real_measurements():
    loader = ArcDatasetLoader()
    task = loader.load_task("3c9b0459")
    solver = IntegratedArcSolver(num_guesses=20)
    sol = solver.solve(task)

    assert "total_duration" in sol.profiling
    assert sol.profiling["total_duration"] > 0.0
    assert sol.execution_time_sec > 0.0


def test_five_task_baseline_artifacts_preserved():
    for rel_path, expected_sha in BASELINE_CHECKSUMS.items():
        full_path = REPO_ROOT / rel_path
        assert full_path.exists(), f"Baseline artifact missing: {rel_path}"
        with open(full_path, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()
        assert actual_sha == expected_sha, f"Baseline artifact modified! {rel_path} expected {expected_sha}, got {actual_sha}"
