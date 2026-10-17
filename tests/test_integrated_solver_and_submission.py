"""Unit and integration tests for Hunter Baseline, IntegratedArcSolver, and Kaggle Submission Adapter."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.representation.grid import ArcGrid
from src.data.dataset_loader import ArcTask, GridPair, ArcDatasetLoader
from src.solvers.hunter import HunterBaselineSolver
from src.solvers.integrated_solver import IntegratedArcSolver
from src.submission.adapter import SubmissionAdapter
from src.benchmark.harness import BenchmarkHarness


def test_hunter_baseline_solver():
    hunter = HunterBaselineSolver()

    # Identity task
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    id_task = ArcTask(
        task_id="test_id",
        train_pairs=(GridPair(arr, arr),),
        test_inputs=(arr,),
        dataset_split="training",
    )
    prog = hunter.solve(id_task)
    assert prog is not None
    assert prog.name == "identity"

    # Flip horizontal task
    arr_flip = np.array([[2, 1], [4, 3]], dtype=np.int32)
    flip_task = ArcTask(
        task_id="test_fliph",
        train_pairs=(GridPair(arr, arr_flip),),
        test_inputs=(arr,),
        dataset_split="training",
    )
    prog_flip = hunter.solve(flip_task)
    assert prog_flip is not None
    assert prog_flip.name == "flip_horizontal"


def test_integrated_solver_multi_test():
    solver = IntegratedArcSolver(search_timeout_seconds=1.0)

    # Multi-test task with 2 test inputs
    in1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    out1 = np.array([[2, 1], [4, 3]], dtype=np.int32)  # flip horizontal

    test1 = np.array([[5, 6], [7, 8]], dtype=np.int32)
    test2 = np.array([[1, 0], [0, 1]], dtype=np.int32)

    multi_task = ArcTask(
        task_id="multi_test_task",
        train_pairs=(GridPair(in1, out1),),
        test_inputs=(test1, test2),
        dataset_split="training",
    )

    solution = solver.solve(multi_task)
    assert solution.solved
    assert len(solution.test_predictions) == 2
    assert solution.test_predictions[0].test_index == 0
    assert solution.test_predictions[1].test_index == 1

    # Check attempt predictions
    pred1_np = solution.test_predictions[0].attempt_1.to_numpy()
    assert np.array_equal(pred1_np, np.array([[6, 5], [8, 7]], dtype=np.int32))


def test_submission_adapter_schema_validation():
    adapter = SubmissionAdapter()

    # Create dummy tasks
    task1 = ArcTask("task_001", (GridPair(np.zeros((2, 2), dtype=np.int32), np.zeros((2, 2), dtype=np.int32)),), (np.zeros((2, 2), dtype=np.int32),), "test")
    task2 = ArcTask("task_002", (GridPair(np.zeros((3, 3), dtype=np.int32), np.zeros((3, 3), dtype=np.int32)),), (np.zeros((3, 3), dtype=np.int32), np.zeros((3, 3), dtype=np.int32)), "test")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "submission.json"
        sub_dict = adapter.generate_submission([task1, task2], out_file, dry_run=True)

        assert out_file.exists()
        assert "task_001" in sub_dict
        assert "task_002" in sub_dict
        assert len(sub_dict["task_001"]) == 1
        assert len(sub_dict["task_002"]) == 2
        for entry in sub_dict["task_001"]:
            assert "attempt_1" in entry
            assert "attempt_2" in entry


def test_benchmark_harness_scoring():
    solver = IntegratedArcSolver(search_timeout_seconds=0.5)
    harness = BenchmarkHarness(solver)

    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    task = ArcTask("bench_task", (GridPair(arr, arr),), (arr,), "training")
    solutions = {"bench_task": [arr]}

    report = harness.run_benchmark([task], solutions=solutions, split_name="synthetic_test")
    assert report.total_tasks_attempted == 1
    assert report.tasks_solved == 1
    assert report.task_level_accuracy == 1.0
    assert report.test_input_level_accuracy == 1.0
