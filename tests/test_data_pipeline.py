"""
Automated unit and integration tests for the ARC-AGI-2 data pipeline.

Verifies:
1. File discovery and path resolution.
2. Loading known ARC tasks from the training and evaluation splits.
3. Shape and color integrity (no truncation, correct numpy dtypes).
4. Strict isolation of evaluation ground-truth from ArcTask instances.
5. Detection and reporting of malformed grid structures.
6. Exact dataset task counts matching actual files.
7. Determinism and consistency of computed statistics.
"""

import unittest
from pathlib import Path
import numpy as np

from src.data.dataset_loader import (
    ArcDatasetLoader,
    ArcTask,
    GridPair,
    discover_dataset_files,
)
from src.data.validation import (
    validate_dataset,
    validate_grid,
    validate_task,
)
from src.data.statistics import (
    check_task_overlaps,
    compute_dataset_statistics,
)


class TestArcDataPipeline(unittest.TestCase):
    """Test suite for ARC data loader, validation, and statistics."""

    @classmethod
    def setUpClass(cls):
        """Set up shared loader and discover dataset files."""
        cls.loader = ArcDatasetLoader()
        cls.discovered_files = cls.loader.files

    def test_file_discovery(self):
        """Verifies that all 6 canonical dataset files are discovered."""
        expected_roles = [
            "training_challenges",
            "training_solutions",
            "evaluation_challenges",
            "evaluation_solutions",
            "test_challenges",
            "sample_submission",
        ]
        for role in expected_roles:
            self.assertIn(role, self.discovered_files, f"Missing discovered file for role: {role}")
            path = self.discovered_files[role]
            self.assertTrue(path.exists(), f"Path for role {role} does not exist: {path}")

    def test_dataset_task_counts(self):
        """Verifies exact task counts match the official competition dataset."""
        train_ids = self.loader.get_task_ids("training")
        eval_ids = self.loader.get_task_ids("evaluation")
        test_ids = self.loader.get_task_ids("test")

        self.assertEqual(len(train_ids), 1000, f"Expected 1000 training tasks, found {len(train_ids)}")
        self.assertEqual(len(eval_ids), 120, f"Expected 120 evaluation tasks, found {len(eval_ids)}")
        self.assertEqual(len(test_ids), 240, f"Expected 240 test tasks, found {len(test_ids)}")

    def test_task_id_uniqueness_and_overlaps(self):
        """Verifies task ID uniqueness and disjointness between train and eval."""
        train_set = set(self.loader.get_task_ids("training"))
        eval_set = set(self.loader.get_task_ids("evaluation"))
        test_set = set(self.loader.get_task_ids("test"))

        # Train and Eval must be completely disjoint
        train_eval_overlap = train_set.intersection(eval_set)
        self.assertEqual(
            len(train_eval_overlap), 0,
            f"Training and evaluation sets MUST be disjoint. Found overlap: {train_eval_overlap}"
        )

        # Test set in public offline distribution is a 240-task subset of training
        test_train_overlap = test_set.intersection(train_set)
        self.assertEqual(
            len(test_train_overlap), 240,
            f"Public offline test set should overlap 240 tasks with training set. Found {len(test_train_overlap)}"
        )

    def test_load_known_task(self):
        """Verifies loading a specific known task parses train pairs and test inputs correctly."""
        known_id = "00576224"
        task = self.loader.load_task(known_id, split="training")

        self.assertEqual(task.task_id, known_id)
        self.assertEqual(task.dataset_split, "training")
        self.assertGreater(task.num_train, 0)
        self.assertGreater(task.num_test, 0)

        # Inspect first train pair
        pair0 = task.train_pairs[0]
        self.assertIsInstance(pair0.input, np.ndarray)
        self.assertIsInstance(pair0.output, np.ndarray)
        self.assertEqual(pair0.input.ndim, 2)
        self.assertEqual(pair0.output.ndim, 2)

        # Verify grid cell values are canonical integers in [0, 9]
        self.assertTrue(np.all((pair0.input >= 0) & (pair0.input <= 9)))
        self.assertTrue(np.all((pair0.output >= 0) & (pair0.output <= 9)))

    def test_safety_ground_truth_isolation(self):
        """CRITICAL: Verifies that ArcTask objects NEVER expose test solutions to solvers."""
        task = self.loader.load_task("00576224", split="training")
        self.assertFalse(hasattr(task, "test_outputs"))
        self.assertFalse(hasattr(task, "solutions"))

        # Verify that ground truth can ONLY be retrieved via the explicit accessor
        ground_truth = self.loader.load_ground_truth("00576224", split="training")
        self.assertIsInstance(ground_truth, list)
        self.assertEqual(len(ground_truth), task.num_test)
        self.assertIsInstance(ground_truth[0], np.ndarray)

    def test_multi_test_tasks(self):
        """Verifies that multi-test tasks are handled and identified correctly."""
        # Task 8dab14c2 has 4 test inputs in training set
        task_id = "8dab14c2"
        task = self.loader.load_task(task_id, split="training")
        self.assertTrue(task.is_multi_test)
        self.assertEqual(task.num_test, 4)

        solutions = self.loader.load_ground_truth(task_id, split="training")
        self.assertEqual(len(solutions), 4)

    def test_grid_validation_rules(self):
        """Tests that invalid, ragged, or out-of-range grids are detected cleanly."""
        # Valid grid
        valid_g = [[0, 1, 2], [3, 4, 5]]
        ok, errs = validate_grid(valid_g)
        self.assertTrue(ok)
        self.assertEqual(len(errs), 0)

        # Ragged/non-rectangular grid
        ragged_g = [[0, 1, 2], [3, 4]]
        ok, errs = validate_grid(ragged_g)
        self.assertFalse(ok)
        self.assertTrue(any("non-rectangular" in e for e in errs))

        # Out of bounds color (> 9)
        invalid_color_g = [[0, 10], [2, 3]]
        ok, errs = validate_grid(invalid_color_g)
        self.assertFalse(ok)
        self.assertTrue(any("invalid color" in e for e in errs))

        # Negative color (< 0)
        neg_color_g = [[-1, 1], [2, 3]]
        ok, errs = validate_grid(neg_color_g)
        self.assertFalse(ok)
        self.assertTrue(any("invalid color" in e for e in errs))

        # Empty grid
        empty_g = []
        ok, errs = validate_grid(empty_g)
        self.assertFalse(ok)
        self.assertTrue(any("height 0" in e for e in errs))

    def test_dataset_validation_cleanliness(self):
        """Verifies that the entire official training and evaluation datasets pass validation with 0 errors."""
        train_challenges = self.loader.get_raw_challenges("training")
        train_solutions = self.loader.get_raw_solutions("training")
        train_report = validate_dataset(train_challenges, train_solutions, "Training Set")
        self.assertTrue(train_report.is_valid, f"Training validation failed: {train_report.errors[:5]}")
        self.assertEqual(train_report.malformed_tasks, 0)
        self.assertEqual(train_report.valid_tasks, 1000)

        eval_challenges = self.loader.get_raw_challenges("evaluation")
        eval_solutions = self.loader.get_raw_solutions("evaluation")
        eval_report = validate_dataset(eval_challenges, eval_solutions, "Evaluation Set")
        self.assertTrue(eval_report.is_valid, f"Evaluation validation failed: {eval_report.errors[:5]}")
        self.assertEqual(eval_report.malformed_tasks, 0)
        self.assertEqual(eval_report.valid_tasks, 120)

    def test_statistics_determinism(self):
        """Verifies that statistical calculation is deterministic across repeated runs."""
        train_challenges = self.loader.get_raw_challenges("training")
        train_solutions = self.loader.get_raw_solutions("training")

        stats1 = compute_dataset_statistics(train_challenges, train_solutions, "Run 1")
        stats2 = compute_dataset_statistics(train_challenges, train_solutions, "Run 2")

        self.assertEqual(stats1.total_tasks, stats2.total_tasks)
        self.assertEqual(stats1.total_train_examples, stats2.total_train_examples)
        self.assertEqual(stats1.total_test_inputs, stats2.total_test_inputs)
        self.assertEqual(stats1.total_solutions, stats2.total_solutions)
        self.assertEqual(stats1.color_cell_counts, stats2.color_cell_counts)
        self.assertEqual(stats1.dim_same_pairs, stats2.dim_same_pairs)


if __name__ == "__main__":
    unittest.main()
