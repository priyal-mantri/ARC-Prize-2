"""
ARC-AGI-2 Dataset Loader and Discovery Utilities.

Provides a clean, modular, and safe data access layer for ARC-AGI-2 tasks:
- Auto-discovers local challenge and solution files.
- Separates solver-facing task representations (inputs only for test) from evaluation ground-truth.
- Validates task presence and supports both training and evaluation splits.
- Preserves original task IDs and raw grid representations.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class GridPair:
    """An input/output example pair from an ARC task's training demonstration set."""
    input: np.ndarray
    output: np.ndarray

    @property
    def input_grid(self) -> np.ndarray:
        return self.input

    @property
    def output_grid(self) -> np.ndarray:
        return self.output

    @property
    def input_shape(self) -> Tuple[int, int]:
        return int(self.input.shape[0]), int(self.input.shape[1])

    @property
    def output_shape(self) -> Tuple[int, int]:
        return int(self.output.shape[0]), int(self.output.shape[1])

    @property
    def preserves_dimensions(self) -> bool:
        return self.input_shape == self.output_shape


@dataclass(frozen=True)
class ArcTask:
    """
    Solver-facing representation of an ARC task.

    CRITICAL SAFETY GUARANTEE:
    This class NEVER stores ground-truth test solutions.
    Solvers only receive train pairs and test inputs, preventing data leakage.
    """
    task_id: str
    train_pairs: Tuple[GridPair, ...]
    test_inputs: Tuple[np.ndarray, ...]
    dataset_split: str

    @property
    def train_examples(self) -> Tuple[GridPair, ...]:
        return self.train_pairs

    @property
    def split(self) -> str:
        return self.dataset_split

    @property
    def num_train(self) -> int:
        return len(self.train_pairs)

    @property
    def num_test(self) -> int:
        return len(self.test_inputs)

    @property
    def is_multi_test(self) -> bool:
        return len(self.test_inputs) > 1

    def __repr__(self) -> str:
        return (
            f"ArcTask(task_id='{self.task_id}', split='{self.dataset_split}', "
            f"train_pairs={self.num_train}, test_inputs={self.num_test})"
        )


def discover_dataset_files(search_dir: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
    """
    Discovers ARC challenge and solution JSON files across candidate search directories.

    Search priority:
    1. Explicit `search_dir` parameter.
    2. Environment variable `ARC_DATA_DIR`.
    3. `data/raw` relative to current working directory.
    4. `../data/raw` (common when running from `notebooks/` directory).
    5. `/home/naya/Documents/ARC - 2` (legacy local directory fallback).

    Returns:
        Dict mapping canonical role keys to existing Path objects.
    """
    candidates: List[Path] = []

    if search_dir is not None:
        candidates.append(Path(search_dir))

    env_dir = os.environ.get("ARC_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    cwd = Path.cwd()
    candidates.append(cwd / "data" / "raw")
    candidates.append(cwd / ".." / "data" / "raw")
    candidates.append(cwd)
    candidates.append(Path("/home/naya/Documents/ARC - 2"))

    role_filenames = {
        "training_challenges": "arc-agi_training_challenges.json",
        "training_solutions": "arc-agi_training_solutions.json",
        "evaluation_challenges": "arc-agi_evaluation_challenges.json",
        "evaluation_solutions": "arc-agi_evaluation_solutions.json",
        "test_challenges": "arc-agi_test_challenges.json",
        "sample_submission": "sample_submission.json",
    }

    discovered: Dict[str, Path] = {}

    for cand in candidates:
        if not cand.exists() or not cand.is_dir():
            continue
        for role, filename in role_filenames.items():
            if role not in discovered:
                target_path = cand / filename
                if target_path.exists() and target_path.is_file():
                    discovered[role] = target_path.resolve()

    return discovered


class ArcDatasetLoader:
    """
    Modular loader for ARC-AGI-2 datasets.

    Handles loading, caching, task instantiation, and strict evaluation solution isolation.
    """

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """
        Initializes the dataset loader, discovering local files.

        Args:
            data_dir: Optional custom path to directory containing dataset JSON files.
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.files = discover_dataset_files(self.data_dir)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get_available_roles(self) -> List[str]:
        """Returns list of discovered dataset role names."""
        return sorted(list(self.files.keys()))

    def _load_json_cached(self, role: str) -> Dict[str, Any]:
        """Loads and caches a JSON dataset file by role."""
        if role not in self._cache:
            if role not in self.files:
                raise FileNotFoundError(
                    f"Dataset file for role '{role}' could not be discovered. "
                    f"Available roles: {list(self.files.keys())}"
                )
            with open(self.files[role], "r", encoding="utf-8") as f:
                self._cache[role] = json.load(f)
        return self._cache[role]

    def get_raw_challenges(self, split: str = "training") -> Dict[str, Any]:
        """
        Returns the raw challenges dictionary for a split ('training', 'evaluation', or 'test').
        """
        role = f"{split}_challenges"
        return self._load_json_cached(role)

    def get_raw_solutions(self, split: str = "training") -> Dict[str, Any]:
        """
        Returns the raw solutions dictionary for a split ('training' or 'evaluation').
        """
        role = f"{split}_solutions"
        return self._load_json_cached(role)

    def get_task_ids(self, split: str = "training") -> List[str]:
        """Returns sorted list of unique task IDs present in the specified challenge split."""
        challenges = self.get_raw_challenges(split)
        return sorted(list(challenges.keys()))

    def has_task(self, task_id: str, split: str = "training") -> bool:
        """Checks if a task ID exists in the given split."""
        challenges = self.get_raw_challenges(split)
        return task_id in challenges

    def load_task(self, task_id: str, split: Optional[str] = None) -> ArcTask:
        """
        Loads a single ARC task as a safe ArcTask instance.

        If split is None, searches in order: 'training', 'evaluation', 'test'.

        CRITICAL NOTE:
        This returns ONLY training pairs and test inputs.
        Test outputs are never attached to ArcTask.
        """
        splits_to_check = [split] if split else ["training", "evaluation", "test"]

        found_split = None
        task_data = None

        for s in splits_to_check:
            try:
                challenges = self.get_raw_challenges(s)
                if task_id in challenges:
                    found_split = s
                    task_data = challenges[task_id]
                    break
            except FileNotFoundError:
                continue

        if task_data is None or found_split is None:
            raise KeyError(f"Task ID '{task_id}' not found in split(s) {splits_to_check}.")

        train_pairs = []
        for pair in task_data.get("train", []):
            inp = np.array(pair["input"], dtype=np.int32)
            out = np.array(pair["output"], dtype=np.int32)
            train_pairs.append(GridPair(input=inp, output=out))

        test_inputs = []
        for item in task_data.get("test", []):
            inp = np.array(item["input"], dtype=np.int32)
            test_inputs.append(inp)

        return ArcTask(
            task_id=task_id,
            train_pairs=tuple(train_pairs),
            test_inputs=tuple(test_inputs),
            dataset_split=found_split,
        )

    def load_ground_truth(self, task_id: str, split: Optional[str] = None) -> List[np.ndarray]:
        """
        EVALUATION-ONLY ACCESSOR:
        Retrieves ground-truth test outputs for a task.

        WARNING:
        This function is intended solely for offline benchmark evaluation, verification,
        and human inspection. It MUST NOT be invoked by or exposed to any solver reasoning pipeline.
        """
        splits_to_check = [split] if split else ["training", "evaluation"]

        for s in splits_to_check:
            try:
                solutions = self.get_raw_solutions(s)
                if task_id in solutions:
                    raw_sols = solutions[task_id]
                    return [np.array(g, dtype=np.int32) for g in raw_sols]
            except FileNotFoundError:
                continue

        raise KeyError(
            f"Ground-truth solution for task ID '{task_id}' not found in solution sets {splits_to_check}."
        )

    def load_split_tasks(self, split: str = "training") -> List[ArcTask]:
        """Loads all tasks in a split as safe ArcTask instances."""
        task_ids = self.get_task_ids(split)
        return [self.load_task(t_id, split=split) for t_id in task_ids]

    def load_training_tasks(self) -> List[ArcTask]:
        return self.load_split_tasks("training")

    def load_evaluation_tasks(self) -> List[ArcTask]:
        return self.load_split_tasks("evaluation")

    def load_test_challenges(self) -> List[ArcTask]:
        return self.load_split_tasks("test")

    def load_solutions_dict(self, split: str = "training") -> Dict[str, List[np.ndarray]]:
        """EVALUATION-ONLY ACCESSOR: Loads ground truth solutions for an entire split."""
        raw = self.get_raw_solutions(split)
        return {
            t_id: [np.array(g, dtype=np.int32) for g in sols]
            for t_id, sols in raw.items()
        }

    def load_training_solutions(self) -> Dict[str, List[np.ndarray]]:
        return self.load_solutions_dict("training")

    def load_evaluation_solutions(self) -> Dict[str, List[np.ndarray]]:
        return self.load_solutions_dict("evaluation")


# Global convenience function
def load_task(task_id: str, split: Optional[str] = None, data_dir: Optional[Union[str, Path]] = None) -> ArcTask:
    """Convenience function to load a task using default discovery paths."""
    loader = ArcDatasetLoader(data_dir=data_dir)
    return loader.load_task(task_id, split=split)
