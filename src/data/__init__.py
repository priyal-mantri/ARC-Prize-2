"""
ARC-AGI-2 Data Pipeline Package.
"""
from src.data.dataset_loader import (
    ArcDatasetLoader,
    ArcTask,
    GridPair,
    discover_dataset_files,
    load_task,
)
from src.data.validation import (
    ValidationResult,
    validate_dataset,
    validate_grid,
    validate_task,
)
from src.data.statistics import (
    compute_dataset_statistics,
    check_task_overlaps,
)

__all__ = [
    "ArcDatasetLoader",
    "ArcTask",
    "GridPair",
    "discover_dataset_files",
    "load_task",
    "ValidationResult",
    "validate_dataset",
    "validate_grid",
    "validate_task",
    "compute_dataset_statistics",
    "check_task_overlaps",
]
