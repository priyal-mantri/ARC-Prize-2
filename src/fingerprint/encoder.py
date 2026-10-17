"""
Structural Fingerprint Encoder for ARC-AGI-2 grids and tasks.

Extracts a deterministic 20-dimensional topological and geometric feature vector
compatible with the existing MirrorMatrix memory bank, with schema versioning.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.ndimage

from src.representation.grid import ArcGrid

FINGERPRINT_DIM_V1 = 20
CURRENT_SCHEMA_VERSION = "v1_20d"


def grid_to_barcode(grid: Union[ArcGrid, np.ndarray, List[List[int]]]) -> np.ndarray:
    """
    Extracts a 20-dimensional topological and geometric feature barcode from a grid.

    Vector layout (20 float32 features):
      [0]     num_components: Connected components (foreground)
      [1]     max_component_size: Maximum component pixel count
      [2]     min_component_size: Minimum component pixel count
      [3]     avg_component_size: Mean component pixel count
      [4]     unique_colors_count: Number of distinct colors in grid
      [5..14] color_histogram: Normalized cell frequency for colors 0 to 9
      [15]    grid_height: Grid height in rows
      [16]    grid_width: Grid width in columns
      [17]    aspect_ratio: Height / Width
      [18]    symmetry_score: Mean horizontal and vertical symmetry ratio (0.0 to 1.0)
      [19]    enclosed_holes_count: Number of enclosed background voids

    Returns:
        np.ndarray of shape (20,) and dtype np.float32.
    """
    if isinstance(grid, ArcGrid):
        raw = grid.raw
    elif isinstance(grid, np.ndarray):
        raw = grid
    elif isinstance(grid, (list, tuple)):
        raw = np.array(grid, dtype=np.int32)
    else:
        return np.zeros(FINGERPRINT_DIM_V1, dtype=np.float32)

    if raw.size == 0 or raw.ndim != 2:
        return np.zeros(FINGERPRINT_DIM_V1, dtype=np.float32)

    h, w = raw.shape
    total_pixels = float(h * w)
    barcode = np.zeros(FINGERPRINT_DIM_V1, dtype=np.float32)

    # 1. Connected components (Objects on foreground > 0)
    mask = (raw > 0)
    labeled, num_features = scipy.ndimage.label(mask)
    barcode[0] = float(num_features)

    if num_features > 0:
        sizes = [float(np.sum(labeled == i)) for i in range(1, num_features + 1)]
        barcode[1] = float(max(sizes))
        barcode[2] = float(min(sizes))
        barcode[3] = float(np.mean(sizes))
    else:
        barcode[1] = 0.0
        barcode[2] = 0.0
        barcode[3] = 0.0

    # 2. Unique colors count & 10-color normalized histogram
    unique_colors = np.unique(raw)
    barcode[4] = float(len(unique_colors))

    hist, _ = np.histogram(raw, bins=10, range=(0, 10))
    norm_hist = hist.astype(np.float32) / total_pixels
    barcode[5:15] = norm_hist

    # 3. Geometric shape features
    barcode[15] = float(h)
    barcode[16] = float(w)
    barcode[17] = float(h) / float(w) if w > 0 else 1.0

    # 4. Symmetry score (mean of horizontal and vertical reflection match ratio)
    sym_h = np.mean(raw == np.fliplr(raw)) if w > 1 else 1.0
    sym_v = np.mean(raw == np.flipud(raw)) if h > 1 else 1.0
    barcode[18] = float((sym_h + sym_v) / 2.0)

    # 5. Enclosed background holes
    filled_mask = scipy.ndimage.binary_fill_holes(mask)
    holes_mask = filled_mask & (~mask)
    _, num_holes = scipy.ndimage.label(holes_mask)
    barcode[19] = float(num_holes)

    return barcode


def task_to_fingerprint(task_or_pairs: Any) -> np.ndarray:
    """
    Computes an aggregated task-level fingerprint averaging input barcodes across demonstrations.
    Accepts ArcTask, List[TaskDemonstration], or List[Tuple[ArcGrid, ArcGrid]].
    """
    grids: List[ArcGrid] = []
    if hasattr(task_or_pairs, "train_examples"):
        grids = [ArcGrid(np_grid=ex.input_grid) for ex in task_or_pairs.train_examples]
    elif hasattr(task_or_pairs, "train_pairs"):
        grids = [ArcGrid(np_grid=p.input) for p in task_or_pairs.train_pairs]
    elif isinstance(task_or_pairs, (list, tuple)):
        for p in task_or_pairs:
            if hasattr(p, "input_grid"):
                grids.append(p.input_grid)
            elif isinstance(p, tuple) and len(p) > 0:
                grids.append(p[0] if isinstance(p[0], ArcGrid) else ArcGrid(np_grid=p[0]))
    if not grids:
        return np.zeros(FINGERPRINT_DIM_V1, dtype=np.float32)
    barcodes = [grid_to_barcode(g) for g in grids]
    return np.mean(barcodes, axis=0).astype(np.float32)
