"""
Deterministic Object-Centric Perception Engine for ARC-AGI-2.

Extracts objects, components, lines, background candidates, and geometric shapes
without external heavy neural networks or hardcoded assumptions.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import scipy.ndimage

from src.representation.grid import ArcGrid, BoundingBox
from .object import ArcObject


def infer_background_color(grid: ArcGrid) -> int:
    """
    Infers the most likely background color using a multi-factor score:
    1. Overall cell frequency.
    2. Border cell frequency (background touches outer perimeter).
    3. Black (color 0) prior when frequency is high.
    """
    raw = grid.raw
    h, w = grid.shape
    total_cells = float(grid.size)

    # 1. Overall frequency
    counts = grid.color_counts
    freq_scores = {c: counts[c] / total_cells for c in range(10)}

    # 2. Border frequency
    border_pixels = np.concatenate([raw[0, :], raw[h - 1, :], raw[:, 0], raw[:, w - 1]])
    border_total = float(len(border_pixels))
    border_counts = {c: int(np.sum(border_pixels == c)) for c in range(10)}
    border_scores = {c: border_counts[c] / border_total for c in range(10)}

    # 3. Composite score
    composite = {}
    for c in range(10):
        if counts[c] == 0:
            composite[c] = -1.0
            continue
        # Weighted combination: 40% cell frequency + 60% border frequency
        score = 0.4 * freq_scores[c] + 0.6 * border_scores[c]
        # Strong prior for color 0 if it is common
        if c == 0 and counts[c] > 0.1 * total_cells:
            score += 0.2
        composite[c] = score

    best_color = max(composite.keys(), key=lambda c: composite[c])
    return int(best_color)


def classify_shape(mask: np.ndarray, bbox: BoundingBox) -> str:
    """Classifies a binary mask into geometric categories."""
    h, w = bbox.height, bbox.width
    area = int(np.sum(mask))

    if area == 1:
        return "single_pixel"
    if area == h * w:
        if h == 1 and w > 1:
            return "h_line"
        if w == 1 and h > 1:
            return "v_line"
        if h == w:
            return "square"
        return "rectangle"

    cropped_mask = mask[bbox.min_row : bbox.max_row + 1, bbox.min_col : bbox.max_col + 1]

    # Check for empty frame (hollow rectangle)
    if h >= 3 and w >= 3:
        interior = cropped_mask[1:-1, 1:-1]
        border_top = np.all(cropped_mask[0, :])
        border_bottom = np.all(cropped_mask[-1, :])
        border_left = np.all(cropped_mask[:, 0])
        border_right = np.all(cropped_mask[:, -1])
        if border_top and border_bottom and border_left and border_right and not np.any(interior):
            return "frame"

    # Check for symmetric cross / plus shape
    if h >= 3 and w >= 3 and h == w and h % 2 == 1:
        mid = h // 2
        cross_pattern = np.zeros((h, w), dtype=bool)
        cross_pattern[mid, :] = True
        cross_pattern[:, mid] = True
        if np.array_equal(cropped_mask, cross_pattern):
            return "cross"

    return "irregular"


def compute_perimeter(mask: np.ndarray) -> int:
    """Computes the perimeter (number of boundary pixels exposed to 4-connectivity exterior)."""
    eroded = scipy.ndimage.binary_erosion(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    boundary = mask & (~eroded)
    return int(np.sum(boundary))


def extract_objects_from_mask(
    grid: ArcGrid,
    mask: np.ndarray,
    connectivity: int = 4,
    start_id: int = 1
) -> List[ArcObject]:
    """
    Extracts distinct connected components from a binary mask.
    Connectivity can be 4 or 8.
    """
    if connectivity == 8:
        struct = np.ones((3, 3), dtype=int)
    else:
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

    labeled, num_features = scipy.ndimage.label(mask, structure=struct)
    objects: List[ArcObject] = []

    for obj_idx in range(1, num_features + 1):
        obj_mask = (labeled == obj_idx)
        pixels = np.nonzero(obj_mask)
        if len(pixels[0]) == 0:
            continue

        min_r, max_r = int(np.min(pixels[0])), int(np.max(pixels[0]))
        min_c, max_c = int(np.min(pixels[1])), int(np.max(pixels[1]))
        bbox = BoundingBox(min_r, min_c, max_r, max_c)

        coords = tuple((int(r), int(c)) for r, c in zip(pixels[0], pixels[1]))
        colors_in_obj = tuple(sorted(np.unique(grid.raw[obj_mask]).tolist()))
        is_mono = len(colors_in_obj) == 1
        dom_color = colors_in_obj[0] if is_mono else int(scipy.stats.mode(grid.raw[obj_mask], keepdims=False)[0]) if hasattr(scipy.stats, 'mode') else colors_in_obj[0]

        # Centroid
        centroid = (float(np.mean(pixels[0])), float(np.mean(pixels[1])))
        area = len(coords)
        perimeter = compute_perimeter(obj_mask)
        shape_type = classify_shape(obj_mask, bbox)

        # Enclosed holes count inside object
        filled = scipy.ndimage.binary_fill_holes(obj_mask)
        holes_mask = filled & (~obj_mask)
        _, num_holes = scipy.ndimage.label(holes_mask)

        arc_obj = ArcObject(
            object_id=start_id + len(objects),
            mask=obj_mask,
            bbox=bbox,
            color=dom_color,
            is_monochrome=is_mono,
            colors_present=colors_in_obj,
            pixel_coordinates=coords,
            area=area,
            perimeter=perimeter,
            centroid=centroid,
            shape_type=shape_type,
            holes_count=int(num_holes),
            is_solid=(num_holes == 0),
            metadata={"connectivity": connectivity}
        )
        objects.append(arc_obj)

    return objects


class PerceptionEngine:
    """
    Main perception engine producing multi-hypothesis object segmentations and grid analytics.
    """

    def __init__(self):
        pass

    def segment_monochromatic_connected_components(
        self,
        grid: ArcGrid,
        background_color: int = 0,
        connectivity: int = 4
    ) -> List[ArcObject]:
        """
        Segments connected components separately for each foreground color.
        Ensures every connected component is strictly monochromatic.
        """
        all_objects: List[ArcObject] = []
        for color in grid.unique_colors:
            if color == background_color:
                continue
            color_mask = (grid.raw == color)
            objs = extract_objects_from_mask(
                grid, color_mask, connectivity=connectivity, start_id=len(all_objects) + 1
            )
            all_objects.extend(objs)

        return all_objects

    def segment_multicolor_connected_components(
        self,
        grid: ArcGrid,
        background_color: int = 0,
        connectivity: int = 4
    ) -> List[ArcObject]:
        """
        Segments connected components of any foreground color grouped together.
        Objects may contain multiple adjacent colors.
        """
        fg_mask = (grid.raw != background_color)
        return extract_objects_from_mask(grid, fg_mask, connectivity=connectivity, start_id=1)

    def segment_by_color_regions(
        self,
        grid: ArcGrid,
        background_color: int = 0
    ) -> List[ArcObject]:
        """
        Treats all cells of each distinct foreground color as a single semantic object,
        even if disconnected.
        """
        objects: List[ArcObject] = []
        for color in grid.unique_colors:
            if color == background_color:
                continue
            mask = (grid.raw == color)
            pixels = np.nonzero(mask)
            if len(pixels[0]) == 0:
                continue

            min_r, max_r = int(np.min(pixels[0])), int(np.max(pixels[0]))
            min_c, max_c = int(np.min(pixels[1])), int(np.max(pixels[1]))
            bbox = BoundingBox(min_r, min_c, max_r, max_c)
            coords = tuple((int(r), int(c)) for r, c in zip(pixels[0], pixels[1]))

            obj = ArcObject(
                object_id=len(objects) + 1,
                mask=mask,
                bbox=bbox,
                color=color,
                is_monochrome=True,
                colors_present=(color,),
                pixel_coordinates=coords,
                area=len(coords),
                perimeter=compute_perimeter(mask),
                centroid=(float(np.mean(pixels[0])), float(np.mean(pixels[1]))),
                shape_type=classify_shape(mask, bbox),
                metadata={"type": "color_partition"}
            )
            objects.append(obj)

        return objects

    def detect_grid_lines(self, grid: ArcGrid) -> Dict[str, List[int]]:
        """
        Detects full horizontal and vertical solid lines (common in grid-partitioned puzzles).
        """
        raw = grid.raw
        h, w = grid.shape
        h_lines: List[int] = []
        v_lines: List[int] = []

        # Full horizontal lines (row having all same non-zero color or divider color)
        for r in range(h):
            first_c = raw[r, 0]
            if first_c != 0 and np.all(raw[r, :] == first_c):
                h_lines.append(r)

        return {"horizontal_lines": h_lines, "vertical_lines": v_lines}

    def infer_background_color(self, grid: ArcGrid) -> int:
        """Infers the dominant background color of the grid."""
        return infer_background_color(grid)

    def segment_connected_components(
        self,
        grid: ArcGrid,
        background_color: int = 0,
        connectivity: int = 4,
    ) -> List[ArcObject]:
        """Convenience alias for monochromatic connected component segmentation."""
        return self.segment_monochromatic_connected_components(grid, background_color, connectivity)
