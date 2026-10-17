"""
Object and Region representation for ARC perception.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from src.representation.grid import ArcGrid, BoundingBox


@dataclass(frozen=True)
class ArcObject:
    """
    Represents an extracted visual object, connected component, or coherent region in an ARC grid.
    """
    object_id: int
    mask: np.ndarray  # 2D boolean mask of shape (H, W)
    bbox: BoundingBox
    color: int        # Dominant color or single color; -1 if multi-colored
    is_monochrome: bool
    colors_present: Tuple[int, ...]
    pixel_coordinates: Tuple[Tuple[int, int], ...]
    area: int
    perimeter: int
    centroid: Tuple[float, float]
    shape_type: str   # 'single_pixel', 'rectangle', 'h_line', 'v_line', 'square', 'cross', 'frame', 'irregular'
    holes_count: int = 0
    is_solid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    @property
    def height(self) -> int:
        return self.bbox.height

    @property
    def width(self) -> int:
        return self.bbox.width

    @property
    def density(self) -> float:
        """Ratio of object pixels to bounding box area."""
        bbox_area = self.bbox.area
        return float(self.area) / float(bbox_area) if bbox_area > 0 else 0.0

    def extract_subgrid(self, source_grid: ArcGrid, background_color: int = 0) -> np.ndarray:
        """Extracts the object rendered in its minimal bounding box."""
        cropped_source = source_grid.raw[
            self.bbox.min_row : self.bbox.max_row + 1,
            self.bbox.min_col : self.bbox.max_col + 1
        ]
        cropped_mask = self.mask[
            self.bbox.min_row : self.bbox.max_row + 1,
            self.bbox.min_col : self.bbox.max_col + 1
        ]
        out = np.full_like(cropped_source, fill_value=background_color)
        out[cropped_mask] = cropped_source[cropped_mask]
        return out

    def overlaps(self, other: "ArcObject") -> bool:
        """Returns True if this object shares any pixels with another."""
        if not self.bbox.intersects(other.bbox):
            return False
        return bool(np.any(self.mask & other.mask))

    def contains(self, other: "ArcObject") -> bool:
        """Returns True if this object's bounding box completely contains the other's."""
        return self.bbox.contains(other.bbox)
