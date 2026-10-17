"""
Canonical ARC Grid and Bounding Box representation.

Enforces domain constraints:
1. Exact integer color values in range [0, 9].
2. 2D rectangular matrices with heights and widths in range [1, 30].
3. Immutability and hashability for robust memoization and cache keys.
4. Non-mutating transformations and geometry extraction.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Represents an axis-aligned bounding box on a 2D ARC grid."""
    min_row: int
    min_col: int
    max_row: int
    max_col: int

    @property
    def height(self) -> int:
        return self.max_row - self.min_row + 1

    @property
    def width(self) -> int:
        return self.max_col - self.min_col + 1

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.min_row + self.max_row) / 2.0, (self.min_col + self.max_col) / 2.0)

    @property
    def is_square(self) -> bool:
        return self.height == self.width

    def contains(self, other: "BoundingBox") -> bool:
        """Returns True if this bounding box completely encloses other."""
        return (
            self.min_row <= other.min_row
            and self.max_row >= other.max_row
            and self.min_col <= other.min_col
            and self.max_col >= other.max_col
        )

    def intersects(self, other: "BoundingBox") -> bool:
        """Returns True if this bounding box overlaps other."""
        return not (
            self.max_row < other.min_row
            or self.min_row > other.max_row
            or self.max_col < other.min_col
            or self.min_col > other.max_col
        )

    def intersection(self, other: "BoundingBox") -> Optional["BoundingBox"]:
        """Computes the intersection bounding box, or None if disjoint."""
        if not self.intersects(other):
            return None
        return BoundingBox(
            min_row=max(self.min_row, other.min_row),
            min_col=max(self.min_col, other.min_col),
            max_row=min(self.max_row, other.max_row),
            max_col=min(self.max_col, other.max_col),
        )

    def union(self, other: "BoundingBox") -> "BoundingBox":
        """Computes the minimal bounding box enclosing both."""
        return BoundingBox(
            min_row=min(self.min_row, other.min_row),
            min_col=min(self.min_col, other.min_col),
            max_row=max(self.max_row, other.max_row),
            max_col=max(self.max_col, other.max_col),
        )


class ArcGrid:
    """
    Immutable representation of an ARC grid.

    Wraps a 2D NumPy array with dtype int32.
    Guarantees that cell values are integers 0-9 and shape is valid [1, 30] x [1, 30].
    """
    __slots__ = ("_data", "_hash", "_shape", "_colors", "_color_counts")

    def __init__(
        self,
        data: Optional[Union[np.ndarray, List[List[int]], "ArcGrid"]] = None,
        np_grid: Optional[np.ndarray] = None,
    ):
        if data is None and np_grid is not None:
            data = np_grid
        elif data is None:
            raise ValueError("ArcGrid requires data or np_grid argument")

        if isinstance(data, ArcGrid):
            self._data = data._data.copy()
        elif isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError(f"ArcGrid must be 2-dimensional, got {data.ndim}D array with shape {data.shape}")
            if not np.issubdtype(data.dtype, np.integer):
                raise TypeError(f"ArcGrid dtype must be integer, got {data.dtype}")
            self._data = np.ascontiguousarray(data, dtype=np.int32)
        elif isinstance(data, (list, tuple)):
            if len(data) == 0:
                raise ValueError("ArcGrid cannot be empty (0 rows).")
            row_len = len(data[0]) if isinstance(data[0], (list, tuple)) else -1
            if row_len <= 0:
                raise ValueError("ArcGrid cannot have empty columns.")
            for r_idx, row in enumerate(data):
                if len(row) != row_len:
                    raise ValueError(f"Non-rectangular grid: row {r_idx} length {len(row)} != {row_len}")
            self._data = np.array(data, dtype=np.int32)
        else:
            raise TypeError(f"Unsupported data type for ArcGrid: {type(data)}")

        h, w = self._data.shape
        if h < 1 or h > 30 or w < 1 or w > 30:
            raise ValueError(f"Grid dimensions ({h}, {w}) outside allowed range [1, 30] x [1, 30]")

        # Check values
        if np.any((self._data < 0) | (self._data > 9)):
            invalid = np.unique(self._data[(self._data < 0) | (self._data > 9)]).tolist()
            raise ValueError(f"Grid contains invalid color integers: {invalid}. Expected range [0, 9].")

        # Make read-only to guarantee immutability
        self._data.flags.writeable = False

        self._shape = (int(h), int(w))
        self._hash: Optional[int] = None
        self._colors: Optional[Tuple[int, ...]] = None
        self._color_counts: Optional[Dict[int, int]] = None

    @property
    def raw(self) -> np.ndarray:
        """Returns read-only numpy array view."""
        return self._data

    @property
    def height(self) -> int:
        return self._shape[0]

    @property
    def width(self) -> int:
        return self._shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def size(self) -> int:
        return self.height * self.width

    @property
    def is_square(self) -> bool:
        return self.height == self.width

    @property
    def aspect_ratio(self) -> float:
        return float(self.height) / float(self.width)

    @property
    def unique_colors(self) -> Tuple[int, ...]:
        """Returns tuple of distinct color integers present in grid, sorted."""
        if self._colors is None:
            self._colors = tuple(sorted(np.unique(self._data).tolist()))
        return self._colors

    @property
    def color_counts(self) -> Dict[int, int]:
        """Returns dictionary mapping color ID to cell count."""
        if self._color_counts is None:
            counts = {c: 0 for c in range(10)}
            unique, count = np.unique(self._data, return_counts=True)
            for u, cnt in zip(unique, count):
                counts[int(u)] = int(cnt)
            self._color_counts = counts
        return self._color_counts

    def count_color(self, color: int) -> int:
        """Returns count of cells matching the given color."""
        return self.color_counts.get(color, 0)

    def get(self, row: int, col: int) -> int:
        """Gets color value at (row, col)."""
        return int(self._data[row, col])

    def to_list(self) -> List[List[int]]:
        """Converts grid to Python list of lists."""
        return self._data.tolist()

    def copy_numpy(self) -> np.ndarray:
        """Returns a writeable copy as a NumPy int32 array."""
        return self._data.copy()

    def to_numpy(self) -> np.ndarray:
        """Alias returning writeable NumPy array."""
        return self._data.copy()

    def crop(self, bbox: BoundingBox) -> "ArcGrid":
        """Crops subgrid using given BoundingBox."""
        sub = self._data[bbox.min_row : bbox.max_row + 1, bbox.min_col : bbox.max_col + 1]
        return ArcGrid(sub)

    def crop_coordinates(self, min_row: int, min_col: int, max_row: int, max_col: int) -> "ArcGrid":
        """Crops subgrid by coordinates (inclusive)."""
        bbox = BoundingBox(min_row, min_col, max_row, max_col)
        return self.crop(bbox)

    def content_bounding_box(self, background_color: int = 0) -> Optional[BoundingBox]:
        """Returns bounding box enclosing all non-background pixels, or None if all background."""
        non_bg = np.nonzero(self._data != background_color)
        if len(non_bg[0]) == 0:
            return None
        min_r, max_r = int(np.min(non_bg[0])), int(np.max(non_bg[0]))
        min_c, max_c = int(np.min(non_bg[1])), int(np.max(non_bg[1]))
        return BoundingBox(min_r, min_c, max_r, max_c)

    def color_bounding_box(self, color: int) -> Optional[BoundingBox]:
        """Returns bounding box enclosing all pixels of specified color, or None if absent."""
        matching = np.nonzero(self._data == color)
        if len(matching[0]) == 0:
            return None
        min_r, max_r = int(np.min(matching[0])), int(np.max(matching[0]))
        min_c, max_c = int(np.min(matching[1])), int(np.max(matching[1]))
        return BoundingBox(min_r, min_c, max_r, max_c)

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ArcGrid):
            return self._shape == other._shape and np.array_equal(self._data, other._data)
        if isinstance(other, np.ndarray):
            return self._shape == other.shape and np.array_equal(self._data, other)
        if isinstance(other, list):
            try:
                arr = np.array(other, dtype=np.int32)
                return self._shape == arr.shape and np.array_equal(self._data, arr)
            except Exception:
                return False
        return False

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self._data.tobytes())
        return self._hash

    def __repr__(self) -> str:
        return f"ArcGrid(shape={self._shape}, colors={self.unique_colors})"

    def __str__(self) -> str:
        lines = ["[" + " ".join(f"{c:1d}" for c in row) + "]" for row in self._data]
        return "\n".join(lines)
