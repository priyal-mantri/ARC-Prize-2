"""Unit tests for ARC Grid and BoundingBox representation."""

import numpy as np
import pytest

from src.representation.grid import ArcGrid, BoundingBox


def test_bounding_box_geometry():
    box = BoundingBox(min_row=2, min_col=3, max_row=5, max_col=7)
    assert box.height == 4
    assert box.width == 5
    assert box.area == 20
    assert box.center == (3.5, 5.0)
    assert not box.is_square

    sq = BoundingBox(1, 1, 3, 3)
    assert sq.is_square
    assert sq.area == 9


def test_bounding_box_intersections():
    b1 = BoundingBox(0, 0, 4, 4)
    b2 = BoundingBox(2, 2, 6, 6)
    b3 = BoundingBox(5, 5, 8, 8)

    assert b1.intersects(b2)
    assert b2.intersects(b1)
    assert not b1.intersects(b3)

    inter = b1.intersection(b2)
    assert inter == BoundingBox(2, 2, 4, 4)

    union = b1.union(b3)
    assert union == BoundingBox(0, 0, 8, 8)


def test_arc_grid_creation_and_validation():
    # Valid grid
    arr = np.zeros((5, 5), dtype=np.int32)
    arr[1, 1] = 3
    grid = ArcGrid(arr)
    assert grid.shape == (5, 5)
    assert grid.height == 5
    assert grid.width == 5
    assert grid.size == 25
    assert grid.is_square
    assert grid.unique_colors == (0, 3)
    assert grid.count_color(3) == 1
    assert grid.count_color(0) == 24
    assert grid.get(1, 1) == 3

    # Invalid dimension (>30)
    huge = np.zeros((35, 5), dtype=np.int32)
    with pytest.raises(ValueError):
        ArcGrid(huge)

    # Invalid color value (>9)
    bad_val = np.zeros((3, 3), dtype=np.int32)
    bad_val[0, 0] = 12
    with pytest.raises(ValueError):
        ArcGrid(bad_val)

    # Negative color
    bad_neg = np.zeros((3, 3), dtype=np.int32)
    bad_neg[0, 0] = -1
    with pytest.raises(ValueError):
        ArcGrid(bad_neg)


def test_arc_grid_immutability_and_hashing():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    grid1 = ArcGrid(arr)
    grid2 = ArcGrid(arr)

    # Equality and hashing
    assert grid1 == grid2
    assert hash(grid1) == hash(grid2)

    # Immutability: underlying array is read-only
    with pytest.raises(ValueError):
        grid1.raw[0, 0] = 9

    # Dictionary key
    d = {grid1: "val"}
    assert d[grid2] == "val"


def test_arc_grid_cropping():
    arr = np.zeros((6, 6), dtype=np.int32)
    arr[2:4, 2:4] = 5
    grid = ArcGrid(arr)

    bbox = grid.content_bounding_box(background_color=0)
    assert bbox == BoundingBox(2, 2, 3, 3)

    cropped = grid.crop(bbox)
    assert cropped.shape == (2, 2)
    assert np.all(cropped.to_numpy() == 5)
