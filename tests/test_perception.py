"""Unit tests for Perception Engine, Component Segmentation, and Hypotheses."""

import numpy as np
import pytest

from src.representation.grid import ArcGrid
from src.perception.detector import PerceptionEngine, infer_background_color
from src.perception.hypotheses import PerceptionHypothesisGenerator


def test_infer_background_color():
    # Grid surrounded by 0
    arr = np.zeros((8, 8), dtype=np.int32)
    arr[2:5, 2:5] = 4
    grid = ArcGrid(arr)
    assert infer_background_color(grid) == 0

    # Grid surrounded by 3
    arr2 = np.full((6, 6), 3, dtype=np.int32)
    arr2[1:3, 1:3] = 1
    grid2 = ArcGrid(arr2)
    assert infer_background_color(grid2) == 3


def test_connected_component_segmentation():
    engine = PerceptionEngine()
    arr = np.zeros((7, 7), dtype=np.int32)
    # Object 1: horizontal line of color 2
    arr[1, 1:5] = 2
    # Object 2: 2x2 square of color 4
    arr[4:6, 4:6] = 4
    grid = ArcGrid(arr)

    objs = engine.segment_monochromatic_connected_components(grid, background_color=0, connectivity=4)
    assert len(objs) == 2

    # Verify shapes and attributes
    obj1 = [o for o in objs if o.color == 2][0]
    assert obj1.shape_type == "h_line"
    assert obj1.area == 4

    obj2 = [o for o in objs if o.color == 4][0]
    assert obj2.shape_type == "square"
    assert obj2.area == 4
    assert obj2.is_solid


def test_4conn_vs_8conn_connectivity():
    engine = PerceptionEngine()
    arr = np.zeros((4, 4), dtype=np.int32)
    # Diagonal pixels touching only at corners
    arr[1, 1] = 5
    arr[2, 2] = 5
    grid = ArcGrid(arr)

    objs_4 = engine.segment_monochromatic_connected_components(grid, background_color=0, connectivity=4)
    assert len(objs_4) == 2  # Disconnected under 4-connectivity

    objs_8 = engine.segment_monochromatic_connected_components(grid, background_color=0, connectivity=8)
    assert len(objs_8) == 1  # Connected under 8-connectivity


def test_perception_hypotheses_generator():
    engine = PerceptionEngine()
    hypo_gen = PerceptionHypothesisGenerator(engine)

    arr = np.zeros((5, 5), dtype=np.int32)
    arr[1, 1] = 1
    arr[1, 2] = 2
    arr[3, 3] = 1
    grid = ArcGrid(arr)

    hypotheses = hypo_gen.generate_hypotheses(grid)
    assert len(hypotheses) >= 3

    hypo_names = [h.name for h in hypotheses]
    assert "mono_4conn" in hypo_names
    assert "mono_8conn" in hypo_names
    assert "color_partition" in hypo_names

    # Check confidence ordering
    assert all(0.0 <= h.confidence <= 1.0 for h in hypotheses)
