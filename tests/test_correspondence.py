"""Unit tests for Object Correspondence and Relational Change Analysis."""

import numpy as np
import pytest

from src.representation.grid import ArcGrid
from src.perception.detector import PerceptionEngine
from src.correspondence.matcher import ObjectCorrespondenceMatcher
from src.correspondence.relational_change import RelationalChangeAnalyzer
from src.search.verifier import TaskDemonstration


def test_object_correspondence_recolored_and_moved():
    engine = PerceptionEngine()
    matcher = ObjectCorrespondenceMatcher()

    # Input: 2x2 square of color 1 at (1, 1)
    inp_arr = np.zeros((6, 6), dtype=np.int32)
    inp_arr[1:3, 1:3] = 1
    inp_grid = ArcGrid(inp_arr)
    objs_in = engine.segment_monochromatic_connected_components(inp_grid, background_color=0)

    # Output: 2x2 square moved to (3, 3) and recolored to 2
    out_arr = np.zeros((6, 6), dtype=np.int32)
    out_arr[3:5, 3:5] = 2
    out_grid = ArcGrid(out_arr)
    objs_out = engine.segment_monochromatic_connected_components(out_grid, background_color=0)

    corr = matcher.match_pair(objs_in, objs_out, inp_grid, out_grid)
    assert len(corr.matches) == 1
    m = corr.matches[0]
    assert m.output_obj_id is not None
    assert m.color_change == (1, 2)
    assert m.movement_vector == (2, 2)


def test_relational_change_analyzer_consistent_recoloring():
    analyzer = RelationalChangeAnalyzer()

    # Create 2 demonstrations where color 1 consistently maps to color 2, dimensions preserved
    demos = []
    for _ in range(2):
        inp_arr = np.zeros((4, 4), dtype=np.int32)
        inp_arr[1, 1] = 1
        out_arr = np.zeros((4, 4), dtype=np.int32)
        out_arr[1, 1] = 2
        demos.append(TaskDemonstration(input_grid=ArcGrid(inp_arr), output_grid=ArcGrid(out_arr)))

    report = analyzer.analyze_task(demos)
    assert report.always_preserves_shape
    assert report.consistent_recolor_rules.get(1) == 2
    assert "recolor_mapping" in report.hypothesized_transformations
