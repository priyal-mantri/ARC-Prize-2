"""Unit tests for Scene Graph Construction and Spatial Relational Reasoning."""

import numpy as np
import pytest

from src.representation.grid import ArcGrid
from src.perception.detector import PerceptionEngine
from src.scene_graph.builder import SceneGraphBuilder


def test_scene_graph_construction_and_relations():
    engine = PerceptionEngine()
    builder = SceneGraphBuilder()

    arr = np.zeros((8, 8), dtype=np.int32)
    # Object 1: color 1, top-left
    arr[1:3, 1:3] = 1
    # Object 2: color 1, bottom-right
    arr[5:7, 5:7] = 1
    # Object 3: color 2, 1x1, middle
    arr[4, 2] = 2

    grid = ArcGrid(arr)
    objects = engine.segment_monochromatic_connected_components(grid, background_color=0)
    assert len(objects) == 3

    graph = builder.build(grid, objects, background_color=0)
    assert graph.num_nodes == 3
    assert graph.num_edges > 0

    # Test spatial relations
    node_by_color = {}
    for nid, node in graph.nodes.items():
        if node.color not in node_by_color:
            node_by_color[node.color] = []
        node_by_color[node.color].append(nid)

    # Object 1 is above Object 2
    o1_id = [nid for nid, n in graph.nodes.items() if n.bbox.min_row == 1][0]
    o2_id = [nid for nid, n in graph.nodes.items() if n.bbox.min_row == 5][0]

    rels_1_to_2 = graph.get_relations_between(o1_id, o2_id)
    assert "above" in rels_1_to_2
    assert "left_of" in rels_1_to_2
    assert "same_color" in rels_1_to_2

    rels_2_to_1 = graph.get_relations_between(o2_id, o1_id)
    assert "below" in rels_2_to_1
    assert "right_of" in rels_2_to_1

    # Check serialization
    graph_dict = graph.to_dict()
    assert "nodes" in graph_dict
    assert "edges" in graph_dict
    assert len(graph_dict["nodes"]) == 3

    # Check summary string
    summary = graph.summary()
    assert "SceneGraph" in summary
    assert "Node" in summary
