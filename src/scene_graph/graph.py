"""
Scene Graph representation for ARC-AGI-2 grids and objects.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx

from src.representation.grid import ArcGrid, BoundingBox
from src.perception.object import ArcObject


@dataclass
class SceneNode:
    """Represents a node in the ARC scene graph."""
    node_id: int
    object_ref: Optional[ArcObject] = None
    color: int = -1
    area: int = 0
    shape_type: str = "unknown"
    centroid: Tuple[float, float] = (0.0, 0.0)
    bbox: Optional[BoundingBox] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "color": self.color,
            "area": self.area,
            "shape_type": self.shape_type,
            "centroid": self.centroid,
            "bbox": (
                (self.bbox.min_row, self.bbox.min_col, self.bbox.max_row, self.bbox.max_col)
                if self.bbox
                else None
            ),
            "attributes": self.attributes,
        }


@dataclass
class SceneEdge:
    """Represents a directed relational edge in the ARC scene graph."""
    source_id: int
    target_id: int
    relation: str  # 'left_of', 'above', 'adjacent_to', 'contains', 'aligned_h', 'same_color', etc.
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "relation": self.relation,
            "attributes": self.attributes,
        }


class ArcSceneGraph:
    """
    Structured relational scene graph of an ARC grid.

    Nodes represent perceived objects or regions; edges encode spatial, geometric,
    containment, and categorical relationships. Wraps NetworkX DiGraph.
    """

    def __init__(self, grid_shape: Tuple[int, int], background_color: int = 0):
        self.grid_shape = grid_shape
        self.background_color = background_color
        self.graph = nx.DiGraph()
        self.nodes: Dict[int, SceneNode] = {}
        self.edges: List[SceneEdge] = []

    def add_object_node(self, obj: ArcObject) -> SceneNode:
        """Adds an ArcObject as a node in the scene graph."""
        node = SceneNode(
            node_id=obj.object_id,
            object_ref=obj,
            color=obj.color,
            area=obj.area,
            shape_type=obj.shape_type,
            centroid=obj.centroid,
            bbox=obj.bbox,
            attributes={
                "is_monochrome": obj.is_monochrome,
                "is_solid": obj.is_solid,
                "holes_count": obj.holes_count,
                "perimeter": obj.perimeter,
                "density": round(obj.density, 2),
            }
        )
        self.nodes[obj.object_id] = node
        self.graph.add_node(
            obj.object_id,
            color=node.color,
            area=node.area,
            shape=node.shape_type,
            centroid=node.centroid,
            bbox=node.bbox,
            **node.attributes
        )
        return node

    def add_relation(self, source_id: int, target_id: int, relation: str, **attrs: Any) -> SceneEdge:
        """Adds a directed relation edge between source and target nodes."""
        edge = SceneEdge(source_id=source_id, target_id=target_id, relation=relation, attributes=attrs)
        self.edges.append(edge)
        self.graph.add_edge(source_id, target_id, relation=relation, **attrs)
        return edge

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def get_neighbors(self, node_id: int) -> List[int]:
        """Returns target node IDs directly reachable from node_id."""
        return list(self.graph.successors(node_id))

    def get_relations_between(self, source_id: int, target_id: int) -> List[str]:
        """Returns all relation names from source_id to target_id."""
        return [e.relation for e in self.edges if e.source_id == source_id and e.target_id == target_id]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the scene graph to a dictionary."""
        return {
            "grid_shape": self.grid_shape,
            "background_color": self.background_color,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    def summary(self) -> str:
        """Returns a human-readable textual summary of the scene graph."""
        lines = [
            f"SceneGraph (Shape: {self.grid_shape}, Bg: {self.background_color}, Nodes: {self.num_nodes}, Edges: {self.num_edges})"
        ]
        for nid, node in sorted(self.nodes.items()):
            lines.append(
                f"  Node {nid}: color={node.color}, shape={node.shape_type}, area={node.area}, centroid={node.centroid}"
            )
        for edge in self.edges:
            lines.append(f"  Edge: ({edge.source_id}) --[{edge.relation}]--> ({edge.target_id})")
        return "\n".join(lines)
