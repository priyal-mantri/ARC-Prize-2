"""
Scene Graph Builder that constructs relational edges between perceived objects.
"""

from typing import List, Optional, Tuple
import numpy as np
import scipy.ndimage

from src.representation.grid import ArcGrid
from src.perception.object import ArcObject
from .graph import ArcSceneGraph


class SceneGraphBuilder:
    """
    Constructs an ArcSceneGraph by populating nodes from objects and computing
    spatial, topological, and categorical relations.
    """

    def __init__(self, include_transitive_directions: bool = False):
        self.include_transitive = include_transitive_directions

    def build(
        self,
        grid: ArcGrid,
        objects: List[ArcObject],
        background_color: int = 0
    ) -> ArcSceneGraph:
        """
        Builds a comprehensive scene graph for the provided objects on the grid.
        """
        sg = ArcSceneGraph(grid_shape=grid.shape, background_color=background_color)

        # 1. Add all object nodes
        for obj in objects:
            sg.add_object_node(obj)

        n = len(objects)
        if n < 2:
            return sg

        # 2. Pairwise relational analysis
        for i in range(n):
            obj_a = objects[i]
            id_a = obj_a.object_id

            for j in range(i + 1, n):
                obj_b = objects[j]
                id_b = obj_b.object_id

                # A. Directional spatial relations (above / below, left_of / right_of)
                # We check non-overlapping bounds or clear centroid differences
                cr_a, cc_a = obj_a.centroid
                cr_b, cc_b = obj_b.centroid

                if obj_a.bbox.max_row < obj_b.bbox.min_row:
                    sg.add_relation(id_a, id_b, "above")
                    sg.add_relation(id_b, id_a, "below")
                elif obj_b.bbox.max_row < obj_a.bbox.min_row:
                    sg.add_relation(id_b, id_a, "above")
                    sg.add_relation(id_a, id_b, "below")

                if obj_a.bbox.max_col < obj_b.bbox.min_col:
                    sg.add_relation(id_a, id_b, "left_of")
                    sg.add_relation(id_b, id_a, "right_of")
                elif obj_b.bbox.max_col < obj_a.bbox.min_col:
                    sg.add_relation(id_b, id_a, "left_of")
                    sg.add_relation(id_a, id_b, "right_of")

                # B. Containment
                if obj_a.contains(obj_b):
                    sg.add_relation(id_a, id_b, "contains")
                    sg.add_relation(id_b, id_a, "inside")
                elif obj_b.contains(obj_a):
                    sg.add_relation(id_b, id_a, "contains")
                    sg.add_relation(id_a, id_b, "inside")

                # C. Alignment (center alignment tolerance: within 0.5 pixels)
                if abs(cr_a - cr_b) < 0.6:
                    sg.add_relation(id_a, id_b, "aligned_horizontal")
                    sg.add_relation(id_b, id_a, "aligned_horizontal")
                if abs(cc_a - cc_b) < 0.6:
                    sg.add_relation(id_a, id_b, "aligned_vertical")
                    sg.add_relation(id_b, id_a, "aligned_vertical")

                # D. Color & Shape equivalence
                if obj_a.color == obj_b.color and obj_a.color != -1:
                    sg.add_relation(id_a, id_b, "same_color", color=obj_a.color)
                    sg.add_relation(id_b, id_a, "same_color", color=obj_a.color)

                if obj_a.shape_type == obj_b.shape_type and obj_a.shape_type != "irregular":
                    sg.add_relation(id_a, id_b, "same_shape", shape=obj_a.shape_type)
                    sg.add_relation(id_b, id_a, "same_shape", shape=obj_a.shape_type)

                # E. Relative Size
                if obj_a.area > obj_b.area:
                    sg.add_relation(id_a, id_b, "larger_than", delta=obj_a.area - obj_b.area)
                    sg.add_relation(id_b, id_a, "smaller_than", delta=obj_a.area - obj_b.area)
                elif obj_b.area > obj_a.area:
                    sg.add_relation(id_b, id_a, "larger_than", delta=obj_b.area - obj_a.area)
                    sg.add_relation(id_a, id_b, "smaller_than", delta=obj_b.area - obj_a.area)
                else:
                    sg.add_relation(id_a, id_b, "equal_area", area=obj_a.area)
                    sg.add_relation(id_b, id_a, "equal_area", area=obj_a.area)

                # F. Adjacency / Contact (distance between masks <= 1)
                if obj_a.bbox.intersects(obj_b.bbox):
                    # Check if bounding boxes touch or overlap
                    # Fast check: dilate mask_a by 1 and check overlap with mask_b
                    # Restrict check to common bounding box to keep it fast
                    union_bbox = obj_a.bbox.union(obj_b.bbox)
                    sub_a = obj_a.mask[union_bbox.min_row : union_bbox.max_row + 1, union_bbox.min_col : union_bbox.max_col + 1]
                    sub_b = obj_b.mask[union_bbox.min_row : union_bbox.max_row + 1, union_bbox.min_col : union_bbox.max_col + 1]
                    
                    dilated_a = scipy.ndimage.binary_dilation(sub_a, structure=np.ones((3, 3), dtype=bool))
                    if np.any(dilated_a & sub_b):
                        sg.add_relation(id_a, id_b, "adjacent_to")
                        sg.add_relation(id_b, id_a, "adjacent_to")

        return sg
