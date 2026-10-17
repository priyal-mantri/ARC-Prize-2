"""
Object Correspondence Matcher for ARC-AGI-2 demonstration pairs.

Maps objects from input grid to output grid, detecting:
- Invariant objects (unchanged position & color)
- Moved objects (spatial displacement)
- Recolored objects (color change, preserved shape)
- Resized objects
- Created objects (in output, not in input)
- Deleted objects (in input, not in output)
- Split and Merged components
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.representation.grid import ArcGrid, BoundingBox
from src.perception.object import ArcObject


@dataclass
class ObjectMatch:
    """Represents a correspondence between an input object and an output object."""
    input_obj_id: Optional[int]
    output_obj_id: Optional[int]
    match_type: str  # 'exact_identity', 'moved', 'recolored', 'recolored_and_moved', 'resized', 'created', 'deleted'
    confidence: float
    color_transition: Optional[Tuple[int, int]] = None  # (input_color, output_color)
    translation_delta: Optional[Tuple[int, int]] = None  # (row_shift, col_shift)
    area_ratio: float = 1.0
    shape_preserved: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def color_change(self) -> Optional[Tuple[int, int]]:
        return self.color_transition

    @property
    def movement_vector(self) -> Optional[Tuple[int, int]]:
        return self.translation_delta


@dataclass
class DemonstrationCorrespondence:
    """Stores all object correspondences and changes for a single demonstration pair."""
    pair_index: int
    matches: List[ObjectMatch]
    created_objects: List[int]   # output obj IDs with no input match
    deleted_objects: List[int]   # input obj IDs with no output match
    color_mappings: Dict[int, int]  # input color -> output color
    global_movement: Optional[Tuple[int, int]] = None
    dimension_change: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 0), (0, 0))


class ObjectCorrespondenceMatcher:
    """
    Computes correspondences between objects perceived in input and output grids.
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def match_pair(
        self,
        input_objects: List[ArcObject],
        output_objects: List[ArcObject],
        input_grid: ArcGrid,
        output_grid: ArcGrid,
        pair_index: int = 0
    ) -> DemonstrationCorrespondence:
        """
        Finds the optimal bipartite object correspondences between input and output objects.
        """
        dim_change = (input_grid.shape, output_grid.shape)
        same_dims = (input_grid.shape == output_grid.shape)

        matches: List[ObjectMatch] = []
        matched_in: Set[int] = set()
        matched_out: Set[int] = set()
        color_maps: Dict[int, int] = {}

        # 1. First Pass: Exact Identity (exact mask + exact color match)
        if same_dims:
            for obj_in in input_objects:
                for obj_out in output_objects:
                    if obj_out.object_id in matched_out:
                        continue
                    if obj_in.color == obj_out.color and np.array_equal(obj_in.mask, obj_out.mask):
                        matches.append(
                            ObjectMatch(
                                input_obj_id=obj_in.object_id,
                                output_obj_id=obj_out.object_id,
                                match_type="exact_identity",
                                confidence=1.0,
                                color_transition=(obj_in.color, obj_out.color),
                                translation_delta=(0, 0),
                                area_ratio=1.0,
                                shape_preserved=True
                            )
                        )
                        matched_in.add(obj_in.object_id)
                        matched_out.add(obj_out.object_id)
                        color_maps[obj_in.color] = obj_out.color
                        break

        # 2. Second Pass: Pure Recoloring (identical spatial mask, different color)
        if same_dims:
            for obj_in in input_objects:
                if obj_in.object_id in matched_in:
                    continue
                for obj_out in output_objects:
                    if obj_out.object_id in matched_out:
                        continue
                    if np.array_equal(obj_in.mask, obj_out.mask):
                        matches.append(
                            ObjectMatch(
                                input_obj_id=obj_in.object_id,
                                output_obj_id=obj_out.object_id,
                                match_type="recolored",
                                confidence=0.95,
                                color_transition=(obj_in.color, obj_out.color),
                                translation_delta=(0, 0),
                                area_ratio=1.0,
                                shape_preserved=True
                            )
                        )
                        matched_in.add(obj_in.object_id)
                        matched_out.add(obj_out.object_id)
                        color_maps[obj_in.color] = obj_out.color
                        break

        # 3. Third Pass: Rigid Movement (same shape and color, shifted position)
        for obj_in in input_objects:
            if obj_in.object_id in matched_in:
                continue
            for obj_out in output_objects:
                if obj_out.object_id in matched_out:
                    continue

                if obj_in.area == obj_out.area and obj_in.shape_type == obj_out.shape_type:
                    # Check if bounding box shapes match
                    if (obj_in.bbox.height, obj_in.bbox.width) == (obj_out.bbox.height, obj_out.bbox.width):
                        # Extract subgrids
                        sub_in = obj_in.extract_subgrid(input_grid)
                        sub_out = obj_out.extract_subgrid(output_grid)
                        # Check binary shape equality
                        if np.array_equal(sub_in > 0, sub_out > 0):
                            dr = obj_out.bbox.min_row - obj_in.bbox.min_row
                            dc = obj_out.bbox.min_col - obj_in.bbox.min_col
                            m_type = "moved" if obj_in.color == obj_out.color else "recolored_and_moved"
                            conf = 0.90 if m_type == "moved" else 0.80

                            matches.append(
                                ObjectMatch(
                                    input_obj_id=obj_in.object_id,
                                    output_obj_id=obj_out.object_id,
                                    match_type=m_type,
                                    confidence=conf,
                                    color_transition=(obj_in.color, obj_out.color),
                                    translation_delta=(dr, dc),
                                    area_ratio=1.0,
                                    shape_preserved=True
                                )
                            )
                            matched_in.add(obj_in.object_id)
                            matched_out.add(obj_out.object_id)
                            color_maps[obj_in.color] = obj_out.color
                            break

        # 4. Fourth Pass: IoU Overlap (partial match or reshaping)
        if same_dims:
            for obj_in in input_objects:
                if obj_in.object_id in matched_in:
                    continue
                best_iou = 0.0
                best_out = None
                for obj_out in output_objects:
                    if obj_out.object_id in matched_out:
                        continue
                    intersection = np.sum(obj_in.mask & obj_out.mask)
                    if intersection > 0:
                        union = np.sum(obj_in.mask | obj_out.mask)
                        iou = float(intersection) / float(union) if union > 0 else 0.0
                        if iou > best_iou:
                            best_iou = iou
                            best_out = obj_out

                if best_out is not None and best_iou >= self.iou_threshold:
                    matches.append(
                        ObjectMatch(
                            input_obj_id=obj_in.object_id,
                            output_obj_id=best_out.object_id,
                            match_type="resized" if obj_in.color == best_out.color else "recolored_and_resized",
                            confidence=0.75 * best_iou,
                            color_transition=(obj_in.color, best_out.color),
                            translation_delta=(0, 0),
                            area_ratio=float(best_out.area) / float(obj_in.area),
                            shape_preserved=(obj_in.shape_type == best_out.shape_type)
                        )
                    )
                    matched_in.add(obj_in.object_id)
                    matched_out.add(best_out.object_id)
                    color_maps[obj_in.color] = best_out.color

        # 5. Record unmatched created and deleted objects
        deleted_ids = [o.object_id for o in input_objects if o.object_id not in matched_in]
        created_ids = [o.object_id for o in output_objects if o.object_id not in matched_out]

        for d_id in deleted_ids:
            matches.append(
                ObjectMatch(
                    input_obj_id=d_id,
                    output_obj_id=None,
                    match_type="deleted",
                    confidence=1.0,
                    shape_preserved=False
                )
            )

        for c_id in created_ids:
            matches.append(
                ObjectMatch(
                    input_obj_id=None,
                    output_obj_id=c_id,
                    match_type="created",
                    confidence=1.0,
                    shape_preserved=False
                )
            )

        # Detect consensus movement if multiple objects moved by the same delta
        deltas = [m.translation_delta for m in matches if m.translation_delta is not None and m.translation_delta != (0, 0)]
        global_move = deltas[0] if len(deltas) > 0 and all(d == deltas[0] for d in deltas) else None

        return DemonstrationCorrespondence(
            pair_index=pair_index,
            matches=matches,
            created_objects=created_ids,
            deleted_objects=deleted_ids,
            color_mappings=color_maps,
            global_movement=global_move,
            dimension_change=dim_change
        )
