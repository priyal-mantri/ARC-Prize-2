"""
Relational Change Analyzer evaluating transformation invariants across all demonstrations of a task.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from src.representation.grid import ArcGrid
from src.perception.object import ArcObject
from src.perception.detector import PerceptionEngine, infer_background_color
from .matcher import DemonstrationCorrespondence, ObjectCorrespondenceMatcher


@dataclass
class TaskRelationalSummary:
    """Summarizes consistent cross-demonstration transformation rules across an entire ARC task."""
    preserves_dimensions: bool
    scale_factor: Optional[Tuple[float, float]]  # (height_scale, width_scale)
    constant_output_shape: Optional[Tuple[int, int]]
    consistent_color_map: Dict[int, int]
    consistent_movement: Optional[Tuple[int, int]]
    is_pure_recoloring: bool
    is_content_crop: bool
    is_object_filtering: bool
    filtered_color: Optional[int]
    pair_correspondences: List[DemonstrationCorrespondence]
    hypothesized_transformations: List[str]

    @property
    def always_preserves_shape(self) -> bool:
        return self.preserves_dimensions

    @property
    def always_preserves_colors(self) -> bool:
        return len(self.consistent_color_map) == 0

    @property
    def consistent_recolor_rules(self) -> Dict[int, int]:
        return self.consistent_color_map

    @property
    def always_creates_or_destroys(self) -> bool:
        return self.is_object_filtering or self.is_content_crop

    @property
    def motion_vectors(self) -> List[Tuple[int, int]]:
        return [self.consistent_movement] if self.consistent_movement is not None else []

    @property
    def summary(self) -> str:
        return f"hypotheses={self.hypothesized_transformations}, preserves_dims={self.preserves_dimensions}"


# Canonical exported alias
RelationalChangeReport = TaskRelationalSummary


class RelationalChangeAnalyzer:
    """
    Analyzes multiple training demonstration pairs to extract invariant transformation rules.
    """

    def __init__(
        self,
        perception: Optional[PerceptionEngine] = None,
        matcher: Optional[ObjectCorrespondenceMatcher] = None,
    ):
        self.perception = perception or PerceptionEngine()
        self.matcher = matcher or ObjectCorrespondenceMatcher()

    def analyze_task(
        self,
        demonstrations: Union[List[Tuple[ArcGrid, ArcGrid]], List[Any]],
    ) -> TaskRelationalSummary:
        """Unified method accepting either list of (ArcGrid, ArcGrid) tuples or TaskDemonstration instances."""
        pairs: List[Tuple[ArcGrid, ArcGrid]] = []
        for d in demonstrations:
            if isinstance(d, tuple):
                pairs.append(d)
            elif hasattr(d, "input_grid") and hasattr(d, "output_grid"):
                pairs.append((d.input_grid, d.output_grid))
            else:
                raise ValueError(f"Unsupported demonstration format: {type(d)}")
        return self.analyze_task_demonstrations(pairs)

    def analyze_task_demonstrations(
        self,
        train_pairs: List[Tuple[ArcGrid, ArcGrid]]
    ) -> TaskRelationalSummary:
        """
        Analyzes all input/output demonstration pairs for a task to identify invariant rules.
        """
        corrs: List[DemonstrationCorrespondence] = []
        all_same_dim = True
        scale_factors: List[Tuple[float, float]] = []
        output_shapes: List[Tuple[int, int]] = []

        for idx, (inp, out) in enumerate(train_pairs):
            if inp.shape != out.shape:
                all_same_dim = False
            scale_factors.append((out.height / inp.height, out.width / inp.width))
            output_shapes.append(out.shape)

            # Extract objects with 4-connectivity
            bg_in = infer_background_color(inp)
            bg_out = infer_background_color(out)
            objs_in = self.perception.segment_monochromatic_connected_components(inp, bg_in, connectivity=4)
            objs_out = self.perception.segment_monochromatic_connected_components(out, bg_out, connectivity=4)

            corr = self.matcher.match_pair(objs_in, objs_out, inp, out, pair_index=idx)
            corrs.append(corr)

        # 1. Dimension preservation / scaling
        preserves_dims = all_same_dim
        consistent_scale = (
            scale_factors[0]
            if len(scale_factors) > 0 and all(s == scale_factors[0] for s in scale_factors)
            else None
        )
        constant_out_shape = (
            output_shapes[0]
            if len(output_shapes) > 0 and all(sh == output_shapes[0] for sh in output_shapes)
            else None
        )

        # 2. Consistent Color Mapping across all demonstrations
        consistent_color_map: Dict[int, int] = {}
        candidate_maps = [c.color_mappings for c in corrs]
        if candidate_maps:
            # Check for colors that consistently map to the same target in every demonstration where they appear
            all_in_colors = set().union(*[m.keys() for m in candidate_maps])
            for c_in in all_in_colors:
                targets = [m[c_in] for m in candidate_maps if c_in in m]
                if len(targets) > 0 and all(t == targets[0] for t in targets):
                    consistent_color_map[c_in] = targets[0]

        # 3. Consistent Movement
        movements = [c.global_movement for c in corrs if c.global_movement is not None]
        consistent_move = movements[0] if len(movements) == len(corrs) and all(m == movements[0] for m in movements) else None

        # 4. Check for pure recoloring (dimensions preserved, identical object masks, only colors changed)
        pure_recolor = (
            preserves_dims
            and all(
                all(m.match_type in ("exact_identity", "recolored") for m in c.matches if m.output_obj_id is not None)
                for c in corrs
            )
        )

        # 5. Check for content cropping (output equals crop_to_content(input))
        is_content_crop = not preserves_dims and all(
            np.array_equal(
                out.raw,
                inp.crop(bbox).raw
            ) if (bbox := inp.content_bounding_box()) is not None and bbox.height == out.height and bbox.width == out.width
            else False
            for inp, out in train_pairs
        )

        # 6. Check for color-filtering / keep_only_color
        is_filtering = False
        filtered_color = None
        for color in range(1, 10):
            if all(
                set(out.unique_colors) <= {0, color} and set(inp.unique_colors) > {0, color}
                for inp, out in train_pairs
            ):
                is_filtering = True
                filtered_color = color
                break

        # Synthesize rule hypotheses strings
        hypotheses: List[str] = []
        if preserves_dims:
            hypotheses.append("preserve_dimensions")
        if pure_recolor and consistent_color_map:
            hypotheses.append("recolor_mapping")
        if consistent_move:
            hypotheses.append(f"translate_{consistent_move[0]}_{consistent_move[1]}")
        if is_content_crop:
            hypotheses.append("crop_to_content")
        if is_filtering and filtered_color is not None:
            hypotheses.append(f"keep_only_color_{filtered_color}")
        if consistent_scale and consistent_scale[0] == consistent_scale[1] and consistent_scale[0] > 1.0:
            hypotheses.append(f"tile_or_scale_{int(consistent_scale[0])}")

        return TaskRelationalSummary(
            preserves_dimensions=preserves_dims,
            scale_factor=consistent_scale,
            constant_output_shape=constant_out_shape,
            consistent_color_map=consistent_color_map,
            consistent_movement=consistent_move,
            is_pure_recoloring=pure_recolor,
            is_content_crop=is_content_crop,
            is_object_filtering=is_filtering,
            filtered_color=filtered_color,
            pair_correspondences=corrs,
            hypothesized_transformations=hypotheses,
        )
