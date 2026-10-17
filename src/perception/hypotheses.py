"""
Perception hypotheses container managing alternative segmentations and confidence ranking.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.representation.grid import ArcGrid
from .object import ArcObject
from .detector import PerceptionEngine, infer_background_color


@dataclass
class SegmentationHypothesis:
    """Represents one candidate interpretation / segmentation of an ARC grid."""
    name: str
    description: str
    background_color: int
    objects: List[ArcObject]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_objects(self) -> int:
        return len(self.objects)


class PerceptionHypothesisGenerator:
    """
    Generates ranked alternative perception segmentations for an ARC grid.
    """

    def __init__(self, engine: Optional[PerceptionEngine] = None):
        self.engine = engine or PerceptionEngine()

    def generate_hypotheses(self, grid: ArcGrid) -> List[SegmentationHypothesis]:
        """
        Extracts multiple competing segmentation hypotheses for a grid.

        Hypotheses:
        1. 'mono_4conn': Monochromatic 4-connected components (default canonical ARC prior).
        2. 'mono_8conn': Monochromatic 8-connected components (diagonal touches count as single object).
        3. 'multicolor_4conn': Multi-color contiguous components (adjacent colored blocks form composite object).
        4. 'color_partition': All pixels of same color form a single disjoint region.
        """
        bg = infer_background_color(grid)
        hypotheses: List[SegmentationHypothesis] = []

        # 1. Monochromatic 4-connectivity
        mono_4 = self.engine.segment_monochromatic_connected_components(grid, background_color=bg, connectivity=4)
        hypotheses.append(
            SegmentationHypothesis(
                name="mono_4conn",
                description="Monochromatic 4-connected components (standard ARC prior)",
                background_color=bg,
                objects=mono_4,
                confidence=0.85,
                metadata={"connectivity": 4, "monochromatic": True}
            )
        )

        # 2. Monochromatic 8-connectivity
        mono_8 = self.engine.segment_monochromatic_connected_components(grid, background_color=bg, connectivity=8)
        # Score higher if 8-conn joins diagonally touching elements without collapsing everything
        conf_8 = 0.70 if len(mono_8) != len(mono_4) else 0.50
        hypotheses.append(
            SegmentationHypothesis(
                name="mono_8conn",
                description="Monochromatic 8-connected components (diagonal connectivity included)",
                background_color=bg,
                objects=mono_8,
                confidence=conf_8,
                metadata={"connectivity": 8, "monochromatic": True}
            )
        )

        # 3. Multi-color 4-connectivity
        multi_4 = self.engine.segment_multicolor_connected_components(grid, background_color=bg, connectivity=4)
        # Only relevant if there are multi-color foregrounds and count is non-trivial
        conf_multi = 0.65 if (len(grid.unique_colors) > 2 and len(multi_4) < len(mono_4)) else 0.40
        hypotheses.append(
            SegmentationHypothesis(
                name="multicolor_4conn",
                description="Multi-color contiguous components grouped into compound objects",
                background_color=bg,
                objects=multi_4,
                confidence=conf_multi,
                metadata={"connectivity": 4, "monochromatic": False}
            )
        )

        # 4. Color region partition
        color_part = self.engine.segment_by_color_regions(grid, background_color=bg)
        hypotheses.append(
            SegmentationHypothesis(
                name="color_partition",
                description="Global color masks (each foreground color is one logical entity)",
                background_color=bg,
                objects=color_part,
                confidence=0.55,
                metadata={"type": "color_partition"}
            )
        )

        # Sort hypotheses by confidence descending
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses
