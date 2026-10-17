"""
Object-Centric Perception Package for ARC-AGI-2.
"""

from .object import ArcObject
from .detector import PerceptionEngine, infer_background_color, classify_shape, compute_perimeter
from .hypotheses import SegmentationHypothesis, PerceptionHypothesisGenerator

__all__ = [
    "ArcObject",
    "PerceptionEngine",
    "infer_background_color",
    "classify_shape",
    "compute_perimeter",
    "SegmentationHypothesis",
    "PerceptionHypothesisGenerator",
]
