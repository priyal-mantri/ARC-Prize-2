"""
Structural Fingerprint Package for ARC-AGI-2.
"""

from .encoder import (
    FINGERPRINT_DIM_V1,
    CURRENT_SCHEMA_VERSION,
    grid_to_barcode,
    task_to_fingerprint,
)

__all__ = [
    "FINGERPRINT_DIM_V1",
    "CURRENT_SCHEMA_VERSION",
    "grid_to_barcode",
    "task_to_fingerprint",
]
