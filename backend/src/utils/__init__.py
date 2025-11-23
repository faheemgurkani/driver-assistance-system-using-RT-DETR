"""
Utility modules for driver assistance system.
"""

from .saliency_integration import (
    apply_saliency_enhancement,
    process_frame_with_saliency,
    frame_to_pil_with_saliency,
)

__all__ = [
    'apply_saliency_enhancement',
    'process_frame_with_saliency',
    'frame_to_pil_with_saliency',
]

