"""
Saliency Integration Module
Provides utilities for integrating saliency model outputs with RT-DETR input pipeline.

This module will be used when the saliency model (from Hammad/Hunain) is ready.
"""

import numpy as np
from typing import Optional, Callable
from PIL import Image


def apply_saliency_enhancement(
    frame: np.ndarray, 
    saliency_mask: np.ndarray
) -> np.ndarray:
    """
    Apply saliency mask to frame using element-wise multiplication.
    
    Formula: Enhanced Frame = Original Frame × Saliency Mask
    
    Args:
        frame: Original frame (H × W × 3) in RGB format, uint8 [0, 255]
        saliency_mask: Saliency mask (H × W) with values in [0.0, 1.0]
                      - 0.0 → background/unimportant
                      - 1.0 → highly important region
                      - Values in-between → partially important
    
    Returns:
        Enhanced frame (H × W × 3) in RGB format, uint8 [0, 255]
        - Important regions (cars, road, pedestrians) stay bright
        - Background regions are dimmed/suppressed
    """
    # Ensure mask is 2D
    if len(saliency_mask.shape) == 3:
        # If mask is (H, W, 1), squeeze it
        saliency_mask = saliency_mask.squeeze()
    
    # Ensure mask values are in [0, 1]
    saliency_mask = np.clip(saliency_mask, 0.0, 1.0)
    
    # Expand mask to 3 channels for element-wise multiplication
    if len(saliency_mask.shape) == 2:
        mask_3d = np.stack([saliency_mask] * 3, axis=2)
    else:
        mask_3d = saliency_mask
    
    # Ensure frame and mask have same dimensions
    if frame.shape[:2] != saliency_mask.shape[:2]:
        raise ValueError(
            f"Frame shape {frame.shape[:2]} does not match "
            f"saliency mask shape {saliency_mask.shape[:2]}"
        )
    
    # Convert frame to float for multiplication
    frame_float = frame.astype(np.float32)
    
    # Element-wise multiplication: Enhanced Frame = Frame × Mask
    enhanced_frame = frame_float * mask_3d
    
    # Clip to valid range and convert back to uint8
    enhanced_frame = np.clip(enhanced_frame, 0, 255).astype(np.uint8)
    
    return enhanced_frame


def process_frame_with_saliency(
    frame: np.ndarray,
    saliency_model: Optional[Callable] = None,
    saliency_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Process a frame with saliency enhancement.
    
    This function can be used in two ways:
    1. Provide a saliency_model function that generates the mask
    2. Provide a pre-computed saliency_mask
    
    Args:
        frame: Original frame (H × W × 3) in RGB format
        saliency_model: Optional function that takes frame and returns saliency mask
                       Function signature: saliency_mask = saliency_model(frame)
        saliency_mask: Optional pre-computed saliency mask (H × W)
    
    Returns:
        Enhanced frame (H × W × 3) in RGB format
    """
    # If no saliency model or mask provided, return original frame
    if saliency_model is None and saliency_mask is None:
        return frame
    
    # Generate saliency mask if model is provided
    if saliency_mask is None and saliency_model is not None:
        saliency_mask = saliency_model(frame)
    
    # Apply saliency enhancement
    if saliency_mask is not None:
        enhanced_frame = apply_saliency_enhancement(frame, saliency_mask)
        return enhanced_frame
    
    return frame


def frame_to_pil_with_saliency(
    frame: np.ndarray,
    saliency_model: Optional[Callable] = None,
    saliency_mask: Optional[np.ndarray] = None
) -> Image.Image:
    """
    Convert frame to PIL Image with optional saliency enhancement.
    
    This is a convenience function that combines saliency processing
    and PIL Image conversion.
    
    Args:
        frame: Original frame (H × W × 3) in RGB format
        saliency_model: Optional saliency model function
        saliency_mask: Optional pre-computed saliency mask
    
    Returns:
        PIL Image (enhanced if saliency is applied, original otherwise)
    """
    enhanced_frame = process_frame_with_saliency(
        frame, saliency_model, saliency_mask
    )
    return Image.fromarray(enhanced_frame)


# Example usage (commented out until saliency model is available):
"""
# Example 1: Using a saliency model function
def my_saliency_model(frame: np.ndarray) -> np.ndarray:
    # This function will be provided by Hammad/Hunain
    # Returns saliency mask (H × W) with values in [0, 1]
    pass

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
enhanced_frame = process_frame_with_saliency(frame_rgb, saliency_model=my_saliency_model)
img = Image.fromarray(enhanced_frame)

# Example 2: Using pre-computed saliency mask
saliency_mask = load_saliency_mask_from_file("mask.npy")
enhanced_frame = process_frame_with_saliency(frame_rgb, saliency_mask=saliency_mask)
img = Image.fromarray(enhanced_frame)

# Example 3: Direct enhancement
enhanced_frame = apply_saliency_enhancement(frame_rgb, saliency_mask)
img = Image.fromarray(enhanced_frame)
"""

