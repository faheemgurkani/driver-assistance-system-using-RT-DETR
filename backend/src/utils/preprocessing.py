"""
Universal Data Preprocessing Module
Converts images to RT-DETR compatible format
"""

import cv2
import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional


class PreprocessTransforms:
    """
    Universal preprocessing pipeline for RT-DETR.
    Converts images to:
    - RGB format
    - Resized to target size (default 640x640)
    - Normalized with ImageNet statistics
    - PyTorch tensor (C×H×W format)
    """
    
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        """
        Args:
            target_size: Target image size (width, height). Default: (640, 640)
        """
        self.target_size = target_size
        # ImageNet normalization statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply preprocessing to a sample.
        
        Args:
            sample: Dictionary with keys:
                - "image": numpy array (BGR or RGB)
                - "annotations": list of annotation dicts
                - "original_size": (H, W) tuple
                - other metadata
        
        Returns:
            Preprocessed sample dictionary with:
                - "image": torch.Tensor (C×H×W, normalized)
                - "annotations": adjusted annotations (if any)
                - "original_size": original (H, W)
                - other metadata preserved
        """
        img = sample["image"].copy()
        original_size = sample["original_size"]  # (H, W)
        annotations = sample.get("annotations", [])
        
        # Convert BGR to RGB (OpenCV loads as BGR)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get original dimensions
        H_orig, W_orig = original_size
        
        # Resize image
        img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to float32 and normalize to [0, 1]
        img_float = img_resized.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        # Shape: (H, W, C) -> normalize per channel
        img_normalized = (img_float - self.mean) / self.std
        
        # Convert to PyTorch tensor and permute to C×H×W
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)
        
        # Adjust annotations if present
        if annotations:
            annotations = self._adjust_annotations(
                annotations, 
                (W_orig, H_orig),  # original (W, H)
                self.target_size   # target (W, H)
            )
        
        # Update sample
        sample["image"] = img_tensor
        sample["annotations"] = annotations
        
        return sample
    
    def _adjust_annotations(self, annotations: list, 
                           original_size: Tuple[int, int],
                           target_size: Tuple[int, int]) -> list:
        """
        Adjust bounding box coordinates after resizing.
        
        Args:
            annotations: List of annotation dicts with "bbox" key
            original_size: Original image size (W, H)
            target_size: Target image size (W, H)
        
        Returns:
            Adjusted annotations
        """
        W_orig, H_orig = original_size
        W_target, H_target = target_size
        
        # Calculate scale factors
        scale_x = W_target / W_orig
        scale_y = H_target / H_orig
        
        adjusted_annotations = []
        for ann in annotations:
            bbox = ann["bbox"]  # [x1, y1, x2, y2]
            
            # Scale bounding boxes
            x1 = bbox[0] * scale_x
            y1 = bbox[1] * scale_y
            x2 = bbox[2] * scale_x
            y2 = bbox[3] * scale_y
            
            # Clamp to image boundaries
            x1 = max(0, min(x1, W_target))
            y1 = max(0, min(y1, H_target))
            x2 = max(0, min(x2, W_target))
            y2 = max(0, min(y2, H_target))
            
            # Only keep valid boxes
            if x2 > x1 and y2 > y1:
                adjusted_ann = ann.copy()
                adjusted_ann["bbox"] = [x1, y1, x2, y2]
                adjusted_annotations.append(adjusted_ann)
        
        return adjusted_annotations

