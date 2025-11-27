"""
KITTI Dataset Loader for RT-DETR
Registered with RT-DETR's system, returns PIL Images and proper target format
Based on actual KITTI format: https://www.cvlibs.net/datasets/kitti/
"""

import os
import torch
import torch.utils.data
from PIL import Image
import numpy as np
from typing import Dict, List, Any, Optional

# Import RT-DETR registration system
import sys
# From backend/src/datasets/ -> backend/ -> project_root/ -> rtdetr_pytorch/
rtdetr_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'rtdetr_pytorch')
if os.path.exists(rtdetr_path):
    sys.path.insert(0, rtdetr_path)
    from src.core import register
else:
    # Fallback decorator if RT-DETR not available
    def register(cls):
        return cls


@register
class KITTIDatasetRTDETR(torch.utils.data.Dataset):
    """
    KITTI Dataset loader compatible with RT-DETR.
    
    KITTI format (from https://www.cvlibs.net/datasets/kitti/):
    - Images: training/image_2/*.png
    - Labels: training/label_2/*.txt
    - Label format: type truncated occluded alpha bbox_x1 bbox_y1 bbox_x2 bbox_y2 dimensions_3d location_3d rotation_y score
    
    Returns:
        (PIL.Image, target_dict) where target_dict has:
        - 'boxes': tensor(N, 4) in xyxy format
        - 'labels': tensor(N,) with class IDs
        - 'image_id': tensor([image_id])
        - 'orig_size': tensor([H, W])
    """
    __inject__ = ['transforms']
    
    # KITTI category mapping to COCO-like format
    # KITTI has: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc, DontCare
    KITTI_CATEGORIES = {
        'Car': 0,
        'Van': 1,
        'Truck': 2,
        'Pedestrian': 3,
        'Person_sitting': 3,  # Map to Pedestrian
        'Cyclist': 4,
        'Tram': 5,
        'Misc': 6,
        'DontCare': -1,  # Ignore
    }
    
    def __init__(self, root_dir: str, transforms=None, split: str = 'training', 
                 return_masks: bool = False):
        """
        Args:
            root_dir: Root directory of KITTI dataset
            transforms: RT-DETR transform pipeline (injected)
            split: Dataset split ('training' or 'testing')
            return_masks: Whether to return masks (not supported for KITTI)
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.split = split
        self.return_masks = return_masks
        
        self.img_dir = os.path.join(root_dir, split, "image_2")
        self.label_dir = os.path.join(root_dir, split, "label_2")
        
        if not os.path.exists(self.img_dir):
            raise ValueError(f"Image directory not found: {self.img_dir}")
        
        # Get all image files
        self.images = sorted([f for f in os.listdir(self.img_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not self.images:
            raise ValueError(f"No images found in {self.img_dir}")
        
        self.ids = list(range(len(self.images)))
    
    def parse_kitti_label(self, label_path: str) -> tuple:
        """
        Parse KITTI label file.
        
        Returns:
            (boxes, labels) where:
            - boxes: list of [x1, y1, x2, y2]
            - labels: list of category IDs
        """
        boxes = []
        labels = []
        
        if not os.path.exists(label_path):
            return boxes, labels
        
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 15:
                    continue
                
                cls_name = parts[0]
                
                # Skip DontCare
                if cls_name == 'DontCare' or cls_name not in self.KITTI_CATEGORIES:
                    continue
                
                # Extract bounding box (x1, y1, x2, y2) - already in image coordinates
                x1 = float(parts[4])
                y1 = float(parts[5])
                x2 = float(parts[6])
                y2 = float(parts[7])
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Get category ID
                category_id = self.KITTI_CATEGORIES[cls_name]
                
                boxes.append([x1, y1, x2, y2])
                labels.append(category_id)
        
        return boxes, labels
    
    def __getitem__(self, idx: int):
        """
        Get a sample.
        
        Returns:
            (PIL.Image, target_dict) compatible with RT-DETR
        """
        fname = self.images[idx]
        img_path = os.path.join(self.img_dir, fname)
        
        # Load image as PIL Image (RT-DETR expects PIL)
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        # Load annotations
        label_fname = fname.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
        label_path = os.path.join(self.label_dir, label_fname)
        boxes_list, labels_list = self.parse_kitti_label(label_path)
        
        # Convert to tensors
        if boxes_list:
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
            labels = torch.tensor(labels_list, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # Create target dict (RT-DETR format)
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([self.ids[idx]], dtype=torch.int64),
            'orig_size': torch.tensor([h, w], dtype=torch.int64),
            'size': torch.tensor([h, w], dtype=torch.int64),
        }
        
        # Apply transforms if provided
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.images)
    
    def extra_repr(self) -> str:
        s = f' root_dir: {self.root_dir}\n split: {self.split}\n'
        s += f' num_images: {len(self.images)}\n'
        if hasattr(self, 'transforms') and self.transforms is not None:
            s += f' transforms:\n   {repr(self.transforms)}'
        return s

