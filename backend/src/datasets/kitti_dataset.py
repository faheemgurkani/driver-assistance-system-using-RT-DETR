"""
KITTI Dataset Loader
Loads from training/image_2/*.png and training/label_2/*.txt
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from .base_dataset import BaseDataset


class KITTIDataset(BaseDataset):
    """
    KITTI Dataset loader.
    Expected structure:
        kitti/
            training/
                image_2/  # images
                label_2/  # annotations
    """
    
    # KITTI category mapping (common classes for autonomous driving)
    KITTI_CATEGORIES = {
        'Car': 0,
        'Van': 1,
        'Truck': 2,
        'Pedestrian': 3,
        'Person_sitting': 4,
        'Cyclist': 5,
        'Tram': 6,
        'Misc': 7,
        'DontCare': -1,  # Ignore these
    }
    
    def __init__(self, root_dir: str, transforms: Optional[Any] = None, split: str = 'training'):
        """
        Args:
            root_dir: Root directory of KITTI dataset
            transforms: Optional transform pipeline
            split: Dataset split ('training' or 'testing')
        """
        super().__init__(root_dir, transforms)
        self.split = split
        
        self.img_dir = os.path.join(root_dir, split, "image_2")
        self.label_dir = os.path.join(root_dir, split, "label_2")
        
        if not os.path.exists(self.img_dir):
            raise ValueError(f"Image directory not found: {self.img_dir}")
        
        # Get all image files
        self.images = sorted([f for f in os.listdir(self.img_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not self.images:
            raise ValueError(f"No images found in {self.img_dir}")
    
    def parse_kitti_label(self, label_path: str) -> List[Dict[str, Any]]:
        """
        Parse KITTI label file format:
        type truncated occluded alpha bbox_x1 bbox_y1 bbox_x2 bbox_y2 dimensions_3d location_3d rotation_y score
        
        Args:
            label_path: Path to label file
            
        Returns:
            List of annotation dictionaries
        """
        boxes = []
        
        if not os.path.exists(label_path):
            return boxes
        
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 15:
                    continue
                
                cls_name = parts[0]
                
                # Skip DontCare and other ignored classes
                if cls_name == 'DontCare' or cls_name not in self.KITTI_CATEGORIES:
                    continue
                
                # Extract bounding box (x1, y1, x2, y2)
                x1 = float(parts[4])
                y1 = float(parts[5])
                x2 = float(parts[6])
                y2 = float(parts[7])
                
                # Get category ID
                category_id = self.KITTI_CATEGORIES[cls_name]
                
                boxes.append({
                    "bbox": [x1, y1, x2, y2],
                    "category_id": category_id,
                    "class_name": cls_name
                })
        
        return boxes
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with image, annotations, etc.
        """
        fname = self.images[idx]
        img_path = os.path.join(self.img_dir, fname)
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Get original size
        H, W = img.shape[:2]
        
        # Load annotations
        label_fname = fname.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
        label_path = os.path.join(self.label_dir, label_fname)
        annotations = self.parse_kitti_label(label_path)
        
        # Create sample dictionary
        sample = {
            "image": img,
            "annotations": annotations,
            "img_id": idx,
            "original_size": (H, W),
            "image_path": img_path
        }
        
        # Apply transforms if provided
        if self.transforms:
            sample = self.transforms(sample)
        
        return sample
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.images)
    
    def get_category_mapping(self) -> Dict[str, int]:
        """Return KITTI category mapping."""
        return self.KITTI_CATEGORIES.copy()

