"""
RT-DETR Format Conversion Utilities
Converts our universal dataset format to RT-DETR expected format
"""

import torch
import torch.utils.data
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# Add RT-DETR path to sys.path
# From backend/src/utils/ -> backend/ -> project_root/ -> rtdetr_pytorch/
RTDETR_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'rtdetr_pytorch')
if os.path.exists(RTDETR_PATH):
    sys.path.insert(0, RTDETR_PATH)

from src.utils.preprocessing import PreprocessTransforms


def convert_to_rtdetr_format(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert our universal sample format to RT-DETR expected format.
    
    Our format:
    {
        "image": tensor(C, H, W),
        "annotations": [{"bbox": [x1, y1, x2, y2], "category_id": int}],
        "original_size": (H, W),
        "img_id": int
    }
    
    RT-DETR format:
    {
        "img": tensor(C, H, W),
        "gt_boxes": tensor(N, 4) in xyxy format,
        "gt_class": tensor(N,),
        "orig_size": tensor([H, W])
    }
    
    Args:
        sample: Sample in our universal format
        
    Returns:
        Sample in RT-DETR format
    """
    img = sample["image"]
    H, W = sample["original_size"]
    
    # Extract boxes and classes from annotations
    boxes = []
    classes = []
    
    for ann in sample.get("annotations", []):
        bbox = ann["bbox"]  # [x1, y1, x2, y2]
        category_id = ann["category_id"]
        
        boxes.append(bbox)
        classes.append(category_id)
    
    # Convert to tensors
    if boxes:
        gt_boxes = torch.tensor(boxes, dtype=torch.float32)
        gt_class = torch.tensor(classes, dtype=torch.int64)
    else:
        # Empty annotations
        gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
        gt_class = torch.zeros((0,), dtype=torch.int64)
    
    return {
        "img": img,
        "gt_boxes": gt_boxes,
        "gt_class": gt_class,
        "orig_size": torch.tensor([H, W], dtype=torch.int64),
        "image_id": torch.tensor([sample.get("img_id", 0)], dtype=torch.int64)
    }


def rtdetr_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for RT-DETR DataLoader.
    Batches multiple samples together.
    
    Args:
        batch: List of samples in RT-DETR format
        
    Returns:
        Batched dictionary
    """
    # Stack images
    images = torch.stack([item["img"] for item in batch], dim=0)
    
    # Collect targets (list of dicts, as RT-DETR expects)
    targets = []
    for item in batch:
        target = {
            "boxes": item["gt_boxes"],
            "labels": item["gt_class"],
            "orig_size": item["orig_size"],
            "image_id": item["image_id"]
        }
        targets.append(target)
    
    return images, targets


def build_dataloader(dataset_name: Optional[str] = None,
                    root_dir: Optional[str] = None,
                    batch_size: int = 8,
                    num_workers: int = 4,
                    shuffle: bool = True,
                    transforms: Optional[PreprocessTransforms] = None,
                    **dataset_kwargs) -> torch.utils.data.DataLoader:
    """
    Build a DataLoader for RT-DETR training.
    
    Args:
        dataset_name: Explicit dataset name (kitti, d2_city, etc.) or None for auto-detect
        root_dir: Root directory of the dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the dataset
        transforms: Optional custom transforms (default: PreprocessTransforms with 640x640)
        **dataset_kwargs: Additional arguments to pass to dataset constructor
    
    Returns:
        DataLoader that yields (images, targets) tuples compatible with RT-DETR
    """
    # Import here to avoid circular imports
    from src.datasets import DatasetFactory
    
    if root_dir is None:
        raise ValueError("root_dir must be provided")
    
    # Create transforms if not provided
    if transforms is None:
        transforms = PreprocessTransforms(target_size=(640, 640))
    
    # Get dataset class from factory
    dataset_class = DatasetFactory.create(dataset_name=dataset_name, root_dir=root_dir)
    
    # Create dataset instance
    dataset = dataset_class(root_dir=root_dir, transforms=transforms, **dataset_kwargs)
    
    # Create DataLoader with custom collate function
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: rtdetr_collate_fn([convert_to_rtdetr_format(s) for s in batch]),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader

