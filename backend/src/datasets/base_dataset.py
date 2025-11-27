"""
Base Dataset Class for Universal Data Loading
"""

import torch
import torch.utils.data
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


class BaseDataset(torch.utils.data.Dataset):
    """
    Base class for all dataset loaders.
    Each dataset must return a dictionary with:
    {
        "image": preprocessed_image_tensor or numpy array,
        "img_id": sample_id,
        "original_size": (H, W),
        "annotations": [
            {"bbox": [x1, y1, x2, y2], "category_id": int}
        ]
    }
    """
    
    def __init__(self, root_dir: str, transforms: Optional[Any] = None):
        """
        Args:
            root_dir: Root directory of the dataset
            transforms: Optional transform pipeline to apply
        """
        self.root = root_dir
        self.transforms = transforms
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Must be implemented by subclasses.
        Returns a dictionary with image, annotations, etc.
        """
        raise NotImplementedError("Subclasses must implement __getitem__")
    
    def __len__(self) -> int:
        """
        Must be implemented by subclasses.
        Returns the number of samples in the dataset.
        """
        raise NotImplementedError("Subclasses must implement __len__")
    
    def parse_annotations(self, annotation_path: str) -> List[Dict[str, Any]]:
        """
        Helper method to parse annotation files.
        Can be overridden by subclasses for custom formats.
        """
        return []
    
    def get_category_mapping(self) -> Dict[str, int]:
        """
        Returns a mapping from category names to category IDs.
        Can be overridden by subclasses.
        """
        return {}

