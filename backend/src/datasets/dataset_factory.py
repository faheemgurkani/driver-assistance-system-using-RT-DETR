"""
Dataset Factory for D2-City Dataset
"""

import os
import torch.utils.data
from typing import Optional, Type
from .d2city_dataset_rtdetr import D2CityDatasetRTDETR


class DatasetFactory:
    """
    Factory class for D2-City dataset loader.
    """
    
    # Supported dataset types (only D2-City)
    DATASET_TYPES = {
        'd2_city': D2CityDatasetRTDETR,
        'd2city': D2CityDatasetRTDETR,
    }
    
    @classmethod
    def detect_dataset_type(cls, root_dir: str) -> Optional[str]:
        """
        Auto-detect D2-City dataset type based on folder structure.
        
        Args:
            root_dir: Root directory of the dataset
            
        Returns:
            Dataset type string or None if cannot detect
        """
        if not os.path.exists(root_dir):
            return None
        
        # Check for D2-City (MP4 videos)
        mp4_files = [f for f in os.listdir(root_dir) if f.endswith('.mp4')]
        if mp4_files:
            return "d2_city"
        
        return None
    
    @classmethod
    def create(cls, dataset_name: Optional[str] = None, root_dir: Optional[str] = None) -> Type[torch.utils.data.Dataset]:
        """
        Create D2-City dataset class.
        
        Args:
            dataset_name: Explicit dataset name ('d2_city' or 'd2city')
            root_dir: Root directory for auto-detection
            
        Returns:
            Dataset class (not instance)
            
        Raises:
            ValueError: If dataset type cannot be determined
        """
        # If explicit name provided, use it
        if dataset_name:
            dataset_name = dataset_name.lower()
            if dataset_name in cls.DATASET_TYPES:
                return cls.DATASET_TYPES[dataset_name]
            else:
                raise ValueError(
                    f"Unknown dataset type: {dataset_name}. "
                    f"Supported types: {list(cls.DATASET_TYPES.keys())}"
                )
        
        # Try auto-detection
        if root_dir:
            detected_type = cls.detect_dataset_type(root_dir)
            if detected_type:
                return cls.DATASET_TYPES[detected_type]
        
        raise ValueError(
            "Cannot determine dataset type. "
            "Please provide dataset_name='d2_city' or ensure root_dir contains MP4 video files."
        )
    
    @classmethod
    def list_supported_datasets(cls) -> list:
        """Return list of supported dataset types."""
        return list(cls.DATASET_TYPES.keys())
