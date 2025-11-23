"""
Saliency-Enhanced D2-City Dataset Loader for RT-DETR

This dataset loader loads PRE-PROCESSED saliency-enhanced frames directly.
These frames have already gone through:
1. Input acquisition (D2-City videos)
2. Basic preprocessing
3. Saliency model processing (saliency mask generation)
4. Enhanced frame creation (Frame × Saliency Mask)

This loader skips all preprocessing steps and loads the enhanced frames directly.
Use this when working with a pre-processed saliency-enhanced D2-City dataset.
"""

import os
import torch
import torch.utils.data
from PIL import Image
import glob
from typing import Dict, List, Any, Optional

# Import RT-DETR registration system (local copy)
import sys
# From backend/src/datasets/ -> backend/src/rtdetr/
rtdetr_path = os.path.join(os.path.dirname(__file__), '..', 'rtdetr')
if os.path.exists(rtdetr_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.rtdetr.core import register
else:
    def register(cls):
        return cls


@register
class SaliencyEnhancedD2CityDatasetRTDETR(torch.utils.data.Dataset):
    """
    Saliency-Enhanced D2-City Dataset loader for RT-DETR.
    
    Loads PRE-PROCESSED saliency-enhanced frames directly from disk.
    These frames are already enhanced (Original Frame × Saliency Mask).
    
    Expected dataset structure:
    root_dir/
        frame_00001.jpg  (or .png)
        frame_00002.jpg
        frame_00003.jpg
        ...
    
    Or organized in subdirectories:
    root_dir/
        video_001/
            frame_00001.jpg
            frame_00002.jpg
            ...
        video_002/
            frame_00001.jpg
            ...
    
    Returns:
        (PIL.Image, target_dict) where target_dict has empty annotations
        (compatible with RT-DETR training/inference)
    """
    __inject__ = ['transforms']
    
    def __init__(
        self, 
        root_dir: str, 
        transforms=None, 
        image_extensions: List[str] = None,
        return_masks: bool = False
    ):
        """
        Args:
            root_dir: Root directory containing pre-processed saliency-enhanced frames
            transforms: RT-DETR transform pipeline (injected)
            image_extensions: List of image file extensions to search for
                            (default: ['.jpg', '.jpeg', '.png', '.bmp'])
            return_masks: Whether to return masks (not supported for this dataset)
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.return_masks = return_masks
        
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
        self.image_extensions = image_extensions
        
        # Find all image files
        self.image_files = []
        self._build_image_list()
        
        if not self.image_files:
            raise ValueError(
                f"No image files found in {root_dir}. "
                f"Looking for extensions: {image_extensions}"
            )
        
        self.ids = list(range(len(self.image_files)))
    
    def _build_image_list(self):
        """Build list of image file paths."""
        # Try flat structure first (all images in root_dir)
        for ext in self.image_extensions:
            pattern = os.path.join(self.root_dir, f"*{ext}")
            files = sorted(glob.glob(pattern))
            self.image_files.extend(files)
        
        # If no files found, try recursive search in subdirectories
        if not self.image_files:
            for ext in self.image_extensions:
                pattern = os.path.join(self.root_dir, "**", f"*{ext}")
                files = sorted(glob.glob(pattern, recursive=True))
                self.image_files.extend(files)
        
        # Remove duplicates and sort
        self.image_files = sorted(list(set(self.image_files)))
    
    def __getitem__(self, idx: int):
        """
        Get a sample (pre-processed saliency-enhanced frame) from the dataset.
        
        Returns:
            (PIL.Image, target_dict) compatible with RT-DETR
        """
        image_path = self.image_files[idx]
        
        # Load pre-processed saliency-enhanced frame
        # These frames are already:
        # - Preprocessed (RGB, normalized, etc.)
        # - Enhanced with saliency mask (Frame × Saliency Mask)
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")
        
        w, h = img.size
        
        # D2-City doesn't have annotations (or annotations would be in separate files)
        target = {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64),
            'image_id': torch.tensor([self.ids[idx]], dtype=torch.int64),
            'orig_size': torch.tensor([h, w], dtype=torch.int64),
            'size': torch.tensor([h, w], dtype=torch.int64),
        }
        
        # Apply RT-DETR transforms (resize, tensor conversion, etc.)
        # Note: These frames are already saliency-enhanced, so we only need
        # RT-DETR's standard transforms (resize, normalization, tensor conversion)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.image_files)
    
    def extra_repr(self) -> str:
        s = f' root_dir: {self.root_dir}\n'
        s += f' num_images: {len(self.image_files)}\n'
        s += f' image_extensions: {self.image_extensions}\n'
        return s

