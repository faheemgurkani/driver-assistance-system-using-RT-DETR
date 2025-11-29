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

# Import datapoints for bounding box format
try:
    from torchvision import datapoints
    HAS_DATAPOINTS = True
except ImportError:
    HAS_DATAPOINTS = False
    # Create a dummy datapoints module for compatibility
    class DummyDatapoints:
        class BoundingBox:
            pass
        class BoundingBoxFormat:
            XYXY = 'xyxy'
            CXCYWH = 'cxcywh'
    datapoints = DummyDatapoints()

# Import annotation parser
try:
    from utils.d2city_annotation_parser import get_annotations_for_image
except ImportError:
    # Fallback: try direct import if utils is not in path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
        from utils.d2city_annotation_parser import get_annotations_for_image
except ImportError:
    get_annotations_for_image = None


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
        return_masks: bool = False,
        annotations_dir: Optional[str] = None
    ):
        """
        Args:
            root_dir: Root directory containing pre-processed saliency-enhanced frames
            transforms: RT-DETR transform pipeline (injected)
            image_extensions: List of image file extensions to search for
                            (default: ['.jpg', '.jpeg', '.png', '.bmp'])
            return_masks: Whether to return masks (not supported for this dataset)
            annotations_dir: Directory containing XML annotation files (optional)
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.return_masks = return_masks
        self.annotations_dir = annotations_dir
        self.use_annotations = annotations_dir is not None and get_annotations_for_image is not None
        
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
        
        # Safety check: Ensure all images are from enhanced_frames directory
        # This prevents accidentally loading from frames/ or saliency_masks/
        invalid_paths = []
        for path in self.image_files:
            # Normalize path for comparison
            normalized_path = os.path.normpath(path)
            normalized_root = os.path.normpath(self.root_dir)
            
            # Check if path is within root_dir
            if not normalized_path.startswith(normalized_root):
                invalid_paths.append(path)
            # Additional check: ensure we're not accidentally in frames/ or saliency_masks/
            path_parts = normalized_path.split(os.sep)
            if 'frames' in path_parts and 'enhanced_frames' not in path_parts:
                invalid_paths.append(path)
            if 'saliency_masks' in path_parts:
                invalid_paths.append(path)
        
        if invalid_paths:
            raise ValueError(
                f"Invalid image paths detected! All images must be from enhanced_frames directory. "
                f"Found {len(invalid_paths)} invalid paths. Examples: {invalid_paths[:3]}"
            )
    
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
        
        # Load annotations if available
        if self.use_annotations:
            # Extract video ID from image path
            # Path format: root_dir/video_id/frame_xxxxx.jpg
            # root_dir is typically: ./data/T2-saliency/enhanced_frames
            # So image_path is: ./data/T2-saliency/enhanced_frames/video_id/frame_xxxxx.jpg
            normalized_path = os.path.normpath(image_path)
            normalized_root = os.path.normpath(self.root_dir)
            
            # Get relative path from root_dir
            if normalized_path.startswith(normalized_root):
                relative_path = os.path.relpath(normalized_path, normalized_root)
                # Split path components: video_id/frame_xxxxx.jpg
                path_parts = relative_path.split(os.sep)
                if len(path_parts) >= 2:
                    video_id = path_parts[0]  # First component is video_id
                    image_filename = path_parts[-1]  # Last component is filename
                else:
                    # Fallback: try to extract from directory structure
                    image_dir = os.path.dirname(image_path)
                    video_id = os.path.basename(image_dir) if os.path.basename(image_dir) else os.path.basename(os.path.dirname(image_dir))
                    image_filename = os.path.basename(image_path)
            else:
                # Fallback: extract from directory structure
            image_dir = os.path.dirname(image_path)
            video_id = os.path.basename(image_dir) if os.path.basename(image_dir) else os.path.basename(os.path.dirname(image_dir))
            image_filename = os.path.basename(image_path)
            
            xml_path = os.path.join(self.annotations_dir, f"{video_id}.xml")
            
            # Check if XML file exists before trying to load
            if os.path.exists(xml_path):
                try:
            boxes, labels = get_annotations_for_image(xml_path, image_filename, w, h)
                    # Validate boxes and labels have matching dimensions
                    if len(boxes) != len(labels):
                        # If mismatch, return empty annotations
                        boxes = torch.zeros((0, 4), dtype=torch.float32)
                        labels = torch.zeros((0,), dtype=torch.int64)
                except Exception as e:
                    # If loading fails, return empty annotations
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    labels = torch.zeros((0,), dtype=torch.int64)
            else:
                # If XML doesn't exist, return empty annotations
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # Calculate area for each bounding box (required for COCO API)
        # Area = width * height of each box
        if boxes.numel() > 0:
            # boxes format: [x1, y1, x2, y2]
            box_widths = boxes[:, 2] - boxes[:, 0]
            box_heights = boxes[:, 3] - boxes[:, 1]
            areas = box_widths * box_heights
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        
        # Convert boxes to datapoints format if available (required by torchvision transforms)
        # Even empty boxes need to be in datapoints format for transforms to work
        if HAS_DATAPOINTS:
            if boxes.numel() > 0:
                boxes_dp = datapoints.BoundingBox(
                    boxes,
                    format=datapoints.BoundingBoxFormat.XYXY,
                    spatial_size=(h, w)
                )
            else:
                # Create empty BoundingBox datapoint
                boxes_dp = datapoints.BoundingBox(
                    boxes,  # Empty tensor (0, 4)
                    format=datapoints.BoundingBoxFormat.XYXY,
                    spatial_size=(h, w)
                )
        else:
            boxes_dp = boxes
        
        target = {
            'boxes': boxes_dp,
            'labels': labels,
            'area': areas,
            'iscrowd': iscrowd,
            'image_id': torch.tensor([self.ids[idx]], dtype=torch.int64),
            'orig_size': torch.tensor([h, w], dtype=torch.int64),
            'size': torch.tensor([h, w], dtype=torch.int64),
        }
        
        # Apply RT-DETR transforms (resize, tensor conversion, etc.)
        # IMPORTANT: These frames are already saliency-enhanced (pre-processed),
        # so we SKIP all preprocessing steps and only apply RT-DETR's standard
        # transforms (resize, normalization, tensor conversion).
        # This is the key difference from the original D2-City branch which
        # applies preprocessing before transfer learning.
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

