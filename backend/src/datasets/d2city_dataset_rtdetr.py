"""
D2-City Dataset Loader for RT-DETR
Registered with RT-DETR's system
Based on D2-City dataset from SciDB: https://www.scidb.cn/en/detail?dataSetId=804399692560465920
"""

import os
import torch
import torch.utils.data
from PIL import Image
import cv2
import glob
import numpy as np
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
class D2CityDatasetRTDETR(torch.utils.data.Dataset):
    """
    D2-City Dataset loader for RT-DETR.
    
    D2-City format:
    - MP4 video files
    - No official annotations (use for inference or transfer learning)
    - Frames extracted at specified intervals
    
    Returns:
        (PIL.Image, target_dict) where target_dict has empty annotations
    """
    __inject__ = ['transforms']
    
    def __init__(self, root_dir: str, transforms=None, frame_skip: int = 5,
                 max_frames_per_video: Optional[int] = None, return_masks: bool = False):
        """
        Args:
            root_dir: Root directory containing MP4 files
            transforms: RT-DETR transform pipeline (injected)
            frame_skip: Extract every Nth frame
            max_frames_per_video: Maximum frames per video
            return_masks: Whether to return masks (not supported)
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.frame_skip = frame_skip
        self.max_frames_per_video = max_frames_per_video
        self.return_masks = return_masks
        
        # Find all video files
        self.video_files = sorted(glob.glob(os.path.join(root_dir, "*.mp4")))
        if not self.video_files:
            self.video_files = sorted(glob.glob(os.path.join(root_dir, "**", "*.mp4"), recursive=True))
        
        if not self.video_files:
            raise ValueError(f"No MP4 files found in {root_dir}")
        
        # Build sample list: (video_path, frame_index)
        self.samples = []
        self._build_sample_list()
        
        self.ids = list(range(len(self.samples)))
    
    def _build_sample_list(self):
        """Build list of (video_path, frame_index) tuples."""
        for video_path in self.video_files:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}")
                continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            frame_indices = list(range(0, total_frames, self.frame_skip))
            
            if self.max_frames_per_video:
                frame_indices = frame_indices[:self.max_frames_per_video]
            
            for fidx in frame_indices:
                self.samples.append((video_path, fidx))
    
    def __getitem__(self, idx: int):
        """
        Get a sample (frame) from the dataset.
        
        Returns:
            (PIL.Image, target_dict) compatible with RT-DETR
        """
        video_path, frame_idx = self.samples[idx]
        
        # Open video and seek to frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")
        
        # Convert BGR to RGB and then to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        w, h = img.size
        
        # D2-City doesn't have annotations
        target = {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64),
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
        return len(self.samples)
    
    def extra_repr(self) -> str:
        s = f' root_dir: {self.root_dir}\n'
        s += f' num_videos: {len(self.video_files)}\n'
        s += f' num_samples: {len(self.samples)}\n'
        s += f' frame_skip: {self.frame_skip}\n'
        return s

