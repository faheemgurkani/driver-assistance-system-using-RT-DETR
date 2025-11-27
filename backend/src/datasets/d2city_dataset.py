"""
D2-City Dataset Loader
Loads MP4 videos and extracts frames
"""

import os
import cv2
import glob
import numpy as np
from typing import Dict, List, Any, Optional
from .base_dataset import BaseDataset


class D2CityDataset(BaseDataset):
    """
    D2-City Dataset loader for MP4 videos.
    Extracts frames from videos at specified intervals.
    
    Expected structure:
        d2_city/
            *.mp4  # video files
    """
    
    def __init__(self, root_dir: str, transforms: Optional[Any] = None, 
                 frame_skip: int = 5, max_frames_per_video: Optional[int] = None):
        """
        Args:
            root_dir: Root directory containing MP4 files
            transforms: Optional transform pipeline
            frame_skip: Extract every Nth frame (default: 5)
            max_frames_per_video: Maximum frames to extract per video (None = all)
        """
        super().__init__(root_dir, transforms)
        self.frame_skip = frame_skip
        self.max_frames_per_video = max_frames_per_video
        
        # Find all video files
        self.video_files = sorted(glob.glob(os.path.join(root_dir, "*.mp4")))
        if not self.video_files:
            # Also check subdirectories
            self.video_files = sorted(glob.glob(os.path.join(root_dir, "**", "*.mp4"), recursive=True))
        
        if not self.video_files:
            raise ValueError(f"No MP4 files found in {root_dir}")
        
        # Build sample list: (video_path, frame_index)
        self.samples = []
        self._build_sample_list()
    
    def _build_sample_list(self):
        """Build list of (video_path, frame_index) tuples."""
        self.samples = []
        
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
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample (frame) from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with image, annotations, etc.
        """
        video_path, frame_idx = self.samples[idx]
        
        # Open video and seek to frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")
        
        # Get original size
        H, W = frame.shape[:2]
        
        # D2-City doesn't have official annotations, return empty list
        # You can use KITTI annotations for training if needed
        sample = {
            "image": frame,
            "annotations": [],  # No annotations available
            "img_id": idx,
            "original_size": (H, W),
            "video_path": video_path,
            "frame_index": frame_idx
        }
        
        # Apply transforms if provided
        if self.transforms:
            sample = self.transforms(sample)
        
        return sample
    
    def __len__(self) -> int:
        """Return number of samples (frames) in dataset."""
        return len(self.samples)

