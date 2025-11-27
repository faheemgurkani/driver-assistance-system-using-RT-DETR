"""
Kaggle Dashcam Dataset Loader
Similar to D2-City but for Kaggle dashcam datasets
"""

import os
import cv2
import glob
import numpy as np
from typing import Dict, List, Any, Optional
from .base_dataset import BaseDataset


class KaggleDashcamDataset(BaseDataset):
    """
    Kaggle Dashcam Dataset loader.
    Handles various Kaggle dashcam dataset structures.
    
    Expected structures:
        kaggle_dashcam/
            train/
                *.mp4 or *.jpg
            test/
                *.mp4 or *.jpg
    """
    
    def __init__(self, root_dir: str, transforms: Optional[Any] = None,
                 split: str = 'train', frame_skip: int = 5,
                 max_frames_per_video: Optional[int] = None):
        """
        Args:
            root_dir: Root directory of Kaggle dataset
            transforms: Optional transform pipeline
            split: Dataset split ('train' or 'test')
            frame_skip: For videos, extract every Nth frame
            max_frames_per_video: Maximum frames per video
        """
        super().__init__(root_dir, transforms)
        self.split = split
        self.frame_skip = frame_skip
        self.max_frames_per_video = max_frames_per_video
        
        # Try to find images or videos in split directory
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            # Try root directory directly
            split_dir = root_dir
        
        # Find all image and video files
        self.image_files = sorted(glob.glob(os.path.join(split_dir, "*.jpg")) +
                                 glob.glob(os.path.join(split_dir, "*.jpeg")) +
                                 glob.glob(os.path.join(split_dir, "*.png")))
        self.video_files = sorted(glob.glob(os.path.join(split_dir, "*.mp4")) +
                                  glob.glob(os.path.join(split_dir, "*.avi")))
        
        # Build sample list
        self.samples = []
        self._build_sample_list()
    
    def _build_sample_list(self):
        """Build list of samples (images or video frames)."""
        self.samples = []
        
        # Add image files
        for img_path in self.image_files:
            self.samples.append(('image', img_path, None))
        
        # Add video frames
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
                self.samples.append(('video', video_path, fidx))
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with image, annotations, etc.
        """
        sample_type, file_path, frame_idx = self.samples[idx]
        
        if sample_type == 'image':
            # Load image directly
            frame = cv2.imread(file_path)
            if frame is None:
                raise ValueError(f"Failed to load image: {file_path}")
        else:
            # Load frame from video
            cap = cv2.VideoCapture(file_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                raise ValueError(f"Failed to read frame {frame_idx} from {file_path}")
        
        # Get original size
        H, W = frame.shape[:2]
        
        # Kaggle dashcam datasets typically don't have annotations
        sample = {
            "image": frame,
            "annotations": [],  # No annotations
            "img_id": idx,
            "original_size": (H, W),
            "file_path": file_path,
            "frame_index": frame_idx if sample_type == 'video' else None
        }
        
        # Apply transforms if provided
        if self.transforms:
            sample = self.transforms(sample)
        
        return sample
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

