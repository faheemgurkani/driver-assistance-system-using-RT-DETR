"""
Custom Video Dataset Loader
For raw local dashcam videos or any custom video files
"""

import os
import cv2
import glob
import numpy as np
from typing import Dict, List, Any, Optional
from .base_dataset import BaseDataset


class CustomVideoDataset(BaseDataset):
    """
    Custom Video Dataset loader.
    Handles any local video files (MP4, AVI, etc.)
    
    Expected structure:
        custom_video/
            *.mp4, *.avi, *.mov, etc.
    """
    
    # Supported video extensions
    VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    
    def __init__(self, root_dir: str, transforms: Optional[Any] = None,
                 frame_skip: int = 1, max_frames_per_video: Optional[int] = None,
                 video_extensions: Optional[List[str]] = None):
        """
        Args:
            root_dir: Root directory containing video files
            transforms: Optional transform pipeline
            frame_skip: Extract every Nth frame (default: 1 = all frames)
            max_frames_per_video: Maximum frames to extract per video
            video_extensions: List of video file extensions to search for
        """
        super().__init__(root_dir, transforms)
        self.frame_skip = frame_skip
        self.max_frames_per_video = max_frames_per_video
        
        if video_extensions is None:
            video_extensions = self.VIDEO_EXTENSIONS
        
        # Find all video files
        self.video_files = []
        for ext in video_extensions:
            self.video_files.extend(glob.glob(os.path.join(root_dir, f"*{ext}")))
            # Also check subdirectories
            self.video_files.extend(glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True))
        
        self.video_files = sorted(list(set(self.video_files)))  # Remove duplicates
        
        if not self.video_files:
            raise ValueError(
                f"No video files found in {root_dir}. "
                f"Supported extensions: {video_extensions}"
            )
        
        # Build sample list
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
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            frame_indices = list(range(0, total_frames, self.frame_skip))
            
            if self.max_frames_per_video:
                frame_indices = frame_indices[:self.max_frames_per_video]
            
            for fidx in frame_indices:
                self.samples.append((video_path, fidx, fps))
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample (frame) from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with image, annotations, etc.
        """
        video_path, frame_idx, fps = self.samples[idx]
        
        # Open video and seek to frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")
        
        # Get original size
        H, W = frame.shape[:2]
        
        # Custom videos typically don't have annotations
        sample = {
            "image": frame,
            "annotations": [],  # No annotations
            "img_id": idx,
            "original_size": (H, W),
            "video_path": video_path,
            "frame_index": frame_idx,
            "fps": fps
        }
        
        # Apply transforms if provided
        if self.transforms:
            sample = self.transforms(sample)
        
        return sample
    
    def __len__(self) -> int:
        """Return number of samples (frames) in dataset."""
        return len(self.samples)

