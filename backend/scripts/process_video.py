"""
Video Processing Script
Processes video files and outputs videos with bounding boxes, class names
Also saves predictions as txt/json files

This script is used by the FastAPI backend for processing videos uploaded through the frontend.
It handles frame extraction, inference, and video reconstruction with bounding boxes.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.cuda.amp import autocast
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json
from typing import List, Dict, Optional

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Import RT-DETR from local copy
from src.rtdetr.core import YAMLConfig
from src.rtdetr.data.coco.coco_dataset import mscoco_label2category, mscoco_category2name
# Import inference functions
sys.path.insert(0, str(Path(__file__).parent))
from inference import load_model, detect_frame, calculate_centroid


def draw_detections(image: Image.Image, results: Dict, conf_threshold: float = 0.5) -> Image.Image:
    """
    Draw bounding boxes and labels on image.
    
    Args:
        image: PIL Image
        results: Detection results from detect_frame
        conf_threshold: Confidence threshold
    
    Returns:
        Annotated PIL Image
    """
    draw = ImageDraw.Draw(image)
    
    # Try to load font
    try:
        font_size = max(20, int(image.size[0] / 30))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Draw bounding boxes and labels
    for bbox, label, score, class_name in zip(
        results['bboxes'], results['labels'], results['scores'], results['class_names']
    ):
        if score < conf_threshold:
            continue
        
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Draw label with background
        label_text = f"{class_name}: {score:.2f}"
        bbox_text = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        # Background rectangle for text
        draw.rectangle(
            [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
            fill="red",
            outline="red"
        )
        
        # Text
        draw.text((x1 + 2, y1 - text_height - 2), label_text, fill="white", font=font)
    
    return image


def process_video(
    model: nn.Module,
    video_path: str,
    output_path: str,
    device: str = 'cuda',
    conf_threshold: float = 0.5,
    save_predictions: bool = True,
    output_format: str = 'json'
) -> Dict:
    """
    Process a video file: extract frames, run inference, draw bounding boxes, save video.
    
    Args:
        model: RT-DETR model (already loaded)
        video_path: Path to input video
        output_path: Path to save output video
        device: Device to run inference on
        conf_threshold: Confidence threshold
        save_predictions: Whether to save predictions to file
        output_format: Format for predictions ('json' or 'txt')
    
    Returns:
        Dictionary with processing statistics
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_predictions = []
    frame_idx = 0
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}, Resolution: {width}x{height}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Run inference
        results = detect_frame(model, frame_pil, device, conf_threshold)
        
        # Draw detections
        annotated_frame = draw_detections(frame_pil, results, conf_threshold)
        
        # Convert back to BGR for video writer
        annotated_array = np.array(annotated_frame)
        annotated_bgr = cv2.cvtColor(annotated_array, cv2.COLOR_RGB2BGR)
        out.write(annotated_bgr)
        
        # Save predictions
        if save_predictions:
            frame_predictions.append({
                'frame': frame_idx,
                'num_detections': results['num_detections'],
                'detections': [
                    {
                        'bbox': bbox,
                        'label': int(label),
                        'class_name': class_name,
                        'score': float(score),
                        'centroid': centroid
                    }
                    for bbox, label, score, centroid, class_name in zip(
                        results['bboxes'], results['labels'], results['scores'],
                        results['centroids'], results['class_names']
                    )
                ]
            })
        
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    out.release()
    
    # Save predictions if requested
    if save_predictions and frame_predictions:
        if output_format == 'json':
            pred_path = output_path.replace('.mp4', '_predictions.json')
            with open(pred_path, 'w') as f:
                json.dump(frame_predictions, f, indent=2)
        elif output_format == 'txt':
            pred_path = output_path.replace('.mp4', '_predictions.txt')
            with open(pred_path, 'w') as f:
                for frame_pred in frame_predictions:
                    f.write(f"Frame {frame_pred['frame']}:\n")
                    for det in frame_pred['detections']:
                        f.write(f"  {det['class_name']}: {det['score']:.3f} "
                               f"bbox={det['bbox']} centroid={det['centroid']}\n")
                    f.write("\n")
    
    stats = {
        'total_frames': frame_idx,
        'output_video': output_path,
        'predictions_file': pred_path if save_predictions else None
    }
    
    print(f"✓ Video processing complete: {output_path}")
    return stats


def main(args):
    """Main function for command-line usage."""
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model, cfg = load_model(args.checkpoint, args.config, device)
    
    # Process video
    stats = process_video(
        model=model,
        video_path=args.input,
        output_path=args.output,
        device=device,
        conf_threshold=args.conf_threshold,
        save_predictions=args.save_predictions,
        output_format=args.output_format
    )
    
    print(f"\nProcessing Statistics:")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Output video: {stats['output_video']}")
    if stats['predictions_file']:
        print(f"  Predictions file: {stats['predictions_file']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process video with RT-DETR - Draw bounding boxes and save predictions"
    )
    
    parser.add_argument(
        '--checkpoint', '-c', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to RT-DETR config (auto-detected if not provided)'
    )
    parser.add_argument(
        '--input', '-i', type=str, required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Path to save output video with bounding boxes'
    )
    parser.add_argument(
        '--conf-threshold', type=float, default=0.5,
        help='Confidence threshold for detections'
    )
    parser.add_argument(
        '--save-predictions', action='store_true', default=False,
        help='Save predictions to JSON/TXT file'
    )
    parser.add_argument(
        '--output-format', type=str, choices=['json', 'txt'], default='json',
        help='Format for predictions file'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on'
    )
    
    args = parser.parse_args()
    main(args)

