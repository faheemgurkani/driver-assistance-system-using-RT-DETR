"""
Video Processing Script
Processes video files and outputs videos with bounding boxes, class names
Also saves predictions as txt/json files
"""

import os
import sys
import argparse
import subprocess
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.cuda.amp import autocast
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json
from typing import List, Dict, Optional, Callable

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
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            try:
                font = ImageFont.truetype("/Library/Fonts/Arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    for bbox, label, score, centroid, class_name in zip(
        results['bboxes'], results['labels'], results['scores'],
        results['centroids'], results['class_names']
    ):
        if score < conf_threshold:
            continue
        
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline='blue', width=4)
        
        # Prepare text
        text = f"{class_name} {score:.2f}"
        
        # Get text bounding box
        try:
            bbox_text = draw.textbbox((x1, y1), text, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
        except AttributeError:
            text_width = len(text) * 8
            text_height = 20
            bbox_text = [x1, y1, x1 + text_width, y1 + text_height]
        
        # Draw text background
        padding = 4
        draw.rectangle(
            [x1 - padding, y1 - padding, x1 + text_width + padding, y1 + text_height + padding],
            fill='white',
            outline='blue',
            width=2
        )
        
        # Draw text
        draw.text((x1, y1), text, font=font, fill='blue')
    
    return image


def ensure_h264_compat(output_path: str,
                       progress_callback: Optional[Callable[[int, int], None]] = None) -> None:
    """Re-encode MP4 to H.264/AAC to guarantee browser compatibility."""
    output_path = str(output_path)
    temp_path = Path(output_path).with_suffix(".h264.mp4")
    print("Re-encoding video to H.264 for browser playback...")
    if progress_callback is not None:
        progress_callback(-1, -1)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                output_path,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-movflags",
                "faststart",
                "-c:a",
                "aac",
                "-shortest",
                str(temp_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.replace(temp_path, output_path)
        print("✓ Video converted to H.264 for browser compatibility")
        if progress_callback is not None:
            progress_callback(-2, -2)
    except FileNotFoundError:
        print("⚠ ffmpeg binary not found. Please install ffmpeg to enable browser preview.")
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    except subprocess.CalledProcessError:
        print("⚠ Failed to re-encode video to H.264. Preview may not work in browser.")
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def process_video(model: nn.Module, video_path: str, output_path: str,
                device: str = 'cuda', conf_threshold: float = 0.5,
                save_predictions: bool = True, output_format: str = 'both',
                progress_callback: Optional[Callable[[int, int], None]] = None):
    """
    Process video and save output with bounding boxes.
    
    Args:
        model: RT-DETR model
        video_path: Path to input video
        output_path: Path to save output video
        device: Device to run inference on
        conf_threshold: Confidence threshold
        save_predictions: Whether to save predictions as txt/json
        output_format: 'json', 'txt', or 'both'
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Store all predictions
    all_predictions = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Run detection
        results = detect_frame(model, frame_pil, device, conf_threshold)
        
        # Draw detections
        annotated_frame = draw_detections(frame_pil, results, conf_threshold)
        
        # Convert back to BGR for video writer
        annotated_array = np.array(annotated_frame)
        annotated_bgr = cv2.cvtColor(annotated_array, cv2.COLOR_RGB2BGR)
        out.write(annotated_bgr)
        
        # Store predictions
        all_predictions.append({
            'frame_number': frame_count,
            'num_detections': results['num_detections'],
            'detections': []
        })
        
        for bbox, label, score, centroid, class_name in zip(
            results['bboxes'], results['labels'], results['scores'],
            results['centroids'], results['class_names']
        ):
            all_predictions[-1]['detections'].append({
                'bbox': bbox,
                'label': int(label),
                'class_name': class_name,
                'score': float(score),
                'centroid': centroid
            })
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")
            if progress_callback is not None:
                progress_callback(frame_count, total_frames)
    
    cap.release()
    out.release()

    if progress_callback is not None:
        progress_callback(total_frames, total_frames)

    ensure_h264_compat(output_path, progress_callback)
    
    print(f"✓ Saved output video to: {output_path}")
    
    # Save predictions
    if save_predictions:
        base_path = Path(output_path).stem
        output_dir = Path(output_path).parent
        
        if output_format in ['json', 'both']:
            json_path = output_dir / f"{base_path}_predictions.json"
            with open(json_path, 'w') as f:
                json.dump({
                    'video_path': video_path,
                    'total_frames': frame_count,
                    'fps': fps,
                    'resolution': [width, height],
                    'predictions': all_predictions
                }, f, indent=2)
            print(f"✓ Saved JSON predictions to: {json_path}")
        
        if output_format in ['txt', 'both']:
            txt_path = output_dir / f"{base_path}_predictions.txt"
            with open(txt_path, 'w') as f:
                f.write(f"Video: {video_path}\n")
                f.write(f"Total Frames: {frame_count}\n")
                f.write(f"FPS: {fps}\n")
                f.write(f"Resolution: {width}x{height}\n\n")
                
                for frame_pred in all_predictions:
                    f.write(f"Frame {frame_pred['frame_number']}:\n")
                    f.write(f"  Detections: {frame_pred['num_detections']}\n")
                    for det in frame_pred['detections']:
                        f.write(f"    - Class: {det['class_name']} (ID: {det['label']})\n")
                        f.write(f"      Score: {det['score']:.4f}\n")
                        f.write(f"      BBox: [{det['bbox'][0]:.2f}, {det['bbox'][1]:.2f}, "
                               f"{det['bbox'][2]:.2f}, {det['bbox'][3]:.2f}]\n")
                        f.write(f"      Centroid: [{det['centroid'][0]:.2f}, {det['centroid'][1]:.2f}]\n")
                    f.write("\n")
            print(f"✓ Saved TXT predictions to: {txt_path}")


def main(args):
    """Main function."""
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model, cfg = load_model(args.checkpoint, args.config, device)
    
    # Process video
    process_video(
        model, args.input, args.output, device,
        args.conf_threshold, args.save_predictions, args.output_format
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process video with RT-DETR and save output with bounding boxes"
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
        help='Path to input video'
    )
    parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Path to save output video'
    )
    parser.add_argument(
        '--conf-threshold', type=float, default=0.5,
        help='Confidence threshold for detections'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on'
    )
    parser.add_argument(
        '--save-predictions', action='store_true', default=True,
        help='Save predictions as txt/json files'
    )
    parser.add_argument(
        '--output-format', type=str, default='both',
        choices=['json', 'txt', 'both'],
        help='Format for prediction files'
    )
    
    args = parser.parse_args()
    main(args)

