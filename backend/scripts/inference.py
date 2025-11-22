"""
Detection Script (detect.py)
Loads checkpoint and runs inference on frames
Returns: bbox + labels + scores + centroids
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Import RT-DETR from local copy
from src.rtdetr.core import YAMLConfig
from src.rtdetr.data.coco.coco_dataset import mscoco_label2category, mscoco_category2name


def calculate_centroid(bbox: np.ndarray) -> np.ndarray:
    """
    Calculate centroid of bounding box.
    
    Args:
        bbox: [x1, y1, x2, y2]
    
    Returns:
        [cx, cy]
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return np.array([cx, cy])


def load_model(checkpoint_path: str, config_path: Optional[str] = None, device: str = 'cuda'):
    """
    Load RT-DETR model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config_path: Path to config file (auto-detected if None)
        device: Device to load model on
    
    Returns:
        Model in deploy mode
    """
    if config_path is None:
        # Try to find default config in backend/configs
        default_config = backend_path / "configs" / "d2city_rtdetr.yml"
        if default_config.exists():
            config_path = str(default_config)
        else:
            raise ValueError("No config provided and default not found")
    
    print(f"Loading config from: {config_path}")
    
    # Step 1: Establish model architecture from config
    # This creates the model structure (backbone, encoder, decoder) from YAML
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    print("✓ Model architecture established from config")
    
    # Step 2: Load pretrained weights into the established model
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint.get('model', checkpoint)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load state dict into the established model architecture
    cfg.model.load_state_dict(state, strict=False)
    print("✓ Loaded pretrained weights into model")
    
    # Create deploy model
    class DeployModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = DeployModel().to(device)
    model.eval()
    return model, cfg


def detect_frame(model: nn.Module, frame: Image.Image, device: str = 'cuda',
                conf_threshold: float = 0.5) -> Dict:
    """
    Run detection on a single frame.
    
    Args:
        model: RT-DETR model in deploy mode
        frame: PIL Image
        device: Device to run inference on
        conf_threshold: Confidence threshold
    
    Returns:
        Dictionary with:
        - 'bboxes': List of [x1, y1, x2, y2]
        - 'labels': List of class IDs
        - 'scores': List of confidence scores
        - 'centroids': List of [cx, cy]
        - 'class_names': List of class names
    """
    w, h = frame.size
    orig_size = torch.tensor([w, h])[None].to(device)
    
    # Preprocess
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(frame)[None].to(device)
    
    # Inference
    with torch.no_grad():
        with autocast():
            output = model(im_data, orig_size)
    
    labels, boxes, scores = output
    
    # Convert to numpy
    labels = labels.cpu().numpy()
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    
    # Filter by confidence threshold
    valid_mask = scores[0] > conf_threshold
    valid_labels = labels[0][valid_mask]
    valid_boxes = boxes[0][valid_mask]
    valid_scores = scores[0][valid_mask]
    
    # Calculate centroids
    centroids = [calculate_centroid(bbox) for bbox in valid_boxes]
    
    # Convert labels to class names
    class_names = []
    for label_idx in valid_labels:
        category_id = mscoco_label2category.get(int(label_idx), int(label_idx))
        class_name = mscoco_category2name.get(category_id, f"class_{label_idx}")
        class_names.append(class_name)
    
    return {
        'bboxes': valid_boxes.tolist(),
        'labels': valid_labels.tolist(),
        'scores': valid_scores.tolist(),
        'centroids': [c.tolist() for c in centroids],
        'class_names': class_names,
        'num_detections': len(valid_boxes)
    }


def detect_image(model: nn.Module, image_path: str, device: str = 'cuda',
                conf_threshold: float = 0.5) -> Dict:
    """
    Run detection on a single image.
    
    Args:
        model: RT-DETR model
        image_path: Path to image file
        device: Device to run inference on
        conf_threshold: Confidence threshold
    
    Returns:
        Detection results dictionary
    """
    frame = Image.open(image_path).convert('RGB')
    results = detect_frame(model, frame, device, conf_threshold)
    results['image_path'] = image_path
    return results


def detect_video_frame(model: nn.Module, frame_array: np.ndarray, device: str = 'cuda',
                       conf_threshold: float = 0.5) -> Dict:
    """
    Run detection on a video frame (numpy array).
    
    Args:
        model: RT-DETR model
        frame_array: numpy array (H, W, C) in RGB format
        device: Device to run inference on
        conf_threshold: Confidence threshold
    
    Returns:
        Detection results dictionary
    """
    frame = Image.fromarray(frame_array)
    return detect_frame(model, frame, device, conf_threshold)


def main(args):
    """Main function for command-line usage."""
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model, cfg = load_model(args.checkpoint, args.config, device)
    
    # Process input
    if args.input_type == 'image':
        results = detect_image(model, args.input, device, args.conf_threshold)
        
        # Print results
        print(f"\nDetection Results for {args.input}:")
        print(f"Number of detections: {results['num_detections']}")
        for i, (bbox, label, score, centroid, class_name) in enumerate(zip(
            results['bboxes'], results['labels'], results['scores'],
            results['centroids'], results['class_names']
        )):
            print(f"  Detection {i+1}:")
            print(f"    Class: {class_name} (ID: {label})")
            print(f"    Confidence: {score:.4f}")
            print(f"    BBox: [{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]")
            print(f"    Centroid: [{centroid[0]:.2f}, {centroid[1]:.2f}]")
        
        # Save results if output specified
        if args.output:
            save_results(results, args.output, args.input)
    
    elif args.input_type == 'video':
        print("Video processing not implemented in detect.py")
        print("Use scripts/process_video.py for video processing")
    
    else:
        raise ValueError(f"Unknown input type: {args.input_type}")


def save_results(results: Dict, output_path: str, image_path: str):
    """
    Save detection results to JSON file.
    
    Args:
        results: Detection results dictionary
        output_path: Path to save JSON file
        image_path: Path to input image
    """
    output_data = {
        'image_path': image_path,
        'num_detections': results['num_detections'],
        'detections': []
    }
    
    for bbox, label, score, centroid, class_name in zip(
        results['bboxes'], results['labels'], results['scores'],
        results['centroids'], results['class_names']
    ):
        output_data['detections'].append({
            'bbox': bbox,
            'label': int(label),
            'class_name': class_name,
            'score': float(score),
            'centroid': centroid
        })
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RT-DETR Detection Script - Returns bbox, labels, scores, centroids"
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
        help='Path to input image or video'
    )
    parser.add_argument(
        '--input-type', type=str, choices=['image', 'video'],
        default='image', help='Type of input'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Path to save results JSON file'
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
    
    args = parser.parse_args()
    main(args)

