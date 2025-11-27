"""
Inference Script for Driver Assistance System using RT-DETR
Supports single images, videos, and webcam input
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Add project root to path (for RT-DETR)
project_root = backend_path.parent
sys.path.insert(0, str(project_root))

# Add RT-DETR to path
rtdetr_path = project_root.parent / "rtdetr_pytorch"
if rtdetr_path.exists():
    sys.path.insert(0, str(rtdetr_path))

from src.utils import PreprocessTransforms
from typing import Optional


def load_model(checkpoint_path: str, config_path: Optional[str] = None):
    """Load RT-DETR model from checkpoint."""
    from src.core import YAMLConfig
    
    if config_path is None:
        # Try to find default config
        default_config = rtdetr_path / "configs" / "rtdetr" / "rtdetr_r50vd_6x_coco.yml"
        if default_config.exists():
            config_path = str(default_config)
        else:
            raise ValueError("No config provided and default not found")
    
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    
    from src.core import GLOBAL_CONFIG
    model = GLOBAL_CONFIG['model']()
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.eval()
    return model, cfg


def postprocess_predictions(preds: Dict[str, torch.Tensor], 
                           original_size: Tuple[int, int],
                           target_size: Tuple[int, int],
                           conf_threshold: float = 0.5) -> List[Dict]:
    """
    Post-process RT-DETR predictions.
    
    Args:
        preds: Model predictions with 'pred_logits' and 'pred_boxes'
        original_size: Original image size (H, W)
        target_size: Target/preprocessed size (H, W)
        conf_threshold: Confidence threshold for filtering
    
    Returns:
        List of detections: [{"bbox": [x1,y1,x2,y2], "score": float, "class": int}, ...]
    """
    logits = preds['pred_logits']  # [batch, num_queries, num_classes]
    boxes = preds['pred_boxes']    # [batch, num_queries, 4] in cxcywh format
    
    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=-1)
    
    # Get max scores and class indices
    scores, class_ids = probs.max(dim=-1)
    
    # Filter by confidence threshold
    valid_mask = scores > conf_threshold
    
    results = []
    for b in range(logits.shape[0]):  # batch dimension
        batch_valid = valid_mask[b]
        batch_scores = scores[b][batch_valid]
        batch_classes = class_ids[b][batch_valid]
        batch_boxes = boxes[b][batch_valid]
        
        H_orig, W_orig = original_size
        H_target, W_target = target_size
        
        # Scale factors
        scale_x = W_orig / W_target
        scale_y = H_orig / H_target
        
        for score, cls, box in zip(batch_scores, batch_classes, batch_boxes):
            # Convert from cxcywh to xyxy
            cx, cy, w, h = box.cpu().numpy()
            x1 = (cx - w/2) * scale_x
            y1 = (cy - h/2) * scale_y
            x2 = (cx + w/2) * scale_x
            y2 = (cy + h/2) * scale_y
            
            # Clamp to image boundaries
            x1 = max(0, min(x1, W_orig))
            y1 = max(0, min(y1, H_orig))
            x2 = max(0, min(x2, W_orig))
            y2 = max(0, min(y2, H_orig))
            
            results.append({
                "bbox": [x1, y1, x2, y2],
                "score": float(score),
                "class": int(cls)
            })
    
    return results


def draw_detections(image: np.ndarray, detections: List[Dict], 
                   class_names: Optional[Dict[int, str]] = None) -> np.ndarray:
    """Draw bounding boxes on image."""
    img = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = [int(coord) for coord in det["bbox"]]
        score = det["score"]
        cls = det["class"]
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = class_names.get(cls, f"Class {cls}") if class_names else f"Class {cls}"
        label_text = f"{label}: {score:.2f}"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw text background
        cv2.rectangle(img, (x1, y1 - text_height - baseline - 5),
                     (x1 + text_width, y1), (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(img, label_text, (x1, y1 - baseline - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img


def infer_single_frame(model: torch.nn.Module, 
                      frame: np.ndarray,
                      preprocess: PreprocessTransforms,
                      device: torch.device,
                      conf_threshold: float = 0.5) -> Tuple[np.ndarray, List[Dict]]:
    """
    Run inference on a single frame.
    
    Args:
        model: RT-DETR model
        frame: Input frame (BGR numpy array)
        preprocess: Preprocessing transform
        device: Torch device
        conf_threshold: Confidence threshold
    
    Returns:
        Tuple of (annotated_frame, detections)
    """
    # Preprocess
    sample = {
        "image": frame,
        "annotations": [],
        "original_size": frame.shape[:2],
        "img_id": 0
    }
    processed = preprocess(sample)
    
    # Add batch dimension and move to device
    img_tensor = processed["image"].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        preds = model(img_tensor)
    
    # Post-process
    detections = postprocess_predictions(
        preds,
        processed["original_size"],
        preprocess.target_size,
        conf_threshold
    )
    
    # Draw detections
    annotated_frame = draw_detections(frame, detections)
    
    return annotated_frame, detections


def infer_image(model, image_path, output_path, preprocess, device, conf_threshold=0.5):
    """Run inference on a single image."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    annotated_frame, detections = infer_single_frame(
        model, frame, preprocess, device, conf_threshold
    )
    
    cv2.imwrite(output_path, annotated_frame)
    print(f"Saved result to: {output_path}")
    print(f"Found {len(detections)} detections")


def infer_video(model, video_path, output_path, preprocess, device, conf_threshold=0.5):
    """Run inference on a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated_frame, detections = infer_single_frame(
            model, frame, preprocess, device, conf_threshold
        )
        
        out.write(annotated_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    print(f"Saved result to: {output_path}")


def infer_webcam(model, preprocess, device, conf_threshold=0.5):
    """Run inference on webcam feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Failed to open webcam")
    
    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated_frame, detections = infer_single_frame(
            model, frame, preprocess, device, conf_threshold
        )
        
        cv2.imshow("Driver Assistance System", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main(args):
    """Main inference function."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, cfg = load_model(args.checkpoint, args.config)
    model = model.to(device)
    
    # Create preprocessor
    preprocess = PreprocessTransforms(target_size=(640, 640))
    
    # Run inference based on input type
    if args.input_type == "image":
        infer_image(model, args.input, args.output, preprocess, device, args.conf_threshold)
    elif args.input_type == "video":
        infer_video(model, args.input, args.output, preprocess, device, args.conf_threshold)
    elif args.input_type == "webcam":
        infer_webcam(model, preprocess, device, args.conf_threshold)
    else:
        raise ValueError(f"Unknown input type: {args.input_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Driver Assistance System")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to RT-DETR config (auto-detected if not provided)")
    
    parser.add_argument("--input", type=str, default=None,
                       help="Input image or video path")
    parser.add_argument("--input-type", type=str, choices=["image", "video", "webcam"],
                       default="image", help="Type of input")
    parser.add_argument("--output", type=str, default="output.jpg",
                       help="Output path for image/video")
    
    parser.add_argument("--conf-threshold", type=float, default=0.5,
                       help="Confidence threshold for detections")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU usage")
    
    args = parser.parse_args()
    
    if args.input_type != "webcam" and args.input is None:
        parser.error("--input is required when --input-type is not 'webcam'")
    
    main(args)

