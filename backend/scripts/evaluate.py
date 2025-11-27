"""
Evaluation Script
Calculates mAP and generates evaluation report
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np
from pathlib import Path
import json
from typing import List, Dict
import matplotlib.pyplot as plt
from collections import defaultdict

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Import RT-DETR from local copy
from src.rtdetr.core import YAMLConfig
from src.rtdetr.data import get_coco_api_from_dataset
# Import inference functions
sys.path.insert(0, str(Path(__file__).parent))
from inference import load_model, detect_frame


def calculate_ap(recall, precision):
    """Calculate Average Precision (AP) from recall and precision arrays."""
    # Add sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    
    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Look for points where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)
    
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou


def evaluate_model(model: nn.Module, dataloader, device: str = 'cuda',
                  conf_threshold: float = 0.5, iou_threshold: float = 0.5):
    """
    Evaluate model on dataset and calculate mAP.
    
    Args:
        model: RT-DETR model
        dataloader: DataLoader for evaluation
        device: Device to run inference on
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            
            # Get predictions
            orig_sizes = torch.stack([t['orig_size'] for t in targets]).to(device)
            
            with autocast():
                outputs = model(images, orig_sizes)
            
            labels_pred, boxes_pred, scores_pred = outputs
            
            # Process each image in batch
            for i in range(len(images)):
                # Filter by confidence
                valid_mask = scores_pred[i].cpu().numpy() > conf_threshold
                pred_boxes = boxes_pred[i].cpu().numpy()[valid_mask]
                pred_labels = labels_pred[i].cpu().numpy()[valid_mask]
                pred_scores = scores_pred[i].cpu().numpy()[valid_mask]
                
                # Get ground truth
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                gt_labels = targets[i]['labels'].cpu().numpy()
                
                all_predictions.append({
                    'boxes': pred_boxes,
                    'labels': pred_labels,
                    'scores': pred_scores
                })
                
                all_ground_truths.append({
                    'boxes': gt_boxes,
                    'labels': gt_labels
                })
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches...")
    
    # Calculate mAP
    print("Calculating mAP...")
    aps = []
    class_aps = defaultdict(list)
    
    for pred, gt in zip(all_predictions, all_ground_truths):
        # Group by class
        for class_id in np.unique(gt['labels']):
            gt_mask = gt['labels'] == class_id
            pred_mask = pred['labels'] == class_id
            
            if not np.any(pred_mask):
                continue
            
            gt_boxes_class = gt['boxes'][gt_mask]
            pred_boxes_class = pred['boxes'][pred_mask]
            pred_scores_class = pred['scores'][pred_mask]
            
            # Sort by score
            sorted_indices = np.argsort(pred_scores_class)[::-1]
            pred_boxes_class = pred_boxes_class[sorted_indices]
            pred_scores_class = pred_scores_class[sorted_indices]
            
            # Calculate TP/FP
            tp = np.zeros(len(pred_boxes_class))
            fp = np.zeros(len(pred_boxes_class))
            matched = np.zeros(len(gt_boxes_class), dtype=bool)
            
            for i, pred_box in enumerate(pred_boxes_class):
                best_iou = 0
                best_gt_idx = -1
                
                for j, gt_box in enumerate(gt_boxes_class):
                    if matched[j]:
                        continue
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_iou >= iou_threshold:
                    tp[i] = 1
                    matched[best_gt_idx] = True
                else:
                    fp[i] = 1
            
            # Calculate precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recall = tp_cumsum / len(gt_boxes_class) if len(gt_boxes_class) > 0 else np.zeros_like(tp_cumsum)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum) if len(tp_cumsum + fp_cumsum) > 0 else np.zeros_like(tp_cumsum)
            
            # Calculate AP
            ap = calculate_ap(recall, precision)
            class_aps[int(class_id)].append(ap)
            aps.append(ap)
    
    # Calculate mean AP
    mean_ap = np.mean(aps) if len(aps) > 0 else 0.0
    
    # Calculate per-class AP
    class_mean_aps = {class_id: np.mean(aps_list) for class_id, aps_list in class_aps.items()}
    
    return {
        'mAP': mean_ap,
        'class_APs': class_mean_aps,
        'num_samples': len(all_predictions)
    }


def plot_map_curves(results: Dict, output_path: str):
    """Plot mAP charts."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = sorted(results['class_APs'].keys())
    aps = [results['class_APs'][c] for c in classes]
    
    ax.bar(range(len(classes)), aps)
    ax.set_xlabel('Class ID')
    ax.set_ylabel('Average Precision (AP)')
    ax.set_title(f'mAP: {results["mAP"]:.4f}')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([f'Class {c}' for c in classes])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✓ Saved mAP chart to: {output_path}")


def generate_report(results: Dict, output_path: str):
    """Generate evaluation report."""
    report = f"""
# Evaluation Report

## Overall Metrics
- **mAP**: {results['mAP']:.4f}
- **Number of Samples**: {results['num_samples']}

## Per-Class Average Precision
"""
    for class_id in sorted(results['class_APs'].keys()):
        ap = results['class_APs'][class_id]
        report += f"- Class {class_id}: {ap:.4f}\n"
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Saved evaluation report to: {output_path}")


def main(args):
    """Main function."""
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    # Load config and create dataloader
    print(f"Loading config from: {args.config}")
    cfg = YAMLConfig(args.config)
    
    # Load model
    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model, _ = load_model(args.checkpoint, args.config, device)
    else:
        model = cfg.model.to(device)
        model.eval()
    
    # Get validation dataloader
    val_dataloader = cfg.val_dataloader
    
    # Evaluate
    results = evaluate_model(
        model, val_dataloader, device,
        args.conf_threshold, args.iou_threshold
    )
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"mAP: {results['mAP']:.4f}")
    print(f"Number of samples: {results['num_samples']}")
    print("\nPer-Class AP:")
    for class_id in sorted(results['class_APs'].keys()):
        print(f"  Class {class_id}: {results['class_APs'][class_id]:.4f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_dir / "evaluation_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved JSON results to: {json_path}")
    
    # Generate report
    report_path = output_dir / "evaluation_report.md"
    generate_report(results, str(report_path))
    
    # Plot charts
    chart_path = output_dir / "map_chart.png"
    plot_map_curves(results, str(chart_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RT-DETR model and calculate mAP")
    
    parser.add_argument(
        '--config', '-c', type=str, required=True,
        help='Path to RT-DETR config YAML file'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to model checkpoint (optional if using config)'
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, default='./evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--conf-threshold', type=float, default=0.5,
        help='Confidence threshold for detections'
    )
    parser.add_argument(
        '--iou-threshold', type=float, default=0.5,
        help='IoU threshold for matching'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run evaluation on'
    )
    
    args = parser.parse_args()
    main(args)

