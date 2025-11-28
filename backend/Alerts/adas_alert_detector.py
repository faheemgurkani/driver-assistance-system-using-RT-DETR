#!/usr/bin/env python3
"""
ADAS Alert Detector - Unified Blind Spot & Collision Warning System

This script combines Blind Spot Detection and Collision Warning modules.
It processes frames, detects alerts, stores exact coordinates, and visualizes results.

Usage:
    python adas_alert_detector.py --frame <image_path> --json <detections_json>
    
    Or import and use:
    from adas_alert_detector import process_adas_alerts
    result = process_adas_alerts(frame_path, json_path, output_dir="output")
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# Import ADAS modules
from modules.blind_spot import check_blind_spot, draw_blind_spot_zones, compute_centroid
from modules.collision import (
    compute_collision_risk,
    draw_collision_risk,
    estimate_distance,
    in_lane as check_in_lane
)


def load_detections_from_json(json_path: str) -> List[Dict]:
    """
    Load detections from a JSON file with flexible format support.
    
    Args:
        json_path: Path to JSON file containing detections
    
    Returns:
        list: List of normalized detection dictionaries
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            detections = data
        elif isinstance(data, dict):
            if 'detections' in data:
                detections = data['detections']
            elif 'objects' in data:
                detections = data['objects']
            elif 'results' in data:
                detections = data['results']
            else:
                detections = [data] if 'class' in data or 'label' in data else []
        else:
            detections = []
        
        # Normalize detection format
        normalized_detections = []
        for det in detections:
            normalized = {}
            
            # Handle class/label
            if 'class' in det:
                normalized['class'] = det['class']
            elif 'label' in det:
                normalized['class'] = det['label']
            
            # Handle confidence/score
            if 'confidence' in det:
                normalized['confidence'] = det['confidence']
                normalized['score'] = det['confidence']  # Also add as score for collision module
            elif 'score' in det:
                normalized['confidence'] = det['score']
                normalized['score'] = det['score']
            
            # Handle bounding box
            if 'bbox' in det:
                bbox = det['bbox']
                if isinstance(bbox, list) and len(bbox) == 4:
                    normalized['bbox'] = bbox
                    normalized['x1'] = bbox[0]
                    normalized['y1'] = bbox[1]
                    normalized['x2'] = bbox[2]
                    normalized['y2'] = bbox[3]
            elif all(k in det for k in ['x1', 'y1', 'x2', 'y2']):
                normalized['x1'] = det['x1']
                normalized['y1'] = det['y1']
                normalized['x2'] = det['x2']
                normalized['y2'] = det['y2']
                normalized['bbox'] = [det['x1'], det['y1'], det['x2'], det['y2']]
            elif all(k in det for k in ['xtl', 'ytl', 'xbr', 'ybr']):
                normalized['xtl'] = det['xtl']
                normalized['ytl'] = det['ytl']
                normalized['xbr'] = det['xbr']
                normalized['ybr'] = det['ybr']
                normalized['bbox'] = [det['xtl'], det['ytl'], det['xbr'], det['ybr']]
                normalized['x1'] = det['xtl']
                normalized['y1'] = det['ytl']
                normalized['x2'] = det['xbr']
                normalized['y2'] = det['ybr']
            
            if normalized:
                normalized_detections.append(normalized)
        
        return normalized_detections
    
    except FileNotFoundError:
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {json_path}: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error loading detections from JSON: {e}", file=sys.stderr)
        return []


def get_blind_spot_coordinates(detections: List[Dict], frame_width: int, frame_height: int) -> Dict:
    """
    Detect blind spots and return exact coordinates of objects in blind spot zones.
    
    Args:
        detections: List of detection dictionaries
        frame_width: Width of the frame
        frame_height: Height of the frame
    
    Returns:
        dict: Detailed blind spot information with coordinates
    """
    from modules.blind_spot import VALID_CLASSES, MIN_CONFIDENCE, MIN_BBOX_HEIGHT
    
    left_objects = []
    right_objects = []
    
    # Left blind spot zone boundaries
    left_x_min = 0.0 * frame_width
    left_x_max = 0.25 * frame_width
    left_y_min = 0.60 * frame_height
    left_y_max = 1.00 * frame_height
    
    # Right blind spot zone boundaries
    right_x_min = 0.75 * frame_width
    right_x_max = 1.00 * frame_width
    right_y_min = 0.60 * frame_height
    right_y_max = 1.00 * frame_height
    
    for detection in detections:
        # Filter by class
        class_name = detection.get('class') or detection.get('label', '').lower()
        if class_name not in VALID_CLASSES:
            continue
        
        # Filter by confidence
        confidence = detection.get('confidence', 0.0)
        if confidence < MIN_CONFIDENCE:
            continue
        
        # Extract bounding box
        if 'bbox' in detection:
            bbox = detection['bbox']
        else:
            bbox = detection
        
        # Calculate bounding box height
        if isinstance(bbox, dict):
            if 'x1' in bbox and 'y1' in bbox:
                bbox_height = abs(bbox['y2'] - bbox['y1'])
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            elif 'xtl' in bbox and 'ytl' in bbox:
                bbox_height = abs(bbox['ybr'] - bbox['ytl'])
                x1, y1, x2, y2 = bbox['xtl'], bbox['ytl'], bbox['xbr'], bbox['ybr']
            else:
                continue
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bbox_height = abs(bbox[3] - bbox[1])
            x1, y1, x2, y2 = bbox
        else:
            continue
        
        # Filter by size
        if bbox_height < MIN_BBOX_HEIGHT:
            continue
        
        # Calculate centroid
        try:
            cx, cy = compute_centroid(bbox)
        except (ValueError, KeyError):
            continue
        
        # Check left blind spot
        if (left_x_min <= cx <= left_x_max) and (left_y_min <= cy <= left_y_max):
            left_objects.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'centroid': (float(cx), float(cy)),
                'zone': 'left',
                'zone_coordinates': {
                    'x_min': left_x_min,
                    'x_max': left_x_max,
                    'y_min': left_y_min,
                    'y_max': left_y_max
                }
            })
        
        # Check right blind spot
        if (right_x_min <= cx <= right_x_max) and (right_y_min <= cy <= right_y_max):
            right_objects.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'centroid': (float(cx), float(cy)),
                'zone': 'right',
                'zone_coordinates': {
                    'x_min': right_x_min,
                    'x_max': right_x_max,
                    'y_min': right_y_min,
                    'y_max': right_y_max
                }
            })
    
    return {
        'left_blind_spot': len(left_objects) > 0,
        'right_blind_spot': len(right_objects) > 0,
        'objects_in_left': len(left_objects),
        'objects_in_right': len(right_objects),
        'left_objects': left_objects,
        'right_objects': right_objects
    }


def get_collision_warning_details(detections: List[Dict], frame_width: int, frame_height: int, k: float = 800.0) -> Dict:
    """
    Detect collision warnings and return exact positions and reasons.
    
    Args:
        detections: List of detection dictionaries
        frame_width: Width of the frame
        frame_height: Height of the frame
        k: Distance scaling constant
    
    Returns:
        dict: Detailed collision warning information with positions and reasons
    """
    from modules.collision import VALID_CLASSES, MIN_SCORE, MIN_BBOX_HEIGHT, HIGH_RISK_DISTANCE, MEDIUM_RISK_DISTANCE
    
    nearest_object = None
    nearest_distance = float('inf')
    all_valid_objects = []
    
    # Lane zone boundaries
    lane_x_min = 0.30 * frame_width
    lane_x_max = 0.70 * frame_width
    
    for detection in detections:
        # Filter by class
        class_name = detection.get('class', '').lower()
        if class_name not in VALID_CLASSES:
            continue
        
        # Filter by score
        score = detection.get('score', detection.get('confidence', 0.0))
        if score < MIN_SCORE:
            continue
        
        # Extract bounding box
        bbox = detection.get('bbox')
        if bbox is None:
            continue
        
        # Calculate bounding box height
        if isinstance(bbox, dict):
            bbox_height = abs(bbox['y2'] - bbox['y1'])
            bbox_list = [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bbox_height = abs(bbox[3] - bbox[1])
            bbox_list = list(bbox)
        else:
            continue
        
        # Filter by size
        if bbox_height < MIN_BBOX_HEIGHT:
            continue
        
        # Calculate centroid
        try:
            cx, cy = compute_centroid(bbox)
        except (ValueError, KeyError):
            continue
        
        # Filter by lane
        if not check_in_lane(cx, frame_width):
            continue
        
        # Estimate distance
        distance = estimate_distance(bbox_height, k)
        
        # Store object info
        object_info = {
            'class': class_name,
            'score': score,
            'bbox': bbox_list,
            'centroid': (float(cx), float(cy)),
            'distance': float(distance),
            'bbox_height': float(bbox_height),
            'lane_zone': {
                'x_min': lane_x_min,
                'x_max': lane_x_max
            }
        }
        
        all_valid_objects.append(object_info)
        
        # Track nearest
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_object = object_info
    
    # Determine risk level and reason
    if nearest_distance == float('inf') or nearest_object is None:
        risk_level = "LOW"
        reason = "No valid vehicles detected in frontal lane"
    elif nearest_distance <= HIGH_RISK_DISTANCE:
        risk_level = "HIGH"
        reason = f"Vehicle detected at {nearest_distance:.2f}m (HIGH RISK: <= {HIGH_RISK_DISTANCE}m)"
    elif nearest_distance <= MEDIUM_RISK_DISTANCE:
        risk_level = "MEDIUM"
        reason = f"Vehicle detected at {nearest_distance:.2f}m (MEDIUM RISK: <= {MEDIUM_RISK_DISTANCE}m)"
    else:
        risk_level = "LOW"
        reason = f"Vehicle detected at {nearest_distance:.2f}m (LOW RISK: > {MEDIUM_RISK_DISTANCE}m)"
    
    return {
        'collision_risk': risk_level,
        'nearest_distance': nearest_distance if nearest_distance != float('inf') else None,
        'nearest_object': nearest_object,
        'all_objects': all_valid_objects,
        'reason': reason,
        'risk_thresholds': {
            'high': HIGH_RISK_DISTANCE,
            'medium': MEDIUM_RISK_DISTANCE
        }
    }


def draw_all_alerts(frame: np.ndarray, blind_spot_info: Dict, collision_info: Dict, detections: List[Dict]) -> np.ndarray:
    """
    Draw all alerts (blind spot and collision) on the frame.
    
    Args:
        frame: OpenCV image
        blind_spot_info: Blind spot detection results with coordinates
        collision_info: Collision warning results with positions
        detections: All detections for visualization
    
    Returns:
        numpy.ndarray: Frame with all alerts drawn
    """
    vis_frame = frame.copy()
    h, w = vis_frame.shape[:2]
    
    # Draw blind spot zones
    vis_frame = draw_blind_spot_zones(vis_frame)
    
    # Draw collision risk indicator
    collision_result_for_draw = {
        'collision_risk': collision_info['collision_risk'],
        'nearest_distance': collision_info['nearest_distance'],
        'object_bbox': collision_info['nearest_object']['bbox'] if collision_info['nearest_object'] else None
    }
    vis_frame = draw_collision_risk(vis_frame, collision_result_for_draw)
    
    # Draw all detections
    for det in detections:
        # Get bounding box
        if 'bbox' in det and isinstance(det['bbox'], list):
            x1, y1, x2, y2 = map(int, det['bbox'])
        elif all(k in det for k in ['x1', 'y1', 'x2', 'y2']):
            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
        else:
            continue
        
        class_name = det.get('class') or det.get('label', 'unknown')
        confidence = det.get('confidence') or det.get('score', 0.0)
        
        # Determine color
        if class_name.lower() in ['car', 'bus', 'truck', 'van', 'motorcycle', 'bicycle']:
            color = (0, 255, 0)  # Green
        else:
            color = (255, 0, 0)  # Red
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name} {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 5),
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(vis_frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Highlight blind spot objects with red boxes
    for obj in blind_spot_info.get('left_objects', []):
        x1, y1, x2, y2 = map(int, obj['bbox'])
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red thick border
        cv2.putText(vis_frame, f"BLIND SPOT LEFT", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    for obj in blind_spot_info.get('right_objects', []):
        x1, y1, x2, y2 = map(int, obj['bbox'])
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red thick border
        cv2.putText(vis_frame, f"BLIND SPOT RIGHT", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Highlight collision risk object
    if collision_info.get('nearest_object'):
        obj = collision_info['nearest_object']
        x1, y1, x2, y2 = map(int, obj['bbox'])
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 165, 255), 3)  # Orange thick border
        distance_text = f"COLLISION RISK: {obj['distance']:.1f}m"
        cv2.putText(vis_frame, distance_text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    # Draw blind spot alert text
    alert_y = 30
    if blind_spot_info['left_blind_spot']:
        alert_text = f"LEFT BLIND SPOT ALERT! ({blind_spot_info['objects_in_left']} objects)"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.rectangle(vis_frame, (text_x - 10, alert_y - 25),
                     (text_x + text_size[0] + 10, alert_y + 5), (0, 0, 255), -1)
        cv2.putText(vis_frame, alert_text, (text_x, alert_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        alert_y += 40
    
    if blind_spot_info['right_blind_spot']:
        alert_text = f"RIGHT BLIND SPOT ALERT! ({blind_spot_info['objects_in_right']} objects)"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.rectangle(vis_frame, (text_x - 10, alert_y - 25),
                     (text_x + text_size[0] + 10, alert_y + 5), (0, 0, 255), -1)
        cv2.putText(vis_frame, alert_text, (text_x, alert_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return vis_frame


def process_adas_alerts(frame_path: str, json_path: str, output_dir: str = "output", save_output: bool = True) -> Dict:
    """
    Main function to process ADAS alerts (blind spot + collision warning).
    
    This function can be imported and used in other Python files.
    
    Args:
        frame_path: Path to input image file
        json_path: Path to JSON file containing detections
        output_dir: Directory to save output image (default: "output")
        save_output: Whether to save the output image (default: True)
    
    Returns:
        dict: Complete ADAS alert results with coordinates and reasons
    """
    # Validate inputs
    if not os.path.exists(frame_path):
        return {
            "success": False,
            "message": f"Frame file not found: {frame_path}",
            "blind_spot": None,
            "collision": None,
            "output_path": None
        }
    
    if not os.path.exists(json_path):
        return {
            "success": False,
            "message": f"JSON file not found: {json_path}",
            "blind_spot": None,
            "collision": None,
            "output_path": None
        }
    
    # Load image
    frame = cv2.imread(frame_path)
    if frame is None:
        return {
            "success": False,
            "message": f"Failed to load image: {frame_path}",
            "blind_spot": None,
            "collision": None,
            "output_path": None
        }
    
    frame_height, frame_width = frame.shape[:2]
    
    # Load detections
    detections = load_detections_from_json(json_path)
    if not detections:
        return {
            "success": False,
            "message": f"No valid detections found in {json_path}",
            "blind_spot": None,
            "collision": None,
            "output_path": None
        }
    
    # STEP 1: Detect blind spots and get exact coordinates
    blind_spot_info = get_blind_spot_coordinates(detections, frame_width, frame_height)
    
    # STEP 2: Check collision warnings and get exact positions/reasons
    collision_info = get_collision_warning_details(detections, frame_width, frame_height)
    
    # STEP 3: Draw all alerts on frame
    vis_frame = draw_all_alerts(frame, blind_spot_info, collision_info, detections)
    
    # Save output
    output_path = None
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        frame_name = Path(frame_path).stem
        output_path = os.path.join(output_dir, f"adas_alerts_{frame_name}.jpg")
        cv2.imwrite(output_path, vis_frame)
    
    return {
        "success": True,
        "message": "Processing completed successfully",
        "blind_spot": blind_spot_info,
        "collision": collision_info,
        "output_path": output_path,
        "frame_dimensions": {
            "width": frame_width,
            "height": frame_height
        }
    }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="ADAS Alert Detector - Blind Spot & Collision Warning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python adas_alert_detector.py --frame image.jpg --json detections.json
  python adas_alert_detector.py -f image.jpg -j detections.json -o custom_output
        """
    )
    
    parser.add_argument(
        '--frame', '-f',
        type=str,
        required=True,
        help='Path to input image file'
    )
    
    parser.add_argument(
        '--json', '-j',
        type=str,
        required=True,
        help='Path to JSON file containing detections'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory for processed images (default: output)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save output image (only print results)'
    )
    
    args = parser.parse_args()
    
    # Process the frame
    result = process_adas_alerts(
        frame_path=args.frame,
        json_path=args.json,
        output_dir=args.output,
        save_output=not args.no_save
    )
    
    # Print results
    if result['success']:
        print("=" * 70)
        print("ADAS ALERT DETECTION RESULTS")
        print("=" * 70)
        print(f"Frame: {args.frame}")
        print(f"Detections: {args.json}")
        print(f"Frame Dimensions: {result['frame_dimensions']['width']}x{result['frame_dimensions']['height']}")
        print()
        
        # Blind Spot Results
        print("-" * 70)
        print("BLIND SPOT DETECTION")
        print("-" * 70)
        bs = result['blind_spot']
        print(f"Left Blind Spot Alert: {bs['left_blind_spot']}")
        print(f"Right Blind Spot Alert: {bs['right_blind_spot']}")
        print(f"Objects in Left Zone: {bs['objects_in_left']}")
        print(f"Objects in Right Zone: {bs['objects_in_right']}")
        
        if bs['left_objects']:
            print("\nLeft Blind Spot Objects (with coordinates):")
            for i, obj in enumerate(bs['left_objects'], 1):
                print(f"  {i}. {obj['class']} (conf: {obj['confidence']:.2f})")
                print(f"     BBox: {obj['bbox']}")
                print(f"     Centroid: {obj['centroid']}")
                print(f"     Zone: x=[{obj['zone_coordinates']['x_min']:.0f}, {obj['zone_coordinates']['x_max']:.0f}], "
                      f"y=[{obj['zone_coordinates']['y_min']:.0f}, {obj['zone_coordinates']['y_max']:.0f}]")
        
        if bs['right_objects']:
            print("\nRight Blind Spot Objects (with coordinates):")
            for i, obj in enumerate(bs['right_objects'], 1):
                print(f"  {i}. {obj['class']} (conf: {obj['confidence']:.2f})")
                print(f"     BBox: {obj['bbox']}")
                print(f"     Centroid: {obj['centroid']}")
                print(f"     Zone: x=[{obj['zone_coordinates']['x_min']:.0f}, {obj['zone_coordinates']['x_max']:.0f}], "
                      f"y=[{obj['zone_coordinates']['y_min']:.0f}, {obj['zone_coordinates']['y_max']:.0f}]")
        
        # Collision Warning Results
        print("\n" + "-" * 70)
        print("COLLISION WARNING")
        print("-" * 70)
        col = result['collision']
        print(f"Collision Risk: {col['collision_risk']}")
        print(f"Reason: {col['reason']}")
        if col['nearest_distance']:
            print(f"Nearest Distance: {col['nearest_distance']:.2f}m")
        
        if col['nearest_object']:
            obj = col['nearest_object']
            print(f"\nNearest Vehicle (with coordinates):")
            print(f"  Class: {obj['class']}")
            print(f"  Score: {obj['score']:.2f}")
            print(f"  BBox: {obj['bbox']}")
            print(f"  Centroid: {obj['centroid']}")
            print(f"  Distance: {obj['distance']:.2f}m")
            print(f"  Lane Zone: x=[{obj['lane_zone']['x_min']:.0f}, {obj['lane_zone']['x_max']:.0f}]")
        
        if col['all_objects']:
            print(f"\nAll Valid Objects in Lane ({len(col['all_objects'])} total):")
            for i, obj in enumerate(col['all_objects'], 1):
                print(f"  {i}. {obj['class']} at {obj['distance']:.2f}m (centroid: {obj['centroid']})")
        
        print("\n" + "=" * 70)
        if result['output_path']:
            print(f"Output saved to: {result['output_path']}")
        print("=" * 70)
    else:
        print(f"Error: {result['message']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

