"""
Collision Warning Module (Module 4B)
ADAS Driver Assistance System

This module estimates the risk of frontal collision using distance heuristics,
lane-relative position, and vehicle filtering.
"""

import math

# Valid vehicle classes for collision detection
VALID_CLASSES = {"car", "bus", "truck", "van", "motorcycle"}

# Minimum confidence threshold
MIN_SCORE = 0.5

# Minimum bounding box height in pixels
MIN_BBOX_HEIGHT = 40

# Default distance scaling constant
DEFAULT_K = 800

# Collision risk thresholds
HIGH_RISK_DISTANCE = 15.0
MEDIUM_RISK_DISTANCE = 25.0


def compute_centroid(bbox):
    """
    Calculate the centroid (center point) of a bounding box.
    
    Args:
        bbox: List/tuple [x1, y1, x2, y2] or dict with 'x1', 'y1', 'x2', 'y2'
    
    Returns:
        tuple: (cx, cy) - centroid coordinates
    """
    if isinstance(bbox, dict):
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
    else:
        raise ValueError("Invalid bbox format. Expected [x1, y1, x2, y2] or dict")
    
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    return (cx, cy)


def estimate_distance(bbox_height, k=DEFAULT_K):
    """
    Estimate distance to object using bounding box height heuristic.
    
    Distance approximation: distance ≈ k / bbox_height
    
    Args:
        bbox_height: Height of the bounding box in pixels (float)
        k: Distance scaling constant (default: 800)
    
    Returns:
        float: Estimated distance to the object
    """
    if bbox_height <= 0:
        return float('inf')
    
    return k / bbox_height


def in_lane(cx, frame_width):
    """
    Check if a centroid is within the frontal lane zone.
    
    Frontal collision can only happen with vehicles directly ahead.
    Lane zone: cx ∈ [0.30 * frame_width → 0.70 * frame_width]
    
    Args:
        cx: Centroid x-coordinate
        frame_width: Width of the frame
    
    Returns:
        bool: True if centroid is in lane, False otherwise
    """
    x_min = 0.30 * frame_width
    x_max = 0.70 * frame_width
    
    return (x_min <= cx <= x_max)


def compute_collision_risk(detections, frame_width, frame_height, k=DEFAULT_K):
    """
    Main function to compute collision risk based on detections.
    
    Args:
        detections: List of detection dictionaries. Each detection should have:
            - 'class': class name (str)
            - 'bbox': [x1, y1, x2, y2] or dict with 'x1', 'y1', 'x2', 'y2'
            - 'score': confidence score (float, 0-1)
        frame_width: Width of the frame (int)
        frame_height: Height of the frame (int)
        k: Distance scaling constant (default: 800)
    
    Returns:
        dict: {
            "collision_risk": "LOW" | "MEDIUM" | "HIGH",
            "nearest_distance": float,
            "object_bbox": [x1, y1, x2, y2] or None
        }
    """
    nearest_distance = float('inf')
    nearest_bbox = None
    
    # Process each detection
    for detection in detections:
        # STEP 1: Vehicle Class Filtering
        class_name = detection.get('class', '').lower()
        if class_name not in VALID_CLASSES:
            continue
        
        # STEP 1: Score filtering
        score = detection.get('score', 0.0)
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
        
        # STEP 1: Bounding box height filtering
        if bbox_height < MIN_BBOX_HEIGHT:
            continue
        
        # STEP 2: Compute Centroid
        try:
            cx, cy = compute_centroid(bbox)
        except (ValueError, KeyError):
            continue
        
        # STEP 3: Lane Filtering
        if not in_lane(cx, frame_width):
            continue
        
        # STEP 4: Distance Estimation
        distance = estimate_distance(bbox_height, k)
        
        # STEP 5: Track Nearest Vehicle
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_bbox = bbox_list
    
    # STEP 6: Determine Collision Risk Level
    if nearest_distance == float('inf'):
        collision_risk = "LOW"
    elif nearest_distance <= HIGH_RISK_DISTANCE:
        collision_risk = "HIGH"
    elif nearest_distance <= MEDIUM_RISK_DISTANCE:
        collision_risk = "MEDIUM"
    else:
        collision_risk = "LOW"
    
    # STEP 7: Return Final Output
    return {
        "collision_risk": collision_risk,
        "nearest_distance": nearest_distance if nearest_distance != float('inf') else None,
        "object_bbox": nearest_bbox
    }


def draw_collision_risk(frame, collision_info):
    """
    Draw collision risk indicator on the frame.
    
    - HIGH risk: Red banner at top
    - MEDIUM risk: Yellow banner at top
    - LOW risk: Hidden or shown subtly (optional)
    
    Args:
        frame: OpenCV image (numpy array) with shape (H, W, 3)
        collision_info: Dictionary from compute_collision_risk() output
    
    Returns:
        numpy.ndarray: Frame with collision risk indicator drawn
    """
    import cv2
    import numpy as np
    
    H, W = frame.shape[:2]
    risk_level = collision_info.get("collision_risk", "LOW")
    
    # Only draw if risk is HIGH or MEDIUM
    if risk_level == "HIGH":
        # Red banner
        color = (0, 0, 255)  # BGR format: red
        text = "HIGH COLLISION RISK!"
        font_scale = 1.2
        thickness = 3
    elif risk_level == "MEDIUM":
        # Yellow banner
        color = (0, 255, 255)  # BGR format: yellow
        text = "MEDIUM COLLISION RISK"
        font_scale = 1.0
        thickness = 2
    else:
        # LOW risk - return frame as is (or draw subtle indicator)
        return frame
    
    # Draw filled rectangle at top
    banner_height = 60
    cv2.rectangle(frame, (0, 0), (W, banner_height), color, -1)
    
    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (W - text_size[0]) // 2
    text_y = (banner_height + text_size[1]) // 2
    
    # Draw text with black outline for better visibility
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    # Optionally draw distance info
    distance = collision_info.get("nearest_distance")
    if distance is not None:
        distance_text = f"Distance: {distance:.1f}m"
        distance_size = cv2.getTextSize(distance_text, font, 0.6, 1)[0]
        distance_x = (W - distance_size[0]) // 2
        distance_y = banner_height + 25
        cv2.putText(frame, distance_text, (distance_x, distance_y), font, 0.6, (255, 255, 255), 1)
    
    return frame

