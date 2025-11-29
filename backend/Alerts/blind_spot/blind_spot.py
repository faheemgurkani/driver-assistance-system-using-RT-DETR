"""
Blind Spot Detection Module (Module 4A)
ADAS Driver Assistance System

This module detects vehicles in the left and right blind spot zones
of a vehicle using object detection results.
"""

# Valid vehicle classes for blind spot detection
VALID_CLASSES = {"car", "bus", "truck", "van", "motorcycle", "bicycle"}

# Minimum confidence threshold
MIN_CONFIDENCE = 0.5

# Minimum bounding box height in pixels
MIN_BBOX_HEIGHT = 40


def compute_centroid(bbox):
    """
    Calculate the centroid (center point) of a bounding box.
    
    Args:
        bbox: Dictionary with keys 'x1', 'y1', 'x2', 'y2' or 
              list/tuple [x1, y1, x2, y2] or dict with 'xtl', 'ytl', 'xbr', 'ybr'
    
    Returns:
        tuple: (cx, cy) - centroid coordinates
    """
    # Handle different bbox formats
    if isinstance(bbox, dict):
        if 'x1' in bbox and 'y1' in bbox:
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        elif 'xtl' in bbox and 'ytl' in bbox:
            x1, y1, x2, y2 = bbox['xtl'], bbox['ytl'], bbox['xbr'], bbox['ybr']
        else:
            raise ValueError("Invalid bbox format. Expected 'x1', 'y1', 'x2', 'y2' or 'xtl', 'ytl', 'xbr', 'ybr'")
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
    else:
        raise ValueError("Invalid bbox format. Expected dict or list/tuple of 4 elements")
    
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    return (cx, cy)


def in_left_blind_spot(cx, cy, W, H):
    """
    Check if a point (centroid) is inside the left blind spot zone.
    
    Left Blind Spot Zone:
        x in [0.0W → 0.25W]
        y in [0.60H → 1.00H]
    
    Args:
        cx: Centroid x-coordinate
        cy: Centroid y-coordinate
        W: Frame width
        H: Frame height
    
    Returns:
        bool: True if point is in left blind spot zone, False otherwise
    """
    x_min = 0.0 * W
    x_max = 0.25 * W
    y_min = 0.60 * H
    y_max = 1.00 * H
    
    return (x_min <= cx <= x_max) and (y_min <= cy <= y_max)


def in_right_blind_spot(cx, cy, W, H):
    """
    Check if a point (centroid) is inside the right blind spot zone.
    
    Right Blind Spot Zone:
        x in [0.75W → 1.00W]
        y in [0.60H → 1.00H]
    
    Args:
        cx: Centroid x-coordinate
        cy: Centroid y-coordinate
        W: Frame width
        H: Frame height
    
    Returns:
        bool: True if point is in right blind spot zone, False otherwise
    """
    x_min = 0.75 * W
    x_max = 1.00 * W
    y_min = 0.60 * H
    y_max = 1.00 * H
    
    return (x_min <= cx <= x_max) and (y_min <= cy <= y_max)


def check_blind_spot(detections, frame_width, frame_height):
    """
    Main function to check for vehicles in blind spot zones.
    
    Args:
        detections: List of detection dictionaries. Each detection should have:
            - 'class' or 'label': class name (str)
            - 'confidence': confidence score (float, 0-1)
            - 'bbox' or bounding box coordinates: 'x1', 'y1', 'x2', 'y2' 
               or 'xtl', 'ytl', 'xbr', 'ybr'
        frame_width: Width of the frame (int)
        frame_height: Height of the frame (int)
    
    Returns:
        dict: {
            "left_blind_spot": bool,
            "right_blind_spot": bool,
            "objects_in_left": int,
            "objects_in_right": int
        }
    """
    # Initialize counters
    objects_in_left = 0
    objects_in_right = 0
    left_blind_spot = False
    right_blind_spot = False
    
    # Process each detection
    for detection in detections:
        # Early continue: Check if class is valid
        class_name = detection.get('class') or detection.get('label', '').lower()
        if class_name not in VALID_CLASSES:
            continue
        
        # Early continue: Check confidence threshold
        confidence = detection.get('confidence', 0.0)
        if confidence < MIN_CONFIDENCE:
            continue
        
        # Extract bounding box
        if 'bbox' in detection:
            bbox = detection['bbox']
        else:
            # Try to construct bbox from detection dict
            bbox = detection
        
        # Calculate bounding box height
        if isinstance(bbox, dict):
            if 'x1' in bbox and 'y1' in bbox:
                bbox_height = abs(bbox['y2'] - bbox['y1'])
            elif 'xtl' in bbox and 'ytl' in bbox:
                bbox_height = abs(bbox['ybr'] - bbox['ytl'])
            else:
                continue
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bbox_height = abs(bbox[3] - bbox[1])
        else:
            continue
        
        # Early continue: Check minimum bounding box height
        if bbox_height < MIN_BBOX_HEIGHT:
            continue
        
        # Calculate centroid
        try:
            cx, cy = compute_centroid(bbox)
        except (ValueError, KeyError):
            continue
        
        # Check if centroid is in blind spot zones
        if in_left_blind_spot(cx, cy, frame_width, frame_height):
            objects_in_left += 1
            left_blind_spot = True
        
        if in_right_blind_spot(cx, cy, frame_width, frame_height):
            objects_in_right += 1
            right_blind_spot = True
    
    return {
        "left_blind_spot": left_blind_spot,
        "right_blind_spot": right_blind_spot,
        "objects_in_left": objects_in_left,
        "objects_in_right": objects_in_right
    }


def draw_blind_spot_zones(frame):
    """
    Draw the left and right blind spot zones on a frame.
    
    Args:
        frame: OpenCV image (numpy array) with shape (H, W, 3)
    
    Returns:
        numpy.ndarray: Frame with blind spot zones drawn as green rectangles
    """
    import cv2
    import numpy as np
    
    H, W = frame.shape[:2]
    
    # Left Blind Spot Zone
    left_x1 = int(0.0 * W)
    left_x2 = int(0.25 * W)
    left_y1 = int(0.60 * H)
    left_y2 = int(1.00 * H)
    
    # Right Blind Spot Zone
    right_x1 = int(0.75 * W)
    right_x2 = int(1.00 * W)
    right_y1 = int(0.60 * H)
    right_y2 = int(1.00 * H)
    
    # Draw rectangles with green border (thickness=2)
    cv2.rectangle(frame, (left_x1, left_y1), (left_x2, left_y2), (0, 255, 0), 2)
    cv2.rectangle(frame, (right_x1, right_y1), (right_x2, right_y2), (0, 255, 0), 2)
    
    return frame

