"""
D2-City Annotation Parser

Parses D2-City XML annotation files and extracts bounding boxes for specific frames.
D2-City uses a custom label set that needs to be mapped to COCO classes.
"""

import xml.etree.ElementTree as ET
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# D2-City label to COCO class ID mapping
# D2-City labels: car,van,bus,truck,person,bicycle,motorcycle,open-tricycle,closed-tricycle,forklift,large-block,small-block
# COCO classes: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, ...
D2CITY_TO_COCO_LABEL_MAP = {
    'car': 2,           # COCO class 2: car
    'van': 2,           # Map van to car
    'bus': 5,           # COCO class 5: bus
    'truck': 7,         # COCO class 7: truck
    'person': 0,        # COCO class 0: person
    'bicycle': 1,       # COCO class 1: bicycle
    'motorcycle': 3,    # COCO class 3: motorcycle
    'open-tricycle': 3, # Map to motorcycle
    'closed-tricycle': 3, # Map to motorcycle
    'forklift': 7,      # Map to truck
    'large-block': -1,  # Ignore (not in COCO)
    'small-block': -1,  # Ignore (not in COCO)
}

# Default D2-City image dimensions (can be overridden)
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 360


def parse_d2city_xml(xml_path: str) -> Dict[int, List[Dict]]:
    """
    Parse a D2-City XML annotation file and return a dictionary mapping frame numbers to annotations.
    
    Args:
        xml_path: Path to the XML annotation file
        
    Returns:
        Dictionary mapping frame numbers to lists of annotation dicts.
        Each annotation dict has: 'label', 'xtl', 'ytl', 'xbr', 'ybr', 'occluded', 'cut'
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    frame_annotations = defaultdict(list)
    
    # Parse all tracks
    for track in root.findall('track'):
        label = track.get('label', '')
        track_id = track.get('id', '')
        
        # Parse all boxes in this track
        for box in track.findall('box'):
            frame_num = int(box.get('frame', '0'))
            xtl = float(box.get('xtl', '0'))
            ytl = float(box.get('ytl', '0'))
            xbr = float(box.get('xbr', '0'))
            ybr = float(box.get('ybr', '0'))
            occluded = box.get('occluded', 'no')
            cut = box.get('cut', 'no')
            
            annotation = {
                'label': label,
                'xtl': xtl,
                'ytl': ytl,
                'xbr': xbr,
                'ybr': ybr,
                'occluded': occluded,
                'cut': cut,
                'track_id': track_id,
            }
            
            frame_annotations[frame_num].append(annotation)
    
    return dict(frame_annotations)


def get_annotations_for_image(
    xml_path: str,
    image_filename: str,
    image_width: int,
    image_height: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get bounding boxes and labels for a specific image from an XML annotation file.
    
    Args:
        xml_path: Path to the XML annotation file
        image_filename: Name of the image file (e.g., "000000.jpg" or "000742.jpg")
        image_width: Width of the image (used for validation)
        image_height: Height of the image (used for validation)
        
    Returns:
        Tuple of (boxes, labels) where:
        - boxes: Tensor of shape (N, 4) with format [x1, y1, x2, y2] in pixel coordinates
        - labels: Tensor of shape (N,) with COCO class IDs
    """
    # Extract frame number from image filename
    # Filenames are like "000000.jpg", "000742.jpg", etc.
    try:
        frame_num = int(image_filename.split('.')[0])
    except (ValueError, IndexError):
        # If we can't parse the frame number, return empty annotations
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
    
    # Parse XML file
    try:
        frame_annotations = parse_d2city_xml(xml_path)
    except Exception as e:
        # If parsing fails, return empty annotations
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
    
    # Get annotations for this frame
    annotations = frame_annotations.get(frame_num, [])
    
    if len(annotations) == 0:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
    
    # Convert annotations to boxes and labels
    boxes = []
    labels = []
    
    for ann in annotations:
        label = ann['label']
        coco_class_id = D2CITY_TO_COCO_LABEL_MAP.get(label, -1)
        
        # Skip labels that don't map to COCO classes
        if coco_class_id == -1:
            continue
        
        # Extract box coordinates
        xtl = ann['xtl']
        ytl = ann['ytl']
        xbr = ann['xbr']
        ybr = ann['ybr']
        
        # Validate and clamp box coordinates
        xtl = max(0, min(xtl, image_width))
        ytl = max(0, min(ytl, image_height))
        xbr = max(0, min(xbr, image_width))
        ybr = max(0, min(ybr, image_height))
        
        # Ensure valid box (x2 > x1, y2 > y1)
        if xbr > xtl and ybr > ytl:
            boxes.append([xtl, ytl, xbr, ybr])
            labels.append(coco_class_id)
    
    if len(boxes) == 0:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
    
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)
    
    return boxes_tensor, labels_tensor

