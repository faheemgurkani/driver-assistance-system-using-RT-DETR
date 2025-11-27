#!/usr/bin/env python3
"""
Verify Data Structure and Annotations
Checks that:
1. Enhanced frames are organized by video_id
2. XML annotations match video_ids
3. Frame indices match between images and XML files
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path / "src"))

from utils.d2city_annotation_parser import parse_d2city_xml, get_annotations_for_image

def verify_data_structure():
    """Verify the data structure matches expected format."""
    print("=" * 80)
    print("VERIFYING DATA STRUCTURE")
    print("=" * 80)
    
    enhanced_frames_dir = backend_path / "data" / "T2-saliency" / "enhanced_frames"
    annotations_dir = backend_path / "data" / "annotations"
    
    # 1. Check enhanced_frames directory structure
    print("\n1. Checking enhanced_frames directory structure...")
    if not enhanced_frames_dir.exists():
        print(f"   ERROR: {enhanced_frames_dir} does not exist!")
        return False
    
    video_dirs = [d for d in enhanced_frames_dir.iterdir() if d.is_dir()]
    print(f"   Found {len(video_dirs)} video directories")
    
    if len(video_dirs) == 0:
        print("   ERROR: No video directories found!")
        return False
    
    # Check structure of first few directories
    print("\n   Sample video directories:")
    for video_dir in sorted(video_dirs)[:5]:
        image_files = list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png"))
        print(f"   - {video_dir.name}: {len(image_files)} images")
        if len(image_files) > 0:
            print(f"     Sample files: {image_files[0].name}, {image_files[-1].name if len(image_files) > 1 else ''}")
    
    # 2. Check annotations directory
    print("\n2. Checking annotations directory...")
    if not annotations_dir.exists():
        print(f"   ERROR: {annotations_dir} does not exist!")
        return False
    
    xml_files = list(annotations_dir.glob("*.xml"))
    print(f"   Found {len(xml_files)} XML annotation files")
    
    if len(xml_files) == 0:
        print("   ERROR: No XML files found!")
        return False
    
    # 3. Verify video_id matching
    print("\n3. Verifying video_id matching between frames and annotations...")
    video_ids_from_frames = {d.name for d in video_dirs}
    video_ids_from_annotations = {f.stem for f in xml_files}
    
    matched = video_ids_from_frames & video_ids_from_annotations
    only_in_frames = video_ids_from_frames - video_ids_from_annotations
    only_in_annotations = video_ids_from_annotations - video_ids_from_frames
    
    print(f"   Matched video_ids: {len(matched)}")
    print(f"   Only in frames (no annotation): {len(only_in_frames)}")
    print(f"   Only in annotations (no frames): {len(only_in_annotations)}")
    
    if len(matched) == 0:
        print("   ERROR: No matching video_ids found!")
        return False
    
    if len(only_in_frames) > 0:
        print(f"   WARNING: {len(only_in_frames)} video_ids have frames but no annotations")
        print(f"   Sample: {list(only_in_frames)[:5]}")
    
    if len(only_in_annotations) > 0:
        print(f"   WARNING: {len(only_in_annotations)} video_ids have annotations but no frames")
        print(f"   Sample: {list(only_in_annotations)[:5]}")
    
    # 4. Verify frame index matching
    print("\n4. Verifying frame index matching...")
    test_video_ids = list(matched)[:5]  # Test first 5 matched video_ids
    
    frame_mismatches = []
    total_tested = 0
    total_with_boxes = 0
    
    for video_id in test_video_ids:
        video_dir = enhanced_frames_dir / video_id
        xml_path = annotations_dir / f"{video_id}.xml"
        
        if not xml_path.exists():
            continue
        
        # Parse XML to get frame indices
        frame_annotations = parse_d2city_xml(str(xml_path))
        xml_frame_indices = set(frame_annotations.keys())
        
        # Get image frame indices
        image_files = sorted(video_dir.glob("*.jpg")) + sorted(video_dir.glob("*.png"))
        image_frame_indices = set()
        for img_file in image_files:
            try:
                frame_idx = int(img_file.stem)
                image_frame_indices.add(frame_idx)
            except ValueError:
                continue
        
        # Check overlap
        overlapping = xml_frame_indices & image_frame_indices
        only_in_xml = xml_frame_indices - image_frame_indices
        only_in_images = image_frame_indices - xml_frame_indices
        
        total_tested += len(image_frame_indices)
        
        # Test loading annotations for a few frames
        for img_file in image_files[:3]:  # Test first 3 images
            try:
                frame_idx = int(img_file.stem)
                w, h = 640, 360  # Default size, will be updated when image is loaded
                boxes, labels = get_annotations_for_image(
                    str(xml_path), img_file.name, w, h
                )
                if len(boxes) > 0:
                    total_with_boxes += 1
            except Exception as e:
                frame_mismatches.append((video_id, img_file.name, str(e)))
        
        print(f"   {video_id}:")
        print(f"     XML frames: {len(xml_frame_indices)}, Image frames: {len(image_frame_indices)}")
        print(f"     Overlapping: {len(overlapping)}")
        if len(overlapping) > 0:
            print(f"     Sample overlapping frames: {sorted(list(overlapping))[:5]}")
        if len(only_in_xml) > 0:
            print(f"     WARNING: {len(only_in_xml)} frames in XML but not in images")
        if len(only_in_images) > 0:
            print(f"     WARNING: {len(only_in_images)} frames in images but not in XML")
    
    print(f"\n   Tested {total_tested} frames, {total_with_boxes} have annotations")
    
    if frame_mismatches:
        print(f"\n   ERRORS loading annotations:")
        for video_id, img_file, error in frame_mismatches[:5]:
            print(f"     {video_id}/{img_file}: {error}")
        return False
    
    # 5. Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"✓ Enhanced frames directory structure: OK ({len(video_dirs)} video directories)")
    print(f"✓ Annotations directory: OK ({len(xml_files)} XML files)")
    print(f"✓ Video ID matching: {len(matched)} matched, {len(only_in_frames)} only in frames, {len(only_in_annotations)} only in annotations")
    print(f"✓ Frame index matching: Tested {total_tested} frames, {total_with_boxes} have annotations")
    
    if len(matched) > 0 and total_with_boxes > 0:
        print("\n✓ DATA STRUCTURE VERIFICATION: PASSED")
        return True
    else:
        print("\n✗ DATA STRUCTURE VERIFICATION: FAILED")
        return False


if __name__ == "__main__":
    success = verify_data_structure()
    sys.exit(0 if success else 1)

