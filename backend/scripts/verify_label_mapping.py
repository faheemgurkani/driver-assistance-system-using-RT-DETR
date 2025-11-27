#!/usr/bin/env python3
"""
Verify label mapping between D2-City labels, COCO class indices, and COCO category IDs.
"""

# D2-City to COCO class index mapping (from d2city_annotation_parser.py)
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

# COCO category ID to class index mapping (from coco_dataset.py)
# COCO categories: 1=person, 2=bicycle, 3=car, 4=motorcycle, 5=airplane, 6=bus, 7=train, 8=truck, ...
mscoco_category2label = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
    22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
    35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39,
    46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59,
    67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69,
    80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
}

# Reverse mapping: class index to category ID
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}

print("=" * 80)
print("LABEL MAPPING VERIFICATION")
print("=" * 80)

print("\n1. D2-City to COCO Class Index Mapping (Dataset Output):")
print("-" * 80)
for d2_label, coco_class_idx in sorted(D2CITY_TO_COCO_LABEL_MAP.items()):
    if coco_class_idx != -1:
        print(f"  {d2_label:20s} → COCO class index: {coco_class_idx:2d}")

print("\n2. COCO Class Index to COCO Category ID Mapping (For Evaluation):")
print("-" * 80)
print("  (Used when remap_mscoco_category=True)")
d2_labels_used = [label for label, idx in D2CITY_TO_COCO_LABEL_MAP.items() if idx != -1]
for d2_label in sorted(d2_labels_used):
    coco_class_idx = D2CITY_TO_COCO_LABEL_MAP[d2_label]
    coco_category_id = mscoco_label2category.get(coco_class_idx, "NOT FOUND")
    print(f"  {d2_label:20s} → class idx {coco_class_idx:2d} → category ID {coco_category_id}")

print("\n3. Expected Label Flow:")
print("-" * 80)
print("  Dataset (D2-City) → COCO class index (0-79) → COCO category ID (1-90)")
print("  Example: 'car' → class idx 2 → category ID 3")

print("\n4. Current Issue:")
print("-" * 80)
print("  ❌ Dataset outputs: COCO class indices (0, 1, 2, 3, 5, 7)")
print("  ❌ convert_to_coco_api uses these DIRECTLY as category_id (WRONG!)")
print("  ✅ Postprocessor remaps predictions: class idx → category ID (CORRECT)")
print("  ⚠️  MISMATCH: Ground truth has class indices, predictions have category IDs")
print("  ⚠️  This causes zero AP because labels don't match!")

print("\n5. Required Fix:")
print("-" * 80)
print("  In convert_to_coco_api (coco_utils.py line 176):")
print("    Change: ann['category_id'] = labels[i]")
print("    To:     ann['category_id'] = mscoco_label2category[labels[i]]")
print("=" * 80)
