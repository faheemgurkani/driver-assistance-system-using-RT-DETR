"""
Register D2-City dataset with RT-DETR
Import this module before using RT-DETR configs to register dataset
"""

import sys
import os

# Add RT-DETR to path (local copy)
# From backend/src/datasets/ -> backend/src/rtdetr/
rtdetr_path = os.path.join(os.path.dirname(__file__), '..', 'rtdetr')
if os.path.exists(rtdetr_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import RT-DETR compatible dataset (this registers it)
try:
    from .d2city_dataset_rtdetr import D2CityDatasetRTDETR
    print("✓ Registered D2CityDatasetRTDETR with RT-DETR")
except Exception as e:
    print(f"Warning: Could not register RT-DETR dataset: {e}")

__all__ = ['D2CityDatasetRTDETR']
