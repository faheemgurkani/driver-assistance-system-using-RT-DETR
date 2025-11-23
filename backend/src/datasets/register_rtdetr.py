"""
Register D2-City datasets with RT-DETR
Import this module before using RT-DETR configs to register datasets
"""

import sys
import os

# Add RT-DETR to path (local copy)
# From backend/src/datasets/ -> backend/src/rtdetr/
rtdetr_path = os.path.join(os.path.dirname(__file__), '..', 'rtdetr')
if os.path.exists(rtdetr_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import RT-DETR compatible datasets (this registers them)
try:
    from .d2city_dataset_rtdetr import D2CityDatasetRTDETR
    print("✓ Registered D2CityDatasetRTDETR with RT-DETR")
except Exception as e:
    print(f"Warning: Could not register D2CityDatasetRTDETR: {e}")

try:
    from .saliency_enhanced_d2city_dataset_rtdetr import SaliencyEnhancedD2CityDatasetRTDETR
    print("✓ Registered SaliencyEnhancedD2CityDatasetRTDETR with RT-DETR")
except Exception as e:
    print(f"Warning: Could not register SaliencyEnhancedD2CityDatasetRTDETR: {e}")

__all__ = ['D2CityDatasetRTDETR', 'SaliencyEnhancedD2CityDatasetRTDETR']
