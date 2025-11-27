"""
Register D2-City datasets with RT-DETR
Import this module before using RT-DETR configs to register datasets
"""

import sys
import os
from pathlib import Path

# Add backend/src to path for absolute imports
backend_src_path = Path(__file__).parent.parent
if str(backend_src_path) not in sys.path:
    sys.path.insert(0, str(backend_src_path))

# Add RT-DETR to path (local copy)
rtdetr_path = backend_src_path / "rtdetr"
if rtdetr_path.exists() and str(backend_src_path) not in sys.path:
    sys.path.insert(0, str(backend_src_path))

# Import RT-DETR compatible datasets (this registers them)
# Use relative imports since we're in the datasets directory
try:
    from .d2city_dataset_rtdetr import D2CityDatasetRTDETR
except Exception:
    # Fallback to absolute import
    try:
        from src.datasets.d2city_dataset_rtdetr import D2CityDatasetRTDETR
    except Exception:
        D2CityDatasetRTDETR = None

try:
    from .saliency_enhanced_d2city_dataset_rtdetr import SaliencyEnhancedD2CityDatasetRTDETR
except Exception:
    # Fallback to absolute import
    try:
        from src.datasets.saliency_enhanced_d2city_dataset_rtdetr import SaliencyEnhancedD2CityDatasetRTDETR
    except Exception:
        SaliencyEnhancedD2CityDatasetRTDETR = None

# Ensure classes are exported
__all__ = ['D2CityDatasetRTDETR', 'SaliencyEnhancedD2CityDatasetRTDETR']
