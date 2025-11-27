# Transfer Learning

## Overview

Transfer learning fine-tunes a pretrained RT-DETR model (trained on COCO) on the D2-City dataset without modifying the model architecture. All runs use the **ResNet-101-VD COCO checkpoint (`rtdetr_r101vd_6x_coco.pth`)**; lighter backbones are only kept for historical reference.

## Architecture Preservation

**No architecture changes are made during transfer learning.**

- Model architecture remains identical to pretrained model
- Classification head keeps 80 classes (COCO classes)
- All layers are fine-tuned (no frozen layers by default)

## Class Mapping Strategy

Instead of changing the classification head, **label mapping** is used to map D2-City labels to COCO classes.

### D2-City to COCO Mapping

**File**: `src/utils/d2city_annotation_parser.py`

```python
D2CITY_TO_COCO_LABEL_MAP = {
    'person': 0,        # COCO class 0: person
    'bicycle': 1,       # COCO class 1: bicycle
    'car': 2,           # COCO class 2: car
    'motorcycle': 3,    # COCO class 3: motorcycle
    'bus': 5,           # COCO class 5: bus
    'truck': 7,         # COCO class 7: truck
    'van': 2,           # Maps to car
    'forklift': 7,      # Maps to truck
    'open-tricycle': 3, # Maps to motorcycle
    'closed-tricycle': 3, # Maps to motorcycle
    'large-block': -1,  # Ignored (not in COCO)
    'small-block': -1,  # Ignored (not in COCO)
}
```

### Mapping Flow

1. **Dataset**: D2-City labels → COCO class indices (0-79)
2. **Training**: Model predicts 80 COCO classes
3. **Evaluation**: Class indices → COCO category IDs (1-90) via `remap_mscoco_category`

### Evaluation Consistency Fix

- Ground-truth labels are remapped inside `backend/src/rtdetr/data/coco/coco_utils.py` by indexing `mscoco_label2category` before passing annotations to the COCO API.
- This change ensures that both predictions **and** references use COCO category IDs, preventing the “zero AP” issue observed when category IDs and indices were mixed.

## Transfer Learning Process

### 1. Model Loading

```python
cfg = YAMLConfig(
    config_path,
    tuning=pretrained_checkpoint_path,  # COCO pretrained weights
    use_amp=True
)
```

**What happens:**
- Model architecture created from YAML
- Pretrained COCO weights loaded via `load_tuning_state()`
- Mismatched keys skipped (`strict=False`)
- All layers initialized with COCO weights

### 2. Training Setup

```python
solver = TASKS['detection'](cfg)
solver.fit()  # Starts fine-tuning
```

**Training parameters:**
- Learning rates: `[1e-5, 1e-4, 1e-4, 1e-4]` (backbone, encoder, decoder, others)
- Optimizer: AdamW with weight decay
- Loss: Varied Focal Loss + GIoU + L1
- Epochs: Configurable (default: 1 for testing)

### 3. Fine-tuning Details

**All components are fine-tuned:**
- Backbone (PResNet): Feature extraction layers
- HybridEncoder: Multi-scale feature encoding
- RTDETRTransformer: Object detection decoder

**No frozen layers** - full end-to-end training.

## Label Format Conversion

### During Training

- **Input labels**: COCO class indices (0-79) from D2-City mapping
- **Model output**: 80 COCO class logits
- **Loss calculation**: Uses mapped class indices

### During Evaluation

- **Model predictions**: COCO class indices (0-79)
- **Remapping**: `remap_mscoco_category=True` converts to COCO category IDs (1-90)
- **Ground truth**: Also remapped to category IDs in `convert_to_coco_api()`
- **Evaluation**: COCO evaluator uses category IDs

## Key Files

- **Fine-tuning script**: `scripts/finetuning.py`
- **Label mapping**: `src/utils/d2city_annotation_parser.py`
- **Model loading**: `src/rtdetr/solver/solver.py` (`load_tuning_state`)
- **Config**: `configs/d2city_saliency_enhanced_rtdetr_r101vd.yml`

## Advantages of Label Mapping

1. **No architecture changes**: Reuse pretrained model as-is
2. **Flexible**: Easy to add new label mappings
3. **Compatible**: Works with COCO evaluation metrics
4. **Efficient**: No need to retrain classification head from scratch

## Summary

- ✅ Architecture: **Unchanged** (80 COCO classes)
- ✅ Strategy: **Label mapping** (D2-City → COCO classes)
- ✅ Fine-tuning: **End-to-end** (all layers trainable)
- ✅ Evaluation: **COCO format** (category IDs via remapping)

