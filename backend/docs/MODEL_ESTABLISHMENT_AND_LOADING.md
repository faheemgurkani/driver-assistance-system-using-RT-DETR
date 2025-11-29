# Model Establishment and Loading

## Overview

RT-DETR models are established from YAML configuration files and loaded with pretrained weights through RT-DETR's `YAMLConfig` system. All production runs in this repository target the **ResNet-101-VD** backbone and load the official COCO checkpoint `rtdetr_r101vd_6x_coco.pth`.

## Model Architecture Establishment

### 1. YAML Configuration System

The model architecture is defined in YAML config files (e.g., `d2city_rtdetr.yml`):

```yaml
__include__:
  - ./rtdetr_base/runtime.yml
  - ./rtdetr_base/rtdetr/include/dataloader.yml
  - ./rtdetr_base/rtdetr/include/optimizer.yml
  - ./rtdetr_base/rtdetr/include/rtdetr_r50vd.yml

task: detection
num_classes: 80

PResNet:
  depth: 101  # default backbone (R101VD) used throughout the project

HybridEncoder:
  hidden_dim: 256
  dim_feedforward: 1024

RTDETRTransformer:
  feat_channels: [256, 256, 256]
```

### 2. Component Registration

Before loading configs, all RT-DETR components must be registered:

```python
import src.rtdetr.nn.backbone  # Registers PResNet, etc.
import src.rtdetr.data         # Registers transforms, datasets
import src.rtdetr.optim         # Registers optimizers
import src.rtdetr.zoo           # Registers RTDETR, HybridEncoder, etc.
```

> **Why this matters**: when running inference outside the training solver (e.g., `scripts/inference.py` or the FastAPI API), the `_pymodule` registry must be populated manually. The inference script now imports `src.rtdetr.nn.backbone`, `src.rtdetr.data`, `src.rtdetr.optim`, and `src.rtdetr.zoo` before instantiating `YAMLConfig` to prevent the `KeyError: '_pymodule'` seen previously.

### 3. Model Creation

```python
from src.rtdetr.core import YAMLConfig

# Create config (establishes architecture)
cfg = YAMLConfig(config_path, tuning=checkpoint_path)

# Model is accessible via cfg.model
model = cfg.model  # RTDETR instance with full architecture
```

## Checkpoint Loading

### Pretrained Weights (Fine-tuning)

```python
cfg = YAMLConfig(
    config_path,
    tuning=pretrained_checkpoint_path,  # Path to .pth file
    use_amp=True
)

# Model weights are automatically loaded via load_tuning_state()
# This loads matching weights and skips mismatched keys (strict=False)
```

**Process:**
1. Model architecture created from YAML
2. Checkpoint loaded: `torch.load(checkpoint_path, map_location='cpu')`
3. State dict extracted: `state['ema']['module']` or `state['model']`
4. Weights loaded: `model.load_state_dict(state, strict=False)`

### Resuming Training

```python
cfg = YAMLConfig(
    config_path,
    resume=checkpoint_path  # Full training state
)

# Loads: model, optimizer, lr_scheduler, ema, scaler, last_epoch
```

## Model Components

### Architecture Structure

```
RTDETR
├── backbone (PResNet-101-VD, primary production model)
├── encoder (HybridEncoder)
│   ├── input_proj (conv layers)
│   ├── encoder (TransformerEncoder layers)
│   └── fpn (Feature Pyramid Network)
└── decoder (RTDETRTransformer)
    ├── denoising (optional)
    └── decoder (TransformerDecoder)
```

### Key Parameters

- **num_classes**: 80 (COCO classes) - fixed, not changed during transfer learning
- **remap_mscoco_category**: True - remaps class indices to COCO category IDs for evaluation
- **Backbone depth**: 101 (R101VD) in production configs (legacy R50 files remain for reference)

## Device Selection

Automatic device detection:

```python
# Priority: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'  # Apple Silicon GPU
else:
    device = 'cpu'
```

## Deploy Mode

For inference, models are converted to deploy mode:

```python
model = cfg.model.deploy()  # Removes training-specific components
postprocessor = cfg.postprocessor.deploy()
```

## Files

- **Config files**: `backend/configs/*.yml`
- **Checkpoints**: `backend/checkpoints/*.pth`
- **Model loading**: `backend/scripts/finetuning.py`
- **Solver**: `backend/src/rtdetr/solver/solver.py` (load_tuning_state)

