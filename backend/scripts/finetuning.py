"""
Fine-tuning Script for D2-City Dataset
Loads pretrained RT-DETR weights and fine-tunes on D2-City dataset
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Enable MPS fallback for operations not yet supported on Apple Silicon
# This allows operations like grid_sampler_2d_backward to fall back to CPU
if 'PYTORCH_ENABLE_MPS_FALLBACK' not in os.environ:
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Register custom datasets (suppress warnings)
try:
    # Add backend/src/datasets to path for imports (matching test.py pattern)
    sys.path.insert(0, str(backend_path / "src" / "datasets"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from register_rtdetr import D2CityDatasetRTDETR, SaliencyEnhancedD2CityDatasetRTDETR
except Exception:
    pass  # Silently fail - will use fallback registration

# Import RT-DETR modules from local copy (suppress warnings)
# Import all modules to register classes before YAMLConfig is used
# This must happen before YAMLConfig is used, as it needs registered classes
# Import order matters - backbone and other components must be imported before zoo
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import src.rtdetr.nn.backbone  # This imports and registers backbone classes (PResNet, etc.)
    import src.rtdetr.data  # This imports and registers data transforms and loaders
    import src.rtdetr.optim  # This imports and registers optimizers and EMA
    import src.rtdetr.zoo  # This imports and registers all RT-DETR model classes (RTDETR, HybridEncoder, etc.)
    import src.rtdetr.misc.dist as dist
    from src.rtdetr.core import YAMLConfig
    from src.rtdetr.solver import TASKS


def main(args):
    """Main fine-tuning function using RT-DETR's solver."""
    import torch
    
    # Suppress torch warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torchvision
        torchvision.disable_beta_transforms_warning()
    
    # Check and display device information
    if torch.cuda.is_available():
        device_info = f"CUDA (GPU: {torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_info = "MPS (Apple Silicon GPU)"
    else:
        device_info = "CPU"
    
    # Initialize distributed training if needed
    dist.init_distributed()
    
    if args.seed is not None:
        dist.set_seed(args.seed)
    
    # Validate arguments - cannot use both pretrained checkpoint and resume at the same time
    if args.pretrained_checkpoint and args.resume:
        raise ValueError(
            'Cannot use both --pretrained-checkpoint and --resume at the same time. '
            'Use --pretrained-checkpoint for fine-tuning from pretrained weights, '
            'or --resume to continue training from a checkpoint.'
        )
    
    # Check if pretrained checkpoint exists
    if args.pretrained_checkpoint:
        pretrained_path = args.pretrained_checkpoint
    else:
        # Try to find checkpoint in checkpoints directory
        checkpoint_dir = backend_path / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if checkpoints:
            pretrained_path = str(checkpoints[0])
        else:
            raise ValueError(
                "No pretrained checkpoint found. "
                "Please specify --pretrained-checkpoint or place .pth file in backend/checkpoints/"
            )
    
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
    
    # Load RT-DETR config
    config_path = args.config or str(backend_path / "configs" / "d2city_rtdetr.yml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config and create solver (suppress warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = YAMLConfig(
            config_path,
            tuning=pretrained_path,  # Use tuning flag for fine-tuning
            use_amp=args.amp,
        )
        solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        # Validation only
        solver.train()  # Setup model on device
        actual_device = solver.device
        if actual_device.type == 'mps':
            print(f"✓ MPS (Apple Silicon GPU) active for validation")
        solver.val()
    else:
        # Fine-tuning - setup() is called in solver.fit() -> solver.train()
        # But we can verify device selection here
        actual_device = solver.cfg.device
        if actual_device.type == 'mps':
            print(f"✓ MPS (Apple Silicon GPU) will be used for transfer learning")
        elif actual_device.type == 'cuda':
            print(f"✓ CUDA GPU will be used for transfer learning")
        else:
            print(f"⚠ CPU will be used (MPS/CUDA not available)")
        
        # Gradient checkpointing is completely disabled to avoid MPS inplace operation errors
        print(f"✓ Gradient checkpointing is DISABLED (avoids inplace operation errors on MPS)")
        print(f"  - HybridEncoder: checkpointing disabled")
        print(f"  - RTDETRTransformer: checkpointing disabled")
        
        # Fine-tuning
        solver.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune RT-DETR on D2-City Dataset"
    )
    
    parser.add_argument(
        '--config', '-c', type=str, default=None,
        help='Path to RT-DETR config YAML file (default: configs/d2city_rtdetr.yml)'
    )
    parser.add_argument(
        '--pretrained-checkpoint', '-p', type=str, default=None,
        help='Path to pretrained checkpoint (default: auto-detect from checkpoints/)'
    )
    parser.add_argument(
        '--resume', '-r', type=str, default=None,
        help='Path to checkpoint to resume training from (for continuing training)'
    )
    parser.add_argument(
        '--test-only', action='store_true', default=False,
        help='Run validation only'
    )
    parser.add_argument(
        '--amp', action='store_true', default=False,
        help='Use automatic mixed precision'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed'
    )
    
    args = parser.parse_args()
    main(args)

