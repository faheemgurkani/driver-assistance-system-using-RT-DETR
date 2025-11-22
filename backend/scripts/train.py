"""
Training Script for Driver Assistance System using RT-DETR
Inspired by rtdetr_pytorch/tools/train.py
Establishes model from config, loads pretrained weights, then trains
"""

import os
import sys
import argparse
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Register custom datasets
try:
    sys.path.insert(0, str(backend_path / "src" / "datasets"))
    from register_rtdetr import D2CityDatasetRTDETR
    print("✓ Registered D2-City dataset with RT-DETR")
except Exception as e:
    print(f"Warning: Could not register dataset: {e}")

# Import RT-DETR modules from local copy
import src.rtdetr.misc.dist as dist 
from src.rtdetr.core import YAMLConfig 
from src.rtdetr.solver import TASKS


def main(args) -> None:
    """
    Main training function.
    
    Flow:
    1. Load config -> establishes model architecture from YAML
    2. If tuning: load pretrained weights into established model
    3. If resume: load full checkpoint (model + optimizer + scheduler)
    4. Train using RT-DETR's solver
    """
    # Initialize distributed training
    dist.init_distributed()
    
    if args.seed is not None:
        dist.set_seed(args.seed)

    # Validate arguments
    assert not all([args.tuning, args.resume]), \
        'Only support from_scratch or resume or tuning at one time'

    # Load RT-DETR config
    # This establishes the model architecture from YAML config
    config_path = args.config or str(backend_path / "configs" / "d2city_rtdetr.yml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading config from: {config_path}")
    print("Step 1: Establishing model architecture from config...")
    
    cfg = YAMLConfig(
        config_path,
        resume=args.resume,  # For resuming training
        use_amp=args.amp,
        tuning=args.tuning   # For fine-tuning from pretrained weights
    )
    
    # Step 2: Model is established via cfg.model property
    # If tuning is set, pretrained weights will be loaded in solver.setup()
    # If resume is set, full checkpoint will be loaded in solver.train()
    print("Step 2: Model architecture established")
    if args.tuning:
        print(f"  → Will load pretrained weights from: {args.tuning}")
    elif args.resume:
        print(f"  → Will resume from checkpoint: {args.resume}")
    else:
        print("  → Training from scratch")

    # Step 3: Create solver (handles model setup, weight loading, training)
    print("Step 3: Creating solver...")
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        # Validation only
        print("Running validation only...")
        solver.val()
    else:
        # Training
    print("Starting training...")
        solver.fit()
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train RT-DETR on D2-City Dataset (inspired by rtdetr_pytorch/tools/train.py)"
    )
    
    parser.add_argument(
        '--config', '-c', type=str, default=None,
        help='Path to RT-DETR config YAML file (default: configs/d2city_rtdetr.yml)'
    )
    parser.add_argument(
        '--resume', '-r', type=str, default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--tuning', '-t', type=str, default=None,
        help='Path to pretrained checkpoint for fine-tuning'
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
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    main(args)

