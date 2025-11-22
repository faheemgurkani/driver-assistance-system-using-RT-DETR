"""
Fine-tuning Script for D2-City Dataset
Loads pretrained RT-DETR weights and fine-tunes on D2-City dataset
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


def main(args):
    """Main fine-tuning function using RT-DETR's solver."""
    # Initialize distributed training if needed
    dist.init_distributed()
    
    if args.seed is not None:
        dist.set_seed(args.seed)
    
    # Validate arguments
    assert not all([args.tuning, args.resume]), \
        'Only support from_scratch or resume or tuning at one time'
    
    # Check if pretrained checkpoint exists
    if args.pretrained_checkpoint:
        pretrained_path = args.pretrained_checkpoint
    else:
        # Try to find checkpoint in checkpoints directory
        checkpoint_dir = backend_path / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if checkpoints:
            pretrained_path = str(checkpoints[0])
            print(f"Using pretrained checkpoint: {pretrained_path}")
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
    
    print(f"Loading config from: {config_path}")
    print(f"Fine-tuning from checkpoint: {pretrained_path}")
    
    # Step 1: Establish model architecture from config
    # This creates the model structure (backbone, encoder, decoder) from YAML
    print("Step 1: Establishing model architecture from config...")
    cfg = YAMLConfig(
        config_path,
        tuning=pretrained_path,  # Use tuning flag for fine-tuning
        use_amp=args.amp,
    )
    print("✓ Model architecture established")
    
    # Step 2: Create solver (will load pretrained weights in setup())
    # The solver.setup() method will call load_tuning_state() to load weights
    print("Step 2: Creating solver (pretrained weights will be loaded in setup())...")
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        # Validation only
        print("Running validation only...")
        solver.val()
    else:
        # Fine-tuning
        print("Starting fine-tuning on D2-City dataset...")
        solver.fit()
        print("Fine-tuning completed!")


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

