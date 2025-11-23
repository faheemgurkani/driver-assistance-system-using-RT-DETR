"""
Test Script for Driver Assistance System using RT-DETR
Inspired by rtdetr_pytorch/tools/train.py (test-only mode)
Runs evaluation/validation on test set
"""

import os 
import sys 
from pathlib import Path
import argparse

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Register custom datasets
try:
    sys.path.insert(0, str(backend_path / "src" / "datasets"))
    from register_rtdetr import D2CityDatasetRTDETR, SaliencyEnhancedD2CityDatasetRTDETR
    print("✓ Registered D2-City datasets with RT-DETR")
    print("  - D2CityDatasetRTDETR (original videos with preprocessing)")
    print("  - SaliencyEnhancedD2CityDatasetRTDETR (pre-processed enhanced frames)")
except Exception as e:
    print(f"Warning: Could not register datasets: {e}")

# Import RT-DETR modules from local copy
import src.rtdetr.misc.dist as dist 
from src.rtdetr.core import YAMLConfig 
from src.rtdetr.solver import TASKS


def main(args) -> None:
    """
    Main test/evaluation function.
    
    Flow:
    1. Load config -> establishes model architecture from YAML
    2. Load checkpoint weights into established model
    3. Run evaluation on test/validation set
    """
    # Initialize distributed training (needed for RT-DETR)
    dist.init_distributed()
    
    if args.seed is not None:
        dist.set_seed(args.seed)

    # Load RT-DETR config
    # This establishes the model architecture from YAML config
    # Supports both scenarios:
    # - Original D2-City: d2city_rtdetr.yml (data loading + preprocessing)
    # - Saliency-Enhanced: d2city_saliency_enhanced_rtdetr.yml (data loading only)
    if args.config:
        config_path = args.config
    elif args.saliency_enhanced:
        config_path = str(backend_path / "configs" / "d2city_saliency_enhanced_rtdetr.yml")
    else:
        config_path = str(backend_path / "configs" / "d2city_rtdetr.yml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading config from: {config_path}")
    print("Step 1: Establishing model architecture from config...")
    
    # Checkpoint is required for testing
    if not args.resume:
        # Try to find checkpoint in output directory
        output_dir = backend_path.parent / "output" / "d2city_rtdetr_r50vd"
        checkpoint_path = output_dir / "checkpoint.pth"
        if checkpoint_path.exists():
            args.resume = str(checkpoint_path)
            print(f"Auto-detected checkpoint: {args.resume}")
        else:
            raise ValueError(
                "Checkpoint required for testing. "
                "Specify --resume or ensure checkpoint exists in output directory."
            )
    
    cfg = YAMLConfig(
        config_path,
        resume=args.resume,  # Load checkpoint weights
        use_amp=args.amp,
    )
    
    # Step 2: Model is established via cfg.model property
    # Checkpoint weights will be loaded in solver.setup()
    print("Step 2: Model architecture established")
    print(f"  → Will load weights from checkpoint: {args.resume}")

    # Step 3: Create solver (handles model setup, weight loading)
    print("Step 3: Creating solver...")
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    # Step 4: Run evaluation
    print("Step 4: Running evaluation on test set...")
    solver.val()
    print("Evaluation completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Test/Evaluate RT-DETR on D2-City Dataset (inspired by rtdetr_pytorch/tools/train.py)"
    )
    
    parser.add_argument(
        '--config', '-c', type=str, default=None,
        help='Path to RT-DETR config YAML file. '
             'Defaults: d2city_rtdetr.yml (original) or d2city_saliency_enhanced_rtdetr.yml (if --saliency-enhanced)'
    )
    parser.add_argument(
        '--saliency-enhanced', action='store_true', default=False,
        help='Use saliency-enhanced D2-City dataset (pre-processed frames, no preprocessing needed)'
    )
    parser.add_argument(
        '--resume', '-r', type=str, default=None,
        help='Path to checkpoint to load for evaluation'
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

