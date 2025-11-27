"""
Pre-training (Fine-tuning) Script for Driver Assistance System
Uses RT-DETR's checkpoint loading mechanism for fine-tuning
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add RT-DETR to path
rtdetr_path = project_root.parent / "rtdetr_pytorch"
if rtdetr_path.exists():
    sys.path.insert(0, str(rtdetr_path))

# Register custom datasets before importing RT-DETR modules
try:
    datasets_path = project_root / "src" / "datasets"
    sys.path.insert(0, str(datasets_path))
    from register_rtdetr import KITTIDatasetRTDETR, D2CityDatasetRTDETR
    print("✓ Registered custom datasets with RT-DETR")
except Exception as e:
    print(f"Warning: Could not register datasets: {e}")

# Import RT-DETR modules
import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS


def main(args):
    """Main pre-training function using RT-DETR's solver."""
    # Initialize distributed training if needed
    dist.init_distributed()
    
    if args.seed is not None:
        dist.set_seed(args.seed)
    
    # Validate arguments
    if args.resume and args.tuning:
        raise ValueError("Cannot use both --resume and --tuning. Use --tuning for fine-tuning from checkpoint.")
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Use default checkpoint
        checkpoint_path = project_root / "checkpoints" / "rtdetr_r101vd_6x_coco.pth"
        if not checkpoint_path.exists():
            # Try alternative location
            alt_path = rtdetr_path / "checkpoints" / "rtdetr_r101vd_6x_coco.pth"
            if alt_path.exists():
                checkpoint_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found. Please provide --checkpoint or place "
                    f"rtdetr_r101vd_6x_coco.pth in {project_root / 'checkpoints'}"
                )
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Load RT-DETR config
    # For fine-tuning, use 'tuning' parameter (loads only model weights, skips mismatched keys)
    cfg = YAMLConfig(
        args.config,
        tuning=str(checkpoint_path),  # Use tuning for fine-tuning
        use_amp=args.amp,
        resume=args.resume if args.resume else None
    )
    
    print(f"Fine-tuning from checkpoint: {checkpoint_path}")
    print(f"Config: {args.config}")
    print(f"Output directory: {cfg.output_dir}")
    
    # Create solver (DetSolver for detection tasks)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        # Validation only
        print("Running validation only...")
        solver.val()
    else:
        # Training (fine-tuning)
        print("Starting fine-tuning...")
        solver.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-train (Fine-tune) Driver Assistance System with RT-DETR checkpoint"
    )
    
    parser.add_argument(
        '--config', '-c', type=str, required=True,
        help='Path to RT-DETR config YAML file (e.g., configs/kitti_rtdetr.yml)'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to pre-trained checkpoint (default: checkpoints/rtdetr_r101vd_6x_coco.pth)'
    )
    parser.add_argument(
        '--resume', '-r', type=str, default=None,
        help='Path to checkpoint to resume from (for continuing training, not fine-tuning)'
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

