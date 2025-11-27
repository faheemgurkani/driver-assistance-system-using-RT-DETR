"""
Training Script using RT-DETR's Solver Pattern
Properly integrates with RT-DETR's training infrastructure
"""

import os
import sys
import argparse
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Add project root to path (for RT-DETR)
project_root = backend_path.parent
sys.path.insert(0, str(project_root))

# Add RT-DETR to path
rtdetr_path = project_root.parent / "rtdetr_pytorch"
if rtdetr_path.exists():
    sys.path.insert(0, str(rtdetr_path))

# Import RT-DETR modules
import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS


def main(args):
    """Main training function using RT-DETR's solver."""
    # Initialize distributed training if needed
    dist.init_distributed()
    
    if args.seed is not None:
        dist.set_seed(args.seed)
    
    assert not all([args.tuning, args.resume]), \
        'Only support from_scratch or resume or tuning at one time'
    
    # Load RT-DETR config
    cfg = YAMLConfig(
        args.config,
        resume=args.resume,
        use_amp=args.amp,
        tuning=args.tuning
    )
    
    # Create solver (DetSolver for detection tasks)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        # Validation only
        solver.val()
    else:
        # Training
        solver.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Driver Assistance System with RT-DETR (using RT-DETR solver)"
    )
    
    parser.add_argument(
        '--config', '-c', type=str, required=True,
        help='Path to RT-DETR config YAML file'
    )
    parser.add_argument(
        '--resume', '-r', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--tuning', '-t', type=str, default=None,
        help='Path to checkpoint for fine-tuning'
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

