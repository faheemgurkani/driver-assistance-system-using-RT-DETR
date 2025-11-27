"""
Training Script for Driver Assistance System using RT-DETR
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

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

from src.utils import build_dataloader, PreprocessTransforms
from src.datasets import DatasetFactory


def load_rtdetr_model(config_path: str, checkpoint_path: Optional[str] = None):
    """
    Load RT-DETR model from config.
    
    Args:
        config_path: Path to RT-DETR config YAML
        checkpoint_path: Optional path to checkpoint to resume from
    
    Returns:
        RT-DETR model, criterion, optimizer
    """
    from src.core import YAMLConfig
    
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    
    # Get model from config
    from src.core import GLOBAL_CONFIG
    model = GLOBAL_CONFIG['model']()
    
    # Get criterion
    criterion = GLOBAL_CONFIG['criterion']()
    
    # Get optimizer
    optimizer = GLOBAL_CONFIG['optimizer'](model.parameters())
    
    return model, criterion, optimizer, cfg


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        # Forward pass
        outputs = model(images, targets)
        
        # Compute loss
        loss_dict = criterion(outputs, targets)
        loss = sum(loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                  f"Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main(args):
    """Main training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Build dataloader
    print(f"Loading dataset from: {args.data_root}")
    train_loader = build_dataloader(
        dataset_name=args.dataset_name,
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    print(f"Dataset loaded: {len(train_loader.dataset)} samples")
    
    # Load RT-DETR model
    if args.rtdetr_config:
        print(f"Loading RT-DETR model from config: {args.rtdetr_config}")
        model, criterion, optimizer, cfg = load_rtdetr_model(
            args.rtdetr_config,
            args.resume
        )
    else:
        # Use default RT-DETR config
        default_config = rtdetr_path / "configs" / "rtdetr" / "rtdetr_r50vd_6x_coco.yml"
        if default_config.exists():
            print(f"Using default config: {default_config}")
            model, criterion, optimizer, cfg = load_rtdetr_model(
                str(default_config),
                args.resume
            )
        else:
            raise ValueError("No RT-DETR config provided and default not found")
    
    model = model.to(device)
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"Epoch {epoch+1}/{args.epochs} completed. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Driver Assistance System with RT-DETR")
    
    # Dataset arguments
    parser.add_argument("--data-root", type=str, required=True,
                       help="Root directory of the dataset")
    parser.add_argument("--dataset-name", type=str, default=None,
                       help="Dataset name (kitti, d2_city, etc.). Auto-detected if not provided")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loading workers")
    
    # Model arguments
    parser.add_argument("--rtdetr-config", type=str, default=None,
                       help="Path to RT-DETR config YAML file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--save-interval", type=int, default=10,
                       help="Save checkpoint every N epochs")
    
    # Other arguments
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU usage")
    
    args = parser.parse_args()
    main(args)

