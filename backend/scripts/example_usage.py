"""
Example Usage Script
Demonstrates how to use the universal dataset loaders and preprocessing
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Add project root to path (for RT-DETR)
project_root = backend_path.parent
sys.path.insert(0, str(project_root))

from src.datasets import DatasetFactory, KITTIDataset, D2CityDataset
from src.utils import PreprocessTransforms, build_dataloader


def example_dataset_factory():
    """Example: Using DatasetFactory for auto-detection."""
    print("=" * 60)
    print("Example 1: Dataset Factory Auto-Detection")
    print("=" * 60)
    
    # Auto-detect dataset type
    root_dir = "datasets/kitti"  # Change to your dataset path
    
    try:
        dataset_class = DatasetFactory.create(root_dir=root_dir)
        print(f"Detected dataset type: {dataset_class.__name__}")
        
        # Create dataset instance
        transforms = PreprocessTransforms(target_size=(640, 640))
        dataset = dataset_class(root_dir=root_dir, transforms=transforms)
        print(f"Dataset size: {len(dataset)} samples")
        
        # Get a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Image shape: {sample['image'].shape if hasattr(sample['image'], 'shape') else type(sample['image'])}")
            print(f"Number of annotations: {len(sample.get('annotations', []))}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your dataset is in the correct format.")


def example_explicit_dataset():
    """Example: Using explicit dataset class."""
    print("\n" + "=" * 60)
    print("Example 2: Explicit Dataset Class")
    print("=" * 60)
    
    root_dir = "datasets/kitti"  # Change to your dataset path
    
    try:
        transforms = PreprocessTransforms(target_size=(640, 640))
        dataset = KITTIDataset(root_dir=root_dir, transforms=transforms)
        
        print(f"Dataset: {dataset.__class__.__name__}")
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample image_id: {sample.get('img_id')}")
            print(f"Original size: {sample.get('original_size')}")
    except Exception as e:
        print(f"Error: {e}")


def example_dataloader():
    """Example: Building a DataLoader for training."""
    print("\n" + "=" * 60)
    print("Example 3: Building DataLoader for RT-DETR")
    print("=" * 60)
    
    root_dir = "datasets/kitti"  # Change to your dataset path
    
    try:
        dataloader = build_dataloader(
            dataset_name="kitti",
            root_dir=root_dir,
            batch_size=4,
            num_workers=2,
            shuffle=True
        )
        
        print(f"DataLoader created successfully")
        print(f"Dataset size: {len(dataloader.dataset)}")
        print(f"Number of batches: {len(dataloader)}")
        
        # Get a batch
        for batch_idx, (images, targets) in enumerate(dataloader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Number of targets: {len(targets)}")
            if targets:
                print(f"  First target keys: {targets[0].keys()}")
                if 'boxes' in targets[0]:
                    print(f"  First target boxes shape: {targets[0]['boxes'].shape}")
            break  # Just show first batch
    except Exception as e:
        print(f"Error: {e}")


def example_preprocessing():
    """Example: Using preprocessing transforms."""
    print("\n" + "=" * 60)
    print("Example 4: Preprocessing Pipeline")
    print("=" * 60)
    
    import cv2
    import numpy as np
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create sample
    sample = {
        "image": dummy_image,
        "annotations": [
            {"bbox": [100, 100, 200, 200], "category_id": 0},
            {"bbox": [300, 300, 400, 400], "category_id": 1}
        ],
        "original_size": (480, 640),
        "img_id": 0
    }
    
    # Apply preprocessing
    preprocess = PreprocessTransforms(target_size=(640, 640))
    processed = preprocess(sample)
    
    print(f"Original image shape: {sample['image'].shape}")
    print(f"Processed image shape: {processed['image'].shape}")
    print(f"Processed image type: {type(processed['image'])}")
    print(f"Original annotations: {len(sample['annotations'])}")
    print(f"Processed annotations: {len(processed['annotations'])}")
    if processed['annotations']:
        print(f"First processed bbox: {processed['annotations'][0]['bbox']}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Driver Assistance System - Example Usage")
    print("=" * 60)
    
    # Run examples
    example_preprocessing()
    example_dataset_factory()
    example_explicit_dataset()
    example_dataloader()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nNote: Make sure to update dataset paths in the examples above.")

