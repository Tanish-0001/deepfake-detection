"""
DataLoader utilities for creating training and evaluation dataloaders.
"""

from typing import Optional, Tuple, Union
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .ff_dataset import FFDataset, FFVideoDataset, get_ff_dataset
from preprocessing.transforms import get_train_transforms, get_val_transforms, TransformConfig


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset=None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampler: bool = True,
    persistent_workers: bool = False
) -> Tuple:
    """
    Create DataLoaders for training, validation, and optionally test sets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU
        use_weighted_sampler: Whether to use weighted sampling for imbalanced data
        persistent_workers: Whether to keep workers alive between epochs
        
    Returns:
        Tuple of (train_loader, val_loader) or (train_loader, val_loader, test_loader)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for DataLoaders")
    
    # Use 'spawn' multiprocessing context to avoid CUDA context issues
    # This is necessary when using CUDA with multiprocessing DataLoader workers
    import multiprocessing
    mp_context = 'spawn' if num_workers > 0 else None
    
    # Only use persistent_workers if num_workers > 0
    use_persistent = persistent_workers and num_workers > 0
    
    # Create weighted sampler for training if dataset is imbalanced
    train_sampler = None
    shuffle_train = True
    
    if use_weighted_sampler and hasattr(train_dataset, 'labels'):
        labels = train_dataset.labels
        
        # For frame-level dataset, expand labels
        if hasattr(train_dataset, 'sample_index'):
            expanded_labels = [labels[vid_idx] for vid_idx, _ in train_dataset.sample_index]
            labels = expanded_labels
        
        # Compute class weights
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1

        # Weight inversely proportional to class frequency
        weights = [1.0 / class_counts[label] for label in labels]
        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        shuffle_train = False  # Sampler handles shuffling
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        multiprocessing_context=mp_context,
        persistent_workers=use_persistent
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        multiprocessing_context=mp_context,
        persistent_workers=use_persistent
    )
    
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=mp_context,
            persistent_workers=use_persistent
        )
        return train_loader, val_loader, test_loader
    
    return train_loader, val_loader


def create_ff_dataloaders(
    root_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 0,  # Default 0 since preload_cache=True loads data into RAM
    frames_per_video: int = 10,
    manipulation_types: Optional[list] = None,
    compression: str = "c23",
    video_level: bool = False,
    config=None,
    use_cache: bool = True,
    require_cache: bool = True,
    preload_cache: bool = True
) -> Tuple:
    """
    Create DataLoaders for FaceForensics++ dataset.
    
    Args:
        root_dir: Root directory of FF++ dataset
        batch_size: Batch size
        num_workers: Number of worker processes (0 recommended when preload_cache=True)
        frames_per_video: Number of frames per video
        manipulation_types: List of manipulation types
        compression: Compression level
        video_level: Whether to use video-level dataset
        config: Optional configuration object
        use_cache: Whether to use cached preprocessed data
        require_cache: If True, only use videos with existing cache
        preload_cache: If True, load all cache into RAM at startup
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from preprocessing.pipeline import PreprocessingPipeline
    
    root_dir = Path(root_dir)
    
    # Get default config values
    if manipulation_types is None:
        manipulation_types = [
            "Deepfakes",
            "Face2Face", 
            "FaceSwap",
            "NeuralTextures",
            "FaceShifter"
        ]
    
    # Override with config if provided
    if config is not None:
        batch_size = config.data.batch_size
        num_workers = config.data.num_workers
        frames_per_video = config.data.frames_per_video
        manipulation_types = config.data.manipulation_types
        compression = config.data.compression
        use_cache = config.data.use_cache
        if hasattr(config.data, 'preload_cache'):
            preload_cache = config.data.preload_cache
    
    # When preloading cache, num_workers=0 is optimal (data already in RAM)
    if preload_cache and num_workers > 0:
        print(f"Note: Using num_workers=0 since preload_cache=True (data loaded into RAM)")
        num_workers = 0
    
    # Create transforms
    transform_config = TransformConfig()
    train_transform = get_train_transforms(transform_config)
    val_transform = get_val_transforms(transform_config)
    
    # Create preprocessing pipelines (only used if cache miss)
    train_pipeline = PreprocessingPipeline(
        num_frames=frames_per_video,
        sampling_strategy="random"  # Random sampling for training
    )
    
    val_pipeline = PreprocessingPipeline(
        num_frames=frames_per_video,
        sampling_strategy="uniform"  # Uniform sampling for validation/test
    )
    
    # Select dataset class
    DatasetClass = FFVideoDataset if video_level else FFDataset
    
    # Create datasets with caching options
    train_dataset = DatasetClass(
        root_dir=root_dir,
        split="train",
        manipulation_types=manipulation_types,
        compression=compression,
        frames_per_video=frames_per_video,
        transform=train_transform,
        preprocessing_pipeline=train_pipeline,
        use_cache=use_cache,
        require_cache=require_cache,
        preload_cache=preload_cache
    )
    
    val_dataset = DatasetClass(
        root_dir=root_dir,
        split="val",
        manipulation_types=manipulation_types,
        compression=compression,
        frames_per_video=frames_per_video,
        transform=val_transform,
        preprocessing_pipeline=val_pipeline,
        use_cache=use_cache,
        require_cache=require_cache,
        preload_cache=preload_cache
    )
    
    test_dataset = DatasetClass(
        root_dir=root_dir,
        split="test",
        manipulation_types=manipulation_types,
        compression=compression,
        frames_per_video=frames_per_video,
        transform=val_transform,
        preprocessing_pipeline=val_pipeline,
        use_cache=use_cache,
        require_cache=require_cache,
        preload_cache=preload_cache
    )
    
    # Create dataloaders
    return create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )


def video_collate_fn(batch):
    """
    Custom collate function for video-level batches.
    
    Args:
        batch: List of (frames, label) tuples
        
    Returns:
        Tuple of (batched_frames, labels)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    
    frames = torch.stack([item[0] for item in batch], dim=0)  # [B, T, C, H, W]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    
    return frames, labels
