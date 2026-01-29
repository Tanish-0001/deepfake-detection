"""
DataLoader utilities for creating training and evaluation dataloaders.

Supports:
- Single dataset (FF++ or Celeb-DF)
- Combined/multi-dataset training
- Weighted sampling for imbalanced data
"""

from typing import Optional, Tuple, Union, List, Dict, Any
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .ff_dataset import FFDataset, FFVideoDataset, get_ff_dataset
from .celeb_df_dataset import CelebDFDataset, CelebDFVideoDataset, get_celeb_df_dataset
from .combined_dataset import (
    CombinedDeepfakeDataset, 
    DatasetConfig, 
    DatasetRegistry,
    create_combined_dataset
)
from preprocessing.transforms import get_train_transforms, get_val_transforms, TransformConfig


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset=None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampler: bool = True,
    persistent_workers: bool = False,
    collate_fn=None
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
        collate_fn: Custom collate function (use video_collate_fn for video-level datasets)
        
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
        persistent_workers=use_persistent,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        multiprocessing_context=mp_context,
        persistent_workers=use_persistent,
        collate_fn=collate_fn
    )
    
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=mp_context,
            persistent_workers=use_persistent,
            collate_fn=collate_fn
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
    preload_cache: bool = True,
    collate_fn=None
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
    
    # Auto-use video_collate_fn for video-level datasets if not specified
    if video_level and collate_fn is None:
        collate_fn = video_collate_fn
    
    # Create dataloaders
    return create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn
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


def create_celeb_df_dataloaders(
    root_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 0,
    frames_per_video: int = 10,
    video_level: bool = False,
    config=None,
    use_cache: bool = True,
    require_cache: bool = True,
    preload_cache: bool = True,
    collate_fn=None
) -> Tuple:
    """
    Create DataLoaders for Celeb-DF-v2 dataset.
    
    Args:
        root_dir: Root directory of Celeb-DF-v2 dataset
        batch_size: Batch size
        num_workers: Number of worker processes (0 recommended when preload_cache=True)
        frames_per_video: Number of frames per video
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
    
    # Override with config if provided
    if config is not None:
        batch_size = config.data.batch_size
        num_workers = config.data.num_workers
        frames_per_video = config.data.frames_per_video
        use_cache = config.data.use_cache
        if hasattr(config.data, 'preload_cache'):
            preload_cache = config.data.preload_cache
    
    # When preloading cache, num_workers=0 is optimal
    if preload_cache and num_workers > 0:
        print(f"Note: Using num_workers=0 since preload_cache=True (data loaded into RAM)")
        num_workers = 0
    
    # Create transforms
    transform_config = TransformConfig()
    train_transform = get_train_transforms(transform_config)
    val_transform = get_val_transforms(transform_config)
    
    # Create preprocessing pipelines
    train_pipeline = PreprocessingPipeline(
        num_frames=frames_per_video,
        sampling_strategy="random"
    )
    
    val_pipeline = PreprocessingPipeline(
        num_frames=frames_per_video,
        sampling_strategy="uniform"
    )
    
    # Select dataset class
    DatasetClass = CelebDFVideoDataset if video_level else CelebDFDataset
    
    # Create datasets
    train_dataset = DatasetClass(
        root_dir=root_dir,
        split="train",
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
        frames_per_video=frames_per_video,
        transform=val_transform,
        preprocessing_pipeline=val_pipeline,
        use_cache=use_cache,
        require_cache=require_cache,
        preload_cache=preload_cache
    )
    
    # Auto-use video_collate_fn for video-level datasets if not specified
    if video_level and collate_fn is None:
        collate_fn = video_collate_fn
    
    return create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


def create_combined_dataloaders(
    dataset_configs: List[Union[DatasetConfig, Dict[str, Any]]],
    batch_size: int = 32,
    num_workers: int = 0,
    frames_per_video: int = 10,
    video_level: bool = False,
    use_cache: bool = True,
    require_cache: bool = True,
    preload_cache: bool = True,
    use_dataset_weights: bool = False,
    pin_memory: bool = True,
    config=None,
    test_only: bool = False,
    collate_fn=None
) -> Tuple:
    """
    Create DataLoaders for combined multi-dataset training.
    
    This function creates dataloaders that combine multiple datasets (e.g., FF++ and Celeb-DF)
    for training a more robust deepfake detector.
    
    Args:
        dataset_configs: List of dataset configurations. Each can be:
            - DatasetConfig object
            - Dict with keys: 'name', 'root_dir', optionally 'weight' and other kwargs
        batch_size: Batch size
        num_workers: Number of worker processes (0 recommended when preload_cache=True)
        frames_per_video: Number of frames per video
        video_level: Whether to use video-level datasets
        use_cache: Whether to use cached preprocessed data
        require_cache: If True, only use videos with existing cache
        preload_cache: If True, load all cache into RAM at startup
        use_dataset_weights: If True, use dataset weights for sampling
        pin_memory: Whether to pin memory for GPU
        config: Optional configuration object
        test_only: Whether to create only test dataloader (skip train and val)
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        
    Example:
        configs = [
            {"name": "ff", "root_dir": "Datasets/FF", "weight": 1.0},
            {"name": "celeb_df", "root_dir": "Datasets/Celeb-DF-v2", "weight": 1.5},
        ]
        train_loader, val_loader, test_loader = create_combined_dataloaders(configs)
    """
    # Override with config if provided
    if config is not None:
        batch_size = getattr(config.data, 'batch_size', batch_size)
        num_workers = getattr(config.data, 'num_workers', num_workers)
        frames_per_video = getattr(config.data, 'frames_per_video', frames_per_video)
        use_cache = getattr(config.data, 'use_cache', use_cache)
        preload_cache = getattr(config.data, 'preload_cache', preload_cache)
    
    # When preloading cache, num_workers=0 is optimal
    if preload_cache and num_workers > 0:
        print(f"Note: Using num_workers=0 since preload_cache=True (data loaded into RAM)")
        num_workers = 0
    
    # Create transforms
    transform_config = TransformConfig()
    train_transform = get_train_transforms(transform_config)
    val_transform = get_val_transforms(transform_config)
    
    print(f"\nCreating combined datasets from {len(dataset_configs)} sources...")
    
    if not test_only:
        # Create combined datasets for each split
        print("\n--- Training Set ---")
        train_dataset = create_combined_dataset(
            dataset_configs=dataset_configs,
            split="train",
            frames_per_video=frames_per_video,
            transform=train_transform,
            video_level=video_level,
            use_cache=use_cache,
            require_cache=require_cache,
            preload_cache=preload_cache
        )
        
        print("\n--- Validation Set ---")
        val_dataset = create_combined_dataset(
            dataset_configs=dataset_configs,
            split="val",
            frames_per_video=frames_per_video,
            transform=val_transform,
            video_level=video_level,
            use_cache=use_cache,
            require_cache=require_cache,
            preload_cache=preload_cache
        )
    
    print("\n--- Test Set ---")
    test_dataset = create_combined_dataset(
        dataset_configs=dataset_configs,
        split="test",
        frames_per_video=frames_per_video,
        transform=val_transform,
        video_level=video_level,
        use_cache=use_cache,
        require_cache=require_cache,
        preload_cache=preload_cache
    )
    
    if not test_only:
        # Print combined stats
        print("\n--- Combined Dataset Statistics ---")
        for split_name, dataset in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
            stats = dataset.get_dataset_stats()
            print(f"\n{split_name}:")
            for name, s in stats.items():
                print(f"  {name}: {s['total_samples']} samples, {s['num_videos']} videos "
                  f"(Real: {s['real_count']}, Fake: {s['fake_count']}, weight: {s['weight']})")
    
    # Auto-use video_collate_fn for video-level datasets if not specified
    if video_level and collate_fn is None:
        collate_fn = video_collate_fn
    
    # Use spawn multiprocessing context
    import multiprocessing
    mp_context = 'spawn' if num_workers > 0 else None
    
    if not test_only:
        # Create weighted sampler for training
        train_sampler = None
        shuffle_train = True
        
        # Get weights for sampling
        if use_dataset_weights:
            # Use both class balance and dataset weights
            class_weights = train_dataset.get_class_balanced_weights()
            dataset_weights = train_dataset.get_dataset_weights()
            weights = [c * d for c, d in zip(class_weights, dataset_weights)]
        else:
            # Only use class balance weights
            weights = train_dataset.get_class_balanced_weights()
        
        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        shuffle_train = False
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            multiprocessing_context=mp_context,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=mp_context,
            collate_fn=collate_fn
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        multiprocessing_context=mp_context,
        collate_fn=collate_fn
    )
    
    if not test_only:
        return train_loader, val_loader, test_loader
    
    return (test_loader,)


def get_dataloaders(
    dataset_type: str = "ff",
    root_dir: Optional[Union[str, Path]] = None,
    dataset_configs: Optional[List[Union[DatasetConfig, Dict[str, Any]]]] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    frames_per_video: int = 10,
    video_level: bool = False,
    use_cache: bool = True,
    require_cache: bool = True,
    preload_cache: bool = True,
    config=None,
    collate_fn=None,
    **kwargs
) -> Tuple:
    """
    Universal function to get dataloaders for any supported dataset or combination.
    
    This is the recommended entry point for creating dataloaders. It automatically
    selects the appropriate function based on the dataset_type.
    
    Args:
        dataset_type: Type of dataset(s) to use:
            - 'ff' or 'faceforensics': FaceForensics++ dataset
            - 'celeb_df' or 'celebdf': Celeb-DF-v2 dataset  
            - 'combined' or 'multi': Combined multi-dataset (requires dataset_configs)
        root_dir: Root directory of the dataset (for single dataset mode)
        dataset_configs: List of dataset configs (required for combined mode)
        batch_size: Batch size
        num_workers: Number of worker processes
        frames_per_video: Number of frames per video
        video_level: Whether to use video-level datasets
        use_cache: Whether to use cached preprocessed data
        require_cache: If True, only use videos with existing cache
        preload_cache: If True, load all cache into RAM
        config: Optional configuration object
        **kwargs: Additional arguments passed to the specific dataloader function
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        
    Examples:
        # Single FF++ dataset
        loaders = get_dataloaders("ff", root_dir="Datasets/FF")
        
        # Single Celeb-DF dataset
        loaders = get_dataloaders("celeb_df", root_dir="Datasets/Celeb-DF-v2")
        
        # Combined datasets
        configs = [
            {"name": "ff", "root_dir": "Datasets/FF"},
            {"name": "celeb_df", "root_dir": "Datasets/Celeb-DF-v2"},
        ]
        loaders = get_dataloaders("combined", dataset_configs=configs)
    """
    dataset_type = dataset_type.lower()
    
    if dataset_type in ['combined', 'multi', 'multiple']:
        if dataset_configs is None:
            raise ValueError("dataset_configs is required for combined dataset mode")
        return create_combined_dataloaders(
            dataset_configs=dataset_configs,
            batch_size=batch_size,
            num_workers=num_workers,
            frames_per_video=frames_per_video,
            video_level=video_level,
            use_cache=use_cache,
            require_cache=require_cache,
            preload_cache=preload_cache,
            config=config,
            collate_fn=collate_fn,
            **kwargs
        )
    
    elif dataset_type in ['ff', 'ff++', 'faceforensics', 'faceforensics++']:
        if root_dir is None:
            root_dir = Path("Datasets/FF")
        return create_ff_dataloaders(
            root_dir=root_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            frames_per_video=frames_per_video,
            video_level=video_level,
            use_cache=use_cache,
            require_cache=require_cache,
            preload_cache=preload_cache,
            config=config,
            collate_fn=collate_fn,
            **kwargs
        )
    
    elif dataset_type in ['celeb_df', 'celebdf', 'celeb-df', 'celeb_df_v2']:
        if root_dir is None:
            root_dir = Path("Datasets/Celeb-DF-v2")
        return create_celeb_df_dataloaders(
            root_dir=root_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            frames_per_video=frames_per_video,
            video_level=video_level,
            use_cache=use_cache,
            require_cache=require_cache,
            preload_cache=preload_cache,
            config=config,
            collate_fn=collate_fn,
            **kwargs
        )
    
    else:
        available = ['ff', 'celeb_df', 'combined']
        raise ValueError(f"Unknown dataset type: '{dataset_type}'. Available: {available}")
