"""
Combined Dataset implementation for training on multiple deepfake detection datasets.

This module provides functionality to combine multiple datasets (FF++, Celeb-DF, etc.)
into a single dataset for training. It uses a registry pattern for easy extensibility.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any, Callable, Type, TYPE_CHECKING

try:
    import torch
    from torch.utils.data import Dataset, ConcatDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object
    ConcatDataset = object


# =============================================================================
# Dataset Registry - For easy extensibility
# =============================================================================

class DatasetRegistry:
    """
    Registry for deepfake detection datasets.
    
    Allows registering new datasets and retrieving them by name.
    This makes it easy to add new datasets in the future.
    
    Usage:
        # Register a new dataset
        @DatasetRegistry.register("my_dataset")
        class MyDataset(Dataset):
            ...
        
        # Or register manually
        DatasetRegistry.register_dataset("my_dataset", MyDataset)
        
        # Get dataset class
        dataset_cls = DatasetRegistry.get("my_dataset")
    """
    
    _datasets: Dict[str, type] = {}
    _video_datasets: Dict[str, type] = {}
    _factory_functions: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str, video_level: bool = False):
        """Decorator to register a dataset class."""
        def decorator(dataset_cls: type):
            if video_level:
                cls._video_datasets[name.lower()] = dataset_cls
            else:
                cls._datasets[name.lower()] = dataset_cls
            return dataset_cls
        return decorator
    
    @classmethod
    def register_dataset(cls, name: str, dataset_cls: type, video_level: bool = False):
        """Manually register a dataset class."""
        if video_level:
            cls._video_datasets[name.lower()] = dataset_cls
        else:
            cls._datasets[name.lower()] = dataset_cls
    
    @classmethod
    def register_factory(cls, name: str, factory_fn: Callable):
        """Register a factory function for creating datasets."""
        cls._factory_functions[name.lower()] = factory_fn
    
    @classmethod
    def get(cls, name: str, video_level: bool = False) -> Optional[type]:
        """Get a registered dataset class by name."""
        if video_level:
            return cls._video_datasets.get(name.lower())
        return cls._datasets.get(name.lower())
    
    @classmethod
    def get_factory(cls, name: str) -> Optional[Callable]:
        """Get a registered factory function by name."""
        return cls._factory_functions.get(name.lower())
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered dataset names."""
        all_names = set(cls._datasets.keys()) | set(cls._video_datasets.keys())
        return sorted(list(all_names))
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a dataset is registered."""
        name_lower = name.lower()
        return name_lower in cls._datasets or name_lower in cls._video_datasets


# =============================================================================
# Register existing datasets
# =============================================================================

def _register_builtin_datasets():
    """Register built-in datasets (FF++, Celeb-DF)."""
    from .ff_dataset import FFDataset, FFVideoDataset, get_ff_dataset
    from .celeb_df_dataset import CelebDFDataset, CelebDFVideoDataset, get_celeb_df_dataset
    
    # Register FF++ datasets
    DatasetRegistry.register_dataset("ff", FFDataset, video_level=False)
    DatasetRegistry.register_dataset("ff++", FFDataset, video_level=False)
    DatasetRegistry.register_dataset("faceforensics", FFDataset, video_level=False)
    DatasetRegistry.register_dataset("faceforensics++", FFDataset, video_level=False)
    
    DatasetRegistry.register_dataset("ff", FFVideoDataset, video_level=True)
    DatasetRegistry.register_dataset("ff++", FFVideoDataset, video_level=True)
    DatasetRegistry.register_dataset("faceforensics", FFVideoDataset, video_level=True)
    DatasetRegistry.register_dataset("faceforensics++", FFVideoDataset, video_level=True)
    
    DatasetRegistry.register_factory("ff", get_ff_dataset)
    DatasetRegistry.register_factory("ff++", get_ff_dataset)
    DatasetRegistry.register_factory("faceforensics", get_ff_dataset)
    DatasetRegistry.register_factory("faceforensics++", get_ff_dataset)
    
    # Register Celeb-DF datasets
    DatasetRegistry.register_dataset("celeb_df", CelebDFDataset, video_level=False)
    DatasetRegistry.register_dataset("celeb-df", CelebDFDataset, video_level=False)
    DatasetRegistry.register_dataset("celebdf", CelebDFDataset, video_level=False)
    DatasetRegistry.register_dataset("celeb_df_v2", CelebDFDataset, video_level=False)
    
    DatasetRegistry.register_dataset("celeb_df", CelebDFVideoDataset, video_level=True)
    DatasetRegistry.register_dataset("celeb-df", CelebDFVideoDataset, video_level=True)
    DatasetRegistry.register_dataset("celebdf", CelebDFVideoDataset, video_level=True)
    DatasetRegistry.register_dataset("celeb_df_v2", CelebDFVideoDataset, video_level=True)
    
    DatasetRegistry.register_factory("celeb_df", get_celeb_df_dataset)
    DatasetRegistry.register_factory("celeb-df", get_celeb_df_dataset)
    DatasetRegistry.register_factory("celebdf", get_celeb_df_dataset)
    DatasetRegistry.register_factory("celeb_df_v2", get_celeb_df_dataset)


# =============================================================================
# Dataset Configuration
# =============================================================================

class DatasetConfig:
    """
    Configuration for a single dataset in a combined dataset.
    
    Attributes:
        name: Dataset name (e.g., 'ff', 'celeb_df')
        root_dir: Root directory of the dataset
        weight: Sampling weight (higher = more samples from this dataset)
        extra_kwargs: Additional keyword arguments for the dataset constructor
    """
    
    def __init__(
        self,
        name: str,
        root_dir: Union[str, Path],
        weight: float = 1.0,
        **extra_kwargs
    ):
        self.name = name.lower()
        self.root_dir = Path(root_dir)
        self.weight = weight
        self.extra_kwargs = extra_kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "root_dir": str(self.root_dir),
            "weight": self.weight,
            **self.extra_kwargs
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetConfig":
        """Create from dictionary."""
        name = d.pop("name")
        root_dir = d.pop("root_dir")
        weight = d.pop("weight", 1.0)
        return cls(name=name, root_dir=root_dir, weight=weight, **d)


# =============================================================================
# Combined Dataset
# =============================================================================

class CombinedDeepfakeDataset(Dataset):
    """
    Combined dataset that merges multiple deepfake detection datasets.
    
    This dataset wraps multiple individual datasets (FF++, Celeb-DF, etc.)
    and presents them as a single unified dataset for training.
    
    Features:
    - Supports any number of datasets via the registry
    - Optional weighted sampling across datasets
    - Tracks which dataset each sample came from (for analysis)
    - Compatible with standard PyTorch DataLoader
    
    Example:
        configs = [
            DatasetConfig("ff", "Datasets/FF"),
            DatasetConfig("celeb_df", "Datasets/Celeb-DF-v2"),
        ]
        dataset = CombinedDeepfakeDataset(configs, split="train")
    """
    
    def __init__(
        self,
        dataset_configs: List[DatasetConfig],
        split: str = "train",
        frames_per_video: int = 10,
        transform=None,
        video_level: bool = False,
        use_cache: bool = True,
        require_cache: bool = True,
        preload_cache: bool = True
    ):
        """
        Initialize combined dataset.
        
        Args:
            dataset_configs: List of DatasetConfig objects specifying datasets to combine
            split: Data split ('train', 'val', 'test')
            frames_per_video: Number of frames per video
            transform: Transform to apply to images
            video_level: Whether to use video-level datasets
            use_cache: Whether to use cached preprocessed data
            require_cache: If True, only use videos with existing cache
            preload_cache: If True, load all cache into RAM
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CombinedDeepfakeDataset")
        
        # Ensure registry is initialized
        _register_builtin_datasets()
        
        self.dataset_configs = dataset_configs
        self.split = split
        self.frames_per_video = frames_per_video
        self.transform = transform
        self.video_level = video_level
        self.use_cache = use_cache
        self.require_cache = require_cache
        self.preload_cache = preload_cache
        
        # Create individual datasets
        self.datasets: List[Any] = []
        self.dataset_names: List[str] = []
        self.dataset_offsets: List[int] = []  # Starting index for each dataset
        
        self._create_datasets()
        
        # Create combined dataset using ConcatDataset
        self._concat_dataset = ConcatDataset(self.datasets)
        
        # Build sample-to-dataset mapping for tracking
        self._build_dataset_mapping()
        
        # Compute combined labels for weighted sampling
        self._compute_combined_labels()
    
    def _create_datasets(self):
        """Create individual datasets based on configs."""
        from preprocessing.pipeline import PreprocessingPipeline
        from preprocessing.transforms import get_train_transforms, get_val_transforms, TransformConfig
        
        # Get transforms if not provided
        if self.transform is None:
            transform_config = TransformConfig()
            if self.split == "train":
                self.transform = get_train_transforms(transform_config)
            else:
                self.transform = get_val_transforms(transform_config)
        
        current_offset = 0
        
        for config in self.dataset_configs:
            dataset_cls = DatasetRegistry.get(config.name, video_level=self.video_level)
            
            if dataset_cls is None:
                raise ValueError(
                    f"Unknown dataset: '{config.name}'. "
                    f"Available datasets: {DatasetRegistry.list_datasets()}"
                )
            
            # Build kwargs for dataset constructor
            kwargs = {
                "root_dir": config.root_dir,
                "split": self.split,
                "frames_per_video": self.frames_per_video,
                "transform": self.transform,
                "use_cache": self.use_cache,
                "require_cache": self.require_cache,
                "preload_cache": self.preload_cache,
            }
            
            # Add extra kwargs from config (e.g., manipulation_types for FF++)
            kwargs.update(config.extra_kwargs)
            
            # Create dataset
            try:
                dataset = dataset_cls(**kwargs)
                self.datasets.append(dataset)
                self.dataset_names.append(config.name)
                self.dataset_offsets.append(current_offset)
                current_offset += len(dataset)
                
                print(f"  Added {config.name}: {len(dataset)} samples")
            except Exception as e:
                print(f"Warning: Failed to load dataset '{config.name}': {e}")
                continue
        
        if not self.datasets:
            raise ValueError("No datasets were loaded successfully!")
        
        print(f"Combined dataset: {current_offset} total samples from {len(self.datasets)} datasets")
    
    def _build_dataset_mapping(self):
        """Build mapping from global index to (dataset_idx, local_idx)."""
        self.sample_to_dataset: List[Tuple[int, int]] = []
        
        for dataset_idx, dataset in enumerate(self.datasets):
            for local_idx in range(len(dataset)):
                self.sample_to_dataset.append((dataset_idx, local_idx))
    
    def _compute_combined_labels(self):
        """Compute combined labels list for weighted sampling."""
        self.labels = []
        
        for dataset in self.datasets:
            if hasattr(dataset, 'labels'):
                # Frame-level dataset - expand labels
                if hasattr(dataset, 'sample_index'):
                    for vid_idx, _ in dataset.sample_index:
                        self.labels.append(dataset.labels[vid_idx])
                else:
                    # Video-level dataset
                    self.labels.extend(dataset.labels)
            else:
                # Fallback: iterate through dataset
                for i in range(len(dataset)):
                    _, label = dataset[i]
                    self.labels.append(label if isinstance(label, int) else label.item())
    
    def get_dataset_for_sample(self, idx: int) -> Tuple[str, int]:
        """
        Get the source dataset name and local index for a global sample index.
        
        Args:
            idx: Global sample index
            
        Returns:
            Tuple of (dataset_name, local_index)
        """
        dataset_idx, local_idx = self.sample_to_dataset[idx]
        return self.dataset_names[dataset_idx], local_idx
    
    def get_dataset_weights(self) -> List[float]:
        """
        Get weights for each sample based on dataset config weights.
        
        Returns:
            List of weights, one per sample
        """
        weights = []
        for dataset_idx, dataset in enumerate(self.datasets):
            config = self.dataset_configs[dataset_idx]
            weights.extend([config.weight] * len(dataset))
        return weights
    
    def get_class_balanced_weights(self) -> List[float]:
        """
        Get weights for class-balanced sampling across all datasets.
        
        Returns:
            List of weights, one per sample
        """
        # Count classes
        class_counts = {}
        for label in self.labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Compute weights
        weights = [1.0 / class_counts[label] for label in self.labels]
        return weights
    
    def __len__(self) -> int:
        """Return total number of samples across all datasets."""
        return len(self._concat_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image/frames, label)
        """
        return self._concat_dataset[idx]
    
    @property
    def num_datasets(self) -> int:
        """Return number of combined datasets."""
        return len(self.datasets)
    
    @property
    def num_videos(self) -> int:
        """Return total number of videos across all datasets."""
        total = 0
        for dataset in self.datasets:
            if hasattr(dataset, 'num_videos'):
                total += dataset.num_videos
            elif hasattr(dataset, 'video_list'):
                total += len(dataset.video_list)
        return total
    
    def get_dataset_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics for each dataset.
        
        Returns:
            Dictionary with stats per dataset
        """
        stats = {}
        for idx, (name, dataset) in enumerate(zip(self.dataset_names, self.datasets)):
            labels = dataset.labels if hasattr(dataset, 'labels') else []
            stats[name] = {
                "total_samples": len(dataset),
                "num_videos": dataset.num_videos if hasattr(dataset, 'num_videos') else len(dataset),
                "real_count": labels.count(0) if labels else 0,
                "fake_count": labels.count(1) if labels else 0,
                "weight": self.dataset_configs[idx].weight
            }
        return stats


# =============================================================================
# Factory function
# =============================================================================

def create_combined_dataset(
    dataset_configs: List[Union[DatasetConfig, Dict[str, Any]]],
    split: str = "train",
    frames_per_video: int = 10,
    transform=None,
    video_level: bool = False,
    use_cache: bool = True,
    require_cache: bool = True,
    preload_cache: bool = True
) -> CombinedDeepfakeDataset:
    """
    Factory function to create a combined dataset.
    
    Args:
        dataset_configs: List of DatasetConfig objects or dicts with config info
        split: Data split ('train', 'val', 'test')
        frames_per_video: Number of frames per video
        transform: Transform to apply
        video_level: Whether to use video-level datasets
        use_cache: Whether to use cache
        require_cache: If True, only use cached videos
        preload_cache: If True, preload cache into RAM
        
    Returns:
        CombinedDeepfakeDataset instance
    """
    # Convert dicts to DatasetConfig objects if needed
    configs = []
    for cfg in dataset_configs:
        if isinstance(cfg, dict):
            configs.append(DatasetConfig.from_dict(cfg.copy()))
        else:
            configs.append(cfg)
    
    return CombinedDeepfakeDataset(
        dataset_configs=configs,
        split=split,
        frames_per_video=frames_per_video,
        transform=transform,
        video_level=video_level,
        use_cache=use_cache,
        require_cache=require_cache,
        preload_cache=preload_cache
    )


# Initialize registry when module is imported
try:
    _register_builtin_datasets()
except ImportError:
    # Datasets not yet available, will be registered on first use
    pass
