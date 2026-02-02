"""
Real-Only Dataset Wrapper.

Filters existing datasets to return only real (label=0) samples for training
the autoencoder in the AutoencoderDetector model.
"""

from typing import List, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object


class RealOnlyDataset(Dataset):
    """
    Wrapper dataset that filters for real samples only (label=0).
    
    This is used to train the autoencoder on real images only,
    so it learns the manifold of real image embeddings.
    """
    
    def __init__(self, base_dataset, real_label: int = 0):
        """
        Initialize real-only dataset wrapper.
        
        Args:
            base_dataset: The underlying dataset (FFDataset, CelebDFDataset, etc.)
            real_label: Label value for real samples (default 0)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for RealOnlyDataset")
        
        self.base_dataset = base_dataset
        self.real_label = real_label
        
        # Build index of real samples
        self._build_real_index()
    
    def _build_real_index(self):
        """Build index mapping to only real samples."""
        self.real_indices = []
        
        # Check if base dataset has labels attribute
        if hasattr(self.base_dataset, 'labels'):
            labels = self.base_dataset.labels
            
            # For frame-level datasets with sample_index
            if hasattr(self.base_dataset, 'sample_index'):
                for idx, (video_idx, frame_idx) in enumerate(self.base_dataset.sample_index):
                    if labels[video_idx] == self.real_label:
                        self.real_indices.append(idx)
            else:
                # For video-level datasets
                for idx, label in enumerate(labels):
                    if label == self.real_label:
                        self.real_indices.append(idx)
        else:
            # Fallback: iterate through dataset (slower)
            print("Warning: Dataset doesn't have labels attribute, iterating to find real samples...")
            for idx in range(len(self.base_dataset)):
                _, label = self.base_dataset[idx]
                if label == self.real_label:
                    self.real_indices.append(idx)
        
        print(f"RealOnlyDataset: Found {len(self.real_indices)} real samples out of {len(self.base_dataset)} total")
    
    def __len__(self) -> int:
        """Return number of real samples."""
        return len(self.real_indices)
    
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """
        Get a real sample.
        
        Args:
            idx: Index into real samples
            
        Returns:
            Tuple of (image, label) where label is always real_label
        """
        real_idx = self.real_indices[idx]
        return self.base_dataset[real_idx]
    
    @property
    def labels(self) -> List[int]:
        """Return labels for all samples (all real)."""
        return [self.real_label] * len(self.real_indices)


class RealOnlyVideoDataset(Dataset):
    """
    Wrapper for video-level datasets that filters for real videos only.
    """
    
    def __init__(self, base_dataset, real_label: int = 0):
        """
        Initialize real-only video dataset wrapper.
        
        Args:
            base_dataset: The underlying video dataset
            real_label: Label value for real samples (default 0)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for RealOnlyVideoDataset")
        
        self.base_dataset = base_dataset
        self.real_label = real_label
        
        # Build index of real videos
        self._build_real_index()
    
    def _build_real_index(self):
        """Build index mapping to only real videos."""
        self.real_indices = []
        
        if hasattr(self.base_dataset, 'labels'):
            for idx, label in enumerate(self.base_dataset.labels):
                if label == self.real_label:
                    self.real_indices.append(idx)
        else:
            # Fallback
            for idx in range(len(self.base_dataset)):
                _, label = self.base_dataset[idx]
                if label == self.real_label:
                    self.real_indices.append(idx)
        
        print(f"RealOnlyVideoDataset: Found {len(self.real_indices)} real videos out of {len(self.base_dataset)} total")
    
    def __len__(self) -> int:
        """Return number of real videos."""
        return len(self.real_indices)
    
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """
        Get a real video sample.
        
        Args:
            idx: Index into real videos
            
        Returns:
            Tuple of (frames, label) where label is always real_label
        """
        real_idx = self.real_indices[idx]
        return self.base_dataset[real_idx]
    
    @property
    def labels(self) -> List[int]:
        """Return labels for all videos (all real)."""
        return [self.real_label] * len(self.real_indices)


def create_real_only_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders with only real samples for autoencoder training.
    
    Args:
        train_dataset: Training dataset (will be wrapped to filter real only)
        val_dataset: Validation dataset (will be wrapped to filter real only)
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU
        
    Returns:
        Tuple of (train_loader, val_loader) with only real samples
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for DataLoaders")
    
    # Wrap datasets to filter real samples only
    train_real = RealOnlyDataset(train_dataset)
    val_real = RealOnlyDataset(val_dataset)
    
    # Use 'spawn' multiprocessing context
    import multiprocessing
    mp_context = 'spawn' if num_workers > 0 else None
    
    train_loader = DataLoader(
        train_real,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        multiprocessing_context=mp_context
    )
    
    val_loader = DataLoader(
        val_real,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        multiprocessing_context=mp_context
    )
    
    return train_loader, val_loader
