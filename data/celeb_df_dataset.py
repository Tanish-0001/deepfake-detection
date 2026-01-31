"""
Celeb-DF-v2 Dataset implementation.

Loads data from the Celeb-DF-v2 dataset structure.
- Celeb-real: Real celebrity videos
- Celeb-synthesis: Fake deepfake videos
- YouTube-real: Real YouTube videos

The dataset uses List_of_testing_videos.txt for official test split.
Remaining videos are used for training and validation.
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
import cv2


def _deterministic_hash(s: str) -> int:
    """Compute a deterministic hash for a string (consistent across Python sessions)."""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object

from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.transforms import get_train_transforms, get_val_transforms, TransformConfig


class CelebDFDataset(Dataset):
    """
    Celeb-DF-v2 Dataset for deepfake detection.
    
    This dataset loads face crops from videos in the Celeb-DF-v2 dataset.
    Real videos are from Celeb-real and YouTube-real, fake videos from Celeb-synthesis.
    
    Labels:
        0 = Real (Celeb-real, YouTube-real)
        1 = Fake (Celeb-synthesis)
        
    For faster training, preprocess the dataset first:
        python -m preprocessing.preprocess_celeb_df --all
    
    Then set use_cache=True and require_cache=True for fastest loading.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        frames_per_video: int = 10,
        transform=None,
        preprocessing_pipeline: Optional[PreprocessingPipeline] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        use_cache: bool = True,
        require_cache: bool = True,  # If True, only use cached videos (no preprocessing)
        preload_cache: bool = True   # If True, preload all cache into memory
    ):
        """
        Initialize Celeb-DF-v2 dataset.
        
        Args:
            root_dir: Root directory of Celeb-DF-v2 dataset
            split: Data split ('train', 'val', 'test')
            frames_per_video: Number of frames to sample per video
            transform: Optional transform to apply to images
            preprocessing_pipeline: Pipeline for face extraction
            cache_dir: Directory for caching preprocessed data
            use_cache: Whether to use cached data if available
            require_cache: If True, skip videos without cache (faster startup, no preprocessing)
            preload_cache: If True, load all cache into RAM (uses more memory but faster)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CelebDFDataset")
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.frames_per_video = frames_per_video
        self.transform = transform
        self.use_cache = use_cache
        self.require_cache = require_cache
        self.preload_cache = preload_cache
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = self.root_dir / "cache" / split
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup preprocessing pipeline
        if preprocessing_pipeline is None:
            preprocessing_pipeline = PreprocessingPipeline(
                num_frames=frames_per_video,
                sampling_strategy="uniform"
            )
        self.pipeline = preprocessing_pipeline
        
        # Load video list from JSON
        self._load_video_list()
        
        # Build sample index (maps sample idx to video and frame)
        self._build_sample_index()
        
        # Preload cache if requested
        self._preloaded_cache = {}
        if self.preload_cache:
            self._preload_all_cache()
    
    def _load_video_list(self):
        """Load video paths from JSON split file."""
        json_path = self.root_dir / f"{self.split}_paths.json"
        
        with open(json_path, 'r') as f:
            video_entries = json.load(f)
        
        self.video_list = []
        self.labels = []
        skipped_no_cache = 0
        
        # Load videos from the paths file which contains path and label for each video
        for entry in video_entries:
            video_path = Path(entry["path"])
            label = entry["label"]
            
            # Check if cache exists when require_cache is True
            if self.require_cache:
                cache_path = self._compute_cache_path(video_path)
                if not cache_path.exists():
                    skipped_no_cache += 1
                    continue
            
            if video_path.exists() or self.require_cache:
                self.video_list.append(video_path)
                self.labels.append(label)
        
        print(f"Loaded {len(self.video_list)} videos for {self.split} split")
        print(f"  Real: {self.labels.count(0)}, Fake: {self.labels.count(1)}")
        if skipped_no_cache > 0:
            print(f"  Skipped {skipped_no_cache} videos without cache (require_cache=True)")
    
    def _build_sample_index(self):
        """Build index mapping sample idx to (video_idx, frame_idx)."""
        self.sample_index = []
        
        for video_idx in range(len(self.video_list)):
            for frame_idx in range(self.frames_per_video):
                self.sample_index.append((video_idx, frame_idx))
    
    def _compute_cache_path(self, video_path: Path) -> Path:
        """Compute cache file path for a video without storing."""
        cache_name = f"{video_path.stem}_{_deterministic_hash(str(video_path))}.npz"
        return self.cache_dir / cache_name
    
    def _get_cache_path(self, video_idx: int) -> Path:
        """Get cache file path for a video."""
        video_path = self.video_list[video_idx]
        return self._compute_cache_path(video_path)
    
    def _preload_all_cache(self):
        """Preload all cached data into memory for faster access."""
        from tqdm import tqdm
        print(f"Preloading {len(self.video_list)} cached videos into memory...")
        
        for video_idx in tqdm(range(len(self.video_list)), desc="Preloading cache"):
            cache_path = self._get_cache_path(video_idx)
            if cache_path.exists():
                try:
                    data = np.load(cache_path)
                    faces = [data[f'face_{i}'] for i in range(len(data.files))]
                    self._preloaded_cache[video_idx] = faces
                except Exception as e:
                    print(f"Warning: Could not preload {cache_path}: {e}")
        
        print(f"Preloaded {len(self._preloaded_cache)} videos into memory")
    
    def _load_or_process_video(self, video_idx: int) -> List[np.ndarray]:
        """Load video faces from cache or process."""
        # Check preloaded cache first (fastest)
        if video_idx in self._preloaded_cache:
            return self._preloaded_cache[video_idx]
        
        cache_path = self._get_cache_path(video_idx)
        
        # Try to load from cache
        if self.use_cache and cache_path.exists():
            try:
                data = np.load(cache_path)
                faces = [data[f'face_{i}'] for i in range(len(data.files))]
                return faces
            except Exception:
                pass
        
        # Skip processing if require_cache is True
        if self.require_cache:
            # Return black frames as fallback
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.frames_per_video)]
        
        # Process video (slow path - only if cache doesn't exist)
        video_path = self.video_list[video_idx]
        faces = self.pipeline.process_video(video_path, return_frames_without_faces=True)
        
        # Ensure we have exactly frames_per_video faces
        if len(faces) < self.frames_per_video:
            # Duplicate last face if not enough
            while len(faces) < self.frames_per_video:
                if faces:
                    faces.append(faces[-1].copy())
                else:
                    # Create black image if no faces at all
                    faces.append(np.zeros((224, 224, 3), dtype=np.uint8))
        elif len(faces) > self.frames_per_video:
            faces = faces[:self.frames_per_video]
        
        # Cache results
        if self.use_cache:
            try:
                cache_dict = {f'face_{i}': face for i, face in enumerate(faces)}
                np.savez_compressed(cache_path, **cache_dict)
            except Exception as e:
                print(f"Warning: Could not cache {video_path}: {e}")
        
        return faces
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.sample_index)
    
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, label)
        """
        video_idx, frame_idx = self.sample_index[idx]
        
        # Load faces for this video
        faces = self._load_or_process_video(video_idx)
        
        # Get specific frame
        face = faces[frame_idx]
        label = self.labels[video_idx]
        
        # Apply transform
        if self.transform is not None:
            face = self.transform(face)
        
        # Convert to tensor if not already
        if isinstance(face, np.ndarray):
            face = torch.from_numpy(face).float()
        
        return face, label
    
    def get_video_sample(self, video_idx: int) -> Tuple[List[Any], int]:
        """
        Get all frames for a specific video.
        
        Args:
            video_idx: Video index
            
        Returns:
            Tuple of (list of images, label)
        """
        faces = self._load_or_process_video(video_idx)
        label = self.labels[video_idx]
        
        if self.transform is not None:
            faces = [self.transform(face) for face in faces]
        
        return faces, label
    
    @property
    def num_videos(self) -> int:
        """Return number of videos in dataset."""
        return len(self.video_list)


class CelebDFVideoDataset(Dataset):
    """
    Celeb-DF-v2 Dataset that returns video-level samples.
    
    Each sample is a video represented by multiple frames.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        frames_per_video: int = 10,
        transform=None,
        preprocessing_pipeline: Optional[PreprocessingPipeline] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        use_cache: bool = True,
        require_cache: bool = False,  # If True, only use cached videos
        preload_cache: bool = False   # If True, preload all cache files into memory
    ):
        """
        Initialize Celeb-DF-v2 video-level dataset.
        
        Args:
            root_dir: Root directory of Celeb-DF-v2 dataset
            split: Data split ('train', 'val', 'test')
            frames_per_video: Number of frames to sample per video
            transform: Optional transform to apply
            preprocessing_pipeline: Pipeline for face extraction
            cache_dir: Directory for caching
            use_cache: Whether to use cache
            require_cache: If True, skip videos without cache (faster, no preprocessing)
            preload_cache: If True, preload all cache files into RAM (uses more memory but faster)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CelebDFVideoDataset")
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.frames_per_video = frames_per_video
        self.transform = transform
        self.use_cache = use_cache
        self.require_cache = require_cache
        self.preload_cache = preload_cache
        
        if cache_dir is None:
            cache_dir = self.root_dir / "cache" / split
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if preprocessing_pipeline is None:
            preprocessing_pipeline = PreprocessingPipeline(
                num_frames=frames_per_video,
                sampling_strategy="uniform"
            )
        self.pipeline = preprocessing_pipeline
        
        self._load_video_list()
        
        # Preload cache if requested (uses more RAM but much faster)
        self._preloaded_cache = {}
        if self.preload_cache:
            self._preload_all_cache()
    
    def _load_video_list(self):
        """Load video paths from JSON split file."""
        json_path = self.root_dir / f"{self.split}_paths.json"
        
        with open(json_path, 'r') as f:
            video_entries = json.load(f)
        
        self.video_list = []
        self.labels = []
        skipped_no_cache = 0
        
        # Load videos from the paths file which contains path and label for each video
        for entry in video_entries:
            video_path = Path(entry["path"])
            label = entry["label"]
            
            # Check if cache exists when require_cache is True
            if self.require_cache:
                cache_path = self._compute_cache_path(video_path)
                if not cache_path.exists():
                    skipped_no_cache += 1
                    continue
            
            if video_path.exists() or self.require_cache:
                self.video_list.append(video_path)
                self.labels.append(label)
        
        print(f"Loaded {len(self.video_list)} videos for {self.split} split")
        print(f"  Real: {self.labels.count(0)}, Fake: {self.labels.count(1)}")
        if skipped_no_cache > 0:
            print(f"  Skipped {skipped_no_cache} videos without cache (require_cache=True)")
    
    def _compute_cache_path(self, video_path: Path) -> Path:
        """Compute cache file path for a video without storing."""
        cache_name = f"{video_path.stem}_{_deterministic_hash(str(video_path))}.npz"
        return self.cache_dir / cache_name
    
    def _get_cache_path(self, video_idx: int) -> Path:
        """Get cache file path for a video."""
        video_path = self.video_list[video_idx]
        return self._compute_cache_path(video_path)
    
    def _preload_all_cache(self):
        """Preload all cached data into memory."""
        from tqdm import tqdm
        print(f"Preloading {len(self.video_list)} cached videos into memory...")
        
        for video_idx in tqdm(range(len(self.video_list)), desc="Preloading cache"):
            cache_path = self._get_cache_path(video_idx)
            if cache_path.exists():
                try:
                    data = np.load(cache_path)
                    faces = [data[f'face_{i}'] for i in range(len(data.files))]
                    self._preloaded_cache[video_idx] = faces
                except Exception as e:
                    print(f"Warning: Could not preload {cache_path}: {e}")
        
        print(f"Preloaded {len(self._preloaded_cache)} videos into memory")
    
    def _load_or_process_video(self, video_idx: int) -> List[np.ndarray]:
        """Load video faces from cache or process."""
        # Check preloaded cache first
        if video_idx in self._preloaded_cache:
            return self._preloaded_cache[video_idx]
        
        cache_path = self._get_cache_path(video_idx)
        
        if self.use_cache and cache_path.exists():
            try:
                data = np.load(cache_path)
                faces = [data[f'face_{i}'] for i in range(len(data.files))]
                return faces
            except Exception:
                pass
        
        # Skip processing if require_cache is True
        if self.require_cache:
            # Return black frames as fallback
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.frames_per_video)]
        
        video_path = self.video_list[video_idx]
        faces = self.pipeline.process_video(video_path, return_frames_without_faces=True)
        
        if len(faces) < self.frames_per_video:
            while len(faces) < self.frames_per_video:
                if faces:
                    faces.append(faces[-1].copy())
                else:
                    faces.append(np.zeros((224, 224, 3), dtype=np.uint8))
        elif len(faces) > self.frames_per_video:
            faces = faces[:self.frames_per_video]
        
        if self.use_cache:
            try:
                cache_dict = {f'face_{i}': face for i, face in enumerate(faces)}
                np.savez_compressed(cache_path, **cache_dict)
            except Exception as e:
                print(f"Warning: Could not cache {video_path}: {e}")
        
        return faces
    
    def __len__(self) -> int:
        """Return number of videos."""
        return len(self.video_list)
    
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """
        Get a video sample.
        
        Args:
            idx: Video index
            
        Returns:
            Tuple of (frames tensor [T, C, H, W], label)
        """
        faces = self._load_or_process_video(idx)
        label = self.labels[idx]
        
        # Apply transform to each frame
        if self.transform is not None:
            faces = [self.transform(face) for face in faces]
        
        # Convert to tensors
        if isinstance(faces[0], np.ndarray):
            faces = [torch.from_numpy(face).float() for face in faces]
        
        # Stack frames: [T, C, H, W]
        frames = torch.stack(faces, dim=0)
        
        return frames, label


def get_celeb_df_dataset(
    root_dir: Union[str, Path],
    split: str = "train",
    config=None,
    video_level: bool = False
) -> Union[CelebDFDataset, CelebDFVideoDataset]:
    """
    Factory function to create Celeb-DF-v2 dataset.
    
    Args:
        root_dir: Root directory of Celeb-DF-v2 dataset
        split: Data split
        config: Optional configuration object
        video_level: Whether to return video-level dataset
        
    Returns:
        CelebDFDataset or CelebDFVideoDataset
    """
    # Default parameters
    frames_per_video = 10
    
    # Override with config if provided
    if config is not None:
        frames_per_video = config.preprocessing.frames_per_video
    
    # Get transforms
    transform_config = TransformConfig()
    if split == "train":
        transform = get_train_transforms(transform_config)
    else:
        transform = get_val_transforms(transform_config)
    
    # Create preprocessing pipeline
    pipeline = PreprocessingPipeline(
        num_frames=frames_per_video,
        sampling_strategy="uniform" if split != "train" else "random"
    )
    
    DatasetClass = CelebDFVideoDataset if video_level else CelebDFDataset
    
    return DatasetClass(
        root_dir=root_dir,
        split=split,
        frames_per_video=frames_per_video,
        transform=transform,
        preprocessing_pipeline=pipeline
    )
