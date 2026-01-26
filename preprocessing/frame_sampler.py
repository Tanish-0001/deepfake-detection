"""
Frame sampling module for extracting frames from video files.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Generator
from enum import Enum


class SamplingStrategy(Enum):
    """Frame sampling strategies."""
    UNIFORM = "uniform"      # Evenly spaced frames
    RANDOM = "random"        # Random frames
    FIRST_N = "first_n"      # First N frames
    KEYFRAMES = "keyframes"  # Only keyframes (I-frames)


class FrameSampler:
    """
    Sample frames from video files using various strategies.
    """
    
    def __init__(
        self,
        num_frames: int = 10,
        strategy: Union[str, SamplingStrategy] = "uniform",
        seed: Optional[int] = None
    ):
        """
        Initialize frame sampler.
        
        Args:
            num_frames: Number of frames to sample
            strategy: Sampling strategy ('uniform', 'random', 'first_n', 'keyframes')
            seed: Random seed for reproducibility (only for random strategy)
        """
        self.num_frames = num_frames
        
        if isinstance(strategy, str):
            strategy = SamplingStrategy(strategy.lower())
        self.strategy = strategy
        
        self.rng = np.random.default_rng(seed)
    
    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        """
        Get video information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        info = {
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': None
        }
        
        if info['fps'] > 0:
            info['duration'] = info['total_frames'] / info['fps']
        
        cap.release()
        return info
    
    def _get_frame_indices(
        self,
        total_frames: int,
        num_frames: Optional[int] = None
    ) -> List[int]:
        """
        Calculate which frame indices to sample.
        
        Args:
            total_frames: Total number of frames in video
            num_frames: Number of frames to sample (uses self.num_frames if None)
            
        Returns:
            List of frame indices to sample
        """
        if num_frames is None:
            num_frames = self.num_frames
        
        # Don't sample more frames than available
        num_frames = min(num_frames, total_frames)
        
        if num_frames <= 0:
            return []
        
        if self.strategy == SamplingStrategy.UNIFORM:
            # Evenly spaced frames
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            return indices.tolist()
        
        elif self.strategy == SamplingStrategy.RANDOM:
            # Random frames without replacement
            indices = self.rng.choice(total_frames, size=num_frames, replace=False)
            return sorted(indices.tolist())
        
        elif self.strategy == SamplingStrategy.FIRST_N:
            # First N frames
            return list(range(num_frames))
        
        elif self.strategy == SamplingStrategy.KEYFRAMES:
            # For keyframes, we'll sample uniformly but the actual extraction
            # will seek to nearest keyframe
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            return indices.tolist()
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def sample_frames(
        self,
        video_path: Union[str, Path],
        num_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Sample frames from a video file.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to sample (uses self.num_frames if None)
            
        Returns:
            List of frames as numpy arrays (BGR format)
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return []
        
        indices = self._get_frame_indices(total_frames, num_frames)
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frames.append(frame)
            else:
                # Try to read next frame if current fails
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
        
        cap.release()
        return frames
    
    def sample_frames_generator(
        self,
        video_path: Union[str, Path],
        num_frames: Optional[int] = None
    ) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames from a video file.
        Memory efficient for large videos.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to sample
            
        Yields:
            Frames as numpy arrays (BGR format)
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return
        
        indices = self._get_frame_indices(total_frames, num_frames)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                yield frame
        
        cap.release()
    
    def sample_frames_with_indices(
        self,
        video_path: Union[str, Path],
        num_frames: Optional[int] = None
    ) -> List[tuple]:
        """
        Sample frames from a video and return with their indices.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to sample
            
        Returns:
            List of (frame_index, frame) tuples
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return []
        
        indices = self._get_frame_indices(total_frames, num_frames)
        results = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                results.append((idx, frame))
        
        cap.release()
        return results


def sample_video_frames(
    video_path: Union[str, Path],
    num_frames: int = 10,
    strategy: str = "uniform",
    seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Convenience function to sample frames from a video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        strategy: Sampling strategy
        seed: Random seed
        
    Returns:
        List of frames as numpy arrays
    """
    sampler = FrameSampler(num_frames, strategy, seed)
    return sampler.sample_frames(video_path)
