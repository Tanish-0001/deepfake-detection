"""
Complete preprocessing pipeline combining frame sampling, face extraction, and transforms.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import pickle

from .face_extractor import FaceExtractor, RetinaFaceExtractor, create_face_extractor
from .frame_sampler import FrameSampler
from .transforms import get_train_transforms, get_val_transforms, TransformConfig


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for deepfake detection.
    
    Pipeline steps:
    1. Sample frames from video
    2. Extract face from each frame using RetinaFace
    3. Align face and find bounding box
    4. Enlarge bounding box, crop and resize to target size
    5. Apply transforms (normalization, augmentation)
    """
    
    def __init__(
        self,
        num_frames: int = 10,
        sampling_strategy: str = "uniform",
        face_detector: str = "retinaface",
        detection_threshold: float = 0.9,
        output_size: Tuple[int, int] = (224, 224),
        bbox_enlargement: float = 1.3,
        align_faces: bool = True,
        device: str = "cuda",
        seed: Optional[int] = None
    ):
        """
        Initialize preprocessing pipeline.
        
        Args:
            num_frames: Number of frames to sample per video
            sampling_strategy: Frame sampling strategy
            face_detector: Face detector type
            detection_threshold: Minimum confidence for face detection
            output_size: Target output size for face crops
            bbox_enlargement: Factor to enlarge bounding boxes
            align_faces: Whether to align faces based on landmarks
            device: Device for face detection model
            seed: Random seed for reproducibility
        """
        self.num_frames = num_frames
        self.output_size = output_size
        self.bbox_enlargement = bbox_enlargement
        self.align_faces = align_faces
        
        # Initialize components
        self.frame_sampler = FrameSampler(
            num_frames=num_frames,
            strategy=sampling_strategy,
            seed=seed
        )
        
        self.face_extractor = create_face_extractor(
            detector_type=face_detector,
            detection_threshold=detection_threshold,
            device=device
        )
    
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single frame: extract and crop face.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            Cropped and resized face, or None if no face detected
        """
        face = self.face_extractor.extract_face(
            frame,
            output_size=self.output_size,
            enlargement_factor=self.bbox_enlargement,
            align=self.align_faces
        )
        return face
    
    def process_video(
        self,
        video_path: Union[str, Path],
        return_frames_without_faces: bool = False
    ) -> List[np.ndarray]:
        """
        Process a video: sample frames and extract faces.
        
        Args:
            video_path: Path to video file
            return_frames_without_faces: If True, return original frame when no face detected
            
        Returns:
            List of face crops (BGR format)
        """
        # Sample frames
        frames = self.frame_sampler.sample_frames(video_path)
        
        faces = []
        for frame in frames:
            face = self.process_frame(frame)
            
            if face is not None:
                faces.append(face)
            elif return_frames_without_faces:
                # Resize original frame as fallback
                resized = cv2.resize(frame, self.output_size)
                faces.append(resized)
        
        return faces
    
    def process_video_with_info(
        self,
        video_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Process a video and return detailed information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with processed faces and metadata
        """
        video_path = Path(video_path)
        
        # Get video info
        video_info = self.frame_sampler.get_video_info(video_path)
        
        # Sample frames with indices
        frame_data = self.frame_sampler.sample_frames_with_indices(video_path)
        
        faces = []
        face_indices = []
        failed_indices = []
        
        for idx, frame in frame_data:
            face = self.process_frame(frame)
            
            if face is not None:
                faces.append(face)
                face_indices.append(idx)
            else:
                failed_indices.append(idx)
        
        return {
            'video_path': str(video_path),
            'video_info': video_info,
            'faces': faces,
            'face_indices': face_indices,
            'failed_indices': failed_indices,
            'num_faces_extracted': len(faces),
            'num_frames_sampled': len(frame_data)
        }
    
    def process_video_batch(
        self,
        video_paths: List[Union[str, Path]],
        num_workers: int = 4,
        show_progress: bool = True
    ) -> Dict[str, List[np.ndarray]]:
        """
        Process multiple videos in parallel.
        
        Args:
            video_paths: List of video paths
            num_workers: Number of parallel workers
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping video paths to their extracted faces
        """
        results = {}
        
        # Note: Face detection models may not be thread-safe,
        # so we process sequentially but could parallelize I/O
        iterator = tqdm(video_paths, desc="Processing videos") if show_progress else video_paths
        
        for video_path in iterator:
            try:
                faces = self.process_video(video_path)
                results[str(video_path)] = faces
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                results[str(video_path)] = []
        
        return results
    
    def preprocess_and_cache(
        self,
        video_paths: List[Union[str, Path]],
        cache_dir: Union[str, Path],
        show_progress: bool = True
    ) -> Dict[str, Path]:
        """
        Preprocess videos and cache results to disk.
        
        Args:
            video_paths: List of video paths
            cache_dir: Directory to save cached results
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping video paths to cache file paths
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_map = {}
        
        iterator = tqdm(video_paths, desc="Preprocessing") if show_progress else video_paths
        
        for video_path in iterator:
            video_path = Path(video_path)
            
            # Create unique cache filename
            cache_name = f"{video_path.stem}_{hash(str(video_path)) % 10**8}.pkl"
            cache_path = cache_dir / cache_name
            
            # Check if already cached
            if cache_path.exists():
                cache_map[str(video_path)] = cache_path
                continue
            
            try:
                # Process video
                result = self.process_video_with_info(video_path)
                
                # Save to cache
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
                
                cache_map[str(video_path)] = cache_path
                
            except Exception as e:
                print(f"Error preprocessing {video_path}: {e}")
        
        return cache_map
    
    @staticmethod
    def load_cached(cache_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load cached preprocessing results.
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            Cached preprocessing results
        """
        with open(cache_path, 'rb') as f:
            return pickle.load(f)


def create_pipeline_from_config(config) -> PreprocessingPipeline:
    """
    Create preprocessing pipeline from configuration object.
    
    Args:
        config: Configuration object (from config module)
        
    Returns:
        PreprocessingPipeline instance
    """
    return PreprocessingPipeline(
        num_frames=config.data.frames_per_video,
        sampling_strategy=config.data.frame_sampling_strategy,
        face_detector=config.data.face_detector,
        detection_threshold=config.data.face_detection_threshold,
        output_size=config.data.output_size,
        bbox_enlargement=config.data.bbox_enlargement_factor,
        align_faces=True,
        device=config.training.device,
        seed=config.seed
    )
