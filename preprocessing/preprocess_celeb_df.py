#!/usr/bin/env python3
"""
Batch preprocessing script for Celeb-DF-v2 dataset.

This script preprocesses all videos in the Celeb-DF-v2 dataset (extracting faces, 
cropping, enlarging, aligning) and caches the results to disk. This allows training 
to run much faster since expensive face detection only needs to happen once.

Usage:
    # First generate splits (if not already done)
    python -m data.generate_celeb_df_splits --data_dir ./Datasets/Celeb-DF-v2
    
    # Preprocess all splits
    python -m preprocessing.preprocess_celeb_df --all
    
    # Preprocess specific split
    python -m preprocessing.preprocess_celeb_df --split train
    
    # Preprocess with custom settings
    python -m preprocessing.preprocess_celeb_df --split train --workers 4 --frames 10
    
    # Force reprocessing (ignore existing cache)
    python -m preprocessing.preprocess_celeb_df --split train --force
"""

import argparse
import json
import hashlib
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
from tqdm import tqdm


def _deterministic_hash(s: str) -> int:
    """Compute a deterministic hash for a string (consistent across Python sessions)."""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from preprocessing.pipeline import PreprocessingPipeline


class CelebDFPreprocessor:
    """
    Batch preprocessor for Celeb-DF-v2 dataset.
    
    Preprocesses all videos and saves face crops to cache for fast loading.
    """
    
    def __init__(
        self,
        dataset_root: Path,
        cache_dir: Path,
        frames_per_video: int = 10,
        output_size: tuple = (224, 224),
        bbox_enlargement: float = 1.3,
        face_detector: str = "retinaface",
        detection_threshold: float = 0.9,
        device: str = "cuda"
    ):
        """
        Initialize dataset preprocessor.
        
        Args:
            dataset_root: Root directory of Celeb-DF-v2 dataset
            cache_dir: Directory to store preprocessed cache files
            frames_per_video: Number of frames to extract per video
            output_size: Target size for face crops (width, height)
            bbox_enlargement: Factor to enlarge face bounding boxes
            face_detector: Face detector to use ('retinaface', etc.)
            detection_threshold: Minimum confidence for face detection
            device: Device for face detection ('cuda' or 'cpu')
        """
        self.dataset_root = Path(dataset_root)
        self.cache_dir = Path(cache_dir)
        self.frames_per_video = frames_per_video
        self.output_size = output_size
        self.bbox_enlargement = bbox_enlargement
        
        # Create pipeline
        self.pipeline = PreprocessingPipeline(
            num_frames=frames_per_video,
            sampling_strategy="uniform",
            face_detector=face_detector,
            detection_threshold=detection_threshold,
            output_size=output_size,
            bbox_enlargement=bbox_enlargement,
            align_faces=True,
            device=device
        )
        
        # Stats tracking
        self.stats = {
            'processed': 0,
            'cached': 0,
            'failed': 0,
            'total_faces': 0,
            'no_face_frames': 0
        }
    
    def get_cache_path(self, video_path: Path, split: str) -> Path:
        """
        Get cache file path for a video.
        
        Uses consistent naming with video stem and deterministic hash for uniqueness.
        """
        cache_subdir = self.cache_dir / split
        cache_subdir.mkdir(parents=True, exist_ok=True)
        cache_name = f"{video_path.stem}_{_deterministic_hash(str(video_path))}.npz"
        return cache_subdir / cache_name
    
    def is_cached(self, video_path: Path, split: str) -> bool:
        """Check if video is already cached."""
        cache_path = self.get_cache_path(video_path, split)
        return cache_path.exists()
    
    def process_video(
        self,
        video_path: Path,
        split: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single video and cache results.
        
        Args:
            video_path: Path to video file
            split: Dataset split (train/val/test)
            force: Force reprocessing even if cached
            
        Returns:
            Dict with processing results and stats
        """
        cache_path = self.get_cache_path(video_path, split)
        
        # Check cache
        if not force and cache_path.exists():
            return {
                'status': 'cached',
                'video': str(video_path),
                'cache_path': str(cache_path)
            }
        
        try:
            # Process video
            faces = self.pipeline.process_video(
                video_path,
                return_frames_without_faces=True
            )
            
            # Ensure we have exactly frames_per_video faces
            num_no_face = 0
            if len(faces) < self.frames_per_video:
                num_no_face = self.frames_per_video - len(faces)
                while len(faces) < self.frames_per_video:
                    if faces:
                        faces.append(faces[-1].copy())
                    else:
                        faces.append(np.zeros((*self.output_size, 3), dtype=np.uint8))
            elif len(faces) > self.frames_per_video:
                faces = faces[:self.frames_per_video]
            
            # Save to cache
            cache_dict = {f'face_{i}': face for i, face in enumerate(faces)}
            np.savez_compressed(cache_path, **cache_dict)
            
            return {
                'status': 'processed',
                'video': str(video_path),
                'cache_path': str(cache_path),
                'num_faces': len(faces),
                'no_face_frames': num_no_face
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'video': str(video_path),
                'error': str(e)
            }
    
    def load_split_videos(self, split: str) -> List[Dict]:
        """Load video list for a specific split."""
        json_path = self.dataset_root / f"{split}_paths.json"
        
        if not json_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {json_path}\n"
                f"Please run: python -m data.generate_celeb_df_splits --data_dir {self.dataset_root}"
            )
        
        with open(json_path, 'r') as f:
            videos = json.load(f)
        
        return videos
    
    def preprocess_split(
        self,
        split: str,
        force: bool = False,
        num_workers: int = 1,
        max_videos: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Preprocess all videos in a split.
        
        Args:
            split: Dataset split to preprocess
            force: Force reprocessing even if cached
            num_workers: Number of worker processes (1 = sequential)
            max_videos: Maximum number of videos to process (for testing)
            
        Returns:
            Dict with preprocessing statistics
        """
        print(f"\nPreprocessing {split} split...")
        
        # Load video list
        videos = self.load_split_videos(split)
        
        if max_videos:
            videos = videos[:max_videos]
        
        print(f"Found {len(videos)} videos")
        
        # Reset stats
        stats = {
            'processed': 0,
            'cached': 0,
            'failed': 0,
            'total_faces': 0,
            'no_face_frames': 0,
            'failed_videos': []
        }
        
        # Process videos
        start_time = time.time()
        
        # For now, always process sequentially (face detection uses GPU)
        # Multi-processing can cause issues with GPU resources
        for video_entry in tqdm(videos, desc=f"Processing {split}"):
            video_path = Path(video_entry["path"])
            result = self.process_video(video_path, split, force)
            
            if result['status'] == 'cached':
                stats['cached'] += 1
            elif result['status'] == 'processed':
                stats['processed'] += 1
                stats['total_faces'] += result.get('num_faces', 0)
                stats['no_face_frames'] += result.get('no_face_frames', 0)
            else:
                stats['failed'] += 1
                stats['failed_videos'].append({
                    'video': result['video'],
                    'error': result.get('error', 'Unknown error')
                })
        
        elapsed = time.time() - start_time
        stats['elapsed_time'] = elapsed
        stats['videos_per_second'] = len(videos) / elapsed if elapsed > 0 else 0
        
        return stats
    
    def preprocess_all(
        self,
        force: bool = False,
        num_workers: int = 1,
        max_videos: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Preprocess all splits.
        
        Args:
            force: Force reprocessing even if cached
            num_workers: Number of worker processes
            max_videos: Maximum videos per split (for testing)
            
        Returns:
            Dict mapping split names to their stats
        """
        all_stats = {}
        
        for split in ['train', 'val', 'test']:
            try:
                stats = self.preprocess_split(
                    split,
                    force=force,
                    num_workers=num_workers,
                    max_videos=max_videos
                )
                all_stats[split] = stats
                
                # Print split summary
                print(f"\n{split.upper()} Split Summary:")
                print(f"  Processed: {stats['processed']}")
                print(f"  Cached: {stats['cached']}")
                print(f"  Failed: {stats['failed']}")
                print(f"  Time: {stats['elapsed_time']:.1f}s")
                
                if stats['failed_videos']:
                    print(f"  Failed videos:")
                    for fv in stats['failed_videos'][:5]:
                        print(f"    - {fv['video']}: {fv['error']}")
                    if len(stats['failed_videos']) > 5:
                        print(f"    ... and {len(stats['failed_videos']) - 5} more")
                        
            except FileNotFoundError as e:
                print(f"Skipping {split}: {e}")
                all_stats[split] = {'error': str(e)}
        
        return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Celeb-DF-v2 dataset for deepfake detection"
    )
    
    # Dataset options
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./Datasets/Celeb-DF-v2",
        help="Path to Celeb-DF-v2 dataset"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Path to cache directory (default: data_dir/cache)"
    )
    
    # Split selection
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        help="Specific split to preprocess"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Preprocess all splits"
    )
    
    # Processing options
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="Number of frames per video (default: 10)"
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=224,
        help="Output face crop size (default: 224)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if cached"
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Maximum videos to process per split (for testing)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for face detection (default: cuda)"
    )
    
    args = parser.parse_args()
    
    if not args.split and not args.all:
        parser.error("Please specify --split or --all")
    
    # Setup paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {data_dir}")
    
    cache_dir = Path(args.cache_dir) if args.cache_dir else data_dir / "cache"
    
    print(f"Celeb-DF-v2 Preprocessing")
    print(f"========================")
    print(f"Dataset: {data_dir}")
    print(f"Cache: {cache_dir}")
    print(f"Frames per video: {args.frames}")
    print(f"Output size: {args.output_size}x{args.output_size}")
    print(f"Device: {args.device}")
    print(f"Force reprocess: {args.force}")
    
    # Create preprocessor
    preprocessor = CelebDFPreprocessor(
        dataset_root=data_dir,
        cache_dir=cache_dir,
        frames_per_video=args.frames,
        output_size=(args.output_size, args.output_size),
        device=args.device
    )
    
    # Run preprocessing
    if args.all:
        all_stats = preprocessor.preprocess_all(
            force=args.force,
            num_workers=args.workers,
            max_videos=args.max_videos
        )
        
        # Print final summary
        print("\n" + "=" * 50)
        print("FINAL SUMMARY")
        print("=" * 50)
        
        total_processed = sum(s.get('processed', 0) for s in all_stats.values() if isinstance(s, dict))
        total_cached = sum(s.get('cached', 0) for s in all_stats.values() if isinstance(s, dict))
        total_failed = sum(s.get('failed', 0) for s in all_stats.values() if isinstance(s, dict))
        
        print(f"Total processed: {total_processed}")
        print(f"Total cached: {total_cached}")
        print(f"Total failed: {total_failed}")
        
    else:
        stats = preprocessor.preprocess_split(
            args.split,
            force=args.force,
            num_workers=args.workers,
            max_videos=args.max_videos
        )
        
        print("\n" + "=" * 50)
        print(f"{args.split.upper()} SPLIT SUMMARY")
        print("=" * 50)
        print(f"Processed: {stats['processed']}")
        print(f"Cached: {stats['cached']}")
        print(f"Failed: {stats['failed']}")
        print(f"Time: {stats['elapsed_time']:.1f}s")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
