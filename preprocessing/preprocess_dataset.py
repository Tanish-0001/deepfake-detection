#!/usr/bin/env python3
"""
Batch preprocessing script for deepfake detection dataset.

This script preprocesses all videos in the dataset (extracting faces, cropping,
enlarging, aligning) and caches the results to disk. This allows training to
run much faster since expensive face detection only needs to happen once.

Usage:
    # Preprocess all splits
    python -m preprocessing.preprocess_dataset --all
    
    # Preprocess specific split
    python -m preprocessing.preprocess_dataset --split train
    
    # Preprocess with custom settings
    python -m preprocessing.preprocess_dataset --split train --workers 4 --frames 10
    
    # Force reprocessing (ignore existing cache)
    python -m preprocessing.preprocess_dataset --split train --force
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

from config.config import Config, DataConfig
from preprocessing.pipeline import PreprocessingPipeline


class DatasetPreprocessor:
    """
    Batch preprocessor for entire dataset.
    
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
        device: str = "cuda",
        compression: str = "c23"
    ):
        """
        Initialize dataset preprocessor.
        
        Args:
            dataset_root: Root directory of dataset (e.g., Datasets/FF)
            cache_dir: Directory to store preprocessed cache files
            frames_per_video: Number of frames to extract per video
            output_size: Target size for face crops (width, height)
            bbox_enlargement: Factor to enlarge face bounding boxes
            face_detector: Face detector to use ('retinaface', etc.)
            detection_threshold: Minimum confidence for face detection
            device: Device for face detection ('cuda' or 'cpu')
            compression: Compression level for FF++ ('c23', 'c40', 'raw')
        """
        self.dataset_root = Path(dataset_root)
        self.cache_dir = Path(cache_dir)
        self.frames_per_video = frames_per_video
        self.output_size = output_size
        self.bbox_enlargement = bbox_enlargement
        self.compression = compression
        
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
            # Extract faces
            faces = self.pipeline.process_video(
                video_path,
                return_frames_without_faces=True
            )
            
            # Ensure consistent number of frames
            original_count = len(faces)
            if len(faces) < self.frames_per_video:
                while len(faces) < self.frames_per_video:
                    if faces:
                        faces.append(faces[-1].copy())
                    else:
                        faces.append(np.zeros((*self.output_size, 3), dtype=np.uint8))
            elif len(faces) > self.frames_per_video:
                faces = faces[:self.frames_per_video]
            
            # Save to cache (compressed numpy format)
            cache_dict = {f'face_{i}': face for i, face in enumerate(faces)}
            np.savez_compressed(cache_path, **cache_dict)
            
            return {
                'status': 'processed',
                'video': str(video_path),
                'cache_path': str(cache_path),
                'faces_extracted': original_count,
                'faces_stored': len(faces)
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'video': str(video_path),
                'error': str(e)
            }
    
    def load_video_list(self, split: str) -> List[Dict[str, Any]]:
        """Load video list from JSON split file."""
        json_path = self.dataset_root / f"{split}_paths.json"
        
        if not json_path.exists():
            print(f"Warning: {json_path} not found")
            return []
        
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def preprocess_split(
        self,
        split: str,
        force: bool = False,
        max_videos: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Preprocess all videos in a split.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            force: Force reprocessing even if cached
            max_videos: Maximum number of videos to process (for testing)
            show_progress: Show progress bar
            
        Returns:
            Dict with preprocessing statistics
        """
        print(f"\n{'='*60}")
        print(f"Preprocessing {split} split")
        print(f"{'='*60}")
        
        # Load video list
        video_entries = self.load_video_list(split)
        
        if not video_entries:
            print(f"No videos found for {split} split")
            return {'split': split, 'total': 0}
        
        # Limit videos if requested
        if max_videos:
            video_entries = video_entries[:max_videos]
        
        # Count existing cache
        cached_count = sum(
            1 for entry in video_entries
            if self.is_cached(Path(entry['path']), split)
        )
        
        if not force:
            print(f"Found {cached_count}/{len(video_entries)} videos already cached")
            if cached_count == len(video_entries):
                print("All videos already cached! Use --force to reprocess.")
                return {
                    'split': split,
                    'total': len(video_entries),
                    'cached': cached_count,
                    'processed': 0,
                    'failed': 0
                }
        
        # Process videos
        stats = {
            'split': split,
            'total': len(video_entries),
            'processed': 0,
            'cached': 0,
            'failed': 0,
            'failed_videos': []
        }
        
        iterator = tqdm(video_entries, desc=f"Processing {split}") if show_progress else video_entries
        
        for entry in iterator:
            video_path = Path(entry['path'])
            
            if not video_path.exists():
                stats['failed'] += 1
                stats['failed_videos'].append({
                    'path': str(video_path),
                    'error': 'File not found'
                })
                continue
            
            result = self.process_video(video_path, split, force=force)
            
            if result['status'] == 'cached':
                stats['cached'] += 1
            elif result['status'] == 'processed':
                stats['processed'] += 1
            else:
                stats['failed'] += 1
                stats['failed_videos'].append({
                    'path': result['video'],
                    'error': result.get('error', 'Unknown error')
                })
        
        # Print summary
        print(f"\n{split.upper()} Split Summary:")
        print(f"  Total videos: {stats['total']}")
        print(f"  Already cached: {stats['cached']}")
        print(f"  Newly processed: {stats['processed']}")
        print(f"  Failed: {stats['failed']}")
        
        if stats['failed_videos']:
            print(f"\n  Failed videos:")
            for fv in stats['failed_videos'][:5]:
                print(f"    - {fv['path']}: {fv['error']}")
            if len(stats['failed_videos']) > 5:
                print(f"    ... and {len(stats['failed_videos']) - 5} more")
        
        return stats
    
    def preprocess_all(
        self,
        force: bool = False,
        max_videos_per_split: Optional[int] = None
    ) -> Dict[str, Dict]:
        """
        Preprocess all splits (train, val, test).
        
        Args:
            force: Force reprocessing
            max_videos_per_split: Max videos per split (for testing)
            
        Returns:
            Dict with stats for each split
        """
        start_time = time.time()
        
        all_stats = {}
        for split in ['train', 'val', 'test']:
            all_stats[split] = self.preprocess_split(
                split,
                force=force,
                max_videos=max_videos_per_split
            )
        
        elapsed = time.time() - start_time
        
        # Print overall summary
        print(f"\n{'='*60}")
        print("PREPROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        
        total_processed = sum(s.get('processed', 0) for s in all_stats.values())
        total_cached = sum(s.get('cached', 0) for s in all_stats.values())
        total_failed = sum(s.get('failed', 0) for s in all_stats.values())
        
        print(f"Total processed: {total_processed}")
        print(f"Total cached: {total_cached}")
        print(f"Total failed: {total_failed}")
        print(f"\nCache location: {self.cache_dir}")
        
        return all_stats
    
    def verify_cache(self, split: str) -> Dict[str, Any]:
        """
        Verify integrity of cached files for a split.
        
        Args:
            split: Dataset split to verify
            
        Returns:
            Verification results
        """
        print(f"\nVerifying {split} cache...")
        
        video_entries = self.load_video_list(split)
        
        results = {
            'split': split,
            'total': len(video_entries),
            'valid': 0,
            'missing': 0,
            'corrupted': 0,
            'issues': []
        }
        
        for entry in tqdm(video_entries, desc="Verifying"):
            video_path = Path(entry['path'])
            cache_path = self.get_cache_path(video_path, split)
            
            if not cache_path.exists():
                results['missing'] += 1
                results['issues'].append({
                    'video': str(video_path),
                    'issue': 'Cache file missing'
                })
                continue
            
            try:
                data = np.load(cache_path)
                num_faces = len(data.files)
                
                if num_faces != self.frames_per_video:
                    results['corrupted'] += 1
                    results['issues'].append({
                        'video': str(video_path),
                        'issue': f'Wrong number of faces: {num_faces} (expected {self.frames_per_video})'
                    })
                else:
                    results['valid'] += 1
                    
            except Exception as e:
                results['corrupted'] += 1
                results['issues'].append({
                    'video': str(video_path),
                    'issue': f'Load error: {str(e)}'
                })
        
        print(f"\n{split.upper()} Cache Verification:")
        print(f"  Valid: {results['valid']}/{results['total']}")
        print(f"  Missing: {results['missing']}")
        print(f"  Corrupted: {results['corrupted']}")
        
        return results


def create_preprocessor_from_config(config: Config) -> DatasetPreprocessor:
    """Create DatasetPreprocessor from config object."""
    return DatasetPreprocessor(
        dataset_root=config.data.dataset_root,
        cache_dir=config.data.dataset_root / "cache",
        frames_per_video=config.data.frames_per_video,
        output_size=config.data.output_size,
        bbox_enlargement=config.data.bbox_enlargement_factor,
        face_detector=config.data.face_detector,
        detection_threshold=config.data.face_detection_threshold,
        device=config.training.device,
        compression=config.data.compression
    )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess dataset videos and cache face crops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess all splits
  python -m preprocessing.preprocess_dataset --all
  
  # Preprocess only training data
  python -m preprocessing.preprocess_dataset --split train
  
  # Force reprocessing (ignore existing cache)
  python -m preprocessing.preprocess_dataset --split train --force
  
  # Verify cache integrity
  python -m preprocessing.preprocess_dataset --split train --verify
  
  # Custom settings
  python -m preprocessing.preprocess_dataset --all --frames 10 --output-size 224
        """
    )
    
    # What to process
    parser.add_argument('--all', action='store_true',
                        help='Preprocess all splits (train, val, test)')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'],
                        help='Preprocess specific split')
    
    # Processing options
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing even if cached')
    parser.add_argument('--verify', action='store_true',
                        help='Verify cache integrity instead of processing')
    parser.add_argument('--max-videos', type=int, default=None,
                        help='Maximum videos to process (for testing)')
    
    # Pipeline settings
    parser.add_argument('--frames', type=int, default=None,
                        help='Frames per video (default: from config)')
    parser.add_argument('--output-size', type=int, default=None,
                        help='Face crop size (default: from config)')
    parser.add_argument('--bbox-enlarge', type=float, default=None,
                        help='Bounding box enlargement factor (default: from config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device for face detection (cuda/cpu)')
    
    # Dataset paths
    parser.add_argument('--dataset-root', type=str, default=None,
                        help='Dataset root directory')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Cache directory')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and not args.split:
        parser.error("Please specify --all or --split")
    
    # Load default config
    config = Config()
    
    # Override with command line arguments
    if args.frames:
        config.data.frames_per_video = args.frames
    if args.output_size:
        config.data.output_size = (args.output_size, args.output_size)
    if args.bbox_enlarge:
        config.data.bbox_enlargement_factor = args.bbox_enlarge
    if args.device:
        config.training.device = args.device
    if args.dataset_root:
        config.data.dataset_root = Path(args.dataset_root)
    
    # Create preprocessor
    cache_dir = Path(args.cache_dir) if args.cache_dir else config.data.dataset_root / "cache"
    
    preprocessor = DatasetPreprocessor(
        dataset_root=config.data.dataset_root,
        cache_dir=cache_dir,
        frames_per_video=config.data.frames_per_video,
        output_size=config.data.output_size,
        bbox_enlargement=config.data.bbox_enlargement_factor,
        face_detector=config.data.face_detector,
        detection_threshold=config.data.face_detection_threshold,
        device=config.training.device,
        compression=config.data.compression
    )
    
    print(f"Dataset root: {config.data.dataset_root}")
    print(f"Cache directory: {cache_dir}")
    print(f"Frames per video: {config.data.frames_per_video}")
    print(f"Output size: {config.data.output_size}")
    print(f"BBox enlargement: {config.data.bbox_enlargement_factor}")
    print(f"Device: {config.training.device}")
    
    if args.verify:
        # Verify mode
        if args.all:
            for split in ['train', 'val', 'test']:
                preprocessor.verify_cache(split)
        else:
            preprocessor.verify_cache(args.split)
    else:
        # Preprocess mode
        if args.all:
            preprocessor.preprocess_all(
                force=args.force,
                max_videos_per_split=args.max_videos
            )
        else:
            preprocessor.preprocess_split(
                args.split,
                force=args.force,
                max_videos=args.max_videos
            )


if __name__ == "__main__":
    main()
