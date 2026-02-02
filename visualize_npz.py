#!/usr/bin/env python3
"""
Visualize preprocessed .npz files from the cache directories.

These files contain face crops extracted from videos during preprocessing.

Usage:
    # Visualize a specific npz file
    python visualize_npz.py --file Datasets/Celeb-DF-v2/cache/train/00001_1234567890.npz
    
    # Visualize random samples from a dataset split
    python visualize_npz.py --dataset Celeb-DF-v2 --split train --num_samples 5
    
    # Visualize from FF++ dataset
    python visualize_npz.py --dataset FF --split train --num_samples 5
    
    # Save visualization to file instead of displaying
    python visualize_npz.py --dataset Celeb-DF-v2 --split train --save output.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random


def load_npz_faces(npz_path: Path) -> list:
    """
    Load face crops from an npz file.
    
    Args:
        npz_path: Path to the .npz file
        
    Returns:
        List of face images as numpy arrays
    """
    data = np.load(npz_path)
    faces = []
    
    # Get all face keys and sort them
    face_keys = sorted([k for k in data.keys() if k.startswith('face_')],
                       key=lambda x: int(x.split('_')[1]))
    
    for key in face_keys:
        faces.append(data[key])
    
    return faces


def visualize_single_npz(npz_path: Path, save_path: Path = None):
    """
    Visualize all frames from a single npz file.
    
    Args:
        npz_path: Path to the .npz file
        save_path: Optional path to save the figure
    """
    faces = load_npz_faces(npz_path)
    num_faces = len(faces)
    
    if num_faces == 0:
        print(f"No faces found in {npz_path}")
        return
    
    # Create subplot grid
    cols = min(5, num_faces)
    rows = (num_faces + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    fig.suptitle(f"Faces from: {npz_path.name}\n({num_faces} frames)", fontsize=12)
    
    # Handle single row/col case
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, face in enumerate(faces):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        
        # Convert BGR to RGB (OpenCV stores in BGR, matplotlib expects RGB)
        face_rgb = face[:, :, ::-1]
        ax.imshow(face_rgb)
        ax.set_title(f"Frame {idx}", fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_faces, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_multiple_npz(npz_paths: list, save_path: Path = None):
    """
    Visualize multiple npz files in a grid (one row per file).
    
    Args:
        npz_paths: List of paths to .npz files
        save_path: Optional path to save the figure
    """
    if not npz_paths:
        print("No npz files provided")
        return
    
    # Load all faces
    all_faces = []
    valid_paths = []
    max_frames = 0
    
    for path in npz_paths:
        try:
            faces = load_npz_faces(path)
            if faces:
                all_faces.append(faces)
                valid_paths.append(path)
                max_frames = max(max_frames, len(faces))
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    if not all_faces:
        print("No valid npz files found")
        return
    
    num_videos = len(all_faces)
    
    fig, axes = plt.subplots(num_videos, max_frames, 
                             figsize=(2.5 * max_frames, 2.5 * num_videos))
    fig.suptitle(f"Preprocessed Face Crops ({num_videos} videos, {max_frames} frames each)", 
                 fontsize=14)
    
    # Handle edge cases for axes shape
    if num_videos == 1 and max_frames == 1:
        axes = np.array([[axes]])
    elif num_videos == 1:
        axes = axes.reshape(1, -1)
    elif max_frames == 1:
        axes = axes.reshape(-1, 1)
    
    for vid_idx, (faces, path) in enumerate(zip(all_faces, valid_paths)):
        for frame_idx in range(max_frames):
            ax = axes[vid_idx, frame_idx]
            
            if frame_idx < len(faces):
                # Convert BGR to RGB (OpenCV stores in BGR, matplotlib expects RGB)
                face_rgb = faces[frame_idx][:, :, ::-1]
                ax.imshow(face_rgb)
                if frame_idx == 0:
                    # Show filename on first frame
                    ax.set_ylabel(path.stem[:20], fontsize=8, rotation=0, 
                                  ha='right', va='center')
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def get_npz_files(dataset: str, split: str, base_dir: Path = None) -> list:
    """
    Get list of all npz files for a dataset split.
    
    Args:
        dataset: Dataset name ('Celeb-DF-v2' or 'FF')
        split: Data split ('train', 'val', or 'test')
        base_dir: Base directory (defaults to script location)
        
    Returns:
        List of Path objects to npz files
    """
    if base_dir is None:
        base_dir = Path(__file__).parent
    
    cache_dir = base_dir / "Datasets" / dataset / "cache" / split
    
    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        return []
    
    npz_files = list(cache_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} npz files in {cache_dir}")
    
    return npz_files


def print_npz_info(npz_path: Path):
    """Print information about an npz file."""
    data = np.load(npz_path)
    
    print(f"\nFile: {npz_path}")
    print(f"Keys: {list(data.keys())}")
    
    for key in sorted(data.keys()):
        arr = data[key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}, "
              f"min={arr.min()}, max={arr.max()}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize preprocessed npz face crop files"
    )
    parser.add_argument(
        "--file", "-f", type=str,
        help="Path to a specific npz file to visualize"
    )
    parser.add_argument(
        "--dataset", "-d", type=str, choices=["Celeb-DF-v2", "FF"],
        help="Dataset to visualize from"
    )
    parser.add_argument(
        "--split", "-s", type=str, default="train",
        choices=["train", "val", "test"],
        help="Data split to visualize"
    )
    parser.add_argument(
        "--num_samples", "-n", type=int, default=3,
        help="Number of random samples to visualize"
    )
    parser.add_argument(
        "--save", type=str,
        help="Path to save the visualization (instead of displaying)"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Print detailed info about the npz file(s)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible sampling"
    )
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    save_path = Path(args.save) if args.save else None
    
    if args.file:
        # Visualize specific file
        npz_path = Path(args.file)
        if not npz_path.exists():
            print(f"File not found: {npz_path}")
            return
        
        if args.info:
            print_npz_info(npz_path)
        
        visualize_single_npz(npz_path, save_path)
        
    elif args.dataset:
        # Visualize random samples from dataset
        npz_files = get_npz_files(args.dataset, args.split)
        
        if not npz_files:
            return
        
        # Sample random files
        num_samples = min(args.num_samples, len(npz_files))
        sampled_files = random.sample(npz_files, num_samples)
        
        print(f"\nSampled files:")
        for f in sampled_files:
            print(f"  - {f.name}")
        
        if args.info:
            for f in sampled_files:
                print_npz_info(f)
        
        if num_samples == 1:
            visualize_single_npz(sampled_files[0], save_path)
        else:
            visualize_multiple_npz(sampled_files, save_path)
    else:
        parser.print_help()
        print("\n\nExample usage:")
        print("  python visualize_npz.py --dataset Celeb-DF-v2 --split train --num_samples 3")
        print("  python visualize_npz.py --file Datasets/Celeb-DF-v2/cache/train/00011_3676378839.npz")


if __name__ == "__main__":
    main()
