"""
Generate train/val/test splits for Celeb-DF-v2 dataset.

This script reads the List_of_testing_videos.txt file to get the official test split,
then splits the remaining videos into train and validation sets.

Celeb-DF-v2 structure:
- Celeb-real/: Real celebrity videos (label 0)
- Celeb-synthesis/: Fake deepfake videos (label 1)
- YouTube-real/: Real YouTube videos (label 0)
- List_of_testing_videos.txt: Official test split

Labels (consistent with FF dataset):
- 0 = Real
- 1 = Fake

Usage:
    python -m data.generate_celeb_df_splits --data_dir ./Datasets/Celeb-DF-v2
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple, Set


def parse_test_list(data_dir: Path) -> List[Dict]:
    """
    Parse List_of_testing_videos.txt to get test videos.
    
    Format in file: <label> <relative_path>
    - file label 0 = Fake -> converted to 1
    - file label 1 = Real -> converted to 0
    
    Output labels (consistent with FF dataset):
    - 0 = Real
    - 1 = Fake
    
    Returns:
        List of dicts with 'path' and 'label' keys
    """
    test_list_path = data_dir / "List_of_testing_videos.txt"
    
    if not test_list_path.exists():
        raise FileNotFoundError(f"Test list not found: {test_list_path}")
    
    test_videos = []
    with open(test_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(' ', 1)
            if len(parts) != 2:
                print(f"Warning: Skipping malformed line: {line}")
                continue
            
            file_label = int(parts[0])
            rel_path = parts[1]
            full_path = data_dir / rel_path
            
            # Convert labels: file has 1=real, 0=fake
            # We want: 0=real, 1=fake (consistent with FF dataset)
            label = 0 if file_label == 1 else 1
            
            test_videos.append({
                "path": str(full_path),
                "label": label
            })
    
    return test_videos


def get_all_videos(data_dir: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Get all videos from Celeb-DF-v2 dataset directories.
    
    Returns:
        Tuple of (celeb_real_videos, celeb_synthesis_videos, youtube_real_videos)
    """
    celeb_real_dir = data_dir / "Celeb-real"
    celeb_synthesis_dir = data_dir / "Celeb-synthesis"
    youtube_real_dir = data_dir / "YouTube-real"
    
    celeb_real_videos = []
    celeb_synthesis_videos = []
    youtube_real_videos = []
    
    # Get Celeb-real videos (label 0 = Real)
    if celeb_real_dir.exists():
        for video_path in sorted(celeb_real_dir.glob("*.mp4")):
            celeb_real_videos.append({
                "path": str(video_path),
                "label": 0  # Real
            })
    
    # Get Celeb-synthesis videos (label 1 = Fake)
    if celeb_synthesis_dir.exists():
        for video_path in sorted(celeb_synthesis_dir.glob("*.mp4")):
            celeb_synthesis_videos.append({
                "path": str(video_path),
                "label": 1  # Fake
            })
    
    # Get YouTube-real videos (label 0 = Real)
    if youtube_real_dir.exists():
        for video_path in sorted(youtube_real_dir.glob("*.mp4")):
            youtube_real_videos.append({
                "path": str(video_path),
                "label": 0  # Real
            })
    
    return celeb_real_videos, celeb_synthesis_videos, youtube_real_videos


def generate_splits(
    data_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Generate train/val/test splits for Celeb-DF-v2.
    
    The test split is determined by List_of_testing_videos.txt.
    The remaining videos are split into train and val sets.
    
    Args:
        data_dir: Root directory of Celeb-DF-v2 dataset
        val_ratio: Ratio of remaining videos to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_videos, val_videos, test_videos)
    """
    random.seed(seed)
    
    # Get all videos from directories
    celeb_real, celeb_synthesis, youtube_real = get_all_videos(data_dir)
    all_videos = celeb_real + celeb_synthesis + youtube_real
    
    print(f"Found {len(celeb_real)} Celeb-real videos")
    print(f"Found {len(celeb_synthesis)} Celeb-synthesis videos")
    print(f"Found {len(youtube_real)} YouTube-real videos")
    print(f"Total: {len(all_videos)} videos")
    
    # Get test videos from official list
    test_videos = parse_test_list(data_dir)
    test_paths = {v["path"] for v in test_videos}
    
    print(f"\nOfficial test split: {len(test_videos)} videos")
    
    # Get remaining videos for train/val split
    remaining_videos = [v for v in all_videos if v["path"] not in test_paths]
    
    print(f"Remaining for train/val: {len(remaining_videos)} videos")
    
    # Separate remaining by label for stratified split
    remaining_real = [v for v in remaining_videos if v["label"] == 0]
    remaining_fake = [v for v in remaining_videos if v["label"] == 1]
    
    print(f"  Remaining real: {len(remaining_real)}")
    print(f"  Remaining fake: {len(remaining_fake)}")
    
    # Shuffle for random split
    random.shuffle(remaining_real)
    random.shuffle(remaining_fake)
    
    # Split real videos
    val_real_count = int(len(remaining_real) * val_ratio)
    val_real = remaining_real[:val_real_count]
    train_real = remaining_real[val_real_count:]
    
    # Split fake videos
    val_fake_count = int(len(remaining_fake) * val_ratio)
    val_fake = remaining_fake[:val_fake_count]
    train_fake = remaining_fake[val_fake_count:]
    
    # Combine train and val sets
    train_videos = train_real + train_fake
    val_videos = val_real + val_fake
    
    # Shuffle final sets
    random.shuffle(train_videos)
    random.shuffle(val_videos)
    
    return train_videos, val_videos, test_videos


def save_splits(
    data_dir: Path,
    train_videos: List[Dict],
    val_videos: List[Dict],
    test_videos: List[Dict]
):
    """Save splits to JSON files."""
    # Save paths files (contains path and label)
    train_paths = data_dir / "train_paths.json"
    val_paths = data_dir / "val_paths.json"
    test_paths = data_dir / "test_paths.json"
    
    with open(train_paths, 'w') as f:
        json.dump(train_videos, f, indent=2)
    
    with open(val_paths, 'w') as f:
        json.dump(val_videos, f, indent=2)
    
    with open(test_paths, 'w') as f:
        json.dump(test_videos, f, indent=2)
    
    print(f"\nSaved splits to:")
    print(f"  {train_paths}")
    print(f"  {val_paths}")
    print(f"  {test_paths}")
    
    # Also save summary files (just the video IDs for reference)
    train_ids = data_dir / "train.json"
    val_ids = data_dir / "val.json"
    test_ids = data_dir / "test.json"
    
    train_summary = [{"video": Path(v["path"]).stem, "label": v["label"]} for v in train_videos]
    val_summary = [{"video": Path(v["path"]).stem, "label": v["label"]} for v in val_videos]
    test_summary = [{"video": Path(v["path"]).stem, "label": v["label"]} for v in test_videos]
    
    with open(train_ids, 'w') as f:
        json.dump(train_summary, f, indent=2)
    
    with open(val_ids, 'w') as f:
        json.dump(val_summary, f, indent=2)
    
    with open(test_ids, 'w') as f:
        json.dump(test_summary, f, indent=2)


def print_split_stats(
    train_videos: List[Dict],
    val_videos: List[Dict],
    test_videos: List[Dict]
):
    """Print statistics about the splits."""
    def count_labels(videos):
        real = sum(1 for v in videos if v["label"] == 0)
        fake = sum(1 for v in videos if v["label"] == 1)
        return real, fake
    
    train_real, train_fake = count_labels(train_videos)
    val_real, val_fake = count_labels(val_videos)
    test_real, test_fake = count_labels(test_videos)
    
    print("\n" + "=" * 50)
    print("Split Statistics")
    print("=" * 50)
    print(f"\nTrain set: {len(train_videos)} videos")
    print(f"  Real: {train_real} ({train_real/len(train_videos)*100:.1f}%)")
    print(f"  Fake: {train_fake} ({train_fake/len(train_videos)*100:.1f}%)")
    
    print(f"\nVal set: {len(val_videos)} videos")
    print(f"  Real: {val_real} ({val_real/len(val_videos)*100:.1f}%)")
    print(f"  Fake: {val_fake} ({val_fake/len(val_videos)*100:.1f}%)")
    
    print(f"\nTest set: {len(test_videos)} videos")
    print(f"  Real: {test_real} ({test_real/len(test_videos)*100:.1f}%)")
    print(f"  Fake: {test_fake} ({test_fake/len(test_videos)*100:.1f}%)")
    
    total = len(train_videos) + len(val_videos) + len(test_videos)
    print(f"\nTotal: {total} videos")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Generate train/val/test splits for Celeb-DF-v2 dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./Datasets/Celeb-DF-v2",
        help="Path to Celeb-DF-v2 dataset directory"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of non-test videos to use for validation (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    print(f"Generating splits for Celeb-DF-v2 at: {data_dir}")
    print(f"Validation ratio: {args.val_ratio}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Generate splits
    train_videos, val_videos, test_videos = generate_splits(
        data_dir,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    # Print statistics
    print_split_stats(train_videos, val_videos, test_videos)
    
    # Save splits
    save_splits(data_dir, train_videos, val_videos, test_videos)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
