"""
Probe cosine similarity between real and fake image activations at every ViT layer.

This script analyzes where the maximum distinction between real and fake faces
occurs in the DINOv2 Vision Transformer by computing cosine similarity of
activations at each intermediate layer.

Usage:
    python probe_layer_similarity.py --dataset celeb_df --split test --num_samples 100
    python probe_layer_similarity.py --dataset ff --split val --num_samples 50
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.celeb_df_dataset import CelebDFDataset
from data.ff_dataset import FFDataset
from preprocessing.transforms import get_val_transforms, TransformConfig


def load_dino_model(device: str = "cuda") -> torch.nn.Module:
    """Load frozen DINOv2 ViT-B/14 model."""
    print("Loading DINOv2 ViT-B/14 model...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model = model.to(device)
    model.eval()
    
    # Print model info
    num_layers = len(model.blocks)
    print(f"Model loaded with {num_layers} transformer blocks")
    
    return model


def get_all_layer_features(
    model: torch.nn.Module, 
    x: torch.Tensor
) -> List[torch.Tensor]:
    """
    Extract features from ALL intermediate layers of the ViT.
    
    Args:
        model: DINOv2 model
        x: Input images of shape (B, 3, H, W)
        
    Returns:
        List of tensors, one per layer. Each tensor has shape (B, num_patches, 768)
    """
    num_layers = len(model.blocks)  # 12 for ViT-B/14
    
    with torch.no_grad():
        # Get outputs from all intermediate layers
        outputs = model.get_intermediate_layers(
            x,
            n=num_layers,  # Get all layers
            reshape=False,
            return_class_token=False,
            norm=True
        )
    
    return list(outputs)


def create_dataset(
    dataset_type: str,
    split: str,
    root_dir: Path
) -> torch.utils.data.Dataset:
    """Create dataset with validation transforms."""
    transform_config = TransformConfig()
    transform = get_val_transforms(transform_config)
    
    if dataset_type == "celeb_df":
        dataset = CelebDFDataset(
            root_dir=root_dir,
            split=split,
            transform=transform,
            use_cache=True,
            require_cache=True,
            preload_cache=True
        )
    else:  # ff
        dataset = FFDataset(
            root_dir=root_dir,
            split=split,
            transform=transform,
            use_cache=True,
            require_cache=True,
            preload_cache=True
        )
    
    return dataset


def collect_samples_by_label(
    dataset: torch.utils.data.Dataset,
    num_samples: int,
    device: str
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Collect real and fake samples from the dataset.
    
    Returns:
        Tuple of (real_images, fake_images) lists
    """
    real_images = []
    fake_images = []
    
    print(f"Collecting up to {num_samples} samples per class...")
    
    for i in tqdm(range(len(dataset)), desc="Scanning dataset"):
        if len(real_images) >= num_samples and len(fake_images) >= num_samples:
            break
            
        image, label = dataset[i]  # Dataset returns (image, label) tuple
        
        if label == 0 and len(real_images) < num_samples:  # Real
            real_images.append(image)
        elif label == 1 and len(fake_images) < num_samples:  # Fake
            fake_images.append(image)
    
    print(f"Collected {len(real_images)} real and {len(fake_images)} fake samples")
    
    return real_images, fake_images


def compute_layer_similarities(
    model: torch.nn.Module,
    real_images: List[torch.Tensor],
    fake_images: List[torch.Tensor],
    device: str,
    batch_size: int = 16
) -> Dict[str, List[float]]:
    """
    Compute cosine similarity between real and fake activations at each layer.
    
    For each layer, we:
    1. Extract features for all real images and compute mean activation
    2. Extract features for all fake images and compute mean activation
    3. Compute cosine similarity between these mean activations
    
    We also compute per-sample similarities and report statistics.
    """
    num_layers = len(model.blocks)
    
    # Accumulate features per layer
    real_features_per_layer = [[] for _ in range(num_layers)]
    fake_features_per_layer = [[] for _ in range(num_layers)]
    
    # Process real images
    print("Processing real images...")
    for i in tqdm(range(0, len(real_images), batch_size), desc="Real batches"):
        batch = torch.stack(real_images[i:i+batch_size]).to(device)
        layer_features = get_all_layer_features(model, batch)
        
        for layer_idx, features in enumerate(layer_features):
            # Mean pool over patches: (B, num_patches, 768) -> (B, 768)
            pooled = features.mean(dim=1)
            real_features_per_layer[layer_idx].append(pooled.cpu())
    
    # Process fake images
    print("Processing fake images...")
    for i in tqdm(range(0, len(fake_images), batch_size), desc="Fake batches"):
        batch = torch.stack(fake_images[i:i+batch_size]).to(device)
        layer_features = get_all_layer_features(model, batch)
        
        for layer_idx, features in enumerate(layer_features):
            pooled = features.mean(dim=1)
            fake_features_per_layer[layer_idx].append(pooled.cpu())
    
    # Concatenate all batches
    for layer_idx in range(num_layers):
        real_features_per_layer[layer_idx] = torch.cat(real_features_per_layer[layer_idx], dim=0)
        fake_features_per_layer[layer_idx] = torch.cat(fake_features_per_layer[layer_idx], dim=0)
    
    # Compute statistics per layer
    results = {
        'layer': [],
        'mean_cos_sim': [],
        'std_cos_sim': [],
        'min_cos_sim': [],
        'max_cos_sim': [],
        'mean_real_vs_mean_fake': [],
        'distinction_score': []  # 1 - cos_sim (higher = more distinct)
    }
    
    print("\n" + "="*80)
    print("LAYER-BY-LAYER COSINE SIMILARITY ANALYSIS")
    print("="*80)
    print(f"{'Layer':<10} {'Mean Sim':<12} {'Std':<10} {'Min':<10} {'Max':<10} {'Distinction':<12}")
    print("-"*80)
    
    for layer_idx in range(num_layers):
        z_real = real_features_per_layer[layer_idx]  # (N_real, 768)
        z_fake = fake_features_per_layer[layer_idx]  # (N_fake, 768)
        
        # Method 1: Cosine similarity between mean embeddings
        mean_real = z_real.mean(dim=0)  # (768,)
        mean_fake = z_fake.mean(dim=0)  # (768,)
        mean_vs_mean_sim = F.cosine_similarity(
            mean_real.unsqueeze(0), 
            mean_fake.unsqueeze(0)
        ).item()
        
        # Method 2: Pairwise cosine similarities between all real and fake pairs
        # (For large datasets, sample a subset for efficiency)
        n_pairs = min(len(z_real), len(z_fake), 500)
        z_real_subset = z_real[:n_pairs]
        z_fake_subset = z_fake[:n_pairs]
        
        pairwise_sims = F.cosine_similarity(z_real_subset, z_fake_subset, dim=1)
        
        mean_sim = pairwise_sims.mean().item()
        std_sim = pairwise_sims.std().item()
        min_sim = pairwise_sims.min().item()
        max_sim = pairwise_sims.max().item()
        distinction = 1 - mean_sim
        
        results['layer'].append(layer_idx + 1)  # 1-indexed for readability
        results['mean_cos_sim'].append(mean_sim)
        results['std_cos_sim'].append(std_sim)
        results['min_cos_sim'].append(min_sim)
        results['max_cos_sim'].append(max_sim)
        results['mean_real_vs_mean_fake'].append(mean_vs_mean_sim)
        results['distinction_score'].append(distinction)
        
        print(f"Layer {layer_idx + 1:<4} {mean_sim:<12.4f} {std_sim:<10.4f} {min_sim:<10.4f} {max_sim:<10.4f} {distinction:<12.4f}")
    
    print("="*80)
    
    # Find layer with maximum distinction
    max_distinction_idx = results['distinction_score'].index(max(results['distinction_score']))
    min_similarity_idx = results['mean_cos_sim'].index(min(results['mean_cos_sim']))
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Layer with MAXIMUM DISTINCTION (lowest similarity): Layer {max_distinction_idx + 1}")
    print(f"  - Mean cosine similarity: {results['mean_cos_sim'][max_distinction_idx]:.4f}")
    print(f"  - Distinction score: {results['distinction_score'][max_distinction_idx]:.4f}")
    print()
    print(f"Layer with HIGHEST similarity: Layer {results['mean_cos_sim'].index(max(results['mean_cos_sim'])) + 1}")
    print(f"  - Mean cosine similarity: {max(results['mean_cos_sim']):.4f}")
    print("="*80)
    
    # Also compute within-class similarities for reference
    print("\n" + "="*80)
    print("WITHIN-CLASS SIMILARITY (for reference)")
    print("="*80)
    print(f"{'Layer':<10} {'Real-Real':<15} {'Fake-Fake':<15} {'Real-Fake':<15}")
    print("-"*80)
    
    for layer_idx in range(num_layers):
        z_real = real_features_per_layer[layer_idx]
        z_fake = fake_features_per_layer[layer_idx]
        
        # Sample pairs for within-class similarity
        n = min(len(z_real) // 2, len(z_fake) // 2, 250)
        
        # Real-Real similarity
        real_real_sim = F.cosine_similarity(z_real[:n], z_real[n:2*n], dim=1).mean().item()
        
        # Fake-Fake similarity  
        fake_fake_sim = F.cosine_similarity(z_fake[:n], z_fake[n:2*n], dim=1).mean().item()
        
        # Real-Fake similarity (already computed)
        real_fake_sim = results['mean_cos_sim'][layer_idx]
        
        print(f"Layer {layer_idx + 1:<4} {real_real_sim:<15.4f} {fake_fake_sim:<15.4f} {real_fake_sim:<15.4f}")
    
    print("="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Probe layer-wise cosine similarity between real and fake images")
    parser.add_argument("--dataset", type=str, default="celeb_df", choices=["celeb_df", "ff"],
                        help="Dataset to use")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples per class to analyze")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Dataset paths
    if args.dataset == "celeb_df":
        root_dir = Path("Datasets/Celeb-DF-v2")
    else:
        root_dir = Path("Datasets/FF")
    
    # Load model
    model = load_dino_model(device)
    
    # Create dataset
    dataset = create_dataset(args.dataset, args.split, root_dir)
    print(f"Dataset: {args.dataset}, Split: {args.split}, Total samples: {len(dataset)}")
    
    # Collect samples
    real_images, fake_images = collect_samples_by_label(dataset, args.num_samples, device)
    
    if len(real_images) == 0 or len(fake_images) == 0:
        print("ERROR: Could not collect enough real or fake samples!")
        return
    
    # Compute similarities
    results = compute_layer_similarities(model, real_images, fake_images, device, args.batch_size)
    
    # Additional analysis: L2 distance in feature space
    print("\n" + "="*80)
    print("L2 DISTANCE ANALYSIS (between mean embeddings)")
    print("="*80)
    
    num_layers = len(model.blocks)
    
    # Recompute for L2 distance
    real_features_per_layer = [[] for _ in range(num_layers)]
    fake_features_per_layer = [[] for _ in range(num_layers)]
    
    for i in range(0, len(real_images), args.batch_size):
        batch = torch.stack(real_images[i:i+args.batch_size]).to(device)
        layer_features = get_all_layer_features(model, batch)
        for layer_idx, features in enumerate(layer_features):
            pooled = features.mean(dim=1)
            real_features_per_layer[layer_idx].append(pooled.cpu())
    
    for i in range(0, len(fake_images), args.batch_size):
        batch = torch.stack(fake_images[i:i+args.batch_size]).to(device)
        layer_features = get_all_layer_features(model, batch)
        for layer_idx, features in enumerate(layer_features):
            pooled = features.mean(dim=1)
            fake_features_per_layer[layer_idx].append(pooled.cpu())
    
    print(f"{'Layer':<10} {'L2 Distance':<15} {'Normalized L2':<15}")
    print("-"*50)
    
    l2_distances = []
    for layer_idx in range(num_layers):
        z_real = torch.cat(real_features_per_layer[layer_idx], dim=0)
        z_fake = torch.cat(fake_features_per_layer[layer_idx], dim=0)
        
        mean_real = z_real.mean(dim=0)
        mean_fake = z_fake.mean(dim=0)
        
        l2_dist = torch.norm(mean_real - mean_fake).item()
        
        # Normalize by feature magnitude
        real_norm = torch.norm(mean_real).item()
        fake_norm = torch.norm(mean_fake).item()
        avg_norm = (real_norm + fake_norm) / 2
        normalized_l2 = l2_dist / avg_norm if avg_norm > 0 else 0
        
        l2_distances.append(l2_dist)
        print(f"Layer {layer_idx + 1:<4} {l2_dist:<15.4f} {normalized_l2:<15.4f}")
    
    max_l2_layer = l2_distances.index(max(l2_distances)) + 1
    print(f"\nLayer with maximum L2 distance: Layer {max_l2_layer} (L2 = {max(l2_distances):.4f})")
    print("="*80)


if __name__ == "__main__":
    main()
