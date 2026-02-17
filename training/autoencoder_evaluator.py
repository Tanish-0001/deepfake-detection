"""
Autoencoder-based Deepfake Detector Evaluator.

Evaluates the autoencoder detector using intervention cost
and computes AUROC and other metrics.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class AutoencoderEvaluator:
    """
    Evaluator for the autoencoder-based deepfake detector.
    
    Uses intervention cost (reconstruction error) to distinguish
    real from fake images.
    """
    
    def __init__(
        self,
        weights_path: str,
        dataset_name: str = "ff",
        dataloader: DataLoader = None,
        batch_size: int = 32,
        num_workers: int = 0,
        device: Optional[str] = None,
        config=None,
        model_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            weights_path: Path to the checkpoint file
            dataset_name: Name of the dataset to evaluate on ('ff', 'celeb_df')
            dataloader: Optional pre-created dataloader
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            config: Configuration object
            model_kwargs: Additional model kwargs
        """
        self.weights_path = Path(weights_path)
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.config = config
        self.model_kwargs = model_kwargs or {}
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
        # Create or use provided dataloader
        if dataloader is not None:
            self.dataloader = dataloader
        else:
            self.dataloader = self._create_dataloader()
    
    def _infer_model_config_from_checkpoint(self, state_dict: Dict) -> Dict[str, Any]:
        """
        Infer model configuration from checkpoint state_dict.
        
        This allows loading checkpoints even when the model config wasn't saved.
        """
        config = {}
        
        # Infer bottleneck_dim from the encoder's last layer
        # The last layer in encoder is autoencoder.encoder.{last_idx}.weight
        encoder_keys = [k for k in state_dict.keys() if k.startswith('autoencoder.encoder.') and k.endswith('.weight')]
        if encoder_keys:
            # Find the highest index
            last_encoder_layer = max(encoder_keys, key=lambda x: int(x.split('.')[2]))
            # Shape is [bottleneck_dim, hidden_dim] for the last encoder layer
            bottleneck_dim = state_dict[last_encoder_layer].shape[0]
            config['bottleneck_dim'] = bottleneck_dim
            print(f"Inferred bottleneck_dim: {bottleneck_dim}")
        
        # Infer hidden_dims from encoder layers
        # Encoder layers alternate: Linear, LayerNorm, GELU, Dropout
        # So we look for Linear layers (indices 0, 4, 8, ...)
        hidden_dims = []
        for key in sorted(encoder_keys)[:-1]:  # Exclude last layer (bottleneck)
            idx = int(key.split('.')[2])
            if idx % 4 == 0:  # Linear layers at indices 0, 4, 8, etc.
                hidden_dim = state_dict[key].shape[0]
                hidden_dims.append(hidden_dim)
        
        if hidden_dims:
            config['hidden_dims'] = hidden_dims
            print(f"Inferred hidden_dims: {hidden_dims}")
        
        return config
    
    def _load_model(self) -> nn.Module:
        """Load the autoencoder detector model."""
        from models.autoencoder_detector import AutoencoderDetector
        
        # Load weights
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
        
        print(f"Loading weights from: {self.weights_path}")
        checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Get model config from checkpoint if available, otherwise infer
        if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
            print("Using model config from checkpoint...")
            inferred_config = checkpoint['model_config']
        elif not self.model_kwargs or 'bottleneck_dim' not in self.model_kwargs:
            print("Model config not provided, inferring from checkpoint state_dict...")
            inferred_config = self._infer_model_config_from_checkpoint(state_dict)
        else:
            inferred_config = {}
        
        # Merge with provided kwargs (provided kwargs take precedence)
        model_kwargs = {**inferred_config, **self.model_kwargs}
        
        # Remove keys that AutoencoderDetector doesn't accept
        # (input_dim is stored in checkpoint config but is hardcoded in the model)
        keys_to_remove = ['input_dim']
        for key in keys_to_remove:
            model_kwargs.pop(key, None)
        
        print(f"Loading AutoencoderDetector model")
        model = AutoencoderDetector(**model_kwargs)
        
        model.load_state_dict(state_dict)
        
        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        
        # Print checkpoint info
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                print(f"Checkpoint epoch: {checkpoint['epoch']}")
            if 'best_auroc' in checkpoint:
                print(f"Checkpoint best AUROC: {checkpoint['best_auroc']:.4f}")
            if 'threshold' in checkpoint:
                print(f"Checkpoint threshold: {checkpoint['threshold']:.4f}")
        
        return model
    
    def _create_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        from data import get_dataloaders
        
        test_loader = get_dataloaders(
            dataset_type=self.dataset_name,
            dataset_configs=self.config.data.get_enabled_dataset_configs() if self.config else None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            test_only=True
        )[0]
        
        return test_loader
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """
        Run evaluation using intervention cost.
        
        Returns:
            Dictionary with all computed metrics
        """
        print("\nRunning evaluation with intervention cost...")
        
        all_costs = []
        all_labels = []
        all_patch_embeddings = []  # (B, num_patches, D) per batch
        all_per_patch_costs = []   # (B, num_patches) per batch
        
        for images, labels in tqdm(self.dataloader, desc="Evaluating"):
            images = images.to(self.device)
            
            # Extract features - now returns (B, num_patches, 768)
            z = self.model.extract_features(images)
            
            # Compute aggregated intervention cost - returns (B,)
            costs = self.model.intervention_cost(z)
            
            # Compute per-patch costs for analysis - returns (B, num_patches)
            patch_costs = self.model.intervention_cost_per_patch(z)
            
            all_costs.extend(costs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_patch_embeddings.append(z.cpu().numpy())
            all_per_patch_costs.append(patch_costs.cpu().numpy())
        
        all_costs = np.array(all_costs)
        all_labels = np.array(all_labels)
        # Stack patch embeddings: (total_samples, num_patches, D)
        all_patch_embeddings = np.vstack(all_patch_embeddings)
        all_per_patch_costs = np.vstack(all_per_patch_costs)
        
        # Compute metrics
        metrics = self._compute_metrics(all_costs, all_labels)
        
        # Add embedding statistics (uses patch-level embeddings)
        metrics['embedding_stats'] = self._compute_embedding_stats(
            all_patch_embeddings, all_labels
        )
        
        # Add per-patch cost statistics
        metrics['patch_cost_stats'] = self._compute_patch_cost_stats(
            all_per_patch_costs, all_labels
        )
        
        return metrics
    
    def _compute_metrics(
        self,
        costs: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.
        
        Args:
            costs: Intervention costs for all samples
            labels: Ground truth labels (0=real, 1=fake)
            
        Returns:
            Dictionary with metrics
        """
        # AUROC - higher cost means more likely fake
        auroc = roc_auc_score(labels, costs)
        
        # Average Precision (AUPRC)
        ap = average_precision_score(labels, costs)
        
        # ROC curve for threshold analysis
        fpr, tpr, thresholds = roc_curve(labels, costs)
        
        # Find Youden's J optimal threshold (balances TPR and FPR equally)
        j_scores = tpr - fpr
        j_optimal_idx = np.argmax(j_scores)
        j_optimal_threshold = thresholds[j_optimal_idx]
        
        # Find F1-maximizing threshold (best for imbalanced data)
        # F1 balances precision and recall, better than accuracy for imbalanced classes
        f1_scores = []
        for thresh in thresholds:
            preds = (costs > thresh).astype(int)
            f1_scores.append(f1_score(labels, preds, zero_division=0))
        f1_optimal_idx = np.argmax(f1_scores)
        f1_optimal_threshold = thresholds[f1_optimal_idx]
        
        # Use F1-maximizing as the primary "optimal" threshold
        optimal_threshold = f1_optimal_threshold
        
        # Also try using the model's learned threshold
        model_threshold = float(self.model.threshold.cpu())
        
        # Compute metrics with optimal threshold (accuracy-maximizing)
        predictions_optimal = (costs > optimal_threshold).astype(int)
        
        # Compute metrics with Youden's J threshold
        predictions_youden = (costs > j_optimal_threshold).astype(int)
        
        # Compute metrics with model threshold
        predictions_model = (costs > model_threshold).astype(int)
        
        # Metrics with optimal threshold
        metrics_optimal = {
            'accuracy': accuracy_score(labels, predictions_optimal),
            'precision': precision_score(labels, predictions_optimal, zero_division=0),
            'recall': recall_score(labels, predictions_optimal, zero_division=0),
            'f1': f1_score(labels, predictions_optimal, zero_division=0),
            'confusion_matrix': confusion_matrix(labels, predictions_optimal).tolist()
        }
        
        # Metrics with model threshold
        metrics_model = {
            'accuracy': accuracy_score(labels, predictions_model),
            'precision': precision_score(labels, predictions_model, zero_division=0),
            'recall': recall_score(labels, predictions_model, zero_division=0),
            'f1': f1_score(labels, predictions_model, zero_division=0),
            'confusion_matrix': confusion_matrix(labels, predictions_model).tolist()
        }
        
        # Cost statistics by class
        real_mask = labels == 0
        fake_mask = labels == 1
        
        cost_stats = {
            'real_mean': float(np.mean(costs[real_mask])),
            'real_std': float(np.std(costs[real_mask])),
            'real_min': float(np.min(costs[real_mask])),
            'real_max': float(np.max(costs[real_mask])),
            'fake_mean': float(np.mean(costs[fake_mask])),
            'fake_std': float(np.std(costs[fake_mask])),
            'fake_min': float(np.min(costs[fake_mask])),
            'fake_max': float(np.max(costs[fake_mask])),
            'separation': float(np.mean(costs[fake_mask]) - np.mean(costs[real_mask]))
        }
        
        # Find thresholds at specific FPR values
        target_fprs = [0.01, 0.05, 0.10]
        thresholds_at_fpr = {}
        tpr_at_fpr = {}
        
        for target_fpr in target_fprs:
            idx = np.argmin(np.abs(fpr - target_fpr))
            thresholds_at_fpr[f'threshold_at_fpr_{int(target_fpr*100)}'] = float(thresholds[idx])
            tpr_at_fpr[f'tpr_at_fpr_{int(target_fpr*100)}'] = float(tpr[idx])
        
        return {
            'auroc': auroc,
            'average_precision': ap,
            'optimal_threshold': float(optimal_threshold),
            'youden_j_threshold': float(j_optimal_threshold),
            'model_threshold': model_threshold,
            'metrics_optimal_threshold': metrics_optimal,
            'metrics_model_threshold': metrics_model,
            'metrics_youden_threshold': {
                'accuracy': accuracy_score(labels, predictions_youden),
                'precision': precision_score(labels, predictions_youden, zero_division=0),
                'recall': recall_score(labels, predictions_youden, zero_division=0),
                'f1': f1_score(labels, predictions_youden, zero_division=0),
                'confusion_matrix': confusion_matrix(labels, predictions_youden).tolist()
            },
            'cost_statistics': cost_stats,
            **thresholds_at_fpr,
            **tpr_at_fpr,
            'fpr_at_optimal': float(fpr[f1_optimal_idx]),
            'tpr_at_optimal': float(tpr[f1_optimal_idx]),
            'num_real': int(np.sum(real_mask)),
            'num_fake': int(np.sum(fake_mask))
        }
    
    def _compute_embedding_stats(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute statistics on the patch embeddings.
        
        Args:
            embeddings: Patch embeddings of shape (N, num_patches, D)
            labels: Labels of shape (N,)
        """
        real_mask = labels == 0
        fake_mask = labels == 1
        
        real_embeddings = embeddings[real_mask]  # (N_real, num_patches, D)
        fake_embeddings = embeddings[fake_mask]  # (N_fake, num_patches, D)
        
        # Compute mean embedding per sample by averaging across patches
        # Then compute class-level statistics
        real_mean_per_sample = np.mean(real_embeddings, axis=1)  # (N_real, D)
        fake_mean_per_sample = np.mean(fake_embeddings, axis=1)  # (N_fake, D)
        
        # Mean embeddings across all samples
        real_mean = np.mean(real_mean_per_sample, axis=0)  # (D,)
        fake_mean = np.mean(fake_mean_per_sample, axis=0)  # (D,)
        
        # Distance between class means
        mean_distance = np.linalg.norm(real_mean - fake_mean)
        
        # Average within-class variance (across sample means)
        real_var = np.mean(np.var(real_mean_per_sample, axis=0))
        fake_var = np.mean(np.var(fake_mean_per_sample, axis=0))
        
        # Also compute patch-level variance (how much patches vary within images)
        real_patch_var = np.mean(np.var(real_embeddings, axis=1))  # Variance across patches
        fake_patch_var = np.mean(np.var(fake_embeddings, axis=1))
        
        return {
            'mean_distance_between_classes': float(mean_distance),
            'real_embedding_variance': float(real_var),
            'fake_embedding_variance': float(fake_var),
            'real_within_image_patch_variance': float(real_patch_var),
            'fake_within_image_patch_variance': float(fake_patch_var),
            'num_patches': int(embeddings.shape[1]),
            'embedding_dim': int(embeddings.shape[2])
        }
    
    def _compute_patch_cost_stats(
        self,
        patch_costs: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute statistics on per-patch intervention costs.
        
        This reveals how costs are distributed across patches,
        which can indicate localized manipulation regions.
        
        Args:
            patch_costs: Per-patch costs of shape (N, num_patches)
            labels: Labels of shape (N,)
            
        Returns:
            Dictionary with patch-level cost statistics
        """
        real_mask = labels == 0
        fake_mask = labels == 1
        
        real_patch_costs = patch_costs[real_mask]  # (N_real, num_patches)
        fake_patch_costs = patch_costs[fake_mask]  # (N_fake, num_patches)
        
        # Statistics on within-image patch cost distribution
        # Variance across patches within each image (high variance = localized anomaly)
        real_within_var = np.mean(np.var(real_patch_costs, axis=1))
        fake_within_var = np.mean(np.var(fake_patch_costs, axis=1))
        
        # Max patch cost per image (useful for detecting localized fakes)
        real_max_patch = np.mean(np.max(real_patch_costs, axis=1))
        fake_max_patch = np.mean(np.max(fake_patch_costs, axis=1))
        
        # Min patch cost per image
        real_min_patch = np.mean(np.min(real_patch_costs, axis=1))
        fake_min_patch = np.mean(np.min(fake_patch_costs, axis=1))
        
        # Ratio of max to mean (high ratio = concentrated anomaly)
        real_max_mean_ratio = np.mean(
            np.max(real_patch_costs, axis=1) / (np.mean(real_patch_costs, axis=1) + 1e-8)
        )
        fake_max_mean_ratio = np.mean(
            np.max(fake_patch_costs, axis=1) / (np.mean(fake_patch_costs, axis=1) + 1e-8)
        )
        
        # Average cost per patch position (which patches have highest cost on average)
        avg_cost_per_patch_real = np.mean(real_patch_costs, axis=0)  # (num_patches,)
        avg_cost_per_patch_fake = np.mean(fake_patch_costs, axis=0)
        
        return {
            'real_within_image_cost_variance': float(real_within_var),
            'fake_within_image_cost_variance': float(fake_within_var),
            'real_max_patch_cost': float(real_max_patch),
            'fake_max_patch_cost': float(fake_max_patch),
            'real_min_patch_cost': float(real_min_patch),
            'fake_min_patch_cost': float(fake_min_patch),
            'real_max_mean_ratio': float(real_max_mean_ratio),
            'fake_max_mean_ratio': float(fake_max_mean_ratio),
            # Top-5 highest cost patch indices for fake images
            'fake_highest_cost_patches': np.argsort(avg_cost_per_patch_fake)[-5:][::-1].tolist(),
            'real_highest_cost_patches': np.argsort(avg_cost_per_patch_real)[-5:][::-1].tolist(),
        }
    
    def print_results(self, metrics: Dict[str, Any]):
        """Print evaluation results in a formatted way."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n📊 Main Metrics:")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  Average Precision: {metrics['average_precision']:.4f}")
        
        print(f"\n🎯 Threshold Analysis:")
        print(f"  Optimal threshold: {metrics['optimal_threshold']:.4f}")
        print(f"  Model threshold: {metrics['model_threshold']:.4f}")
        
        print(f"\n📈 Performance with Optimal Threshold:")
        opt_metrics = metrics['metrics_optimal_threshold']
        print(f"  Accuracy: {opt_metrics['accuracy']:.4f}")
        print(f"  Precision: {opt_metrics['precision']:.4f}")
        print(f"  Recall: {opt_metrics['recall']:.4f}")
        print(f"  F1 Score: {opt_metrics['f1']:.4f}")
        cm = opt_metrics['confusion_matrix']
        print(f"  Confusion Matrix:")
        print(f"    TN: {cm[0][0]:5d}  FP: {cm[0][1]:5d}")
        print(f"    FN: {cm[1][0]:5d}  TP: {cm[1][1]:5d}")
        
        print(f"\n📉 Intervention Cost Statistics:")
        stats = metrics['cost_statistics']
        print(f"  Real images:  mean={stats['real_mean']:.4f}, std={stats['real_std']:.4f}")
        print(f"  Fake images:  mean={stats['fake_mean']:.4f}, std={stats['fake_std']:.4f}")
        print(f"  Separation (fake_mean - real_mean): {stats['separation']:.4f}")
        
        print(f"\n🔍 TPR at Fixed FPR:")
        if 'tpr_at_fpr_1' in metrics:
            print(f"  TPR @ 1% FPR: {metrics['tpr_at_fpr_1']:.4f}")
        if 'tpr_at_fpr_5' in metrics:
            print(f"  TPR @ 5% FPR: {metrics['tpr_at_fpr_5']:.4f}")
        if 'tpr_at_fpr_10' in metrics:
            print(f"  TPR @ 10% FPR: {metrics['tpr_at_fpr_10']:.4f}")
        
        print(f"\n📊 Dataset Info:")
        print(f"  Real samples: {metrics['num_real']}")
        print(f"  Fake samples: {metrics['num_fake']}")
        
        # Print patch-level statistics if available
        if 'patch_cost_stats' in metrics:
            print(f"\n🧩 Patch-Level Cost Analysis:")
            pstats = metrics['patch_cost_stats']
            print(f"  Real within-image cost variance: {pstats['real_within_image_cost_variance']:.4f}")
            print(f"  Fake within-image cost variance: {pstats['fake_within_image_cost_variance']:.4f}")
            print(f"  Real max patch cost (avg): {pstats['real_max_patch_cost']:.4f}")
            print(f"  Fake max patch cost (avg): {pstats['fake_max_patch_cost']:.4f}")
            print(f"  Fake max/mean ratio: {pstats['fake_max_mean_ratio']:.2f}")
        
        if 'embedding_stats' in metrics:
            print(f"\n🔢 Embedding Info:")
            estats = metrics['embedding_stats']
            print(f"  Num patches: {estats.get('num_patches', 'N/A')}")
            print(f"  Embedding dim: {estats.get('embedding_dim', 'N/A')}")
            print(f"  Class mean distance: {estats['mean_distance_between_classes']:.4f}")
        
        print("="*60)


def main():
    """Main function for command-line evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Autoencoder Deepfake Detector")
    
    parser.add_argument(
        "--weights",
        type=str,
        default="checkpoints/autoencoder_detector/checkpoint_best_autoencoder.pt",
        help="Path to model weights"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ff",
        choices=["ff", "celeb_df"],
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = AutoencoderEvaluator(
        weights_path=args.weights,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Print results
    evaluator.print_results(metrics)
    
    # Save results if output path specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
