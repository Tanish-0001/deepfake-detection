"""
Model Evaluator for Deepfake Detection.

Evaluates trained models on specified datasets and reports comprehensive metrics.

Usage:
    python -m training.evaluator --model simple_cnn --weights checkpoints/deepfake_detection/checkpoint_best.pt --dataset ff
    python -m training.evaluator --model simple_cnn --weights checkpoints/deepfake_detection/checkpoint_best.pt --dataset celeb_df
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_auc_score,
        classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from tqdm import tqdm
from data import get_dataloaders


class Evaluator:
    """
    Evaluator class for deepfake detection models.
    
    Handles model loading, dataset preparation, inference, and metric computation.
    """
    
    def __init__(
        self,
        model_name: str,
        weights_path: str,
        dataset_name: str = "ff",
        dataloader: DataLoader = None,
        batch_size: int = 32,
        num_workers: int = 0,
        device: Optional[str] = None,
        video_level: bool = False,
        config = None,
        model_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_name: Name of the model architecture ('simple_cnn', 'simple_cnn_large')
            weights_path: Path to the checkpoint file containing model weights
            dataset_name: Name of the dataset to evaluate on ('ff', 'celeb_df')
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            video_level: Whether to use video-level evaluation
            config: Configuration object
            model_kwargs: Additional keyword arguments for model creation (e.g., max_seq_length)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for evaluation")
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for metrics computation")
        
        self.model_name = model_name
        self.weights_path = Path(weights_path)
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.video_level = video_level
        self.model_kwargs = model_kwargs or {}

        self.config = config
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model and weights
        self.model = self._load_model()
        
        # Create dataset and dataloader
        if dataloader is not None:
            self.dataloader = dataloader
        else:
            self.dataloader = self._create_dataloader()
    
    def _load_model(self) -> nn.Module:
        """
        Load the model architecture and weights.
        
        Returns:
            Loaded model in evaluation mode
        """
        from models import get_model
        
        # Create model with kwargs
        print(f"Loading model: {self.model_name}")
        if self.model_kwargs:
            print(f"Model kwargs: {self.model_kwargs}")
        model = get_model(self.model_name, **self.model_kwargs)
        
        # Load weights
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
        
        print(f"Loading weights from: {self.weights_path}")
        # Note: weights_only=False is needed because checkpoints contain Path objects
        # in metadata (config, checkpoint_dir, etc.). This is safe for self-created checkpoints.
        checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume the checkpoint is the state dict itself
            state_dict = checkpoint
        
        # Load state dict
        model.load_state_dict(state_dict)
        
        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        
        # Print checkpoint info if available
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                print(f"Checkpoint epoch: {checkpoint['epoch']}")
            if 'best_val_acc' in checkpoint:
                print(f"Checkpoint best val accuracy: {checkpoint['best_val_acc']:.4f}")
        
        return model
    
    def _create_dataloader(self) -> DataLoader:
        """
        Create the test dataloader for the specified dataset.
        
        Returns:
            DataLoader for the test set
        """

        test_loader = get_dataloaders(
            dataset_type=self.dataset_name,
            dataset_configs=self.config.data.get_enabled_dataset_configs(),
            batch_size=self.config.data.batch_size,
            num_workers=self.num_workers,
            test_only=True
        )[0]

        return test_loader
    
    def _create_ff_dataset(self, transform):
        """Create FaceForensics++ dataset."""
        from data.ff_dataset import FFDataset, FFVideoDataset
        from preprocessing.pipeline import PreprocessingPipeline
        
        root_dir = Path("Datasets/FF")
        
        if not root_dir.exists():
            raise FileNotFoundError(f"FF dataset not found at: {root_dir}")
        
        pipeline = PreprocessingPipeline(
            num_frames=10,
            sampling_strategy="uniform"
        )
        
        DatasetClass = FFVideoDataset if self.video_level else FFDataset
        
        return DatasetClass(
            root_dir=root_dir,
            split="test",
            frames_per_video=10,
            transform=transform,
            preprocessing_pipeline=pipeline
        )
    
    def _create_celeb_df_dataset(self, transform):
        """Create Celeb-DF-v2 dataset."""
        from data.celeb_df_dataset import CelebDFDataset, CelebDFVideoDataset
        from preprocessing.pipeline import PreprocessingPipeline
        
        root_dir = Path("Datasets/Celeb-DF-v2")
        
        if not root_dir.exists():
            raise FileNotFoundError(f"Celeb-DF-v2 dataset not found at: {root_dir}")
        
        pipeline = PreprocessingPipeline(
            num_frames=10,
            sampling_strategy="uniform"
        )
        
        DatasetClass = CelebDFVideoDataset if self.video_level else CelebDFDataset
        
        return DatasetClass(
            root_dir=root_dir,
            split="test",
            frames_per_video=10,
            transform=transform,
            preprocessing_pipeline=pipeline
        )
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """
        Run evaluation and compute metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        print("\nRunning evaluation...")
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        for batch_idx, (inputs, labels) in enumerate(tqdm(self.dataloader, desc="Evaluating")):
            # Move to device
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Handle video-level input: [B, T, C, H, W] -> [B*T, C, H, W]
            if self.video_level and inputs.dim() == 5:
                batch_size, num_frames = inputs.shape[:2]
                inputs = inputs.view(-1, *inputs.shape[2:])
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Average predictions across frames
                if outputs.dim() == 2:
                    outputs = outputs.view(batch_size, num_frames, -1).mean(dim=1)
            else:
                # Forward pass
                outputs = self.model(inputs)
            
            # Get probabilities (softmax for multi-class or sigmoid for binary)
            if outputs.shape[-1] == 2:
                probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of fake
            else:
                probs = torch.sigmoid(outputs).squeeze()
            
            # Get predictions
            if outputs.shape[-1] == 2:
                preds = outputs.argmax(dim=1)
            else:
                preds = (probs > 0.5).long()
            
            # Collect results
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        
        # Compute metrics
        metrics = self._compute_metrics(y_true, y_pred, y_prob)
        
        return metrics
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities for the positive class (fake)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics['f1_score'] = float(f1_score(y_true, y_pred, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Extract confusion matrix values
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            # Specificity (True Negative Rate)
            metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        # AUC-ROC (handle edge cases)
        try:
            if len(np.unique(y_true)) > 1:
                metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob))
            else:
                metrics['auc_roc'] = None
                print("Warning: Only one class present in y_true. AUC-ROC is not defined.")
        except Exception as e:
            metrics['auc_roc'] = None
            print(f"Warning: Could not compute AUC-ROC: {e}")
        
        # Sample counts
        metrics['total_samples'] = len(y_true)
        metrics['real_samples'] = int(np.sum(y_true == 0))
        metrics['fake_samples'] = int(np.sum(y_true == 1))
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred,
            target_names=['Real (0)', 'Fake (1)'],
            zero_division=0,
            output_dict=True
        )
        
        return metrics
    
    def print_results(self, metrics: Dict[str, Any]) -> None:
        """
        Print evaluation results in a formatted manner.
        
        Args:
            metrics: Dictionary of computed metrics
        """
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        print(f"\nModel: {self.model_name}")
        print(f"Weights: {self.weights_path}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"  - Real: {metrics['real_samples']}")
        print(f"  - Fake: {metrics['fake_samples']}")
        
        print("\n" + "-" * 40)
        print("METRICS")
        print("-" * 40)
        
        print(f"Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1 Score:    {metrics['f1_score']:.4f}")
        if metrics.get('specificity') is not None:
            print(f"Specificity: {metrics['specificity']:.4f}")
        if metrics.get('auc_roc') is not None:
            print(f"AUC-ROC:     {metrics['auc_roc']:.4f}")
        else:
            print("AUC-ROC:     N/A")
        
        print("\n" + "-" * 40)
        print("CONFUSION MATRIX")
        print("-" * 40)
        print("                 Predicted")
        print("              Real    Fake")
        cm = metrics['confusion_matrix']
        if len(cm) == 2:
            print(f"Actual Real  {cm[0][0]:6d}  {cm[0][1]:6d}")
            print(f"Actual Fake  {cm[1][0]:6d}  {cm[1][1]:6d}")
        
        print("\n" + "-" * 40)
        print("CLASSIFICATION REPORT")
        print("-" * 40)
        report = metrics['classification_report']
        print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-" * 52)
        for class_name in ['Real (0)', 'Fake (1)']:
            if class_name in report:
                cls = report[class_name]
                print(f"{class_name:<12} {cls['precision']:>10.4f} {cls['recall']:>10.4f} "
                      f"{cls['f1-score']:>10.4f} {int(cls['support']):>10}")
        print("-" * 52)
        if 'weighted avg' in report:
            avg = report['weighted avg']
            print(f"{'Weighted Avg':<12} {avg['precision']:>10.4f} {avg['recall']:>10.4f} "
                  f"{avg['f1-score']:>10.4f} {int(avg['support']):>10}")
        
        print("\n" + "=" * 60)
    
    def save_results(self, metrics: Dict[str, Any], output_path: str) -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            metrics: Dictionary of computed metrics
            output_path: Path to save the results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        results = {
            'model_name': self.model_name,
            'weights_path': str(self.weights_path),
            'dataset_name': self.dataset_name,
            'device': str(self.device),
            'video_level': self.video_level,
            'metrics': metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate deepfake detection models on various datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        choices=['simple_cnn', 'simple_cnn_large'],
        help="Model architecture to use"
    )
    
    parser.add_argument(
        '--weights', '-w',
        type=str,
        default='checkpoint_best.pt',
        help="Path to the checkpoint file containing model weights"
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=['ff', 'celeb_df', 'celebdf'],
        default='ff',
        help="Dataset to evaluate on"
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        '--num-workers', '-j',
        type=int,
        default=0,
        help="Number of data loading workers"
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help="Device to use for evaluation (default: auto-detect)"
    )
    
    parser.add_argument(
        '--video-level',
        action='store_true',
        help="Use video-level evaluation (average predictions across frames)"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help="Path to save results JSON file (optional)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("DEEPFAKE DETECTION MODEL EVALUATOR")
    print("=" * 60)

    checkpoint_path = "checkpoints/deepfake_detection/" + args.weights
    
    # Create evaluator
    evaluator = Evaluator(
        model_name=args.model,
        weights_path=checkpoint_path,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        video_level=args.video_level
    )
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Print results
    evaluator.print_results(metrics)
    
    # Save results if output path provided
    if args.output:
        evaluator.save_results(metrics, args.output)
    
    return metrics


if __name__ == '__main__':
    main()
