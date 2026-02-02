"""
Autoencoder Trainer for Deepfake Detection.

This trainer handles:
1. Training the autoencoder on real images only
2. Computing intervention cost statistics
3. Finding optimal threshold for real/fake classification
4. Evaluating using AUROC and other metrics
"""

import time
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import asdict
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class AutoencoderTrainer:
    """
    Trainer for the autoencoder-based deepfake detector.
    
    Training process:
    1. Train autoencoder on real images only (minimize reconstruction loss)
    2. Compute intervention costs on validation set (real + fake)
    3. Find optimal threshold using ROC curve
    4. Evaluate on test set with AUROC
    """
    
    def __init__(self, config):
        """
        Initialize the trainer.
        
        Args:
            config: Config object containing training parameters
        """
        self.config = config
        self.training_config = config.training
        
        # Set device
        self.device = self._setup_device()
        
        # Initialize tracking variables
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_auroc = 0.0
        self.epochs_without_improvement = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'val_auroc': [],
            'val_threshold': []
        }
                
        # Create directories
        self.checkpoint_dir = self.training_config.checkpoint_dir / "autoencoder_detector"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.training_config.log_dir / "autoencoder_detector"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self._save_config()
        
        # Set random seed
        self._set_seed(config.seed)
    
    def _setup_device(self) -> torch.device:
        """Setup and return the training device."""
        device_str = self.training_config.device
        
        if device_str == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif device_str == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple MPS")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        
        return device
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def _save_config(self):
        """Save configuration to file."""
        config_path = self.log_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
    
    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer for autoencoder parameters only."""
        optimizer_name = self.training_config.optimizer.lower()
        lr = self.training_config.learning_rate
        weight_decay = self.training_config.weight_decay
        
        # Only optimize autoencoder and classifier parameters
        parameters = [
            {"params": model.autoencoder.parameters(), "lr": lr},
            {"params": model.classifier.parameters(), "lr": lr}
        ]
        
        if optimizer_name == "adam":
            return optim.Adam(parameters, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            return optim.AdamW(parameters, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            return optim.SGD(parameters, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self, optimizer: optim.Optimizer, num_training_steps: int):
        """Create learning rate scheduler."""
        scheduler_name = self.training_config.scheduler.lower()
        
        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps,
                eta_min=1e-7
            )
        elif scheduler_name == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.training_config.scheduler_patience,
                gamma=self.training_config.scheduler_factor
            )
        elif scheduler_name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=self.training_config.scheduler_patience,
                factor=self.training_config.scheduler_factor,
                verbose=True
            )
        elif scheduler_name == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def train(
        self,
        model: nn.Module,
        train_loader_real: DataLoader,
        val_loader_full: DataLoader,
        test_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the autoencoder on real images only.
        
        Args:
            model: AutoencoderDetector model
            train_loader_real: DataLoader with ONLY real images for AE training
            val_loader_full: DataLoader with BOTH real and fake images for threshold tuning
            test_loader: Optional test DataLoader for final evaluation
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Training results dictionary
        """
        # Move model to device
        model = model.to(self.device)
        
        # Create optimizer and scheduler
        optimizer = self._create_optimizer(model)
        num_training_steps = len(train_loader_real) * self.training_config.num_epochs
        scheduler = self._create_scheduler(optimizer, num_training_steps)
        
        # Resume from checkpoint if specified
        if resume_from:
            self._load_checkpoint(model, optimizer, scheduler, resume_from)
        
        print(f"\n{'='*60}")
        print("AUTOENCODER TRAINING (Real Images Only)")
        print(f"{'='*60}")
        print(f"Training samples (real only): {len(train_loader_real.dataset)}")
        print(f"Validation samples (real + fake): {len(val_loader_full.dataset)}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.training_config.num_epochs}")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.training_config.num_epochs):
            self.current_epoch = epoch
            
            # Phase 1: Train autoencoder on real images
            train_loss = self._train_epoch_autoencoder(
                model, train_loader_real, optimizer, scheduler
            )
            
            # Phase 2: Compute costs and find threshold on validation set
            val_metrics = self._evaluate_with_threshold(model, val_loader_full)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['val_loss_real'])
            self.history['val_auroc'].append(val_metrics['auroc'])
            self.history['val_threshold'].append(val_metrics['threshold'])
            self.history['learning_rates'].append(current_lr)
            
            # Update scheduler
            if scheduler is not None and self.training_config.scheduler.lower() in ["plateau", "step"]:
                scheduler.step(train_loss)
            
            # Print progress
            print(f"Epoch [{epoch+1}/{self.training_config.num_epochs}] "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val AUROC: {val_metrics['auroc']:.4f} | "
                  f"Threshold: {val_metrics['threshold']:.4f} | "
                  f"Acc: {val_metrics['accuracy']:.4f} | "
                  f"LR: {current_lr:.2e}")
            
            # Check for improvement (use AUROC as primary metric)
            improved = False
            if val_metrics['auroc'] > self.best_auroc:
                self.best_auroc = val_metrics['auroc']
                self.best_val_loss = train_loss
                improved = True
                self.epochs_without_improvement = 0
                
                # Update model threshold and statistics
                model.set_threshold(val_metrics['threshold'])
                model.set_normalization_stats(val_metrics['mean_cost'], val_metrics['std_cost'])
                
                # Save best checkpoint
                self._save_checkpoint(model, optimizer, scheduler, is_best=True, metrics=val_metrics)
                print(f"  ✓ New best model saved (AUROC: {val_metrics['auroc']:.4f})")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.training_config.early_stopping:
                if self.epochs_without_improvement >= self.training_config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
        
        total_time = time.time() - start_time
        
        # Final evaluation on test set if provided
        test_metrics = None
        if test_loader is not None:
            print("\n" + "="*60)
            print("FINAL TEST EVALUATION")
            print("="*60)
            # Load best checkpoint
            best_checkpoint = self.checkpoint_dir / "checkpoint_best_autoencoder.pt"
            if best_checkpoint.exists():
                self._load_checkpoint(model, optimizer, scheduler, str(best_checkpoint))
            test_metrics = self._evaluate_with_threshold(model, test_loader)
            self._print_detailed_metrics(test_metrics, "Test")
        
        # Save history
        self._save_history()
        
        print("\n" + "-" * 60)
        print(f"Training completed in {total_time/60:.2f} minutes")
        print(f"Best Validation AUROC: {self.best_auroc:.4f}")
        
        return {
            'history': self.history,
            'best_auroc': self.best_auroc,
            'best_val_loss': self.best_val_loss,
            'test_metrics': test_metrics,
            'total_time': total_time
        }
    
    def _train_epoch_autoencoder(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler
    ) -> float:
        """
        Train autoencoder for one epoch on real images only.
        
        Returns:
            Average reconstruction loss
        """
        model.train()
        # Keep backbone in eval mode (frozen)
        model.backbone.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training AE", file=sys.stdout, dynamic_ncols=True)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            
            # Verify all samples are real (label=0)
            assert (labels == 0).all(), "Training data should only contain real images!"
            
            optimizer.zero_grad()

            loss = model.get_autoencoder_loss(images)
            loss.backward()
            optimizer.step()
            
            # Update scheduler (for cosine annealing)
            if scheduler is not None and self.training_config.scheduler.lower() == "cosine":
                scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def _evaluate_with_threshold(
        self,
        model: nn.Module,
        dataloader: DataLoader
    ) -> Dict[str, Any]:
        """
        Evaluate model and find optimal threshold.
        
        Args:
            model: AutoencoderDetector model
            dataloader: DataLoader with both real and fake samples
            
        Returns:
            Dictionary with metrics and optimal threshold
        """
        model.eval()
        
        all_costs = []
        all_labels = []
        real_costs = []
        
        for images, labels in tqdm(dataloader, desc="Computing costs", file=sys.stdout, dynamic_ncols=True):
            images = images.to(self.device)
            
            # Extract features and compute intervention cost
            z = model.extract_features(images)
            costs = model.intervention_cost(z)
            
            all_costs.extend(costs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            # Track real costs separately for statistics
            real_mask = labels == 0
            if real_mask.any():
                real_costs.extend(costs[real_mask].cpu().numpy())
        
        all_costs = np.array(all_costs)
        all_labels = np.array(all_labels)
        real_costs = np.array(real_costs)
        
        # Compute statistics for normalization
        mean_cost = float(np.mean(real_costs))
        std_cost = float(np.std(real_costs))
        
        # Compute AUROC
        # Higher cost = more likely fake (label=1)
        auroc = roc_auc_score(all_labels, all_costs)
        
        # Find optimal threshold using ROC curve (Youden's J statistic)
        fpr, tpr, thresholds = roc_curve(all_labels, all_costs)
        j_scores = tpr - fpr
        j_optimal_idx = np.argmax(j_scores)
        j_optimal_threshold = thresholds[j_optimal_idx]
        
        # Find F1-maximizing threshold (best for imbalanced data)
        # F1 balances precision and recall, better than accuracy for imbalanced classes
        f1_scores = []
        for thresh in thresholds:
            preds = (all_costs > thresh).astype(int)
            f1_scores.append(f1_score(all_labels, preds, zero_division=0))
        f1_optimal_idx = np.argmax(f1_scores)
        f1_optimal_threshold = thresholds[f1_optimal_idx]
        
        # Use F1-maximizing threshold as the primary threshold
        # (better for imbalanced datasets like deepfake detection)
        optimal_threshold = f1_optimal_threshold
        
        # Compute predictions with optimal threshold
        predictions = (all_costs > optimal_threshold).astype(int)
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, predictions)
        precision = precision_score(all_labels, predictions, zero_division=0)
        recall = recall_score(all_labels, predictions, zero_division=0)
        f1 = f1_score(all_labels, predictions, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, predictions)
        
        # Compute reconstruction loss on real samples only
        val_loss_real = float(np.mean(real_costs ** 2))
        
        # Cost statistics by class
        fake_mask = all_labels == 1
        real_mean_cost = float(np.mean(all_costs[~fake_mask]))
        fake_mean_cost = float(np.mean(all_costs[fake_mask])) if fake_mask.any() else 0.0
        
        return {
            'auroc': auroc,
            'threshold': float(optimal_threshold),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist(),
            'val_loss_real': val_loss_real,
            'mean_cost': mean_cost,
            'std_cost': std_cost,
            'real_mean_cost': real_mean_cost,
            'fake_mean_cost': fake_mean_cost,
            'fpr_at_threshold': float(fpr[f1_optimal_idx]),
            'tpr_at_threshold': float(tpr[f1_optimal_idx]),
            'youden_j_threshold': float(j_optimal_threshold)
        }
    
    def _print_detailed_metrics(self, metrics: Dict[str, Any], split_name: str):
        """Print detailed metrics."""
        print(f"\n{split_name} Results:")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  Optimal Threshold: {metrics['threshold']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  TPR @ threshold: {metrics['tpr_at_threshold']:.4f}")
        print(f"  FPR @ threshold: {metrics['fpr_at_threshold']:.4f}")
        print(f"\n  Cost Statistics:")
        print(f"    Real mean cost: {metrics['real_mean_cost']:.4f}")
        print(f"    Fake mean cost: {metrics['fake_mean_cost']:.4f}")
        print(f"\n  Confusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"    TN: {cm[0][0]}, FP: {cm[0][1]}")
        print(f"    FN: {cm[1][0]}, TP: {cm[1][1]}")
    
    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        is_best: bool = False,
        metrics: Optional[Dict] = None
    ):
        """Save a checkpoint."""
        # Extract model configuration for future loading
        model_config = {}
        if hasattr(model, 'autoencoder'):
            if hasattr(model.autoencoder, 'bottleneck_dim'):
                model_config['bottleneck_dim'] = model.autoencoder.bottleneck_dim
            if hasattr(model.autoencoder, 'input_dim'):
                model_config['input_dim'] = model.autoencoder.input_dim
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model.state_dict(),
            'model_config': model_config,  # Save model architecture config
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_auroc': self.best_auroc,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'threshold': float(model.threshold),
            'mean_cost': float(model.mean_cost),
            'std_cost': float(model.std_cost),
            'metrics': metrics
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / "checkpoint_latest_autoencoder.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best_autoencoder.pt"
            torch.save(checkpoint, best_path)
    
    def _load_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        checkpoint_path: str
    ):
        """Load a checkpoint."""
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_auroc = checkpoint.get('best_auroc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)
        
        # Restore threshold and normalization stats
        if 'threshold' in checkpoint:
            model.set_threshold(checkpoint['threshold'])
        if 'mean_cost' in checkpoint and 'std_cost' in checkpoint:
            model.set_normalization_stats(checkpoint['mean_cost'], checkpoint['std_cost'])
    
    def _save_history(self):
        """Save training history."""
        history_path = self.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
