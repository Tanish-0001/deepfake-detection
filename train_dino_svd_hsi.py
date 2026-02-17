#!/usr/bin/env python3
"""
Train DINO SVD + HSI Model for Deepfake Detection.

This script trains a combined model that uses:
- DINO SVD backbone: DINO with SVD residual learning where principal components
  are frozen and residual components are trainable
- HSI Encoder: RGB to Hyperspectral conversion using MST++ followed by feature extraction

Usage:
    # Train on single dataset
    python train_dino_svd_hsi.py --dataset ff
    python train_dino_svd_hsi.py --dataset celeb_df
    
    # Train on combined datasets
    python train_dino_svd_hsi.py --dataset combined
    
    # With custom SVD rank
    python train_dino_svd_hsi.py --dataset combined --svd_rank 767
    
    # With additional loss terms
    python train_dino_svd_hsi.py --dataset combined --use_orthogonal_loss --use_keepsv_loss
"""

import argparse
import json
import time
import sys
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any, Optional, List
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import Config, DataConfig, TrainingConfig, PreprocessingConfig, DatasetSourceConfig
from models.DinoSVD_HSI import DinoSVD_with_HSI_Model
from models.dino_svd_model import SVDResidualLinear
from data import get_dataloaders
from data.dataloader import create_ff_dataloaders, create_celeb_df_dataloaders


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DINO SVD + HSI Model for Deepfake Detection"
    )
    
    # Dataset settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="ff",
        choices=["ff", "celeb_df", "combined"],
        help="Dataset to train on"
    )
    parser.add_argument(
        "--frames_per_video",
        type=int,
        default=10,
        help="Number of frames to sample per video"
    )
    
    # Model settings
    parser.add_argument(
        "--dino_model",
        type=str,
        default="dinov2_vitb14",
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
        help="DINO model variant"
    )
    parser.add_argument(
        "--svd_rank",
        type=int,
        default=None,
        help="Number of singular values to keep frozen (default: feature_dim - 1)"
    )
    parser.add_argument(
        "--hsi_encoding_dim",
        type=int,
        default=256,
        help="HSI encoder output dimension"
    )
    parser.add_argument(
        "--hsi_hidden_dims",
        type=int,
        nargs="+",
        default=[64, 128],
        help="Hidden dimensions for HSI encoder CNN"
    )
    parser.add_argument(
        "--classifier_hidden_dims",
        type=int,
        nargs="+",
        default=[256, 16],
        help="Hidden dimensions for classifier head"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate"
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["attn"],
        help="Module name patterns to apply SVD to"
    )
    
    # Loss settings
    parser.add_argument(
        "--use_orthogonal_loss",
        action="store_true",
        help="Use orthogonality loss for SVD components"
    )
    parser.add_argument(
        "--orthogonal_loss_weight",
        type=float,
        default=0.01,
        help="Weight for orthogonality loss"
    )
    parser.add_argument(
        "--use_keepsv_loss",
        action="store_true",
        help="Use singular value preservation loss"
    )
    parser.add_argument(
        "--keepsv_loss_weight",
        type=float,
        default=0.01,
        help="Weight for keepsv loss"
    )
    parser.add_argument(
        "--use_weight_loss",
        action="store_true",
        help="Use weight diversity loss"
    )
    parser.add_argument(
        "--weight_loss_weight",
        type=float,
        default=0.001,
        help="Weight for weight diversity loss"
    )
    
    # Training settings
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--svd_lr",
        type=float,
        default=None,
        help="Learning rate for SVD parameters (default: same as --lr)"
    )
    parser.add_argument(
        "--hsi_lr",
        type=float,
        default=None,
        help="Learning rate for HSI encoder parameters (default: same as --lr)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "sgd"],
        help="Optimizer"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step", "plateau", "none"],
        help="Learning rate scheduler"
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        default=True,
        help="Use class-weighted loss for imbalanced data"
    )

    parser.add_argument(
        "--eval_only",
        action="store_true",
        default=False,
        help="Run evaluation only (no training)"
    )
    
    # Early stopping
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        default=True,
        help="Enable early stopping"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )
    
    # Hardware settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu/mps)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of data loading workers"
    )
    
    # Paths
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/dino_svd_hsi",
        help="Directory for saving checkpoints"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/dino_svd_hsi",
        help="Directory for saving logs"
    )
    
    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


class DinoSVDHSITrainer:
    """
    Trainer for DINO SVD + HSI Model.
    
    Handles:
    - Training with optional auxiliary losses (orthogonal, keepsv, weight)
    - Validation and metric tracking
    - Checkpointing
    - Early stopping
    """
    
    def __init__(
        self,
        config: Config,
        use_orthogonal_loss: bool = False,
        orthogonal_loss_weight: float = 0.01,
        use_keepsv_loss: bool = False,
        keepsv_loss_weight: float = 0.01,
        use_weight_loss: bool = False,
        weight_loss_weight: float = 0.001,
        svd_lr: Optional[float] = None,
        hsi_lr: Optional[float] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration object
            use_orthogonal_loss: Whether to use orthogonality loss
            orthogonal_loss_weight: Weight for orthogonality loss
            use_keepsv_loss: Whether to use keepsv loss
            keepsv_loss_weight: Weight for keepsv loss
            use_weight_loss: Whether to use weight diversity loss
            weight_loss_weight: Weight for weight diversity loss
            svd_lr: Separate learning rate for SVD parameters
            hsi_lr: Separate learning rate for HSI encoder parameters
        """
        self.config = config
        self.training_config = config.training
        
        # Loss settings
        self.use_orthogonal_loss = use_orthogonal_loss
        self.orthogonal_loss_weight = orthogonal_loss_weight
        self.use_keepsv_loss = use_keepsv_loss
        self.keepsv_loss_weight = keepsv_loss_weight
        self.use_weight_loss = use_weight_loss
        self.weight_loss_weight = weight_loss_weight
        self.svd_lr = svd_lr
        self.hsi_lr = hsi_lr
        
        # Set device
        self.device = self._setup_device()
        
        # Initialize tracking variables
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_auc = 0.0
        self.epochs_without_improvement = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_ce_loss': [],
            'train_orth_loss': [],
            'train_keepsv_loss': [],
            'train_weight_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        # Create directories
        self.checkpoint_dir = Path(self.training_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(self.training_config.log_dir)
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
        config_dict = {
            'config': asdict(self.config),
            'trainer_settings': {
                'use_orthogonal_loss': self.use_orthogonal_loss,
                'orthogonal_loss_weight': self.orthogonal_loss_weight,
                'use_keepsv_loss': self.use_keepsv_loss,
                'keepsv_loss_weight': self.keepsv_loss_weight,
                'use_weight_loss': self.use_weight_loss,
                'weight_loss_weight': self.weight_loss_weight,
                'svd_lr': self.svd_lr,
                'hsi_lr': self.hsi_lr
            }
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def _get_svd_params(self, model: DinoSVD_with_HSI_Model) -> List[nn.Parameter]:
        """Get trainable SVD residual parameters from the DINO backbone."""
        params = []
        for name, param in model.dino_svd.backbone.named_parameters():
            if any(x in name for x in ['S_residual', 'U_residual', 'V_residual']):
                if param.requires_grad:
                    params.append(param)
        return params
    
    def _get_hsi_encoder_params(self, model: DinoSVD_with_HSI_Model) -> List[nn.Parameter]:
        """Get trainable HSI encoder parameters (excluding frozen RGB2HSI)."""
        params = []
        for name, param in model.hsi_encoder.named_parameters():
            # Exclude RGB2HSI which is frozen
            if 'RGB2HSI' not in name and param.requires_grad:
                params.append(param)
        return params
    
    def _get_classifier_params(self, model: DinoSVD_with_HSI_Model) -> List[nn.Parameter]:
        """Get classifier parameters."""
        return list(model.classifier.parameters())
    
    def _create_optimizer(self, model: DinoSVD_with_HSI_Model) -> optim.Optimizer:
        """Create optimizer with separate parameter groups."""
        optimizer_name = self.training_config.optimizer.lower()
        lr = self.training_config.learning_rate
        weight_decay = self.training_config.weight_decay
        
        # Learning rates for different components
        svd_lr = self.svd_lr if self.svd_lr is not None else lr
        hsi_lr = self.hsi_lr if self.hsi_lr is not None else lr
        
        # Separate parameter groups
        svd_params = self._get_svd_params(model)
        hsi_params = self._get_hsi_encoder_params(model)
        classifier_params = self._get_classifier_params(model)
        
        parameters = [
            {"params": svd_params, "lr": svd_lr, "weight_decay": weight_decay, "name": "svd"},
            {"params": hsi_params, "lr": hsi_lr, "weight_decay": weight_decay, "name": "hsi"},
            {"params": classifier_params, "lr": lr, "weight_decay": weight_decay, "name": "classifier"}
        ]
        
        # Filter out empty parameter groups
        parameters = [p for p in parameters if len(list(p["params"])) > 0 or 
                      (hasattr(p["params"], '__len__') and len(p["params"]) > 0)]
        
        if optimizer_name == "adam":
            return optim.Adam(parameters)
        elif optimizer_name == "adamw":
            return optim.AdamW(parameters)
        elif optimizer_name == "sgd":
            return optim.SGD(parameters, momentum=0.9)
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
                step_size=5,
                gamma=0.1
            )
        elif scheduler_name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=5,
                factor=0.1,
                verbose=True
            )
        elif scheduler_name == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def _create_criterion(self, train_loader: DataLoader) -> nn.Module:
        """Create loss function with optional class weights."""
        if self.training_config.use_class_weights:
            # Count samples per class
            labels = train_loader.dataset.labels
            num_real = labels.count(0)
            num_fake = labels.count(1)
            total = num_real + num_fake
            
            # Weight inversely proportional to class frequency
            weights = torch.tensor([
                num_fake / total,  # weight for Real (0)
                num_real / total,  # weight for Fake (1)
            ], device=self.device)
            
            print(f"Using class-weighted loss:")
            print(f"  Real (0): {num_real} samples, weight={weights[0]:.4f}")
            print(f"  Fake (1): {num_fake} samples, weight={weights[1]:.4f}")
            
            return nn.CrossEntropyLoss(weight=weights)
        
        return nn.CrossEntropyLoss()
    
    def _compute_auxiliary_losses(self, model: DinoSVD_with_HSI_Model) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses from the DINO SVD backbone."""
        losses = {}
        
        if self.use_orthogonal_loss:
            losses['orthogonal'] = model.dino_svd.compute_orthogonal_loss() * self.orthogonal_loss_weight
        
        if self.use_keepsv_loss:
            losses['keepsv'] = model.dino_svd.compute_keepsv_loss() * self.keepsv_loss_weight
        
        if self.use_weight_loss:
            losses['weight'] = model.dino_svd.compute_weight_loss() * self.weight_loss_weight
        
        return losses
    
    def _setup_model_for_training(self, model: DinoSVD_with_HSI_Model):
        """
        Set up the model for training.
        
        This ensures:
        - RGB2HSI (MST++) is in eval mode and frozen
        - Other components are properly set up for training
        """
        # CRITICAL: Set RGB2HSI to eval mode (it should stay frozen)
        model.hsi_encoder.RGB2HSI.eval()
        model.hsi_encoder.RGB2HSI.requires_grad_(False)
        
        print("Model setup for training:")
        print("  - RGB2HSI (MST++): FROZEN and in EVAL mode")
        print("  - HSI feature extractor: TRAINABLE")
        print("  - DINO SVD residuals: TRAINABLE")
        print("  - Classifier: TRAINABLE")
    
    def _train_epoch(
        self,
        model: DinoSVD_with_HSI_Model,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scheduler,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        
        # CRITICAL: Keep RGB2HSI in eval mode even during training
        model.hsi_encoder.RGB2HSI.eval()
        
        total_loss = 0
        total_ce_loss = 0
        total_orth_loss = 0
        total_keepsv_loss = 0
        total_weight_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)
        
        for batch in pbar:
            images, labels = batch[0], batch[1]
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Handle video-level batches (B, T, C, H, W) -> (B*T, C, H, W)
            if images.dim() == 5:
                B, T, C, H, W = images.shape
                images = images.view(B * T, C, H, W)
                # Repeat labels for each frame
                labels = labels.unsqueeze(1).expand(B, T).reshape(B * T)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(images)
            
            # Classification loss
            ce_loss = criterion(logits, labels)
            total_ce_loss += ce_loss.item()
            
            # Auxiliary losses
            aux_losses = self._compute_auxiliary_losses(model)
            
            # Total loss
            loss = ce_loss
            for loss_name, loss_value in aux_losses.items():
                loss = loss + loss_value
                if loss_name == 'orthogonal':
                    total_orth_loss += loss_value.item()
                elif loss_name == 'keepsv':
                    total_keepsv_loss += loss_value.item()
                elif loss_name == 'weight':
                    total_weight_loss += loss_value.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        n_batches = len(train_loader)
        return {
            'loss': total_loss / n_batches,
            'ce_loss': total_ce_loss / n_batches,
            'orth_loss': total_orth_loss / n_batches if self.use_orthogonal_loss else 0,
            'keepsv_loss': total_keepsv_loss / n_batches if self.use_keepsv_loss else 0,
            'weight_loss': total_weight_loss / n_batches if self.use_weight_loss else 0,
            'acc': 100 * correct / total
        }
    
    @torch.no_grad()
    def _validate(
        self,
        model: DinoSVD_with_HSI_Model,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Validate the model."""
        model.eval()
        model.to(self.device)
        
        total_loss = 0
        all_labels = []
        all_preds = []
        all_probs = []
        
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            # Handle both tuple format (frames, labels) and dict format
            if isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[1]
            else:
                images, labels = batch['frames'], batch['label']
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Handle video-level batches
            if images.dim() == 5:
                B, T, C, H, W = images.shape
                images = images.view(B * T, C, H, W)
                labels_expanded = labels.unsqueeze(1).expand(B, T).reshape(B * T)
            else:
                labels_expanded = labels
            
            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels_expanded)
            total_loss += loss.item()
            
            # Get predictions
            probs = torch.softmax(logits, dim=1)[:, 1]
            _, predicted = torch.max(logits, 1)
            
            # For video-level evaluation, aggregate frame predictions
            if images.dim() == 5 or (len(labels) != len(predicted)):
                # Reshape back to video level
                probs = probs.view(B, T).mean(dim=1)
                predicted = (probs > 0.5).long()
                all_labels.extend(labels.cpu().numpy())
            else:
                all_labels.extend(labels_expanded.cpu().numpy())
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        # Compute metrics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        acc = accuracy_score(all_labels, all_preds) * 100
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5  # When only one class is present
        
        # Compute precision, recall, F1 score
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        return {
            'loss': total_loss / len(val_loader),
            'acc': acc,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def _save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                         scheduler, is_best: bool = False, filename: str = None):
        """Save a checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'best_val_auc': self.best_val_auc,
            'history': self.history
        }
        
        # Save latest checkpoint
        if filename is None:
            filename = 'checkpoint_latest_dino_svd_hsi.pt'
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'checkpoint_best_dino_svd_hsi.pt')
            print(f"  Saved best model (AUC: {self.best_val_auc:.4f})")
    
    def _load_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                         scheduler, checkpoint_path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_auc = checkpoint.get('best_val_auc', 0.0)
        self.history = checkpoint.get('history', self.history)
        
        print(f"Resumed from epoch {self.current_epoch} (Val accuracy: {self.best_val_acc})")
    
    def _save_history(self):
        """Save training history to file."""
        history_path = self.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _print_trainable_params(self, model: DinoSVD_with_HSI_Model):
        """Print information about trainable parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        svd_params = sum(p.numel() for p in self._get_svd_params(model))
        hsi_params = sum(p.numel() for p in self._get_hsi_encoder_params(model))
        classifier_params = sum(p.numel() for p in self._get_classifier_params(model))
        
        # Count RGB2HSI params (frozen)
        rgb2hsi_params = sum(p.numel() for p in model.hsi_encoder.RGB2HSI.parameters())
        
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  - SVD Residual Params: {svd_params:,}")
        print(f"  - HSI Encoder Params: {hsi_params:,}")
        print(f"  - Classifier Params: {classifier_params:,}")
        print(f"  - RGB2HSI (frozen): {rgb2hsi_params:,}")
    
    def train(
        self,
        model: DinoSVD_with_HSI_Model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            model: DinoSVD_with_HSI_Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Training results dictionary
        """
        # Move model to device
        model = model.to(self.device)
        
        # Setup model for training (set RGB2HSI to eval mode)
        self._setup_model_for_training(model)
        
        # Print model info
        print("\nModel Information:")
        self._print_trainable_params(model)
        
        # Create optimizer, scheduler, and criterion
        optimizer = self._create_optimizer(model)
        num_training_steps = len(train_loader) * self.training_config.num_epochs
        scheduler = self._create_scheduler(optimizer, num_training_steps)
        criterion = self._create_criterion(train_loader)
        
        # Resume from checkpoint if specified
        if resume_from:
            self._load_checkpoint(model, optimizer, scheduler, resume_from)
        
        print(f"\nStarting training for {self.training_config.num_epochs} epochs")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Device: {self.device}")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.training_config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train one epoch
            train_metrics = self._train_epoch(
                model, train_loader, optimizer, criterion, scheduler, epoch
            )
            
            # Validate
            val_metrics = self._validate(model, val_loader, criterion)
            
            # Update scheduler for ReduceLROnPlateau
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_ce_loss'].append(train_metrics['ce_loss'])
            self.history['train_orth_loss'].append(train_metrics['orth_loss'])
            self.history['train_keepsv_loss'].append(train_metrics['keepsv_loss'])
            self.history['train_weight_loss'].append(train_metrics['weight_loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Check for improvement
            is_best = val_metrics['acc'] > self.best_val_acc
            if is_best:
                self.best_val_auc = val_metrics['auc']
                self.best_val_acc = val_metrics['acc']
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self._save_checkpoint(model, optimizer, scheduler, is_best)
            self._save_history()
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1}/{self.training_config.num_epochs} "
                  f"[{epoch_time:.1f}s]")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"CE: {train_metrics['ce_loss']:.4f}, "
                  f"Acc: {train_metrics['acc']:.2f}%")
            if self.use_orthogonal_loss or self.use_keepsv_loss or self.use_weight_loss:
                aux_str = "  Aux Losses -"
                if self.use_orthogonal_loss:
                    aux_str += f" Orth: {train_metrics['orth_loss']:.4f}"
                if self.use_keepsv_loss:
                    aux_str += f" KeepSV: {train_metrics['keepsv_loss']:.4f}"
                if self.use_weight_loss:
                    aux_str += f" Weight: {train_metrics['weight_loss']:.4f}"
                print(aux_str)
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['acc']:.2f}%, "
                  f"AUC: {val_metrics['auc']:.4f}")
            print(f"          P: {val_metrics['precision']:.4f}, "
                  f"R: {val_metrics['recall']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
            print(f"  CM:")
            print(f"  TN={val_metrics['confusion_matrix'][0,0]:5d}  FP={val_metrics['confusion_matrix'][0,1]:5d}")
            print(f"  FN={val_metrics['confusion_matrix'][1,0]:5d}  TP={val_metrics['confusion_matrix'][1,1]:5d}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping check
            if self.training_config.early_stopping:
                if self.epochs_without_improvement >= self.training_config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time / 60:.1f} minutes")
        print(f"Best Validation AUC: {self.best_val_auc:.4f}")
        print(f"Best Validation Acc: {self.best_val_acc:.2f}%")
        
        # Load best model for evaluation
        best_checkpoint = torch.load(
            self.checkpoint_dir / 'checkpoint_best_dino_svd_hsi.pt',
            map_location=self.device
        )
        model.load_state_dict(best_checkpoint['model_state_dict'])
        
        # Evaluate on test set if provided
        if test_loader is not None:
            print("\nEvaluating on test set...")
            test_metrics = self._validate(model, test_loader, criterion)
            print(f"Test - Loss: {test_metrics['loss']:.4f}, "
                  f"Acc: {test_metrics['acc']:.2f}%, "
                  f"AUC: {test_metrics['auc']:.4f}")
            print(f"       P: {test_metrics['precision']:.4f}, "
                  f"R: {test_metrics['recall']:.4f}, "
                  f"F1: {test_metrics['f1']:.4f}")
            print(f"Confusion Matrix:")
            print(f"  TN={test_metrics['confusion_matrix'][0,0]:5d}  FP={test_metrics['confusion_matrix'][0,1]:5d}")
            print(f"  FN={test_metrics['confusion_matrix'][1,0]:5d}  TP={test_metrics['confusion_matrix'][1,1]:5d}")
            
            return {
                'history': self.history,
                'best_val_auc': self.best_val_auc,
                'best_val_acc': self.best_val_acc,
                'test_metrics': test_metrics
            }
        
        return {
            'history': self.history,
            'best_val_auc': self.best_val_auc,
            'best_val_acc': self.best_val_acc
        }


def create_config(args) -> Config:
    """Create configuration from arguments."""
    # Determine dataset root
    if args.dataset == "combined":
        dataset_root = Path("Datasets")
        dataset_mode = "combined"
    else:
        dataset_root = Path(f"Datasets/{'FF' if args.dataset == 'ff' else 'Celeb-DF-v2'}")
        dataset_mode = "single"
    
    data_config = DataConfig(
        dataset_type=args.dataset,
        dataset_mode=dataset_mode,
        dataset_root=dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_cache=True,
        preload_cache=True
    )
    
    training_config = TrainingConfig(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.patience,
        checkpoint_dir=Path(args.checkpoint_dir),
        log_dir=Path(args.log_dir),
        device=args.device,
        use_class_weights=args.use_class_weights
    )
    
    preprocessing_config = PreprocessingConfig(
        frames_per_video=args.frames_per_video,
        use_cache=True
    )
    
    return Config(
        data=data_config,
        training=training_config,
        preprocessing=preprocessing_config,
        experiment_name="dino_svd_hsi",
        seed=args.seed
    )


def main():
    """Main training function."""
    args = parse_args()
    config = create_config(args)
    
    print("=" * 60)
    print("DINO SVD + HSI Model Training for Deepfake Detection")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  DINO Model: {args.dino_model}")
    print(f"  SVD Rank: {args.svd_rank or 'auto (feature_dim - 1)'}")
    print(f"  Target Modules: {args.target_modules}")
    print(f"  HSI Encoding Dim: {args.hsi_encoding_dim}")
    print(f"  HSI Hidden Dims: {args.hsi_hidden_dims}")
    print(f"  Classifier Hidden Dims: {args.classifier_hidden_dims}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  SVD Learning Rate: {args.svd_lr or args.lr}")
    print(f"  HSI Learning Rate: {args.hsi_lr or args.lr}")
    
    if args.use_orthogonal_loss or args.use_keepsv_loss or args.use_weight_loss:
        print(f"\nAuxiliary Losses:")
        if args.use_orthogonal_loss:
            print(f"  Orthogonal Loss: weight={args.orthogonal_loss_weight}")
        if args.use_keepsv_loss:
            print(f"  KeepSV Loss: weight={args.keepsv_loss_weight}")
        if args.use_weight_loss:
            print(f"  Weight Loss: weight={args.weight_loss_weight}")
    
    print()
    
    # Create data loaders
    print("Loading data...")
    if args.dataset == "ff":
        train_loader, val_loader, test_loader = create_ff_dataloaders(
            root_dir=Path("Datasets/FF"),
            config=config,
            frames_per_video=args.frames_per_video,
            video_level=True
        )
    elif args.dataset == "celeb_df":
        train_loader, val_loader, test_loader = create_celeb_df_dataloaders(
            root_dir=Path("Datasets/Celeb-DF-v2"),
            config=config,
            frames_per_video=args.frames_per_video,
            video_level=True
        )
    else:  # combined
        dataset_configs = [
            {"name": "ff", "root_dir": "Datasets/FF", "weight": 1.0},
            {"name": "celeb_df", "root_dir": "Datasets/Celeb-DF-v2", "weight": 1.0}
        ]
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_type="combined",
            dataset_configs=dataset_configs,
            batch_size=args.batch_size,
            frames_per_video=args.frames_per_video,
            video_level=True
        )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = DinoSVD_with_HSI_Model(
        num_classes=2,
        hsi_encoding_dim=args.hsi_encoding_dim,
        hsi_hidden_dims=args.hsi_hidden_dims,
        classifier_hidden_dims=args.classifier_hidden_dims,
        dropout=args.dropout
    )
    
    # Note: The DinoSVD_with_HSI_Model internally creates DinoSVDModel
    # We need to ensure SVD rank and target modules are set properly
    # This requires modifying the model or passing these args
    # For now, the model uses default settings from DinoSVDModel
    
    print("\nCreating trainer...")
    # Create trainer
    trainer = DinoSVDHSITrainer(
        config=config,
        use_orthogonal_loss=args.use_orthogonal_loss,
        orthogonal_loss_weight=args.orthogonal_loss_weight,
        use_keepsv_loss=args.use_keepsv_loss,
        keepsv_loss_weight=args.keepsv_loss_weight,
        use_weight_loss=args.use_weight_loss,
        weight_loss_weight=args.weight_loss_weight,
        svd_lr=args.svd_lr,
        hsi_lr=args.hsi_lr
    )
    print("\nModel and Trainer created successfully!")
    
    # Train
    if not args.eval_only:
        results = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            resume_from=args.resume
        )
    
        print("\nTraining complete!")
        print(f"Best Validation AUC: {results['best_val_auc']:.4f}")
        print(f"Best Validation Acc: {results['best_val_acc']:.2f}%")
        
        if 'test_metrics' in results:
            print(f"Test AUC: {results['test_metrics']['auc']:.4f}")
            print(f"Test Acc: {results['test_metrics']['acc']:.2f}%")

    # Load best checkpoint for final evaluation
    best_checkpoint = torch.load(
        trainer.checkpoint_dir / 'checkpoint_best_dino_svd_hsi.pt',
        map_location=trainer.device
    )
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model = model.to(trainer.device)
    
    # Set RGB2HSI to eval mode for evaluation
    model.hsi_encoder.RGB2HSI.eval()

    print("\nEvaluating on test set...")
    test_metrics = trainer._validate(model, test_loader, trainer._create_criterion(train_loader))
    print(f"Test - Loss: {test_metrics['loss']:.4f}, "
          f"Acc: {test_metrics['acc']:.2f}%, "
          f"AUC: {test_metrics['auc']:.4f}")
    print(f"       P: {test_metrics['precision']:.4f}, "
          f"R: {test_metrics['recall']:.4f}, "
          f"F1: {test_metrics['f1']:.4f}")
    print(f"Confusion Matrix:")
    print(f"  TN={test_metrics['confusion_matrix'][0,0]:5d}  FP={test_metrics['confusion_matrix'][0,1]:5d}")
    print(f"  FN={test_metrics['confusion_matrix'][1,0]:5d}  TP={test_metrics['confusion_matrix'][1,1]:5d}")


if __name__ == "__main__":
    main()
