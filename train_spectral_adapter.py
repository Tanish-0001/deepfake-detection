#!/usr/bin/env python3
"""
Train DinoSVD + Parameter-Efficient Spectral Adapter Model for Deepfake Detection.

This script trains a model that uses:
- MST++ to upsample RGB → 31-channel HSI (frozen, pretrained)
- A trainable Conv2d to tokenise the HSI into spectral tokens
- A fully-frozen DinoSVD backbone with SpectralAdapters injected at each block
- A trainable classifier head

Only the HSI tokenizer, spectral adapters, and classifier are trained.

Usage:
    # Train on FF
    python train_spectral_adapter.py --dataset ff

    # With custom adapter settings
    python train_spectral_adapter.py --dataset ff --bottleneck_dim 64 --adapter_scale 0.1

    # Quick test run
    python train_spectral_adapter.py --dataset ff --epochs 2 --batch_size 2
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
    roc_curve,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import Config, DataConfig, TrainingConfig, PreprocessingConfig
from models.DinoSVD_SpectralAdapter import DinoSVD_SpectralAdapter_Model
from data import get_dataloaders
from data.dataloader import create_ff_dataloaders, create_celeb_df_dataloaders


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DinoSVD + Spectral Adapter Model for Deepfake Detection"
    )

    # Dataset settings
    parser.add_argument(
        "--dataset", type=str, default="ff", choices=["ff", "celeb_df", "combined"],
        help="Dataset to train on"
    )
    parser.add_argument(
        "--frames_per_video", type=int, default=10,
        help="Number of frames to sample per video"
    )

    # Model settings
    parser.add_argument(
        "--dino_model", type=str, default="dinov2_vitb14",
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
        help="DINO model variant"
    )
    parser.add_argument(
        "--svd_rank", type=int, default=None,
        help="Number of singular values to keep in DinoSVD (default: feature_dim - 1)"
    )
    parser.add_argument(
        "--target_modules", type=str, nargs="+", default=["attn"],
        help="Module name patterns to apply SVD to in DinoSVD"
    )
    parser.add_argument(
        "--bottleneck_dim", type=int, default=64,
        help="Bottleneck dimension d for spectral adapters"
    )
    parser.add_argument(
        "--adapter_scale", type=float, default=0.1,
        help="Scaling factor s for adapter residuals"
    )
    parser.add_argument(
        "--classifier_hidden_dims", type=int, nargs="+", default=[256, 16],
        help="Hidden dimensions for classifier head"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate"
    )

    # Training settings
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for classifier")
    parser.add_argument(
        "--adapter_lr", type=float, default=None,
        help="Separate learning rate for spectral adapters (default: same as --lr)"
    )
    parser.add_argument(
        "--tokenizer_lr", type=float, default=None,
        help="Separate learning rate for HSI tokenizer (default: same as --lr)"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"],
        help="Optimizer"
    )
    parser.add_argument(
        "--scheduler", type=str, default="cosine",
        choices=["cosine", "step", "plateau", "none"],
        help="Learning rate scheduler"
    )
    parser.add_argument(
        "--use_class_weights", action="store_true", default=False,
        help="Use class-weighted loss for imbalanced data"
    )

    # DinoSVD finetuning
    parser.add_argument(
        "--finetune_dino", action="store_true", default=False,
        help="Unfreeze DinoSVD backbone after --unfreeze_after epochs"
    )
    parser.add_argument(
        "--unfreeze_after", type=int, default=10,
        help="Epoch after which to unfreeze DinoSVD (used with --finetune_dino)"
    )
    parser.add_argument(
        "--unfreeze_schedule", type=int, nargs="+", default=[25, 30, 35],
        help="Epochs to progressively unfreeze DinoSVD stages (last 4, middle 4, first 4 blocks)"
    )
    parser.add_argument(
        "--dino_lr", type=float, default=1e-5,
        help="Learning rate for DinoSVD backbone after unfreezing"
    )

    # Early stopping & Hardware
    parser.add_argument("--early_stopping", action="store_true", default=True, help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loading workers")

    # Paths & Misc
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/spectral_adapter")
    parser.add_argument("--log_dir", type=str, default="logs/spectral_adapter")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--eval_only", action="store_true", default=False, help="Run evaluation only (no training)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


class SpectralAdapterTrainer:
    """Trainer for DinoSVD + Spectral Adapter Model."""

    def __init__(
        self,
        config: Config,
        adapter_lr: Optional[float] = None,
        tokenizer_lr: Optional[float] = None,
        finetune_dino: bool = False,
        unfreeze_after: int = 10,
        unfreeze_schedule: List[int] = None,
        dino_lr: float = 1e-5,
    ):
        self.config = config
        self.training_config = config.training
        self.adapter_lr = adapter_lr
        self.tokenizer_lr = tokenizer_lr
        self.finetune_dino = finetune_dino
        self.unfreeze_after = unfreeze_after
        self.unfreeze_schedule = unfreeze_schedule if unfreeze_schedule is not None else []
        self.dino_lr = dino_lr
        self.unfrozen_stages = set()

        self.device = self._setup_device()

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_auc = 0.0
        self.best_threshold = 0.5
        self.epochs_without_improvement = 0

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': [],
            'val_precision': [], 'val_recall': [], 'val_f1': [],
            'learning_rates': []
        }

        self.checkpoint_dir = Path(self.training_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(self.training_config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._save_config()
        self._set_seed(config.seed)

    def _setup_device(self) -> torch.device:
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
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def _save_config(self):
        config_path = self.log_dir / "config.json"
        config_dict = {
            'config': asdict(self.config),
            'trainer_settings': {
                'adapter_lr': self.adapter_lr,
                'tokenizer_lr': self.tokenizer_lr,
                'finetune_dino': self.finetune_dino,
                'unfreeze_after': self.unfreeze_after,
                'unfreeze_schedule': self.unfreeze_schedule,
                'dino_lr': self.dino_lr,
            }
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

    def _create_optimizer(self, model: DinoSVD_SpectralAdapter_Model) -> optim.Optimizer:
        """Create optimizer with separate parameter groups."""
        opt_name = self.training_config.optimizer.lower()
        lr = self.training_config.learning_rate
        wd = self.training_config.weight_decay

        adapter_lr = self.adapter_lr if self.adapter_lr is not None else lr
        tokenizer_lr = self.tokenizer_lr if self.tokenizer_lr is not None else lr

        parameters = [
            {"params": model.get_tokenizer_params(), "lr": tokenizer_lr,
             "weight_decay": wd, "name": "hsi_tokenizer"},
            {"params": model.get_adapter_params(), "lr": adapter_lr,
             "weight_decay": wd, "name": "adapters"},
            {"params": model.get_classifier_params(), "lr": lr,
             "weight_decay": wd, "name": "classifier"},
        ]

        # Filter out empty parameter groups
        parameters = [p for p in parameters if len(p["params"]) > 0]

        if opt_name == "adam":
            return optim.Adam(parameters)
        elif opt_name == "adamw":
            return optim.AdamW(parameters)
        elif opt_name == "sgd":
            return optim.SGD(parameters, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _add_new_dino_params_to_optimizer(self, model: DinoSVD_SpectralAdapter_Model, optimizer: optim.Optimizer):
        existing_params = set()
        for group in optimizer.param_groups:
            for p in group['params']:
                existing_params.add(p)
                
        new_dino_params = [
            p for p in model.dino_svd.parameters()
            if p.requires_grad and p not in existing_params
        ]
        
        if new_dino_params:
            optimizer.add_param_group({
                'params': new_dino_params,
                'lr': self.dino_lr,
                'weight_decay': self.training_config.weight_decay,
                'name': f'dino_svd_stage_{len(optimizer.param_groups)}'
            })
            print(f"Added param group for {len(new_dino_params)} newly unfrozen DinoSVD tensors.")

    def _create_optimizer_with_dino(self, model: DinoSVD_SpectralAdapter_Model) -> optim.Optimizer:
        """Rebuild optimizer adding DinoSVD params at a small learning rate."""
        opt_name = self.training_config.optimizer.lower()
        lr = self.training_config.learning_rate
        wd = self.training_config.weight_decay

        adapter_lr = self.adapter_lr if self.adapter_lr is not None else lr
        tokenizer_lr = self.tokenizer_lr if self.tokenizer_lr is not None else lr
        dino_params = [p for p in model.dino_svd.parameters() if p.requires_grad]

        parameters = [
            {"params": model.get_tokenizer_params(), "lr": tokenizer_lr,
             "weight_decay": wd, "name": "hsi_tokenizer"},
            {"params": model.get_adapter_params(), "lr": adapter_lr,
             "weight_decay": wd, "name": "adapters"},
            {"params": model.get_classifier_params(), "lr": lr,
             "weight_decay": wd, "name": "classifier"},
            {"params": dino_params, "lr": self.dino_lr,
             "weight_decay": wd, "name": "dino_svd"},
        ]

        parameters = [p for p in parameters if len(p["params"]) > 0]
        print(f"Rebuilt optimizer with DinoSVD params (lr={self.dino_lr:.1e})")

        if opt_name == "adam":
            return optim.Adam(parameters)
        elif opt_name == "adamw":
            return optim.AdamW(parameters)
        elif opt_name == "sgd":
            return optim.SGD(parameters, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _create_scheduler(self, optimizer: optim.Optimizer, num_training_steps: int):
        sched_name = self.training_config.scheduler.lower()
        if sched_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_training_steps, eta_min=1e-7
            )
        elif sched_name == "step":
            return optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        elif sched_name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.1, verbose=True
            )
        elif sched_name == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {sched_name}")

    def _create_criterion(self, train_loader: DataLoader) -> nn.Module:
        if self.training_config.use_class_weights:
            labels = train_loader.dataset.labels
            num_real = labels.count(0)
            num_fake = labels.count(1)
            total = num_real + num_fake

            weights = torch.tensor([
                num_fake / total,   # weight for Real (0)
                num_real / total,   # weight for Fake (1)
            ], device=self.device)

            print(f"Using class-weighted loss:")
            print(f"  Real (0): {num_real} samples, weight={weights[0]:.4f}")
            print(f"  Fake (1): {num_fake} samples, weight={weights[1]:.4f}")

            return nn.CrossEntropyLoss(weight=weights)

        return nn.CrossEntropyLoss()

    def _setup_model_for_training(self, model: DinoSVD_SpectralAdapter_Model):
        """Ensure frozen and trainable components are set correctly."""
        # MST++ frozen + eval
        model.rgb2hsi.eval()
        model.rgb2hsi.requires_grad_(False)

        # DinoSVD frozen + eval
        model.dino_svd.eval()
        model.dino_svd.requires_grad_(False)

        # Trainable components
        model.hsi_tokenizer.requires_grad_(True)
        for ab in model.adapted_blocks:
            ab.adapter.requires_grad_(True)
        model.classifier.requires_grad_(True)

        print("Model setup for training:")
        print("  - MST++ (RGB→HSI):      FROZEN, eval mode")
        print("  - DinoSVD backbone:      FROZEN, eval mode")
        print("  - HSI Tokenizer (31→D):  TRAINABLE")
        print(f"  - Spectral Adapters:     TRAINABLE ({len(model.adapted_blocks)} blocks)")
        print("  - Classifier head:       TRAINABLE")

    def _train_epoch(self, model, train_loader, optimizer, criterion, scheduler, epoch):
        model.train()
        # Keep frozen components in eval mode
        model.rgb2hsi.eval()
        model.dino_svd.eval()

        total_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)

        for batch in pbar:
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            if images.dim() == 5:
                B, T, C, H, W = images.shape
                images = images.view(B * T, C, H, W)
                labels = labels.unsqueeze(1).expand(B, T).reshape(B * T)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

            if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

        return {'loss': total_loss / len(train_loader), 'acc': 100 * correct / total}

    @torch.no_grad()
    def _validate(self, model, val_loader, criterion, threshold=None):
        model.eval()
        total_loss, all_labels, all_probs = 0, [], []

        for batch in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            if images.dim() == 5:
                B, T, C, H, W = images.shape
                images = images.view(B * T, C, H, W)
                labels_expanded = labels.unsqueeze(1).expand(B, T).reshape(B * T)
            else:
                labels_expanded = labels

            logits = model(images)
            loss = criterion(logits, labels_expanded)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]

            if images.dim() == 5:
                probs = probs.view(B, T).mean(dim=1)
                all_labels.extend(labels.cpu().numpy())
            else:
                all_labels.extend(labels_expanded.cpu().numpy())

            all_probs.extend(probs.cpu().numpy())

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        if threshold is None:
            # roc_curve calculates the False Positive Rate (FPR) and True Positive Rate (TPR) 
            # for all possible thresholds in your probability array.
            fpr, tpr, thresholds_roc = roc_curve(all_labels, all_probs)
            
            # Youden's J statistic = Sensitivity (TPR) + Specificity (1 - FPR) - 1
            # Mathematically, this simplifies to: J = TPR - FPR
            j_scores = tpr - fpr
            
            # Find the index of the highest J score
            best_idx = np.argmax(j_scores)
            
            # Extract the corresponding optimal threshold
            threshold = float(thresholds_roc[best_idx])
            threshold = max(0.1, min(0.9, threshold))  # Clamp the threshold to prevent extreme edge cases

        all_preds = (all_probs >= threshold).astype(int)

        acc = accuracy_score(all_labels, all_preds) * 100
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5

        return {
            'loss': total_loss / len(val_loader),
            'acc': acc, 'auc': auc,
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'confusion_matrix': confusion_matrix(all_labels, all_preds),
            'threshold': threshold
        }

    def _save_checkpoint(self, model, optimizer, scheduler, is_best=False, filename=None):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'best_val_auc': self.best_val_auc,
            'best_threshold': self.best_threshold,
            'history': self.history,
            'unfrozen_stages': list(self.unfrozen_stages),
        }

        if filename is None:
            filename = 'checkpoint_latest_spectral_adapter.pt'

        torch.save(checkpoint, self.checkpoint_dir / filename)

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'checkpoint_best_spectral_adapter.pt')
            print(f"  Saved best model (AUC: {self.best_val_auc:.4f})")

    def _load_checkpoint(self, model, optimizer, scheduler, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model.load_state_dict(checkpoint['model_state_dict'])

        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_auc = checkpoint.get('best_val_auc', 0.0)
        self.best_threshold = checkpoint.get('best_threshold', 0.5)
        self.history = checkpoint.get('history', self.history)
        self.unfrozen_stages = set(checkpoint.get('unfrozen_stages', []))

        # Backwards compatibility check
        if checkpoint.get('dino_unfrozen', False) and not self.unfrozen_stages:
            model.unfreeze_dino()
            optimizer = self._create_optimizer_with_dino(model)
            self.unfrozen_stages = {0}
        else:
            for stage in sorted(list(self.unfrozen_stages)):
                model.unfreeze_dino_stage(stage)
                self._add_new_dino_params_to_optimizer(model, optimizer)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Resumed from epoch {self.current_epoch} (Val accuracy: {self.best_val_acc:.2f}%)")
        if self.unfrozen_stages:
            print(f"  DinoSVD stages {list(self.unfrozen_stages)} were already unfrozen in checkpoint")

        return optimizer, scheduler

    def _save_history(self):
        history_path = self.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def train(self, model, train_loader, val_loader, test_loader=None, resume_from=None):
        model = model.to(self.device)
        self._setup_model_for_training(model)

        print("\nModel Information:")
        model.print_trainable_params()

        optimizer = self._create_optimizer(model)
        num_training_steps = len(train_loader) * self.training_config.num_epochs
        scheduler = self._create_scheduler(optimizer, num_training_steps)
        criterion = self._create_criterion(train_loader)

        if resume_from:
            optimizer, scheduler = self._load_checkpoint(
                model, optimizer, scheduler, resume_from
            )

        print(f"\nStarting training for {self.training_config.num_epochs} epochs")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Device: {self.device}")
        print("-" * 60)

        start_time = time.time()

        for epoch in range(self.current_epoch, self.training_config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Check if we should unfreeze DinoSVD this epoch
            if self.finetune_dino:
                unfreeze_triggered = False
                
                if self.unfreeze_schedule:
                    for stage_idx, target_epoch in enumerate(self.unfreeze_schedule):
                        # Epochs loop is 0-indexed, display is 1-indexed. Target epoch matches display.
                        if (epoch + 1) >= target_epoch and stage_idx not in self.unfrozen_stages:
                            print(f"\n{'='*60}")
                            print(f"Unfreezing DinoSVD Stage {stage_idx} at epoch {epoch + 1}")
                            print(f"{'='*60}")
                            model.unfreeze_dino_stage(stage_idx)
                            self.unfrozen_stages.add(stage_idx)
                            unfreeze_triggered = True

                elif not self.unfrozen_stages and epoch >= self.unfreeze_after:
                    print(f"\n{'='*60}")
                    print(f"Unfreezing DinoSVD entirely at epoch {epoch + 1}")
                    print(f"{'='*60}")
                    model.unfreeze_dino()
                    self.unfrozen_stages.add(0)
                    unfreeze_triggered = True
                
                if unfreeze_triggered:
                    self._add_new_dino_params_to_optimizer(model, optimizer)
                    self.epochs_without_improvement = 0
                    model.print_trainable_params()

            # Train & validate
            train_metrics = self._train_epoch(
                model, train_loader, optimizer, criterion, scheduler, epoch
            )
            val_metrics = self._validate(model, val_loader, criterion)

            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])

            # Track history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rates'].append(
                [pg['lr'] for pg in optimizer.param_groups]
            )

            # Check improvement
            is_best = val_metrics['acc'] > self.best_val_acc
            if is_best:
                self.best_val_auc = val_metrics['auc']
                self.best_val_acc = val_metrics['acc']
                self.best_val_loss = val_metrics['loss']
                self.best_threshold = val_metrics['threshold']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self._save_checkpoint(model, optimizer, scheduler, is_best=is_best)
            self._save_history()

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch + 1}/{self.training_config.num_epochs} "
                f"({epoch_time:.1f}s) | "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"Acc: {train_metrics['acc']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} "
                f"Acc: {val_metrics['acc']:.2f}% "
                f"AUC: {val_metrics['auc']:.4f} "
                f"F1: {val_metrics['f1']:.4f} "
                f"(Thresh: {val_metrics['threshold']:.2f})"
                f"{' ★' if is_best else ''}"
            )
            print(f"  CM:")
            print(f"  TN={val_metrics['confusion_matrix'][0,0]:5d}  FP={val_metrics['confusion_matrix'][0,1]:5d}")
            print(f"  FN={val_metrics['confusion_matrix'][1,0]:5d}  TP={val_metrics['confusion_matrix'][1,1]:5d}")

            # Early stopping
            if (self.training_config.early_stopping
                    and self.epochs_without_improvement >= self.training_config.early_stopping_patience):
                print(f"\nEarly stopping at epoch {epoch + 1} "
                      f"(no improvement for {self.epochs_without_improvement} epochs)")
                break

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time / 60:.1f} minutes")
        print(f"Best Val Acc: {self.best_val_acc:.2f}%  AUC: {self.best_val_auc:.4f}")

        # Final evaluation on test set
        if test_loader:
            best_ckpt = self.checkpoint_dir / 'checkpoint_best_spectral_adapter.pt'
            if best_ckpt.exists():
                ckpt = torch.load(best_ckpt, map_location=self.device)
                model.load_state_dict(ckpt['model_state_dict'])
                if 'best_threshold' in ckpt:
                    self.best_threshold = ckpt['best_threshold']
            test_metrics = self._validate(model, test_loader, criterion, threshold=self.best_threshold)
            print(f"\nTest Results (Threshold: {self.best_threshold:.4f}):")
            print(f"  Accuracy:  {test_metrics['acc']:.2f}%")
            print(f"  AUC:       {test_metrics['auc']:.4f}")
            print(f"  Precision: {test_metrics['precision']:.4f}")
            print(f"  Recall:    {test_metrics['recall']:.4f}")
            print(f"  F1:        {test_metrics['f1']:.4f}")
            print(f"  Confusion Matrix:\n{test_metrics['confusion_matrix']}")

        return {
            'best_val_auc': self.best_val_auc,
            'best_val_acc': self.best_val_acc,
        }


def create_config(args):
    dataset_root = (
        Path("Datasets") if args.dataset == "combined"
        else Path(f"Datasets/{'FF' if args.dataset == 'ff' else 'Celeb-DF-v2'}")
    )
    return Config(
        data=DataConfig(
            dataset_type=args.dataset,
            dataset_mode="combined" if args.dataset == "combined" else "single",
            dataset_root=dataset_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_cache=True,
            preload_cache=True,
        ),
        training=TrainingConfig(
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
            use_class_weights=args.use_class_weights,
        ),
        preprocessing=PreprocessingConfig(
            frames_per_video=args.frames_per_video,
            use_cache=True,
        ),
        experiment_name="spectral_adapter",
        seed=args.seed,
    )


def main():
    args = parse_args()
    config = create_config(args)

    use_weighted_sampler = not args.use_class_weights

    print("Loading data...")
    if args.dataset == "ff":
        train_loader, val_loader, test_loader = create_ff_dataloaders(
            root_dir=Path("Datasets/FF"), config=config,
            frames_per_video=args.frames_per_video, video_level=True,
            use_weighted_sampler=use_weighted_sampler
        )
    elif args.dataset == "celeb_df":
        train_loader, val_loader, test_loader = create_celeb_df_dataloaders(
            root_dir=Path("Datasets/Celeb-DF-v2"), config=config,
            frames_per_video=args.frames_per_video, video_level=True,
            use_weighted_sampler=use_weighted_sampler
        )
    else:
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_type="combined",
            dataset_configs=[
                {"name": "ff", "root_dir": "Datasets/FF", "weight": 1.0},
                {"name": "celeb_df", "root_dir": "Datasets/Celeb-DF-v2", "weight": 1.0},
            ],
            batch_size=args.batch_size,
            frames_per_video=args.frames_per_video,
            video_level=True,
            use_weighted_sampler=use_weighted_sampler
        )

    print("\nCreating model...")
    model = DinoSVD_SpectralAdapter_Model(
        num_classes=2,
        bottleneck_dim=args.bottleneck_dim,
        adapter_scale=args.adapter_scale,
        classifier_hidden_dims=args.classifier_hidden_dims,
        dropout=args.dropout,
        dino_model=args.dino_model,
        svd_rank=args.svd_rank,
        target_modules=args.target_modules,
    )

    trainer = SpectralAdapterTrainer(
        config=config,
        adapter_lr=args.adapter_lr,
        tokenizer_lr=args.tokenizer_lr,
        finetune_dino=args.finetune_dino,
        unfreeze_after=args.unfreeze_after,
        unfreeze_schedule=args.unfreeze_schedule,
        dino_lr=args.dino_lr,
    )

    if not args.eval_only:
        trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            resume_from=args.resume,
        )

    # Load best checkpoint for final evaluation
    best_checkpoint = torch.load(
        trainer.checkpoint_dir / 'checkpoint_best_spectral_adapter.pt',
        map_location=trainer.device
    )
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model = model.to(trainer.device)

    best_threshold = best_checkpoint.get('best_threshold', 0.5)

    print(f"\nEvaluating on test set (Threshold: {best_threshold:.4f})...")
    test_metrics = trainer._validate(model, test_loader, trainer._create_criterion(train_loader), threshold=best_threshold)
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
