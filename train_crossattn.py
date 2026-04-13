#!/usr/bin/env python3
"""
Train DINO SVD + HSI Cross-Attention Model for Deepfake Detection.

This script trains a model that uses:
- MST++ (frozen) → 3D/2D CNN (trainable)
- DINO SVD backbone for feature extraction (frozen initially)
- Cross-Attention fusion (trainable)
- MLP classifier head (trainable)

Usage:
    # Train on FF
    python train_crossattn.py --dataset ff
    
    # Quick test run
    python train_crossattn.py --dataset ff --epochs 2 --batch_size 2
"""

import argparse
import json
import time
import sys
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any, Optional
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
)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import Config, DataConfig, TrainingConfig, PreprocessingConfig
from models.DinoSVD_HSI_CrossAttention import DinoSVD_HSI_CrossAttention_Model
from data import get_dataloaders
from data.dataloader import create_ff_dataloaders, create_celeb_df_dataloaders


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DINO SVD + HSI Cross-Attention Model for Deepfake Detection"
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
        help="DINO model variant"
    )
    parser.add_argument(
        "--svd_rank", type=int, default=None,
        help="Number of singular values to keep in DinoSVD"
    )
    parser.add_argument(
        "--cross_attn_d_model", type=int, default=256,
        help="Dimension of cross-attention layer"
    )
    parser.add_argument(
        "--cross_attn_heads", type=int, default=8,
        help="Number of heads in cross-attention layer"
    )
    parser.add_argument(
        "--classifier_hidden_dim", type=int, default=128,
        help="Hidden dimension for classifier head MLP"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate"
    )
    
    # Training settings
    parser.add_argument(
        "--epochs", type=int, default=60,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate for classifier and general parameters"
    )
    parser.add_argument(
        "--hsi_lr", type=float, default=None,
        help="Separate learning rate for HSI encoder (default: same as --lr)"
    )
    parser.add_argument(
        "--attn_lr", type=float, default=None,
        help="Separate learning rate for cross-attention (default: same as --lr)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="Weight decay"
    )
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"],
        help="Optimizer"
    )
    parser.add_argument(
        "--scheduler", type=str, default="cosine", choices=["cosine", "step", "plateau", "none"],
        help="Learning rate scheduler"
    )
    
    # DinoSVD finetuning
    parser.add_argument(
        "--finetune_dino", action="store_true", default=False,
        help="Unfreeze DinoSVD backbone after --unfreeze_after epochs"
    )
    parser.add_argument(
        "--unfreeze_after", type=int, default=10,
        help="Epoch after which to unfreeze DinoSVD"
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
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/dino_svd_crossattn")
    parser.add_argument("--log_dir", type=str, default="logs/dino_svd_crossattn")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--eval_only", action="store_true", default=False, help="Run evaluation only (no training)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


class CrossAttnTrainer:
    """Trainer for DINO SVD + HSI Cross-Attention Model."""
    
    def __init__(
        self,
        config: Config,
        hsi_lr: Optional[float] = None,
        attn_lr: Optional[float] = None,
        finetune_dino: bool = False,
        unfreeze_after: int = 10,
        dino_lr: float = 1e-5,
    ):
        self.config = config
        self.training_config = config.training
        self.hsi_lr = hsi_lr
        self.attn_lr = attn_lr
        self.finetune_dino = finetune_dino
        self.unfreeze_after = unfreeze_after
        self.dino_lr = dino_lr
        self.dino_unfrozen = False
        
        self.device = self._setup_device()
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_auc = 0.0
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
        
        self._set_seed(config.seed)
    
    def _setup_device(self) -> torch.device:
        device_str = self.training_config.device
        if device_str == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif device_str == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        return device
    
    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def _create_optimizer(self, model: DinoSVD_HSI_CrossAttention_Model) -> optim.Optimizer:
        """Create optimizer with parameter groups for HSI encoder, attention, and classifier."""
        opt_name = self.training_config.optimizer.lower()
        lr = self.training_config.learning_rate
        wd = self.training_config.weight_decay
        
        hsi_lr = self.hsi_lr if self.hsi_lr is not None else lr
        attn_lr = self.attn_lr if self.attn_lr is not None else lr
        
        parameters = [
            {"params": list(model.hsi_encoder.parameters()), "lr": hsi_lr, "weight_decay": wd, "name": "hsi_encoder"},
            {"params": list(model.cross_attn.parameters()), "lr": attn_lr, "weight_decay": wd, "name": "cross_attn"},
            {"params": list(model.classifier.parameters()), "lr": lr, "weight_decay": wd, "name": "classifier"},
        ]
        
        # Filter out empty parameter groups (e.g., if a component is frozen)
        parameters = [p for p in parameters if sum(1 for par in p["params"] if par.requires_grad) > 0]
        
        if opt_name == "adam": return optim.Adam(parameters)
        elif opt_name == "adamw": return optim.AdamW(parameters)
        elif opt_name == "sgd": return optim.SGD(parameters, momentum=0.9)
        else: raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _create_optimizer_with_dino(self, model: DinoSVD_HSI_CrossAttention_Model) -> optim.Optimizer:
        """Rebuild optimizer adding DinoSVD params at a small learning rate."""
        opt_name = self.training_config.optimizer.lower()
        lr = self.training_config.learning_rate
        wd = self.training_config.weight_decay
        
        hsi_lr = self.hsi_lr if self.hsi_lr is not None else lr
        attn_lr = self.attn_lr if self.attn_lr is not None else lr
        
        dino_params = [p for p in model.dino_svd.parameters() if p.requires_grad]
        
        parameters = [
            {"params": list(model.hsi_encoder.parameters()), "lr": hsi_lr, "weight_decay": wd, "name": "hsi_encoder"},
            {"params": list(model.cross_attn.parameters()), "lr": attn_lr, "weight_decay": wd, "name": "cross_attn"},
            {"params": list(model.classifier.parameters()), "lr": lr, "weight_decay": wd, "name": "classifier"},
            {"params": dino_params, "lr": self.dino_lr, "weight_decay": wd, "name": "dino_svd"},
        ]
        
        parameters = [p for p in parameters if sum(1 for par in p["params"] if par.requires_grad) > 0]
        print(f"Rebuilt optimizer with DinoSVD params (lr={self.dino_lr:.1e})")
        
        if opt_name == "adam": return optim.Adam(parameters)
        elif opt_name == "adamw": return optim.AdamW(parameters)
        elif opt_name == "sgd": return optim.SGD(parameters, momentum=0.9)
        pass

    def _create_scheduler(self, optimizer: optim.Optimizer, num_training_steps: int):
        sched_name = self.training_config.scheduler.lower()
        if sched_name == "cosine": return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-7)
        elif sched_name == "step": return optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        elif sched_name == "plateau": return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
        elif sched_name == "none": return None
        
    def _create_criterion(self, train_loader: DataLoader) -> nn.Module:
        return nn.CrossEntropyLoss()
    
    def _setup_model_for_training(self, model: DinoSVD_HSI_CrossAttention_Model):
        model.hsi_encoder.rgb2hsi.eval()
        model.hsi_encoder.rgb2hsi.requires_grad_(False)
        model.dino_svd.eval()
        model.dino_svd.requires_grad_(False)
        
        # Ensure our new modules are trainable
        model.hsi_encoder.requires_grad_(True)
        model.cross_attn.requires_grad_(True)
        model.classifier.requires_grad_(True)
        model.hsi_encoder.rgb2hsi.requires_grad_(False) # Re-freeze MST++ just in case
        
    def _train_epoch(self, model, train_loader, optimizer, criterion, scheduler, epoch):
        model.train()
        model.hsi_encoder.rgb2hsi.eval()
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
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            
            if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})
        
        return {'loss': total_loss / len(train_loader), 'acc': 100 * correct / total}
    
    @torch.no_grad()
    def _validate(self, model, val_loader, criterion):
        model.eval()
        total_loss, all_labels, all_preds, all_probs = 0, [], [], []
        
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
            _, predicted = torch.max(logits, 1)
            
            if images.dim() == 5:
                probs = probs.view(B, T).mean(dim=1)
                predicted = (probs > 0.5).long()
                all_labels.extend(labels.cpu().numpy())
            else:
                all_labels.extend(labels_expanded.cpu().numpy())
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
        all_labels, all_preds, all_probs = np.array(all_labels), np.array(all_preds), np.array(all_probs)
        acc = accuracy_score(all_labels, all_preds) * 100
        try: auc = roc_auc_score(all_labels, all_probs)
        except ValueError: auc = 0.5
        
        return {
            'loss': total_loss / len(val_loader),
            'acc': acc, 'auc': auc,
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
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
            'history': self.history,
            'dino_unfrozen': self.dino_unfrozen
        }
        
        if filename is None:
            filename = 'checkpoint_latest_dino_svd_crossattn.pt'
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
        
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'checkpoint_best_dino_svd_crossattn.pt')
            print(f"  Saved best model (ACC: {self.best_val_acc:.4f})")
    
    def _save_history(self):
        """Save training history to file."""
        history_path = self.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def train(self, model, train_loader, val_loader, test_loader=None, resume_from=None):
        model = model.to(self.device)
        self._setup_model_for_training(model)
        
        print("\nModel Information:")
        model.print_trainable_params()
        
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer, len(train_loader) * self.training_config.num_epochs)
        criterion = self._create_criterion(train_loader)
        
        print(f"\nStarting training on {self.device}...")
        for epoch in range(self.current_epoch, self.training_config.num_epochs):
            self.current_epoch = epoch
            
            if self.finetune_dino and not self.dino_unfrozen and epoch >= self.unfreeze_after:
                print(f"\nUnfreezing DinoSVD at epoch {epoch + 1}")
                model.unfreeze_dino()
                self.dino_unfrozen = True
                optimizer = self._create_optimizer_with_dino(model)
                scheduler = self._create_scheduler(optimizer, len(train_loader) * (self.training_config.num_epochs - epoch))
                self.epochs_without_improvement = 0
                model.print_trainable_params()
            
            train_metrics = self._train_epoch(model, train_loader, optimizer, criterion, scheduler, epoch)
            val_metrics = self._validate(model, val_loader, criterion)
            
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
                
            is_best = val_metrics['acc'] > self.best_val_acc
            if is_best:
                self.best_val_auc, self.best_val_acc, self.best_val_loss = val_metrics['auc'], val_metrics['acc'], val_metrics['loss']
                self.epochs_without_improvement = 0
                torch.save({'model_state_dict': model.state_dict()}, self.checkpoint_dir / 'checkpoint_best_crossattn.pt')
            else:
                self.epochs_without_improvement += 1

            self._save_checkpoint(model, optimizer, scheduler, is_best)
            self._save_history()

            print(f"Epoch {epoch + 1}/{self.training_config.num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['acc']:.2f}%")
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
                
            if self.training_config.early_stopping and self.epochs_without_improvement >= self.training_config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
                
        if test_loader:
            model.load_state_dict(torch.load(self.checkpoint_dir / 'checkpoint_best_crossattn.pt', map_location=self.device)['model_state_dict'])
            test_metrics = self._validate(model, test_loader, criterion)
            print(f"\nTest AUC: {test_metrics['auc']:.4f} Acc: {test_metrics['acc']:.2f}%")
        
        return {'best_val_auc': self.best_val_auc, 'best_val_acc': self.best_val_acc}

def create_config(args):
    dataset_root = Path("Datasets") if args.dataset == "combined" else Path(f"Datasets/{'FF' if args.dataset == 'ff' else 'Celeb-DF-v2'}")
    return Config(
        data=DataConfig(dataset_type=args.dataset, dataset_mode="combined" if args.dataset == "combined" else "single", dataset_root=dataset_root, batch_size=args.batch_size, num_workers=args.num_workers, use_cache=True, preload_cache=True),
        training=TrainingConfig(num_epochs=args.epochs, learning_rate=args.lr, weight_decay=args.weight_decay, optimizer=args.optimizer, scheduler=args.scheduler, early_stopping=args.early_stopping, early_stopping_patience=args.patience, checkpoint_dir=Path(args.checkpoint_dir), log_dir=Path(args.log_dir), device=args.device, use_class_weights=False),
        preprocessing=PreprocessingConfig(frames_per_video=args.frames_per_video, use_cache=True),
        experiment_name="dino_svd_crossattn", seed=args.seed
    )

def main():
    args = parse_args()
    config = create_config(args)
    
    print("Loading data...")
    if args.dataset == "ff":
        train_loader, val_loader, test_loader = create_ff_dataloaders(root_dir=Path("Datasets/FF"), config=config, frames_per_video=args.frames_per_video, video_level=True)
    elif args.dataset == "celeb_df":
        train_loader, val_loader, test_loader = create_celeb_df_dataloaders(root_dir=Path("Datasets/Celeb-DF-v2"), config=config, frames_per_video=args.frames_per_video, video_level=True)
    else:
        train_loader, val_loader, test_loader = get_dataloaders(dataset_type="combined", dataset_configs=[{"name": "ff", "root_dir": "Datasets/FF", "weight": 1.0}, {"name": "celeb_df", "root_dir": "Datasets/Celeb-DF-v2", "weight": 1.0}], batch_size=args.batch_size, frames_per_video=args.frames_per_video, video_level=True)
    
    print("\nCreating model...")
    model = DinoSVD_HSI_CrossAttention_Model(
        num_classes=2,
        cross_attn_d_model=args.cross_attn_d_model,
        cross_attn_heads=args.cross_attn_heads,
        classifier_hidden_dim=args.classifier_hidden_dim,
        dropout=args.dropout,
        dino_model=args.dino_model,
        svd_rank=args.svd_rank,
    )
    
    trainer = CrossAttnTrainer(
        config=config, hsi_lr=args.hsi_lr, attn_lr=args.attn_lr,
        finetune_dino=args.finetune_dino, unfreeze_after=args.unfreeze_after, dino_lr=args.dino_lr
    )
    
    if not args.eval_only:
        trainer.train(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, resume_from=args.resume)

    # Load best checkpoint for final evaluation
    best_checkpoint = torch.load(
        trainer.checkpoint_dir / 'checkpoint_best_crossattn.pt',
        map_location=trainer.device
    )
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model = model.to(trainer.device)

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
