"""
Trainer module for deepfake detection models.

This module provides a Trainer class that handles the training loop,
validation, checkpointing, and logging for deepfake detection models.
"""

import time
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import asdict
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import classification_report


class Trainer:
    """
    Trainer class for training deepfake detection models.
    
    Handles:
    - Training loop with validation
    - Learning rate scheduling
    - Early stopping
    - Checkpointing (save/load)
    - Logging and metrics tracking
    - Mixed precision training
    """
    
    def __init__(self, config):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: Config object containing training, data, and model configurations
        """
        
        self.config = config
        self.training_config = config.training
        
        # Set device
        self.device = self._setup_device()
        
        # Initialize tracking variables
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
                
        # Save config
        self._save_config()
        
        # Set random seed for reproducibility
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
        # Save config
        config_path = self.training_config.log_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
    
    def _create_optimizer(self, model: nn.Module, unfreeze_backbone: bool = False) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = self.training_config.optimizer.lower()
        lr = self.training_config.learning_rate
        weight_decay = self.training_config.weight_decay

        # Build parameter groups based on model type
        parameters = []
        
        if unfreeze_backbone and hasattr(model, 'get_backbone_params'):
            parameters.append({"params": model.get_backbone_params(), "lr": 1e-5})
        
        # Add temporal transformer parameters if present (for DinoTemporalModel)
        if hasattr(model, 'get_temporal_params'):
            parameters.append({"params": model.get_temporal_params(), "lr": lr})
        
        # Add other temporal model components (cls_token, positional_encoding, feature_projection)
        if hasattr(model, 'cls_token'):
            parameters.append({"params": [model.cls_token], "lr": lr})
        if hasattr(model, 'positional_encoding'):
            parameters.append({"params": [model.positional_encoding], "lr": lr})
        if hasattr(model, 'feature_projection') and hasattr(model.feature_projection, 'parameters'):
            # Only add if it's not Identity
            proj_params = list(model.feature_projection.parameters())
            if proj_params:
                parameters.append({"params": proj_params, "lr": lr})
        
        # Add classifier parameters
        parameters.append({"params": model.classifier.parameters(), "lr": lr})

        
        if optimizer_name == "adam":
            return optim.Adam(parameters, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            return optim.AdamW(parameters, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            return optim.SGD(parameters, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self, optimizer: optim.Optimizer, num_training_steps: int):
        """Create learning rate scheduler based on configuration."""
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
    
    def _create_criterion(self, train_loader: DataLoader = None) -> nn.Module:
        """
        Create loss function, optionally with class weights for imbalanced data.
        
        Args:
            train_loader: Training dataloader to compute class weights from
            
        Returns:
            CrossEntropyLoss with or without class weights
        """
        if self.training_config.use_class_weights and train_loader is not None:
            # Count samples per class
            labels = train_loader.dataset.labels
            num_real = labels.count(0)
            num_fake = labels.count(1)
            total = num_real + num_fake
            
            # Weight inversely proportional to class frequency
            # More weight to minority class
            weights = torch.tensor([
                num_fake / total,  # weight for Real (0) - minority gets higher weight
                num_real / total,  # weight for Fake (1)
            ], device=self.device)
            
            print(f"Using class-weighted loss:")
            print(f"  Real (0): {num_real} samples, weight={weights[0]:.4f}")
            print(f"  Fake (1): {num_fake} samples, weight={weights[1]:.4f}")
            
            return nn.CrossEntropyLoss(weight=weights)
        
        return nn.CrossEntropyLoss()
    
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            resume_from: Path to checkpoint to resume from (optional)
        Returns:
            Dictionary containing training history and best metrics
        """
        # Move model to device
        model = model.to(self.device)
        
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
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.training_config.num_epochs):
            self.current_epoch = epoch

            if self.training_config.unfreeze_backbone and epoch == (self.training_config.unfreeze_backbone_after_epochs):
                print("\nUnfreezing backbone for fine-tuning.")
                model.unfreeze_backbone()
                remaining_steps = (self.training_config.num_epochs - epoch) * len(train_loader)

                optimizer = self._create_optimizer(model, unfreeze_backbone=True)
                optimizer.zero_grad(set_to_none=True)

                scheduler = self._create_scheduler(
                    optimizer,
                    num_training_steps=remaining_steps
                )
                            
            # Training phase
            train_loss, train_acc = self._train_epoch(
                model, train_loader, optimizer, criterion, scheduler
            )
            
            # Validation phase
            val_loss, val_acc = self._validate(model, val_loader, criterion)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Update scheduler if using ReduceLROnPlateau/StepLR
            if scheduler is not None and self.training_config.scheduler.lower() in ["plateau", "step"]:
                scheduler.step(val_loss)
            
            # Print progress
            print(f"Epoch [{epoch+1}/{self.training_config.num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.2e}")
            
            # Check for improvement
            improved = False
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                improved = True
            
            # Save checkpoint
            if improved:
                self.epochs_without_improvement = 0
                self._save_checkpoint(model, optimizer, scheduler, is_best=True)
                print(f"  ✓ New best model saved (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f})")
            else:
                self.epochs_without_improvement += 1
            
            if self.training_config.save_individual_epoch:
                self._save_checkpoint(model, optimizer, scheduler, is_best=improved)
            
            # Early stopping
            if self.training_config.early_stopping:
                if self.epochs_without_improvement >= self.training_config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
        
        total_time = time.time() - start_time
        
        # Save final history
        self._save_history()
        
        print("-" * 50)
        print(f"Training completed in {total_time/60:.2f} minutes")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"Best Validation Accuracy: {self.best_val_acc:.4f}")
        
        return {
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'total_time': total_time
        }
    
    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scheduler
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, total=len(train_loader), file=sys.stdout, dynamic_ncols=True)

        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Update scheduler (except for ReduceLROnPlateau)
            if scheduler is not None and self.training_config.scheduler.lower() not in ["plateau", "step", "none"]:
                scheduler.step()
            
            # Calculate metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Log progress
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0

        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(targets.cpu().numpy())
        
        print("\nValidation Classification Report:")
        print(classification_report(y_true, y_pred, digits=4))
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            model: PyTorch model to evaluate
            test_loader: DataLoader for test data
        
        Returns:
            Dictionary containing evaluation metrics
        """
        model = model.to(self.device)
        model.eval()
        
        criterion = self._create_criterion()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                probs = torch.softmax(outputs, dim=1)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake class
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        
        # Calculate additional metrics
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }
        
        print(f"\nEvaluation Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        
        return results
    
    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        is_best: bool = False
    ):
        """Save a checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': asdict(self.config)
        }
        
        # Save latest checkpoint
        checkpoint_path = self.training_config.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.training_config.checkpoint_dir / self.training_config.checkpoint_file_name
            torch.save(checkpoint, best_path)
        
        # Optionally save epoch checkpoint
        if self.training_config.save_individual_epoch:
            epoch_path = self.training_config.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch+1}.pt"
            torch.save(checkpoint, epoch_path)
    
    def _load_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        checkpoint_path: str
    ):
        """Load a checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        print(f"Resumed from epoch {self.current_epoch}")
    
    def _save_history(self):
        """Save training history to file."""
        history_path = self.training_config.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_best_model(self, model: nn.Module) -> nn.Module:
        """
        Load the best model from checkpoint.
        
        Args:
            model: Model instance to load weights into
        
        Returns:
            Model with loaded weights
        """
        best_path = self.training_config.checkpoint_dir / self.training_config.checkpoint_file_name
        
        if not best_path.exists():
            raise FileNotFoundError(f"No best checkpoint found at {best_path}")
        
        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
        print(f"  Val Loss: {checkpoint['best_val_loss']:.4f}")
        print(f"  Val Acc: {checkpoint['best_val_acc']:.4f}")

        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint['best_val_loss']
        
        return model
