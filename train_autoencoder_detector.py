#!/usr/bin/env python3
"""
Train Autoencoder-based Deepfake Detector.

This script trains an autoencoder on real images only, then uses the
intervention cost (reconstruction error) to detect deepfakes.

Usage:
    # Train on single dataset
    python train_autoencoder_detector.py --dataset ff
    python train_autoencoder_detector.py --dataset celeb_df
    
    # Train on combined datasets (FF++ and Celeb-DF)
    python train_autoencoder_detector.py --dataset combined
    python train_autoencoder_detector.py --dataset combined --ff_weight 1.0 --celeb_df_weight 1.5
    
    # With additional options
    python train_autoencoder_detector.py --dataset ff --epochs 50 --batch_size 64
"""

import argparse
from pathlib import Path

import torch

from config.config import Config, DataConfig, TrainingConfig, PreprocessingConfig
from models import AutoencoderDetector
from data import (
    FFDataset,
    CelebDFDataset,
    RealOnlyDataset,
    create_dataloaders,
    CombinedDeepfakeDataset,
    DatasetConfig as CombinedDatasetConfig,
    create_combined_dataset
)
from data.dataloader import create_ff_dataloaders, create_celeb_df_dataloaders, create_combined_dataloaders
from training.autoencoder_trainer import AutoencoderTrainer
from training.autoencoder_evaluator import AutoencoderEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Autoencoder-based Deepfake Detector"
    )
    
    # Dataset settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="combined",
        choices=["ff", "celeb_df", "combined"],
        help="Dataset to train on (ff, celeb_df, or combined for both)"
    )
    parser.add_argument(
        "--ff_weight",
        type=float,
        default=1.0,
        help="Sampling weight for FF++ dataset (only used with --dataset combined)"
    )
    parser.add_argument(
        "--celeb_df_weight",
        type=float,
        default=1.0,
        help="Sampling weight for Celeb-DF dataset (only used with --dataset combined)"
    )
    
    # Model settings
    parser.add_argument(
        "--bottleneck_dim",
        type=int,
        default=16,  # Smaller for better compression
        help="Bottleneck dimension for autoencoder (smaller = more compression)"
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[],
        help="Hidden layer dimensions for encoder/decoder"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate"
    )
    parser.add_argument(
        "--intermediate_layers",
        type=int,
        default=1,
        help="Number of intermediate backbone layers to extract (from the end)"
    )
    parser.add_argument(
        "--layer_index",
        type=int,
        default=-1,
        help="Which layer to use (0=earliest extracted, -1=final layer)"
    )
    
    # Autoencoder architecture settings
    parser.add_argument(
        "--normalize_features",
        action="store_true",
        default=True,
        help="L2 normalize extracted features (default: False, keeps magnitude info)"
    )
    parser.add_argument(
        "--add_noise",
        action="store_true",
        default=False,
        help="Use denoising autoencoder (add noise during training)"
    )
    parser.add_argument(
        "--no_noise",
        action="store_true",
        help="Disable denoising (no noise added)"
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.1,
        help="Standard deviation of noise for denoising autoencoder"
    )
    
    # Margin loss settings
    parser.add_argument(
        "--margin_alpha",
        type=float,
        default=2.0,
        help="Alpha coefficient for margin: margin = mean(cost_real) + alpha * std(cost_real)"
    )
    parser.add_argument(
        "--margin_lambda",
        type=float,
        default=0.1,
        help="Lambda weight for margin loss: loss = loss_real + lambda * loss_margin"
    )
    parser.add_argument(
        "--no_margin_loss",
        action="store_true",
        help="Disable margin loss on fake samples (train on real only)"
    )
    
    # Training settings
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
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
        help="Number of data loading workers (0 when using preloaded cache)"
    )
    
    # Paths
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
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


def create_config(args) -> Config:
    """Create configuration from arguments."""
    # Determine dataset root based on dataset type
    if args.dataset == "combined":
        dataset_root = Path("Datasets")  # Parent directory for combined
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
        mixed_precision=False  # Disable for stability
    )
    
    preprocessing_config = PreprocessingConfig(
        frames_per_video=10,
        use_cache=True
    )
    
    return Config(
        data=data_config,
        training=training_config,
        preprocessing=preprocessing_config,
        experiment_name="autoencoder_detector",
        seed=args.seed
    )


def main():
    """Main training function."""
    args = parse_args()
    config = create_config(args)
    
    # Handle negation flags
    add_noise = args.add_noise and not args.no_noise
    use_margin_loss = not args.no_margin_loss
    
    print("="*60)
    print("AUTOENCODER-BASED DEEPFAKE DETECTOR")
    print("="*60)
    print(f"\nDataset: {args.dataset}")
    if args.dataset == "combined":
        print(f"  FF++ weight: {args.ff_weight}")
        print(f"  Celeb-DF weight: {args.celeb_df_weight}")
    print(f"Bottleneck dim: {args.bottleneck_dim}")
    print(f"Hidden dims: {args.hidden_dims}")
    print(f"Intermediate layers: {args.intermediate_layers}")
    print(f"Layer index: {args.layer_index} ({'earliest' if args.layer_index == 0 else 'final' if args.layer_index == -1 else args.layer_index})")
    print(f"Normalize features: {args.normalize_features}")
    print(f"Denoising AE: {add_noise} (noise_std={args.noise_std})")
    print(f"Margin loss: {use_margin_loss} (alpha={args.margin_alpha}, lambda={args.margin_lambda})")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Create model
    print("\n📦 Creating model...")
    model = AutoencoderDetector(
        num_classes=2,
        bottleneck_dim=args.bottleneck_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        intermediate_layers=args.intermediate_layers,
        layer_index=args.layer_index,
        normalize_features=args.normalize_features,
        add_noise=add_noise,
        noise_std=args.noise_std
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Create dataloaders
    print("\n📂 Loading datasets...")
    
    if args.dataset == "ff":
        train_loader, val_loader, test_loader = create_ff_dataloaders(
            root_dir=Path("Datasets/FF"),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            config=config
        )
    elif args.dataset == "celeb_df":
        train_loader, val_loader, test_loader = create_celeb_df_dataloaders(
            root_dir=Path("Datasets/Celeb-DF-v2"),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            config=config
        )
    elif args.dataset == "combined":
        # Create combined dataset configs
        dataset_configs = [
            {
                "name": "ff",
                "root_dir": "Datasets/FF",
                "weight": args.ff_weight
            },
            {
                "name": "celeb_df", 
                "root_dir": "Datasets/Celeb-DF-v2",
                "weight": args.celeb_df_weight
            }
        ]
        
        train_loader, val_loader, test_loader = create_combined_dataloaders(
            dataset_configs=dataset_configs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            config=config
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Wrap training dataset to only use real samples
    print("\n🔄 Filtering training data for real samples only...")
    train_dataset_real = RealOnlyDataset(train_loader.dataset, real_label=0)
    
    # Create new train loader with only real samples
    from torch.utils.data import DataLoader
    train_loader_real = DataLoader(
        train_dataset_real,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Training samples (real only): {len(train_dataset_real)}")
    print(f"Training samples (all): {len(train_loader.dataset)}")
    print(f"Validation samples (all): {len(val_loader.dataset)}")
    print(f"Test samples (all): {len(test_loader.dataset)}")
    
    # Create trainer with margin loss parameters
    trainer = AutoencoderTrainer(
        config,
        margin_alpha=args.margin_alpha,
        margin_lambda=args.margin_lambda,
        use_margin_loss=use_margin_loss
    )
    
    # Train
    print("\n🚀 Starting training...")
    results = trainer.train(
        model=model,
        train_loader_real=train_loader_real,
        val_loader_full=val_loader,
        test_loader=test_loader,
        resume_from=args.resume,
        train_loader_full=train_loader  # Pass full loader for margin loss
    )
    
    print("\n✅ Training complete!")
    print(f"Best AUROC: {results['best_auroc']:.4f}")
    
    if results['test_metrics']:
        print(f"\n📊 Test Results:")
        print(f"  AUROC: {results['test_metrics']['auroc']:.4f}")
        print(f"  Accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"  F1 Score: {results['test_metrics']['f1']:.4f}")

    # Create evaluator with matching model configuration
    model_kwargs = {
        'num_classes': 2,
        'bottleneck_dim': args.bottleneck_dim,
        'hidden_dims': args.hidden_dims,
        'dropout': args.dropout,
        'intermediate_layers': args.intermediate_layers,
        'layer_index': args.layer_index,
        'normalize_features': args.normalize_features,
        'add_noise': add_noise,
        'noise_std': args.noise_std
    }
    evaluator = AutoencoderEvaluator(
        "checkpoints/autoencoder_detector/checkpoint_best_autoencoder.pt", 
        dataloader=test_loader, 
        config=config,
        model_kwargs=model_kwargs
    )
    results = evaluator.evaluate()

    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
