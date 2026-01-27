"""
Example usage of the deepfake detection pipeline.

This script demonstrates how to:
1. Configure the pipeline
2. Preprocess videos
3. Load data
4. Create and use models
"""

import sys
from pathlib import Path
import os

# Fix TensorFlow/Keras 3 compatibility issue with retinaface
# Must be set before importing tensorflow
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hide INFO + WARNING

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def example_preprocessing():
    """Example of using the preprocessing pipeline."""
    print("=" * 50)
    print("Preprocessing Pipeline Example")
    print("=" * 50)
    
    from preprocessing import PreprocessingPipeline
    
    # Create pipeline with custom settings
    pipeline = PreprocessingPipeline(
        num_frames=10,
        sampling_strategy="uniform",
        face_detector="retinaface",
        detection_threshold=0.9,
        output_size=(224, 224),
        bbox_enlargement=1.3,
        align_faces=True
    )
    
    print("Pipeline created with:")
    print(f"  - Frames per video: 10")
    print(f"  - Sampling strategy: uniform")
    print(f"  - Face detector: RetinaFace")
    print(f"  - Output size: 224x224")
    print(f"  - Bounding box enlargement: 1.3x")
    
    # Example: Process a video (uncomment with actual video path)
    video_path = "Datasets/FF/original_sequences/youtube/c23/videos/000.mp4"
    faces = pipeline.process_video(video_path)
    print(f"Extracted {len(faces)} faces from video")
    
    return pipeline


def example_dataset_loading():
    """Example of loading the FF++ dataset."""
    print("\n" + "=" * 50)
    print("Dataset Loading Example (Single Dataset)")
    print("=" * 50)
    
    from config import Config
    from data import create_ff_dataloaders
    
    # Create config
    config = Config()
    config.data.frames_per_video = 10
    config.data.batch_size = 32
    
    print("Configuration:")
    print(f"  - Dataset root: {config.data.dataset_root}")
    print(f"  - Frames per video: {config.data.frames_per_video}")
    print(f"  - Batch size: {config.data.batch_size}")
    print(f"  - Manipulation types: {config.data.manipulation_types}")
    
    # Note: Actual data loading requires the dataset to be present
    train_loader, val_loader, test_loader = create_ff_dataloaders(
        root_dir="Datasets/FF",
        batch_size=32,
        frames_per_video=10
    )
    
    print(f"Dataloaders created:\n  - Train batches: {len(train_loader)}\n  - Val batches: {len(val_loader)}\n  - Test batches: {len(test_loader)}")
    return train_loader, val_loader, test_loader


def example_combined_dataset_loading():
    """Example of loading combined FF++ and Celeb-DF datasets."""
    print("\n" + "=" * 50)
    print("Combined Dataset Loading Example")
    print("=" * 50)
    
    from data import (
        create_combined_dataloaders,
        get_dataloaders,
        DatasetConfig,
        DatasetRegistry
    )
    
    # Show available datasets
    print("\nRegistered datasets:")
    for name in DatasetRegistry.list_datasets():
        print(f"  - {name}")
    
    # Method 1: Using DatasetConfig objects (more explicit)
    print("\n--- Method 1: Using DatasetConfig objects ---")
    dataset_configs = [
        DatasetConfig(
            name="ff",
            root_dir="Datasets/FF",
            weight=1.0,
            manipulation_types=["Deepfakes", "Face2Face", "FaceSwap"]
        ),
        DatasetConfig(
            name="celeb_df",
            root_dir="Datasets/Celeb-DF-v2",
            weight=1.5  # Higher weight = more samples from this dataset
        ),
    ]
    
    train_loader, val_loader, test_loader = create_combined_dataloaders(
        dataset_configs=dataset_configs,
        batch_size=32,
        frames_per_video=10,
        preload_cache=True
    )
    
    print(f"\nCombined dataloaders created:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    
    # Method 2: Using dictionaries (more concise)
    print("\n--- Method 2: Using dictionaries ---")
    configs_as_dicts = [
        {"name": "ff", "root_dir": "Datasets/FF"},
        {"name": "celeb_df", "root_dir": "Datasets/Celeb-DF-v2"},
    ]
    
    # Using the universal get_dataloaders function
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_type="combined",
        dataset_configs=configs_as_dicts,
        batch_size=32
    )
    
    return train_loader, val_loader, test_loader


def example_combined_dataset_with_config():
    """Example using Config class for combined dataset training."""
    print("\n" + "=" * 50)
    print("Combined Dataset with Config Example")
    print("=" * 50)
    
    from config import Config, DatasetSourceConfig
    from data import get_dataloaders
    from pathlib import Path
    
    # Create config for combined training
    config = Config()
    
    # Set to combined mode
    config.data.dataset_mode = "combined"
    
    # Configure dataset sources
    config.data.dataset_sources = [
        DatasetSourceConfig(
            name="ff",
            root_dir=Path("Datasets/FF"),
            weight=1.0,
            enabled=True,
            extra_kwargs={
                "manipulation_types": ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
            }
        ),
        DatasetSourceConfig(
            name="celeb_df",
            root_dir=Path("Datasets/Celeb-DF-v2"),
            weight=1.2,
            enabled=True
        ),
        # Example: Disabled dataset (won't be loaded)
        # DatasetSourceConfig(
        #     name="dfdc",
        #     root_dir=Path("Datasets/DFDC"),
        #     weight=1.0,
        #     enabled=False  # Disabled
        # ),
    ]
    
    print("Combined training configuration:")
    print(f"  - Mode: {config.data.dataset_mode}")
    print(f"  - Dataset sources:")
    for src in config.data.dataset_sources:
        status = "enabled" if src.enabled else "disabled"
        print(f"    - {src.name}: {src.root_dir} (weight={src.weight}, {status})")
    
    # Get enabled dataset configs
    dataset_configs = config.data.get_enabled_dataset_configs()
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_type="combined",
        dataset_configs=dataset_configs,
        batch_size=config.data.batch_size,
        frames_per_video=config.data.frames_per_video,
        use_dataset_weights=config.data.use_dataset_weights
    )
    
    return train_loader, val_loader, test_loader


def example_model_creation():
    """Example of creating models."""
    print("\n" + "=" * 50)
    print("Model Creation Example")
    print("=" * 50)
    
    try:
        import torch
        from models import SimpleCNN, SimpleCNNLarge, get_model
        
        # Create simple CNN
        model_small = SimpleCNN(
            num_classes=2,
            hidden_dims=[64, 128, 256, 512],
            dropout_rate=0.5
        )
        
        # Create larger CNN
        model_large = SimpleCNNLarge(
            num_classes=2,
            base_channels=64,
            dropout_rate=0.5
        )
        
        # Print model info
        print("\nSimpleCNN:")
        info = model_small.get_model_info()
        for key, value in info.items():
            print(f"  - {key}: {value}")
        
        print("\nSimpleCNNLarge:")
        info = model_large.get_model_info()
        for key, value in info.items():
            print(f"  - {key}: {value}")
        
        # Test forward pass
        print("\nTest forward pass:")
        dummy_input = torch.randn(4, 3, 224, 224)
        
        output_small = model_small(dummy_input)
        print(f"  SimpleCNN output shape: {output_small.shape}")
        
        output_large = model_large(dummy_input)
        print(f"  SimpleCNNLarge output shape: {output_large.shape}")
        
        # Using factory function
        print("\nUsing get_model factory:")
        model = get_model("simple_cnn_large", num_classes=2)
        print(f"  Created: {model.__class__.__name__}")
        
    except ImportError:
        print("PyTorch not available. Install with: pip install torch")


def example_full_pipeline():
    """Example of the full pipeline from config to model."""
    
    from config import Config, DatasetSourceConfig
    from data import create_ff_dataloaders, get_dataloaders
    
    # Create configuration
    config = Config()
    
    # Modify as needed
    config.data.frames_per_video = 10
    config.data.batch_size = 32
    
    config.model.model_name = "dino_model"
    config.model.dropout_rate = 0.3

    config.training.optimizer = "adamw"
    config.training.weight_decay = 1e-4
    
    config.training.num_epochs = 30
    config.training.learning_rate = 1e-4

    # Dataset Configuration

    # Only FF
    # config.data.dataset_mode = "single"
    # config.data.dataset_type = "ff"
    # config.data.dataset_root = Path("Datasets/FF")

    # train_loader, val_loader, test_loader = create_ff_dataloaders(
    #     root_dir=config.data.dataset_root,
    #     config=config
    # )

    # Only Celeb-DF
    # config.data.dataset_mode = "single"
    # config.data.dataset_type = "celeb_df"
    # config.data.dataset_root = Path("Datasets/Celeb-DF-v2")

    # train_loader, val_loader, test_loader = get_dataloaders(
    #     dataset_type="celeb_df",
    #     root_dir="Datasets/Celeb-DF-v2",
    #     batch_size=config.data.batch_size
    # )

    # Both
    config.data.dataset_mode = "combined"
    config.data.dataset_sources = [
        DatasetSourceConfig(
            name="ff",
            root_dir=Path("Datasets/FF"),
            weight=1.0,
            enabled=True
        ),
        DatasetSourceConfig(
            name="celeb_df",
            root_dir=Path("Datasets/Celeb-DF-v2"),
            weight=1.0,
            enabled=True
        ),
    ]

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_type="combined",
        dataset_configs=config.data.get_enabled_dataset_configs(),
        batch_size=config.data.batch_size
    )
        
    print("Full pipeline configuration:")
    print(f"  Data:")
    print(f"    - Frames per video: {config.data.frames_per_video}")
    print(f"    - Batch size: {config.data.batch_size}")
    print(f"    - Manipulations: {config.data.manipulation_types}")
    
    print(f"  Model:")
    print(f"    - Name: {config.model.model_name}")
    print(f"    - Dropout: {config.model.dropout_rate}")
    
    print(f"  Training:")
    print(f"    - Epochs: {config.training.num_epochs}")
    print(f"    - Learning rate: {config.training.learning_rate}")
    
    # 1. Create model
    from models import get_model
    config.model.dropout_rate = 0.3
    config.model.hidden_dims = [256]

    model = get_model(
        config.model.model_name, 
        num_classes=config.model.num_classes, 
        dropout=config.model.dropout_rate,
        hidden_dims=config.model.hidden_dims
    )
    
    # 2. Training
    from training import create_trainer

    config.training.checkpoint_file_name = f"checkpoint_best_{config.model.model_name}.pt"

    trainer = create_trainer(config)
    _ = trainer.train(model, train_loader, val_loader)
    
    # 3. Evaluate on test set
    from training import Evaluator

    evaluator = Evaluator(
        model_name=config.model.model_name,
        weights_path=config.training.checkpoint_dir / config.training.checkpoint_file_name,
        # dataset_name="ff",
        dataloader=test_loader,  # uncomment if testing from the same dataset as training
        batch_size=config.data.batch_size,
    )

    # model = trainer.load_best_model(model)
    metrics = evaluator.evaluate()
    
    # Print results
    evaluator.print_results(metrics)


if __name__ == "__main__":
    # print("Deepfake Detection Pipeline - Usage Examples")
    # print("=" * 50)
    
    # Single dataset examples
    # example_preprocessing()
    # example_dataset_loading()
    # example_model_creation()
    
    # Combined/multi-dataset examples
    # example_combined_dataset_loading()
    # example_combined_dataset_with_config()
    
    # Full pipeline example
    example_full_pipeline()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("=" * 50)
