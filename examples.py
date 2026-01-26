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
    print("Dataset Loading Example")
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
    print("\n" + "=" * 50)
    print("Full Pipeline Example")
    print("=" * 50)
    
    from config import Config
    
    # Create configuration
    config = Config()
    
    # Modify as needed
    config.data.frames_per_video = 10
    config.data.batch_size = 32
    config.data.manipulation_types = ["Deepfakes", "Face2Face"]
    
    config.model.model_name = "simple_cnn_large"
    config.model.dropout_rate = 0.5
    
    config.training.num_epochs = 30
    config.training.learning_rate = 1e-4
    
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
    
    # 1. Create preprocessing pipeline from config
    from preprocessing import create_pipeline_from_config
    pipeline = create_pipeline_from_config(config)
    
    # 2. Create dataloaders
    from data import create_ff_dataloaders
    train_loader, val_loader, test_loader = create_ff_dataloaders(
        root_dir=config.data.dataset_root,
        config=config
    )
    
    # 3. Create model
    from models import get_model
    model = get_model(config.model.model_name, num_classes=config.model.num_classes)
    
    # # 4. Training loop
    from training import create_trainer
    trainer = create_trainer(config)
    # results = trainer.train(model, train_loader, val_loader)
    
    # 5. Evaluate on test set
    # Load best model and evaluate
    model = trainer.load_best_model(model)
    test_results = trainer.evaluate(model, test_loader)


if __name__ == "__main__":
    # print("Deepfake Detection Pipeline - Usage Examples")
    # print("=" * 50)
    
    # example_preprocessing()
    # example_dataset_loading()
    # example_model_creation()
    example_full_pipeline()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("=" * 50)
