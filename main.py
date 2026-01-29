"""
Main
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


def main():
    """Example of the full pipeline from config to model."""
    
    from config import Config, DatasetSourceConfig
    from data import create_ff_dataloaders, get_dataloaders
    
    # Create configuration
    config = Config()
    
    # Modify as needed
    config.data.frames_per_video = 10
    config.data.batch_size = 8
    
    config.model.model_name = "dino_temporal_model"
    config.model.dropout_rate = 0.3

    config.training.optimizer = "adamw"
    config.training.weight_decay = 1e-4
    
    config.training.num_epochs = 30
    config.training.learning_rate = 1e-4

    config.training.save_individual_epoch = False

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
        batch_size=config.data.batch_size,
        frames_per_video=config.data.frames_per_video,
        video_level=True,
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
        # hidden_dims=config.model.hidden_dims,
        num_transformer_layers=4,
        max_seq_length=10,  # Match frames_per_video

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
        weights_path=config.training.checkpoint_dir / config.experiment_name / config.training.checkpoint_file_name,
        dataset_name="combined",
        dataloader=test_loader,  # uncomment if testing from the same dataset as training
        batch_size=config.data.batch_size,
        config=config,
        model_kwargs={
            "num_classes": config.model.num_classes,
            "dropout": config.model.dropout_rate,
            "num_transformer_layers": 4,
            "max_seq_length": config.data.frames_per_video,  # Must match training
        }
    )

    # model = trainer.load_best_model(model)
    metrics = evaluator.evaluate()
    
    # Print results
    evaluator.print_results(metrics)


if __name__ == "__main__":
    main()
