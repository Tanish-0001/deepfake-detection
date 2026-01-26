"""
Configuration classes for the deepfake detection pipeline.
Modify these settings to change data processing, model, or training parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class PreprocessingConfig:
    """
    Configuration for preprocessing pipeline.
    
    These settings control how videos are preprocessed and cached.
    Run preprocessing once with: python -m preprocessing.preprocess_dataset --all
    """
    
    # Face extraction settings
    face_detector: str = "retinaface"  # 'retinaface', 'mtcnn', 'dlib'
    face_detection_threshold: float = 0.9
    
    # Face cropping and resizing
    output_size: Tuple[int, int] = (224, 224)
    bbox_enlargement_factor: float = 1.3  # How much to enlarge the bounding box
    align_faces: bool = True  # Whether to align faces based on landmarks
    
    # Frame sampling
    frames_per_video: int = 10
    frame_sampling_strategy: str = "uniform"  # 'uniform', 'random', 'first_n'
    
    # Caching - ENABLE THIS FOR FASTER TRAINING
    use_cache: bool = True  # Use cached preprocessed faces
    cache_dir: Optional[Path] = None  # None = use dataset_root/cache
    skip_preprocessing_if_cached: bool = True  # Skip face detection if cache exists
    
    # Multi-processing (for batch preprocessing)
    preprocessing_device: str = "cuda"  # Device for face detection
    num_preprocessing_workers: int = 1  # Workers for batch preprocessing
    
    def get_cache_dir(self, dataset_root: Path) -> Path:
        """Get cache directory, defaulting to dataset_root/cache."""
        if self.cache_dir is not None:
            return self.cache_dir
        return dataset_root / "cache"


@dataclass
class DataConfig:
    """Configuration for data preprocessing and loading."""
    
    # Dataset paths
    dataset_root: Path = Path("Datasets/FF")
    train_json: str = "train.json"
    val_json: str = "val.json"
    test_json: str = "test.json"
    
    # Frame sampling (legacy - use preprocessing config for new projects)
    frames_per_video: int = 10
    frame_sampling_strategy: str = "uniform"  # 'uniform', 'random', 'first_n'
    
    # Face extraction (legacy - use preprocessing config for new projects)
    face_detector: str = "retinaface"  # 'retinaface', 'mtcnn', 'dlib'
    face_detection_threshold: float = 0.9
    
    # Face cropping and resizing (legacy - use preprocessing config for new projects)
    output_size: Tuple[int, int] = (224, 224)
    bbox_enlargement_factor: float = 1.3  # How much to enlarge the bounding box
    
    # Manipulation types to include (for FF++ dataset)
    manipulation_types: List[str] = field(default_factory=lambda: [
        "Deepfakes",
        "Face2Face", 
        "FaceSwap",
        "NeuralTextures",
        "FaceShifter"
    ])
    
    # Compression level (for FF++ dataset)
    compression: str = "c23"  # 'c23', 'c40', 'raw'
    
    # Data augmentation (for training)
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    
    # Preprocessing cache - SET TO TRUE FOR FASTER TRAINING
    use_cache: bool = True  # Load from cache instead of preprocessing on-the-fly
    cache_dir: Path = Path("cache/preprocessed")  # Legacy, prefer dataset_root/cache
    preload_cache: bool = True  # Preload all cache files into memory (faster training)
    
    # DataLoader settings
    batch_size: int = 32
    num_workers: int = 0  # 0 is optimal when preload_cache=True (data already in RAM)
    pin_memory: bool = True
    prefetch_factor: int = 2  # Number of batches to prefetch per worker


@dataclass
class ModelConfig:
    """Configuration for the model architecture."""
    
    # Model type
    model_name: str = "simple_cnn"  # 'simple_cnn', 'efficientnet', 'xception', etc.
    
    # Input configuration
    input_channels: int = 3
    input_size: Tuple[int, int] = (224, 224)
    
    # Model architecture (for simple_cnn)
    num_classes: int = 2  # Binary classification: real vs fake
    dropout_rate: float = 0.5
    
    # Feature dimensions
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    
    # Pretrained weights (for transfer learning models)
    pretrained: bool = True
    freeze_backbone: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training parameters
    num_epochs: int = 25
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Optimizer
    optimizer: str = "adam"  # 'adam', 'adamw', 'sgd'
    
    # Learning rate scheduler
    scheduler: str = "cosine"  # 'cosine', 'step', 'plateau', 'none'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.1
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_best_only: bool = True
    
    # Logging
    log_dir: Path = Path("logs")
    log_interval: int = 10
    
    # Device
    device: str = "cuda"  # 'cuda', 'cpu', 'mps'
    mixed_precision: bool = False


@dataclass
class Config:
    """Main configuration class combining all configs."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    
    # Experiment settings
    experiment_name: str = "deepfake_detection"
    seed: int = 42
    
    def __post_init__(self):
        """Create necessary directories."""
        self.data.cache_dir.mkdir(parents=True, exist_ok=True)
        self.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.training.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Sync preprocessing config with data config for backwards compatibility
        self._sync_preprocessing_settings()
    
    def _sync_preprocessing_settings(self):
        """Sync preprocessing config with data config."""
        # Use preprocessing config as source of truth, update data config
        self.data.frames_per_video = self.preprocessing.frames_per_video
        self.data.frame_sampling_strategy = self.preprocessing.frame_sampling_strategy
        self.data.face_detector = self.preprocessing.face_detector
        self.data.face_detection_threshold = self.preprocessing.face_detection_threshold
        self.data.output_size = self.preprocessing.output_size
        self.data.bbox_enlargement_factor = self.preprocessing.bbox_enlargement_factor
        self.data.use_cache = self.preprocessing.use_cache
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create config from dictionary."""
        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        preprocessing_config = PreprocessingConfig(**config_dict.get("preprocessing", {}))
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            preprocessing=preprocessing_config,
            experiment_name=config_dict.get("experiment_name", "deepfake_detection"),
            seed=config_dict.get("seed", 42)
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)
