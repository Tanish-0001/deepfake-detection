# Deepfake Detection Pipeline

A modular pipeline for deepfake detection using computer vision models.

## Project Structure

```
DD/
├── config/                 # Configuration module
│   ├── __init__.py
│   └── config.py           # Data, model, and training configurations
│
├── preprocessing/          # Data preprocessing module
│   ├── __init__.py
│   ├── face_extractor.py    # RetinaFace-based face detection and extraction
│   ├── frame_sampler.py      # Video frame sampling strategies
│   ├── transforms.py         # Image augmentation and normalization
│   ├── pipeline.py           # Complete preprocessing pipeline
|   └── preprocess_dataset.py  # preprocesses videos and saves output to speed up training
│
├── data/                    # Dataset module
│   ├── __init__.py
│   ├── ff_dataset.py          # FaceForensics++ dataset implementation
│   ├── dataloader.py          # DataLoader utilities
|   └── generate_ff_splits.py  # Generates paths and labels from train/val/test.json
│
├── models/                 # Model definitions
│   ├── __init__.py
│   ├── base_model.py      # Abstract base class for models
│   └── simple_cnn.py      # Simple CNN placeholder model
│
├── Datasets/               # Dataset storage
│   └── FF/                 # FaceForensics++ dataset
│       ├── train.json
│       ├── val.json
│       ├── test.json
│       ├── original_sequences/
│       └── manipulated_sequences/
│
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Configuration

```python
from config import Config, DataConfig, ModelConfig

# Create default config
config = Config()

# Or customize
config.preprocessing.frames_per_video = 15
config.data.batch_size = 64
config.model.dropout_rate = 0.3
```

### 2. Data Preprocessing

```python
from preprocessing import PreprocessingPipeline

# Create preprocessing pipeline
pipeline = PreprocessingPipeline(
    num_frames=10,
    sampling_strategy="uniform",
    face_detector="retinaface",
    output_size=(224, 224),
    bbox_enlargement=1.3
)

# Process a single video
faces = pipeline.process_video("path/to/video.mp4")

# Process with detailed info
result = pipeline.process_video_with_info("path/to/video.mp4")
```

### 3. Dataset Loading

```python
from data import create_ff_dataloaders

# Create dataloaders for FaceForensics++
train_loader, val_loader, test_loader = create_ff_dataloaders(
    root_dir="Datasets/FF",
    batch_size=32,
    num_workers=4,
    frames_per_video=10
)

# Iterate through data
for images, labels in train_loader:
    # images: [B, C, H, W] tensor
    # labels: [B] tensor (0=real, 1=fake)
    pass
```

### 4. Model Definition

```python
from models import SimpleCNN, SimpleCNNLarge, get_model

# Create simple CNN
model = SimpleCNN(num_classes=2, dropout_rate=0.5)

# Or create larger model
model = SimpleCNNLarge(num_classes=2, base_channels=64)

# Or use factory function
model = get_model("simple_cnn_large", num_classes=2)

# Get model info
print(model.get_model_info())
```

## Preprocessing Pipeline

The preprocessing pipeline performs the following steps:

1. **Frame Sampling**: Sample X frames from each video
   - Strategies: `uniform`, `random`, `first_n`, `keyframes`

2. **Face Detection**: Detect faces using RetinaFace
   - High accuracy face detection
   - Returns bounding boxes and 5-point landmarks

3. **Face Alignment**: Align faces based on eye landmarks
   - Corrects for rotation

4. **Bounding Box Enlargement**: Enlarge detected bbox
   - Default factor: 1.3x

5. **Crop and Resize**: Crop face region and resize
   - Default size: 224x224

## Modifying Components

### Adding a New Face Detector

1. Create a new class in `preprocessing/face_extractor.py`:

```python
class MyFaceExtractor(FaceExtractor):
    def detect_faces(self, image):
        # Your implementation
        pass
    
    def extract_face(self, image, output_size, enlargement_factor):
        # Your implementation
        pass
```

2. Update the factory function in `face_extractor.py`

### Adding a New Model

1. Create a new file in `models/`:

```python
from .base_model import BaseModel

class MyModel(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__(num_classes)
        # Define layers
    
    def forward(self, x):
        # Forward pass
        return logits
```

2. Register in `models/__init__.py` and `models/simple_cnn.py`

### Adding a New Dataset

1. Create a new file in `data/`:

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        # Initialize
        pass
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Return (image, label)
        pass
```

## Configuration Options

### DataConfig

- `frames_per_video`: Number of frames to sample (default: 10)
- `frame_sampling_strategy`: 'uniform', 'random', 'first_n' (default: 'uniform')
- `face_detector`: 'retinaface' (default: 'retinaface')
- `output_size`: Target image size (default: (224, 224))
- `bbox_enlargement_factor`: Bounding box enlargement (default: 1.3)
- `batch_size`: Batch size for training (default: 32)

### ModelConfig

- `model_name`: Model architecture (default: 'simple_cnn')
- `num_classes`: Number of output classes (default: 2)
- `dropout_rate`: Dropout rate (default: 0.5)
- `hidden_dims`: Hidden dimensions for CNN (default: [64, 128, 256, 512])

## License

MIT License
