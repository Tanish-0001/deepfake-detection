"""
Simple CNN model for binary deepfake classification.

This is a placeholder model that can be used as a baseline.
For better performance, consider using more advanced architectures like:
- EfficientNet
- XceptionNet
- Vision Transformer (ViT)
"""

from typing import List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None

from .base_model import BaseModel


class ConvBlock(nn.Module):
    """
    Convolutional block with Conv -> BatchNorm -> ReLU -> Optional MaxPool.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_pool: bool = True,
        pool_size: int = 2
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        
        if use_pool:
            layers.append(nn.MaxPool2d(pool_size))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class SimpleCNN(BaseModel):
    """
    Simple CNN for binary deepfake classification.
    
    Architecture:
    - 4 convolutional blocks with increasing channels
    - Global average pooling
    - Fully connected classifier
    
    Input: [B, 3, 224, 224]
    Output: [B, 2] (binary classification logits)
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 3,
        hidden_dims: List[int] = None,
        dropout_rate: float = 0.5,
        input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize SimpleCNN.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (3 for RGB)
            hidden_dims: List of hidden dimensions for conv layers
            dropout_rate: Dropout rate for classifier
            input_size: Input image size (height, width)
        """
        super().__init__(num_classes)
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]
        
        self.input_channels = input_channels
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        
        # Build convolutional layers
        self.features = self._build_features()
        
        # Calculate feature map size after conv layers
        self.feature_size = self._get_feature_size()
        
        # Build classifier
        self.classifier = self._build_classifier()
    
    def _build_features(self) -> nn.Sequential:
        """Build convolutional feature extractor."""
        layers = []
        in_channels = self.input_channels
        
        for i, out_channels in enumerate(self.hidden_dims):
            layers.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    use_pool=True
                )
            )
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _get_feature_size(self) -> int:
        """Calculate the size of features after conv layers."""
        # After each pool layer, spatial size is halved
        # With 4 pool layers: 224 -> 112 -> 56 -> 28 -> 14
        h, w = self.input_size
        for _ in self.hidden_dims:
            h = h // 2
            w = w // 2
        
        return self.hidden_dims[-1] * h * w
    
    def _build_classifier(self) -> nn.Sequential:
        """Build classification head."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dims[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate * 0.5),
            nn.Linear(256, self.num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Logits [B, num_classes]
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x):
        """
        Extract features without classification.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Feature tensor [B, hidden_dims[-1], H', W']
        """
        return self.features(x)


class SimpleCNNLarge(BaseModel):
    """
    Larger Simple CNN with more layers for better feature extraction.
    
    Architecture:
    - 5 convolutional blocks with residual-like connections
    - Deeper feature extractor
    - More robust classifier
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 3,
        base_channels: int = 64,
        dropout_rate: float = 0.5,
        input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize SimpleCNNLarge.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            base_channels: Base number of channels (multiplied in deeper layers)
            dropout_rate: Dropout rate
            input_size: Input image size
        """
        super().__init__(num_classes)
        
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Convolutional blocks
        self.block1 = self._make_block(base_channels, base_channels, 2)
        self.block2 = self._make_block(base_channels, base_channels * 2, 2, stride=2)
        self.block3 = self._make_block(base_channels * 2, base_channels * 4, 2, stride=2)
        self.block4 = self._make_block(base_channels * 4, base_channels * 8, 2, stride=2)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_block(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Create a block of convolutional layers."""
        layers = []
        
        # First layer may have stride > 1
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Remaining layers
        for _ in range(1, num_layers):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Logits [B, num_classes]
        """
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.avgpool(x)
        x = self.classifier(x)
        
        return x
    
    def extract_features(self, x):
        """
        Extract features without classification.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Feature tensor [B, base_channels * 8]
        """
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


# Model registry for easy instantiation
MODEL_REGISTRY = {
    "simple_cnn": SimpleCNN,
    "simple_cnn_large": SimpleCNNLarge,
}


def get_model(model_name: str, **kwargs) -> BaseModel:
    """
    Get a model by name.
    
    Args:
        model_name: Name of the model
        **kwargs: Model arguments
        
    Returns:
        Model instance
    """
    if model_name.lower() not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name.lower()](**kwargs)
