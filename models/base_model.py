"""
Base model class defining the interface for all deepfake detection models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None


class BaseModel(nn.Module if TORCH_AVAILABLE else ABC):
    """
    Abstract base class for deepfake detection models.
    
    All models should inherit from this class and implement the required methods.
    """
    
    def __init__(self, num_classes: int = 2):
        """
        Initialize base model.
        
        Args:
            num_classes: Number of output classes (default 2 for binary classification)
        """
        if TORCH_AVAILABLE:
            super().__init__()
        self.num_classes = num_classes
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [B, C, H, W] or [B, T, C, H, W] for video
            
        Returns:
            Output logits of shape [B, num_classes]
        """
        pass
    
    def predict(self, x) -> Tuple:
        """
        Make predictions with the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        return preds, probs
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Get the number of parameters in the model.
        
        Args:
            trainable_only: If True, count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model info
        """
        return {
            'model_name': self.__class__.__name__,
            'num_classes': self.num_classes,
            'total_parameters': self.get_num_parameters(),
            'trainable_parameters': self.get_num_parameters(trainable_only=True)
        }
    
    def freeze_backbone(self):
        """Freeze backbone layers (for transfer learning)."""
        pass  # Override in subclasses
    
    def unfreeze_backbone(self):
        """Unfreeze backbone layers."""
        pass  # Override in subclasses
    
    def save(self, path: str):
        """
        Save model to file.
        
        Args:
            path: Path to save model
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info()
        }, path)
    
    @classmethod
    def load(cls, path: str, **kwargs):
        """
        Load model from file.
        
        Args:
            path: Path to model file
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded model
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        checkpoint = torch.load(path, map_location='cpu')
        model_info = checkpoint.get('model_info', {})
        
        # Create model instance
        model = cls(num_classes=model_info.get('num_classes', 2), **kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
