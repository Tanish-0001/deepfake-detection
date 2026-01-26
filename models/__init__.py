# Models module
from .simple_cnn import SimpleCNN, SimpleCNNLarge
from .base_model import BaseModel

__all__ = [
    'BaseModel',
    'SimpleCNN',
    'SimpleCNNLarge',
    'get_model'
]


def get_model(model_name: str, **kwargs):
    """
    Factory function to create models by name.
    
    Args:
        model_name: Name of the model ('simple_cnn', 'simple_cnn_large')
        **kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        Instantiated model
    """
    models = {
        'simple_cnn': SimpleCNN,
        'simple_cnn_large': SimpleCNNLarge,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)
