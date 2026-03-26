# Models module
from .simple_cnn import SimpleCNN, SimpleCNNLarge
from .base_model import BaseModel
from .dino_model import DinoModel
from .dino_temporal_model import DinoTemporalModel
from .autoencoder_detector import AutoencoderDetector, LatentAutoencoder, intervention_cost
from .dino_svd_model import DinoSVDModel, SVDResidualLinear
from .DinoSVD_MSTPP import DinoSVD_MSTPP_Model
from .DinoSVD_HSI_CrossAttention import DinoSVD_HSI_CrossAttention_Model
from .DinoSVD_SpectralAdapter import DinoSVD_SpectralAdapter_Model

__all__ = [
    'BaseModel',
    'SimpleCNN',
    'SimpleCNNLarge',
    'DinoModel',
    'DinoTemporalModel',
    'AutoencoderDetector',
    'LatentAutoencoder',
    'intervention_cost',
    'DinoSVDModel',
    'SVDResidualLinear',
    'DinoSVD_MSTPP_Model',
    'DinoSVD_HSI_CrossAttention_Model',
    'DinoSVD_SpectralAdapter_Model',
    'get_model'
]


def get_model(model_name: str, **kwargs):
    """
    Factory function to create models by name.
    
    Args:
        model_name: Name of the model ('simple_cnn', 'simple_cnn_large', 'dino_model', 
                   'dino_temporal_model', 'autoencoder_detector', 'dino_svd')
        **kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        Instantiated model
    """
    models = {
        'simple_cnn': SimpleCNN,
        'simple_cnn_large': SimpleCNNLarge,
        'dino_model': DinoModel,
        'dino_temporal_model': DinoTemporalModel,
        'autoencoder_detector': AutoencoderDetector,
        'dino_svd': DinoSVDModel,
        'dino_svd_mstpp': DinoSVD_MSTPP_Model,
        'dino_svd_hsi_crossattn': DinoSVD_HSI_CrossAttention_Model,
        'dino_svd_spectral_adapter': DinoSVD_SpectralAdapter_Model
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)
