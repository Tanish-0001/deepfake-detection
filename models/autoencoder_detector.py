"""
Autoencoder-based Deepfake Detection Model.

This model uses a frozen pretrained ViT encoder (DINOv2) and trains an autoencoder
on real-image embeddings only. Deepfakes are detected by measuring how much the
autoencoder has to "move" a test embedding to make it look real (intervention cost).

Design principles:
1. Small bottleneck - forces compression of real image features
2. No overcapacity - prevents memorizing all embeddings
3. Reconstruction in latent space, not pixels - more efficient and meaningful

Detection logic:
- Real images: Low reconstruction error (AE has seen similar embeddings)
- Fake images: High reconstruction error (AE must "move" embeddings significantly)
"""

from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel


class LatentAutoencoder(nn.Module):
    """
    Autoencoder operating in the latent space of a vision encoder.
    
    Trained only on real image embeddings to learn the manifold of real images.
    Fake images will have higher reconstruction error as they lie off this manifold.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        bottleneck_dim: int = 64,
        hidden_dims: List[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize the latent autoencoder.
        
        Args:
            input_dim: Dimension of input embeddings (768 for ViT-B/14)
            bottleneck_dim: Dimension of the bottleneck (small for compression)
            hidden_dims: Hidden layer dimensions for encoder/decoder
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Final bottleneck layer
        encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (reverse of encoder)
        decoder_layers = []
        prev_dim = bottleneck_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Final reconstruction layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Encode embeddings to bottleneck representation."""
        return self.encoder(z)
    
    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode bottleneck representation back to embedding space."""
        return self.decoder(h)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode and decode.
        
        Args:
            z: Input embeddings of shape (B, input_dim)
            
        Returns:
            Reconstructed embeddings of shape (B, input_dim)
        """
        h = self.encode(z)
        z_hat = self.decode(h)
        return z_hat
    
    def reconstruction_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss (MSE).
        
        Args:
            z: Input embeddings
            
        Returns:
            Mean squared error loss
        """
        z_hat = self.forward(z)
        return F.mse_loss(z_hat, z)


class AutoencoderDetector(BaseModel):
    """
    Deepfake detector using autoencoder-based anomaly detection.
    
    Architecture:
    1. Frozen DINOv2 ViT-B/14 encoder extracts embeddings
    2. Latent autoencoder trained on real images only
    3. Detection via intervention cost (reconstruction error)
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        bottleneck_dim: int = 128,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        threshold: float = None
    ):
        """
        Initialize the autoencoder detector.
        
        Args:
            num_classes: Number of classes (kept for compatibility, always 2)
            bottleneck_dim: Bottleneck dimension for the autoencoder
            hidden_dims: Hidden dimensions for encoder/decoder
            dropout: Dropout rate
            threshold: Decision threshold for real vs fake (learned during training)
        """
        super().__init__(num_classes)
        
        # Load frozen DINOv2 backbone (same as dino_model.py)
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14'
        )
        
        # Freeze backbone completely
        self.freeze_backbone()
        self.backbone.eval()
        
        # Feature dimension from ViT-B/14
        self.feature_dim = 768
        
        # Build autoencoder
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.autoencoder = LatentAutoencoder(
            input_dim=self.feature_dim,
            bottleneck_dim=bottleneck_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        # Classification head for optional supervised fine-tuning
        # (can be used after AE training for threshold learning)
        self.classifier = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, num_classes)
        )
        
        # Learned threshold (initialized to None, set during training)
        self.register_buffer('threshold', torch.tensor(0.0) if threshold is None else torch.tensor(threshold))
        
        # Statistics for normalization (computed during training)
        self.register_buffer('mean_cost', torch.tensor(0.0))
        self.register_buffer('std_cost', torch.tensor(1.0))
    
    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone is not supported for this model."""
        raise NotImplementedError("Backbone should remain frozen for autoencoder detector")
    
    def get_backbone_params(self):
        """Return empty list as backbone is always frozen."""
        return []
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images using frozen backbone.
        
        Args:
            x: Input images of shape (B, 3, H, W)
            
        Returns:
            Feature embeddings of shape (B, 768)
        """
        with torch.no_grad():
            features = self.backbone(x)
            
            # Mean-pool patch tokens (skip CLS token if present)
            if features.dim() == 3:
                features = features[:, 1:, :].mean(dim=1)
        
        features = F.normalize(features, dim=1)  # L2 NORMALIZATION
        return features
    
    def intervention_cost(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute intervention cost: how much the AE moves an embedding.
        
        This is the core detection metric. Real images should have low cost
        (already on the learned manifold), while fake images have high cost
        (AE needs to "move" them to the real manifold).
        
        Args:
            z: Embeddings of shape (B, 768)
            
        Returns:
            Intervention cost of shape (B,)
        """
        with torch.no_grad():
            z_hat = self.autoencoder(z)
            cost = torch.norm(z_hat - z, dim=1)
        return cost
    
    def intervention_cost_trainable(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute intervention cost with gradient flow (for training).
        
        Args:
            z: Embeddings of shape (B, 768)
            
        Returns:
            Intervention cost of shape (B,)
        """
        z_hat = self.autoencoder(z)
        cost = torch.norm(z_hat - z, dim=1)
        return cost
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference.
        
        Args:
            x: Input images of shape (B, 3, H, W)
            
        Returns:
            Logits of shape (B, 2) where higher probability for class 1 = fake
        """
        # Extract features
        z = self.extract_features(x)
        
        # Compute intervention cost
        cost = self.intervention_cost(z)
        
        # Normalize cost using learned statistics
        normalized_cost = (cost - self.mean_cost) / (self.std_cost + 1e-8)
        
        # Convert cost to logits via classifier
        logits = self.classifier(normalized_cost.unsqueeze(-1))
        
        return logits
    
    def forward_with_cost(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both logits and raw intervention cost.
        
        Args:
            x: Input images of shape (B, 3, H, W)
            
        Returns:
            Tuple of (logits, cost) where:
            - logits: Shape (B, 2)
            - cost: Shape (B,) raw intervention cost
        """
        # Extract features
        z = self.extract_features(x)
        
        # Compute intervention cost
        cost = self.intervention_cost(z)
        
        # Normalize cost using learned statistics
        normalized_cost = (cost - self.mean_cost) / (self.std_cost + 1e-8)
        
        # Convert cost to logits via classifier
        logits = self.classifier(normalized_cost.unsqueeze(-1))
        
        return logits, cost
    
    def predict_from_cost(self, cost: torch.Tensor) -> torch.Tensor:
        """
        Predict labels from intervention cost using threshold.
        
        Args:
            cost: Intervention costs of shape (B,)
            
        Returns:
            Predicted labels (0=real, 1=fake) of shape (B,)
        """
        return (cost > self.threshold).long()
    
    def set_threshold(self, threshold: float):
        """Set the decision threshold."""
        self.threshold.fill_(threshold)
    
    def set_normalization_stats(self, mean: float, std: float):
        """Set normalization statistics for intervention cost."""
        self.mean_cost.fill_(mean)
        self.std_cost.fill_(std)
    
    def get_autoencoder_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute autoencoder reconstruction loss for training.
        
        Should only be called with real images during AE training.
        
        Args:
            x: Input images of shape (B, 3, H, W)
            
        Returns:
            Reconstruction loss (scalar)
        """
        z = self.extract_features(x)
        return self.autoencoder.reconstruction_loss(z)
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings for analysis.
        
        Args:
            x: Input images of shape (B, 3, H, W)
            
        Returns:
            Embeddings of shape (B, 768)
        """
        return self.extract_features(x)


def intervention_cost(z: torch.Tensor, ae: LatentAutoencoder) -> torch.Tensor:
    """
    Standalone intervention cost function as specified in requirements.
    
    Args:
        z: Input embeddings
        ae: Trained autoencoder
        
    Returns:
        Intervention cost for each sample
    """
    with torch.no_grad():
        z_hat = ae(z)
        cost = torch.norm(z_hat - z, dim=1)
    return cost
