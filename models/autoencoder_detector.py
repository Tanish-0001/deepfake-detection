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
        dropout: float = 0.1,
        add_noise: bool = True,
        noise_std: float = 0.1
    ):
        """
        Initialize the latent autoencoder.
        
        Args:
            input_dim: Dimension of input embeddings (768 for ViT-B/14)
            bottleneck_dim: Dimension of the bottleneck (small for compression)
            hidden_dims: Hidden layer dimensions for encoder/decoder
            dropout: Dropout rate
            use_cosine_loss: Use cosine similarity loss instead of MSE
            add_noise: Add noise to inputs during training (denoising AE)
            noise_std: Standard deviation of noise
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = []
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.add_noise = add_noise
        self.noise_std = noise_std
        
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
    
    def reconstruction_loss(self, z: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Handles both 2D (B, D) and 3D (B, num_patches, D) inputs.
        For 3D inputs, flattens patches before processing.
        
        Args:
            z: Input embeddings of shape (B, D) or (B, num_patches, D)
            training: Whether in training mode (controls noise injection)
            
        Returns:
            Reconstruction loss (scalar)
        """
        original_shape = z.shape
        
        if z.dim() == 3:
            # Flatten patches: (B, num_patches, D) -> (B * num_patches, D)
            B, num_patches, D = z.shape
            z = z.view(B * num_patches, D)
        
        # Add noise during training (denoising autoencoder)
        if training and self.add_noise and self.training:
            z_noisy = z + torch.randn_like(z) * self.noise_std
        else:
            z_noisy = z
        
        z_hat = self.forward(z_noisy)
        
        loss = F.mse_loss(z_hat, z)
        return loss


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
        bottleneck_dim: int = 8,  # Reduced default for better compression
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        threshold: float = None,
        intermediate_layers: int = 4,
        layer_index: int = 0,
        normalize_features: bool = True,  # normalize by default
        add_noise: bool = False,  # Denoising autoencoder
        noise_std: float = 0.1
    ):
        """
        Initialize the autoencoder detector.
        
        Args:
            num_classes: Number of classes (kept for compatibility, always 2)
            bottleneck_dim: Bottleneck dimension for the autoencoder (smaller = more compression)
            hidden_dims: Hidden dimensions for encoder/decoder
            dropout: Dropout rate
            threshold: Decision threshold for real vs fake (learned during training)
            intermediate_layers: Number of intermediate layers to extract (from the end)
            layer_index: Which layer to use from the extracted layers (0 = earliest, -1 = last/final)
            normalize_features: Whether to L2-normalize extracted features (default: True)
            add_noise: Add noise during training (denoising AE - prevents identity mapping)
            noise_std: Standard deviation of noise
        """
        super().__init__(num_classes)
        
        # Store configuration
        self.intermediate_layers = intermediate_layers
        self.layer_index = layer_index
        self.normalize_features = normalize_features
        
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
        
        # Build autoencoder with improved defaults
        if hidden_dims is None:
            hidden_dims = []  # Smaller network for bottleneck
        
        self.autoencoder = LatentAutoencoder(
            input_dim=self.feature_dim,
            bottleneck_dim=bottleneck_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            add_noise=add_noise,
            noise_std=noise_std
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
        Extract patch-level features from intermediate backbone layers.
        
        Args:
            x: Input images of shape (B, 3, H, W)
            
        Returns:
            Patch feature embeddings of shape (B, num_patches, 768)
            For ViT-B/14 with 224x224 input: num_patches = 256
        """
        with torch.no_grad():
            intermediate_outputs = self.backbone.get_intermediate_layers(x, n=1)[0]
            patch_tokens = intermediate_outputs[:, 1:, :]  # (B, num_patches, 768)
        
        # Optional L2 normalization (default: on)
        if self.normalize_features:
            patch_tokens = F.normalize(patch_tokens, dim=-1)
        
        return patch_tokens
    
    def intervention_cost(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute intervention cost as distribution over patches, then aggregate.
        
        This is the core detection metric. Real images should have low cost
        (already on the learned manifold), while fake images have high cost
        (AE needs to "move" them to the real manifold).
        
        Args:
            z: Patch embeddings of shape (B, num_patches, 768)
            
        Returns:
            Aggregated intervention cost of shape (B,)
        """
        with torch.no_grad():
            B, num_patches, D = z.shape
            
            # Reshape to process all patches through autoencoder: (B * num_patches, 768)
            z_flat = z.view(B * num_patches, D)
            z_hat_flat = self.autoencoder(z_flat)
            
            # Reshape back: (B, num_patches, 768)
            z_hat = z_hat_flat.view(B, num_patches, D)
            patch_costs = torch.norm(z_hat - z, dim=-1)  # (B, num_patches)
            
            k = max(1, int(0.1 * num_patches))  # top 10% patches
            topk_costs = patch_costs.topk(k, dim=-1).values
            cost = topk_costs.mean(dim=-1)  # (B,)
        
        return cost
    
    def intervention_cost_per_patch(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute intervention cost per patch (for visualization/analysis).
        
        Args:
            z: Patch embeddings of shape (B, num_patches, 768)
            
        Returns:
            Per-patch intervention cost of shape (B, num_patches)
        """
        with torch.no_grad():
            B, num_patches, D = z.shape
            
            z_flat = z.view(B * num_patches, D)
            z_hat_flat = self.autoencoder(z_flat)
            z_hat = z_hat_flat.view(B, num_patches, D)

            patch_costs = torch.norm(z_hat - z, dim=-1)
        
        return patch_costs
    
    def intervention_cost_trainable(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute intervention cost with gradient flow (for training).
        
        Args:
            z: Patch embeddings of shape (B, num_patches, 768)
            
        Returns:
            Aggregated intervention cost of shape (B,)
        """
        B, num_patches, D = z.shape
        
        # Reshape to process all patches through autoencoder
        z_flat = z.view(B * num_patches, D)
        z_hat_flat = self.autoencoder(z_flat)
        z_hat = z_hat_flat.view(B, num_patches, D)

        patch_costs = torch.norm(z_hat - z, dim=-1)
        
        k = max(1, int(0.1 * num_patches))  # top 10% patches
        topk_costs = patch_costs.topk(k, dim=-1).values
        cost = topk_costs.mean(dim=-1)  # (B,)
        
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
    
    def get_margin_loss(
        self, 
        z_fake: torch.Tensor, 
        margin: float
    ) -> torch.Tensor:
        """
        Compute hinge-style margin loss on fake samples.
        
        Encourages fake samples to have intervention cost above the margin.
        
        Args:
            z_fake: Fake image features of shape (B, num_patches, 768)
            margin: Margin threshold (typically mean + alpha * std of real costs)
            
        Returns:
            Margin loss (scalar)
        """
        # Compute intervention cost with gradients (trainable version)
        cost_fake = self.intervention_cost_trainable(z_fake)
        
        # Hinge loss: penalize when cost_fake < margin
        # loss = relu(margin - cost_fake).mean()
        loss_margin = torch.relu(margin - cost_fake).mean()
        
        return loss_margin
    
    def get_combined_loss(
        self,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        margin_alpha: float = 2.0,
        margin_lambda: float = 0.1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss: reconstruction on real + margin loss on fake.
        
        Args:
            x_real: Real images of shape (B_real, 3, H, W)
            x_fake: Fake images of shape (B_fake, 3, H, W)
            margin_alpha: Alpha coefficient for margin (margin = mean + alpha * std)
            margin_lambda: Weight for margin loss
            
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses
        """
        # Extract features
        z_real = self.extract_features(x_real)
        z_fake = self.extract_features(x_fake)
        
        # Compute real reconstruction loss
        loss_real = self.autoencoder.reconstruction_loss(z_real)
        
        # Compute intervention costs on real samples for margin calculation
        with torch.no_grad():
            cost_real = self.intervention_cost_trainable(z_real)
            mean_cost = cost_real.mean().item()
            std_cost = cost_real.std().item()
            margin = mean_cost + margin_alpha * std_cost
        
        # Compute margin loss on fake samples
        loss_margin = self.get_margin_loss(z_fake, margin)
        
        # Combined loss
        total_loss = loss_real + margin_lambda * loss_margin
        
        loss_dict = {
            'loss_real': loss_real.item(),
            'loss_margin': loss_margin.item(),
            'margin': margin,
            'mean_cost_real': mean_cost,
            'std_cost_real': std_cost
        }
        
        return total_loss, loss_dict
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patch embeddings for analysis.
        
        Args:
            x: Input images of shape (B, 3, H, W)
            
        Returns:
            Patch embeddings of shape (B, num_patches, 768)
        """
        return self.extract_features(x)


def intervention_cost(z: torch.Tensor, ae: LatentAutoencoder, use_cosine: bool = True) -> torch.Tensor:
    """
    Standalone intervention cost function for patch-level embeddings.
    
    Computes per-patch intervention cost and aggregates via mean.
    
    Args:
        z: Patch embeddings of shape (B, num_patches, D) or (B, D)
        ae: Trained autoencoder
        use_cosine: Use cosine distance instead of L2
        
    Returns:
        Aggregated intervention cost of shape (B,)
    """
    with torch.no_grad():
        if z.dim() == 3:
            B, num_patches, D = z.shape
            z_flat = z.view(B * num_patches, D)
            z_hat_flat = ae(z_flat)
            z_hat = z_hat_flat.view(B, num_patches, D)
            
            # Per-patch cost, then aggregate
            if use_cosine:
                cos_sim = F.cosine_similarity(z_hat, z, dim=-1)
                patch_costs = 1 - cos_sim
            else:
                patch_costs = torch.norm(z_hat - z, dim=-1)
            cost = patch_costs.mean(dim=-1)  # (B,)
        else:
            # Fallback for 2D input
            z_hat = ae(z)
            if use_cosine:
                cos_sim = F.cosine_similarity(z_hat, z, dim=-1)
                cost = 1 - cos_sim
            else:
                cost = torch.norm(z_hat - z, dim=-1)
    return cost
