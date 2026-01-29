from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel


class DinoTemporalModel(BaseModel):
    """
    Temporal deepfake detection model using DINOv2 features with a transformer encoder.
    
    Pipeline: video → frames → DINOv2 → [T x 768] → temporal transformer → pooling → logits
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        num_transformer_layers: int = 4,
        num_attention_heads: int = 8,
        transformer_dim: int = 768,
        transformer_feedforward_dim: int = 2048,
        pooling: str = "cls",  # "cls", "mean", or "max"
        max_seq_length: int = 32,
    ):
        super().__init__(num_classes)
        
        self.pooling = pooling
        self.transformer_dim = transformer_dim
        self.max_seq_length = max_seq_length
        
        # DINOv2 backbone as feature extractor
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14'
        )
        
        self.freeze_backbone()
        self.backbone.eval()
        self.backbone_frozen = True
        
        # Feature dimension from DINOv2 ViT-B/14
        self.dino_feature_dim = 768
        
        # Project DINO features to transformer dimension if needed
        if self.dino_feature_dim != transformer_dim:
            self.feature_projection = nn.Linear(self.dino_feature_dim, transformer_dim)
        else:
            self.feature_projection = nn.Identity()
        
        # Learnable CLS token for sequence-level representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Positional encoding for temporal sequence
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_length + 1, transformer_dim)  # +1 for CLS token
        )
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)
        
        # Temporal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_attention_heads,
            dim_feedforward=transformer_feedforward_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
            norm=nn.LayerNorm(transformer_dim),
        )
        
        # Classification head
        if hidden_dims is None:
            hidden_dims = [256]
        
        layers = []
        input_size = transformer_dim
        
        for hidden_size in hidden_dims:
            layers.extend([
                nn.LayerNorm(input_size),
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            input_size = hidden_size
        
        # Final classification layer
        layers.extend([
            nn.LayerNorm(input_size),
            nn.Linear(input_size, num_classes)
        ])
        
        self.classifier = nn.Sequential(*layers)
    
    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self, num_blocks: int = 1):
        """Unfreeze the last N transformer blocks of the backbone."""
        self.backbone_frozen = False
        for block in self.backbone.blocks[-num_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
    
    def get_backbone_params(self):
        """Return backbone parameters that require gradients."""
        return [p for p in self.backbone.parameters() if p.requires_grad]
    
    def get_temporal_params(self):
        """Return temporal transformer parameters."""
        return list(self.temporal_transformer.parameters())
    
    def get_classifier_params(self):
        """Return classifier parameters."""
        return list(self.classifier.parameters())
    
    def extract_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from individual frames using DINOv2.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W) or (B*T, C, H, W)
            
        Returns:
            Frame features of shape (B, T, D) where D is dino_feature_dim
        """
        input_shape = x.shape
        
        if len(input_shape) == 5:
            # Input is (B, T, C, H, W)
            B, T, C, H, W = input_shape
            x = x.view(B * T, C, H, W)
        else:
            # Input is (B*T, C, H, W), need to infer B and T
            raise ValueError("Input must be of shape (B, T, C, H, W)")
        
        # Extract DINO features
        if self.backbone_frozen:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        
        # Mean-pool patch tokens (skip CLS token if present)
        if features.dim() == 3:
            # features: (B*T, num_patches+1, D) - skip CLS token and mean pool patches
            features = features[:, 1:, :].mean(dim=1)  # (B*T, D)
        
        # Reshape back to (B, T, D)
        features = features.view(B, T, -1)
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for temporal deepfake detection.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W) where:
               - B: batch size
               - T: number of frames (temporal dimension)
               - C: channels (3 for RGB)
               - H, W: height and width
               
        Returns:
            Logits of shape (B, num_classes)
        """
        B, T, C, H, W = x.shape
        
        # Extract frame-level features using DINOv2
        frame_features = self.extract_frame_features(x)  # (B, T, dino_feature_dim)
        
        # Project to transformer dimension
        frame_features = self.feature_projection(frame_features)  # (B, T, transformer_dim)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, transformer_dim)
        sequence = torch.cat([cls_tokens, frame_features], dim=1)  # (B, T+1, transformer_dim)
        
        # Add positional encoding
        seq_len = sequence.size(1)
        sequence = sequence + self.positional_encoding[:, :seq_len, :]
        
        # Apply temporal transformer
        transformed = self.temporal_transformer(sequence)  # (B, T+1, transformer_dim)
        
        # Pool temporal features
        if self.pooling == "cls":
            pooled = transformed[:, 0, :]  # Use CLS token
        elif self.pooling == "mean":
            pooled = transformed[:, 1:, :].mean(dim=1)  # Mean of frame tokens
        elif self.pooling == "max":
            pooled = transformed[:, 1:, :].max(dim=1).values  # Max of frame tokens
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def forward_with_attention(self, x: torch.Tensor) -> tuple:
        """
        Forward pass that also returns attention weights for interpretability.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W)
            
        Returns:
            Tuple of (logits, attention_weights)
        """
        B, T, C, H, W = x.shape
        
        # Extract frame-level features
        frame_features = self.extract_frame_features(x)
        frame_features = self.feature_projection(frame_features)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        sequence = torch.cat([cls_tokens, frame_features], dim=1)
        
        # Add positional encoding
        seq_len = sequence.size(1)
        sequence = sequence + self.positional_encoding[:, :seq_len, :]
        
        # Store attention weights
        attention_weights = []
        
        # Manual forward through transformer layers to capture attention
        for layer in self.temporal_transformer.layers:
            # Get attention weights from multi-head attention
            attn_output, attn_weight = layer.self_attn(
                sequence, sequence, sequence,
                need_weights=True,
                average_attn_weights=True
            )
            attention_weights.append(attn_weight)
            
            # Complete the transformer layer forward pass
            sequence = sequence + layer.dropout1(attn_output)
            sequence = layer.norm1(sequence)
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(sequence))))
            sequence = sequence + layer.dropout2(ff_output)
            sequence = layer.norm2(sequence)
        
        # Final layer norm
        if self.temporal_transformer.norm is not None:
            sequence = self.temporal_transformer.norm(sequence)
        
        # Pool and classify
        if self.pooling == "cls":
            pooled = sequence[:, 0, :]
        elif self.pooling == "mean":
            pooled = sequence[:, 1:, :].mean(dim=1)
        elif self.pooling == "max":
            pooled = sequence[:, 1:, :].max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        logits = self.classifier(pooled)
        
        # Stack attention weights: (num_layers, B, seq_len, seq_len)
        attention_weights = torch.stack(attention_weights, dim=0)
        
        return logits, attention_weights
