"""
DinoSVD + HSI Cross-Attention Fusion Model for Deepfake Detection.

Architecture:
  - RGB Stream: DinoSVD backbone extracts patch tokens (spatial) and CLS token (global)
  - HSI Stream: MST++ (frozen) → 3D CNN (spectral+spatial) → 2D CNN (spatial abstraction)
  - Fusion: Cross-Attention where Q = Dinov2 patch tokens, K/V = HSI spatial features
  - Classifier: Global Average Pooling → 2-layer MLP

The cross-attention lets the rich semantic features from Dinov2 query specific
spectral anomalies captured by the HSI CNN, producing a fused representation
that combines spatial context with spectral forensic cues.
"""

import os
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .MST_plus_plus import MST_Plus_Plus
from .dino_svd_model import DinoSVDModel

# Get the directory where this file is located
_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# HSI Encoder: MST++ (frozen) → 3D CNN → 2D CNN
# ---------------------------------------------------------------------------

class HSI_3D_2D_Encoder(nn.Module):
    """
    Hyperspectral feature extractor using 3D + 2D convolutions.

    Pipeline:
      1. MST++ converts RGB → 31-channel HSI (frozen, pretrained)
      2. 3D convolutions capture joint spectral-spatial patterns
         Input is treated as (B, 1, 31, H, W) — 1 "channel", 31 spectral depth
      3. Reshape to 2D and apply 2D convolutions for high-level spatial abstraction
      4. Output: spatial feature map (B, out_channels, H', W') — NOT pooled,
         so cross-attention can operate on the spatial tokens.
    """

    def __init__(
        self,
        in_channels: int = 3,
        conv3d_channels: List[int] = None,
        conv2d_channels: List[int] = None,
    ):
        super().__init__()

        # ---- MST++ (frozen) ----
        self.rgb2hsi = MST_Plus_Plus(in_channels=in_channels, out_channels=31)
        weights_path = os.path.join(_MODELS_DIR, "hyper_skin_mstpp.pt")
        ckpt = torch.load(weights_path, map_location="cpu")
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        self.rgb2hsi.load_state_dict(ckpt)
        self.rgb2hsi.requires_grad_(False)

        # ---- 3D CNN: spectral + spatial ----
        if conv3d_channels is None:
            conv3d_channels = [16, 32]

        layers_3d = []
        in_ch = 1  # single "channel", spectral bands are the depth dimension
        for out_ch in conv3d_channels:
            layers_3d.extend([
                # kernel (spectral=3, spatial=3x3), stride spectral=2 to reduce depth
                nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch

        self.conv3d = nn.Sequential(*layers_3d)

        # Compute the remaining spectral depth after 3D convolutions
        # Starting depth = 31, each 3D layer with stride=(2,1,1) halves the depth
        spectral_depth = 31
        for _ in conv3d_channels:
            spectral_depth = math.ceil(spectral_depth / 2)  # stride-2 with padding=1

        # After 3D conv: (B, conv3d_channels[-1], spectral_depth, H, W)
        # Reshape to 2D: (B, conv3d_channels[-1] * spectral_depth, H, W)
        self._transition_channels = conv3d_channels[-1] * spectral_depth

        # ---- 2D CNN: high-level spatial abstraction ----
        if conv2d_channels is None:
            conv2d_channels = [128, 256]

        layers_2d = []
        in_ch_2d = self._transition_channels
        for out_ch in conv2d_channels:
            layers_2d.extend([
                nn.Conv2d(in_ch_2d, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch_2d = out_ch

        self.conv2d = nn.Sequential(*layers_2d)
        self.out_channels = conv2d_channels[-1]

    @property
    def feature_dim(self) -> int:
        """Number of channels in the output spatial feature map."""
        return self.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) RGB image

        Returns:
            (B, out_channels, H', W') spatial feature map
        """
        # MST++ (frozen)
        with torch.no_grad():
            hsi = self.rgb2hsi(x)  # (B, 31, H, W)

        # Reshape for 3D conv: (B, 1, 31, H, W)
        hsi_3d = hsi.unsqueeze(1)

        # 3D convolutions (spectral + spatial)
        feat_3d = self.conv3d(hsi_3d)  # (B, C3d, D', H, W)

        # Reshape to 2D: flatten spectral depth into channels
        B, C, D, H, W = feat_3d.shape
        feat_2d = feat_3d.reshape(B, C * D, H, W)

        # 2D convolutions (spatial abstraction)
        feat = self.conv2d(feat_2d)  # (B, out_channels, H', W')

        return feat


# ---------------------------------------------------------------------------
# Cross-Attention Module
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """
    Multi-head cross-attention: Q from one stream, K/V from another.

    Implements scaled dot-product attention:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    The two streams may have different feature dimensions; linear projections
    map them to a common d_model space.
    """

    def __init__(
        self,
        d_query: int,
        d_kv: int,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_query: Dimension of query features (e.g. Dinov2 patch dim = 768)
            d_kv:    Dimension of key/value features (e.g. HSI CNN out channels)
            d_model: Internal attention dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Attention dropout rate
        """
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections
        self.W_q = nn.Linear(d_query, d_model, bias=False)
        self.W_k = nn.Linear(d_kv, d_model, bias=False)
        self.W_v = nn.Linear(d_kv, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query:     (B, N_q, d_query)  — e.g. Dinov2 patch tokens
            key_value: (B, N_kv, d_kv)    — e.g. flattened HSI spatial map

        Returns:
            (B, N_q, d_model) — fused features
        """
        B, N_q, _ = query.shape

        # Project to common dimension
        Q = self.W_q(query)      # (B, N_q, d_model)
        K = self.W_k(key_value)  # (B, N_kv, d_model)
        V = self.W_v(key_value)  # (B, N_kv, d_model)

        # Reshape for multi-head attention: (B, num_heads, N, head_dim)
        Q = Q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, N_q, N_kv)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        out = attn_weights @ V  # (B, H, N_q, head_dim)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.d_model)

        # Output projection + residual-friendly norm
        out = self.W_out(out)
        out = self.out_norm(out)

        return out


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class DinoSVD_HSI_CrossAttention_Model(nn.Module):
    """
    Deepfake detector combining DinoSVD (RGB) and HSI (3D+2D CNN) streams
    via cross-attention fusion.

    Training protocol:
      1. Freeze Dinov2 entirely for the first N epochs
      2. HSI CNN + Cross-Attention + MLP classifier train from scratch
      3. Optionally unfreeze Dinov2 for fine-tuning after warmup
    """

    def __init__(
        self,
        num_classes: int = 2,
        # HSI encoder settings
        conv3d_channels: List[int] = None,
        conv2d_channels: List[int] = None,
        # Cross-attention settings
        cross_attn_d_model: int = 256,
        cross_attn_heads: int = 8,
        # Classifier settings
        classifier_hidden_dim: int = 128,
        dropout: float = 0.1,
        # DinoSVD settings
        dino_model: str = "dinov2_vitb14",
        svd_rank: int = None,
        target_modules: List[str] = None,
    ):
        super().__init__()

        # ---- 1. DinoSVD backbone (frozen initially) ----
        self.dino_svd = DinoSVDModel(
            hidden_dims=[],  # no classifier — we use cross-attention fusion
            dropout=dropout,
            dino_model=dino_model,
            svd_rank=svd_rank,
            target_modules=target_modules,
        )
        self.dino_feature_dim = self.dino_svd.feature_dim  # e.g. 768 for ViT-B

        # Freeze Dinov2 entirely at init
        self.dino_svd.requires_grad_(False)

        # ---- 2. HSI Encoder (3D + 2D CNN, trainable) ----
        self.hsi_encoder = HSI_3D_2D_Encoder(
            in_channels=3,
            conv3d_channels=conv3d_channels,
            conv2d_channels=conv2d_channels,
        )

        # ---- 3. Cross-Attention: Q=Dino patches, K/V=HSI spatial ----
        self.cross_attn = CrossAttention(
            d_query=self.dino_feature_dim,
            d_kv=self.hsi_encoder.feature_dim,
            d_model=cross_attn_d_model,
            num_heads=cross_attn_heads,
            dropout=dropout,
        )

        # ---- 4. Classifier: 2-layer MLP ----
        self.classifier = nn.Sequential(
            nn.LayerNorm(cross_attn_d_model),
            nn.Linear(cross_attn_d_model, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(classifier_hidden_dim),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _get_dino_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patch tokens from the DinoSVD backbone.

        DINOv2's forward_features returns a dict with:
          - 'x_norm_clstoken': (B, D)
          - 'x_norm_patchtokens': (B, N_patches, D)

        Returns:
            (B, N_patches, D) patch token sequence
        """
        feat_dict = self.dino_svd.backbone.forward_features(x)
        return feat_dict["x_norm_patchtokens"]  # (B, N, D)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) RGB image (ImageNet-normalised)

        Returns:
            (B, num_classes) logits
        """
        # ---- RGB stream: Dinov2 patch tokens (Q) ----
        with torch.no_grad():
            dino_patches = self._get_dino_patch_tokens(x)  # (B, N_dino, D)

        # ---- HSI stream: spatial feature map → flattened tokens (K, V) ----
        hsi_spatial = self.hsi_encoder(x)  # (B, C_hsi, H', W')
        B, C, H, W = hsi_spatial.shape
        hsi_tokens = hsi_spatial.reshape(B, C, H * W).permute(0, 2, 1)  # (B, H'*W', C_hsi)

        # ---- Cross-Attention fusion ----
        fused = self.cross_attn(
            query=dino_patches,    # Q: Dinov2 spatial semantics
            key_value=hsi_tokens,  # K, V: HSI spectral anomalies
        )  # (B, N_dino, d_model)

        # ---- Global Average Pooling over the token sequence ----
        pooled = fused.mean(dim=1)  # (B, d_model)

        # ---- Classifier ----
        logits = self.classifier(pooled)
        return logits

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_trainable_params(self) -> list:
        """Return only the parameters that should be optimised."""
        params = []
        params += list(self.hsi_encoder.parameters())
        params += list(self.cross_attn.parameters())
        params += list(self.classifier.parameters())
        # Include DinoSVD params if they've been unfrozen
        params += [p for p in self.dino_svd.parameters() if p.requires_grad]
        return params

    def unfreeze_dino(self):
        """Unfreeze DinoSVD backbone weights for fine-tuning."""
        self.dino_svd.requires_grad_(True)
        trainable = sum(p.numel() for p in self.dino_svd.parameters() if p.requires_grad)
        print(f"DinoSVD unfrozen: {trainable:,} parameters now trainable")

    def print_trainable_params(self):
        """Print trainable vs total parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        hsi_params = sum(p.numel() for p in self.hsi_encoder.parameters() if p.requires_grad)
        attn_params = sum(p.numel() for p in self.cross_attn.parameters() if p.requires_grad)
        clf_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        dino_params = sum(p.numel() for p in self.dino_svd.parameters() if p.requires_grad)

        print(f"Total params:     {total:,}")
        print(f"Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")
        print(f"Frozen params:    {frozen:,}")
        print(f"  - HSI Encoder:    {hsi_params:,}")
        print(f"  - Cross-Attn:     {attn_params:,}")
        print(f"  - Classifier:     {clf_params:,}")
        print(f"  - DinoSVD:        {dino_params:,}")


if __name__ == "__main__":
    # Quick sanity check
    model = DinoSVD_HSI_CrossAttention_Model(num_classes=2)
    model.print_trainable_params()

    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # expect [2, 2]
