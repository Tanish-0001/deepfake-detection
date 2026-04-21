"""
DinoSVD + Parameter-Efficient Spectral Adapter Model for Deepfake Detection.

Architecture:
  - MST++ (frozen, pretrained): RGB → 31-channel HSI
  - HSI Tokenizer: Conv2d(31, D, patch_size, patch_size) → spectral tokens
  - DinoSVD backbone (frozen): Transformer blocks wrapped with SpectralAdapters
  - Spectral Adapters (trainable): Bottleneck fusion of RGB + HSI features at
    each transformer block
  - Classifier (trainable): MLP head on mean-pooled patch tokens

The Spectral Adapter is a lightweight bottleneck module inserted after the
Multi-Head Self-Attention (MSA) in each Dinov2 block and before the FFN.  It
down-projects both the RGB tokens and the HSI tokens into a small bottleneck
space, fuses them with a GELU non-linearity, and up-projects back to the
embedding dimension.  The adapter output is added to the main stream as a
scaled residual, gently nudging frozen Dinov2 features towards spectral
anomaly sensitivity.

Only the HSI tokenizer, adapter weights, and classifier head are trained.
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
# Spectral Adapter (single bottleneck module)
# ---------------------------------------------------------------------------

class SpectralAdapter(nn.Module):
    """
    Bottleneck adapter that fuses RGB and HSI tokens.

    Given RGB tokens X' (B, N, D) from the frozen MSA output and HSI spectral
    tokens X_hsi (B, N, D), the adapter computes:

        Adapter(X', X_hsi) = GELU(X' @ W_down + X_hsi @ W_hsi) @ W_up

    where:
        W_down : (D, d)   — compresses RGB tokens to bottleneck
        W_hsi  : (D, d)   — compresses HSI tokens to bottleneck
        W_up   : (d, D)   — expands fused bottleneck back to embedding dim

    Parameters are initialised small (near-zero up-projection) so the adapter
    starts as a near-identity, preserving the frozen backbone's behaviour.
    """

    def __init__(self, embed_dim: int, bottleneck_dim: int = 64):
        super().__init__()

        self.W_down = nn.Linear(embed_dim, bottleneck_dim, bias=True)
        self.W_hsi = nn.Linear(embed_dim, bottleneck_dim, bias=True)
        self.activation = nn.GELU()
        self.W_up = nn.Linear(bottleneck_dim, embed_dim, bias=True)

        # Initialise up-projection to near-zero so adapter starts neutral
        nn.init.normal_(self.W_up.weight, std=0.002)
        nn.init.zeros_(self.W_up.bias)

    def forward(
        self,
        x_rgb: torch.Tensor,
        x_hsi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_rgb: (B, N, D) — RGB patch tokens after frozen MSA
            x_hsi: (B, N, D) — HSI spectral tokens

        Returns:
            (B, N, D) — adapter residual to add to the main stream
        """
        z = self.W_down(x_rgb) + self.W_hsi(x_hsi)     # (B, N, d)
        z = self.activation(z)                           # (B, N, d)
        out = self.W_up(z)                               # (B, N, D)
        return out


# ---------------------------------------------------------------------------
# Adapted DINOv2 Block (frozen block + trainable adapter)
# ---------------------------------------------------------------------------

class AdaptedDinoBlock(nn.Module):
    """
    Wraps a single frozen DINOv2 transformer block, inserting a
    SpectralAdapter between the MSA and FFN stages.

    Forward pass for input tokens X (with CLS) and spectral tokens X_hsi:

        Step A — Frozen Attention:
            X' = X + ls1(attn(norm1(X)))

        Step B — Spectral Injection (trainable adapter, patch tokens only):
            patch' = patch(X') + s · Adapter(patch(X'), X_hsi)
            X'_fused = [CLS(X'), patch']

        Step C — Frozen FFN:
            X'' = X'_fused + ls2(mlp(norm2(X'_fused)))

    The adapter is applied ONLY to patch tokens (not CLS), since X_hsi has
    no corresponding CLS token.
    """

    def __init__(
        self,
        original_block: nn.Module,
        embed_dim: int,
        bottleneck_dim: int = 64,
        adapter_scale: float = 0.1,
    ):
        super().__init__()

        # Store the original frozen block (not copied — shares parameters)
        self.block = original_block
        self.adapter = SpectralAdapter(embed_dim, bottleneck_dim)
        self.scale = adapter_scale

    def forward(
        self,
        x: torch.Tensor,
        x_hsi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:     (B, 1+N, D) — RGB tokens (CLS prepended)
            x_hsi: (B, N, D)   — HSI spectral tokens (no CLS)

        Returns:
            (B, 1+N, D) — updated tokens
        """
        # ---- Step A: Frozen Attention ----
        x_prime = x + self.block.ls1(
            self.block.attn(self.block.norm1(x))
        )

        # ---- Step B: Spectral Injection (patch tokens only) ----
        cls_token = x_prime[:, :1, :]       # (B, 1, D)
        patch_tokens = x_prime[:, 1:, :]    # (B, N, D)

        adapter_out = self.adapter(patch_tokens, x_hsi)   # (B, N, D)
        patch_tokens = patch_tokens + self.scale * adapter_out

        x_fused = torch.cat([cls_token, patch_tokens], dim=1)  # (B, 1+N, D)

        # ---- Step C: Frozen FFN ----
        x_out = x_fused + self.block.ls2(
            self.block.mlp(self.block.norm2(x_fused))
        )

        return x_out


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class DinoSVD_SpectralAdapter_Model(nn.Module):
    """
    Deepfake detector with Parameter-Efficient Spectral Adapters.

    Pipeline:
      1. MST++ (frozen, pretrained) converts RGB → 31-channel HSI
      2. A trainable Conv2d tokenises the 31-ch HSI into patch tokens
         that match the spatial layout of Dinov2's patch embedding
      3. The frozen DinoSVD backbone processes RGB through its standard
         patch embedding; then each transformer block is wrapped with
         an AdaptedDinoBlock that injects HSI information via a
         lightweight spectral adapter
      4. Mean-pooled patch tokens → trainable MLP classifier

    Only the HSI tokenizer, adapter weights, and classifier are trained.
    """

    def __init__(
        self,
        num_classes: int = 2,
        # Adapter settings
        bottleneck_dim: int = 64,
        adapter_scale: float = 0.1,
        # Classifier settings
        classifier_hidden_dims: List[int] = None,
        dropout: float = 0.1,
        # DinoSVD settings
        dino_model: str = "dinov2_vitb14",
        svd_rank: int = None,
        target_modules: List[str] = None,
    ):
        super().__init__()

        # ---- ImageNet normalisation constants (registered as buffers) ----
        self.register_buffer(
            '_img_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            '_img_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        # ---- 1. MST++: RGB → 31ch HSI (frozen, pretrained) ----
        self.rgb2hsi = MST_Plus_Plus(in_channels=3, out_channels=31)
        weights_path = os.path.join(_MODELS_DIR, "hyper_skin_mstpp.pt")
        ckpt = torch.load(weights_path, map_location="cpu")
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        self.rgb2hsi.load_state_dict(ckpt)
        self.rgb2hsi.requires_grad_(False)

        # ---- 2. DinoSVD backbone (frozen) ----
        self.dino_svd = DinoSVDModel(
            hidden_dims=[],   # no classifier — we use our own head
            dropout=dropout,
            dino_model=dino_model,
            svd_rank=svd_rank,
            target_modules=target_modules,
        )
        self.dino_svd.requires_grad_(False)

        # Grab key dimensions from the backbone
        self.embed_dim = self.dino_svd.feature_dim      # e.g. 768 for ViT-B
        self.patch_size = self.dino_svd.backbone.patch_embed.proj.kernel_size[0]  # 14

        # ---- 3. HSI Tokenizer: 31ch → D-dim patch tokens (trainable) ----
        # Uses the SAME patch_size and stride as Dinov2 so the spatial grid
        # of HSI tokens aligns perfectly with the RGB patch tokens.
        self.hsi_tokenizer = nn.Sequential(
            nn.Conv2d(
                in_channels=31,
                out_channels=self.embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=True,
            ),
        )

        # ---- 4. Adapted Blocks (frozen blocks + trainable adapters) ----
        self.adapted_blocks = nn.ModuleList()
        for block in self.dino_svd.backbone.blocks:
            self.adapted_blocks.append(
                AdaptedDinoBlock(
                    original_block=block,
                    embed_dim=self.embed_dim,
                    bottleneck_dim=bottleneck_dim,
                    adapter_scale=adapter_scale,
                )
            )

        # ---- 5. Classifier head (trainable) ----
        if classifier_hidden_dims is None:
            classifier_hidden_dims = [256, 16]

        layers: list = []
        input_size = self.embed_dim

        for hidden_size in classifier_hidden_dims:
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
            nn.Linear(input_size, num_classes),
        ])

        self.classifier = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Custom forward through adapted blocks
    # ------------------------------------------------------------------

    def _prepare_rgb_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Replicate DINOv2's token preparation: patch embed → CLS → pos embed.

        Args:
            x: (B, 3, H, W) — ImageNet-normalised RGB image

        Returns:
            (B, 1+N, D) — token sequence with CLS prepended
        """
        backbone = self.dino_svd.backbone
        B = x.shape[0]

        # Patch embedding
        tokens = backbone.patch_embed(x)                 # (B, N, D)

        # Prepend CLS token
        cls_tokens = backbone.cls_token.expand(B, -1, -1)  # (B, 1, D)
        tokens = torch.cat((cls_tokens, tokens), dim=1)     # (B, 1+N, D)

        # Add positional encoding
        tokens = tokens + backbone.interpolate_pos_encoding(
            tokens, x.shape[2], x.shape[3]
        )

        return tokens

    def _tokenize_hsi(self, hsi: torch.Tensor) -> torch.Tensor:
        """
        Convert 31-channel HSI to a sequence of spectral tokens matching
        the spatial layout of Dinov2's patch grid.

        Args:
            hsi: (B, 31, H, W) — hyperspectral image

        Returns:
            (B, N, D) — spectral token sequence (same N as RGB patches)
        """
        feat = self.hsi_tokenizer(hsi)         # (B, D, H', W')
        B, D, H, W = feat.shape
        tokens = feat.reshape(B, D, H * W)    # (B, D, N)
        tokens = tokens.permute(0, 2, 1)      # (B, N, D)
        return tokens

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) ImageNet-normalised RGB image

        Returns:
            (B, num_classes) logits
        """
        # ---- Un-normalise to [0, 1] for MST++ ----
        x_rgb = x * self._img_std + self._img_mean          # → [0, 1]

        # ---- MST++ (frozen) — produce 31-channel HSI ----
        with torch.no_grad():
            hsi = self.rgb2hsi(x_rgb)                        # (B, 31, H, W)

        # ---- HSI tokenisation (trainable) ----
        x_hsi_tokens = self._tokenize_hsi(hsi)              # (B, N, D)

        # ---- RGB token preparation (frozen patch embed + CLS + pos) ----
        # No torch.no_grad() — we need the graph for gradient flow through
        # the frozen layers back to the HSI tokenizer and adapters.
        x_tokens = self._prepare_rgb_tokens(x)               # (B, 1+N, D)

        # ---- Pass through adapted blocks ----
        for adapted_block in self.adapted_blocks:
            x_tokens = adapted_block(x_tokens, x_hsi_tokens)

        # ---- Final LayerNorm (frozen) ----
        x_tokens = self.dino_svd.backbone.norm(x_tokens)

        # ---- Mean-pool patch tokens (skip CLS at position 0) ----
        mean_feat = x_tokens[:, 1:, :].mean(dim=1)
        # max_feat = x_tokens[:, 1:, :].max(dim=1)[0]   
        # features = torch.cat([mean_feat, max_feat], dim=1)           # (B, D)
        features = mean_feat

        # ---- Classifier (trainable) ----
        logits = self.classifier(features)
        return logits

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_trainable_params(self) -> list:
        """Return only the parameters that should be optimised."""
        params = []
        params += list(self.hsi_tokenizer.parameters())
        # Adapter parameters only (not the frozen block references)
        for ab in self.adapted_blocks:
            params += list(ab.adapter.parameters())
        params += list(self.classifier.parameters())
        # Include DinoSVD params if they've been unfrozen
        params += [p for p in self.dino_svd.parameters() if p.requires_grad]
        return params

    def get_adapter_params(self) -> list:
        """Return only the spectral adapter parameters."""
        params = []
        for ab in self.adapted_blocks:
            params += list(ab.adapter.parameters())
        return params

    def get_tokenizer_params(self) -> list:
        """Return only the HSI tokenizer parameters."""
        return list(self.hsi_tokenizer.parameters())

    def get_classifier_params(self) -> list:
        """Return only the classifier parameters."""
        return list(self.classifier.parameters())

    def unfreeze_dino(self):
        """Unfreeze DinoSVD backbone weights for fine-tuning."""
        self.dino_svd.requires_grad_(True)
        trainable = sum(
            p.numel() for p in self.dino_svd.parameters() if p.requires_grad
        )
        print(f"DinoSVD unfrozen: {trainable:,} parameters now trainable")

    def unfreeze_dino_stage(self, stage: int):
        """
        Unfreeze portions of the DinoSVD backbone progressively.
        stage=0: Last 4 blocks + final norm
        stage=1: Middle 4 blocks
        stage=2: First 4 blocks + patch embed + cls token + pos embed
        """
        blocks = self.dino_svd.backbone.blocks
        num_blocks = len(blocks)
        
        if stage == 0:
            for block in blocks[-4:]:
                block.requires_grad_(True)
            if hasattr(self.dino_svd.backbone, 'norm'):
                self.dino_svd.backbone.norm.requires_grad_(True)
            print(f"DinoSVD Stage 0 unfrozen: Last 4 blocks out of {num_blocks}")
            
        elif stage == 1:
            for block in blocks[-8:-4]:
                block.requires_grad_(True)
            print(f"DinoSVD Stage 1 unfrozen: Blocks [-8:-4] out of {num_blocks}")
            
        elif stage == 2:
            for block in blocks[:-8]:
                block.requires_grad_(True)
            if hasattr(self.dino_svd.backbone, 'patch_embed'):
                self.dino_svd.backbone.patch_embed.requires_grad_(True)
            if hasattr(self.dino_svd.backbone, 'cls_token'):
                if hasattr(self.dino_svd.backbone.cls_token, 'requires_grad_'):
                    self.dino_svd.backbone.cls_token.requires_grad_(True)
                else:
                    self.dino_svd.backbone.cls_token.requires_grad = True
            if hasattr(self.dino_svd.backbone, 'pos_embed'):
                if hasattr(self.dino_svd.backbone.pos_embed, 'requires_grad_'):
                    self.dino_svd.backbone.pos_embed.requires_grad_(True)
                else:
                    self.dino_svd.backbone.pos_embed.requires_grad = True
            print(f"DinoSVD Stage 2 unfrozen: Remaining {max(0, num_blocks-8)} first blocks and embeddings")
            
        trainable = sum(p.numel() for p in self.dino_svd.parameters() if p.requires_grad)
        print(f"DinoSVD incrementally unfrozen: {trainable:,} parameters now trainable")

    def print_trainable_params(self):
        """Print trainable vs total parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        frozen = total - trainable

        tokenizer_params = sum(
            p.numel() for p in self.hsi_tokenizer.parameters()
            if p.requires_grad
        )
        adapter_params = sum(p.numel() for p in self.get_adapter_params())
        clf_params = sum(
            p.numel() for p in self.classifier.parameters()
            if p.requires_grad
        )
        dino_params = sum(
            p.numel() for p in self.dino_svd.parameters()
            if p.requires_grad
        )

        print(f"Total params:     {total:,}")
        print(f"Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")
        print(f"Frozen params:    {frozen:,}")
        print(f"  - HSI Tokenizer:  {tokenizer_params:,}")
        print(f"  - Adapters:       {adapter_params:,}")
        print(f"  - Classifier:     {clf_params:,}")
        print(f"  - DinoSVD:        {dino_params:,}")
        print(f"Number of adapted blocks: {len(self.adapted_blocks)}")


if __name__ == "__main__":
    # Quick sanity check
    model = DinoSVD_SpectralAdapter_Model(num_classes=2)
    model.print_trainable_params()

    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # expect [2, 2]
