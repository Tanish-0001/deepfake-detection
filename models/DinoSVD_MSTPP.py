import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .MST_plus_plus import MST_Plus_Plus
from .dino_svd_model import DinoSVDModel

# Get the directory where this file is located
_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


class HSICompressor(nn.Module):
    """
    Lightweight convolutional network that compresses 31-channel HSI
    imagery down to 3 channels suitable for input to a pretrained
    RGB backbone (DinoSVD).

    Architecture: 31 → 16 → 8 → 3  (Conv3×3 + BN + ReLU each layer)
    No spatial pooling — preserves full resolution.
    """

    def __init__(self):
        super(HSICompressor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(31, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 31, H, W) hyperspectral image
        Returns:
            (B, 3, H, W) compressed pseudo-RGB image
        """
        return self.net(x)


class DinoSVD_MSTPP_Model(nn.Module):
    """
    Deepfake detection model that:
      1. Upsamples RGB → 31-channel HSI via MST++ (frozen)
      2. Compresses 31ch → 3ch via a lightweight trainable conv network
      3. Extracts features via a fully-frozen DinoSVD backbone
      4. Classifies via a trainable MLP head

    Only the HSICompressor and classifier head are trainable.
    """

    def __init__(
        self,
        num_classes: int = 2,
        classifier_hidden_dims: List[int] = None,
        dropout: float = 0.1,
        dino_model: str = "dinov2_vitb14",
        svd_rank: int = None,
        target_modules: List[str] = None,
    ):
        super(DinoSVD_MSTPP_Model, self).__init__()

        # ---- 1. MST++: RGB → 31ch HSI (frozen) ----
        self.rgb2hsi = MST_Plus_Plus(in_channels=3, out_channels=31)
        weights_path = os.path.join(_MODELS_DIR, "hyper_skin_mstpp.pt")
        ckpt = torch.load(weights_path, map_location="cpu")
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        self.rgb2hsi.load_state_dict(ckpt)
        self.rgb2hsi.requires_grad_(False)

        # ---- 2. HSI Compressor: 31ch → 3ch (trainable) ----
        self.compressor = HSICompressor()

        # ---- 3. DinoSVD backbone (fully frozen) ----
        self.dino_svd = DinoSVDModel(
            hidden_dims=[],  # no classifier layers — we use our own head
            dropout=dropout,
            dino_model=dino_model,
            svd_rank=svd_rank,
            target_modules=target_modules,
        )
        self.dino_svd.requires_grad_(False)

        # ---- 4. Classifier head (trainable) ----
        if classifier_hidden_dims is None:
            classifier_hidden_dims = [256, 16]

        layers = []
        input_size = self.dino_svd.feature_dim

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
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) RGB image
        Returns:
            (B, num_classes) logits
        """
        # MST++ (frozen) — no grad needed, nothing before it needs gradients
        with torch.no_grad():
            hsi = self.rgb2hsi(x)           # (B, 31, H, W)

        # Compressor (trainable) — grad flows here
        pseudo_rgb = self.compressor(hsi)    # (B, 3, H, W)

        # DinoSVD feature extraction (frozen weights via requires_grad=False)
        # NOTE: Do NOT use torch.no_grad() here — we need the computation graph
        # so that gradients can flow THROUGH DinoSVD back to the compressor.
        # The frozen weights won't accumulate gradients, but the graph is needed.
        features = self.dino_svd.get_features(pseudo_rgb)  # (B, feat_dim)

        # Classifier (trainable)
        logits = self.classifier(features)
        return logits

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_trainable_params(self) -> list:
        """Return only the parameters that should be optimized."""
        params = []
        params += list(self.compressor.parameters())
        params += list(self.classifier.parameters())
        # Include DinoSVD params if they've been unfrozen
        params += [p for p in self.dino_svd.parameters() if p.requires_grad]
        return params

    def unfreeze_dino(self):
        """Unfreeze DinoSVD backbone weights for finetuning."""
        self.dino_svd.requires_grad_(True)
        trainable = sum(p.numel() for p in self.dino_svd.parameters() if p.requires_grad)
        print(f"DinoSVD unfrozen: {trainable:,} parameters now trainable")

    def print_trainable_params(self):
        """Print trainable vs total parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"Total params:     {total:,}")
        print(f"Trainable params: {trainable:,}")
        print(f"Frozen params:    {frozen:,}")
        print(f"Trainable ratio:  {trainable / total * 100:.2f}%")


if __name__ == "__main__":
    # Quick sanity check
    model = DinoSVD_MSTPP_Model(num_classes=2)
    model.print_trainable_params()

    dummy_input = torch.randn(2, 3, 128, 128)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # expect [2, 2]
