import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .MST_plus_plus import MST_Plus_Plus
from .dino_svd_model import DinoSVDModel

# Get the directory where this file is located
_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


class HSI_Encoder(nn.Module):
    def __init__(self, in_channels=3, encoding_dim=256, hidden_dims=None):
        super(HSI_Encoder, self).__init__()
        self.RGB2HSI = MST_Plus_Plus(in_channels=in_channels, out_channels=31)

        # Load pre-trained weights from the models directory
        weights_path = os.path.join(_MODELS_DIR, 'hyper_skin_mstpp.pt')

        ckpt = torch.load(weights_path, map_location='cpu')

        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        self.RGB2HSI.load_state_dict(ckpt)
        self.RGB2HSI.requires_grad_(False)

        if hidden_dims is None:
            self.hidden_dims = [64, 128]
        else:
            self.hidden_dims = hidden_dims

        self.feature_size = self.hidden_dims[-1]  # after global average pooling
        self.encoding_dim = encoding_dim

        self.feature_extractor = self._build_features()
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_size, self.encoding_dim)
        )

    def _build_features(self) -> nn.Sequential:
        """Build convolutional feature extractor."""
        layers = []
        in_channels = 31
        
        for _, out_channels in enumerate(self.hidden_dims):
    
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.no_grad():
            x = self.RGB2HSI(x)

        x = self.feature_extractor(x)
        x = self.encoder(x)
        return x

class DinoSVD_with_HSI_Model(nn.Module):
    def __init__(
            self, 
            num_classes=2, 
            hsi_encoding_dim=256,
            hsi_hidden_dims=None,
            classifier_hidden_dims=None, 
            dropout=0.1,
            dino_model="dinov2_vitb14",
            svd_rank=None,
            target_modules=None,
            use_learned_scaling=True
        ):
        super(DinoSVD_with_HSI_Model, self).__init__()
        
        self.hsi_encoder = HSI_Encoder(in_channels=3, encoding_dim=hsi_encoding_dim, hidden_dims=hsi_hidden_dims)

        self.dino_svd = DinoSVDModel(
            hidden_dims=[],  # no classifier layers since we will use the features directly
            dropout=dropout,
            dino_model=dino_model,
            svd_rank=svd_rank,
            target_modules=target_modules,
        )

        if classifier_hidden_dims is None:
            classifier_hidden_dims = [256, 16]

        # Learned scaling factors instead of hard L2 normalization
        self.use_learned_scaling = use_learned_scaling
        if use_learned_scaling:
            self.hsi_scale = nn.Parameter(torch.ones(1))
            self.dino_scale = nn.Parameter(torch.ones(1))

        # Separate LayerNorms for each feature branch before fusion
        self.hsi_norm = nn.LayerNorm(hsi_encoding_dim)
        self.dino_norm = nn.LayerNorm(self.dino_svd.feature_dim)

        layers = []
        input_size = self.dino_svd.feature_dim + hsi_encoding_dim
        
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
            nn.Linear(input_size, num_classes)
        ])
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        hsi_encoding = self.hsi_encoder(x)
        dino_features = self.dino_svd.get_features(x)

        # Normalize each branch independently with LayerNorm (preserves magnitude info)
        hsi_encoding = self.hsi_norm(hsi_encoding)
        dino_features = self.dino_norm(dino_features)

        if self.use_learned_scaling:
            hsi_encoding = hsi_encoding * self.hsi_scale
            dino_features = dino_features * self.dino_scale

        combined_features = torch.cat([hsi_encoding, dino_features], dim=1)
        out = self.classifier(combined_features)

        # Cache normalized features for decorrelation loss computation
        self._cached_hsi_features = hsi_encoding
        self._cached_dino_features = dino_features

        return out

    def compute_decorrelation_loss(self) -> torch.Tensor:
        """
        Compute feature decorrelation loss between HSI and DINO branches.
        
        Encourages the two encoders to capture complementary (non-redundant)
        information, which improves cross-dataset generalization.
        
        Uses the Hilbert-Schmidt Independence Criterion (HSIC) approximation:
        Minimize the squared Frobenius norm of the cross-covariance matrix.
        """
        if not hasattr(self, '_cached_hsi_features') or self._cached_hsi_features is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        hsi = self._cached_hsi_features  # (B, D_hsi)
        dino = self._cached_dino_features  # (B, D_dino)
        
        # Center features
        hsi_centered = hsi - hsi.mean(dim=0, keepdim=True)
        dino_centered = dino - dino.mean(dim=0, keepdim=True)
        
        # Cross-covariance matrix: (D_hsi, D_dino)
        batch_size = hsi.size(0)
        cross_cov = (hsi_centered.T @ dino_centered) / (batch_size - 1 + 1e-8)
        
        # Minimize squared Frobenius norm of cross-covariance
        loss = torch.norm(cross_cov, p='fro') ** 2
        
        return loss


if __name__ == "__main__":
    # Test the model with dummy input
    model = DinoSVD_with_HSI_Model(num_classes=2)
    dummy_input = torch.randn(2, 3, 128, 128)  # Batch of 2 RGB images
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Should be [2, 2] for binary classification