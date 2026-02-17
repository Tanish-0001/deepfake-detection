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
        weights_path = os.path.join(_MODELS_DIR, 'mst_plus_plus.pth')
        self.RGB2HSI.load_state_dict(torch.load(weights_path, weights_only=True))
        self.RGB2HSI.requires_grad_(False)

        if hidden_dims is None:
            self.hidden_dims = [64, 128]
        else:
            self.hidden_dims = hidden_dims

        self.feature_size = self.hidden_dims[-1]  # after global average pooling

        self.feature_extractor = self._build_features()
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_size, encoding_dim)
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
            num_classes=31, 
            hsi_encoding_dim=256,
            hsi_hidden_dims=None,
            classifier_hidden_dims=None, 
            dropout=0.1
        ):
        super(DinoSVD_with_HSI_Model, self).__init__()
        
        self.hsi_encoder = HSI_Encoder(in_channels=3, encoding_dim=hsi_encoding_dim, hidden_dims=hsi_hidden_dims)

        self.dino_svd = DinoSVDModel(
            hidden_dims=[],  # no classifier layers since we will use the features directly
            dropout=0.2,
            dino_model="dinov2_vitb14",
        )

        if classifier_hidden_dims is None:
            classifier_hidden_dims = [256, 16]

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

        hsi_encoding = F.normalize(hsi_encoding, dim=1)
        dino_features = F.normalize(dino_features, dim=1)

        combined_features = torch.cat([hsi_encoding, dino_features], dim=1)
        out = self.classifier(combined_features)
        return out

