from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel


class DinoModel(BaseModel):
    def __init__(self, num_classes: int = 2, hidden_dims: List[int] = None, dropout: float = 0.3):
        super().__init__(num_classes)

        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14'
        )

        self.freeze_backbone()
        self.backbone.eval()

        self.backbone_frozen = True

        if hidden_dims is None:
            hidden_dims = [256]

        layers = []
        input_size = 768  # ViT-B/14 feature size

        for hidden_size in hidden_dims:
            layers.extend([
                nn.LayerNorm(input_size),
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            input_size = hidden_size

        # Final classification head
        layers.extend([
            nn.LayerNorm(input_size),
            nn.Linear(input_size, num_classes)
        ])

        self.classifier = nn.Sequential(*layers)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        self.backbone_frozen = False
        for block in self.backbone.blocks[-1:]:
            for p in block.parameters():
                p.requires_grad = True
    
    def get_backbone_params(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]
    
    def forward(self, x):
        # x: (B, 3, H, W)
        if self.backbone_frozen:
            with torch.no_grad():
                features = self.backbone(x)  # ViT returns tokens
        else:
            features = self.backbone(x)

        # Mean-pool patch tokens (skip CLS)
        if features.dim() == 3:
            features = features[:, 1:, :].mean(dim=1)

        logits = self.classifier(features)
        return logits

