"""
DINO SVD Model for Deepfake Detection.

This model uses DINO (DINOv2) as the backbone and applies Singular Value Decomposition (SVD)
to decompose the original feature space into two orthogonal subspaces:
- Principal subspace: Preserves pre-trained knowledge (frozen)
- Residual subspace: Learns new forgery patterns (trainable)

This approach avoids distortion of the original rich feature space during learning fakes.

Based on the EFFORT approach for deepfake detection.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel


class SVDResidualLinear(nn.Module):
    """
    Linear layer decomposed using SVD with trainable residual components.
    
    The weight matrix W is decomposed as:
        W = W_main + W_residual
    
    Where:
        W_main = U_r @ diag(S_r) @ Vh_r (frozen, preserves pre-trained knowledge)
        W_residual = U_residual @ diag(S_residual) @ Vh_residual (trainable)
    """
    
    def __init__(self, in_features: int, out_features: int, r: int, 
                 bias: bool = True, init_weight: Optional[torch.Tensor] = None):
        """
        Initialize SVDResidualLinear layer.
        
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            r: Number of top singular values to keep in main weight (frozen)
            bias: If True, adds a learnable bias
            init_weight: Initial weight tensor to decompose
        """
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # Number of top singular values to keep fixed
        
        # Original weights (fixed/frozen)
        self.weight_main = nn.Parameter(
            torch.Tensor(out_features, in_features), 
            requires_grad=False
        )
        
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        
        # Placeholders for SVD components (set by replace_with_svd_residual)
        self.S_residual = None
        self.U_residual = None
        self.V_residual = None
        self.S_r = None
        self.U_r = None
        self.V_r = None
        
        # Frobenius norms for loss computation
        self.weight_original_fnorm = None
        self.weight_main_fnorm = None
    
    def compute_current_weight(self) -> torch.Tensor:
        """Compute the current effective weight (main + residual)."""
        if self.S_residual is not None:
            return self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        else:
            return self.weight_main
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with combined main and residual weights."""
        if (hasattr(self, 'U_residual') and hasattr(self, 'V_residual') and 
            self.S_residual is not None):
            # Reconstruct the residual weight
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            # Total weight is the fixed main weight plus the residual
            weight = self.weight_main + residual_weight
        else:
            # If residual components are not set, use only the main weight
            weight = self.weight_main
        
        return F.linear(x, weight, self.bias)
    
    def compute_orthogonal_loss(self) -> torch.Tensor:
        """
        Compute orthogonality loss to maintain orthogonal bases.
        
        From EFFORT algorithm:
        L_orth = ||Û^T Û - I||_F^2 + ||V̂^T V̂ - I||_F^2
        
        Where Û = [U_r, U_{n-r}] and V̂ = [V_r, V_{n-r}]
        """
        if self.S_residual is not None:
            # Concatenate main and residual components
            # Û = [U_r, U_{n-r}] ∈ R^{m×k} where k = min(m,n)
            U_full = torch.cat((self.U_r, self.U_residual), dim=1)
            # V̂ = [V_r, V_{n-r}] ∈ R^{k×n}
            V_full = torch.cat((self.V_r, self.V_residual), dim=0)
            
            # Compute Û^T Û and V̂^T V̂ (should equal I if orthonormal)
            # U^T U ∈ R^{k×k} for column orthonormality
            UTU = U_full.t() @ U_full
            # V^T V ∈ R^{n×n}, but V V^T ∈ R^{k×k} for row orthonormality
            # Since V has orthonormal rows, we check V V^T = I
            VVT = V_full @ V_full.t()
            
            # Construct identity matrices
            UTU_identity = torch.eye(UTU.size(0), device=UTU.device)
            VVT_identity = torch.eye(VVT.size(0), device=VVT.device)
            
            # Compute squared Frobenius norm distance from identity
            # L_orth = ||Û^T Û - I||_F^2 + ||V̂ V̂^T - I||_F^2
            loss = (torch.norm(UTU - UTU_identity, p='fro') ** 2 + 
                    torch.norm(VVT - VVT_identity, p='fro') ** 2)
        else:
            loss = torch.tensor(0.0, device=self.weight_main.device)
        
        return loss
    
    def compute_keepsv_loss(self) -> torch.Tensor:
        """
        Compute loss to maintain similar Frobenius norm as original weights.
        
        This helps preserve the magnitude of the original weight matrix.
        """
        if self.S_residual is not None and self.weight_original_fnorm is not None:
            # Total current weight
            weight_current = self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            # Frobenius norm of current weight
            weight_current_fnorm = torch.norm(weight_current, p='fro')
            
            loss = torch.abs(weight_current_fnorm ** 2 - self.weight_original_fnorm ** 2)
        else:
            loss = torch.tensor(0.0, device=self.weight_main.device)
        
        return loss
    
    def compute_fn_loss(self) -> torch.Tensor:
        """Compute Frobenius norm loss of current weight."""
        if self.S_residual is not None:
            weight_current = self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            loss = torch.norm(weight_current, p='fro') ** 2
        else:
            loss = torch.tensor(0.0, device=self.weight_main.device)
        
        return loss


def replace_with_svd_residual(module: nn.Linear, r: int) -> SVDResidualLinear:
    """
    Replace an nn.Linear module with SVDResidualLinear.
    
    Performs SVD on the original weights and creates a new module where:
    - Top r singular components are fixed (main weight)
    - Remaining components are trainable (residual)
    
    Args:
        module: Original nn.Linear module to replace
        r: Number of top singular values to keep fixed
        
    Returns:
        SVDResidualLinear module with SVD-decomposed weights
    """
    if not isinstance(module, nn.Linear):
        return module
    
    in_features = module.in_features
    out_features = module.out_features
    bias = module.bias is not None
    
    # Create SVDResidualLinear module
    new_module = SVDResidualLinear(
        in_features, out_features, r, 
        bias=bias, 
        init_weight=module.weight.data.clone()
    )
    
    # Copy bias if present
    if bias and module.bias is not None:
        new_module.bias.data.copy_(module.bias.data)
    
    # Store original weight Frobenius norm
    new_module.weight_original_fnorm = torch.norm(module.weight.data, p='fro')
    
    # Perform SVD on the original weight
    U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)
    
    # Ensure r does not exceed the number of singular values
    r = min(r, len(S))
    
    # Keep top r singular components (main weight - frozen)
    U_r = U[:, :r]      # Shape: (out_features, r)
    S_r = S[:r]         # Shape: (r,)
    Vh_r = Vh[:r, :]    # Shape: (r, in_features)
    
    # Reconstruct the main weight (fixed)
    weight_main = U_r @ torch.diag(S_r) @ Vh_r
    
    # Calculate the Frobenius norm of main weight
    new_module.weight_main_fnorm = torch.norm(weight_main, p='fro')
    
    # Set the main weight
    new_module.weight_main.data.copy_(weight_main)
    
    # Residual components (trainable)
    U_residual = U[:, r:]    # Shape: (out_features, n - r)
    S_residual = S[r:]       # Shape: (n - r,)
    Vh_residual = Vh[r:, :]  # Shape: (n - r, in_features)
    
    if len(S_residual) > 0:
        # Register trainable residual components
        new_module.S_residual = nn.Parameter(S_residual.clone())
        new_module.U_residual = nn.Parameter(U_residual.clone())
        new_module.V_residual = nn.Parameter(Vh_residual.clone())
        
        # Register frozen main SVD components (for orthogonality loss)
        new_module.S_r = nn.Parameter(S_r.clone(), requires_grad=False)
        new_module.U_r = nn.Parameter(U_r.clone(), requires_grad=False)
        new_module.V_r = nn.Parameter(Vh_r.clone(), requires_grad=False)
    else:
        new_module.S_residual = None
        new_module.U_residual = None
        new_module.V_residual = None
        new_module.S_r = None
        new_module.U_r = None
        new_module.V_r = None
    
    return new_module


def apply_svd_residual_to_attn(model: nn.Module, r: int, 
                                target_modules: List[str] = None) -> nn.Module:
    """
    Apply SVD residual decomposition to attention layers in a model.
    
    Args:
        model: PyTorch model (e.g., DINO ViT)
        r: Number of top singular values to keep fixed
        target_modules: List of module name patterns to target. 
                       If None, targets 'attn' modules by default.
    
    Returns:
        Model with SVDResidualLinear layers in attention modules
    """
    if target_modules is None:
        target_modules = ['attn']
    
    def _apply_to_module(parent_module, parent_name=''):
        for name, module in parent_module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            
            # Check if this module or any parent matches target patterns
            is_target = any(target in full_name for target in target_modules)
            
            if is_target and isinstance(module, nn.Linear):
                # Replace this Linear layer
                setattr(parent_module, name, replace_with_svd_residual(module, r))
            elif isinstance(module, nn.Linear):
                # Not a target, skip
                pass
            else:
                # Recursively process child modules
                _apply_to_module(module, full_name)
    
    _apply_to_module(model)
    
    # After replacing, set requires_grad for residual components
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
            param.requires_grad = True
        elif any(x in param_name for x in ['weight_main', 'S_r', 'U_r', 'V_r']):
            param.requires_grad = False
        else:
            # Freeze other backbone parameters by default
            # (classifier will be unfrozen separately)
            param.requires_grad = False
    
    return model


class DinoSVDModel(BaseModel):
    """
    DINO-based deepfake detector with SVD residual learning.
    
    Uses DINOv2 as backbone with SVD decomposition applied to attention layers.
    The principal SVD components preserve pre-trained knowledge while
    residual components learn to detect forgeries.
    """
    
    def __init__(self, 
                 num_classes: int = 2,
                 hidden_dims: List[int] = None,
                 dropout: float = 0.3,
                 dino_model: str = 'dinov2_vitb14',
                 svd_rank: int = None,
                 target_modules: List[str] = None):
        """
        Initialize DinoSVDModel.
        
        Args:
            num_classes: Number of output classes
            hidden_dims: Hidden dimensions for classifier head
            dropout: Dropout rate
            dino_model: DINO model variant ('dinov2_vitb14', 'dinov2_vitl14', etc.)
            svd_rank: Number of singular values to keep fixed.
                     If None, defaults to feature_dim - 1 (keep all but one)
            target_modules: Module name patterns for SVD replacement.
                          If None, targets attention ('attn') modules.
        """
        super().__init__(num_classes)
        
        self.dino_model_name = dino_model
        
        # Determine feature dimension based on model
        if 'vitb' in dino_model:
            self.feature_dim = 768
        elif 'vitl' in dino_model:
            self.feature_dim = 1024
        elif 'vitg' in dino_model:
            self.feature_dim = 1536
        else:
            self.feature_dim = 768  # Default to ViT-B
        
        # Set SVD rank (default: keep all but 1 singular value fixed)
        self.svd_rank = svd_rank if svd_rank is not None else self.feature_dim - 1
        self.target_modules = target_modules if target_modules is not None else ['attn']
        
        # Load DINO backbone
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            dino_model
        )
        
        # Apply SVD residual decomposition to attention layers
        self.backbone = apply_svd_residual_to_attn(
            self.backbone, 
            r=self.svd_rank,
            target_modules=self.target_modules
        )
        
        # Build classifier head
        if hidden_dims is None:
            hidden_dims = [256]
        
        layers = []
        input_size = self.feature_dim
        
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
        
        # Loss function
        self.loss_func = nn.CrossEntropyLoss()
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the backbone."""
        features = self.backbone(x)
        
        # Mean-pool patch tokens (skip CLS if present)
        if features.dim() == 3:
            features = features[:, 1:, :].mean(dim=1)
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        features = self.get_features(x)
        logits = self.classifier(features)
        return logits
    
    def forward_with_features(self, x: torch.Tensor) -> dict:
        """
        Forward pass returning both logits and features.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Dictionary with 'cls' (logits), 'prob' (probabilities), and 'feat' (features)
        """
        features = self.get_features(x)
        logits = self.classifier(features)
        prob = torch.softmax(logits, dim=1)[:, 1]
        
        return {
            'cls': logits,
            'prob': prob,
            'feat': features
        }
    
    def compute_orthogonal_loss(self) -> torch.Tensor:
        """
        Compute total orthogonality loss across all SVD layers.
        
        Returns:
            Scalar tensor with mean orthogonality loss
        """
        total_loss = 0.0
        count = 0
        
        for module in self.backbone.modules():
            if isinstance(module, SVDResidualLinear):
                total_loss = total_loss + module.compute_orthogonal_loss()
                count += 1
        
        if count > 0:
            return total_loss / count
        return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def compute_keepsv_loss(self) -> torch.Tensor:
        """
        Compute total singular value preservation loss.
        
        Returns:
            Scalar tensor with mean keepsv loss
        """
        total_loss = 0.0
        count = 0
        
        for module in self.backbone.modules():
            if isinstance(module, SVDResidualLinear):
                total_loss = total_loss + module.compute_keepsv_loss()
                count += 1
        
        if count > 0:
            return total_loss / count
        return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def compute_weight_loss(self) -> torch.Tensor:
        """
        Compute weight diversity loss using SVD on summed weights.
        
        This encourages diverse weight updates across layers.
        
        Returns:
            Scalar tensor with negative mean singular value
        """
        weight_sum_dict = {}
        num_weight_dict = {}
        
        for module in self.backbone.modules():
            if isinstance(module, SVDResidualLinear):
                weight_curr = module.compute_current_weight()
                size_key = str(weight_curr.size())
                
                if size_key not in weight_sum_dict:
                    weight_sum_dict[size_key] = weight_curr
                    num_weight_dict[size_key] = 1
                else:
                    weight_sum_dict[size_key] = weight_sum_dict[size_key] + weight_curr
                    num_weight_dict[size_key] += 1
        
        if not weight_sum_dict:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        loss = 0.0
        for k in weight_sum_dict.keys():
            _, S_sum, _ = torch.linalg.svd(weight_sum_dict[k], full_matrices=False)
            loss = loss - torch.mean(S_sum)
        
        return loss / len(weight_sum_dict)
    
    def get_svd_params(self) -> List[nn.Parameter]:
        """Get trainable SVD residual parameters."""
        params = []
        for name, param in self.backbone.named_parameters():
            if any(x in name for x in ['S_residual', 'U_residual', 'V_residual']):
                if param.requires_grad:
                    params.append(param)
        return params
    
    def get_classifier_params(self) -> List[nn.Parameter]:
        """Get classifier parameters."""
        return list(self.classifier.parameters())
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all trainable parameters."""
        return self.get_svd_params() + self.get_classifier_params()
    
    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_svd_residuals(self):
        """Unfreeze only the SVD residual parameters."""
        for name, param in self.backbone.named_parameters():
            if any(x in name for x in ['S_residual', 'U_residual', 'V_residual']):
                param.requires_grad = True
    
    def print_trainable_params(self):
        """Print information about trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        svd_params = sum(p.numel() for p in self.get_svd_params())
        classifier_params = sum(p.numel() for p in self.get_classifier_params())
        
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  - SVD Residual Params: {svd_params:,}")
        print(f"  - Classifier Params: {classifier_params:,}")
