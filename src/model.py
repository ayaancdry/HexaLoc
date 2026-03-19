# for cbam at every layer
#!/usr/bin/env python3
"""
Neural network architecture for plane-based localization.
Uses 2D CNN to process single XY plane frames and predict poses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from src.model_CBAM import CBAM


class SpatialPyramidPooling(nn.Module):
    """Spatial Pyramid Pooling layer for multi-scale feature extraction."""
    
    def __init__(self, pool_sizes=[1, 2, 4]):
        super(SpatialPyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        features = []
        for size in self.pool_sizes:
            # Pool to specific size (e.g., 4x4)
            pool = F.adaptive_max_pool2d(x, output_size=(size, size))
            # Flatten: (B, C, size, size) -> (B, C*size*size)
            flat = pool.view(x.size(0), -1)
            features.append(flat)
        # Concatenate all features: (B, C*1 + C*4 + C*16) = (B, C*21)
        return torch.cat(features, dim=1)


class MultiScaleCrossAttention(nn.Module):
    """
    Cross-attention between multi-scale features.
    Projects features from different scales to a common dimension,
    applies multi-head self-attention, then projects back.
    """
    
    def __init__(self, dims=[640, 1280, 2560], common_dim=256, num_heads=8, dropout=0.1):
        super(MultiScaleCrossAttention, self).__init__()
        
        self.num_scales = len(dims)
        self.common_dim = common_dim
        
        # Project each scale to common dimension
        self.proj_in = nn.ModuleList([
            nn.Linear(dim, common_dim) for dim in dims
        ])
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm before attention (pre-norm)
        self.norm1 = nn.LayerNorm(common_dim)
        self.norm2 = nn.LayerNorm(common_dim)
        
        # FFN after attention
        self.ffn = nn.Sequential(
            nn.Linear(common_dim, common_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(common_dim * 4, common_dim),
            nn.Dropout(dropout)
        )
        
        # Project back to original dimensions
        self.proj_out = nn.ModuleList([
            nn.Linear(common_dim, dim) for dim in dims
        ])
        
        # Learnable scale tokens to distinguish scales
        self.scale_embeddings = nn.Parameter(torch.randn(self.num_scales, common_dim) * 0.02)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, features_list):
        batch_size = features_list[0].shape[0]
        
        # Project all features to common dimension and add scale embeddings
        projected = []
        for i, (feat, proj) in enumerate(zip(features_list, self.proj_in)):
            proj_feat = proj(feat)  # [B, common_dim]
            proj_feat = proj_feat + self.scale_embeddings[i].unsqueeze(0)  # Add scale embedding
            projected.append(proj_feat)
        
        # Stack as sequence: [B, num_scales, common_dim]
        x = torch.stack(projected, dim=1)
        
        # Pre-norm + Multi-head self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out  # Residual connection
        
        # Pre-norm + FFN
        x = x + self.ffn(self.norm2(x))
        
        # Split back and project to original dimensions
        refined = []
        for i, proj_out in enumerate(self.proj_out):
            scale_feat = x[:, i, :]  # [B, common_dim]
            refined_feat = proj_out(scale_feat)  # [B, original_dim]
            # Residual connection with original features
            refined_feat = refined_feat + features_list[i]
            refined.append(refined_feat)
        
        return refined


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: int = 1, downsample: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # Add this
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)  # Add this
        
        self.downsample = downsample
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.01)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)  # Add this
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)  # Add this
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class BottleneckBlock(nn.Module):
    """
    Bottleneck ResNet block with residual connection.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: int = 1, downsample: Optional[nn.Module] = None):
        """
        Initialize bottleneck ResNet block.
        """
        super(BottleneckBlock, self).__init__()
        
        # 1x1 convolution to reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False)
        # self.ln1 = nn.LayerNorm(out_channels // 4)
        
        # 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        # self.ln2 = nn.LayerNorm(out_channels // 4)
        
        # 1x1 convolution to expand channels
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False)
        # self.ln3 = nn.LayerNorm(out_channels)
        
        self.downsample = downsample
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.01)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        # Apply LayerNorm
        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        # out = self.ln1(out)
        out = out.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        out = self.relu(out)
        
        out = self.conv2(out)
        # Apply LayerNorm
        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        # out = self.ln2(out)
        out = out.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        out = self.relu(out)
        
        out = self.conv3(out)
        # Apply LayerNorm
        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        # out = self.ln3(out)
        out = out.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        if self.downsample is not None:
            # Apply downsample (conv), then LayerNorm
            identity = self.downsample[0](x)  # Conv2d
            # B, C, H, W = identity.shape
            # identity = identity.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
            # identity = self.downsample[1](identity)  # LayerNorm
            # identity = identity.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        out += identity
        out = self.relu(out)
        
        return out

class PlaneEncoder(nn.Module):
    """
    ResNet-like encoder network with multi-scale feature extraction.
    Extracts features from layers 2, 3, and 4, applies SPP to each,
    optionally applies cross-scale attention, and concatenates for 
    rich multi-scale representation.
    """
    
    def __init__(self, 
                 num_planes: int = 50,
                 grid_size: int = 512,
                 feature_dim: int = 256,
                 block_type: str = 'basic',
                 use_cbam: bool = False,
                 use_cross_attention: bool = False,
                 cross_attn_dim: int = 256,
                 cross_attn_heads: int = 8):
        """
        Initialize ResNet-like plane encoder with multi-scale feature extraction.
        """
        super(PlaneEncoder, self).__init__()
        
        self.num_planes = num_planes
        self.grid_size = grid_size
        self.feature_dim = feature_dim
        self.block_type = block_type
        self.use_cbam = use_cbam
        self.use_cross_attention = use_cross_attention
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(num_planes, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.01)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers - channel progression: 32 -> 64 -> 128 -> 256 -> 512
        self.layer1 = self._make_layer(32, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Multi-Stage CBAM: Applied after every layer to refine features at all scales
        if self.use_cbam:
            self.cbam1 = CBAM(64)   # For Layer 1 output
            self.cbam2 = CBAM(128)  # For Layer 2 output
            self.cbam3 = CBAM(256)  # For Layer 3 output
            self.cbam4 = CBAM(512)  # For Layer 4 output
        
        # Multi-scale SPP branches for layers 2, 3, and 4
        # Using pool_sizes=[1, 2] for each branch
        self.spp2 = SpatialPyramidPooling(pool_sizes=[1, 2])  # 128 * (1 + 4) = 640
        self.spp3 = SpatialPyramidPooling(pool_sizes=[1, 2])  # 256 * (1 + 4) = 1280
        self.spp4 = SpatialPyramidPooling(pool_sizes=[1, 2])  # 512 * (1 + 4) = 2560
        
        # SPP output dimensions
        spp_dims = [128 * 5, 256 * 5, 512 * 5]  # [640, 1280, 2560]
        total_features = sum(spp_dims)  # 4480
        
        # Cross-scale attention: allows different scales to communicate
        if self.use_cross_attention:
            self.cross_attention = MultiScaleCrossAttention(
                dims=spp_dims,
                common_dim=cross_attn_dim,
                num_heads=cross_attn_heads,
                dropout=0.1
            )
        
        # Global average pooling (kept for compatibility, not used)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature projection - maps concatenated multi-scale features to feature_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(total_features, feature_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.01)
        )
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        """
        Constructs a sequential "layer" consisting of multiple residual blocks.
        """
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        if self.block_type == 'basic':
            layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
            for _ in range(1, blocks):
                layers.append(BasicBlock(out_channels, out_channels))
        else:  # bottleneck
            layers.append(BottleneckBlock(in_channels, out_channels, stride, downsample))
            for _ in range(1, blocks):
                layers.append(BottleneckBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, planes):
        """
        Forward pass with multi-scale feature extraction and optional cross-attention.
        """
        # Initial convolution and pooling
        x = self.conv1(planes)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Layer 1 (no branch from here)
        x = self.layer1(x)
        if self.use_cbam:
            x = self.cbam1(x)
        
        # Layer 2 - Extract features here (128 channels)
        x = self.layer2(x)
        if self.use_cbam:
            x = self.cbam2(x)
        feat2 = self.spp2(x)  # [B, 128*5] = [B, 640]
        
        # Layer 3 - Extract features here (256 channels)
        x = self.layer3(x)
        if self.use_cbam:
            x = self.cbam3(x)
        feat3 = self.spp3(x)  # [B, 256*5] = [B, 1280]
        
        # Layer 4 - Extract features here (512 channels)
        x = self.layer4(x)
        if self.use_cbam:
            x = self.cbam4(x)
        feat4 = self.spp4(x)  # [B, 512*5] = [B, 2560]
        
        # Apply cross-scale attention if enabled
        # This allows different scales to share context before concatenation
        if self.use_cross_attention:
            feat2, feat3, feat4 = self.cross_attention([feat2, feat3, feat4])
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat([feat2, feat3, feat4], dim=1)  # [B, 4480]
        
        # Feature projection
        features = self.feature_proj(multi_scale_features)  # [B, feature_dim]
        
        return features
        
class PoseDecoder(nn.Module):
    """
    Decoder network for predicting poses from encoded features.
    Uses MLP for pose regression.
    """
    
    def __init__(self, 
                 feature_dim: int = 256,
                 hidden_dim: int = 512,
                 dropout_rate: float = 0.1,
                 num_layers: int = 4):
        """
        Initialize pose decoder.
        """
        super(PoseDecoder, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        
        # Build deeper MLP with dropout
        layers = []
        
        # Input layer
        layers.append(nn.Linear(feature_dim, hidden_dim))
        layers.append(nn.LeakyReLU(inplace=True, negative_slope=0.01))
        #layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers with decreasing dimensions
        for i in range(num_layers - 1):
            current_dim = hidden_dim
            next_dim = hidden_dim 
            
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.LeakyReLU(inplace=True, negative_slope=0.01))
            #layers.append(nn.Dropout(dropout_rate))
        
        # Output layer (no dropout for final prediction)
        final_dim = hidden_dim 
        layers.append(nn.Linear(final_dim, 6))  # 6DOF
        
        self.pose_head = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features):
        """
        Forward pass through pose decoder.
        """
        # Predict poses
        poses = self.pose_head(features)
        
        return poses

class PlaneLocalizationNet(nn.Module):
    """
    Complete ResNet-based neural network for plane-based localization.
    Combines ResNet encoder and pose decoder for single frame processing.
    Supports both CBAM (intra-scale) and cross-attention (inter-scale) mechanisms.
    """
    
    def __init__(self,
                 num_planes: int = 50,
                 grid_size: int = 512,
                 feature_dim: int = 256,
                 hidden_dim: int = 512,
                 block_type: str = 'basic',
                 dropout_rate: float = 0.1,
                 num_layers: int = 4,
                 use_cbam: bool = False,
                 use_cross_attention: bool = False,
                 cross_attn_dim: int = 256,
                 cross_attn_heads: int = 8):
        """
        Initialize complete ResNet-based localization network.
        """
        super(PlaneLocalizationNet, self).__init__()
        
        self.num_planes = num_planes
        self.grid_size = grid_size
        self.block_type = block_type
        
        # ResNet encoder with optional CBAM and cross-attention
        self.encoder = PlaneEncoder(
            num_planes=num_planes,
            grid_size=grid_size,
            feature_dim=feature_dim,
            block_type=block_type,
            use_cbam=use_cbam,
            use_cross_attention=use_cross_attention,
            cross_attn_dim=cross_attn_dim,
            cross_attn_heads=cross_attn_heads
        )
        
        self.decoder = PoseDecoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            num_layers=num_layers
        )
    
    def forward(self, planes):
        """
        Forward pass through complete network.
        """
        # Encode planes to features
        features = self.encoder(planes)
        
        # Decode features to poses
        poses = self.decoder(features)
        
        return poses
    
    def predict_frame(self, planes, return_features=False):
        """
        Predict pose for a single frame with optional feature return.
        """
        features = self.encoder(planes)
        poses = self.decoder(features)
        
        if return_features:
            return poses, features
        return poses

def create_model(config):
    """
    Create ResNet-based model from configuration dictionary.
    """
    model = PlaneLocalizationNet(
        num_planes=config.get('num_planes', 50),
        grid_size=config.get('grid_size', 512),
        feature_dim=config.get('feature_dim', 256),
        hidden_dim=config.get('hidden_dim', 512),
        block_type=config.get('block_type', 'basic'),
        dropout_rate=config.get('dropout_rate', 0.1),
        num_layers=config.get('num_layers', 4),
        use_cbam=config.get('use_cbam', False),
        use_cross_attention=config.get('use_cross_attention', False),
        cross_attn_dim=config.get('cross_attn_dim', 256),
        cross_attn_heads=config.get('cross_attn_heads', 8)
    )
    
    return model

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the ResNet-based model with CBAM and cross-attention
    config = {
        'num_planes': 50,
        'grid_size': 512,
        'feature_dim': 256,
        'hidden_dim': 512,
        'block_type': 'basic',
        'use_cbam': True,
        'use_cross_attention': True,
        'cross_attn_dim': 256,
        'cross_attn_heads': 8
    }
    
    model = create_model(config)
    
    # Test with dummy data
    batch_size = 2
    num_planes = config['num_planes']
    grid_size = config['grid_size']
    
    dummy_planes = torch.randn(batch_size, num_planes, grid_size, grid_size)
    
    print(f"ResNet Model parameters: {count_parameters(model):,}")
    print(f"Input shape: {dummy_planes.shape}")
    print(f"Block type: {config['block_type']}")
    print(f"Use CBAM: {config['use_cbam']}")
    print(f"Use Cross-Attention: {config['use_cross_attention']}")
    
    with torch.no_grad():
        output = model(dummy_planes)
        print(f"Output shape: {output.shape}")
        print(f"Expected: [batch_size, 6] for 6DOF (3 translation + 3 log-quaternion: tx, ty, tz, log_qx, log_qy, log_qz)")
    
    # Test without cross-attention
    config_no_cross = config.copy()
    config_no_cross['use_cross_attention'] = False
    model_no_cross = create_model(config_no_cross)
    
    print(f"\nModel without cross-attention parameters: {count_parameters(model_no_cross):,}")
    
    with torch.no_grad():
        output_no_cross = model_no_cross(dummy_planes)
        print(f"Output shape (no cross-attn): {output_no_cross.shape}")
    
    # Test bottleneck version
    config_bottleneck = config.copy()
    config_bottleneck['block_type'] = 'bottleneck'
    model_bottleneck = create_model(config_bottleneck)
    
    print(f"\nBottleneck Model parameters: {count_parameters(model_bottleneck):,}")
    
    with torch.no_grad():
        output_bottleneck = model_bottleneck(dummy_planes)
        print(f"Bottleneck output shape: {output_bottleneck.shape}")
    
    print("\nResNet model test completed successfully!")
