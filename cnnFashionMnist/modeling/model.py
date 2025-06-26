from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FashionMNISTCNN(nn.Module):
    """CNN model for Fashion-MNIST classification."""

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super(FashionMNISTCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)

        ### DATA FLOW VISUALIZATION - STANDARD MODEL
        # Input: (batch, 1, 28, 28)           # Grayscale Fashion-MNIST images
        # ↓ CONV BLOCK 1
        # ↓ conv1 (1→32) + bn + relu
        # (batch, 32, 28, 28)                 # 32 feature maps, same size
        # ↓ conv2 (32→32) + bn + relu
        # (batch, 32, 28, 28)                 # Refined 32 feature maps
        # ↓ maxpool(2x2) + dropout(0.25)
        # (batch, 32, 14, 14)                 # Downsampled by 2x

        # Second convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)

        # ↓ CONV BLOCK 2
        # ↓ conv3 (32→64) + bn + relu
        # (batch, 64, 14, 14)                 # 64 feature maps
        # ↓ conv4 (64→64) + bn + relu
        # (batch, 64, 14, 14)                 # Refined 64 feature maps
        # ↓ maxpool(2x2) + dropout(0.25)
        # (batch, 64, 7, 7)                   # Downsampled by 2x

        # Third convolutional block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)

        # ↓ CONV BLOCK 3
        # ↓ conv5 (64→128) + bn + relu
        # (batch, 128, 7, 7)                  # 128 feature maps
        # ↓ maxpool(2x2) + dropout(0.25)
        # (batch, 128, 3, 3)                  # Final spatial reduction

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)

        # ↓ CLASSIFICATION HEAD
        # ↓ flatten
        # (batch, 1152)                       # 128 * 3 * 3 = 1152 features
        # ↓ fc1 (1152→512) + bn + relu + dropout(0.5)
        # (batch, 512)                        # First hidden layer
        # ↓ fc2 (512→256) + bn + relu + dropout(0.5)
        # (batch, 256)                        # Second hidden layer
        # ↓ fc3 (256→10)
        # (batch, 10)                         # Final class logits

    def forward(self, x):
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))  # Conv → BatchNorm → ReLU
        x = F.relu(self.bn2(self.conv2(x)))  # Conv → BatchNorm → ReLU
        x = self.pool1(x)  # MaxPool
        x = self.dropout1(x)  # Dropout

        # Second conv block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Third conv block
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # [batch_size, 3*3*128]

        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout4(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout5(x)
        x = self.fc3(x)  # Final output (no activation)

        return x

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": "FashionMNISTCNN",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        }


class ChannelAttention(nn.Module):
    """Channel attention mechanism (Squeeze-and-Excitation)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        attention = self.fc(x).view(b, c, 1, 1)
        return x * attention


class AttentionFashionMNISTCNN(nn.Module):
    """CNN model with Channel Attention for Fashion-MNIST classification."""

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super(AttentionFashionMNISTCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)

        # Second convolutional block with attention
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.attention3 = ChannelAttention(64)  # ← Attention after conv3
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)

        # Third convolutional block with attention
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.attention5 = ChannelAttention(128)  # ← Attention after conv5
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second conv block with attention
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.attention3(x)  # ← Apply attention
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Third conv block with attention
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.attention5(x)  # ← Apply attention
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout4(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout5(x)
        x = self.fc3(x)

        return x

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": "AttentionFashionMNISTCNN",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
        }


# 🔍 Multi-Scale Implementation:
class MultiScaleBlock(nn.Module):
    """Multi-scale feature extraction block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Divide output channels among different scales
        branch_channels = out_channels // 4

        # Different kernel sizes for different scales
        self.branch_1x1 = nn.Conv2d(in_channels, branch_channels, 1, 1, 0)  # Point-wise
        self.branch_3x3 = nn.Conv2d(in_channels, branch_channels, 3, 1, 1)  # Fine details
        self.branch_5x5 = nn.Conv2d(in_channels, branch_channels, 5, 1, 2)  # Medium patterns
        self.branch_7x7 = nn.Conv2d(in_channels, branch_channels, 7, 1, 3)  # Global context

        # Batch normalization for each branch
        self.bn_1x1 = nn.BatchNorm2d(branch_channels)
        self.bn_3x3 = nn.BatchNorm2d(branch_channels)
        self.bn_5x5 = nn.BatchNorm2d(branch_channels)
        self.bn_7x7 = nn.BatchNorm2d(branch_channels)

        # Combine features from different scales
        self.combine = nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        self.bn_combine = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Process input through different scales in parallel
        feat_1x1 = F.relu(self.bn_1x1(self.branch_1x1(x)))
        feat_3x3 = F.relu(self.bn_3x3(self.branch_3x3(x)))
        feat_5x5 = F.relu(self.bn_5x5(self.branch_5x5(x)))
        feat_7x7 = F.relu(self.bn_7x7(self.branch_7x7(x)))

        # Concatenate all scales
        combined = torch.cat([feat_1x1, feat_3x3, feat_5x5, feat_7x7], dim=1)

        # Mix and refine combined features
        output = F.relu(self.bn_combine(self.combine(combined)))

        return output


class MultiScaleFashionCNN(nn.Module):
    """CNN with Multi-scale feature extraction."""

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)

        # Multi-scale blocks replace single convolutions
        self.multi_scale1 = MultiScaleBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)

        self.multi_scale2 = MultiScaleBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)

        # Classification head
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Initial feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Multi-scale feature extraction
        x = self.multi_scale1(x)  # Captures multiple detail levels
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.multi_scale2(x)  # Even richer feature combinations
        x = self.pool3(x)
        x = self.dropout3(x)

        # Classification
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout4(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout5(x)
        x = self.fc3(x)

        return x

    def get_model_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_name": "MultiScaleFashionCNN",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
        }


# ⚡ Enhanced SE Block Implementation:
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block - enhanced channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()

        reduced_channels = max(channels // reduction, 1)  # Ensure at least 1 channel

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Squeeze: Global information
        y = self.squeeze(x).view(b, c)

        # Excitation: Channel importance
        y = self.excitation(y).view(b, c, 1, 1)

        # Scale original features
        return x * y


class SEFashionCNN(nn.Module):
    """CNN with enhanced SE (Squeeze-and-Excitation) blocks."""

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.se1 = SEBlock(32)  # SE after first block
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.se2 = SEBlock(64)  # SE after second block
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)

        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(128)
        self.se3 = SEBlock(128)  # SE after third block
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)

        # Classification head
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # First block with SE
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se1(x)  # Enhanced channel attention
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second block with SE
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.se2(x)  # Focus on important channels
        x = self.pool2(x)
        x = self.dropout2(x)

        # Third block with SE
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.se3(x)  # Final channel refinement
        x = self.pool3(x)
        x = self.dropout3(x)

        # Classification
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout4(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout5(x)
        x = self.fc3(x)

        return x

    def get_model_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_name": "SEFashionCNN",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
        }


# 🚀 Residual Connections Implementation:
class ResidualBlock(nn.Module):
    """Residual block with batch normalization and optional SE."""

    def __init__(self, channels, use_se=True):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)

        # Optional SE block
        self.se = SEBlock(channels) if use_se else nn.Identity()

    def forward(self, x):
        residual = x  # Store input for skip connection

        # First convolution
        out = F.relu(self.bn1(self.conv1(x)))

        # Second convolution
        out = self.bn2(self.conv2(out))

        # SE attention (optional)
        out = self.se(out)

        # Residual connection + activation
        out = F.relu(out + residual)  # Skip connection!

        return out


class ResidualFashionCNN(nn.Module):
    """CNN with Residual connections for deeper learning."""

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3)  # Larger initial conv
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, 2, 1)

        # Residual blocks
        self.res_block1 = ResidualBlock(64, use_se=True)
        self.res_block2 = ResidualBlock(64, use_se=True)

        # Transition to more channels
        self.conv_transition = nn.Conv2d(64, 128, 1)  # 1x1 conv for channel expansion
        self.bn_transition = nn.BatchNorm2d(128)

        self.res_block3 = ResidualBlock(128, use_se=True)
        self.res_block4 = ResidualBlock(128, use_se=True)

        # Final layers
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Initial processing
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, 14, 14]
        x = self.pool1(x)  # [B, 64, 7, 7]

        # Residual learning
        x = self.res_block1(x)  # Deep feature learning with skip connections
        x = self.res_block2(x)  # Even deeper, gradients flow easily

        # Channel expansion
        x = F.relu(self.bn_transition(self.conv_transition(x)))  # [B, 128, 7, 7]

        # More residual learning
        x = self.res_block3(x)  # High-level features
        x = self.res_block4(x)  # Semantic features

        # Classification
        x = self.global_pool(x)  # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 128]
        x = self.dropout(x)
        x = self.fc(x)  # [B, num_classes]

        return x

    def get_model_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_name": "ResidualFashionCNN",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
        }


# 🔥 Combined Advanced Implementation:
class AdvancedFashionCNN(nn.Module):
    """Combined Multi-scale + SE + Residual for maximum performance."""

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super().__init__()

        # Multi-scale input processing
        self.multi_scale_input = MultiScaleBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Residual blocks with SE attention
        self.res_se_block1 = ResidualBlock(64, use_se=True)
        self.res_se_block2 = ResidualBlock(64, use_se=True)

        # Transition and more residual learning
        self.conv_transition = nn.Conv2d(64, 128, 1)
        self.bn_transition = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.res_se_block3 = ResidualBlock(128, use_se=True)
        self.res_se_block4 = ResidualBlock(128, use_se=True)

        # Final classification
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Multi-scale feature extraction
        x = self.multi_scale_input(x)  # Captures fine + coarse details
        x = self.pool1(x)

        # Deep residual learning with SE attention
        x = self.res_se_block1(x)  # Skip connections + channel attention
        x = self.res_se_block2(x)  # Even deeper learning

        # Channel expansion
        x = F.relu(self.bn_transition(self.conv_transition(x)))
        x = self.pool2(x)

        # High-level feature learning
        x = self.res_se_block3(x)  # Semantic features
        x = self.res_se_block4(x)  # Decision-ready features

        # Classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def get_model_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_name": "AdvancedFashionCNN",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
        }


class SimpleFashionMNISTCNN(nn.Module):
    """Simpler CNN model for Fashion-MNIST classification."""

    def __init__(self, num_classes: int = 10):
        super(SimpleFashionMNISTCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

        ### DATA FLOW VISUALIZATION - SIMPLE MODEL
        # Input: (batch, 1, 28, 28)           # Grayscale Fashion-MNIST images
        # ↓ conv1 (1→16, 5x5) + relu
        # (batch, 16, 28, 28)                 # 16 feature maps, larger kernels
        # ↓ maxpool(2x2)
        # (batch, 16, 14, 14)                 # Downsampled by 2x
        #
        # ↓ conv2 (16→32, 5x5) + relu
        # (batch, 32, 14, 14)                 # 32 feature maps
        # ↓ maxpool(2x2)
        # (batch, 32, 7, 7)                   # Downsampled by 2x
        #
        # ↓ flatten
        # (batch, 1568)                       # 32 * 7 * 7 = 1568 features
        # ↓ fc1 (1568→128) + relu + dropout(0.5)
        # (batch, 128)                        # Single hidden layer
        # ↓ fc2 (128→10)
        # (batch, 10)                         # Final class logits
        ###

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": "SimpleFashionMNISTCNN",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
        }


def create_model(model_type: str = "standard", num_classes: int = 10, **kwargs) -> nn.Module:
    """Factory function to create different model architectures."""
    if model_type == "standard":
        return FashionMNISTCNN(num_classes=num_classes, **kwargs)
    elif model_type == "attention":
        return AttentionFashionMNISTCNN(num_classes=num_classes, **kwargs)
    elif model_type == "multiscale":  # ← NEW
        return MultiScaleFashionCNN(num_classes=num_classes, **kwargs)
    elif model_type == "se":  # ← NEW
        return SEFashionCNN(num_classes=num_classes, **kwargs)
    elif model_type == "residual":  # ← NEW
        return ResidualFashionCNN(num_classes=num_classes, **kwargs)
    elif model_type == "advanced":  # ← NEW
        return AdvancedFashionCNN(num_classes=num_classes, **kwargs)
    elif model_type == "simple":
        return SimpleFashionMNISTCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def initialize_weights(model: nn.Module):
    """Initialize model weights using Xavier/Glorot initialization."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:  # ← ADD THIS CHECK
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:  # ← ADD THIS CHECK
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
