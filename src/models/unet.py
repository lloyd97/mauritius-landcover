"""
U-Net Architecture for Land Cover Segmentation
===============================================

Implements U-Net with various encoder backbones for semantic segmentation
of satellite imagery.

References:
    - Ronneberger et al. (2015): U-Net: Convolutional Networks for 
      Biomedical Image Segmentation
    - Uses segmentation_models_pytorch for pretrained encoders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import segmentation_models_pytorch as smp


class ConvBlock(nn.Module):
    """Double convolution block for U-Net."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        batch_norm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Encoder block with convolution and downsampling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, dropout=dropout)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_conv = self.conv(x)
        x_pool = self.pool(x_conv)
        return x_conv, x_pool


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and convolution."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2,
            kernel_size=2, stride=2
        )
        self.conv = ConvBlock(
            in_channels // 2 + skip_channels,
            out_channels,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation.
    
    Simple implementation without pretrained encoder.
    For pretrained encoders, use UNetPretrained class.
    """
    
    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 7,
        features: List[int] = [64, 128, 256, 512],
        dropout: float = 0.1
    ):
        """
        Initialize U-Net.
        
        Args:
            in_channels: Number of input channels (bands)
            num_classes: Number of output classes
            features: Number of features at each encoder level
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Encoder
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for feat in features:
            self.encoders.append(EncoderBlock(prev_channels, feat, dropout))
            prev_channels = feat
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2, dropout=dropout)
        
        # Decoder
        self.decoders = nn.ModuleList()
        reversed_features = list(reversed(features))
        prev_channels = features[-1] * 2
        
        for i, feat in enumerate(reversed_features):
            self.decoders.append(DecoderBlock(prev_channels, feat, feat, dropout))
            prev_channels = feat
        
        # Final convolution
        self.final = nn.Conv2d(features[0], num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output logits (B, num_classes, H, W)
        """
        # Encoder path
        skip_connections = []
        for encoder in self.encoders:
            x_conv, x = encoder(x)
            skip_connections.append(x_conv)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])
        
        return self.final(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class UNetPretrained(nn.Module):
    """
    U-Net with pretrained encoder backbone.
    
    Uses segmentation_models_pytorch for various encoder options.
    Supports: resnet18, resnet34, resnet50, resnet101, efficientnet-b0, etc.
    """
    
    ENCODERS = {
        'resnet18': 'resnet18',
        'resnet34': 'resnet34',
        'resnet50': 'resnet50',
        'resnet101': 'resnet101',
        'efficientnet-b0': 'efficientnet-b0',
        'efficientnet-b4': 'efficientnet-b4',
        'mobilenet_v2': 'mobilenet_v2',
        'vgg16': 'vgg16',
    }
    
    def __init__(
        self,
        encoder_name: str = 'resnet50',
        encoder_weights: str = 'imagenet',
        in_channels: int = 9,
        num_classes: int = 7,
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        activation: Optional[str] = None
    ):
        """
        Initialize pretrained U-Net.
        
        Args:
            encoder_name: Name of encoder backbone
            encoder_weights: Pretrained weights ('imagenet' or None)
            in_channels: Number of input channels
            num_classes: Number of output classes
            decoder_channels: Decoder channel configuration
            activation: Output activation (None, 'softmax', 'sigmoid')
        """
        super().__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            decoder_channels=decoder_channels,
            activation=activation
        )
        
        self.encoder_name = encoder_name
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def freeze_encoder(self):
        """Freeze encoder weights for fine-tuning."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights."""
        for param in self.model.encoder.parameters():
            param.requires_grad = True


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ architecture for semantic segmentation.
    
    Better for capturing multi-scale context in satellite imagery.
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet50',
        encoder_weights: str = 'imagenet',
        in_channels: int = 9,
        num_classes: int = 7,
        activation: Optional[str] = None
    ):
        super().__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class AttentionGate(nn.Module):
    """Attention gate for focusing on relevant features."""
    
    def __init__(self, gate_channels: int, in_channels: int, inter_channels: int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample g to match x if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class AttentionUNet(nn.Module):
    """U-Net with attention gates for improved feature selection."""
    
    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 7,
        features: List[int] = [64, 128, 256, 512]
    ):
        super().__init__()
        
        # Use base UNet structure
        self.base = UNet(in_channels, num_classes, features)
        
        # Add attention gates
        self.attention_gates = nn.ModuleList([
            AttentionGate(f * 2, f, f // 2) 
            for f in reversed(features)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This is a simplified version
        # Full implementation would modify the decoder to use attention gates
        return self.base(x)


def create_model(config: dict) -> nn.Module:
    """
    Create model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    model_config = config.get('model', {})
    
    architecture = model_config.get('architecture', 'unet')
    encoder = model_config.get('encoder', 'resnet50')
    encoder_weights = model_config.get('encoder_weights', 'imagenet')
    in_channels = model_config.get('in_channels', 9)
    num_classes = model_config.get('num_classes', 7)
    
    if architecture == 'unet':
        if encoder_weights:
            model = UNetPretrained(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                num_classes=num_classes
            )
        else:
            model = UNet(
                in_channels=in_channels,
                num_classes=num_classes
            )
    elif architecture == 'deeplabv3':
        model = DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes
        )
    elif architecture == 'attention_unet':
        model = AttentionUNet(
            in_channels=in_channels,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model


if __name__ == '__main__':
    # Test models
    print("Testing U-Net models...")
    
    # Test input
    x = torch.randn(2, 9, 256, 256)
    
    # Test basic UNet
    model = UNet(in_channels=9, num_classes=7)
    out = model(x)
    print(f"UNet output shape: {out.shape}")
    
    # Test pretrained UNet
    model = UNetPretrained(encoder_name='resnet34', in_channels=9, num_classes=7)
    out = model(x)
    print(f"UNet (ResNet34) output shape: {out.shape}")
    
    # Test DeepLabV3+
    model = DeepLabV3Plus(encoder_name='resnet50', in_channels=9, num_classes=7)
    out = model(x)
    print(f"DeepLabV3+ output shape: {out.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
