"""
LSTM-UNet for Temporal Land Cover Analysis
==========================================

Combines U-Net semantic segmentation with LSTM for processing
multi-temporal satellite imagery sequences.

This architecture is particularly effective for:
    - Change detection
    - Seasonal variation handling
    - Time-series land cover mapping

Reference:
    Sefrin et al. (2021): Deep Learning for Land Cover Change Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import segmentation_models_pytorch as smp


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell.
    
    Processes spatial-temporal data by combining convolution operations
    with LSTM gating mechanisms.
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3
    ):
        """
        Initialize ConvLSTM cell.
        
        Args:
            input_channels: Number of input channels
            hidden_channels: Number of hidden state channels
            kernel_size: Convolution kernel size
        """
        super().__init__()
        
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        # Combined convolution for all gates
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,  # i, f, g, o gates
            kernel_size=kernel_size,
            padding=padding
        )
        
        self.bn = nn.BatchNorm2d(4 * hidden_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for single time step.
        
        Args:
            x: Input tensor (B, C, H, W)
            hidden_state: Tuple of (h, c) hidden states
            
        Returns:
            h: New hidden state
            (h, c): New hidden state tuple
        """
        h, c = hidden_state
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)
        
        # Compute all gates
        gates = self.bn(self.conv(combined))
        
        # Split into individual gates
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)
        
        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Cell input
        o = torch.sigmoid(o)  # Output gate
        
        # Update cell state
        c_new = f * c + i * g
        
        # Compute hidden state
        h_new = o * torch.tanh(c_new)
        
        return h_new, (h_new, c_new)
    
    def init_hidden(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state with zeros."""
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        return h, c


class ConvLSTM(nn.Module):
    """
    Multi-layer Convolutional LSTM.
    
    Processes sequences of spatial data with multiple LSTM layers.
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        kernel_size: int = 3,
        bidirectional: bool = False
    ):
        """
        Initialize ConvLSTM.
        
        Args:
            input_channels: Number of input channels
            hidden_channels: List of hidden channels for each layer
            kernel_size: Convolution kernel size
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.num_layers = len(hidden_channels)
        self.hidden_channels = hidden_channels
        self.bidirectional = bidirectional
        
        # Create LSTM cells for each layer
        self.cells = nn.ModuleList()
        for i, h_ch in enumerate(hidden_channels):
            in_ch = input_channels if i == 0 else hidden_channels[i - 1]
            self.cells.append(ConvLSTMCell(in_ch, h_ch, kernel_size))
        
        # For bidirectional
        if bidirectional:
            self.cells_backward = nn.ModuleList()
            for i, h_ch in enumerate(hidden_channels):
                in_ch = input_channels if i == 0 else hidden_channels[i - 1]
                self.cells_backward.append(ConvLSTMCell(in_ch, h_ch, kernel_size))
    
    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through all time steps and layers.
        
        Args:
            x: Input tensor (B, T, C, H, W)
            hidden_state: Optional initial hidden states
            
        Returns:
            output: Output tensor (B, T, hidden_channels[-1], H, W)
            hidden_states: Final hidden states for each layer
        """
        batch_size, seq_len, _, height, width = x.shape
        device = x.device
        
        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = [
                cell.init_hidden(batch_size, height, width, device)
                for cell in self.cells
            ]
        
        # Process sequence
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]  # (B, C, H, W)
            
            # Pass through layers
            for layer_idx, cell in enumerate(self.cells):
                x_t, hidden_state[layer_idx] = cell(x_t, hidden_state[layer_idx])
            
            outputs.append(x_t)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (B, T, C, H, W)
        
        return output, hidden_state


class TemporalEncoder(nn.Module):
    """
    Encoder that processes temporal sequences.
    
    Uses a shared CNN encoder followed by ConvLSTM for temporal modeling.
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet34',
        in_channels: int = 9,
        lstm_hidden: int = 64,
        lstm_layers: int = 2
    ):
        """
        Initialize temporal encoder.
        
        Args:
            encoder_name: Name of CNN encoder backbone
            in_channels: Number of input channels
            lstm_hidden: Hidden channels for LSTM
            lstm_layers: Number of LSTM layers
        """
        super().__init__()
        
        # Create encoder (using smp for feature extraction)
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights='imagenet' if in_channels == 3 else None
        )
        
        # Get encoder output channels
        encoder_channels = self.encoder.out_channels
        
        # ConvLSTM for temporal processing at each scale
        self.conv_lstm = nn.ModuleList([
            ConvLSTM(ch, [lstm_hidden], kernel_size=3)
            for ch in encoder_channels[1:]  # Skip first (input)
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, T, C, H, W)
            
        Returns:
            List of feature maps at different scales
        """
        batch_size, seq_len = x.shape[:2]
        
        # Process each time step through encoder
        all_features = []
        for t in range(seq_len):
            features = self.encoder(x[:, t])
            all_features.append(features)
        
        # Stack features by scale
        stacked_features = []
        for scale_idx in range(len(all_features[0])):
            scale_features = torch.stack(
                [f[scale_idx] for f in all_features],
                dim=1
            )  # (B, T, C, H, W)
            stacked_features.append(scale_features)
        
        # Process through ConvLSTM at each scale
        temporal_features = [stacked_features[0][:, -1]]  # First scale unchanged
        for i, lstm in enumerate(self.conv_lstm):
            out, _ = lstm(stacked_features[i + 1])
            temporal_features.append(out[:, -1])  # Use last time step
        
        return temporal_features


class LSTMUNet(nn.Module):
    """
    U-Net with LSTM for temporal land cover classification.
    
    Architecture:
        1. Shared U-Net encoder processes each time step
        2. ConvLSTM layers process temporal features
        3. U-Net decoder produces final segmentation
    
    Supports:
        - Variable-length sequences
        - Multi-scale temporal features
        - Pre-trained encoder weights
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: str = 'imagenet',
        in_channels: int = 9,
        num_classes: int = 7,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        decoder_channels: List[int] = [256, 128, 64, 32, 16]
    ):
        """
        Initialize LSTM-UNet.
        
        Args:
            encoder_name: Name of encoder backbone
            encoder_weights: Pretrained weights
            in_channels: Number of input channels
            num_classes: Number of output classes
            lstm_hidden: Hidden channels for LSTM
            lstm_layers: Number of LSTM layers
            decoder_channels: Decoder channel configuration
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Encoder (shared across time steps)
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights if in_channels == 3 else None
        )
        
        encoder_channels = self.encoder.out_channels
        
        # ConvLSTM for bottleneck features
        self.conv_lstm = ConvLSTM(
            input_channels=encoder_channels[-1],
            hidden_channels=[lstm_hidden] * lstm_layers,
            kernel_size=3
        )
        
        # Bridge from LSTM to decoder
        self.bridge = nn.Sequential(
            nn.Conv2d(lstm_hidden, encoder_channels[-1], kernel_size=1),
            nn.BatchNorm2d(encoder_channels[-1]),
            nn.ReLU(inplace=True)
        )
        
        # U-Net decoder
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=5
        )
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, T, C, H, W)
            
        Returns:
            Output logits (B, num_classes, H, W)
        """
        batch_size, seq_len = x.shape[:2]
        
        # Encode each time step
        all_encoder_features = []
        for t in range(seq_len):
            features = self.encoder(x[:, t])
            all_encoder_features.append(features)
        
        # Stack bottleneck features for LSTM
        bottleneck_features = torch.stack(
            [f[-1] for f in all_encoder_features],
            dim=1
        )  # (B, T, C, H, W)
        
        # Process through ConvLSTM
        lstm_out, _ = self.conv_lstm(bottleneck_features)
        lstm_features = lstm_out[:, -1]  # Use last time step (B, lstm_hidden, H, W)
        
        # Bridge to decoder
        bridge_features = self.bridge(lstm_features)
        
        # Use encoder features from last time step for skip connections
        last_encoder_features = all_encoder_features[-1]
        last_encoder_features = list(last_encoder_features[:-1]) + [bridge_features]
        
        # Decode
        decoder_out = self.decoder(*last_encoder_features)
        
        # Segmentation
        output = self.segmentation_head(decoder_out)
        
        return output
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class BiTemporalChangeNet(nn.Module):
    """
    Siamese network for bi-temporal change detection.
    
    Compares two time points to detect changes in land cover.
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: str = 'imagenet',
        in_channels: int = 9,
        num_classes: int = 7
    ):
        """
        Initialize change detection network.
        
        Args:
            encoder_name: Name of encoder backbone
            encoder_weights: Pretrained weights
            in_channels: Number of input channels
            num_classes: Number of change classes
        """
        super().__init__()
        
        # Shared encoder for both time points
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights if in_channels == 3 else None
        )
        
        encoder_channels = self.encoder.out_channels
        
        # Difference module at each scale
        self.diff_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch * 2, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            )
            for ch in encoder_channels
        ])
        
        # Decoder for change map
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=[256, 128, 64, 32, 16],
            n_blocks=5
        )
        
        # Output heads
        self.change_head = nn.Conv2d(16, 2, kernel_size=1)  # Binary change
        self.class_head = nn.Conv2d(16, num_classes, kernel_size=1)  # Change type
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for bi-temporal change detection.
        
        Args:
            x1: First time point (B, C, H, W)
            x2: Second time point (B, C, H, W)
            
        Returns:
            change_map: Binary change map (B, 2, H, W)
            change_type: Change type map (B, num_classes, H, W)
        """
        # Encode both time points
        features1 = self.encoder(x1)
        features2 = self.encoder(x2)
        
        # Compute difference features at each scale
        diff_features = []
        for i, (f1, f2) in enumerate(zip(features1, features2)):
            combined = torch.cat([f1, f2], dim=1)
            diff = self.diff_convs[i](combined)
            diff_features.append(diff)
        
        # Decode
        decoder_out = self.decoder(*diff_features)
        
        # Generate outputs
        change_map = self.change_head(decoder_out)
        change_type = self.class_head(decoder_out)
        
        return change_map, change_type


def create_temporal_model(config: dict) -> nn.Module:
    """
    Create temporal model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    model_config = config.get('model', {})
    lstm_config = model_config.get('lstm', {})
    
    encoder = model_config.get('encoder', 'resnet34')
    encoder_weights = model_config.get('encoder_weights', 'imagenet')
    in_channels = model_config.get('in_channels', 9)
    num_classes = model_config.get('num_classes', 7)
    
    if lstm_config.get('enabled', False):
        model = LSTMUNet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
            lstm_hidden=lstm_config.get('hidden_size', 64),
            lstm_layers=lstm_config.get('num_layers', 2)
        )
    else:
        # Fall back to regular UNet
        from .unet import UNetPretrained
        model = UNetPretrained(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes
        )
    
    return model


if __name__ == '__main__':
    # Test models
    print("Testing LSTM-UNet models...")
    
    # Test ConvLSTM
    print("\n1. Testing ConvLSTM...")
    conv_lstm = ConvLSTM(64, [32, 32], kernel_size=3)
    x = torch.randn(2, 6, 64, 32, 32)  # (B, T, C, H, W)
    out, hidden = conv_lstm(x)
    print(f"ConvLSTM output shape: {out.shape}")
    
    # Test LSTMUNet
    print("\n2. Testing LSTMUNet...")
    model = LSTMUNet(
        encoder_name='resnet18',
        in_channels=9,
        num_classes=7,
        lstm_hidden=32
    )
    x = torch.randn(2, 4, 9, 256, 256)  # (B, T, C, H, W)
    out = model(x)
    print(f"LSTMUNet output shape: {out.shape}")
    
    # Test BiTemporalChangeNet
    print("\n3. Testing BiTemporalChangeNet...")
    model = BiTemporalChangeNet(
        encoder_name='resnet18',
        in_channels=9,
        num_classes=7
    )
    x1 = torch.randn(2, 9, 256, 256)
    x2 = torch.randn(2, 9, 256, 256)
    change_map, change_type = model(x1, x2)
    print(f"Change map shape: {change_map.shape}")
    print(f"Change type shape: {change_type.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
