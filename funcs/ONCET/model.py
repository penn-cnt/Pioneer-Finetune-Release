"""
model.py

Lightweight Seizure Detector - Option 1 (Depthwise Separable TCN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable convolution for efficient temporal processing.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        causal: bool = True
    ):
        super().__init__()
        
        # Causal padding (don't look into future)
        if causal:
            self.padding = (kernel_size - 1) * dilation
            self.causal = True
        else:
            self.padding = ((kernel_size - 1) * dilation) // 2
            self.causal = False
        
        # Depthwise: each feature processed independently
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
            groups=in_channels,
            bias=False
        )
        
        # Pointwise: mix features
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        # Depthwise
        x = self.depthwise(x)
        if self.causal:
            x = x[..., :-self.padding]  # Remove future-looking padding
        x = self.bn1(x)
        x = F.gelu(x)
        
        # Pointwise
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.gelu(x)
        
        return x


class SqueezeExcitation1d(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _ = x.shape
        # Global context
        y = self.pool(x).view(b, c)
        # Learn channel importance
        y = self.fc(y).view(b, c, 1)
        # Reweight
        return x * y


class TemporalResidualBlock(nn.Module):
    """
    Residual block with depthwise separable convolution.
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.conv1 = DepthwiseSeparableConv1d(
            channels, channels,
            kernel_size, dilation
        )
        self.conv2 = DepthwiseSeparableConv1d(
            channels, channels,
            kernel_size, dilation
        )
        self.se = SqueezeExcitation1d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.dropout(x)
        return x + residual


class LightweightSeizureDetector(nn.Module):
    """
    Lightweight seizure detector using depthwise separable TCN.
    
    Input: [batch, 1, 256] - single electrode, 1 second at 256Hz
    Output: [batch, 2] - logits for [not_seizing, seizing]
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        base_filters: int = 32,
        num_blocks: int = 3,
        kernel_size: int = 16,
        dilations: list = None,
        dropout_stem: float = 0.1,
        dropout_blocks: float = 0.2,
        dropout_head: float = 0.3
    ):
        super().__init__()
        
        if dilations is None:
            dilations = [2, 4, 8]
        
        assert len(dilations) == num_blocks, "Number of dilations must match num_blocks"
        
        self.num_classes = num_classes
        self.base_filters = base_filters
        
        # Stem: Convert raw EEG to initial features
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_filters, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.GELU(),
            nn.Dropout(dropout_stem)
        )
        
        # Temporal processing blocks
        self.blocks = nn.ModuleList([
            TemporalResidualBlock(
                base_filters,
                kernel_size,
                dilation,
                dropout_blocks
            )
            for dilation in dilations
        ])
        
        # Feature compression
        self.compress = DepthwiseSeparableConv1d(
            base_filters,
            base_filters // 2,
            kernel_size=1,
            causal=False
        )
        
        # Global pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout_head),
            nn.Linear(base_filters, 32),
            nn.GELU(),
            nn.Dropout(dropout_head),
            nn.Linear(32, num_classes)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.0) 
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    

    def forward(self, x):
        """
        Args:
            x: [batch, 1, 256] - single electrode signal
            
        Returns:
            logits: [batch, num_classes]
        """
        # Stem
        x = self.stem(x)  # [batch, base_filters, 128]
        
        # Temporal blocks
        for block in self.blocks:
            x = block(x)  # [batch, base_filters, 128]
        
        # Compress
        x = self.compress(x)  # [batch, base_filters//2, 128]
        
        # Global pooling
        gap = self.gap(x).squeeze(-1)  # [batch, base_filters//2]
        gmp = self.gmp(x).squeeze(-1)  # [batch, base_filters//2]
        x = torch.cat([gap, gmp], dim=1)  # [batch, base_filters]
        
        # Classification
        x = self.head(x)  # [batch, num_classes]
        
        return x
    
    def get_seizure_probability(self, x):
        """
        Convenience method for inference.
        
        Returns:
            probs: [batch] - probability of seizure class
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        return probs[:, 1]
    
    def get_seizure_probability_temp(self, x):
        """
        Convenience method for inference.
        
        Returns:
            probs: [batch] - probability of seizure class
        """
        logits = self.forward(x)
        probs = F.softmax(logits/self.temperature, dim=1)
        return probs[:, 1]


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = LightweightSeizureDetector(
        num_classes=2,
        base_filters=32,
        num_blocks=3,
        dilations=[2, 4, 8]
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(8, 1, 256)
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sample probabilities: {probs[0]}")