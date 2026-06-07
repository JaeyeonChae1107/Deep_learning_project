"""
CBAM: Convolutional Block Attention Module
Woo et al., ECCV 2018 — https://arxiv.org/abs/1807.06521

Channel Attention: squeeze via both AvgPool & MaxPool → shared MLP → element-wise add → sigmoid
Spatial Attention: channel-wise AvgPool & MaxPool concat → 7×7 conv → sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        mid = max(in_channels // reduction_ratio, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = x.mean(dim=[2, 3])                          # (B, C)
        mx = x.amax(dim=[2, 3])                           # (B, C)
        scale = torch.sigmoid(
            self.shared_mlp(avg) + self.shared_mlp(mx)   # (B, C)
        )
        return x * scale.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        mx, _ = x.max(dim=1, keepdim=True)  # (B, 1, H, W)
        cat = torch.cat([avg, mx], dim=1)   # (B, 2, H, W)
        scale = torch.sigmoid(self.conv(cat))
        return x * scale


class CBAM(nn.Module):
    """Sequential application: Channel Attention → Spatial Attention."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
