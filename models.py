"""
models.py — U-Net and Half U-Net Architectures
================================================
Implements a standard U-Net and a reduced-complexity Half U-Net in PyTorch.

Standard U-Net encoder channel progression : [64, 128, 256, 512, 1024]
Half U-Net encoder channel progression     : [32,  64, 128, 256,  512]

Half U-Net halves every channel dimension, yielding ~75% fewer parameters
and >20% reduction in peak GPU VRAM allocation while preserving segmentation
fidelity — mirroring the findings of André et al. (MAGMA, 2025).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool downsampling followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Bilinear upsampling + skip connection + DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd spatial dimensions
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1×1 convolution to produce the segmentation mask logits."""

    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ─────────────────────────────────────────────
# U-Net Architectures
# ─────────────────────────────────────────────

class UNet(nn.Module):
    """
    Standard U-Net (Ronneberger et al., 2015).
    Encoder channels: [64, 128, 256, 512, 1024]
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 1):
        super().__init__()
        self.name = "UNet"
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64,  128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1   = Up(1024 + 512, 512)
        self.up2   = Up(512  + 256, 256)
        self.up3   = Up(256  + 128, 128)
        self.up4   = Up(128  +  64,  64)
        self.outc  = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)


class HalfUNet(nn.Module):
    """
    Half U-Net: channel capacity halved at every encoder stage.
    Encoder channels: [32, 64, 128, 256, 512]

    Achieves ~75% parameter reduction and >20% VRAM savings
    vs. standard U-Net (André et al., MAGMA 2025).
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 1):
        super().__init__()
        self.name = "HalfUNet"
        self.inc   = DoubleConv(n_channels, 32)
        self.down1 = Down(32,  64)
        self.down2 = Down(64,  128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.up1   = Up(512 + 256, 256)
        self.up2   = Up(256 + 128, 128)
        self.up3   = Up(128 +  64,  64)
        self.up4   = Up( 64 +  32,  32)
        self.outc  = OutConv(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)
