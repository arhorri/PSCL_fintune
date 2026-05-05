"""
Minimal U-Net for single-channel SEM patch segmentation.

Architecture: 4 encoder stages + bottleneck + 4 decoder stages.
Input:  [B, 1,   H, W]  (grayscale float32)
Output: [B, C,   H, W]  (class logits, C = n_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    """Two Conv-BN-ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """4-level U-Net.

    Parameters
    ----------
    in_channels:
        Input channels (1 for grayscale SEM images).
    n_classes:
        Number of output segmentation classes.
    base_features:
        Feature width at the first encoder stage.  Doubles at each depth.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 5,
        base_features: int = 64,
    ) -> None:
        super().__init__()
        f = base_features

        # Encoder
        self.enc1 = _Block(in_channels, f)
        self.enc2 = _Block(f,     f * 2)
        self.enc3 = _Block(f * 2, f * 4)
        self.enc4 = _Block(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _Block(f * 8, f * 16)

        # Decoder
        self.up4   = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4  = _Block(f * 16, f * 8)
        self.up3   = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3  = _Block(f * 8, f * 4)
        self.up2   = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2  = _Block(f * 4, f * 2)
        self.up1   = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1  = _Block(f * 2, f)

        self.head = nn.Conv2d(f, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder (with skip connections)
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)
