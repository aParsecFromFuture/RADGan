import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()

        self.down1 = self._down_block(in_channels, features * 1)
        self.down2 = self._down_block(features * 1, features * 2)
        self.down3 = self._down_block(features * 2, features * 4)
        self.down4 = self._down_block(features * 4, features * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 3, 1, 1),
            nn.SiLU())

        self.up1 = self._up_block(features * 8, features * 4)
        self.up2 = self._up_block(features * 4, features * 2)
        self.up3 = self._up_block(features * 2, features * 1)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(features, in_channels, 3, 1, 1),
            nn.Sigmoid())

    def _down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, padding_mode='reflect'),
            nn.GroupNorm(32, out_channels),
            nn.SiLU()).to('cuda')

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU()).to('cuda')

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        bottleneck = self.bottleneck(d4)

        up1 = self.up1(bottleneck)
        up2 = self.up2(up1 + d3)
        up3 = self.up3(up2 + d2)
        out = self.final_up(up3 + d1)

        return out
