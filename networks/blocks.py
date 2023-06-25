import torch
from torch import nn


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU())

        self.network = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode='reflect'),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode='reflect'),
            nn.GroupNorm(32, out_channels),
            nn.SiLU())

    def forward(self, x):
        x = self.up(x)
        return self.network(x) + x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, padding_mode='reflect'),
            nn.GroupNorm(32, out_channels),
            nn.SiLU())

        self.network = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode='reflect'),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode='reflect'),
            nn.GroupNorm(32, out_channels),
            nn.SiLU())

    def forward(self, x):
        x = self.down(x)
        return self.network(x) + x
