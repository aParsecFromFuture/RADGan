import torch
from torch import nn
from networks.blocks import UpBlock, DownBlock


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=32):
        super().__init__()

        self.down1 = DownBlock(in_channels, features * 1)
        self.down2 = DownBlock(features * 1, features * 2)
        self.down3 = DownBlock(features * 2, features * 4)
        self.down4 = DownBlock(features * 4, features * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 3, 1, 1),
            nn.GroupNorm(32, features * 8),
            nn.SiLU())

        self.up1 = UpBlock(features * 4 * 2, features * 4)
        self.up2 = UpBlock(features * 4 * 2, features * 2)
        self.up3 = UpBlock(features * 2 * 2, features * 1)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(features * 1 * 2, in_channels, 3, 1, 1),
            nn.Sigmoid())

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        bottleneck = self.bottleneck(d4)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d3], 1))
        up3 = self.up3(torch.cat([up2, d2], 1))
        out = self.up4(torch.cat([up3, d1], 1))

        return out
