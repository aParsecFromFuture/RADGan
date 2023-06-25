import torch
from torch import nn
from networks.blocks import UpBlock, DownBlock


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=32):
        super().__init__()

        self.start = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, 1, 1, padding_mode='reflect'),
            nn.GroupNorm(32, features),
            nn.SiLU())

        self.down1 = DownBlock(features * 1, features * 1)
        self.down2 = DownBlock(features * 1, features * 2)
        self.down3 = DownBlock(features * 2, features * 4)
        self.down4 = DownBlock(features * 4, features * 8)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 16, 3, 1, 1),
            nn.GroupNorm(32, features * 16),
            nn.SiLU())

        self.up1 = UpBlock(features * 8 * 2, features * 4)
        self.up2 = UpBlock(features * 4 * 2, features * 2)
        self.up3 = UpBlock(features * 2 * 2, features * 1)
        self.up4 = UpBlock(features * 1 * 2, features * 1)
        
        self.final = nn.Sequential(
            nn.Conv2d(features, in_channels, 3, 1, 1),
            nn.Sigmoid())

    def forward(self, x):
        d0 = self.start(x)

        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        bottleneck = self.bottleneck(d4)

        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d3], 1))
        u3 = self.up3(torch.cat([u2, d2], 1))
        u4 = self.up4(torch.cat([u3, d1], 1))

        out = self.final(u4)

        return out

