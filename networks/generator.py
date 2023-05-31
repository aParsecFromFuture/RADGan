from torch import nn
from networks.blocks import UpBlock, DownBlock


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()

        self.down1 = DownBlock(in_channels, features * 1)
        self.down2 = DownBlock(features * 1, features * 2)
        self.down3 = DownBlock(features * 2, features * 4)
        self.down4 = DownBlock(features * 4, features * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 3, 1, 1),
            nn.SiLU(),
            nn.Dropout2d(0.5))

        self.up1 = UpBlock(features * 8, features * 4)
        self.up2 = UpBlock(features * 4, features * 2)
        self.up3 = UpBlock(features * 2, features * 1)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(features, in_channels, 3, 1, 1),
            nn.Sigmoid())

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
