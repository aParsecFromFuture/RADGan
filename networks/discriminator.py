from torch import nn
from networks.blocks import DownBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=32):
        super().__init__()

        self.start = nn.Sequential(
            nn.Conv2d(in_channels + 1, features, 3, 1, 1, padding_mode='reflect'),
            nn.GroupNorm(32, features),
            nn.SiLU())

        self.down1 = DownBlock(features * 1, features * 1)
        self.down2 = DownBlock(features * 1, features * 2)
        self.down3 = DownBlock(features * 2, features * 4)
        self.down4 = DownBlock(features * 4, features * 8)

        self.final = nn.Conv2d(features * 8, 1, 3, 1, 1)

    def forward(self, x):
        d0 = self.start(x)

        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        out = self.final(d4)

        return out

