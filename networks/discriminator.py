from torch import nn
from networks.blocks import DownBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=32):
        super().__init__()

        self.down1 = DownBlock(in_channels + 1, features)
        self.down2 = DownBlock(features * 1, features * 2)
        self.down3 = DownBlock(features * 2, features * 4)
        self.down4 = DownBlock(features * 4, features * 8)
        self.down5 = nn.Conv2d(features * 8, 1, 3, 1, 1)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        out = self.down5(down4)

        return out
