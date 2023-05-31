from torch import nn
from networks.blocks import DownBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()

        self.network = nn.Sequential(
            DownBlock(in_channels * 2, features),
            DownBlock(features * 1, features * 2),
            DownBlock(features * 2, features * 4),
            DownBlock(features * 4, features * 8),
            nn.Conv2d(features * 8, 1, 4, 2, 1),
            nn.Sigmoid())

    def forward(self, x):
        return self.network(x)
