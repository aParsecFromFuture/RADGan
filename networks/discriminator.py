import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], 4, 2, 1),
            nn.SiLU(),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(self._block(in_channels, feature))
            in_channels = feature

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, 1, 4, 2, 1),
                nn.Sigmoid(),
            )
        )

        self.model = nn.Sequential(*layers)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, padding_mode='reflect'),
            nn.GroupNorm(32, out_channels),
            nn.SiLU()).to('cuda')

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x
