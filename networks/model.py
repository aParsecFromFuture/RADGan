import torch
from torch import nn
from networks.discriminator import Discriminator
from networks.generator import Generator


class GAN(nn.Module):
    def __init__(self, in_channels, features=64):
        super().__init__()
        self.net_G = Generator(in_channels, features)
        self.net_D = Discriminator(in_channels, features)

    @torch.no_grad()
    def forward(self, x):
        return self.net_G(x)

