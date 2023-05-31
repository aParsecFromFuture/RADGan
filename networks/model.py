import torch
from torch import nn
from torch import optim
from networks.discriminator import Discriminator
from networks.generator import Generator


class GAN(nn.Module):
    def __init__(self, in_channels, lr=1e-4, l1_lambda=100, grad_clip=0.5):
        super().__init__()
        self.net_G = Generator(in_channels)
        self.net_D = Discriminator(in_channels)

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr, betas=(0.5, 0.999))

        self.g_scaler = torch.cuda.amp.GradScaler()
        self.d_scaler = torch.cuda.amp.GradScaler()

        self.l1_lambda = l1_lambda
        self.grad_clip = grad_clip

    def forward(self, x, y):
        with torch.cuda.amp.autocast():
            y_fake = self.net_G(x)
            D_real = self.net_D(x, y)
            D_real_loss = self.bce_loss(D_real, torch.ones_like(D_real))
            D_fake = self.net_D(x, y_fake.detach())
            D_fake_loss = self.bce_loss(D_fake, torch.zeros_like(D_fake))
            D_loss = 0.5 * (D_real_loss + D_fake_loss)

        self.opt_D.zero_grad()
        self.d_scaler.scale(D_loss).backward()
        nn.utils.clip_grad_norm_(self.net_D.parameters(), self.grad_clip)
        self.d_scaler.step(self.opt_D)
        self.d_scaler.update()

        with torch.cuda.amp.autocast():
            D_fake = self.net_D(x, y_fake)
            G_fake_loss = self.bce_loss(D_fake, torch.ones_like(D_fake))
            L1_loss = self.l1_lambda * self.l1_loss(y_fake, y)
            G_loss = G_fake_loss + L1_loss

        self.opt_G.zero_grad()
        self.g_scaler.scale(G_loss).backward()
        nn.utils.clip_grad_norm_(self.net_G.parameters(), self.grad_clip)
        self.g_scaler.step(self.opt_G)
        self.g_scaler.update()

        return D_loss, G_loss

    @torch.no_grad()
    def evaluate(self, x, y):
        with torch.cuda.amp.autocast():
            y_fake = self.net_G(x)
            D_real = self.net_D(x, y)
            D_real_loss = self.bce_loss(D_real, torch.ones_like(D_real))
            D_fake = self.net_D(x, y_fake.detach())
            D_fake_loss = self.bce_loss(D_fake, torch.zeros_like(D_fake))
            D_loss = 0.5 * (D_real_loss + D_fake_loss)

            D_fake = self.net_D(x, y_fake)
            G_fake_loss = self.bce_loss(D_fake, torch.ones_like(D_fake))
            L1_loss = self.l1_loss(y_fake, y)
            G_loss = G_fake_loss + L1_loss

        return y_fake, D_loss, G_loss

    @torch.no_grad()
    def generate(self, x):
        return self.net_G(x)

