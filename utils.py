import os
import glob
import torch
import numpy as np
from PIL import Image

from torch.nn import init
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import mse_loss, l1_loss

from skimage.metrics import structural_similarity, peak_signal_noise_ratio

import albumentations as A
from albumentations.pytorch import ToTensorV2

from pytorch_msssim import ssim


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    print(f'Network initialized with {init_type}')
    net.apply(init_func)


def set_requires_grad(net, requires_grad=False):
    for param in net.parameters():
        param.requires_grad = requires_grad


def get_loader(data_dir, **kwargs):
    class CustomDataset(Dataset):
        def __init__(self, root):
            super().__init__()
            self.tfs = A.Compose([
                A.Normalize(mean=[0.0], std=[1.0]),
                ToTensorV2()
            ], additional_targets={'image_2': 'image'})

            self.x_paths = sorted(glob.glob(os.path.join(root, 'x_*.png')))
            self.y_paths = sorted(glob.glob(os.path.join(root, 'y_*.png')))

            self.x_imgs = [Image.open(x_path).convert('L') for x_path in self.x_paths]
            self.y_imgs = [Image.open(y_path).convert('L') for y_path in self.y_paths]

        def __getitem__(self, i):
            x = np.array(self.x_imgs[i])
            y = np.array(self.y_imgs[i])

            aug = self.tfs(image=y, image_2=x)

            return {'x': aug['image_2'], 'y': aug['image']}

        def __len__(self):
            return len(self.y_paths)

    dataset = CustomDataset(data_dir)
    loader = DataLoader(dataset, **kwargs)
    return loader


def train_fn(model, x, y, alpha, beta, opt_G, opt_D):
    model.train()

    fake_y = model.net_G(x)
    set_requires_grad(model.net_D, True)
    
    fake_xy = torch.cat((x, fake_y), 1)
    pred_fake = model.net_D(fake_xy.detach())
    loss_D_fake = mse_loss(pred_fake, torch.zeros_like(pred_fake))

    real_xy = torch.cat((x, y), 1)
    pred_real = model.net_D(real_xy)
    loss_D_real = mse_loss(pred_real, torch.ones_like(pred_real))

    loss_D = (loss_D_fake + loss_D_real) * 0.5

    opt_D.zero_grad()
    loss_D.backward()
    clip_grad_norm_(model.net_D.parameters(), 0.5)
    opt_D.step()

    set_requires_grad(model.net_D, False)

    fake_xy = torch.cat((x, fake_y), 1)
    pred_fake = model.net_D(fake_xy)
    loss_G_GAN = mse_loss(pred_fake, torch.ones_like(pred_fake))

    loss_G_P = alpha * (1 - ssim(fake_y, y, data_range=1)) + beta * l1_loss(fake_y, y)
    loss_G = loss_G_GAN + loss_G_P

    opt_G.zero_grad()
    loss_G.backward()
    clip_grad_norm_(model.net_G.parameters(), 0.5)
    opt_G.step()

    return loss_G_GAN.item(), loss_D.item(), loss_G_P.item()


@torch.no_grad()
def eval_fn(model, x, y):
    model.eval()

    y_fake = model.net_G(x)

    y = y.detach().cpu().numpy()
    y_fake = y_fake.detach().cpu().numpy()

    ssim_score, psnr_score, mae_loss = 0, 0, 0
    n = x.shape[0]

    for i in range(n):
        gt, pred = y[i, 0], y_fake[i, 0]
        data_range = gt.max() - gt.min()

        ssim_score += structural_similarity(gt, pred, data_range=data_range)
        psnr_score += peak_signal_noise_ratio(gt, pred, data_range=data_range)
        mae_loss += np.abs(gt - pred).mean()

    return ssim_score / n, psnr_score / n, mae_loss / n

