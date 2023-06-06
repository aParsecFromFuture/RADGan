import os
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch import nn
from torch.nn import init
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_msssim import ssim
from torch.nn.functional import mse_loss, l1_loss
import monai
from monai.data import NibabelReader

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    Identityd,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    Spacingd,
    RandRotate90d,
    Resized,
    ResizeWithPadOrCropd,
    NormalizeIntensityd,
)


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
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def set_requires_grad(net, requires_grad=False):
    for param in net.parameters():
        param.requires_grad = requires_grad


def get_loader(data_dir, **kwargs):
    ct_imgs = sorted(glob.glob(os.path.join(data_dir, '*', 'ct.nii.gz')))
    mr_imgs = sorted(glob.glob(os.path.join(data_dir, '*', 'mr.nii.gz')))

    data_dicts = [{'mr': mr_img, 'ct': ct_img}
                  for mr_img, ct_img in zip(mr_imgs, ct_imgs)]

    transform = Compose([
        LoadImaged(keys=['ct', 'mr']),
        EnsureChannelFirstd(keys=['ct', 'mr']),
        Spacingd(keys=['ct', 'mr'], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear")),
        Orientationd(keys=['ct', 'mr'], axcodes="SPL"),
        Resized(keys=['ct', 'mr'], spatial_size=256, size_mode='longest'),
        ScaleIntensityd(keys=['mr', 'ct'], minv=0.0, maxv=1.0),
        ResizeWithPadOrCropd(keys=['ct', 'mr'], spatial_size=(256, 256, 256)),
    ])

    dataset = monai.data.Dataset(data=data_dicts, transform=transform)
    data_loader = monai.data.DataLoader(dataset, **kwargs)

    return data_loader


def get_loader2d(data_dir, aug_enabled=False, **kwargs):
    class CustomDataset(Dataset):
        def __init__(self, root, transform):
            super().__init__()
            self.tfs = transform
            self.ct_paths = sorted(glob.glob(os.path.join(root, 'ct_*.png')))
            self.mr_paths = sorted(glob.glob(os.path.join(root, 'mr_*.png')))

            self.ct_imgs = [Image.open(ct_path).convert('L') for ct_path in self.ct_paths]
            self.mr_imgs = [Image.open(mr_path).convert('L') for mr_path in self.mr_paths]

        def __getitem__(self, i):
            ct_img = np.array(self.ct_imgs[i])
            mr_img = np.array(self.mr_imgs[i])

            aug = self.tfs(image=ct_img, image_2=mr_img)

            return {'ct': aug['image'], 'mr': aug['image_2']}

        def __len__(self):
            return len(self.ct_paths)

    if aug_enabled:
        tfs = A.Compose([
            A.Flip(),
            A.RandomRotate90(),
            A.Normalize(mean=[0.0], std=[1.0]),
            ToTensorV2()
        ], additional_targets={'image_2': 'image'})
    else:
        tfs = A.Compose([
            A.Normalize(mean=[0.0], std=[1.0]),
            ToTensorV2()
        ], additional_targets={'image_2': 'image'})

    dataset = CustomDataset(data_dir, tfs)
    loader = DataLoader(dataset, **kwargs)
    return loader


def prepare_2d_dataset(src_dir, dst_dir, fraction=0.1):
    os.makedirs(dst_dir)
    os.makedirs(os.path.join(dst_dir, 'train'))
    os.makedirs(os.path.join(dst_dir, 'val'))
    os.makedirs(os.path.join(dst_dir, 'test'))

    for phase in ['train', 'val', 'test']:
        loader = get_loader(os.path.join(src_dir, phase), batch_size=1, shuffle=False)
        for i, data in tqdm(enumerate(loader)):
            slice_indices = list(range(256))
            for j in slice_indices:
                if data['ct'][0, 0, j].max() > 0.1:
                    save_image(data['ct'][0, 0, j], os.path.join(dst_dir, phase, f'ct_{i}_{j}.png'))
                    save_image(data['mr'][0, 0, j], os.path.join(dst_dir, phase, f'mr_{i}_{j}.png'))


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
    clip_grad_norm_(model.net_D.parameters(), 0.5)
    opt_G.step()

    return loss_G_GAN.item(), loss_D.item(), loss_G_P.item()


@torch.no_grad()
def eval_fn(model, x, y):
    model.eval()

    y_fake = model.net_G(x)

    ssim_score, psnr_score, mae_loss = 0, 0, 0

    y = y.detach().cpu().numpy()
    y_fake = y_fake.detach().cpu().numpy()

    for i in range(4):
        img_1, img_2 = y[i, 0], y_fake[i, 0]

        ssim_score += structural_similarity(img_1, img_2, data_range=1.0)
        psnr_score += peak_signal_noise_ratio(img_1, img_2, data_range=1.0)
        mae_loss += np.abs(img_1 - img_2).mean()

    return ssim_score / 4, psnr_score / 4, mae_loss / 4

