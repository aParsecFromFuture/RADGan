import os
import glob
import torch
import random
from torch import nn
from torch.nn import init
from monai.data import Dataset, DataLoader
from monai.metrics.regression import SSIMMetric
from monai.losses import MaskedLoss

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


def get_loader(data_dir, aug_enabled=False, **kwargs):
    ct_imgs = sorted(glob.glob(os.path.join(data_dir, '*', 'ct.nii.gz')))
    mr_imgs = sorted(glob.glob(os.path.join(data_dir, '*', 'mr.nii.gz')))
    masks = sorted(glob.glob(os.path.join(data_dir, '*', 'mask.nii.gz')))

    data_dicts = [{'mr': mr_img, 'ct': ct_img, 'mask': mask}
                  for mr_img, ct_img, mask in zip(mr_imgs, ct_imgs, masks)]

    transform = Compose([
        LoadImaged(keys=['ct', 'mr', 'mask']),
        EnsureChannelFirstd(keys=['ct', 'mr', 'mask']),
        Spacingd(keys=['ct', 'mr', 'mask'], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear", "nearest")),
        Orientationd(keys=['ct', 'mr', 'mask'], axcodes="PLS"),
        ScaleIntensityd(keys=['ct', 'mr'], minv=0.0, maxv=1.0),
        CropForegroundd(keys=['ct', 'mr'], source_key='mask'),
        Resized(keys=['ct', 'mr', 'mask'], spatial_size=256, size_mode='longest'),
        ResizeWithPadOrCropd(keys=['ct', 'mr', 'mask'], spatial_size=(256, 256, 256)),
        Compose([
            RandRotate90d(keys=['ct', 'mr', 'mask'], prob=0.5, spatial_axes=(0, 1)),
            RandRotate90d(keys=['ct', 'mr', 'mask'], prob=0.5, spatial_axes=(1, 2)),
            RandRotate90d(keys=['ct', 'mr', 'mask'], prob=0.5, spatial_axes=(0, 2)),
        ]) if aug_enabled else Identityd(keys=['ct', 'mr', 'mask']),
        NormalizeIntensityd(keys=['mr'], subtrahend=0.0775, divisor=0.1208)
    ])

    dataset = Dataset(data=data_dicts, transform=transform)
    data_loader = DataLoader(dataset, **kwargs)

    return data_loader


def set_requires_grad(net, requires_grad=False):
    for param in net.parameters():
        param.requires_grad = requires_grad


def volume3d_train(model, batch, opt_G, opt_D, fraction=0.1, grad_clip=0.5, l1_lambda=100):
    bce_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()
    G_total_loss, D_total_loss = 0, 0
    model.train()

    slice_indices = random.sample(list(range(30, 220)), int(fraction * 256))
    random.shuffle(slice_indices)

    for i in slice_indices:
        x = batch['mr'][:, :, i, :, :].cuda()
        y = batch['ct'][:, :, i, :, :].cuda()

        set_requires_grad(model.net_D, True)
        y_fake = model.net_G(x)
        D_real = model.net_D(torch.cat([x, y], dim=1))
        D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
        D_fake = model.net_D(torch.cat([x, y_fake.detach()], dim=1))
        D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
        D_loss = 0.5 * (D_real_loss + D_fake_loss)

        opt_D.zero_grad()
        D_loss.backward()
        nn.utils.clip_grad_norm_(model.net_D.parameters(), grad_clip)
        opt_D.step()

        set_requires_grad(model.net_D, False)
        D_fake = model.net_D(torch.cat([x, y_fake], dim=1))
        G_fake_loss = bce_loss(D_fake, torch.ones_like(D_fake))
        L1_loss = l1_lambda * l1_loss(y_fake, y)
        G_loss = G_fake_loss + L1_loss

        opt_G.zero_grad()
        G_loss.backward()
        nn.utils.clip_grad_norm_(model.net_G.parameters(), grad_clip)
        opt_G.step()

        G_total_loss += G_loss.item() / len(slice_indices)
        D_total_loss += D_loss.item() / len(slice_indices)

    return G_total_loss, D_total_loss


@torch.no_grad()
def volume3d_infer(model, batch):
    data_range = torch.tensor([1], device='cuda')
    ssim_fn = MaskedLoss(loss=SSIMMetric(spatial_dims=2, data_range=data_range))

    model.eval()
    total_score = 0
    slice_indices = list(range(256))

    for i in slice_indices:
        x = batch['mr'][:, :, i, :, :].cuda()
        y = batch['ct'][:, :, i, :, :].cuda()
        mask = batch['mask'][:, :, i, :, :].cuda()

        y_fake = model.net_G(x)
        total_score += ssim_fn(y, y_fake, mask=mask).item() / len(slice_indices)

    return total_score

