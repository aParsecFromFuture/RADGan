import os
import argparse

from tqdm import tqdm
import torch
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data import MedDataset
from networks.model import GAN
from utils import init_weights
from monai.metrics.regression import SSIMMetric
from monai.losses import MaskedLoss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', '--train_dir', type=str, default='dataset/train', help='Training Data Directory')
    parser.add_argument('-val', '--val_dir', type=str, default='dataset/val', help='Validation Data Directory')
    parser.add_argument('-lr', '--learning_rate', type=int, default=1e-4, help='Learning Rate')
    parser.add_argument('-l1_lambda', '--l1_lambda', type=int, default=100, help='Lambda L1 penalty')
    parser.add_argument('-clip', '--grad_clip', type=int, default=0.5, help='Gradient Norm Clipping Value')
    parser.add_argument('-b', '--batch', type=int, default=1, help='Batch Size')
    parser.add_argument('-n', '--num_epochs', type=int, default=1000, help='Number of Epochs')
    parser.add_argument('-d', '--device', type=str, default='cuda')

    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    writer = SummaryWriter('runs/train')

    train_ds = MedDataset(root=args.train_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)

    val_ds = MedDataset(root=args.val_dir)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, drop_last=False)

    model = GAN(in_channels=1,
                lr=args.learning_rate,
                l1_lambda=args.l1_lambda,
                grad_clip=args.grad_clip).to(args.device)

    init_weights(model, init_type='normal')

    best_val_score = -1
    data_range = torch.tensor([1], device=args.device)
    ssim = MaskedLoss(loss=SSIMMetric(spatial_dims=2, data_range=data_range))

    for epoch in range(args.num_epochs):
        model.train()
        train_G, train_D = 0, 0
        for x, y, _ in tqdm(train_loader):
            x, y = x.to(args.device), y.to(args.device)
            D_loss, G_loss = model(x, y)
            train_G += G_loss.item()
            train_D += D_loss.item()

        train_G = train_G / len(train_loader)
        train_D = train_D / len(train_loader)

        print(f'Epoch [{epoch}/{args.num_epochs}] Loss G: {train_G:.4f}, Loss D: {train_D:.4f}')
        writer.add_scalar('train/G_loss', train_G, epoch)
        writer.add_scalar('train/D_loss', train_D, epoch)

        model.eval()
        val_score = 0
        for x, y, mask in tqdm(val_loader):
            x, y, mask = x.to(args.device), y.to(args.device), mask.to(args.device)
            y_fake = model.generate(x)
            val_score += ssim(y, y_fake, mask=mask).item()

        val_score = val_score / len(val_loader)

        if val_score > best_val_score:
            print(f'Congratulations! Best new val SSIM: {val_score:.4f}')
            best_val_score = val_score
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch [{epoch}/{args.num_epochs}] SSIM: {val_score:.4f}')
        writer.add_scalar('val/SSIM', val_score, epoch)
        writer.add_image('val/fake_img', y_fake[0], epoch)

        save_image(y[0], os.path.join('runs/train', f'target_{epoch}.png'))
        save_image(y_fake[0], os.path.join('runs/train', f'recon_{epoch}.png'))

    print('Training done!')
    writer.close()

