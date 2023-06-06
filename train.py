import os
import argparse
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from networks.model import GAN
from utils import init_weights
from utils import get_loader2d, train_fn, eval_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', '--train_dir', type=str, default='dataset2d/train', help='Training Data Directory')
    parser.add_argument('-val', '--val_dir', type=str, default='dataset2d/val', help='Validation Data Directory')
    parser.add_argument('-checkpoint', '--checkpoint', type=str, default=None, help='Checkpoint Path')
    parser.add_argument('-lr', '--learning_rate', type=int, default=1e-4, help='Learning Rate')
    parser.add_argument('-alpha', '--alpha', type=int, default=50, help='Alpha')
    parser.add_argument('-beta', '--beta', type=int, default=20, help='Beta')
    parser.add_argument('-b', '--batch', type=int, default=2, help='Batch Size')
    parser.add_argument('-n', '--num_epochs', type=int, default=100, help='Number of Epochs')
    parser.add_argument('-d', '--device', type=str, default='cuda')

    torch.manual_seed(42)
    args = parser.parse_args()

    train_loader = get_loader2d(args.train_dir,
                                shuffle=True,
                                batch_size=args.batch,
                                aug_enabled=True,
                                drop_last=True)

    val_loader = get_loader2d(args.val_dir,
                              shuffle=True,
                              batch_size=4,
                              drop_last=True)

    model = GAN(in_channels=1, features=32).to(args.device)

    if args.checkpoint is not None:
        model = torch.load(args.checkpoint)
    else:
        init_weights(model, init_type='normal')

    opt_G = optim.Adam(model.net_G.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    opt_D = optim.Adam(model.net_D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    scheduler_G = ReduceLROnPlateau(opt_G, patience=10, factor=0.5)
    scheduler_D = ReduceLROnPlateau(opt_D, patience=10, factor=0.5)

    writer = SummaryWriter()
    best_score = 0
    step = 0

    for epoch in range(args.num_epochs):
        G_loss, D_loss, P_loss = 0, 0, 0
        for batch_idx, batch in enumerate(train_loader, 1):
            l1, l2, l3 = train_fn(model,
                                  x=batch['mr'].to(args.device),
                                  y=batch['ct'].to(args.device),
                                  alpha=args.alpha, beta=args.beta,
                                  opt_G=opt_G, opt_D=opt_D)
            G_loss += l1
            D_loss += l2
            P_loss += l3

            if batch_idx % 1000 == 0:
                SSIM_score, PSNR_score, MAE_loss = 0, 0, 0
                for val_batch in val_loader:
                    s1, s2, s3 = eval_fn(model,
                                         x=val_batch['mr'].to(args.device),
                                         y=val_batch['ct'].to(args.device))
                    SSIM_score += s1
                    PSNR_score += s2
                    MAE_loss += s3

                scheduler_G.step(SSIM_score)
                scheduler_D.step(SSIM_score)

                print(f'Epoch [{epoch}/{args.num_epochs}]'
                      f' Batch: [{batch_idx}/{len(train_loader)}]'
                      f' G_loss: {G_loss/batch_idx:.4f}'
                      f' D_loss: {D_loss/batch_idx:.4f}'
                      f' P_loss: {P_loss/batch_idx:.4f}'
                      f' SSIM: {SSIM_score/len(val_loader):.4f},'
                      f' PSNR: {PSNR_score/len(val_loader):.4f}'
                      f' MAE: {MAE_loss/len(val_loader):.4f}')

                writer.add_scalar('G_loss', G_loss/batch_idx, step),
                writer.add_scalar('D_loss', D_loss/batch_idx, step)
                writer.add_scalar('P_loss', P_loss / batch_idx, step)
                writer.add_scalar('SSIM_score', SSIM_score/len(val_loader), step)
                writer.add_scalar('PSNR_score', PSNR_score/len(val_loader), step)
                writer.add_scalar('MAE_loss', MAE_loss / len(val_loader), step)
                step = step + 1

                x_sample = val_batch['mr'].cuda()
                y_sample = val_batch['ct'].cuda()
                y_fake = model(x_sample)

                imgs = []
                imgs.extend([x_sample[i] for i in range(4)])
                imgs.extend([y_sample[i] for i in range(4)])
                imgs.extend([y_fake[i] for i in range(4)])

                grid_img = make_grid(imgs, nrow=4)
                save_image(grid_img, os.path.join(writer.log_dir, f'sample_{epoch:03d}_{batch_idx:04d}.png'))

                if best_score < SSIM_score:
                    print('Congratulations! New checkpoint saved')
                    torch.save(model, os.path.join(writer.log_dir, 'best_model.pt'))
                    best_score = SSIM_score

    print('Training done!')
    writer.close()
