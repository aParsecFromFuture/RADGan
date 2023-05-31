import argparse
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from networks.model import GAN
from utils import init_weights
from utils import get_loader, volume3d_train, volume3d_infer


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
    writer = SummaryWriter()

    train_loader = get_loader(args.train_dir, aug_enabled=True, shuffle=True, batch_size=args.batch)
    val_loader = get_loader(args.val_dir, shuffle=True, batch_size=1)

    model = GAN(in_channels=1).to(args.device)
    init_weights(model, init_type='normal')

    opt_G = optim.Adam(model.net_G.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    opt_D = optim.Adam(model.net_D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    best_score = -1

    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            G_loss, D_loss = volume3d_train(model, batch, opt_G, opt_D, fraction=0.1)
            SSIM_score = volume3d_infer(model, next(iter(val_loader)))

            print(f'Epoch [{epoch}/{args.num_epochs}] Batch: [{batch_idx}/{len(train_loader)}] Loss G: {G_loss:.4f}, Loss D: {D_loss:.4f}, SSIM: {SSIM_score:.4f}')
            writer.add_scalar('G_loss', G_loss, batch_idx)
            writer.add_scalar('D_loss', D_loss, batch_idx)
            writer.add_scalar('SSIM', SSIM_score, batch_idx)

            if SSIM_score > best_score:
                print(f'Congratulations! New best SSIM value: {SSIM_score:.4f}')
                torch.save(model.state_dict(), 'best_model.pth')
                best_score = SSIM_score

    print('Training done!')
    writer.close()

