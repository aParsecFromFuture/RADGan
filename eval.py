import os
import argparse
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from torchvision.utils import make_grid
from data import MedDataset
from torch.utils.data import DataLoader
from networks.model import GAN
from monai.metrics.regression import SSIMMetric
from monai.losses import MaskedLoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, default='best_model.pth', help='Model to Evaluate')
    parser.add_argument('-data', '--data_dir', type=str, default='dataset/test', help='Test Data Directory')
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-pair', '--pair_mode', type=bool, default=True, help='Save Prediction and GT images as pairs')

    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()

    model = GAN(in_channels=1).to(args.device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    test_ds = MedDataset(root=args.data_dir)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, drop_last=False)

    data_range = torch.tensor([1], device=args.device)
    ssim = MaskedLoss(loss=SSIMMetric(spatial_dims=2, data_range=data_range))
    SSIMMetric(spatial_dims=2, data_range=data_range)

    if not os.path.exists('eval'):
        os.mkdir('eval')

    for x, y, mask in tqdm(test_loader):
        x, y, mask = x.to(args.device), y.to(args.device), mask.to(args.device)
        y_fake = model.generate(x)
        test_score = ssim(y, y_fake, mask=mask).item()

        if args.pair_mode:
            img_grid = make_grid([y[0], y_fake[0]])
            save_image(img_grid, os.path.join('eval', f'recon_{test_score}.png'))
        else:
            save_image(y_fake[0], os.path.join('eval', f'recon_{test_score}.png'))

    print('Evaluation done!')
