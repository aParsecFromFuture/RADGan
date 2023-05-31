import argparse
import torch
from monai.inferers import SliceInferer
from monai.transforms import SaveImage
from networks.model import GAN
from utils import get_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, default='best_model.pth', help='Model to Evaluate')
    parser.add_argument('-data', '--data_dir', type=str, default='task/test', help='Test Data Directory')

    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()

    model = GAN(in_channels=1)
    model.load_state_dict(torch.load(args.model_path))
    model.cpu()
    model.eval()

    test_loader = get_loader(args.data_dir, batch_size=1, shuffle=False, drop_last=False)
    inferer = SliceInferer(roi_size=(256, 256), sw_batch_size=1, spatial_dim=1)
    saver = SaveImage(output_dir='./output', output_ext='.nii.gz')

    for batch in test_loader:
        y_fake = inferer(batch['mr'], model)
        saver(y_fake[0])
        break

    print('Evaluation done!')
