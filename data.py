import os
import glob
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class MedDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.mr_paths = sorted(glob.glob(os.path.join(root, '*', 'mr_*.jpg')))
        self.ct_paths = sorted(glob.glob(os.path.join(root, '*', 'ct_*.jpg')))
        self.mask_paths = sorted(glob.glob(os.path.join(root, '*', 'mask_*.jpg')))

        self.tfs_x = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0775], std=[0.1208])
        ])

        self.tfs_y = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        self.tfs_mask = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, i):
        mr_img = self.tfs_x(Image.open(self.mr_paths[i]))
        ct_img = self.tfs_y(Image.open(self.ct_paths[i]))
        mask = self.tfs_mask(Image.open(self.mask_paths[i]))
        return mr_img, ct_img, mask

    def __len__(self):
        return len(self.ct_paths)
