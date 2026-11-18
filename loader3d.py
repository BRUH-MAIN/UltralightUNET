"""Dataset for the 3D PVM Hippocampus splits (see dataprepare/Prepare_Hippocampus.py).

Mirrors loader.py's isic_loader shape (same data_{split}.npy / mask_{split}.npy
naming, same train-flag / Test-flag split selection, same random_rot_flip /
random_rotate augmentation strategy) with two differences forced by the data:

  * per-volume z-score normalization instead of loader.dataset_normalized's
    0-255 natural-image rescale -- standard practice for MRI intensity
    (nnU-Net, MONAI), and NOT dataset-wide like the 2D loader's stats, since
    per-scanner intensity offsets would otherwise leak across volumes.
  * masks are multi-class integer labels (0 bg / 1 anterior / 2 posterior),
    kept as a LongTensor for CrossEntropyLoss rather than the 2D loader's
    single float channel for BCE/Dice.
"""

import random

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset


def _zscore(vol):
    std = vol.std()
    if std < 1e-8:
        return vol - vol.mean()
    return (vol - vol.mean()) / std


class hippocampus_loader(Dataset):
    def __init__(self, path_Data, train=True, Test=False):
        super().__init__()
        self.train = train
        if train:
            self.data = np.load(path_Data + 'data_train.npy')
            self.mask = np.load(path_Data + 'mask_train.npy')
        elif Test:
            self.data = np.load(path_Data + 'data_test.npy')
            self.mask = np.load(path_Data + 'mask_test.npy')
        else:
            self.data = np.load(path_Data + 'data_val.npy')
            self.mask = np.load(path_Data + 'mask_val.npy')

    def __getitem__(self, indx):
        img = _zscore(self.data[indx].astype(np.float32))
        seg = self.mask[indx].astype(np.int64)

        if self.train:
            if random.random() > 0.5:
                img, seg = self.random_rot_flip(img, seg)
            if random.random() > 0.5:
                img, seg = self.random_rotate(img, seg)

        img = torch.from_numpy(img.copy()).unsqueeze(0)   # (1, D, H, W)
        seg = torch.from_numpy(seg.copy())                 # (D, H, W), class indices

        return img, seg

    def random_rot_flip(self, image, label):
        k = np.random.randint(0, 4)
        axes = tuple(np.random.choice(3, 2, replace=False))
        image = np.rot90(image, k, axes=axes)
        label = np.rot90(label, k, axes=axes)
        axis = np.random.randint(0, 3)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label

    def random_rotate(self, image, label):
        angle = np.random.randint(20, 80)
        axes = tuple(np.random.choice(3, 2, replace=False))
        image = ndimage.rotate(image, angle, axes=axes, order=0, reshape=False)
        label = ndimage.rotate(label, angle, axes=axes, order=0, reshape=False)
        return image, label

    def __len__(self):
        return len(self.data)
