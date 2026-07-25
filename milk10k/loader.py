"""MILK10k dataset: one lesion = one example, with its paired images.

The unit is the lesion, not the image. Each item returns the requested
modalities (dermatoscopic and/or clinical) as CHW tensors, plus the label and the
skin-tone class (kept for the per-skin-tone fairness breakdown at test time).

Images are loaded on the fly from JPEG and resized. The full set is only ~314 MB
of JPEG, so decoding is not the bottleneck, and this avoids materialising a ~2 GB
array of both modalities.
"""

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet statistics -- standard, and compatible if a pretrained encoder is added later.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

# manifest column holding the isic_id for each modality
_ID_COL = {'derm': 'derm_isic_id', 'clin': 'clinical_isic_id'}


def build_transforms(input_size, train):
    if train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


class MILK10kDataset(Dataset):
    def __init__(self, manifest, images_dir, split, modalities=('derm',),
                 input_size=256, train=False):
        df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
        self.df = df[df.split == split].reset_index(drop=True)
        self.images_dir = images_dir
        self.modalities = tuple(modalities)
        self.tf = build_transforms(input_size, train and split == 'train')
        for m in self.modalities:
            if m not in _ID_COL:
                raise ValueError(f'unknown modality {m!r}')

    def __len__(self):
        return len(self.df)

    def _load(self, isic_id):
        path = os.path.join(self.images_dir, f'{isic_id}.jpg')
        if not os.path.exists(path):
            raise FileNotFoundError(f'missing image {path}')
        return self.tf(Image.open(path).convert('RGB'))

    def __getitem__(self, i):
        row = self.df.iloc[i]
        item = {m: self._load(row[_ID_COL[m]]) for m in self.modalities}
        item['label'] = int(row['label_idx'])
        # skin tone can be NaN in principle; -1 marks "unknown" for the fairness split
        st = row.get('skin_tone_class', -1)
        item['skin_tone'] = int(st) if not (isinstance(st, float) and np.isnan(st)) else -1
        item['lesion_id'] = row['lesion_id']
        return item
