import random

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset


# ===== normalize over the dataset
def dataset_normalized(imgs):
    # PATCH: float32 instead of the implicit float64. Same arithmetic, half the RAM
    # (the train split is 1250x256x256x3; float64 would be ~2 GB, and this box has 16 GB).
    # engine.py casts to .float() on the GPU anyway, so nothing downstream changes.
    imgs = imgs.astype(np.float32)
    # upstream allocated an empty array here and immediately rebound the name; the
    # allocation was never read.
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def _clahe_augment(img_u8):
    """Contrast-limited adaptive histogram equalization on the L channel
    (LAB space) -- boosts local contrast, which is exactly what a diffuse,
    low-contrast lesion boundary lacks. cv2 imported lazily so opencv is only
    a hard dependency for callers that actually use extra_augment=True."""
    import cv2
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab2 = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def _hair_augment(img_u8, n_range=(2, 6)):
    """Draw a few synthetic dark curved lines (hair/surgical-ink/ruler-mark
    stand-ins) onto the image; the ground truth mask is untouched. The point
    is teaching the model these are not lesion, since real hair/ink marks
    were observed pulling predicted boundaries toward them (see
    results/COMPARISON.md's failure-analysis note)."""
    import cv2
    img = img_u8.copy()
    h, w = img.shape[:2]
    for _ in range(np.random.randint(*n_range)):
        # near-grayscale (dark hair/ink/ruler marks are achromatic, not
        # tinted) -- one base value + small per-channel jitter, not fully
        # independent RGB, which can otherwise land on a colorful line that
        # teaches the wrong lesson.
        base = np.random.randint(5, 45)
        color = tuple(int(np.clip(base + np.random.randint(-8, 8), 0, 255)) for _ in range(3))
        thickness = np.random.randint(1, 3)
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        pts = [(x, y)]
        for _ in range(np.random.randint(4, 9)):
            x = int(np.clip(x + np.random.randint(-18, 18), 0, w - 1))
            y = int(np.clip(y + np.random.randint(-18, 18), 0, h - 1))
            pts.append((x, y))
        for p, q in zip(pts[:-1], pts[1:]):
            cv2.line(img, p, q, color, thickness, cv2.LINE_AA)
    return img


class isic_loader(Dataset):
    """ dataset class for Brats datasets
    """
    def __init__(self, path_Data, train = True, Test = False, extra_augment = False):
        super(isic_loader, self)
        self.train = train
        # Opt-in, defaults False: existing replication runs (results/COMPARISON.md)
        # were produced without this, and changing the default would silently
        # change what every prior documented run.py invocation reproduces.
        # scripts/train_augmented.py is the only current caller that sets it True.
        self.extra_augment = extra_augment
        if train:
          self.data   = np.load(path_Data+'data_train.npy')
          self.mask   = np.load(path_Data+'mask_train.npy')
        else:
          if Test:
            self.data   = np.load(path_Data+'data_test.npy')
            self.mask   = np.load(path_Data+'mask_test.npy')
          else:
            self.data   = np.load(path_Data+'data_val.npy')
            self.mask   = np.load(path_Data+'mask_val.npy')          
        
        self.data   = dataset_normalized(self.data)
        self.mask   = np.expand_dims(self.mask, axis=3)
        self.mask   = (self.mask/255.).astype(np.float32)  # PATCH: float32, see above

    def __getitem__(self, indx):
        img = self.data[indx]
        seg = self.mask[indx]
        if self.train:
            if random.random() > 0.5:
                img, seg = self.random_rot_flip(img, seg)
            if random.random() > 0.5:
                img, seg = self.random_rotate(img, seg)
            if self.extra_augment:
                # Appearance-only: seg (the lesion boundary) is never touched by
                # either of these, only img.
                img_u8 = np.clip(img, 0, 255).astype(np.uint8)
                if random.random() < 0.5:
                    img_u8 = _clahe_augment(img_u8)
                if random.random() < 0.3:
                    img_u8 = _hair_augment(img_u8)
                img = img_u8.astype(np.float32)

        seg = torch.tensor(seg.copy())
        img = torch.tensor(img.copy())
        img = img.permute( 2, 0, 1)
        seg = seg.permute( 2, 0, 1)

        return img, seg
    
    def random_rot_flip(self,image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label
    
    def random_rotate(self,image, label):
        angle = np.random.randint(20, 80)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label


               
    def __len__(self):
        return len(self.data)
    