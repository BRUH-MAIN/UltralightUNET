"""3D-specific additions for the PVM-3D pipeline: multi-class Dice+CE loss,
FLOPs/param counting at a 3D input shape, and a mid-slice sanity-check image.

Everything dimension-agnostic (set_seed, get_logger, get_optimizer,
get_scheduler, log_config_info, check_environment) is reused directly from
utils.py rather than duplicated here -- see train3d.py.
"""

import copy
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss3D(nn.Module):
    """Standard multi-class segmentation loss (nnU-Net / MONAI convention):
    voxelwise cross-entropy + soft Dice averaged over all classes, including
    background. Unlike the 2D repo's BceDiceLoss (binary, sigmoid), this takes
    raw multi-class logits (B, num_classes, D, H, W) and integer targets
    (B, D, H, W), matching UltraLight_VM_UNet_3D / StandardUNet3D's output
    convention.
    """
    def __init__(self, num_classes=3, wb=1, wd=1, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.wb = wb
        self.wd = wd
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        celoss = self.ce(pred, target)

        probs = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        intersection = (probs * target_onehot).sum(dims)
        denom = probs.sum(dims) + target_onehot.sum(dims)
        dice_per_class = (2 * intersection + self.smooth) / (denom + self.smooth)
        diceloss = 1 - dice_per_class.mean()

        return self.wb * celoss + self.wd * diceloss


def per_class_dice(pred_logits, target, num_classes=3, smooth=1e-6):
    """Hard-label per-class DSC for one batch. pred_logits: (B,C,D,H,W) raw
    logits. target: (B,D,H,W) integer labels. Returns a length-num_classes
    numpy array. Used by val/test logging (engine3d.py) for the
    per-class breakdown the plan's failure-analysis step needs."""
    pred = pred_logits.argmax(dim=1)
    dsc = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().item()
        denom = p.sum().item() + t.sum().item()
        dsc[c] = (2 * inter + smooth) / (denom + smooth)
    return dsc


def cal_params_flops3d(model, patch, input_channels, logger):
    """3D analogue of utils.cal_params_flops. Profiles a deep copy (see that
    function's docstring for why: thop registers total_ops/total_params
    buffers that would otherwise pollute checkpoints)."""
    from thop import profile
    flops, params = profile(copy.deepcopy(model),
                             inputs=(torch.randn(1, input_channels, patch, patch, patch).cuda(),),
                             verbose=False)
    print('flops', flops / 1e9)
    print('params', params / 1e6)

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.3fM" % (total / 1e6))
    logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: {total/1e6:.4f}')


def save_mid_slices(img, msk, msk_pred, i, save_path, test_data_name=None):
    """Axial/coronal/sagittal mid-slice overlay, the 3D analogue of
    utils.save_imgs. img/msk/msk_pred: (1, 1-or-C, D, H, W) or (1, D, H, W)
    tensors/arrays for a single volume (batch size 1, same assumption
    engine.test_one_epoch makes for the 2D save_imgs)."""
    img = img.squeeze(0).squeeze(0).detach().cpu().numpy()   # (D, H, W)
    msk = msk.squeeze(0).detach().cpu().numpy() if hasattr(msk, 'detach') else np.squeeze(msk, axis=0)
    msk_pred = msk_pred.squeeze(0).detach().cpu().numpy() if hasattr(msk_pred, 'detach') else np.squeeze(msk_pred, axis=0)

    d, h, w = img.shape
    slices = [(img[d // 2], msk[d // 2], msk_pred[d // 2]),
              (img[:, h // 2], msk[:, h // 2], msk_pred[:, h // 2]),
              (img[:, :, w // 2], msk[:, :, w // 2], msk_pred[:, :, w // 2])]

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for row, (im, gt, pr) in enumerate(slices):
        axes[row, 0].imshow(im, cmap='gray'); axes[row, 0].axis('off')
        axes[row, 1].imshow(gt, cmap='viridis', vmin=0, vmax=2); axes[row, 1].axis('off')
        axes[row, 2].imshow(pr, cmap='viridis', vmin=0, vmax=2); axes[row, 2].axis('off')

    prefix = (test_data_name + '_') if test_data_name is not None else ''
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{prefix}{i}.png'))
    plt.close(fig)
