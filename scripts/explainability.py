"""Explainability figures: SC_Att_Bridge spatial attention + Seg-Grad-CAM.

Two complementary views of what the 49,457-parameter model is "looking at",
neither requiring any change to the model:

1. Spatial attention (SC_Att_Bridge). The model already computes a per-pixel
   spatial attention map at each of its 5 skip-connection levels
   (models/UltraLight_VM_UNet.py: Spatial_Att_Bridge.forward). This hooks
   `model.scab.satt` and reads its return value straight off the forward
   pass -- no backward, no target class, just what the architecture itself
   already produces internally. The shallowest map (satt1, H/2 x W/2) is used
   here since it's closest to the input's own resolution.

2. Seg-Grad-CAM (Vinogradova et al. 2020), adapted from classification
   Grad-CAM: instead of backpropagating a class logit, this backpropagates
   the sum of predicted foreground probability over the model's own predicted
   region (`prob >= threshold`) into the last feature map before the 1x1
   output conv (`self.final`), captured via a forward pre-hook on `final`
   rather than editing the model. Channel-wise gradient-averaged weights,
   ReLU, upsample -- standard Grad-CAM from there.

Case study default: the ISIC2018 low-Dice outliers and high-Dice examples
identified in results/COMPARISON.md's per-image variance section, to see
whether the two views suggest a common failure mode for the outliers.

Usage:
    python scripts/explainability.py --dataset ISIC2018 \
        --indices 42 359 40 347 123 338 444 501 \
        --out results/explainability/ISIC2018
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from configs.config_setting import setting_config
from loader import isic_loader
from models.UltraLight_VM_UNet import UltraLight_VM_UNet
from scripts.cross_eval import find_best_checkpoint, load_model  # reuse, don't reimplement


def dice(pred, gt, threshold=0.5):
    p = (pred >= threshold).astype(np.float32)
    g = (gt >= 0.5).astype(np.float32)
    inter = (p * g).sum()
    denom = p.sum() + g.sum()
    return float(2 * inter / denom) if denom else 0.0


def compute_maps(model, img, threshold=0.5):
    """Run one forward+backward pass, returning (prob, spatial_att, gradcam),
    all as (H, W) numpy arrays at the input's own resolution."""
    satt_capture = {}

    def satt_hook(module, inputs, output):
        satt_capture['maps'] = output  # (satt1..satt5), each (1,1,h,w)

    final_input_capture = {}

    def final_pre_hook(module, inputs):
        x = inputs[0]
        x.retain_grad()
        final_input_capture['x'] = x

    h1 = model.scab.satt.register_forward_hook(satt_hook)
    h2 = model.final.register_forward_pre_hook(final_pre_hook)
    try:
        model.zero_grad(set_to_none=True)
        prob = model(img)  # (1,1,H,W)

        pred_region = (prob >= threshold).float().detach()
        score = (prob * pred_region).sum()
        score.backward()

        final_input = final_input_capture['x']
        grad = final_input.grad  # (1,C,h,w)
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * final_input.detach()).sum(dim=1, keepdim=True))
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=img.shape[2:], mode='bilinear', align_corners=True)

        satt1 = satt_capture['maps'][0].detach()  # (1,1,h1,w1), already in [0,1] (sigmoid)
        satt1 = F.interpolate(satt1, size=img.shape[2:], mode='bilinear', align_corners=True)
    finally:
        h1.remove()
        h2.remove()

    return (prob.detach().squeeze().cpu().numpy(),
           satt1.squeeze().cpu().numpy(),
           cam.squeeze().cpu().numpy())


def panel(ax, arr, title, cmap=None, overlay_img=None):
    if overlay_img is not None:
        ax.imshow(overlay_img)
        ax.imshow(arr, cmap='jet', alpha=0.45)
    else:
        ax.imshow(arr, cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ISIC2018')
    ap.add_argument('--indices', type=int, nargs='+', required=True,
                    help='test-set indices, matching the "test image N" lines in the run log')
    ap.add_argument('--out', default=None)
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(REPO_ROOT, 'results', 'explainability', args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    config = setting_config
    ckpt_path, run_dir = find_best_checkpoint(args.dataset)
    print(f"model: {args.dataset} ({os.path.relpath(ckpt_path, REPO_ROOT)})")
    model = load_model(ckpt_path, config.model_config)

    data_path = os.path.join(REPO_ROOT, 'data', args.dataset) + os.sep
    test_dataset = isic_loader(path_Data=data_path, train=False, Test=True)

    rows = []
    for idx in args.indices:
        img_np, msk_np = test_dataset[idx]
        img = img_np.unsqueeze(0).cuda().float()
        img.requires_grad_(True)
        msk = msk_np.squeeze().numpy()

        prob, satt, cam = compute_maps(model, img, args.threshold)
        d = dice(prob, msk, args.threshold)

        disp_img = img_np.permute(1, 2, 0).detach().cpu().numpy()
        disp_img = disp_img - disp_img.min()
        disp_img = disp_img / (disp_img.max() + 1e-8)

        fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
        panel(axes[0], disp_img, f"image {idx}  (DSC {d:.3f})")
        panel(axes[1], msk, "ground truth", cmap='gray')
        panel(axes[2], (prob >= args.threshold).astype(np.float32), "prediction", cmap='gray')
        panel(axes[3], satt, "spatial attention", overlay_img=disp_img)
        panel(axes[4], cam, "Seg-Grad-CAM", overlay_img=disp_img)
        fig.tight_layout()
        fig_path = os.path.join(out_dir, f"{idx}.png")
        fig.savefig(fig_path, dpi=130)
        plt.close(fig)
        print(f"  image {idx}: DSC={d:.4f}  -> {fig_path}")
        rows.append((idx, d, disp_img, msk, prob, satt, cam))

    # One combined grid, sorted worst -> best, for a single at-a-glance comparison.
    rows.sort(key=lambda r: r[1])
    n = len(rows)
    fig, axes = plt.subplots(n, 5, figsize=(15, 3.0 * n))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["image (DSC)", "ground truth", "prediction", "spatial attention", "Seg-Grad-CAM"]
    for r, (idx, d, disp_img, msk, prob, satt, cam) in enumerate(rows):
        panel(axes[r][0], disp_img, f"#{idx}  DSC {d:.3f}")
        panel(axes[r][1], msk, "", cmap='gray')
        panel(axes[r][2], (prob >= args.threshold).astype(np.float32), "", cmap='gray')
        panel(axes[r][3], satt, "", overlay_img=disp_img)
        panel(axes[r][4], cam, "", overlay_img=disp_img)
        if r == 0:
            for c, t in enumerate(col_titles):
                axes[r][c].set_title(t, fontsize=10)
    fig.tight_layout()
    grid_path = os.path.join(out_dir, "_grid_worst_to_best.png")
    fig.savefig(grid_path, dpi=130)
    plt.close(fig)
    print(f"\nwrote combined grid -> {grid_path}")


if __name__ == "__main__":
    main()
