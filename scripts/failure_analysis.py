"""Systematic look at every low-DSC ISIC2018 test image, not just a handful.

Week 3's explainability case study characterized two failure modes from 8
hand-picked images (4 worst, 4 best). This runs the same idea over every test
image below a DSC threshold -- 39 images at DSC < 0.7 -- and adds objective,
per-image metrics so the failure-mode split is measured, not eyeballed:

  - area_ratio = predicted foreground pixels / ground-truth foreground pixels.
    >1 means the model over-segments (predicts more than the true lesion);
    <1 means it under-segments. This alone distinguishes "attention pulled
    onto a bright artifact" (over-segments onto that artifact) from "attention
    collapses onto one high-contrast fleck" (under-segments the true extent)
    without needing a hand-tuned artifact detector.

  - lesion_contrast = mean image intensity inside the ground-truth mask minus
    mean intensity in a ring immediately outside it (a dilated-GT minus GT
    annulus, i.e. the actual local surrounding skin for *this* image, not a
    fixed skin-tone heuristic). Low value = a diffuse, low-contrast lesion
    boundary.

  - bg_color_std = std of pixel values outside the ground truth mask. A
    "busy" frame -- an artifact and skin both in view -- has a wider color
    spread than a frame that's just skin.

Usage:
    python scripts/failure_analysis.py --dataset ISIC2018 --dsc-below 0.7
    python scripts/failure_analysis.py --dataset ISIC2018 --dsc-below 0.7 --grid-only-first 20
"""

import argparse
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage

from loader import isic_loader
from scripts.cross_eval import find_best_checkpoint, load_model
from scripts.explainability import compute_maps, dice as _dice


def outlier_indices(dataset, dsc_below):
    """Every test-image index logged with dice < dsc_below, from the run's own log."""
    ckpt_path, run_dir = find_best_checkpoint(dataset)
    log_path = os.path.join(run_dir, 'log', 'train.info.log')
    pairs = []
    pat = re.compile(r'test image (\d+): dice: ([\d.]+)')
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                idx, d = int(m.group(1)), float(m.group(2))
                if d < dsc_below:
                    pairs.append((idx, d))
    return sorted(pairs, key=lambda x: x[1])


def image_metrics(disp_img, gt01):
    """disp_img: (H,W,3) uint8. gt01: (H,W) in {0,1}."""
    gray = disp_img.mean(axis=2)
    gt_bool = gt01 > 0.5

    dilated = ndimage.binary_dilation(gt_bool, iterations=15)
    ring = dilated & ~gt_bool
    lesion_contrast = float(gray[gt_bool].mean() - gray[ring].mean()) if ring.any() and gt_bool.any() else float('nan')

    bg = ~gt_bool
    bg_color_std = float(disp_img[bg].std()) if bg.any() else float('nan')

    return lesion_contrast, bg_color_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ISIC2018')
    ap.add_argument('--dsc-below', type=float, default=0.7)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(REPO_ROOT, 'results', 'failure_analysis', args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    outliers = outlier_indices(args.dataset, args.dsc_below)
    print(f"{len(outliers)} test images with DSC < {args.dsc_below}")

    from configs.config_setting import setting_config
    ckpt_path, _ = find_best_checkpoint(args.dataset)
    model = load_model(ckpt_path, setting_config.model_config)
    device = next(model.parameters()).device

    data_path = os.path.join(REPO_ROOT, 'data', args.dataset) + os.sep
    test_dataset = isic_loader(path_Data=data_path, train=False, Test=True)

    rows = []
    for idx, logged_dsc in outliers:
        img_t, msk_t = test_dataset[idx]
        img = img_t.unsqueeze(0).to(device).float()
        img.requires_grad_(True)
        prob, satt, cam = compute_maps(model, img)
        gt = msk_t.squeeze().numpy()
        gt01 = (gt >= 0.5).astype(np.float32)
        pred01 = (prob >= 0.5).astype(np.float32)
        d = _dice(prob, gt)

        disp = img_t.permute(1, 2, 0).numpy()
        disp = ((disp - disp.min()) / (disp.max() - disp.min() + 1e-8) * 255).astype(np.uint8)

        gt_area = gt01.sum()
        pred_area = pred01.sum()
        area_ratio = float(pred_area / gt_area) if gt_area > 0 else float('nan')
        lesion_contrast, bg_color_std = image_metrics(disp, gt01)

        rows.append(dict(idx=idx, logged_dsc=logged_dsc, recomputed_dsc=d, area_ratio=area_ratio,
                         lesion_contrast=lesion_contrast, bg_color_std=bg_color_std,
                         gt_area_frac=float(gt_area / gt01.size)))
        print(f"  #{idx:4d}  DSC={d:.3f}  area_ratio={area_ratio:6.2f}  "
             f"lesion_contrast={lesion_contrast:6.1f}  bg_std={bg_color_std:5.1f}")

    # ---- classify into failure modes by area_ratio ----
    over = [r for r in rows if r['area_ratio'] > 1.5]
    under = [r for r in rows if r['area_ratio'] < 0.67]
    mixed = [r for r in rows if 0.67 <= r['area_ratio'] <= 1.5]

    def mean(key, subset):
        vals = [r[key] for r in subset if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float('nan')

    print(f"\nover-segmenters (area_ratio>1.5): n={len(over)}  "
         f"mean bg_color_std={mean('bg_color_std', over):.1f}  mean lesion_contrast={mean('lesion_contrast', over):.1f}")
    print(f"under-segmenters (area_ratio<0.67): n={len(under)}  "
         f"mean bg_color_std={mean('bg_color_std', under):.1f}  mean lesion_contrast={mean('lesion_contrast', under):.1f}")
    print(f"mixed/other (0.67-1.5, wrong location not area): n={len(mixed)}  "
         f"mean bg_color_std={mean('bg_color_std', mixed):.1f}  mean lesion_contrast={mean('lesion_contrast', mixed):.1f}")
    print(f"\n(for reference) all {len(rows)} outliers: "
         f"mean bg_color_std={mean('bg_color_std', rows):.1f}  mean lesion_contrast={mean('lesion_contrast', rows):.1f}")

    import csv
    csv_path = os.path.join(out_dir, 'failure_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")

    # ---- grid: worst-area-ratio over-segmenter and worst-area-ratio under-segmenter, several each ----
    def panel(ax, arr, title, cmap=None, overlay_img=None):
        if overlay_img is not None:
            ax.imshow(overlay_img)
            ax.imshow(arr, cmap='jet', alpha=0.45)
        else:
            ax.imshow(arr, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    def render_group(subset, name, n=8):
        subset = sorted(subset, key=lambda r: r['recomputed_dsc'])[:n]
        if not subset:
            return
        fig, axes = plt.subplots(len(subset), 4, figsize=(12, 3.0 * len(subset)))
        if len(subset) == 1:
            axes = axes[None, :]
        for r, row in enumerate(subset):
            idx = row['idx']
            img_t, msk_t = test_dataset[idx]
            img = img_t.unsqueeze(0).to(device).float()
            img.requires_grad_(True)
            prob, satt, cam = compute_maps(model, img)
            disp = img_t.permute(1, 2, 0).numpy()
            disp = ((disp - disp.min()) / (disp.max() - disp.min() + 1e-8) * 255).astype(np.uint8)
            pred01 = (prob >= 0.5).astype(np.float32)
            panel(axes[r][0], disp, f"#{idx} DSC={row['recomputed_dsc']:.2f} ratio={row['area_ratio']:.2f}")
            panel(axes[r][1], msk_t.squeeze().numpy(), "ground truth", cmap='gray')
            panel(axes[r][2], pred01, "prediction (thresholded)", cmap='gray')
            panel(axes[r][3], cam, "Seg-Grad-CAM", overlay_img=disp)
        fig.suptitle(name, fontsize=12)
        fig.tight_layout()
        path = os.path.join(out_dir, f"_{name.lower().replace(' ', '_')}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"wrote {path}")

    render_group(over, "Over-segmenters")
    render_group(under, "Under-segmenters")


if __name__ == "__main__":
    main()
