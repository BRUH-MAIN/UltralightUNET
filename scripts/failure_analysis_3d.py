"""Per-case DSC broken down by hippocampus volume size, for PVM-3D (and,
for comparison, StandardUNet3D) -- docs/3d_extension_plan.md's "Next steps"
#3: check whether the 2D repo's measured, size-linked over-segmentation bias
(scripts/failure_analysis.py: 59% of ISIC2018 outliers over-segment, corr(lesion
size, over-segmentation severity) = -0.53) generalizes to the 3D, entirely-
small-ROI Hippocampus task.

For every test volume: true foreground voxel count (size), predicted
foreground voxel count, area_ratio = predicted / true (>1 over-segments, <1
under-segments, mirroring failure_analysis.py's ISIC metric), and DSC. Reports
the size vs. area_ratio correlation directly -- the same diagnostic, on real
volumes instead of hand-picked cases.

Usage:
    python scripts/failure_analysis_3d.py                        # PVM-3D only
    python scripts/failure_analysis_3d.py --network StandardUNet3D --compare
"""

import argparse
import csv
import glob
import os
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
from scipy.stats import pearsonr

from configs.config_setting_3d import setting_config
from loader3d import hippocampus_loader

RESULTS_ROOT = os.path.join(REPO_ROOT, "results")


def find_best_checkpoint(network):
    run_dirs = glob.glob(os.path.join(RESULTS_ROOT, f"{network}_Hippocampus_*"))
    if not run_dirs:
        raise SystemExit(f"no results/{network}_Hippocampus_* run directory found -- "
                         f"train it first: python train3d.py --network {network}")
    run_dirs.sort(key=os.path.getmtime)
    run_dir = run_dirs[-1]
    ckpts = glob.glob(os.path.join(run_dir, "checkpoints", "best-*.pth"))
    if not ckpts:
        raise SystemExit(f"no best-*.pth checkpoint in {run_dir}/checkpoints")
    return ckpts[-1], run_dir


def load_model(network, ckpt_path):
    c_list = setting_config._MODEL_CLIST[network]
    if network == 'UltraLight_VM_UNet_3D':
        from models.UltraLight_VM_UNet_3D import UltraLight_VM_UNet_3D
        model = UltraLight_VM_UNet_3D(num_classes=3, input_channels=1, c_list=c_list)
    elif network == 'StandardUNet3D':
        from models.StandardUNet3D import StandardUNet3D
        model = StandardUNet3D(num_classes=3, input_channels=1, c_list=c_list)
    else:
        raise ValueError(network)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state)
    return model.cuda().eval()


def per_volume_metrics(model, test_dataset):
    rows = []
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            img_t, msk_t = test_dataset[idx]
            img = img_t.unsqueeze(0).cuda().float()
            out = model(img)
            pred = out.argmax(dim=1).squeeze(0).cpu().numpy()
            gt = msk_t.numpy()

            gt_fg = gt > 0
            pred_fg = pred > 0
            gt_size = int(gt_fg.sum())
            pred_size = int(pred_fg.sum())
            area_ratio = float(pred_size / gt_size) if gt_size > 0 else float('nan')

            inter = np.logical_and(gt_fg, pred_fg).sum()
            denom = gt_fg.sum() + pred_fg.sum()
            dsc = float(2 * inter / denom) if denom > 0 else float('nan')

            rows.append(dict(idx=idx, gt_size=gt_size, pred_size=pred_size,
                             area_ratio=area_ratio, dsc=dsc))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--network', default='UltraLight_VM_UNet_3D',
                    choices=['UltraLight_VM_UNet_3D', 'StandardUNet3D'])
    ap.add_argument('--compare', action='store_true',
                    help='also run StandardUNet3D and overlay both on one plot')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(RESULTS_ROOT, 'failure_analysis', 'Hippocampus3D')
    os.makedirs(out_dir, exist_ok=True)

    test_dataset = hippocampus_loader(path_Data=setting_config.data_path, train=False, Test=True)

    networks = [args.network] + (['StandardUNet3D'] if args.compare and args.network != 'StandardUNet3D' else [])
    results = {}
    for net in networks:
        ckpt_path, run_dir = find_best_checkpoint(net)
        print(f"{net}: loading {ckpt_path}")
        model = load_model(net, ckpt_path)
        rows = per_volume_metrics(model, test_dataset)
        results[net] = rows
        del model
        torch.cuda.empty_cache()

        sizes = np.array([r['gt_size'] for r in rows], dtype=np.float64)
        ratios = np.array([r['area_ratio'] for r in rows], dtype=np.float64)
        dscs = np.array([r['dsc'] for r in rows], dtype=np.float64)

        r_ratio, p_ratio = pearsonr(sizes, ratios)
        r_dsc, p_dsc = pearsonr(sizes, dscs)
        n_over = int((ratios > 1.1).sum())
        n_under = int((ratios < 0.9).sum())

        print(f"  {net}: n={len(rows)} test volumes, mean DSC={dscs.mean():.4f}, "
             f"mean area_ratio={ratios.mean():.3f}")
        print(f"  {net}: corr(gt_size, area_ratio)={r_ratio:.3f} (p={p_ratio:.3f}), "
             f"corr(gt_size, dsc)={r_dsc:.3f} (p={p_dsc:.3f})")
        print(f"  {net}: over-segmenters (ratio>1.1): {n_over}/{len(rows)}, "
             f"under-segmenters (ratio<0.9): {n_under}/{len(rows)}")

        csv_path = os.path.join(out_dir, f'{net}_per_volume.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {'UltraLight_VM_UNet_3D': 'tab:orange', 'StandardUNet3D': 'tab:blue'}
    for net, rows in results.items():
        sizes = [r['gt_size'] for r in rows]
        ratios = [r['area_ratio'] for r in rows]
        dscs = [r['dsc'] for r in rows]
        axes[0].scatter(sizes, ratios, label=net, alpha=0.7, color=colors.get(net))
        axes[1].scatter(sizes, dscs, label=net, alpha=0.7, color=colors.get(net))
    axes[0].axhline(1.0, color='gray', linestyle='--', linewidth=1)
    axes[0].set_xlabel('ground-truth hippocampus size (voxels)')
    axes[0].set_ylabel('area_ratio (pred / gt)')
    axes[0].set_title('over/under-segmentation vs. volume size')
    axes[0].legend()
    axes[1].set_xlabel('ground-truth hippocampus size (voxels)')
    axes[1].set_ylabel('DSC')
    axes[1].set_title('DSC vs. volume size')
    axes[1].legend()
    fig.tight_layout()
    plot_path = os.path.join(out_dir, 'size_vs_failure.png')
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"\nwrote {plot_path}")


if __name__ == "__main__":
    main()
