"""Train (or evaluate) the plain U-Net baseline with the exact UltraLight replication recipe.

The UltraLight paper's Table 1 gives a U-Net row for ISIC2017, ISIC2018 and PH2, and those paper
numbers are what results/COMPARISON.md quotes for those datasets. HAM10000 is not in the paper, so
there is no U-Net number to quote -- this script trains one, on this repo's own HAM10000 split
(dataprepare/Prepare_HAM10000.py, seed 42), so it is directly comparable to the repo's UltraLight
HAM10000 result.

Everything except the network is untouched from configs/config_setting.py: batch 8, 250 epochs,
AdamW lr 1e-3 / wd 1e-2, CosineAnnealingLR T_max=50 eta_min=1e-5, BceDiceLoss, seed 42, threshold
0.5, fp32, 256x256 input, the same loader and augmentation. U-Net uses PyTorch's default init.

Runs are directory-named UNet_<dataset>_<timestamp>/, so scripts/cross_eval.py's
`UltraLight_VM_UNet_<dataset>_*` glob never picks one up.

Usage:
    python scripts/train_unet.py --dataset HAM10000
    python scripts/train_unet.py --dataset HAM10000 --eval-weights results/UNet_HAM10000_<ts>/checkpoints/best-*.pth
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from configs.config_setting import setting_config
import train as train_module
import test as test_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='HAM10000')
    ap.add_argument('--base', type=int, default=16, help='first-stage width; 16 -> 1.94 M params')
    ap.add_argument('--seed', type=int, default=None, help='override config.seed (default 42)')
    ap.add_argument('--tag', default='', help='optional suffix for the run directory')
    ap.add_argument('--eval-weights', help='skip training; evaluate this checkpoint on the test split')
    args = ap.parse_args()

    config = setting_config
    config.network = 'UNet'
    config.model_config = {'num_classes': 1, 'input_channels': 3, 'base': args.base}
    config.datasets = args.dataset
    config.data_path = os.path.join(REPO_ROOT, 'data', args.dataset) + os.sep
    config.val_batch_size = config._VAL_BATCH_SIZE[args.dataset]
    config.test_batch_size = 1
    if args.seed is not None:
        config.seed = args.seed

    timestamp = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')
    config.work_dir = (os.path.join(REPO_ROOT, 'results', '') +
                       f"UNet_{args.dataset}{args.tag}_{timestamp}" + os.sep)

    print(f"network=UNet base={args.base} dataset={args.dataset} seed={config.seed} "
          f"work_dir={config.work_dir}")
    if args.eval_weights:
        test_module.main(config, weights=args.eval_weights, work_dir=config.work_dir)
    else:
        train_module.main(config)


if __name__ == "__main__":
    main()
