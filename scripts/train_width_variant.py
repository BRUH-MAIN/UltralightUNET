"""Train a channel-width variant of UltraLight VM-UNet, for the Week 4
efficiency-push Pareto curve (DSC vs. params/FLOPs).

The paper's channel widths (c_list=[8,16,24,32,48,64], 49,457 params) are
already tiny; this trains smaller variants on ISIC2017 -- the
best-understood dataset, per results/COMPARISON.md -- to chart how DSC
degrades as the model shrinks further. Every other hyperparameter (batch
size, epochs, optimizer, schedule, seed) is untouched, matching the
replication runs.

All six c_list entries must be divisible by 4: PVMLayer chunks its input
channels into 4 for the parallel-Mamba split, and every encoder/decoder
stage's GroupNorm is hardcoded to num_groups=4 (models/UltraLight_VM_UNet.py).
c_list[0]=4 is therefore the practical floor for the shallowest layer.

Runs are directory-named UltraLight_VM_UNet_<dataset>-<tag>_<timestamp>/
(hyphen before the tag, not underscore) specifically so
scripts/cross_eval.py's `UltraLight_VM_UNet_<dataset>_*` glob -- which picks
the full-width checkpoint for cross-dataset eval -- never matches a variant
by accident.

Usage:
    python scripts/train_width_variant.py --tag half    --c-list 4,8,12,16,24,32
    python scripts/train_width_variant.py --tag quarter --c-list 4,4,8,8,12,16
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True, help='short name, e.g. "half" or "quarter"')
    ap.add_argument('--c-list', required=True,
                    help='comma-separated channel widths, e.g. 4,8,12,16,24,32 -- all must be div by 4')
    ap.add_argument('--dataset', default='ISIC2017')
    args = ap.parse_args()

    c_list = [int(x) for x in args.c_list.split(',')]
    if len(c_list) != 6:
        raise SystemExit(f"expected 6 channel widths, got {len(c_list)}: {c_list}")
    bad = [c for c in c_list if c % 4 != 0]
    if bad:
        raise SystemExit(f"every c_list entry must be divisible by 4 (GroupNorm num_groups=4 / "
                         f"PVM 4-way chunk); offending values: {bad}")

    config = setting_config
    config.model_config = dict(config.model_config, c_list=c_list)
    config.datasets = f"{args.dataset}-{args.tag}"  # label only, for logging/save_imgs
    config.data_path = os.path.join(REPO_ROOT, 'data', args.dataset) + os.sep
    config.val_batch_size = config._VAL_BATCH_SIZE[args.dataset]
    config.test_batch_size = 1

    timestamp = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')
    config.work_dir = (os.path.join(REPO_ROOT, 'results', '') +
                       f"{config.network}_{args.dataset}-{args.tag}_{timestamp}" + os.sep)

    print(f"c_list={c_list}  dataset={args.dataset}  work_dir={config.work_dir}")
    train_module.main(config)


if __name__ == "__main__":
    main()
