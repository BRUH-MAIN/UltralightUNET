"""Train ISIC2018 with extra appearance augmentation (CLAHE + synthetic
hair/mark lines), to test whether it addresses the failure modes found by
scripts/failure_analysis.py.

That analysis, run over all 39 ISIC2018 test images with DSC < 0.7, found:
  - 23/39 (59%) over-segment. The two extreme cases are non-skin artifact
    confusion (glare, colored gauze); the much larger middle group predicts
    a correctly-located but too-large, over-smoothed blob, often bleeding
    onto nearby hair/ink marks or ruler lines.
  - 12/39 (31%) under-segment: a diffuse, low-contrast lesion, where the
    model fixates on one small high-contrast fleck instead of the full
    extent (mean |lesion_contrast| 15.2 vs 36.2 for the over-segmenters).

Two augmentations target those directly, applied train-time only via
loader.py's isic_loader(extra_augment=True) (opt-in -- the canonical
replication runs in results/COMPARISON.md are unaffected):
  - CLAHE (p=0.5): boosts local contrast, aimed at the diffuse-lesion group.
  - synthetic hair/mark lines (p=0.3): dark curved lines drawn onto the
    image with the ground truth left untouched, aimed at the
    marks-pull-the-boundary group.

Same directory-naming convention as train_width_variant.py -- hyphenated,
not underscored, so scripts/cross_eval.py's dataset glob never picks this up
by accident.

Usage:
    python scripts/train_augmented.py --dataset ISIC2018
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
    ap.add_argument('--dataset', default='ISIC2018')
    ap.add_argument('--tag', default='augmented')
    args = ap.parse_args()

    config = setting_config
    config.datasets = f"{args.dataset}-{args.tag}"  # label only
    config.data_path = os.path.join(REPO_ROOT, 'data', args.dataset) + os.sep
    config.val_batch_size = config._VAL_BATCH_SIZE[args.dataset]
    config.test_batch_size = 1
    config.extra_augment = True

    timestamp = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')
    config.work_dir = (os.path.join(REPO_ROOT, 'results', '') +
                       f"{config.network}_{args.dataset}-{args.tag}_{timestamp}" + os.sep)

    print(f"dataset={args.dataset}  extra_augment=True  work_dir={config.work_dir}")
    train_module.main(config)


if __name__ == "__main__":
    main()
