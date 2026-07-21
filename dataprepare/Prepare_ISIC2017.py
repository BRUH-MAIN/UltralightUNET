# -*- coding: utf-8 -*-
"""Build the ISIC2017 .npy splits.

Rewrite of the upstream dataprepare/Prepare_ISIC2017.py (originally by Reza Azad),
which calls scipy.misc.imread / scipy.misc.imresize -- removed in SciPy >= 1.3.
Upstream's workaround is a second Python 3.7 environment pinned to scipy==1.2.1;
this does the same work with Pillow directly.

Semantics preserved:
  * 256x256 bilinear resize of all 2000 training images and their masks
  * split 1250 train / 150 val / 600 test, sliced in listing order
  * masks stay in 0..255 (the loader divides by 255 and thresholds at 0.5)

Two deliberate deviations, both recorded in README.md:

  1. Sorted listing. Upstream slices raw glob.glob() order, which is filesystem
     dependent and so not reproducible across machines. Sorting makes the split
     deterministic. The paper describes the split only as "random", so this is
     the most likely source of any small metric delta.

  2. uint8 storage instead of float64. scipy.misc.imresize returned uint8 and
     upstream immediately widened it with np.double(); storing the uint8 is
     therefore lossless with respect to the original pipeline, and takes the
     output from ~4 GB to ~525 MB. loader.dataset_normalized casts to float32.
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image

HEIGHT = 256
WIDTH = 256
CHANNELS = 3

N_TOTAL = 2000
N_TRAIN = 1250
N_VAL = 150
N_TEST = 600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data/ISIC2017",
                    help="directory holding the two extracted ISIC folders")
    ap.add_argument("--out", default=None, help="where to write the .npy files (default: --root)")
    args = ap.parse_args()

    out = args.out or args.root
    img_dir = os.path.join(args.root, "ISIC2017_Task1-2_Training_Input")
    gt_dir = os.path.join(args.root, "ISIC2017_Task1_Training_GroundTruth")
    for d in (img_dir, gt_dir):
        if not os.path.isdir(d):
            raise SystemExit(f"missing {d}\nRun: python scripts/download_isic.py --dataset ISIC2017")

    tr_list = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))  # see deviation (1)
    if len(tr_list) != N_TOTAL:
        raise SystemExit(f"expected {N_TOTAL} images in {img_dir}, found {len(tr_list)}")

    data = np.zeros([N_TOTAL, HEIGHT, WIDTH, CHANNELS], dtype=np.uint8)
    label = np.zeros([N_TOTAL, HEIGHT, WIDTH], dtype=np.uint8)

    print(f"Reading ISIC 2017 from {img_dir}")
    for idx, path in enumerate(tr_list):
        img = Image.open(path).convert("RGB").resize((WIDTH, HEIGHT), Image.BILINEAR)
        data[idx] = np.asarray(img, dtype=np.uint8)

        stem = os.path.splitext(os.path.basename(path))[0]  # ISIC_0000000
        gt_path = os.path.join(gt_dir, stem + "_segmentation.png")
        msk = Image.open(gt_path).convert("L").resize((WIDTH, HEIGHT), Image.BILINEAR)
        label[idx] = np.asarray(msk, dtype=np.uint8)

        if (idx + 1) % 200 == 0 or idx + 1 == N_TOTAL:
            print(f"  {idx + 1}/{N_TOTAL}")
    print("Reading ISIC 2017 finished")

    # 1250 train / 150 val / 600 test, same slice points as upstream
    splits = {
        "train": slice(0, N_TRAIN),
        "val": slice(N_TRAIN, N_TRAIN + N_VAL),
        "test": slice(N_TRAIN + N_VAL, N_TOTAL),
    }
    os.makedirs(out, exist_ok=True)
    for name, sl in splits.items():
        np.save(os.path.join(out, f"data_{name}.npy"), data[sl])
        np.save(os.path.join(out, f"mask_{name}.npy"), label[sl])
        print(f"  data_{name}.npy {data[sl].shape}  mask_{name}.npy {label[sl].shape}")

    total = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out) if f.endswith(".npy"))
    print(f"\nwrote {total / 1e6:.0f} MB of .npy to {out}/")


if __name__ == "__main__":
    main()
