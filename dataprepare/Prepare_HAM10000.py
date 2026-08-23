# -*- coding: utf-8 -*-
"""Build the HAM10000 .npy splits.

Same pattern as Prepare_ISIC2017.py / Prepare_ISIC2018.py (256x256 bilinear
resize, seeded shuffle, uint8 storage), with two differences forced by this
not being a fixed-size challenge release the way the ISIC archives are:

  1. Images are paired to masks by ISIC-ID filename stem rather than assumed
     to line up 1:1. scripts/download_ham10000.py's mask archive holds 10,018
     files against ~10,015 official HAM10000 images, so pairing is done by
     set intersection and the result is whatever it is -- reported, not
     asserted against a hardcoded total.

  2. There is no official train/val/test split for this combination (it is
     not an ISIC challenge task), so split sizes are computed proportionally
     from however many pairs are actually found: 70/10/20, matching the ratio
     used for ISIC2018 for consistency, in the absence of a paper precedent
     to anchor to instead.

Same seeded-permutation rationale as the other two prep scripts: ISIC IDs
correlate with acquisition source, so a sorted/unshuffled split would bias
lesion-size distribution across splits (measured on ISIC2017, see
Prepare_ISIC2017.py's docstring). SPLIT_SEED is fixed at 42 to match
configs/config_setting.py:seed.
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image

HEIGHT = 256
WIDTH = 256
CHANNELS = 3

TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
# TEST_FRAC is the remainder, so rounding never drops or duplicates a pair.

SPLIT_SEED = 42  # matches configs/config_setting.py:seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data/HAM10000",
                    help="directory holding HAM10000_images/ and HAM10000_segmentations_lesion_tschandl/")
    ap.add_argument("--out", default=None, help="where to write the .npy files (default: --root)")
    args = ap.parse_args()

    out = args.out or args.root
    img_dir = os.path.join(args.root, "HAM10000_images")
    gt_dir = os.path.join(args.root, "HAM10000_segmentations_lesion_tschandl")
    for d in (img_dir, gt_dir):
        if not os.path.isdir(d):
            raise SystemExit(f"missing {d}\nRun: python scripts/download_ham10000.py")

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not img_paths:
        raise SystemExit(f"no .jpg images found in {img_dir}")

    # Pair by filename stem; keep only images that have a matching mask.
    pairs = []
    for path in img_paths:
        stem = os.path.splitext(os.path.basename(path))[0]  # ISIC_0000000
        gt_path = os.path.join(gt_dir, stem + "_segmentation.png")
        if os.path.exists(gt_path):
            pairs.append((path, gt_path))

    n_total = len(pairs)
    print(f"{len(img_paths)} images, {n_total} with a matching mask "
          f"({len(img_paths) - n_total} unmatched, discarded)")
    if n_total == 0:
        raise SystemExit("no image/mask pairs found -- check filename stems in "
                         f"{img_dir} vs {gt_dir}")

    order = np.random.default_rng(SPLIT_SEED).permutation(n_total)
    pairs = [pairs[i] for i in order]

    data = np.zeros([n_total, HEIGHT, WIDTH, CHANNELS], dtype=np.uint8)
    label = np.zeros([n_total, HEIGHT, WIDTH], dtype=np.uint8)

    print(f"Reading HAM10000 from {img_dir}")
    for idx, (img_path, gt_path) in enumerate(pairs):
        img = Image.open(img_path).convert("RGB").resize((WIDTH, HEIGHT), Image.BILINEAR)
        data[idx] = np.asarray(img, dtype=np.uint8)

        msk = Image.open(gt_path).convert("L").resize((WIDTH, HEIGHT), Image.BILINEAR)
        label[idx] = np.asarray(msk, dtype=np.uint8)

        if (idx + 1) % 1000 == 0 or idx + 1 == n_total:
            print(f"  {idx + 1}/{n_total}")
    print("Reading HAM10000 finished")

    n_train = round(n_total * TRAIN_FRAC)
    n_val = round(n_total * VAL_FRAC)
    n_test = n_total - n_train - n_val
    splits = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, n_total),
    }
    print(f"split: {n_train} train / {n_val} val / {n_test} test "
          f"({TRAIN_FRAC:.0%}/{VAL_FRAC:.0%}/{1 - TRAIN_FRAC - VAL_FRAC:.0%})")
    print(f"NOTE: set configs/config_setting.py val_batch_size to a divisor of {n_val} "
          f"for the HAM10000 branch (train.py asserts this).")

    os.makedirs(out, exist_ok=True)
    for name, sl in splits.items():
        np.save(os.path.join(out, f"data_{name}.npy"), data[sl])
        np.save(os.path.join(out, f"mask_{name}.npy"), label[sl])
        print(f"  data_{name}.npy {data[sl].shape}  mask_{name}.npy {label[sl].shape}")

    total = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out) if f.endswith(".npy"))
    print(f"\nwrote {total / 1e6:.0f} MB of .npy to {out}/")


if __name__ == "__main__":
    main()
