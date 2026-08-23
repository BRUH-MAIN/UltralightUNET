# -*- coding: utf-8 -*-
"""Build the PH2 .npy splits.

Same pattern as Prepare_ISIC2017.py / Prepare_ISIC2018.py (256x256 bilinear
resize, seeded shuffle, uint8 storage) -- see those docstrings for the full
rationale on the seeded permutation (ISIC IDs, and by extension PH2's IMD
IDs, are not assumed exchangeable with acquisition order).

PH2 has no official train/val/test split (it's a 200-image research release,
not a challenge task with a stated protocol), so this uses the same 70/10/20
ratio already used for HAM10000 in the absence of a paper precedent to anchor
to instead: 140 train / 20 val / 40 test.

Unlike the ISIC archives, PH2's images and masks are not two parallel flat
directories -- each of the 200 images lives in its own
`IMD<nnn>/IMD<nnn>_Dermoscopic_Image/IMD<nnn>.bmp` +
`IMD<nnn>/IMD<nnn>_lesion/IMD<nnn>_lesion.bmp` pair (see
scripts/download_ph2.py). This walks that layout directly rather than
flattening it first.
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image

HEIGHT = 256
WIDTH = 256
CHANNELS = 3

N_TOTAL = 200
N_TRAIN = 140
N_VAL = 20
N_TEST = 40
assert N_TRAIN + N_VAL + N_TEST == N_TOTAL

SPLIT_SEED = 42  # matches configs/config_setting.py:seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data/PH2",
                    help="directory holding PH2Dataset/ as extracted by scripts/download_ph2.py")
    ap.add_argument("--out", default=None, help="where to write the .npy files (default: --root)")
    args = ap.parse_args()

    out = args.out or args.root
    img_root = os.path.join(args.root, "PH2Dataset", "PH2 Dataset images")
    if not os.path.isdir(img_root):
        raise SystemExit(f"missing {img_root}\nRun: python scripts/download_ph2.py")

    imd_dirs = sorted(d for d in glob.glob(os.path.join(img_root, "IMD*")) if os.path.isdir(d))
    if len(imd_dirs) != N_TOTAL:
        raise SystemExit(f"expected {N_TOTAL} IMD folders in {img_root}, found {len(imd_dirs)}")

    pairs = []
    for d in imd_dirs:
        stem = os.path.basename(d)  # IMD002
        img_matches = glob.glob(os.path.join(d, f"{stem}_Dermoscopic_Image", "*.bmp"))
        gt_matches = glob.glob(os.path.join(d, f"{stem}_lesion", "*.bmp"))
        if len(img_matches) != 1 or len(gt_matches) != 1:
            raise SystemExit(f"{d}: expected exactly one dermoscopic image and one lesion mask, "
                             f"found {len(img_matches)} and {len(gt_matches)}")
        pairs.append((img_matches[0], gt_matches[0]))

    # sort for a deterministic starting point, then permute with a fixed seed so the
    # split is both reproducible and unbiased with respect to IMD ID.
    order = np.random.default_rng(SPLIT_SEED).permutation(N_TOTAL)
    pairs = [pairs[i] for i in order]

    data = np.zeros([N_TOTAL, HEIGHT, WIDTH, CHANNELS], dtype=np.uint8)
    label = np.zeros([N_TOTAL, HEIGHT, WIDTH], dtype=np.uint8)

    print(f"Reading PH2 from {img_root}")
    for idx, (img_path, gt_path) in enumerate(pairs):
        img = Image.open(img_path).convert("RGB").resize((WIDTH, HEIGHT), Image.BILINEAR)
        data[idx] = np.asarray(img, dtype=np.uint8)

        # PH2 masks are 1-bit (mode "1"); convert("L") maps True/False -> 255/0,
        # matching the 0..255 mask convention the other prep scripts use.
        msk = Image.open(gt_path).convert("L").resize((WIDTH, HEIGHT), Image.BILINEAR)
        label[idx] = np.asarray(msk, dtype=np.uint8)

        if (idx + 1) % 50 == 0 or idx + 1 == N_TOTAL:
            print(f"  {idx + 1}/{N_TOTAL}")
    print("Reading PH2 finished")

    splits = {
        "train": slice(0, N_TRAIN),
        "val": slice(N_TRAIN, N_TRAIN + N_VAL),
        "test": slice(N_TRAIN + N_VAL, N_TOTAL),
    }
    print(f"split: {N_TRAIN} train / {N_VAL} val / {N_TEST} test (70%/10%/20%)")

    os.makedirs(out, exist_ok=True)
    for name, sl in splits.items():
        np.save(os.path.join(out, f"data_{name}.npy"), data[sl])
        np.save(os.path.join(out, f"mask_{name}.npy"), label[sl])
        print(f"  data_{name}.npy {data[sl].shape}  mask_{name}.npy {label[sl].shape}")

    total = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out) if f.endswith(".npy"))
    print(f"\nwrote {total / 1e6:.0f} MB of .npy to {out}/")


if __name__ == "__main__":
    main()
