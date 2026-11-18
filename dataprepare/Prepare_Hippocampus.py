# -*- coding: utf-8 -*-
"""Build the MSD Task04_Hippocampus .npy splits for PVM-3D.

Reads the 263 labeled NIfTI volumes from data/Task04_Hippocampus/{imagesTr,
labelsTr}/ (see scripts/download_hippocampus.py), center-pads/crops each to a
fixed 64x64x64 patch, and writes data_{train,val,test}.npy / mask_{...}.npy
under the output directory -- the same file-naming convention loader.py's
isic_loader uses for the 2D datasets, so the 3D loader (loader3d.py) can follow
the identical load-by-suffix pattern.

No resampling: MSD Hippocampus volumes are natively tiny (~35-50 voxels/side),
smaller than the 64^3 patch in every case observed, so padding to 64 and
center-cropping any that exceed it preserves native resolution rather than
distorting anatomy through interpolation -- see docs/3d_extension_plan.md.

Intensities are stored as raw float32 (no rescale here); per-volume z-score
normalization happens in loader3d.py at load time, following standard
MRI-segmentation practice (nnU-Net, MONAI) rather than the 0-255 natural-image
convention loader.dataset_normalized uses for the 2D skin-lesion datasets.

Labels: 0 = background, 1 = anterior, 2 = posterior (MSD Task04 convention).

Split: seeded permutation (SPLIT_SEED=42, matching configs/config_setting.py's
seed and the same reproducible-but-unbiased approach Prepare_ISIC2017.py uses,
for the same reason -- slicing raw glob order is filesystem dependent and
sorted order can correlate with acquisition batch). 200 train / 30 val / 30
test -- val/test sizes chosen with small non-trivial divisors so
val_batch_size/test_batch_size can divide them exactly (loaders use
drop_last=True).

N_TOTAL is 260, not the 263 docs/3d_extension_plan.md estimated before the
archive was actually inspected: dataset.json's numTraining is 260, and
imagesTr/ holds 260 real (non-AppleDouble) volumes -- confirmed against the
downloaded archive, not carried over from the plan's estimate.
"""

import argparse
import glob
import os

import numpy as np

PATCH = 64
N_TOTAL = 260
N_TRAIN = 200
N_VAL = 30
N_TEST = 30
assert N_TRAIN + N_VAL + N_TEST == N_TOTAL

SPLIT_SEED = 42  # matches configs/config_setting_3d.py:seed


def _pad_crop_to_patch(vol, patch=PATCH, order=0):
    """Center-pad with zeros to at least `patch` in every dim, then
    center-crop to exactly `patch`. `order` is unused for image data (kept as
    a no-op parameter so the same helper signature works for masks)."""
    d, h, w = vol.shape
    pad = [max(patch - d, 0), max(patch - h, 0), max(patch - w, 0)]
    if any(pad):
        pads = [(p // 2, p - p // 2) for p in pad]
        vol = np.pad(vol, pads, mode="constant", constant_values=0)
    d, h, w = vol.shape
    start = [(d - patch) // 2, (h - patch) // 2, (w - patch) // 2]
    return vol[start[0]:start[0] + patch, start[1]:start[1] + patch, start[2]:start[2] + patch]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data/Task04_Hippocampus",
                    help="directory holding imagesTr/ and labelsTr/")
    ap.add_argument("--out", default=None, help="where to write the .npy files (default: --root)")
    args = ap.parse_args()

    try:
        import nibabel as nib
    except ImportError:
        raise SystemExit("nibabel is not installed.\n  pip install nibabel")

    out = args.out or args.root
    img_dir = os.path.join(args.root, "imagesTr")
    lbl_dir = os.path.join(args.root, "labelsTr")
    for d in (img_dir, lbl_dir):
        if not os.path.isdir(d):
            raise SystemExit(f"missing {d}\nRun: python scripts/download_hippocampus.py")

    # MSD tars ship AppleDouble "._*" sidecar files alongside the real .nii.gz;
    # they are not volumes and would break the 1:1 image/label pairing.
    img_paths = sorted(p for p in glob.glob(os.path.join(img_dir, "*.nii.gz"))
                        if not os.path.basename(p).startswith("._"))
    if len(img_paths) != N_TOTAL:
        raise SystemExit(f"expected {N_TOTAL} images in {img_dir}, found {len(img_paths)}")

    order = np.random.default_rng(SPLIT_SEED).permutation(N_TOTAL)
    img_paths = [img_paths[i] for i in order]

    data = np.zeros([N_TOTAL, PATCH, PATCH, PATCH], dtype=np.float32)
    label = np.zeros([N_TOTAL, PATCH, PATCH, PATCH], dtype=np.uint8)

    print(f"Reading {N_TOTAL} volumes from {img_dir}")
    for idx, img_path in enumerate(img_paths):
        stem = os.path.basename(img_path)  # hippocampus_001.nii.gz
        lbl_path = os.path.join(lbl_dir, stem)
        if not os.path.exists(lbl_path):
            raise SystemExit(f"missing label for {stem}: {lbl_path}")

        img = np.asarray(nib.load(img_path).dataobj, dtype=np.float32)
        seg = np.asarray(nib.load(lbl_path).dataobj, dtype=np.uint8)
        if img.shape != seg.shape:
            raise SystemExit(f"{stem}: image shape {img.shape} != label shape {seg.shape}")

        data[idx] = _pad_crop_to_patch(img)
        label[idx] = _pad_crop_to_patch(seg)

        if (idx + 1) % 50 == 0 or idx + 1 == N_TOTAL:
            print(f"  {idx + 1}/{N_TOTAL}")
    print("Reading Task04_Hippocampus finished")

    n_classes = int(label.max()) + 1
    print(f"label values observed: {sorted(np.unique(label).tolist())} ({n_classes} classes)")

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
