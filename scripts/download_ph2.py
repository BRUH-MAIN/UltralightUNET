"""Download and extract the PH2 dataset (dermoscopic images + lesion masks).

PH2 (Mendonca et al., Universidade do Porto) is not served from a plain URL
the way the ISIC challenge archives are -- upstream distribution is a Google
Drive link behind a request form. This uses the public Kaggle mirror instead:
https://www.kaggle.com/datasets/spacesurfer/ph2-dataset

    python scripts/download_ph2.py

Requires a Kaggle API token at ~/.kaggle/kaggle.json (kaggle.com/settings ->
Create New Token) or the KAGGLE_USERNAME/KAGGLE_KEY environment variables --
same as scripts/hf_data.py's HF_TOKEN convention, but for Kaggle's API.

Unlike download_isic.py/download_ham10000.py, this does not resume partial
downloads (the `kaggle` package's dataset_download_files doesn't expose HTTP
Range the way this repo's own urlopen-based downloader does) -- the zip is
~212 MB, small enough that a clean retry is cheap if one is interrupted.

The Kaggle mirror preserves the dataset's native layout, one directory per
image:

    PH2Dataset/PH2 Dataset images/IMD002/IMD002_Dermoscopic_Image/IMD002.bmp
    PH2Dataset/PH2 Dataset images/IMD002/IMD002_lesion/IMD002_lesion.bmp
    PH2Dataset/PH2 Dataset images/IMD002/IMD002_roi/...              (unused)

That layout is left as-is rather than flattened -- dataprepare/Prepare_PH2.py
walks it directly -- since the per-image dermoscopic/lesion filenames don't
collide the way flattening would require care for.
"""

import argparse
import os
import zipfile

DATASET = "spacesurfer/ph2-dataset"
ROOT = "data/PH2"
N_EXPECTED = 200


def _human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def _count_images(root):
    img_root = os.path.join(root, "PH2Dataset", "PH2 Dataset images")
    if not os.path.isdir(img_root):
        return 0
    return len([d for d in os.listdir(img_root)
                if os.path.isdir(os.path.join(img_root, d)) and d.startswith("IMD")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-zip", action="store_true",
                    help="keep the downloaded .zip after extraction (default: delete)")
    args = ap.parse_args()

    have = _count_images(ROOT)
    if have >= N_EXPECTED:
        print(f"{ROOT}/ already has {have} IMD folders, nothing to do")
        return

    from kaggle.api.kaggle_api_extended import KaggleApi

    os.makedirs(ROOT, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {DATASET} from Kaggle...")
    api.dataset_download_files(DATASET, path=ROOT, unzip=False, quiet=False)

    zip_path = os.path.join(ROOT, "ph2-dataset.zip")
    if not os.path.exists(zip_path):
        # kaggle names the zip after the dataset slug, not always "ph2-dataset.zip"
        zips = [f for f in os.listdir(ROOT) if f.endswith(".zip")]
        if not zips:
            raise SystemExit(f"expected a .zip in {ROOT}/ after download, found none")
        zip_path = os.path.join(ROOT, zips[0])

    print(f"  extracting {_human(os.path.getsize(zip_path))}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(ROOT)

    if not args.keep_zip:
        os.remove(zip_path)

    n = _count_images(ROOT)
    print(f"\nPH2 ready under {ROOT}/: {n} IMD folders")
    if n != N_EXPECTED:
        print(f"WARNING: expected {N_EXPECTED} IMD folders, found {n}")
    print("Next: python dataprepare/Prepare_PH2.py")


if __name__ == "__main__":
    main()
