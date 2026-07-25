"""Download the MILK10k classification dataset from the ISIC S3 bucket.

MILK10k is a skin-lesion *diagnosis* benchmark: 5,240 lesions, each with a
paired clinical close-up and dermatoscopic image (10,480 images total), 11
diagnosis classes, scored by macro-F1. CC-BY-NC, no registration gate.

    python milk10k/scripts/download_milk10k.py            # training data + CSVs
    python milk10k/scripts/download_milk10k.py --test     # also the held-out test images

The training GroundTruth gives one label per lesion; the test GroundTruth is
withheld (leaderboard only), so all reportable numbers come from splitting the
training lesions -- see dataprepare/prepare_milk10k.py.

Mirrors the resume/skip logic of the Phase-1 scripts/download_isic.py. URLs
verified live (HEAD 200) on 2026-07-25.
"""

import argparse
import os
import sys
import zipfile
from urllib.request import Request, urlopen

BASE = "https://isic-archive.s3.amazonaws.com/challenges/milk10k"

TRAIN_FILES = [
    ("MILK10k_Training_Input.zip", 10480, ".jpg"),   # 314 MB
    ("MILK10k_Training_GroundTruth.csv", None, None),
    ("MILK10k_Training_Metadata.csv", None, None),
    ("MILK10k_Training_Supplement.csv", None, None),
]
TEST_FILES = [
    ("MILK10k_Test_Input.zip", 958, ".jpg"),          # 29 MB
    ("MILK10k_Test_Metadata.csv", None, None),
]


def _human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def download(url, path):
    """Fetch url to path, resuming a partial file via HTTP Range."""
    have = os.path.getsize(path) if os.path.exists(path) else 0
    head = urlopen(Request(url, method="HEAD"), timeout=60)
    total = int(head.headers["Content-Length"])
    head.close()

    if have == total:
        print(f"  already downloaded ({_human(total)})")
        return
    if have > total:
        os.remove(path)
        have = 0

    req = Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"  resuming at {_human(have)} of {_human(total)}")
    else:
        print(f"  downloading {_human(total)}")

    with urlopen(req, timeout=60) as resp, open(path, "ab" if have else "wb") as fh:
        got, last = have, -1
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            pct = int(got * 100 / total)
            if pct != last:
                print(f"\r  {pct:3d}%  {_human(got)} / {_human(total)}", end="", flush=True)
                last = pct
    print()
    if os.path.getsize(path) != total:
        raise RuntimeError(f"size mismatch for {url}")


def extract(zip_path, root, dest_name, expected, ext):
    dest = os.path.join(root, dest_name)
    if os.path.isdir(dest):
        n = len([f for f in os.listdir(dest) if f.endswith(ext)])
        if n == expected:
            print(f"  {dest_name}/ already has {n} {ext} files, skipping extract")
            return
    print(f"  extracting -> {dest_name}/")
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        # the archive stores images at the top level; flatten just in case
        for m in zf.infolist():
            if m.is_dir():
                continue
            name = os.path.basename(m.filename)
            if not name.endswith(ext):
                continue
            with zf.open(m) as src, open(os.path.join(dest, name), "wb") as out:
                out.write(src.read())
    n = len([f for f in os.listdir(dest) if f.endswith(ext)])
    print(f"  {dest_name}/: {n} {ext} files")
    if expected and n != expected:
        print(f"  WARNING: expected {expected}, found {n}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="milk10k/data")
    ap.add_argument("--test", action="store_true", help="also fetch the held-out test images")
    ap.add_argument("--keep-zips", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    files = TRAIN_FILES + (TEST_FILES if args.test else [])
    for name, expected, ext in files:
        print(name)
        path = os.path.join(args.root, name)
        download(f"{BASE}/{name}", path)
        if name.endswith(".zip"):
            dest = name.replace(".zip", "").replace("_Input", "_Images")
            extract(path, args.root, dest, expected, ext)
            if not args.keep_zips:
                os.remove(path)

    print(f"\nMILK10k ready under {args.root}/")
    print("Next: python milk10k/dataprepare/prepare_milk10k.py")


if __name__ == "__main__":
    main()
