"""Download and extract the ISIC skin-lesion datasets used by UltraLight VM-UNet.

The ISIC challenge archives are served as plain objects from a public S3 bucket,
so no account or API token is needed.

    python scripts/download_isic.py --dataset ISIC2017

Downloads resume: a partial .zip is continued with an HTTP Range request rather
than restarted. Extraction is skipped if the destination already holds the
expected number of files, so re-running the script is cheap.

PH2 is not handled here -- it is distributed via Google Drive rather than S3.
"""

import argparse
import os
import shutil
import sys
import zipfile
from urllib.request import Request, urlopen

BASE = "https://isic-challenge-data.s3.amazonaws.com"

# dest_dir -> (url, expected file count after extraction)
DATASETS = {
    "ISIC2017": {
        "root": "data/ISIC2017",
        "parts": [
            ("ISIC2017_Task1-2_Training_Input",
             f"{BASE}/2017/ISIC-2017_Training_Data.zip", 2000, ".jpg"),
            ("ISIC2017_Task1_Training_GroundTruth",
             f"{BASE}/2017/ISIC-2017_Training_Part1_GroundTruth.zip", 2000, ".png"),
        ],
    },
    "ISIC2018": {
        "root": "data/ISIC2018",
        "parts": [
            ("ISIC2018_Task1-2_Training_Input",
             f"{BASE}/2018/ISIC2018_Task1-2_Training_Input.zip", 2594, ".jpg"),
            ("ISIC2018_Task1_Training_GroundTruth",
             f"{BASE}/2018/ISIC2018_Task1_Training_GroundTruth.zip", 2594, ".png"),
        ],
    },
}


def _human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def download(url, path):
    """Fetch `url` to `path`, resuming a partial file if one is present."""
    have = os.path.getsize(path) if os.path.exists(path) else 0

    head = urlopen(Request(url, method="HEAD"), timeout=60)
    total = int(head.headers["Content-Length"])
    head.close()

    if have == total:
        print(f"  already downloaded ({_human(total)})")
        return
    if have > total:
        print("  local file is larger than remote; re-downloading from scratch")
        os.remove(path)
        have = 0

    req = Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"  resuming at {_human(have)} of {_human(total)}")
    else:
        print(f"  downloading {_human(total)}")

    with urlopen(req, timeout=60) as resp, open(path, "ab" if have else "wb") as fh:
        got = have
        last = -1
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

    got = os.path.getsize(path)
    if got != total:
        raise RuntimeError(f"size mismatch for {url}: got {got}, expected {total}")


def extract(zip_path, root, dest_name, expected, ext):
    """Extract `zip_path` under `root`, flattening to `root/dest_name/*`."""
    dest = os.path.join(root, dest_name)
    if os.path.isdir(dest):
        n = len([f for f in os.listdir(dest) if f.endswith(ext)])
        if n == expected:
            print(f"  {dest_name}/ already has {n} {ext} files, skipping extract")
            return
        print(f"  {dest_name}/ has {n} {ext} files (want {expected}), re-extracting")
        shutil.rmtree(dest)

    staging = os.path.join(root, "_staging")
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    print(f"  extracting -> {dest_name}/")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(staging)

    # The archives wrap everything in a single top-level directory whose name
    # does not match what the prep script expects, so pull the payload up.
    entries = [os.path.join(staging, e) for e in os.listdir(staging)]
    if len(entries) == 1 and os.path.isdir(entries[0]):
        payload = entries[0]
    else:
        payload = staging
    os.makedirs(root, exist_ok=True)
    shutil.move(payload, dest)
    if os.path.isdir(staging):
        shutil.rmtree(staging)

    # ISIC ships a superpixel PNG alongside every JPG in the input archives;
    # they are not inputs and would break the 1:1 image/mask pairing.
    removed = 0
    for f in os.listdir(dest):
        if "superpixels" in f or not f.endswith(ext):
            os.remove(os.path.join(dest, f))
            removed += 1
    n = len([f for f in os.listdir(dest) if f.endswith(ext)])
    print(f"  {dest_name}/: {n} {ext} files ({removed} non-payload files removed)")
    if n != expected:
        print(f"  WARNING: expected {expected} files, found {n}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="ISIC2017")
    ap.add_argument("--keep-zips", action="store_true",
                    help="keep the .zip archives after extraction (default: delete)")
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    root = spec["root"]
    zips = os.path.join(root, "_zips")
    os.makedirs(zips, exist_ok=True)

    for dest_name, url, expected, ext in spec["parts"]:
        print(f"{url.rsplit('/', 1)[-1]}")
        zip_path = os.path.join(zips, url.rsplit("/", 1)[-1])
        dest = os.path.join(root, dest_name)
        if os.path.isdir(dest) and len([f for f in os.listdir(dest) if f.endswith(ext)]) == expected:
            print(f"  {dest_name}/ complete, nothing to do")
            continue
        download(url, zip_path)
        extract(zip_path, root, dest_name, expected, ext)
        if not args.keep_zips:
            os.remove(zip_path)

    if not args.keep_zips and os.path.isdir(zips) and not os.listdir(zips):
        os.rmdir(zips)
    print(f"\n{args.dataset} ready under {root}/")
    print(f"Next: python dataprepare/Prepare_{args.dataset}.py")


if __name__ == "__main__":
    main()
