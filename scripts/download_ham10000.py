"""Download and extract HAM10000 (images + lesion segmentation masks).

HAM10000 itself (Harvard Dataverse, doi:10.7910/DVN/DBW86T) is a 7-class
diagnosis *classification* dataset with no masks. The segmentation masks used
here -- "HAM10000_segmentations_lesion_tschandl.zip", binary lesion masks
curated by a single dermatologist -- are hosted in the *same* Dataverse record,
so no separate Kaggle account/token is needed despite that same file also
being mirrored on Kaggle as tschandl/ham10000-lesion-segmentations.

    python scripts/download_ham10000.py

Downloads resume (HTTP Range) and extraction is skipped if the destination
already holds files, same conventions as scripts/download_isic.py.

Unlike the ISIC challenge archives, this is not a fixed-size challenge
release, so counts are not asserted against a hardcoded constant -- the script
just reports what it finds. The mask archive holds 10,018 files against
10,015 official HAM10000 images; dataprepare/Prepare_HAM10000.py resolves the
mismatch by pairing on filename stem rather than assuming a 1:1 count.
"""

import argparse
import os
import shutil
import sys
import zipfile
from urllib.request import Request, urlopen

BASE = "https://dataverse.harvard.edu/api/access/datafile"
ROOT = "data/HAM10000"

# Dataverse's front end 403s the default "Python-urllib/x.y" User-Agent (a WAF
# rule, not an auth check -- the files are public); any normal-looking UA passes.
HEADERS = {"User-Agent": "Mozilla/5.0"}

# (dest_name, file id, ext, min expected file count -- a floor, not an exact match)
PARTS = [
    ("HAM10000_images_part_1", 3172585, ".jpg", 4000),
    ("HAM10000_images_part_2", 3172584, ".jpg", 4000),
    ("HAM10000_segmentations_lesion_tschandl", 3838943, ".png", 9000),
]


def _human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def download(url, path):
    """Fetch `url` to `path`, resuming a partial file if one is present."""
    have = os.path.getsize(path) if os.path.exists(path) else 0

    head = urlopen(Request(url, method="HEAD", headers=HEADERS), timeout=60)
    total = int(head.headers["Content-Length"])
    head.close()

    if have == total:
        print(f"  already downloaded ({_human(total)})")
        return
    if have > total:
        print("  local file is larger than remote; re-downloading from scratch")
        os.remove(path)
        have = 0

    req = Request(url, headers=HEADERS)
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


def extract(zip_path, root, dest_name, ext, min_expected):
    """Extract `zip_path` under `root`, flattening to `root/dest_name/*`."""
    dest = os.path.join(root, dest_name)
    if os.path.isdir(dest):
        n = len([f for f in os.listdir(dest) if f.endswith(ext)])
        if n >= min_expected:
            print(f"  {dest_name}/ already has {n} {ext} files, skipping extract")
            return
        print(f"  {dest_name}/ has {n} {ext} files (want >= {min_expected}), re-extracting")
        shutil.rmtree(dest)

    staging = os.path.join(root, "_staging")
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    print(f"  extracting -> {dest_name}/")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(staging)

    # The Dataverse zips wrap everything in a single top-level directory whose
    # name does not match what the prep script expects, so pull the payload up.
    # __MACOSX/ (AppleDouble metadata some of these zips carry) is ignored when
    # deciding whether there's a single wrapping directory, so its presence
    # doesn't force payload = staging (which would leave the real files nested
    # one level too deep, and __MACOSX sitting alongside them for the ext-only
    # cleanup below to trip over).
    entries = [os.path.join(staging, e) for e in os.listdir(staging) if e != "__MACOSX"]
    if len(entries) == 1 and os.path.isdir(entries[0]):
        payload = entries[0]
    else:
        payload = staging
    os.makedirs(root, exist_ok=True)
    shutil.move(payload, dest)
    if os.path.isdir(staging):
        shutil.rmtree(staging)

    removed = 0
    for f in os.listdir(dest):
        if not f.endswith(ext):
            p = os.path.join(dest, f)
            # zip archives created on macOS carry a __MACOSX/ metadata directory
            # alongside the payload; strip directories too, not just files.
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
            removed += 1
    n = len([f for f in os.listdir(dest) if f.endswith(ext)])
    print(f"  {dest_name}/: {n} {ext} files ({removed} non-payload files removed)")
    if n < min_expected:
        print(f"  WARNING: expected at least {min_expected} files, found {n}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-zips", action="store_true",
                    help="keep the .zip archives after extraction (default: delete)")
    args = ap.parse_args()

    zips = os.path.join(ROOT, "_zips")
    os.makedirs(zips, exist_ok=True)

    for dest_name, file_id, ext, min_expected in PARTS:
        url = f"{BASE}/{file_id}"
        print(f"{dest_name} (Dataverse file {file_id})")
        zip_path = os.path.join(zips, f"{dest_name}.zip")
        dest = os.path.join(ROOT, dest_name)
        if os.path.isdir(dest) and len([f for f in os.listdir(dest) if f.endswith(ext)]) >= min_expected:
            print(f"  {dest_name}/ complete, nothing to do")
            continue
        download(url, zip_path)
        extract(zip_path, ROOT, dest_name, ext, min_expected)
        if not args.keep_zips:
            os.remove(zip_path)

    if not args.keep_zips and os.path.isdir(zips) and not os.listdir(zips):
        os.rmdir(zips)

    # Flatten the two image parts into one directory, same as Prepare_HAM10000.py expects.
    combined = os.path.join(ROOT, "HAM10000_images")
    os.makedirs(combined, exist_ok=True)
    moved = 0
    for part_name, _, ext, _ in PARTS[:2]:
        part_dir = os.path.join(ROOT, part_name)
        if not os.path.isdir(part_dir):
            continue
        for f in os.listdir(part_dir):
            src = os.path.join(part_dir, f)
            dst = os.path.join(combined, f)
            if not os.path.exists(dst):
                shutil.move(src, dst)
                moved += 1
        if not os.listdir(part_dir):
            os.rmdir(part_dir)
    n_combined = len([f for f in os.listdir(combined) if f.endswith(".jpg")])
    print(f"\nHAM10000_images/: {n_combined} .jpg files ({moved} moved in this run)")

    print(f"\nHAM10000 ready under {ROOT}/")
    print("Next: python dataprepare/Prepare_HAM10000.py")


if __name__ == "__main__":
    main()
