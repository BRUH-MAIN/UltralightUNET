#!/usr/bin/env bash
# Download the benchmark datasets that need no registration and no third-party
# code: MSD Task04_Hippocampus, MSD Task03_Liver (= LiTS training set) and AMOS22.
# Data only -- nothing downloaded here is executed. Resumable (curl -C -).
#
#   bash scripts/cluster/download_open_data.sh <DEST>      # e.g. $PVM3D_DATA_ROOT/downloads
#
# Run on the login node (compute-node egress is untested) inside tmux/screen:
# ~70 GB at the measured ~15 MB/s is roughly 1.5 h.
#
# NOT handled here (need a person):
#   KiTS23     official downloader is code from github.com/neheller/kits23 (pip install + kits23_download_data)
#   ACDC       registration at the CREATIS challenge site
#   BraTS2021  Synapse syn25829067 (account + data-use terms)
#   BTCV       Synapse syn3193805 (account + registration)
set -euo pipefail
DEST="${1:?usage: download_open_data.sh <DEST>}"
mkdir -p "$DEST"
cd "$DEST"

fetch() {  # url outfile
  echo "== $2"
  curl -fL --retry 5 --retry-delay 10 -C - -o "$2" "$1"
}

MSD=https://msd-for-monai.s3-us-west-2.amazonaws.com
for t in Task04_Hippocampus Task03_Liver; do
  if [ -d "$t" ]; then echo "$t/ already extracted"; continue; fi
  fetch "$MSD/$t.tar" "$t.tar"
  mkdir -p "_extract_$t" && tar --no-same-owner -xf "$t.tar" -C "_extract_$t"
  mv "_extract_$t/$t" "$t" && rmdir "_extract_$t" && rm -f "$t.tar"
  # MSD tars carry macOS AppleDouble "._*" sidecars; they are not volumes
  find "$t" -name '._*' -delete
done

# AMOS22: file list from the Zenodo API (record 7262581, the link nnU-Net's converter names)
if [ ! -d amos22 ]; then
  curl -fsSL https://zenodo.org/api/records/7262581 -o amos_record.json
  python3 -I -c '
import json, sys
rec = json.load(open("amos_record.json"))
for f in rec.get("files", []):
    if f["key"].lower().startswith("amos22") and f["key"].endswith(".zip"):
        print(f["key"], f["links"]["self"], f.get("size", 0))
' > amos_files.txt
  cat amos_files.txt
  [ -s amos_files.txt ] || { echo "no amos22*.zip in Zenodo record 7262581 -- check https://zenodo.org/records/7262581"; exit 1; }
  while read -r key url size; do fetch "$url" "$key"; done < amos_files.txt
  mkdir -p _extract_amos && unzip -q -o amos22*.zip -d _extract_amos
  # the converter wants the folder holding imagesTr/ labelsTr/ imagesVa/ ...
  inner="$(dirname "$(find _extract_amos -maxdepth 3 -type d -name imagesTr | head -1)")"
  mv "$inner" amos22 && rm -rf _extract_amos amos22*.zip amos_record.json amos_files.txt
fi
echo
du -sh "$DEST"/* 2>/dev/null
echo "next: python scripts/cluster/convert_datasets.py msd --src $DEST/Task04_Hippocampus  (and Task03_Liver; amos --src $DEST/amos22)"
