#!/usr/bin/env bash
# Download the benchmark datasets that need no registration and no third-party
# code: MSD Task04_Hippocampus, MSD Task03_Liver (= LiTS training set) and AMOS22.
# Data only -- nothing downloaded here is executed. Resumable (curl -C -), and every
# archive is verified against a published checksum before it is extracted.
#
#   bash scripts/cluster/download_open_data.sh <DEST>                 # e.g. $PVM3D_DATA_ROOT/downloads
#   DATASETS="Task03_Liver" bash scripts/cluster/download_open_data.sh <DEST>   # a subset
#   MSD_SOURCE=s3 bash scripts/cluster/download_open_data.sh <DEST>   # official bucket instead of the mirror
#
# Sources:
#   MSD   Hugging Face mirror Novel-BioMedAI/Medical_Segmentation_Decathlon by default.
#         It is byte-identical to the official msd-for-monai S3 bucket: verified
#         2026-11-21 for Task04_Hippocampus (same SHA-256 from both hosts), and
#         Task03_Liver has the same size on both. Because the bytes are identical, a
#         partial .tar from either host resumes from the other; the checks below prove it.
#         Network note (cluster login node, 2026-11-21): every host was throttled PER
#         CONNECTION to ~0.7-1.6 MB/s (S3, Hugging Face, Zenodo). For the 29 GB Liver tar,
#         8 parallel `curl -r` byte ranges from the HF URL appended to the partial file
#         took ~21 min, and the result passed both the SHA-256 and the official S3 ETag.
#         Zenodo did not speed up with parallel ranges, so AMOS runs as one stream (~4 h).
#   AMOS  Zenodo record 7262581 (the link nnU-Net's converter names), MD5 from Zenodo's API.
#
# NOT handled here (need a person):
#   KiTS23     official downloader is code from github.com/neheller/kits23 (pip install + kits23_download_data)
#   ACDC       CREATIS HumanHeart portal (public per the challenge page)
#   BraTS2021  Kaggle mirror (CC BY 4.0): scripts/cluster/download_brats21_kaggle.sh -- needs a Kaggle API token
#   BTCV       Synapse syn3193805 (account + registration)
set -euo pipefail
DEST="${1:?usage: download_open_data.sh <DEST>}"
DATASETS="${DATASETS:-Task04_Hippocampus Task03_Liver amos22}"
MSD_SOURCE="${MSD_SOURCE:-hf}"
mkdir -p "$DEST"
cd "$DEST"

# Two independent identities per MSD tar, both checked in one read:
#  * SHA-256 = the Hugging Face mirror's LFS object id;
#  * the official msd-for-monai S3 object's multipart ETag (MD5 of the 8 MiB part MD5s,
#    "-<parts>"), so a file fetched from the mirror is also proven byte-identical to the
#    official bucket. Method validated 2026-11-21: recomputing Task04_Hippocampus's ETag
#    from the HF-verified file reproduces S3's value exactly.
declare -A MSD_SHA256=(
  [Task04_Hippocampus]=282d808a3e84e5a52f090d9dd4c0b0057b94a6bd51ad41569aef5ff303287771
  [Task03_Liver]=4007d9db1acda850d57a6ceb2b3998b7a0d43f8ad5a3f740dc38bc0cb8b7a2c5
)
declare -A MSD_S3_ETAG=(
  [Task04_Hippocampus]=81aa748a1df6d95ecb6f1b5812f132b1-4
  [Task03_Liver]=288b95e0d9ead5215f43f2aa9d786f2e-3449
)
case "$MSD_SOURCE" in
  hf) MSD_URL=https://huggingface.co/datasets/Novel-BioMedAI/Medical_Segmentation_Decathlon/resolve/main ;;
  s3) MSD_URL=https://msd-for-monai.s3-us-west-2.amazonaws.com ;;
  *) echo "MSD_SOURCE must be hf or s3" >&2; exit 2 ;;
esac

verify_msd() {  # file sha256 s3-etag -- one pass over the file
  echo "   verifying $1 (SHA-256 vs HF mirror, multipart ETag vs official S3) ..."
  if ! python3 -I - "$1" "$2" "$3" <<'EOF'
import hashlib, sys
path, want_sha, want_etag = sys.argv[1:4]
part = 8 * 1024 * 1024
sha, parts = hashlib.sha256(), []
with open(path, 'rb') as f:
    while chunk := f.read(part):
        sha.update(chunk)
        parts.append(hashlib.md5(chunk).digest())
etag = f'{hashlib.md5(b"".join(parts)).hexdigest()}-{len(parts)}'
ok_sha, ok_etag = sha.hexdigest() == want_sha, etag == want_etag
print(f'   sha256 {"OK" if ok_sha else "MISMATCH " + sha.hexdigest()}; '
      f'official S3 ETag {"OK" if ok_etag else "MISMATCH " + etag}')
sys.exit(0 if ok_sha and ok_etag else 1)
EOF
  then
    echo "   CHECKSUM MISMATCH for $1 -- deleting it; rerun to download again" >&2
    rm -f "$1"
    exit 1
  fi
}

fetch() {  # url outfile
  echo "== $2  <- $1"
  curl -fL --progress-bar --retry 5 --retry-delay 10 -C - -o "$2" "$1"
}

verify() {  # file algo expected
  echo "   verifying $2 of $1 ..."
  local got
  got="$("${2}sum" "$1" | cut -d' ' -f1)"
  if [ "$got" != "$3" ]; then
    echo "   CHECKSUM MISMATCH for $1: got $got, expected $3 -- deleting it; rerun to download again" >&2
    rm -f "$1"
    exit 1
  fi
  echo "   $2 OK"
}

for t in Task04_Hippocampus Task03_Liver; do
  case " $DATASETS " in *" $t "*) ;; *) continue ;; esac
  if [ -d "$t" ]; then
    echo "$t/ already extracted"
    rm -rf "_extract_$t" "$t.tar"   # leftovers of an earlier run of this script
    continue
  fi
  fetch "$MSD_URL/$t.tar" "$t.tar"
  verify_msd "$t.tar" "${MSD_SHA256[$t]}" "${MSD_S3_ETAG[$t]}"
  rm -rf "_extract_$t" && mkdir "_extract_$t"
  tar --no-same-owner -xf "$t.tar" -C "_extract_$t"
  mv "_extract_$t/$t" "$t"
  # the tars also carry macOS AppleDouble "._*" sidecars (top level and per file); not volumes
  rm -rf "_extract_$t" "$t.tar"
  find "$t" -name '._*' -delete
  echo "   $t: $(ls "$t/imagesTr" | wc -l) images, $(ls "$t/labelsTr" | wc -l) labels"
done

# AMOS22: file list + MD5 from the Zenodo API (record 7262581)
if [[ " $DATASETS " == *" amos22 "* ]] && [ ! -d amos22 ]; then
  curl -fsSL https://zenodo.org/api/records/7262581 -o amos_record.json
  python3 -I -c '
import json
rec = json.load(open("amos_record.json"))
for f in rec.get("files", []):
    if f["key"].lower().startswith("amos22") and f["key"].endswith(".zip"):
        md5 = f.get("checksum", "").split(":", 1)[-1] if f.get("checksum", "").startswith("md5:") else "-"
        print(f["key"], f["links"]["self"], f.get("size", 0), md5)
' > amos_files.txt
  cat amos_files.txt
  [ -s amos_files.txt ] || { echo "no amos22*.zip in Zenodo record 7262581 -- check https://zenodo.org/records/7262581"; exit 1; }
  while read -r key url size md5; do
    fetch "$url" "$key"
    if [ "$md5" != "-" ]; then verify "$key" md5 "$md5"; else echo "   (Zenodo listed no MD5 for $key)"; fi
  done < amos_files.txt
  rm -rf _extract_amos && mkdir _extract_amos
  if command -v unzip >/dev/null; then unzip -q -o amos22*.zip -d _extract_amos
  else python3 -I -m zipfile -e amos22*.zip _extract_amos; fi
  # the converter wants the folder holding imagesTr/ labelsTr/ imagesVa/ ...
  inner="$(dirname "$(find _extract_amos -maxdepth 3 -type d -name imagesTr | head -1)")"
  mv "$inner" amos22 && rm -rf _extract_amos amos22*.zip amos_record.json amos_files.txt
fi
echo
du -sh "$DEST"/* 2>/dev/null
echo "next: python scripts/cluster/convert_datasets.py msd --src $DEST/Task04_Hippocampus  (and Task03_Liver; amos --src $DEST/amos22)"
