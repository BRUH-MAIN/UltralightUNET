#!/usr/bin/env bash
# BraTS 2021 Task 1 training data (1,251 cases) from its Kaggle mirror, instead of
# waiting for Synapse access. Data only -- nothing downloaded is executed.
#
#   bash scripts/cluster/download_brats21_kaggle.sh <DEST> [owner/dataset]
#   -> <DEST>/BraTS2021_Training_Data/BraTS2021_XXXXX/{..._t1,_t1ce,_t2,_flair,_seg}.nii.gz
#   then: python scripts/cluster/convert_datasets.py brats21 --src <DEST>/BraTS2021_Training_Data
#
# Why the mirror is acceptable: the BraTS 2021 training set is released under
# CC BY 4.0 (TCIA collection RSNA-ASNR-MICCAI-BraTS-2021, DOI 10.7937/jc8x-9874),
# which permits redistribution with attribution. The paper must still cite the
# BraTS 2021 papers (Baid et al. 2021 arXiv:2107.02314; Menze et al. 2015 TMI;
# Bakas et al. 2017 Sci Data) and the TCIA DOI. Integrity is checked below: the
# case count must be 1,251 with all five files per case, and nnU-Net's converter
# rejects any label outside {0,1,2,4}.
#
# BRATS_LOCAL_ZIP=/path/to/archive.zip skips the download (zip fetched some other way).
# BRATS_CHECK_ONLY=1 only runs the integrity check on <DEST>/BraTS2021_Training_Data.
#
# Credentials: a Kaggle API token, set up by the account owner on this machine --
# never pasted into a chat:  ~/.kaggle/kaggle.json (chmod 600)  or
# KAGGLE_USERNAME / KAGGLE_KEY in the environment. The key is passed to curl on
# stdin, not on the command line, so it does not show up in `ps` on a shared node.
set -euo pipefail
DEST="${1:?usage: download_brats21_kaggle.sh <DEST> [owner/dataset]}"
SLUG="${2:-dschettler8845/brats-2021-task1}"
OUT="$DEST/BraTS2021_Training_Data"
mkdir -p "$DEST"
cd "$DEST"

if [ -n "${BRATS_CHECK_ONLY:-}" ] || { [ -d "$OUT" ] && [ "$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name 'BraTS2021_*' | wc -l)" -ge 1251 ]; }; then
  echo "$OUT: checking existing cases (no download)"
else
  if [ -n "${BRATS_LOCAL_ZIP:-}" ]; then   # zip already downloaded (e.g. via the browser)
    cp "$BRATS_LOCAL_ZIP" brats21_kaggle.zip
  else
  if [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
    [ -r "$HOME/.kaggle/kaggle.json" ] || { echo "no Kaggle token: create ~/.kaggle/kaggle.json (chmod 600) or export KAGGLE_USERNAME/KAGGLE_KEY" >&2; exit 1; }
    KAGGLE_USERNAME="$(python3 -I -c 'import json,os;print(json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["username"])')"
    KAGGLE_KEY="$(python3 -I -c 'import json,os;print(json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["key"])')"
  fi
  echo "== downloading kaggle:$SLUG (about 12-13 GB)"
  # -K - : curl reads the credentials from stdin; credentials are not resent on the
  # redirect to the storage host (no --location-trusted)
  printf 'user = "%s:%s"\n' "$KAGGLE_USERNAME" "$KAGGLE_KEY" | \
    curl -fL --retry 5 --retry-delay 10 -C - -K - -o brats21_kaggle.zip \
      "https://www.kaggle.com/api/v1/datasets/download/$SLUG"
  unset KAGGLE_KEY
  fi

  rm -rf _extract_brats && mkdir _extract_brats
  if command -v unzip >/dev/null; then unzip -q brats21_kaggle.zip -d _extract_brats/zip
  else python3 -I -m zipfile -e brats21_kaggle.zip _extract_brats/zip; fi   # zipfile refuses ../ and absolute paths
  # The zip holds the main BraTS2021_Training_Data.tar plus standalone per-case tars
  # (e.g. BraTS2021_00495.tar). Each tar is extracted into its own folder; the main
  # archive's cases are placed first, then standalone ones replace them (assumed to
  # be the mirror's corrections) -- any case where the two differ is listed.
  find _extract_brats/zip -name '*.tar' -print0 | while IFS= read -r -d '' t; do
    name="$(basename "$t" .tar)"
    echo "   extracting $name.tar"
    mkdir -p "_extract_brats/tars/$name"
    tar --no-same-owner -xf "$t" -C "_extract_brats/tars/$name" && rm -f "$t"
  done
  mkdir -p "$OUT"
  place() {  # $1 = tar folder, $2 = 1 if standalone (may replace)
    # -mindepth 1: a standalone tar's own folder is also named BraTS2021_XXXXX; -prune: the
    # matched case folder is moved, so find must not descend into it
    find "$1" -mindepth 1 -type d -regex '.*/BraTS2021_[0-9][0-9][0-9][0-9][0-9]' -prune -print0 | \
        while IFS= read -r -d '' d; do
      b="$(basename "$d")"
      if [ -e "$OUT/$b" ]; then
        [ "$2" = 1 ] || continue
        if ! diff -rq "$d" "$OUT/$b" >/dev/null 2>&1; then
          echo "   WARNING: $b differs between the main archive and its standalone tar; using the standalone copy"
        fi
        rm -rf "${OUT:?}/$b"
      fi
      mv "$d" "$OUT/$b"
    done
  }
  for t in _extract_brats/tars/*; do
    case "$(basename "$t")" in BraTS2021_[0-9][0-9][0-9][0-9][0-9]) ;; *) place "$t" 0 ;; esac
  done
  for t in _extract_brats/tars/BraTS2021_[0-9][0-9][0-9][0-9][0-9]; do
    [ -d "$t" ] && place "$t" 1
  done
  rm -rf _extract_brats brats21_kaggle.zip
fi

# integrity: count + five files per case
python3 -I - "$OUT" <<'EOF'
import os, re, sys
root = sys.argv[1]
cases = sorted(d for d in os.listdir(root) if re.fullmatch(r'BraTS2021_\d{5}', d))
need = ('t1', 't1ce', 't2', 'flair', 'seg')
bad = [c for c in cases if not all(os.path.isfile(os.path.join(root, c, f'{c}_{m}.nii.gz')) for m in need)]
print(f'{len(cases)} cases in {root}; {len(bad)} incomplete' + (f': {bad[:10]}' if bad else ''))
sys.exit(0 if len(cases) == 1251 and not bad else 'expected 1251 complete cases (official BraTS 2021 training set)')
EOF
echo "next: python scripts/cluster/convert_datasets.py brats21 --src $OUT"
