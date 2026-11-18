#!/usr/bin/env bash
# Fetch the external baseline architectures at pinned commits into third_party/
# (gitignored). Nothing is vendored into this repo: SegFormer3D is GPL-3.0 and
# SegMamba ships no license, so their code stays a separately-fetched dependency.
#
#   bash scripts/cluster/fetch_baselines.sh            # -> $REPO/third_party
#   PVM3D_THIRD_PARTY=/scratch/x bash scripts/cluster/fetch_baselines.sh
#
# Commits were the repos' HEADs on 2026-11-13, when the wrappers in
# pvm3d/nnunet_trainers/trainers_external.py were written against them.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${PVM3D_THIRD_PARTY:-$REPO/third_party}"
mkdir -p "$DEST"

fetch() {  # name url sha license
  local name=$1 url=$2 sha=$3
  if [ -d "$DEST/$name/.git" ] && [ "$(git -C "$DEST/$name" rev-parse HEAD)" = "$sha" ]; then
    echo "$name @ ${sha:0:10} already present"; return
  fi
  rm -rf "${DEST:?}/$name"
  git clone -q --filter=blob:limit=1m "$url" "$DEST/$name"
  git -C "$DEST/$name" -c advice.detachedHead=false checkout -q "$sha"
  echo "$name @ ${sha:0:10} ($4)"
}

fetch LightM-UNet https://github.com/MrBlankness/LightM-UNet.git 38148ef8ac408014be0d668d9a98e2e97a68ed99 "Apache-2.0"
fetch Slim-UNETR  https://github.com/deepang-ai/Slim-UNETR.git   ae0391d16dd4cd05bf4b05ffb5fba162321d77d4 "MIT"
fetch SegFormer3D https://github.com/OSUPCVLab/SegFormer3D.git   e314242f14b6731458130809945a0ee27f4298bd "GPL-3.0"
fetch SegMamba    https://github.com/ge-xing/SegMamba.git        cff35970e0c542ad940b5701267f1ac888298b06 "no license file"

cat <<'MSG'

LightM-UNet needs MONAI:   pip install "monai>=1.3"
SegMamba needs ITS OWN VENV: its bundled mamba fork (bimamba_type="v3", nslices)
installs under the package name `mamba_ssm` and would replace the kernel PVM-3D is
tested against. Build it separately:
    python3.11 -m venv $REPO/.venv-segmamba && source $REPO/.venv-segmamba/bin/activate
    (install torch, requirements.txt, requirements-3d.txt, monai as in setup_env.sh)
    pip install --no-build-isolation $DEST/SegMamba/causal-conv1d $DEST/SegMamba/mamba
    python scripts/cluster/install_trainers.py
and give its runs `venv=$REPO/.venv-segmamba` in the pack_runs run file.
MSG
