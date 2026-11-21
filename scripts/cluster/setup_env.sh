#!/usr/bin/env bash
# One-time environment setup on a SLURM cluster (A100 sm_80, RTX 6000 Ada sm_89).
# Idempotent; rerun safely. Safe on a login node: nothing here touches a GPU
# unless it runs inside a SLURM allocation.
#
#   bash scripts/cluster/setup_env.sh [VENV_DIR]          # default: $REPO/.venv-cluster
#   sbatch scripts/cluster/slurm/gpu_check.sbatch         # then: GPU kernel tests on a compute GPU
#
# In every job afterwards:
#   source VENV_DIR/bin/activate && source scripts/cluster/env.sh <DATA_ROOT>
#
# Choices, and why:
#  * Python 3.11 from PATH, else from conda (`conda info --base`), else $PYTHON.
#    The venv is ISOLATED (no --system-site-packages): layering on a shared conda
#    base whose torch differs lets pip install a second torch that shadows the
#    base one and fails at import (libcudart mismatch).
#  * torch 2.10: the newest torch for which mamba_ssm / causal_conv1d publish
#    prebuilt wheels (scripts/install_mamba.py), so no CUDA compile is needed.
#    cu128 runs on sm_80/sm_89 (and sm_120) with any driver >= 570, incl. CUDA-13 drivers.
#    Override with TORCH_SPEC / TORCH_INDEX.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${1:-$REPO/.venv-cluster}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.10.*}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif command -v python3.11 >/dev/null 2>&1; then
  PY="$(command -v python3.11)"
else
  # conda is often only on PATH after `module load conda` on HPC login nodes
  if ! command -v conda >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load conda >/dev/null 2>&1 || true
  fi
  if command -v conda >/dev/null 2>&1 && [ -x "$(conda info --base)/bin/python3.11" ]; then
    PY="$(conda info --base)/bin/python3.11"
  else
    echo "no python3.11 found: set PYTHON=/path/to/python3.11 (e.g. <conda base>/bin/python3.11)" >&2; exit 1
  fi
fi
echo "python: $PY ($("$PY" --version))   torch: $TORCH_SPEC from $TORCH_INDEX"

if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

python -c "import torch" 2>/dev/null || pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX"
# every later install is constrained to the torch we just chose, so no dependency
# can swap it for a different build
TORCH_PIN="$(python -c 'import torch; print(torch.__version__)')"
CONSTRAINTS="$(mktemp)"
echo "torch==$TORCH_PIN" > "$CONSTRAINTS"
pip install -r "$REPO/requirements-3d.txt" -c "$CONSTRAINTS"

# mamba_ssm + causal_conv1d: prebuilt wheel for this torch, else compile for all archs.
if ! python -c "import mamba_ssm" 2>/dev/null; then
  python "$REPO/scripts/install_mamba.py" || python "$REPO/scripts/build_mamba.py" --all-archs
fi

# the mamba wheels are --no-deps: prove the package actually imports before anything else
python -c "import mamba_ssm, causal_conv1d; print('mamba_ssm', mamba_ssm.__version__, 'imports OK')" || {
  echo "mamba_ssm does not import -- missing dependency? (see requirements-3d.txt)" >&2; exit 1; }

python "$REPO/scripts/cluster/install_trainers.py"

echo
if [ -n "${SLURM_JOB_ID:-}" ] && python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "== GPU kernel + pipeline tests (inside SLURM job $SLURM_JOB_ID) =="
  python -m pytest -q "$REPO/tests/test_pvm3d_network.py" "$REPO/tests/test_mamba_equivalence.py" \
    "$REPO/tests/test_eval3d.py" "$REPO/tests/test_stats3d.py"
else
  echo "== CPU-only tests (not in a SLURM allocation: GPUs here may belong to other users) =="
  CUDA_VISIBLE_DEVICES="" python -m pytest -q "$REPO/tests/test_pvm3d_network.py" "$REPO/tests/test_eval3d.py" \
    "$REPO/tests/test_stats3d.py"
  echo "GPU tests: sbatch $REPO/scripts/cluster/slurm/gpu_check.sbatch"
fi
echo "environment ready: source $VENV/bin/activate && source $REPO/scripts/cluster/env.sh <DATA_ROOT>"
