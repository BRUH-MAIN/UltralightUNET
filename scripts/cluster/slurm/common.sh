# Sourced by every sbatch script here. Expects to be submitted from the repo root
# (sbatch sets SLURM_SUBMIT_DIR) and PVM3D_DATA_ROOT to point at the shared data
# folder (nnU-Net raw/preprocessed/results live under it).
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
: "${PVM3D_DATA_ROOT:?export PVM3D_DATA_ROOT=/path/on/shared/storage before sbatch (it is passed to the job)}"
VENV="${PVM3D_VENV:-$REPO/.venv-cluster}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
# shellcheck disable=SC1091
source "$REPO/scripts/cluster/env.sh" "$PVM3D_DATA_ROOT"
cd "$REPO"
echo "job ${SLURM_JOB_ID:-?} on $(hostname) at $(date -Is); commit $(git rev-parse --short HEAD 2>/dev/null || echo ?)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
