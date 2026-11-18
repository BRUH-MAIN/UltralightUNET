#!/usr/bin/env bash
# CPU job: nnU-Net planning + preprocessing (+ verify) for converted datasets.
#   PVM3D_DATA_ROOT=... sbatch scripts/cluster/slurm/preprocess.sbatch 4 27 220
# No GPU is requested: nnU-Net's planner does not need one.
#SBATCH --job-name=pvm3d-preprocess
#SBATCH --partition=workq
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
source "$SLURM_SUBMIT_DIR/scripts/cluster/slurm/common.sh"
[ $# -gt 0 ] || { echo "usage: sbatch preprocess.sbatch <dataset ids...>"; exit 2; }
python scripts/cluster/convert_datasets.py plan -d "$@" -np "$SLURM_CPUS_PER_TASK"
python scripts/cluster/make_plans_variant.py -d "$@" --divisible 32
python scripts/cluster/convert_datasets.py verify || true   # reports every dataset; not-yet-converted ones are listed
