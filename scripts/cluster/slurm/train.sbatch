#!/usr/bin/env bash
# Work through a run file on one GPU; resubmit until `pack_runs.py --status` says all done.
# Several of these may run at once on the same run file (one per GPU): runs are
# claimed with lock files, so no run is ever trained twice concurrently.
#
#   export PVM3D_DATA_ROOT=/dist_home/<user>/.../pvm3d_data
#   sbatch scripts/cluster/slurm/train.sbatch scripts/cluster/runs/S1.txt 4
#   sbatch --time=12:00:00 --nodelist=<node> scripts/cluster/slurm/train.sbatch <runfile> <capacity>
#
# --time: shorter jobs are backfilled sooner on a busy queue, and nothing is lost
# by stopping (runs resume from their last checkpoint, at most 10 epochs old for
# our trainers). 24 h is a reasonable default; the partition cap is 3 days.
#SBATCH --job-name=pvm3d-train
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
source "$SLURM_SUBMIT_DIR/scripts/cluster/slurm/common.sh"
RUNFILE="${1:?usage: sbatch train.sbatch <runfile> [capacity]}"
CAPACITY="${2:-4}"
python scripts/cluster/pack_runs.py "$RUNFILE" --capacity "$CAPACITY" --slurm-auto \
    --gpus "${CUDA_VISIBLE_DEVICES:-0}" --cores "$SLURM_CPUS_PER_TASK"
