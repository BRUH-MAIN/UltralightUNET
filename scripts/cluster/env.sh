# Environment for every PVM-3D nnU-Net command (laptop and cluster).
#
#   source scripts/cluster/env.sh                 # data under $REPO/data/nnunet
#   source scripts/cluster/env.sh /scratch/$USER/pvm3d   # cluster scratch
#   PVM3D_DATA_ROOT=/scratch/... source scripts/cluster/env.sh
#
# nnUNet_extTrainer is what makes `-tr nnUNetTrainerPVM3D_*` resolvable: nnU-Net
# v2.8+ searches that folder when a trainer is not in its own package. Only
# trainer modules live there (pvm3d/nnunet_trainers/); see pvm3d/nnunet_glue.py.

_PVM3D_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_PVM3D_ROOT="${1:-${PVM3D_DATA_ROOT:-$_PVM3D_REPO/data/nnunet}}"

export PVM3D_REPO="$_PVM3D_REPO"
export nnUNet_raw="$_PVM3D_ROOT/raw"
export nnUNet_preprocessed="$_PVM3D_ROOT/preprocessed"
export nnUNet_results="$_PVM3D_ROOT/results"
export nnUNet_extTrainer="$_PVM3D_REPO/pvm3d/nnunet_trainers"
case ":${PYTHONPATH:-}:" in *":$_PVM3D_REPO:"*) ;; *) export PYTHONPATH="$_PVM3D_REPO${PYTHONPATH:+:$PYTHONPATH}";; esac

# data-augmentation workers per training run. pack_runs.py overrides this per
# slot (cores / concurrent runs); this is the single-run default.
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-8}"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
echo "PVM-3D env: raw=$nnUNet_raw preprocessed=$nnUNet_preprocessed results=$nnUNet_results"
unset _PVM3D_REPO _PVM3D_ROOT
