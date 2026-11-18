#!/usr/bin/env bash
# GPU kernel + pipeline tests on a real compute GPU, after setup_env.sh.
# Submit once per GPU type: A100 (sm_80) and RTX 6000 Ada (sm_89), e.g.
#   sbatch --nodelist=<a100-node> scripts/cluster/slurm/gpu_check.sbatch
#SBATCH --job-name=pvm3d-gpucheck
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
export PVM3D_DATA_ROOT="${PVM3D_DATA_ROOT:-$SLURM_SUBMIT_DIR/data/nnunet}"
source "$SLURM_SUBMIT_DIR/scripts/cluster/slurm/common.sh"
python -c "import torch, mamba_ssm; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0), 'mamba_ssm', mamba_ssm.__version__)"
python -m pytest -q tests/test_pvm3d_network.py tests/test_mamba_equivalence.py tests/test_eval3d.py \
    tests/test_stats3d.py tests/test_pack_runs.py
