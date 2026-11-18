# PVM-3D cluster runbook (SLURM)

Commands for the experiments in [docs/3d_journal_roadmap.md](../../docs/3d_journal_roadmap.md), in the order you
run them.

How this cluster works:
- **No interactive GPU block.** Everything runs as SLURM jobs on partition `workq` (3-day cap, shared, contended).
  Shorter `--time` requests backfill sooner.
- **GPUs:** 2× RTX 6000 Ada (sm_89) on the master node, and 2× A100 80 GB (sm_80) on each of node02 and node03.
- **Don't touch the login node's GPUs.** They are visible without an allocation, but they belong to whoever's job
  is scheduled there. Use them only through `sbatch`.
- **Data lives on NFS** (`/dist_home/...`, about 260 TB free). There is no scratch filesystem.

## 0. Code and environment (login node, once)

```bash
git clone -b replication https://github.com/BRUH-MAIN/Ultralightunet.git && cd Ultralightunet
bash scripts/cluster/setup_env.sh        # isolated venv from conda's python3.11, torch 2.10/cu128, prebuilt mamba
                                         # wheels, nnU-Net 2.8.1; CPU-only tests here (no GPU outside SLURM)
export PVM3D_DATA_ROOT=/dist_home/<you>/<project>/pvm3d_data     # shared NFS path, about 400 GB; put it in ~/.bashrc
sbatch scripts/cluster/slurm/gpu_check.sbatch                          # kernel tests on a compute GPU ...
sbatch --nodelist=asaicomputenode02 scripts/cluster/slurm/gpu_check.sbatch   # ... once per GPU type (A100 / Ada)
```

`gpu_check-<jobid>.out` must end with all tests passing before any training.

Every job sources `.venv-cluster` and `scripts/cluster/env.sh "$PVM3D_DATA_ROOT"` through
`scripts/cluster/slurm/common.sh`. Submit jobs from the repo root, with `PVM3D_DATA_ROOT` exported.

## 1. Data

**Open datasets** (login node, inside tmux; about 53 GB, roughly 1 h):

```bash
bash scripts/cluster/download_open_data.sh $PVM3D_DATA_ROOT/downloads    # MSD Hippocampus + Liver (LiTS), AMOS22
```

**Datasets that need a person** (registration or third-party code; see the header of `download_open_data.sh`):
- **KiTS23:** the official downloader from github.com/neheller/kits23.
- **ACDC:** CREATIS registration.
- **BraTS2021, BTCV:** Synapse.

Put each one under `$PVM3D_DATA_ROOT/downloads/`.

**Convert**, which is file copying and fine on the login node, then plan and preprocess as a CPU job:

```bash
source .venv-cluster/bin/activate && source scripts/cluster/env.sh "$PVM3D_DATA_ROOT"
C=scripts/cluster/convert_datasets.py; D=$PVM3D_DATA_ROOT/downloads
python $C msd  --src $D/Task04_Hippocampus        # -> Dataset004
python $C msd  --src $D/Task03_Liver              # -> Dataset003 (LiTS)
python $C amos --src $D/amos22                    # -> Dataset219
# later: acdc (27), kits23 (220), brats21 (137), btcv (17) -- same pattern, see convert_datasets.py --help
sbatch scripts/cluster/slurm/preprocess.sbatch 4 3 219          # plan + preprocess + div32 plans + verify
python scripts/cluster/make_plans_variant.py -d 27 220 --divisible 16   # SegMamba (S7), after those are planned
python $C verify                                               # every dataset must print OK
```

What `verify` checks:
- Case counts.
- Plans and splits are present.
- For AMOS, it writes `modality.json` (CT vs MRI read from image intensities) and expects 300 CT / 60 MRI.
  The id≥500 shortcut is ambiguous: the official metadata lists amos_id 500 as both a CT and an MR scan.

## 2. Training

```bash
sbatch scripts/cluster/slurm/train.sbatch scripts/cluster/runs/S1.txt 4        # 1 GPU, capacity 4, 24 h
sbatch scripts/cluster/slurm/train.sbatch scripts/cluster/runs/S1.txt 4        # a 2nd GPU on the same run file
python scripts/cluster/pack_runs.py scripts/cluster/runs/S1.txt --status        # state/epoch/s-per-epoch/ETA/Dice
```

How jobs behave:
- **Shared run files are safe.** Several jobs can work through the same run file at once: each run is claimed with
  a lock file in its output folder, heart-beaten every 30 s and taken over 15 min after its job dies.
- **Jobs stop cleanly.** `--slurm-auto` stops launching about 20 min, and stops everything about 6 min, before the
  job's time limit.
- **Resubmitting continues.** Finished runs are skipped and partial runs resume (`--c`). Our trainers checkpoint
  every 10 epochs; nnU-Net (org.) every 50.
- **Choosing `--time`:** 24 h is the default. Shorter backfills sooner; longer wastes less on restarts. The
  partition cap is 72 h.
- **Choosing `--cpus-per-task`:** about 6 per concurrent light run, because data augmentation is CPU-bound.
  `train.sbatch` asks for 24 CPUs for capacity 4.

**First S1 job, which decides capacity and the ablation schedule:**
1. Watch `--status` s/epoch for the PVM-3D and LightM-UNet runs, and `sacct -j <id> --format=AveCPU,MaxRSS`.
2. If s/epoch × 1000 > about 8 h for light models, set `ABLATION_EPOCHS = 500` in
   `pvm3d/nnunet_trainers/trainers_pvm3d.py` **before S2**.
3. If the GPU is under about 60% busy while the CPUs are saturated, give the job more CPUs rather than more
   capacity.

**NFS throughput:** untested. If `--status` shows A100 runs much slower than the RTX 5060 laptop (about 12 s/epoch
for PVM-3D-S on Hippocampus), suspect data loading from NFS and say so before scaling up.

## 3. External baselines (needs Rohan's go-ahead: runs third-party code)

```bash
bash scripts/cluster/fetch_baselines.sh      # LightM-UNet, Slim-UNETR, SegFormer3D, SegMamba @ pinned commits
python -m pytest tests/test_external_baselines.py -v    # inside an sbatch job: their first execution anywhere
```

SegMamba (P1, before S7) needs its own venv, as printed by `fetch_baselines.sh`.

## 4. After runs finish

```bash
for ds in 4 3 219; do for tr in nnUNetTrainerPVM3DFinal_S nnUNetTrainer; do
  python scripts/eval3d.py -d $ds -tr $tr -f 0 1 2 3 4         # exits non-zero if DSC != nnU-Net summary.json
done; done
python scripts/eval3d.py -d 219 -tr nnUNetTrainerPVM3DFinal_S_CTOnly -f 0 --modality-split amos
python scripts/stats3d.py --csv "$nnUNet_results"/*/*/eval3d_2mm.csv --ref nnUNetTrainerPVM3DFinal_S \
    --efficiency results/efficiency/*.jsonl --out results/paper
```

Ablation table: run `stats3d.py` on the ablation CSVs with `--ref nnUNetTrainerPVM3D_S_ref`.

## 5. Gates

- **Gate A (after S1):** our ACDC nnU-Net (org.) 5-fold mean must be within ±0.5 DSC of Revisited's 91.54.
  Compare KiTS fold 0 against 86.04 loosely, since it is one fold.
- **Gate B (after S2):**
  1. Choose the design from the ablation table.
  2. Set `FINAL_DESIGN` in `trainers_pvm3d.py`, then commit and push.
  3. `git pull` on the cluster.
  4. Only then submit S3.
  - Runs refuse to resume under a changed definition (`pvm_config.json`).

## 6. Efficiency numbers (no training needed; run as a GPU job)

```bash
python scripts/flops3d.py -d 220 -tr nnUNetTrainerPVM3DFinal_{T,S,B} nnUNetTrainer -o results/efficiency/flops.jsonl
python scripts/bench_infer3d.py -d 220 -tr nnUNetTrainerPVM3DFinal_S nnUNetTrainer --train-step \
    -o results/efficiency/bench_a100.jsonl
```

To benchmark on the laptop without the data, copy `$nnUNet_preprocessed/*/nnUNetPlans.json` and `dataset.json` back
and pass them with `--plans-file` / `--dataset-json`.
