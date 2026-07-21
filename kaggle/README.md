# Running the training on Kaggle (2 × T4)

The local RTX 3050 is for script development, tests, and smoke runs. The 250-epoch runs happen
here.

## Before the first run

1. Push this repo to GitHub (the notebook clones it).
2. Prepare the data locally and push it to HuggingFace — see the Data section of the top-level
   [README](../README.md). The notebook never re-derives the split.
3. If the HuggingFace dataset repo is **private**, add your token as a Kaggle notebook secret named
   `HF_TOKEN` (Add-ons → Secrets). A public repo needs no token.
4. In the notebook settings, set Accelerator to **GPU T4 x2** and turn Internet **on**.

## Notebook cells

```python
# 1. what we got
!nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

```python
# 2. code
!git clone -q https://github.com/<user>/UltralightUNET.git /kaggle/working/repo
%cd /kaggle/working/repo
```

```python
# 3. deps. Kaggle images already carry torch, timm, einops, scipy, sklearn,
#    matplotlib and tqdm; thop and huggingface_hub are the usual gaps.
!pip install -q thop huggingface_hub
```

```python
# 4. data (skip the token lines if the dataset repo is public)
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

!python scripts/hf_data.py pull --dataset ISIC2017 --repo <user>/ultralight-vmunet-data
```

```python
# 5. sanity check before burning hours: scan equivalence + parameter count
!pip install -q pytest && python -m pytest tests/ -q
```

```python
# 6. train. One experiment per GPU -- see below.
!CUDA_VISIBLE_DEVICES=0 python train.py
```

Results land in `results/UltraLight_VM_UNet_ISIC2017_<timestamp>/`, with `log/train.info.log`
holding the metrics and `checkpoints/` the weights. Download that directory before the session
ends, or write it to a Kaggle Dataset output — `/kaggle/working` is wiped when the session dies.

## Why one GPU per run, not DataParallel

The model is 0.049 M parameters and its selective scan is a Python loop issuing many very small
CUDA kernels, so the bottleneck is launch overhead on the CPU rather than GPU arithmetic.
`DataParallel` re-replicates the module every step and runs the replicas in threads that contend on
the GIL — it adds to exactly the cost that dominates here. It also halves the per-GPU batch from
the paper's 8 to 4.

Running two independent experiments concurrently uses both cards without either problem, and each
one matches the paper's single-V100 setup:

```python
# in two separate cells, both backgrounded
!CUDA_VISIBLE_DEVICES=0 nohup python train.py > /kaggle/working/isic2017.log 2>&1 &
!CUDA_VISIBLE_DEVICES=1 nohup python train.py > /kaggle/working/isic2018.log 2>&1 &
```

(Edit `configs/config_setting.py:datasets` between launching the two, or pass the dataset through
an environment variable if you have added one.)

## Session limits

Kaggle caps a GPU session at 12 hours and 30 GPU-hours/week. `train.py` writes
`checkpoints/latest.pth` every epoch and resumes from it automatically on restart, so a run that
outlives a session can be continued by re-running the notebook with the previous
`results/<run>/` restored — keep the run directory in a Kaggle Dataset between sessions.
