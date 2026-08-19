# Running the training on Kaggle (2 × T4)

Optional second machine. Everything here also runs locally — see the root
[README](../README.md) — but a Kaggle session gives you two more GPUs to run experiments on in
parallel.

## Use the notebook

**[`notebooks/train_isic2017.ipynb`](notebooks/train_isic2017.ipynb)** — upload it to Kaggle and
run the cells in order. Nothing to fill in.

Notebook settings:

| setting | value |
|---|---|
| Accelerator | **GPU T4 x2** |
| Internet | **on** |
| Add-ons → Secrets | **`HF_TOKEN`** (the dataset repo is private) |

The cells: hardware check → clone + deps → pull data → **sanity checks** → optional batch-size
benchmark → train → zip results.

Do not skip the sanity-check cell. It costs seconds and asserts the parameter count is exactly
49,457 plus the full kernel-vs-oracle equivalence suite, on *Kaggle's* torch build rather than the
dev machine's. That is where a version-drift failure surfaces, instead of hours into a run.

## The two CUDA extensions

`mamba_ssm` and `causal_conv1d` are not in the Kaggle image. The bootstrap cell runs
`scripts/install_mamba.py`, which picks the prebuilt wheel matching the image's torch, python and
CUDA major. The T4 is **sm_75**, which every cu12 wheel covers natively — the Blackwell caveat in
the root README does not apply here.

The one thing that can go wrong is Kaggle moving to a torch minor the mamba release matrix does not
cover yet. The script says so explicitly rather than installing something that will not import; the
fallback is to pin torch first, or to compile with `scripts/build_mamba.py` (tens of minutes).

## Measured cost

Run 1: **1.64 h** for 250 epochs on a single T4, ~23 s/epoch (train 17 s, val 6 s) — measured with
the pure-PyTorch scan, before the migration to `mamba_ssm`. The fused kernel should be no slower.

This workload is bound by **CPU launch overhead**, not GPU throughput — the network is a long chain
of individually tiny operations and the model is only 0.049 M parameters. Two consequences worth
knowing:

- A batch of 8 uses **under 1 GB of the T4's 15 GB**. That is expected; VRAM is not the limiting
  resource. `scripts/bench_batch.py` (cell 5) measures where throughput actually peaks on the
  session's hardware. **Do not raise `config.batch_size` for a replication run** — 8 is the
  paper's hyperparameter, and 32 would mean 39 optimiser steps per epoch instead of 157.
- Wall clock is sensitive to which CPU the session drew, more than which GPU.

## Using both T4s

Run two independent experiments concurrently rather than splitting one across both.
`DataParallel` re-replicates the module every step and runs replicas in GIL-contending threads,
adding to exactly the launch overhead that dominates here, and it would halve the per-GPU batch
from the paper's 8 to 4. One run per GPU also matches the paper's single-V100 setup.

```python
import os, subprocess
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}
subprocess.Popen(['python', 'train.py'], env=env,
                 stdout=open('/kaggle/working/run0.log', 'w'), stderr=subprocess.STDOUT)
# edit configs/config_setting.py:datasets before launching the second on device 1
```

## Session limits

Kaggle caps a GPU session at 12 h and 30 GPU-hours/week — comfortable against a 1.64 h run.
`train.py` writes `checkpoints/latest.pth` every epoch and resumes automatically, so a run that
outlives a session can continue if you restore the previous `results/<run>/` directory first.

## Regenerating the notebook

`kaggle/notebooks/train_isic2017.ipynb` is checked in directly — edit it in Jupyter or on Kaggle
and commit the result. Keep the cells free of placeholders: an earlier revision of this file used
`<user>` as a stand-in, which bash reads as *redirect stdin from a file named `user`*, so the
clone silently did nothing and only surfaced three cells later as a missing `tests/` directory.
