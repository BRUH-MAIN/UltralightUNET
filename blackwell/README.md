# UltraLight VM-UNet — Blackwell (RTX 50-series) build

Self-contained v2 of the codebase, for an **RTX 5060** or any other Blackwell GPU.
Run everything from this directory. Start with
**[`notebooks/train_test.ipynb`](notebooks/train_test.ipynb)**.

The science is identical to the root codebase — same model, same hyperparameters, same data.
The differences are all compatibility or speed. See [Differences](#differences-from-the-root-codebase).

## Why a separate directory

The RTX 5060 is Blackwell, compute capability **sm_120**. The `torch 2.0.1+cu117` pinned at the
repo root cannot run on it *at all*:

```
torch 2.0.1+cu117 arch_list: ['sm_37','sm_50','sm_60','sm_61','sm_70','sm_75','sm_80','sm_86','compute_37']
RTX 5060                   : sm_120
```

No matching kernel, and the only embedded PTX is `compute_37` — far too old to JIT forward to
Blackwell. The failure is `no kernel image is available for execution on the device`.
**PyTorch 2.7.0** was the first stable release with native sm_120 CUDA 12.8 wheels.

Rather than bump the root environment and risk invalidating the Kaggle T4 results already
recorded in [`../results/COMPARISON.md`](../results/COMPARISON.md), this is a parallel copy.

## Setup

```bash
cd blackwell
uv venv --python 3.11 .venv
.venv\Scripts\activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
python -m ipykernel install --user --name ultralight-bw --display-name "UltraLight (Blackwell)"
```

Then open `notebooks/train_test.ipynb` and select the **UltraLight (Blackwell)** kernel.
Cell 1 verifies the install by executing a real CUDA kernel — it does not trust version strings.

## Data

The dataset repo is private, so authenticate once:

```bash
huggingface-cli login          # or set HF_TOKEN in the environment
python scripts/hf_data.py pull --dataset ISIC2017
```

This downloads the six prepared `.npy` splits (524 MB) into `data/ISIC2017/`. Preprocessing and
the train/val/test split happen **once**, upstream of every machine, so nothing about the split
can drift between environments.

> **The split matters more than it looks.** An earlier run used a *sorted* file listing —
> reproducible, but biased: ISIC IDs correlate with acquisition source, producing train/val/test
> mean lesion areas of 22.9% / 8.0% / 15.0%. That alone cost 4.1 DSC points. The seeded
> permutation (`SPLIT_SEED = 42`) gives 20.0% / 17.6% / 18.6%. Cell 3 prints these — if you do
> not see roughly those numbers, the data is stale.

## Run

```bash
python -m pytest tests/ -q     # 19 tests, expect all passing
python train.py                # 250 epochs, then auto-evaluates the best checkpoint
python test.py --weights results/<run>/checkpoints/best-epoch116-loss0.2545.pth
```

`train.py` prints the parameter count (**must be 49,457**) and GFLOPs at startup, and writes
`checkpoints/latest.pth` every epoch so an interrupted run resumes automatically.

## Differences from the root codebase

Three changes, none of which alter the training trajectory.

### 1. `timm.layers` import, with fallback

timm moved `trunc_normal_` to `timm.layers` in 0.9; `timm.models.layers` is a deprecated shim.
The cu128 environment installs a modern timm, the root environment pins 0.4.12, so
`models/UltraLight_VM_UNet.py` tries the new path and falls back. This is the *only* difference in
that file.

### 2. `torch.load(..., weights_only=False)`

torch ≥ 2.6 flipped the default to `True`, which **rejects these checkpoints**: `min_loss` and
`loss` are `np.float64` (because `engine.py` returns `np.mean(...)`), and numpy scalars are not in
the default allowlist. Verified — loading such a checkpoint with `weights_only=True` raises
`UnpicklingError`. These are our own files, so full unpickling is safe.

Without this, a fresh run works but **resuming crashes**, which is the worst time to find out.

### 3. `val_batch_size = 30` (speed only)

No gradients are taken during validation, so the training trajectory is unaffected. The root
codebase hardcodes batch 1, which is maximally launch-bound for a workload whose bottleneck *is*
launch overhead.

`val_one_epoch` aggregates with `np.mean` over per-batch losses; `BCELoss` reduces over every
pixel and `DiceLoss` averages per sample, so for equal-sized batches the mean of batch means
equals the overall mean. Measured over the 150 val images:

| val batch | images seen | Δ loss vs batch 1 | speedup |
|---|---|---|---|
| 1 | 150 | baseline | 1.0× |
| 5 | 150 | −2e-08 | 8.3× |
| **30** | 150 | **+6e-05** | **20.1×** |
| 8 | **144** ← discards 6 | +3.9e-03 | 11.9× |

6e-05 on a loss of ~1.36 is fp32 reduction-order noise. Validation was 6 s of every 23 s epoch, so
this takes ~25% off total wall clock.

**The batch size must divide the split exactly.** The val/test loaders use `drop_last=True`, so a
non-divisor silently *discards* images — batch 8 would evaluate 144 of 150 and shift the loss 60×
more than the noise floor. `train.py` asserts this rather than trusting it.

One honest caveat: best-checkpoint selection compares val losses, so in a near-exact tie (within
6e-5) a different epoch could win. Not observed, and far below run-to-run variation.

Test batch size stays at **1** because `engine.test_one_epoch` calls `save_imgs`, which does
`img.squeeze(0)` and assumes a batch of one.

## Performance notes

Reference: **1.64 h** for 250 epochs on a Kaggle T4 (~23 s/epoch). The RTX 5060 should beat it —
newer architecture, and since the bottleneck is CPU launch overhead rather than GPU throughput, a
modern laptop CPU helps more than the GPU does. Val batching takes off another ~25%.

**VRAM is not the constraint.** Peak usage at batch 8 is ~0.7 GB of the 5060's 8 GB. The model is
0.049 M parameters; the cost is that the selective scan issues ~176 sequential kernel launches per
forward *regardless of batch size*. `scripts/bench_batch.py` measures where throughput actually
peaks on your hardware — on an RTX 3050 it was 2.07× at batch 32.

**But do not raise `batch_size` for a replication run.** 8 is the paper's hyperparameter; 32 means
39 optimiser steps per epoch instead of 157, which changes the result. It is a fine lever for your
own experiments provided you hold it constant across everything you compare.

**`num_workers = 0`** deliberately. Raising it hides the CPU-side `scipy.ndimage.rotate`
augmentation and is the single biggest remaining speedup, but each worker seeds its own RNG, so
the augmentation stream — and the training trajectory — changes. Left alone so the replication
stays comparable.

## Comparing results across machines

This environment runs a different torch and cuDNN than the Kaggle T4 runs, so results may differ
slightly even with an identical split and seed — convolution algorithm selection is not guaranteed
stable across versions. Fine for a standalone replication; worth a footnote if you put numbers
from both machines in one table.

The full deviation list from the reference implementation is in [`../README.md`](../README.md).
