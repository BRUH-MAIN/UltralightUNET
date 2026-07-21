# UltraLight VM-UNet — replication

Replication of **UltraLight VM-UNet: Parallel Vision Mamba significantly reduces parameters for
skin lesion segmentation** (Wu et al., *Patterns* 6, 101298, 2025) — [paper PDF](paper/) ·
[upstream code](https://github.com/wurenkai/UltraLight-VM-UNet).

The goal is a trustworthy, self-owned baseline: reproduce the paper's ISIC2017 numbers first, then
build novelty on top of a result we can defend.

**Target (Table 1, ISIC2017):** DSC 0.9091 · IoU 0.8334 · ACC 0.9646 · SE 0.9053 · SP 0.9790 ·
Prec 0.9481, at 0.049 M params / 0.060 GFLOPs.

### Status

| check | paper | ours | |
|---|---|---|---|
| parameters | 49,457 (0.049 M) | **49,457** | exact |
| GFLOPs | 0.060 | **0.0602** | see note below |
| equivalence tests | — | 19/19 pass | scan + PVM batching, fwd < 1e-5, grads < 1e-4 |
| epoch time | — | 48 s (train + val) | RTX 3050; **~3.3 h** for 250 epochs |
| ISIC2017 metrics | DSC 0.9091 | _pending_ | |

> **On GFLOPs.** Our raw thop reading is 0.0649. The gap is a measurement artifact, not a model
> difference: thop can only count operations that pass through an `nn.Module.forward`. In the
> paper's environment Mamba's internals run as fused CUDA calls (`causal_conv1d_fn`,
> `mamba_inner_fn`) that thop never sees, whereas we invoke `conv1d`, `x_proj` and `out_proj` as
> real modules. Excluding exactly those — 0.0004 for the depthwise conv, 0.0042 for the two
> projections — leaves **0.0602**, matching the paper to three decimals. The honest total for this
> implementation is 0.065.

## Where things run

| | machine | role |
|---|---|---|
| Debug | Windows 11, RTX 3050 Laptop (4 GB) | script development, tests, smoke runs |
| Train | Kaggle, 2 × T4 | the 250-epoch runs (one experiment per GPU, in parallel) |
| Data | HuggingFace dataset | prepared `.npy` splits, shared by both |

Preprocessing and the train/val/test split happen **once, locally**. Both machines then consume the
identical prepared bytes, so nothing about the split can drift between debugging and training.

## Setup

```bash
uv venv --python 3.11 .venv
.venv\Scripts\activate                      # Windows
uv pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 \
    --index-url https://download.pytorch.org/whl/cu117
uv pip install -r requirements.txt
```

## Data

```bash
python scripts/download_isic.py --dataset ISIC2017   # 5.8 GB from the ISIC S3 bucket
python dataprepare/Prepare_ISIC2017.py               # -> data/ISIC2017/*.npy  (~525 MB)
python scripts/hf_data.py push --dataset ISIC2017 --repo <user>/ultralight-vmunet-data
```

Then on any machine (including a Kaggle notebook), skip straight to:

```bash
python scripts/hf_data.py pull --dataset ISIC2017 --repo <user>/ultralight-vmunet-data
```

### Prepared ISIC2017 splits

Built from `ISIC-2017_Training_Data.zip` (5.8 GB, 2000 JPEGs) and
`ISIC-2017_Training_Part1_GroundTruth.zip` (8.9 MB, 2000 PNGs), both from
`isic-challenge-data.s3.amazonaws.com`. The 2001 superpixel/licence files bundled in the image
archive are discarded. 524 MB total.

| file | shape | dtype | sha256 (file) |
|---|---|---|---|
| `data_train.npy` | (1250, 256, 256, 3) | uint8 | `ff008f4b31e2cd3c…` |
| `data_val.npy` | (150, 256, 256, 3) | uint8 | `4d166e179a28d2ba…` |
| `data_test.npy` | (600, 256, 256, 3) | uint8 | `b07d93f4fdb1210c…` |
| `mask_train.npy` | (1250, 256, 256) | uint8 | `71f79783f5ad69dd…` |
| `mask_val.npy` | (150, 256, 256) | uint8 | `00a26b8fd73be7bd…` |
| `mask_test.npy` | (600, 256, 256) | uint8 | `1eb78e7cf33d3066…` |

Verified: no image appears in more than one split; masks are 22.9% foreground with 1.02%
intermediate values (bilinear edge blur, as expected); the loader yields images in **[0, 255]**
— not [0, 1] — and masks in [0, 1], matching upstream. The copy on HuggingFace was re-downloaded
and confirmed byte-identical to these before the raw archives were deleted.

## Run

```bash
pytest tests/ -v            # scan equivalence + per-layer parameter counts
python train.py             # writes to results/<network>_<dataset>_<timestamp>/
```

`train.py` prints the parameter count and GFLOPs at startup, trains for 250 epochs, and
automatically evaluates the best checkpoint on the test split at the end. Point
`configs/config_setting.py:test_weights` at a checkpoint and run `python test.py` to re-evaluate
without retraining.

On Kaggle, pin each run to one GPU rather than splitting one run across both — at 0.049 M
parameters the bottleneck is kernel-launch overhead, which `DataParallel` adds to:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py   # e.g. ISIC2017
CUDA_VISIBLE_DEVICES=1 python train.py   # e.g. ISIC2018, concurrently
```

## Deviations from upstream

Every difference from https://github.com/wurenkai/UltraLight-VM-UNet is listed here. Anything not
in this list is a verbatim transcription — `models/UltraLight_VM_UNet.py` in particular is
byte-identical from `class PVMLayer` onward.

### 1. Mamba runs in pure PyTorch

Upstream requires `mamba_ssm==1.0.1` + `causal_conv1d==1.0.0`. Both are CUDA extensions with
Linux-only official support, and `mamba_ssm/ops/selective_scan_interface.py` does a top-level
`import selective_scan_cuda` — so on Windows the package cannot be imported at all.

[`models/mamba_pytorch.py`](models/mamba_pytorch.py) reimplements the same module in plain PyTorch.
This is **not** an approximation: `selective_scan_ref` is the reference scan from the official Mamba
repository (the function its CUDA kernel is tested against), and `selective_scan_chunked` — the one
that actually runs — is that identical recurrence with the sequential product reassociated over
chunks. `tests/test_mamba_equivalence.py` asserts they agree to < 1e-5 forward and < 1e-4 on every
gradient, at the six real layer shapes.

The fused kernel exists to make `d_inner` in the thousands tractable. Here the six PVM layers run at
`d_inner` 12–32 over sequences of at most 1024 tokens, so there is very little for it to accelerate:

| PVM layer | `d_model` | `d_inner` | `dt_rank` | seq len @ 256×256 |
|---|---|---|---|---|
| encoder4 (24→32) | 6 | 12 | 1 | 1024 |
| encoder5 (32→48) | 8 | 16 | 1 | 256 |
| encoder6 (48→64) | 12 | 24 | 1 | 64 |
| decoder1 (64→48) | 16 | 32 | 1 | 64 |
| decoder2 (48→32) | 12 | 24 | 1 | 64 |
| decoder3 (32→24) | 8 | 16 | 1 | 256 |

`mamba_ssm` is deliberately **not** imported even where it would work (Kaggle is Linux). A silent
per-machine backend switch would mean the code being debugged is not the code producing the numbers.

> One consequence worth knowing: `UltraLight_VM_UNet.__init__` ends with
> `self.apply(self._init_weights)`, which reinitialises every `nn.Linear` (trunc_normal_ std=0.02,
> zero bias) and `nn.Conv1d` in the model — including the ones inside Mamba, and including
> `dt_proj.bias`, whose `_no_reinit` marker that function does not check. So the vendored module
> must keep the same submodule names and types for `apply()` to touch exactly what it touches
> upstream. That is load-bearing, not cosmetic.

### 2. Two performance patches that do not change the model

A pure-PyTorch scan is launch-bound: the recurrence is a Python loop issuing many very small CUDA
kernels, and CPU launch overhead — not GPU arithmetic — sets the pace. Measured on the RTX 3050 at
batch 8, a training step started at **0.984 s**, which is 13.2 h for 250 epochs — past Kaggle's
12 h session cap. Two changes bring it to **0.151 s/iter (~4 h)**, a 6.5× speedup:

| | s/iter (GPU step) | 250 epochs |
|---|---|---|
| upstream form | 0.984 | 13.2 h |
| + batch the four PVM branches | 0.474 | 7.6 h |
| + `unbind` instead of per-step indexing | **0.151** | — |

End to end, including data loading and augmentation, that is **48 s/epoch → ~3.3 h for 250
epochs** on the RTX 3050 (measured over a full train+val cycle at the real 1250/150 split sizes;
peak VRAM 0.67 GB of 4 GB). The residual gap between 0.151 s/iter and the 0.247 s/iter seen in the
real loop is CPU-side: `num_workers=0` plus `scipy.ndimage.rotate` augmentation on the main thread.
That is left alone deliberately — raising `num_workers` would change the augmentation RNG stream
and so change training, for a speedup we do not need.

**Batched PVM branches.** `PVMLayer.forward` calls the *same* `self.mamba` four times in sequence.
Mamba treats the batch dimension as independent — every operation is either per-token or a scan
along `L` — so the four branches stack onto the batch axis and become one call.

**`unbind` in the scan loops.** Indexing `a[:, :, :, k]` per step registers a separate
`slice_backward` for every timestep, each allocating a full-size zero tensor to scatter one slice
into. Profiling put `slice_backward` at 34% of the step. `unbind` yields the same views with a
single `stack` backward for the entire loop.

Neither touches the mathematics. `tests/test_pvm_batching.py` asserts the batched forward matches
`PVMLayer._forward_reference` (upstream's exact four-call form, retained in the file) at every
layer shape, on outputs *and* on all parameter gradients, plus end-to-end through the whole model.

### 3. Data prep without SciPy 1.2

`dataprepare/Prepare_ISIC2017.py` is rewritten to use Pillow instead of `scipy.misc.imread` /
`imresize` (removed in SciPy ≥ 1.3; upstream's workaround is a second Python 3.7 conda environment).
Two substantive changes:

- **Sorted file listing.** Upstream slices raw `glob.glob()` order, which is filesystem-dependent
  and therefore not reproducible across machines. We sort first. The paper describes the split only
  as "random", so this is the most likely source of any small metric delta.
- **uint8 storage.** `scipy.misc.imresize` returned uint8 and upstream immediately widened it with
  `np.double()`, so keeping the uint8 is lossless with respect to the original pipeline — and takes
  the `.npy` output from ~4 GB to ~525 MB. `loader.dataset_normalized` casts to float32.

### 4. Small environment patches

- `loader.py`: `scipy.ndimage.morphology` → `scipy.ndimage` (removed in SciPy ≥ 1.15); float32
  rather than float64 arrays (`engine.py` casts to `.float()` on the GPU regardless).
- `utils.py`: matplotlib `Agg` backend, since `save_imgs` writes ~600 PNGs headlessly.
- `train.py` / `test.py`: `DataParallel` wrapper dropped for single-GPU use, and the matching
  `.module` indirection with it. Checkpoint keys are unchanged — upstream saved
  `model.module.state_dict()`, which produces the same un-prefixed keys.
- `configs/config_setting.py`: `data_path` filled in. All hyperparameters are untouched — batch 8,
  250 epochs, AdamW lr 1e-3 / wd 1e-2, CosineAnnealingLR `T_max=50` `eta_min=1e-5`, seed 42,
  threshold 0.5, `amp=False`, 256×256 input, `c_list=[8,16,24,32,48,64]`.

### Unchanged on purpose

`engine.py` is verbatim. Its metrics pool a confusion matrix over **all pixels of an entire split**
rather than averaging per-image scores. That is how the paper's numbers are defined, so it stays
as-is even though per-image averaging is more common elsewhere.

## Layout

```
models/mamba_pytorch.py        pure-PyTorch Mamba + selective scans
models/UltraLight_VM_UNet.py   the model (verbatim upstream)
engine.py utils.py loader.py   train/val/test loops, losses, dataset
configs/config_setting.py      all hyperparameters
dataprepare/                   raw images -> .npy splits
scripts/download_isic.py       fetch ISIC archives from S3
scripts/hf_data.py             push/pull prepared splits via HuggingFace
tests/                         scan equivalence, parameter counts
```
