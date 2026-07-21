# Replication results vs. the paper — ISIC2017

Reference: Table 1 of Wu et al., *Patterns* 6, 101298 (2025), ISIC2017 block.

## Structural checks (independent of training)

| | paper | ours | |
|---|---|---|---|
| parameters | 49,457 (0.049 M) | **49,457** | exact |
| GFLOPs | 0.060 | **0.0602** | see README, thop cannot see fused CUDA calls |

The parameter count matching exactly is the strong evidence that the vendored pure-PyTorch Mamba
has the same parameterisation as `mamba_ssm` at every one of the six PVM layers.

## Run 1 — sorted split (superseded)

250 epochs, Tesla T4, 2026-07-21. Best val loss 0.2545 @ epoch 116.

| metric | paper | run 1 | Δ |
|---|---|---|---|
| DSC / F1 | 0.9091 | 0.8682 | **−0.0409** |
| IoU | 0.8334 | 0.7670 | **−0.0664** |
| SE / Recall | 0.9053 | 0.8458 | **−0.0595** |
| Prec | 0.9481 | 0.8917 | **−0.0564** |
| ACC | 0.9646 | 0.9614 | −0.0032 |
| SP | 0.9790 | 0.9818 | +0.0028 |

Confusion matrix: `TN 32,802,805 · FP 607,114 · FN 911,452 · TP 5,000,229`

### Diagnosis: the split, not the model

`dataprepare/Prepare_ISIC2017.py` originally **sorted** the file listing before slicing
`0:1250 / 1250:1400 / 1400:2000`. That was introduced for reproducibility — upstream slices raw
`glob.glob()` order, which is filesystem-dependent — but it turned out to be actively harmful,
because ISIC IDs correlate with acquisition source. Measured on the sorted sequence:

| split | mean lesion area | mean brightness |
|---|---|---|
| train `[0:1250]` | **22.9%** of frame | 160.8 |
| val `[1250:1400]` | **8.0%** | 162.9 |
| test `[1400:2000]` | **15.0%** | 147.4 |

`corr(sorted index, lesion fraction) = −0.281`. Per-200-image blocks run 30–32% foreground at the
start of the sequence and 7–11% around `[1200:1800]`.

So the model trained on lesions averaging 14,975 px and was tested on lesions averaging 9,853 px.
An under-segmenting model is exactly what the error signature shows: sensitivity and precision both
down ~0.06, specificity *up* slightly, accuracy nearly unchanged — dominated by the easy background
class. It also explains the otherwise-backwards result that validation DSC (0.81–0.83 throughout
training) came in *below* test DSC (0.868): the val block is the most extreme, at 8.0% foreground.

Upstream's unsorted `glob.glob` on Linux returns near-arbitrary directory order, which is
effectively the "randomly divided" split the paper describes. Sorting traded fidelity for
reproducibility without either being necessary.

### Fix

A **seeded permutation** (`SPLIT_SEED = 42`, matching `config.seed`) is reproducible *and*
unbiased. After reshuffling:

| split | foreground | brightness |
|---|---|---|
| train | 20.0% | 156.9 |
| val | 17.6% | 157.2 |
| test | 18.6% | 156.8 |

Still a verified clean partition of exactly 2000 unique images.

Prepared split hashes (sha256, first 16 hex):

| file | sha256 |
|---|---|
| `data_train.npy` | `d8f1101f99fd7be6…` |
| `data_val.npy` | `3c7fe7c64546c094…` |
| `data_test.npy` | `4f2e920a0fda6bab…` |
| `mask_train.npy` | `10563338a5bff3b6…` |
| `mask_val.npy` | `ff0ec2327b170f6b…` |
| `mask_test.npy` | `d28c772b730a0847…` |

## Run 2 — seeded shuffle split

_Pending._ Same code, same hyperparameters, corrected split.

| metric | paper | run 2 | Δ |
|---|---|---|---|
| DSC / F1 | 0.9091 | | |
| IoU | 0.8334 | | |
| SE / Recall | 0.9053 | | |
| SP | 0.9790 | | |
| ACC | 0.9646 | | |
| Prec | 0.9481 | | |

## Remaining known deviations

Even with the split corrected, an exact match is not expected, because the paper's split is
described only as "random" with no seed given — a different random partition of 2000 images will
land a little differently. Anything within roughly ±0.01 DSC should be read as a successful
replication.

Other differences, all detailed in [../README.md](../README.md):

- pure-PyTorch selective scan instead of the fused CUDA kernel (verified equivalent, 19/19 tests)
- batched PVM branches and `unbind` in the scan loops (verified equivalent, forward and gradients)
- uint8 intermediate storage (lossless w.r.t. `scipy.misc.imresize`, which returned uint8)
- Pillow resize rather than SciPy 1.2's `imresize` wrapper around the same PIL call
- single GPU rather than the paper's single V100 — matched, no DataParallel

## Reproducing

```bash
python scripts/download_isic.py --dataset ISIC2017
python dataprepare/Prepare_ISIC2017.py
python -m pytest tests/ -q          # expect 19 passed
python train.py
```
