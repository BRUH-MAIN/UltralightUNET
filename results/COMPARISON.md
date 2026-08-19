# Replication results vs. the paper — ISIC2017

Reference: Table 1 of Wu et al., *Patterns* 6, 101298 (2025), ISIC2017 block.

## Structural checks (independent of training)

| | paper | ours | |
|---|---|---|---|
| parameters | 49,457 (0.049 M) | **49,457** | exact |
| GFLOPs | 0.060 | **0.0602** | as thop reads it; see README on the fused-kernel blind spot |

The parameter count matches exactly at every one of the six PVM layers, which is the structural
half of the replication: the model is the paper's model, not merely one shaped like it.

> **Note on the scan.** Runs 1 and 2 were produced by the pure-PyTorch reimplementation of Mamba in
> `models/mamba_pytorch.py`. Run 3 is the first on `mamba_ssm`'s fused CUDA kernel — the same
> dependency the paper uses — with that reimplementation kept as the oracle the test suite checks
> the kernel against. Run 3 landed within 0.0004 DSC of run 2, which is the empirical version of
> the equivalence the tests assert.

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

## Run 2 — seeded shuffle split — **replication successful**

250 epochs, RTX 5060 (Blackwell, sm_120), torch 2.7+/cu128, pure-PyTorch scan. Same model and
hyperparameters as run 1; only the split changed.

| metric | paper | run 2 | Δ | |
|---|---|---|---|---|
| **DSC / F1** | 0.9091 | **0.9030** | **−0.0061** | within tolerance |
| IoU | 0.8334 | 0.8232 | −0.0102 | forced by DSC, see below |
| SE / Recall | 0.9053 | 0.8957 | −0.0096 | |
| SP | 0.9790 | 0.9799 | +0.0009 | |
| ACC | 0.9646 | 0.9643 | −0.0003 | |
| Prec | 0.9481 | _not logged_ | | |

Test loss 0.2429. The `engine.py` log line does not print precision; it is recoverable from the
confusion matrix if needed.

### The split fix worked

Correcting the biased split moved DSC from **0.8682 → 0.9030 (+0.0348)**, recovering essentially
all of the 4.1-point gap that run 1 showed. The remaining gap to the paper is **0.0061 DSC
(0.67%)** — inside the ±0.01 band set before the run, so this is a successful replication of the
paper's headline ISIC2017 result.

### On the IoU flag

The notebook's comparison cell flags IoU as "outside ±0.01", but that is not an independent
finding. For a single foreground class, IoU and DSC are the same measurement:

```
IoU = DSC / (2 − DSC)
```

This reproduces both rows exactly at the reported precision — paper DSC 0.9091 → IoU 0.8333
(reported 0.8334), ours DSC 0.9030 → IoU 0.8232 (reported 0.8232). So the −0.0102 IoU delta is
mechanically forced by the −0.0061 DSC delta, amplified 1.67× by the nonlinearity; it carries no
information beyond the DSC gap. A ±0.01 threshold is simply too tight for IoU when the same
tolerance is applied to DSC — the honest single number to judge is the 0.67% DSC gap.

### Why not an exact match

The residual sub-1% gap is expected and not worth chasing:

- **The split.** The paper's partition is described only as "randomly divided", with no seed. A
  different random partition of the same 2000 images lands slightly differently; ours uses
  `SPLIT_SEED = 42`.
- **The stack.** This run used torch 2.7+/cuDNN on Blackwell; the paper used a single V100 on an
  unstated (older) torch. cuDNN convolution-algorithm selection is not bit-stable across versions
  or architectures.
- **Val batching.** `val_batch_size = 30` shifts the *selection* loss by ~6e-5 (fp32 noise), which
  could in principle pick a different best epoch in a near-tie. Far below the gap above.

None of these is a defect to fix; each is a documented, principled difference from the original.

## Run 3 — `mamba_ssm` fused kernel — **replication confirmed**

250 epochs, RTX 5060 (Blackwell, sm_120), torch 2.13.0+cu130, `mamba_ssm` 2.3.2.post1 +
`causal_conv1d` 1.6.2.post1 compiled from source for sm_120. Identical split, seed and
hyperparameters to run 2; the **only** change is that the selective scan is now the paper's fused
CUDA kernel instead of the pure-PyTorch transcription of it.

Best val loss 0.1960 @ epoch 130. Test loss 0.2453.
Confusion matrix: `TN 31,403,838 · FP 614,506 · FN 790,663 · TP 6,512,593`

| metric | paper | run 2 (PyTorch scan) | run 3 (`mamba_ssm`) | Δ vs paper | |
|---|---|---|---|---|---|
| **DSC / F1** | 0.9091 | 0.9030 | **0.9026** | **−0.0065** | within tolerance |
| IoU | 0.8334 | 0.8232 | 0.8225 | −0.0109 | forced by DSC |
| SE / Recall | 0.9053 | 0.8957 | 0.8917 | −0.0136 | |
| SP | 0.9790 | 0.9799 | 0.9808 | +0.0018 | |
| ACC | 0.9646 | 0.9643 | 0.9643 | −0.0003 | |
| Prec | 0.9481 | _not logged_ | 0.9138 | −0.0343 | but see below |

**The migration changed nothing measurable.** Run 3 lands 0.0004 DSC from run 2 — two independent
scan implementations, 250 epochs apart, agreeing to the fourth decimal. That is the empirical
counterpart to the equivalence the test suite asserts analytically.

**Wall clock: 22.9 min** (21.7 min training at 5.2 s/epoch, 1.1 min for the 600-image test pass
including its overlay PNGs), against 1.64 h for run 1 on a Kaggle T4.

### On the precision gap

Precision is the one metric more than 0.02 from the paper, and the discrepancy is in the paper's
table rather than in this run. DSC is by definition the harmonic mean of precision and recall, so
the paper's own three numbers have to satisfy it — and they do not:

```
2 · 0.9481 · 0.9053 / (0.9481 + 0.9053) = 0.9262    but the paper reports DSC 0.9091
2 · 0.9138 · 0.8917 / (0.9138 + 0.8917) = 0.9026    ours, matching our reported DSC exactly
```

Inverting the identity for the precision consistent with the paper's *own* DSC 0.9091 and SE 0.9053
gives **0.9129** — against our 0.9138, a difference of +0.0009. So on the paper's own definition of
DSC our precision agrees to within a thousandth; the tabulated 0.9481 cannot be the pooled-pixel
precision belonging to the rest of that row. Our six metrics are mutually consistent, all derivable
from the single confusion matrix above.

This is worth knowing before quoting the paper's precision as a target: it is the one cell in that
row that no run reproducing the DSC can also match.

## Remaining known deviations

Even with the split corrected, an exact match is not expected, because the paper's split is
described only as "random" with no seed given — a different random partition of 2000 images will
land a little differently. Anything within roughly ±0.01 DSC should be read as a successful
replication.

Other differences, all detailed in [../README.md](../README.md):

- `mamba_ssm` 2.3.2.post1 rather than the pinned 1.0.1, which cannot run on Blackwell at all
  (`mamba_simple.py`, the file the model imports, is unchanged between the two on the training path)
- batched PVM branches (verified equivalent, forward and gradients)
- uint8 intermediate storage (lossless w.r.t. `scipy.misc.imresize`, which returned uint8)
- Pillow resize rather than SciPy 1.2's `imresize` wrapper around the same PIL call
- single GPU rather than the paper's single V100 — matched, no DataParallel
- for runs 1 and 2 only: the pure-PyTorch selective scan, per the note at the top; run 3 uses the
  fused kernel and lands 0.0004 DSC away

## Reproducing

```bash
python scripts/download_isic.py --dataset ISIC2017
python dataprepare/Prepare_ISIC2017.py
python -m pytest tests/ -q          # kernel-vs-oracle equivalence
python train.py
```
