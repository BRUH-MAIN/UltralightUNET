# Replication results vs. the paper — ISIC2017, ISIC2018, PH2, HAM10000

Reference: Table 1 of Wu et al., *Patterns* 6, 101298 (2025) — the paper's ISIC2017, ISIC2018 and
PH2 blocks, all three now replicated. HAM10000 is not one of the paper's three benchmark datasets;
it's covered in its own section below as an extension beyond the paper's evaluation set.

- [ISIC2017](#isic2017) — the main replication saga (4 runs: split-bias diagnosis/fix, kernel migration, reproducibility check)
- [ISIC2018](#isic2018--replication-successful) — second independent replication, clean run
- [PH2](#ph2--replication-successful-small-test-set-caveat) — third and final paper dataset, tiny test set
- [HAM10000](#ham10000--generalization-beyond-the-papers-evaluation-set) — no paper target; tests generalization
- [Cross-dataset summary](#cross-dataset-summary) — all four side by side
- [Cross-dataset generalization](#cross-dataset-generalization) — zero-shot transfer matrix and findings
- [Explainability](#explainability) — attention/Seg-Grad-CAM case study, two ISIC2018 failure modes
- [Reproducing](#reproducing)

## ISIC2017

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

## Run 4 — reproducibility check (found incidentally, undocumented until now)

While building the Week 2 cross-dataset generalization sweep (`scripts/cross_eval.py`), the
checkpoint it auto-selected for ISIC2017 didn't reproduce run 3's DSC — it turned out a second,
essentially identical run had been sitting undocumented in
`results/UltraLight_VM_UNet_ISIC2017_Wednesday_19_August_2026_18h_51m_11s/` since 2026-08-19. Same
seed, same split, same hyperparameters, same `mamba_ssm 2.3.2.post1` fused kernel as run 3 — both
runs' logs print the identical line `mamba backend: mamba_ssm 2.3.2.post1, fused path:
mamba_inner_fn (causal_conv1d present)`, so this is not a scan-implementation swap the way run 2 →
run 3 was. It's the same code, same config, run again two days later.

| metric | run 3 (documented) | run 4 (this) | Δ |
|---|---|---|---|
| DSC / F1 | 0.9026 | 0.8993 | −0.0033 |
| best epoch / val loss | 130 / 0.1960 | 97 / 0.1954 | |
| test loss | 0.2453 | 0.2492 | |

### Why identical config doesn't mean identical result

`utils.py:set_seed` sets `torch.manual_seed` and pins cuDNN (`cudnn.deterministic = True`,
`cudnn.benchmark = False`) — but that reaches cuDNN's own convolution kernels, not `mamba_ssm`'s
hand-written CUDA kernel (`mamba_inner_fn`). Fused scan kernels commonly use atomic-add-based
reductions for performance, whose accumulation order — and therefore floating-point rounding —
depends on GPU thread scheduling rather than the RNG seed. `tests/test_mamba_equivalence.py`
already only checks this kernel against the pure-PyTorch oracle to fp32 *tolerance*, not
bit-exactness, for the same underlying reason.

Two runs of the exact same code landing 0.0033 DSC apart is a useful number in its own right: it's
an empirical noise floor for this pipeline. Any single-run "gap vs paper" or "beat the paper" claim
smaller than roughly this much — which includes ISIC2018's −0.0029 and PH2's +0.0047 above — is
within normal run-to-run variance, not necessarily a real difference. This doesn't change any
replication verdict in this document: run 3's 0.71%-of-paper gap and run 4's 1.06% gap are both
comfortably inside the ±0.01 band set beforehand. It's the honest error bar to attach to every DSC
number here, and it's why the [cross-dataset generalization matrix](#cross-dataset-generalization)
below — which needed *a* checkpoint per dataset, not necessarily run 3's — uses run 4 for ISIC2017:
whichever run a script picks by "most recent," that policy should be applied consistently rather
than hand-picking the better-looking number after the fact.

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

## ISIC2018 — replication successful

250 epochs, RTX 5060 Laptop GPU (sm_120), torch 2.13.0+cu130, `mamba_ssm` 2.3.2.post1 fused kernel,
seeded split (`SPLIT_SEED = 42`), 2026-08-22. Best val loss 0.2578 @ epoch 106. Wall clock ~61 min
(22:57:22 → 23:58:34). Unlike ISIC2017, `Prepare_ISIC2018.py` used a seeded shuffle from the start —
the split-bias lesson from ISIC2017 run 1 was already known, so there was no bug to diagnose here.

| metric | paper (Table 1, ISIC2018) | ours | Δ | |
|---|---|---|---|---|
| **DSC / F1** | 0.8940 | **0.8911** | **−0.0029** | within tolerance |
| IoU | 0.8056 | 0.8037 | −0.0019 | forced by DSC, same identity as ISIC2017 |
| SE / Recall | 0.8680 | 0.8861 | +0.0181 | |
| SP | 0.9781 | 0.9735 | −0.0046 | |
| ACC | 0.9558 | 0.9556 | −0.0002 | |
| Prec | 0.9197 | 0.8962 | −0.0235 | derived from the confusion matrix below; `engine.py` doesn't log it |

Confusion matrix: `TN 26,366,142 · FP 717,940 · FN 796,408 · TP 6,198,230`

**Replication successful.** The gap is 0.29% DSC — tighter than the 0.65% ISIC2017 gap — well
inside the ±0.01 band. This is a second, independent confirmation of replication fidelity: same
architecture, same seeded-split methodology, same fused kernel, different dataset.

### Per-image variance

Over the 520 test images: mean DSC 0.8809, std **0.1380**. 16 images (3.1%) score below 0.5 DSC,
39 (7.5%) below 0.7 — both notably higher outlier rates than HAM10000 (below). The worst cases:
test images 42 (0.0513), 359 (0.0681), 40 (0.0777), 347 (0.2280), 273 (0.2326). These are a natural
starting point for an explainability pass (attention/saliency maps) to see what characterizes the
failures — small lesions, hair artifacts, low contrast, etc.

## HAM10000 — generalization beyond the paper's evaluation set

HAM10000 is not one of the paper's three benchmark datasets, so there's no Table 1 number to
compare against here — this run instead tests whether the architecture generalizes to a dataset it
was never designed or tuned against, using the Tschandl et al. lesion-segmentation masks paired
with the HAM10000 dermoscopic images by filename stem (`scripts/download_ham10000.py`,
`dataprepare/Prepare_HAM10000.py`).

250 epochs, RTX 5060 Laptop GPU (sm_120), torch 2.13.0+cu130, `mamba_ssm` 2.3.2.post1 fused kernel,
seeded split, 2026-08-22→23. Best val loss 0.1624 @ epoch 138. Wall clock ~2h51m
(23:59:52 → 02:50:29), the longest of the three runs since HAM10000's test split (2,003 images) is
roughly 4x ISIC2017's and 4x ISIC2018's.

| metric | ours |
|---|---|
| **DSC / F1** | **0.9331** |
| IoU | 0.8746 |
| SE / Recall | 0.9213 |
| SP | 0.9807 |
| ACC | 0.9649 |
| Prec | 0.9452 (derived) |

Confusion matrix: `TN 94,506,386 · FP 1,864,389 · FN 2,745,908 · TP 32,151,925`

This is the highest DSC of the three datasets. Plausible drivers: HAM10000's prepared split has far
more images than ISIC2017 or ISIC2018 individually, giving the same 49,457-parameter model more
data per parameter to fit; the Tschandl masks may also be more annotation-consistent than ISIC's
crowd-sourced-style boundaries. Neither is confirmed — worth a closer look if pursued further.

### Per-image variance

Over the 2,003 test images: mean DSC 0.9328, std **0.0897** — the tightest of the three datasets.
17 images (0.8%) score below 0.5 DSC, 56 (2.8%) below 0.7. Worst cases: test image 1761 (0.0000 —
a full miss), 1573 (0.1769), 971 (0.1791), 192 (0.2156), 1019 (0.2208).

## PH2 — replication successful (small-test-set caveat)

250 epochs, RTX 5060 Laptop GPU (sm_120), torch 2.13.0+cu130, `mamba_ssm` 2.3.2.post1 fused kernel,
seeded split, 2026-08-23. Best val loss 0.1752 @ epoch 120. Wall clock ~3m17s — by far the shortest
run, since PH2 has only 140 train images (a ~14x smaller train set than ISIC2017/2018 and ~50x
smaller than HAM10000). Downloaded via the [Kaggle PH2
mirror](https://www.kaggle.com/datasets/spacesurfer/ph2-dataset) (`scripts/download_ph2.py`) since
upstream PH2 is a Google-Drive/request-form release rather than a plain URL; no official train/val
/test protocol is published for it, so this uses a 140/20/40 (70/10/20) seeded split, same ratio as
HAM10000.

| metric | paper (Table 1, PH2) | ours | Δ | |
|---|---|---|---|---|
| **DSC / F1** | 0.9265 | **0.9312** | **+0.0047** | ours slightly higher |
| IoU | 0.8631 | 0.8712 | +0.0081 | forced by DSC, same identity as the other two |
| SE / Recall | 0.9345 | 0.9100 | −0.0245 | |
| SP | 0.9606 | 0.9761 | +0.0155 | |
| ACC | 0.9521 | 0.9530 | +0.0009 | |
| Prec | 0.9187 | 0.9534 | +0.0347 | derived from the confusion matrix below |

Confusion matrix: `TN 1,664,436 · FP 40,782 · FN 82,480 · TP 833,742`

**Replication successful** — and the first of the three where our number comes in *above* the
paper's rather than within a negative tolerance band. Take that "beat the paper" framing with a
grain of salt, though: the test split is only **40 images**, an order of magnitude smaller than
ISIC2017's 600 or ISIC2018's 520, so a handful of images flipping from correct to wrong moves DSC by
more than a percentage point on its own. The honest read is "well within replication," not "better
architecture."

### Per-image variance

Over the 40 test images: mean DSC 0.9309, std **0.0516** — the tightest of all four datasets, and
zero images below 0.7 DSC (worst: image 29 at 0.7736). Consistent with PH2 being a small, curated
clinical release rather than a crowd-sourced challenge archive — but also consistent with 40 images
simply being too few to surface many outliers even if the underlying failure modes are similar to
the other datasets.

## Cross-dataset summary

| dataset | paper DSC | ours DSC | Δ | test n | per-image std | n below 0.5 DSC |
|---|---|---|---|---|---|---|
| ISIC2017 | 0.9091 | 0.9026 (run 3) | −0.0065 | 600 | — | — |
| ISIC2018 | 0.8940 | 0.8911 | −0.0029 | 520 | 0.1380 | 16 (3.1%) |
| PH2 | 0.9265 | 0.9312 | +0.0047 | 40 | 0.0516 | 0 (0%) |
| HAM10000 | — (not in paper) | 0.9331 | — | 2,003 | 0.0897 | 17 (0.8%) |

All four runs use the identical architecture (49,457 params, 0.0602 GFLOPs) and training recipe;
only the dataset changes. This completes replication on all three of the paper's benchmark datasets
(ISIC2017, ISIC2018, PH2) plus one beyond its scope (HAM10000).

## Cross-dataset generalization

`scripts/cross_eval.py` loads each dataset's best checkpoint and evaluates it — zero-shot, no
fine-tuning — on every other dataset's test split, reusing `engine.test_one_epoch` for every cell
so results are directly comparable to every in-domain number elsewhere in this document. Source
checkpoints: ISIC2018/PH2/HAM10000 use each dataset's only run; ISIC2017 uses **run 4** (DSC 0.8993
in-domain, not run 3's 0.9026 — see the [reproducibility note](#run-4--reproducibility-check-found-incidentally-undocumented-until-now)
above for why, and why that's the right call rather than hand-picking run 3).

### DSC matrix

| trained on ↓ / evaluated on → | ISIC2017 | ISIC2018 | PH2 | HAM10000 | avg (off-diag) |
|---|---|---|---|---|---|
| ISIC2017 | **0.8993** | 0.8885 | 0.9038 | 0.8895 | 0.8939 |
| ISIC2018 | 0.9074 | **0.8911** | 0.9094 | 0.9034 | **0.9067** |
| PH2 | 0.8082 | 0.7834 | **0.9312** | 0.7831 | 0.7916 |
| HAM10000 | 0.8496 | 0.8438 | 0.9079 | **0.9331** | 0.8671 |
| avg (off-diag) | 0.8551 | 0.8386 | **0.9070** | 0.8587 | |

Bold on the diagonal = in-domain (each dataset's own already-recorded result, reproduced here as a
sanity check that this sweep's plumbing matches `test.py`'s — it does, to 4 decimals, for
ISIC2018/PH2/HAM10000). Full per-cell mIoU/accuracy/sensitivity/specificity in
`results/cross_eval/cross_eval_matrix.csv`.

### Findings

**ISIC2018 is the best source and the hardest target — the same trait explains both.** Averaged
over the other three datasets, an ISIC2018-trained model scores 0.9067 DSC — the best of any source,
and on ISIC2017 (0.9074) it even beats ISIC2017's own in-domain result (0.8993). But ISIC2018 is
also the hardest *target*: models trained elsewhere average only 0.8386 DSC on it, the worst column
in the table. Week 1 already flagged ISIC2018 as the highest-variance dataset (per-image DSC std
0.1380, 3.1% of test images below 0.5 DSC — see above). That diversity looks like it cuts both ways:
training on a more heterogeneous dataset teaches more transferable features (good source), while
a more heterogeneous *test* set is harder for a model that hasn't seen that diversity to match (hard
target). Same underlying property, opposite consequence depending which side of the split it's on.

**PH2 generalizes worst as a source, by a wide margin.** A PH2-trained model averages only 0.7916
DSC elsewhere — 0.12–0.15 DSC below its own in-domain 0.9312, and clearly the worst row in the
table. The likely driver isn't (only) domain-specificity: PH2's train split is 140 images, ~9x
smaller than ISIC2017's 1,250 and ~50x smaller than HAM10000's 7,010, so this may simply be too
little data for the model to learn features beyond what one small, single-clinic dermoscopy archive
looks like. Dataset size and domain narrowness are confounded here and this sweep can't separate
them — a controlled experiment (subsample ISIC2018 to 140 images, retrain, retest) would.

**PH2 is the easiest target regardless of source.** Every other dataset's model scores its highest
or near-highest off-diagonal DSC *on* PH2 (ISIC2017→PH2 0.9038, ISIC2018→PH2 0.9094, HAM10000→PH2
0.9079 — all higher than those same models score on ISIC2017, ISIC2018, or HAM10000). Combined with
PH2 having zero test images below 0.7 DSC even in-domain (Week 1), the simplest explanation is that
PH2 itself is simply an easy, clean, low-variance test set — a single-source clinical release rather
than a crowd-sourced, multi-source challenge archive — not that PH2-trained models are special.

**More training data doesn't automatically mean better transfer.** HAM10000 has by far the largest
train split (7,010 images) but only the third-best generalization average (0.8671) — worse than
ISIC2018's 0.9067 despite ISIC2018 having 4x fewer training images. Scale alone doesn't predict
transfer quality here; dataset diversity (per the ISIC2018 finding above) looks like the better
predictor, though with only four datasets this is a pattern to note, not a claim to lean on hard.

## Explainability

`scripts/explainability.py` produces two views of what the model attends to, neither requiring a
model change: (1) the spatial attention map `SC_Att_Bridge` already computes internally at its
shallowest skip-connection level, read via a forward hook; (2) Seg-Grad-CAM (Vinogradova et al.
2020) — backpropagate the sum of predicted-foreground probability into the last feature map before
the 1x1 output conv, captured via a forward pre-hook on `model.final`, no model edits needed there
either. Case study: the ISIC2018 model on its own 4 worst and 4 best test images by per-image DSC
(from the [per-image variance](#per-image-variance) section above).

![ISIC2018 explainability grid, worst to best](../ppt_assets/explainability_grid_isic2018.jpg)

*(regenerate the full-resolution PNG with
`python scripts/explainability.py --dataset ISIC2018 --indices 42 359 40 347 123 338 444 501`;
the tracked copy above is a compressed JPEG for repo size)*

### Two distinct failure modes, not one

The four low-DSC cases split cleanly into two different failure patterns, both visible in the
Seg-Grad-CAM column:

- **Over-segmentation from imaging artifacts (#42, DSC 0.051; #359, DSC 0.068).** Both images have a
  large non-lesion, high-salience region in frame — a bright specular reflection off the dermoscope
  contact plate in #42, a bright magenta gauze/backing sheet in #359 — next to a small, comparatively
  low-contrast true lesion. In both cases Grad-CAM lights up the bright artifact, not the small true
  lesion, and the predicted mask follows: a large blob covering the artifact region instead of the
  small ground-truth blob.
- **Under-segmentation from diffuse, low-contrast lesions (#40, DSC 0.078; #347, DSC 0.228).** Here
  the true lesion is a broad area with a gradual, low-contrast boundary. Grad-CAM in both cases
  collapses onto a single small, high-local-contrast fleck inside the true lesion rather than the
  full diffuse region, and the prediction is correspondingly a tiny blob instead of matching the
  true extent.

The four high-DSC cases (0.987–0.989) look qualitatively different from both failure modes: Grad-CAM
lights up the *entire* lesion region fairly uniformly, closely tracking its true boundary, for
lesions that are reasonably high-contrast against the surrounding skin with no competing bright
artifact in frame. That's a common thread across all four good examples, not just a property of one.

### Spatial attention vs. Seg-Grad-CAM

The `SC_Att_Bridge` spatial attention map (4th column) is visibly less informative than Grad-CAM
(5th column) across every image in the grid — it shows a faint, mostly low-level pattern with a
grid-like artifact near the image borders (a `Spatial_Att_Bridge` conv with `padding=9, dilation=3`
on a 7x7 kernel, at H/2 resolution upsampled 2x, is the likely source), rather than object-level
localization. That's a real characterization of the architecture, not a bug in the extraction: the
shallowest attention gate operates on low-level encoder features before any PVM/Mamba layer has run,
so there's no reason to expect it to already encode "where the lesion is" the way a feature map two
stages deeper (Grad-CAM's target layer) does.

### Caveat

Eight images (four failure, four success) is enough to characterize *these* failure modes, not to
claim they're exhaustive or representative of ISIC2018's full 16-below-0.5-DSC population — the
other 12 low-Dice test images aren't inspected here. Both identified patterns (bright artifact
confusion, diffuse-boundary under-segmentation) are plausible root causes for ISIC2018's higher
per-image variance relative to the other three datasets, but confirming that at scale would need
running this same analysis over all 16 outliers, which `scripts/explainability.py --indices` supports
directly.

## Reproducing

```bash
# ISIC2017
python scripts/download_isic.py --dataset ISIC2017
python dataprepare/Prepare_ISIC2017.py

# ISIC2018
python scripts/download_isic.py --dataset ISIC2018
python dataprepare/Prepare_ISIC2018.py

# PH2
python scripts/download_ph2.py      # requires a Kaggle API token at ~/.kaggle/kaggle.json
python dataprepare/Prepare_PH2.py

# HAM10000
python scripts/download_ham10000.py
python dataprepare/Prepare_HAM10000.py

python -m pytest tests/ -q          # kernel-vs-oracle equivalence, applies to all datasets
python train.py                     # set configs/config_setting.py: datasets = 'ISIC2017' | 'ISIC2018' | 'PH2' | 'HAM10000'
```
