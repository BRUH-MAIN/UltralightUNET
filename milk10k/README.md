# MILK10k — Lightweight Multimodal Mamba Classifier (Phase 2)

The semester project's own contribution, built on the Phase-1 replication.

**Idea in one line:** the paper only ever used its parallel Vision Mamba block (the **PVM Layer**)
for *segmentation*; we reuse that exact block to build a **lightweight multimodal classifier** for
skin-lesion diagnosis on **MILK10k**, whose defining feature is *paired clinical + dermatoscopic*
images.

This is new on two axes the UltraLight VM-UNet authors never touched — **classification** and
**multimodal fusion** — while literally importing the component we already validated
(`from models.UltraLight_VM_UNet import PVMLayer`).

## Relationship to Phase 1

Phase 1 (the segmentation replication) is frozen on the **`replication`** branch and tagged
**`v1.0-replication`** (DSC 0.9030 vs the paper's 0.9091). This work lives on `main` and *reuses*
Phase 1 — it does not replace it. The classifier's encoder is UltraLight VM-UNet's contracting path
with a classification head instead of the decoder.

## The dataset, and what makes it hard

MILK10k: **5,240 lesions, 10,480 images** (each lesion has one clinical close-up + one
dermatoscopic image), **11 diagnosis classes**, scored by **macro-F1**. CC-BY-NC, downloadable
directly from ISIC (no registration).

Two facts shaped the whole design:

- **Test ground truth is withheld** (live challenge), so all reportable numbers come from our own
  split of the 5,240 training lesions. The official 958-image test set is leaderboard-only.
- **Severe imbalance — 280×.** BCC is 48% of lesions; MAL_OTH is 9 lesions (0.2%). macro-F1 averages
  per-class F1 equally, so the rare classes dominate the metric. Expect low macro-F1 even for a good
  model, and read the **per-class** breakdown, not just the average.

| class | lesions | | class | lesions |
|---|---|---|---|---|
| BCC | 2522 | | AKIEC | 303 |
| NV | 746 | | DF / INF / VASC / BEN_OTH | 44–52 |
| BKL | 544 | | **MAL_OTH** | **9** |
| SCCKA | 473 | | MEL | 450 |

## Two correctness rules (Phase-1 lessons applied)

1. **Lesion-level split.** Each lesion has two images; splitting by *image* would leak a lesion's
   clinical view into train and its dermatoscopic view into test. `prepare_milk10k.py` splits by
   `lesion_id`.
2. **Class-stratified, seeded.** macro-F1 is undefined for a class absent from the test split, so
   every class must appear in every split. Stratified `SPLIT_SEED = 42`; the rarest class lands
   ~6/2/1. Asserted, not trusted (`milk10k/tests/`).

The split is captured in **`milk10k/milk10k_manifest.csv`** (5,240 rows, ~389 KB), which is
**version-controlled** — the split is reproducible from git without re-hosting the images. The
images themselves re-download from ISIC on any machine.

## Setup & run (from the repo root)

```bash
# environment: reuse the blackwell/ cu128 venv, or create one:
#   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
#   uv pip install pandas scikit-learn timm einops pillow thop tqdm pytest

python milk10k/scripts/download_milk10k.py        # 314 MB from ISIC S3
python milk10k/dataprepare/prepare_milk10k.py     # -> milk10k/milk10k_manifest.csv (already tracked)
python -m pytest milk10k/tests -q                 # split integrity + model sanity

python -m milk10k.train                           # trains per configs/config.py:modality
python -m milk10k.test --weights milk10k/results/<run>/checkpoints/best.pth --modality derm
```

Everything runs as a module from the repo root (`python -m milk10k.*`) so the shared `models/`
package resolves. Data paths in the config are absolute, so cwd does not matter.

## Model

`milk10k/pvm_classifier.py` — `PVMEncoder` (the reused contracting path) + a head. Three modes via
`config.modality`:

| mode | params | what |
|---|---|---|
| `derm` / `clin` | **0.016 M** | single-modality (milestone 1 + ablation baseline) |
| `both` + `concat` | 0.032 M | late fusion of pooled features |
| `both` + `cross_mamba` | 0.039 M | a shared PVM Layer scans across both views' tokens |

Even the multimodal model is **smaller than the 0.049 M segmentation model** — the lightweight
claim carries into the new task.

## Milestones

1. **Feasibility probe** — done. Class histogram, pairing, split integrity all verified above.
2. **Single-modality PVM classifier** (dermatoscopic) vs a standard lightweight baseline, on
   params · GFLOPs · macro-F1.
3. **Multimodal fusion** — report the lift over the best single modality.
4. **Fairness bonus** — `test.py` breaks macro-F1 down by **skin tone 0–5** (honest caveat: tones 0–1
   have too few test lesions to read much into).

## Imbalance handling

`config.loss`: `weighted_ce` (balanced inverse-frequency CE, default) or `focal`. Weights are
computed from the **train** split only (`engine.compute_class_weights`).

## Deferred (decide from results, not upfront)

Which fusion wins (`concat` vs `cross_mamba`), whether the from-scratch lightweight model needs a
pretrained encoder, and the 48-class ISIC-Dx hierarchy — all downstream of milestone 2.
