# Published reference numbers

`nnunet_revisited_dsc.csv`: 5-fold cross-validation DSC (%, mean over classes/regions and folds) from
Isensee et al., *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation*,
MICCAI 2024 (arXiv:2404.09556), as tabulated in the nnU-Net repository's
`documentation/resenc_presets.md`. VRAM (GB, nvidia-smi) and RT (training hours, one fold, A100 40 GB) are
from the same table. Transcribed 2026-11-13; the rows that appear in both the arXiv v2 HTML table and
`resenc_presets.md` agree to every digit.

These are the "reused baselines" of docs/3d_journal_roadmap.md §4. They are directly comparable to our runs
**only if** our datasets use the same case identifiers (so nnU-Net's seeded default split is the same) and the
same nnU-Net version semantics -- which Gate A (our own nnU-Net (org.) rerun on ACDC + KiTS fold 0, within
±0.5 DSC) checks. Per-case predictions are not published, so paired tests are only possible against models we
train ourselves; reused rows get fold-mean comparisons only.

Before citing in the paper, re-check every number against the published PDF.
