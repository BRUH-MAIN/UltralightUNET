# Run files for scripts/cluster/pack_runs.py

One file per 72 h block of docs/3d_journal_roadmap.md §7. Dataset ids:
4 Hippocampus · 27 ACDC · 220 KiTS2023 · 219 AMOS2022 (task 2) · 3 LiTS (MSD Liver) · 137 BraTS2021 · 17 BTCV.

`weight=2` marks runs that saturate a GPU on their own (nnU-Net (org.), SegMamba). External baselines that need
a divisible patch use `plans=nnUNetPlans_div32` / `div16` (scripts/cluster/make_plans_variant.py).

Order inside a file = priority. A block that ends early simply leaves the tail for the next block: the packer
skips finished runs and resumes partial ones. P1/P2 items are in separate files so they are dropped first.

`PVM3DFinal_*` trainers only exist meaningfully after **Gate B** (end of S2): set `FINAL_DESIGN` in
pvm3d/nnunet_trainers/trainers_pvm3d.py, commit, and only then start S3.
