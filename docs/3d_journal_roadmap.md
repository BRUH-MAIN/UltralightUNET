# PVM-3D → top-tier journal (MedIA / IEEE TMI): experiment & dataset roadmap

**Status (2026-11-13): Phase 0 (engineering) is done and tested on the laptop. Next: register for the gated
datasets, run `scripts/cluster/runs/laptop_pre.txt`, then cluster block S1.** How to run things:
[`scripts/cluster/README.md`](../scripts/cluster/README.md). The design history behind PVM-3D:
[`docs/3d_extension_plan.md`](3d_extension_plan.md).

## Context

The repo has a working 3D port of UltraLight VM-UNet's PVM layer (`models/UltraLight_VM_UNet_3D.py`, 64,804
params), trained once on MSD Hippocampus: **0.8531 DSC vs 0.8972** for a hand-built 3D U-Net
(`models/StandardUNet3D.py`, 5.6M params), with a size-linked over-segmentation bias
(`scripts/failure_analysis_3d.py`). Goal: a MedIA/TMI submission, with **all runs finished in under a month**.

Compute (confirmed on the cluster on 2026-11-18):
- A shared SLURM partition with a 3-day job cap.
- 2× RTX 6000 Ada 48 GB on the master node, and 2× A100 80 GB on each of two compute nodes.
- Jobs queue alongside other users'.
- The "block" in §7 means about 72 GPU-hours of work, which can be split across parallel jobs. More than one GPU
  at a time would shorten the calendar a lot; how many we may use concurrently is still open.

Novelty was re-checked on 2026-11-12. There is still no published 3D use of PVM's shared-weight 4-way channel split.
SegMaFormer (arXiv 2603.22002), Mamba-SAM (2602.00650), MobileViM, LightM-UNet and SegResMamba all use different
mechanisms. Re-check once more right before submission.

The reviewer bar is set by **nnU-Net Revisited** (Isensee et al., MICCAI 2024, arXiv 2404.09556). It names four
common validation failures, and this plan is built to avoid each one:
- **P1, confounders:** gains that come from residual blocks or the training recipe rather than the new module.
- **P2, weak baselines.**
- **P3, too few or unsuitable datasets.** The median paper uses 3.
- **P4, inconsistent reporting:** TTA, ensembling, post-processing, cherry-picked classes.

That paper also published 5-fold DSC for about 15 strong baselines on 6 datasets, using nnU-Net v2's own default
splits (`pvm3d/reference/nnunet_revisited_dsc.csv`). **Because PVM-3D runs inside nnU-Net v2 on the same datasets,
those numbers can be compared directly.** This saves roughly 2,000+ GPU-hours of heavy-baseline retraining, and it
is the only way the benchmark fits in a month.

---

## 0. Phase-0 results and what they change

**Built and tested** (119 tests pass; the 4 external-baseline tests skip until that code is fetched):

| Piece | Where | Verified by |
|---|---|---|
| Generalized PVM-3D, built from nnU-Net plans | `pvm3d/network.py` | Legacy config reproduces **exactly 64,804 params and identical outputs** from converted weights; anisotropic strides; deep-supervision scales match nnU-Net's; every ablation variant back-propagates into every parameter |
| nnU-Net trainers: sizes, 18 ablations, Gate-B final design, data-fraction, CT-only, smoke | `pvm3d/nnunet_trainers/trainers_pvm3d.py` | End-to-end Hippocampus smoke runs (train → sliding-window validation); the config guard refuses to resume a run whose architecture or epoch count changed |
| Parameter-matched tiny nnU-Net (within 0.02–6% of target) | `trainers_baselines.py` | `tests/test_pvm3d_network.py` |
| LightM-UNet, SlimUNETR, SegFormer3D, SegMamba wrappers, each with its authors' recipe | `trainers_external.py` | **Not yet executed** (see risks) |
| Per-case DSC / NSD@2mm / HD95, region-aware | `scripts/eval3d.py` | Matches nnU-Net's own evaluator exactly (labels and regions, anisotropic spacing); closed-form geometry tests |
| Fold aggregation, Wilcoxon + Holm, paired bootstrap CIs, rank stability, LaTeX tables, Pareto figure | `scripts/stats3d.py` | Synthetic data with a planted effect: detects it, does not flag the null |
| MACs counted with Mamba included | `scripts/flops3d.py` | mamba_ssm and the PyTorch oracle give identical totals |
| Latency / VRAM / memory-time / train-step benchmark | `scripts/bench_infer3d.py` | Runs on GPU and CPU |
| Cluster: env, setup, dataset conversion + verify, plans variants, baseline fetch, run packer, per-block run files | `scripts/cluster/` | Packer tested with concurrent runs, completion, status, and hard-stop with no orphaned processes |

**Findings that change the plan**

1. **thop's FLOP numbers are unusable in both directions.**
   - On Hippocampus plans it undercounts PVM-3D-S by about 3× (0.061 vs 0.178 GMACs) once PVM sits at
     higher-resolution stages.
   - It overcounts nnU-Net's U-Net by about 8× (148 vs 19 GMACs).
   - The legacy model's 0.26 G was nearly right only because its Mamba stages see tiny grids.
   - Every efficiency number in the paper comes from `flops3d.py`. Say this in the methods section, since the
     2D paper's 0.060 GFLOPs has the same blind spot.
2. **"Fewer FLOPs" does not automatically mean faster.**
   - Hippocampus GPU inference: PVM-3D-S 26 ms vs nnU-Net 23 ms. Fixed overheads dominate tiny volumes.
   - The training step is 3.4× cheaper (30 vs 102 ms/it; 430 vs 986 MB).
   - On CPU both take about 1.0 s, because the pure-PyTorch Mamba is a correctness oracle, not a fast kernel.
   - H6 ("CPU-viable") must be tested on the large CT volumes, and it may need an optimized CPU selective scan.
     Otherwise phrase the claim around memory and training cost.
3. **Model size depends on the plans.**
   - nnU-Net plans Hippocampus with 4 stages (40×56×40 patch), so PVM-3D-S has 32.8K params there; the legacy
     design gives 25.7K.
   - CT datasets get 6 stages and therefore larger models.
   - Report parameters per dataset, not as one number.
4. **The compute budget is tighter than first estimated.**
   - The run files hold about 360 P0+P1 runs. At about 8 run-hours each and 4 runs per GPU, that is about
     10 blocks, against 9 available.
   - The real per-run time and concurrency get measured in S1.
   - The first lever is pre-wired: `ABLATION_EPOCHS` (trainers_pvm3d.py), 1000 → 500 before S2. It halves
     about 70 ablation runs, with ablations compared at equal schedule.
   - Further levers are listed in §7.
5. **nnU-Net 2.8.1 bug:** its `nnUNet_extTrainer` lookup can crash on a second lookup in one process. Trainers are
   therefore registered by symlink (`scripts/cluster/install_trainers.py`); our tools resolve them directly.
6. **The AMOS CT/MRI id rule is wrong at the boundary.**
   - The official metadata (zenodo 7262581) lists `amos_id` 500 twice: once as a CT (Optima CT540) and once as an
     MR (Ingenia).
   - Modality is now read from the images (`convert_datasets.py verify` writes `modality.json`; CT = Hounsfield
     values below −200), and E8 uses that map.
7. **H4 (Mamba is load-bearing) is the make-or-break hypothesis.**
   - nnU-Net Revisited found that U-Mamba's gains came from its residual blocks, not its Mamba layers.
   - Finding 8 gives a concrete reason PVM's Mamba may be unused here too.
   - Whether it is gets tested first (S1_mamba: retraining ablations plus inference-time knockout), before the
     benchmark budget is spent.
8. **A likely mechanism: the inherited init cripples Mamba** (verified on `pvm3d/network.py`).
   - UltraLight's `_init_weights` (kept for fidelity to the 2D original) re-initialises Mamba's projections
     (std 0.02) and sets `dt_proj.bias` to 0, ignoring mamba_ssm's `_no_reinit` marker.
   - At init this gives a step size of softplus(0) = 0.693 in every channel. **The slowest SSM mode then remembers
     about 1.4 tokens**, versus a median of 60 (28–258) with mamba_ssm's own init.
   - Mamba's output also starts at about 1e-3 of the skip path, versus about 3e-2.
   - So every PVM model built this way, including probably the published 2D one, starts with Mamba as a 1-token
     local filter.
   - Ablation **A9** tests whether fixing this makes Mamba matter:
     - `_S_nativeInit`
     - `_S_noInnerSkip` (no bypass, so information must pass through Mamba)
     - the combination of both.
   - `scripts/mamba_knockout.py` reports the knockout drop, the output ratio, and the learned memory horizon for
     every trained PVM run.
9. **DD-PVM novelty is weak.**
    - VMamba's 2D cross-scan (2 axis orders × 2 directions) carries straight over to 3D as a zero-parameter
      per-branch scan (our `cross` mode), and SegMamba's tri-orientation is published prior art.
    - DD-PVM (3 distinct axis orders + 1 reversal) is a variant, so it is an ablation axis (A4, now including
      `_S_cross`), not a headline contribution.
    - If A9 succeeds, the init/bypass mechanism is the stronger contribution.
10. **The cluster is SLURM, not a live 72 h block** (2026-11-18). Runs go through `scripts/cluster/slurm/*.sbatch`.
   - Several jobs can share one run file safely (per-run locks in `pack_runs.py`), so a second or third GPU
     directly shortens the schedule.
   - The cluster environment is torch 2.10/cu128 with prebuilt mamba wheels; the laptop stays on 2.13.

---

## 1. Audit of the original 3D implementation

**Kept unchanged (as the paper's "same mechanism" story):**
- `PVMLayer`: `PVMLayer3D` with default settings is numerically identical to it.
- The UltraLight topology, `SC_Att_Bridge`, and the weight init.

**Fixed:**

| # | Gap in the original pipeline | Resolution |
|---|---|---|
| G1 | Fixed 64³ pad/crop, no resampling, no sliding window | nnU-Net v2 pipeline. The old crop lost 0 foreground voxels on Hippocampus, so the preliminary number stands as preliminary. |
| G2 | Hard-wired 4 isotropic pools, 2 PVM stages, no deep supervision | `PVMUNet3D`: plans-derived strides and kernels, configurable PVM placement, deep-supervision heads |
| G3 | Hand-built, untuned baseline | nnU-Net + Revisited table + lightweight baselines in the same framework |
| G4 | 2D recipe (`T_max=50` cosine over 300 epochs, so the LR cycles) | nnU-Net recipe. The 2D recipe becomes ablation A7 (`_S_AdamWCos50`). |
| G5 | 1 seed, 30 test volumes, DSC only | 5-fold CV, DSC + NSD + HD95, paired tests |
| G6 | thop FLOPs | `flops3d.py` (finding 1) |
| G7 | No evidence that Mamba matters | A1 (`_S_noMamba`, `_S_convMixer`) + parameter-matched tiny nnU-Net |

The original pipeline (`train3d.py`, `loader3d.py`, `engine3d.py`) is untouched and stays as the
preliminary-result pipeline.

---

## 2. Thesis, contributions, hypotheses

**Thesis.** PVM's shared-weight parallel channel split carries over to 3D. With one 3D-specific extension, it gives
the best accuracy-per-parameter Pareto front among 3D segmenters.

**Method contribution C2: direction-diverse PVM (DD-PVM, `scan='dd'`).**
- Each of the 4 existing branches scans the 3D token grid in a different order: W-fastest, H-fastest, D-fastest,
  and reversed.
- It costs **zero parameters** (tested) and zero FLOPs beyond a gather.
- It brings SegMamba-style tri-orientation into PVM's channel parallelism.

**Hypotheses**
- **H1:** PVM-3D-{T,S,B} dominates every <5M-parameter baseline on DSC per parameter, across 7 datasets.
- **H2:** The shared 4-way split matches or beats a single full-width Mamba (A2/A3).
- **H3:** DD-PVM beats raster and is at least as good as tri-orientation at ⅓ of its scan cost (A4).
- **H4:** Mamba is load-bearing (A1).
- **H5:** The light model degrades no worse than nnU-Net with less data or under domain shift (E7/E8).
- **H6:** Deployability. Revised by finding 2: test it on the CT volumes before claiming it.

**Decision gates**
- **Gate A (end of S1):** our nnU-Net (org.) must be within ±0.5 DSC of Revisited on ACDC (5-fold) and KiTS
  (fold 0). If it isn't, compare only against models we train, and spend the buffer on nnU-Net reruns.
- **Gate B (end of S2):** set `FINAL_DESIGN` in `trainers_pvm3d.py` to the winning ablation settings, commit, and
  never edit it again. The `nnUNetTrainerPVM3DFinal_*`, tiny-U-Net-matched, data-fraction and CT-only trainers all
  derive from it.
  - If No-Mamba ≈ PVM (H4 fails), reframe the paper around efficiency and the Pareto front.
  - If DD-PVM fails, drop C2.

---

## 3. Datasets

**Tier A.** 5-fold CV on nnU-Net's default splits, `3d_fullres`. Converted by
`scripts/cluster/convert_datasets.py`, which uses nnU-Net's own converters so case identifiers (and therefore
splits) match Revisited's.

| ID | Dataset | Modality | n | Targets | Why it is in | Access |
|---|---|---|---|---|---|---|
| 27 | **ACDC** | cine MRI, anisotropic | 200 | RV, MYO, LV | Revisited "recommended"; anisotropy; nnFormer/UNETR++ comparability | creatis.insa-lyon.fr (registration). The converter writes patient-grouped splits. |
| 220 | **KiTS2023** | CT | 489 | kidney+masses / masses / tumor (regions) | Revisited "recommended", least saturated | `github.com/neheller/kits23` download script |
| 219 | **AMOS2022 task 2** | CT + MRI | 360 | 15 organs | Revisited "recommended"; long-range multi-organ context; CT/MRI split for E8 | zenodo.org/record/7262581 |
| 3 | **LiTS** (= MSD Task03 Liver) | CT | 131 | liver, tumor | Revisited "middle ground"; LightM-UNet's own 3D benchmark | MSD (HF mirror, as for Hippocampus) |
| 137 | **BraTS2021** | 4-channel MRI | 1251 | WT / TC / ET (regions) | Saturated, but reviewers expect it; multi-channel input | Kaggle mirror of the official training set (CC BY 4.0; `download_brats21_kaggle.sh`, integrity-checked). Cite Baid 2021, Menze 2015, Bakas 2017 and TCIA DOI 10.7937/jc8x-9874 |
| 17 | **BTCV** | CT | 30 | 13 organs | Noisy per Revisited, but ubiquitous; report with that caveat | Synapse syn3193805 (registration) |
| 4 | **MSD Hippocampus** | MRI | 260 | anterior, posterior | Continuity with existing work; tiny structures | Done (converted and planned on the laptop) |

**Tier B: secondary analyses.**
- AMOS CT→MRI (E8; no extra data).
- BraTS 2023 SSA/PED zero-shot (P2; Synapse).
- TorchIO corruption sweep (inference only).
- FLARE24 laptop track (stretch).

**Before S1:**
- **Start the Synapse, ACDC and AMOS registrations now.**
- Budget about 300–400 GB of cluster scratch.
- Run preprocessing as CPU jobs outside GPU blocks.
- Then run `convert_datasets.py verify`. It checks case counts and the **AMOS CT/MRI id threshold (300/60)** that
  E8 relies on, which has not been checked against the real data yet.

---

## 4. Baselines

**Reused from Revisited** (same framework and splits; no retraining):
- Models: nnU-Net (org.), ResEnc M/L/XL, MedNeXt L k3/k5, STU-Net S/B/L, SwinUNETR(V2), nnFormer, CoTr, U-Mamba
  Bot/Enc, No-Mamba Base.
- Source: `pvm3d/reference/nnunet_revisited_dsc.csv`, with provenance in its README.
- Only fold-mean comparisons are possible for these models, since their per-case predictions are not published.

**Trained by us:**

| Priority | Model | Trainer | Recipe |
|---|---|---|---|
| P0 | Tiny nnU-Net matched to PVM-3D-{T,S,B} | `nnUNetTrainerTinyUNet_match{T,S,B}` | nnU-Net |
| P0 | LightM-UNet | `nnUNetTrainerLightMUNet` | Authors': Adam 1e-4, poly, fp32 |
| P0 | SlimUNETR | `nnUNetTrainerSlimUNETR` + `-p nnUNetPlans_div32` | Authors': AdamW 1e-3, warmup + cosine |
| P0 | nnU-Net (org.) on Hippocampus + Gate A | `nnUNetTrainer` | nnU-Net |
| P1 | SegFormer3D | `nnUNetTrainerSegFormer3D` + `div32` | Authors': AdamW 1e-4, warmup + cosine |
| P1 | SegMamba (ACDC 5-fold, KiTS fold 0) | `nnUNetTrainerSegMamba` + `div16`, **own venv** | Authors' = nnU-Net's |
| P2 | MobileViM (if code is released), UNETR++ | — | — |

External code is fetched at pinned commits (`scripts/cluster/fetch_baselines.sh`), never vendored: SegFormer3D is
GPL-3.0 and SegMamba has no license file. SegMamba's mamba fork installs as `mamba_ssm`, so it gets its own venv
to avoid replacing the kernel PVM-3D is tested against.

---

## 5. Protocol, metrics, statistics

**Protocol.**
- nnU-Net defaults: 1000 × 250 iterations, SGD + poly, deep supervision, mirroring TTA at validation, no
  post-processing, no ensembling.
- External baselines keep only their authors' optimizer settings ("closest configuration").
- Every run records its config (`pvm_config.json`), trainer, plans and GPU.

**Accuracy** (`eval3d.py`):
- DSC (primary), NSD@2mm, HD95 (mm), per class/region and per case.
- Fold score = nnU-Net's `foreground_mean` definition.
- Cases where exactly one of GT/prediction is empty are counted and reported.

**Efficiency** (`flops3d.py`, `bench_infer3d.py`):
- Inference params (deep-supervision heads excluded for every model alike) and GMACs at the plans patch.
- Sliding-window latency, peak VRAM and memory-time area on the A100/RTX 6000, the RTX 5060 laptop and CPU.
- Train-step ms/it and VRAM.
- GPU-hours per run, from the nnU-Net logs.

**Statistics** (`stats3d.py`):
- Mean ± SD over folds.
- Per-case Wilcoxon signed-rank tests, Holm-corrected within dataset × metric.
- Paired bootstrap 95% CIs (10k resamples).
- Bootstrap rank stability.

---

## 6. Experiment matrix (trainer names in `scripts/cluster/runs/`)

**E1. Infrastructure (P0).**
- Gate A parity: nnU-Net ACDC 5-fold, KiTS fold 0.
- Hippocampus: nnU-Net, `PVM3D_Legacy`, `PVM3D_S`, 5-fold each. Does the recipe alone close part of the
  4.4-point gap?

**E2. Width pilot (P0).** `PVM3D_{T,S,B}` on ACDC fold 0. Base widths 8/16/32, giving the UltraLight progression
×{1,2,4}.

**E3. Ablations (P0).** All against `PVM3D_S_ref`, at `ABLATION_EPOCHS`. Fold 0 on ACDC, KiTS and AMOS, plus 5-fold
ACDC for A1/A3/A4.

| ID | Trainers (`nnUNetTrainerPVM3D_S_*`) |
|---|---|
| A1 Mamba load-bearing? | `noMamba`, `convMixer` |
| A2 branches | `br1`, `br2`, (ref = 4), `br8` |
| A3 weight sharing | `unshared` |
| A4 scan order | `dd`, `cross` (VMamba-style), `bidir`, `tri`, `random` |
| A5 placement | `enc2dec1` (legacy), `encAll` |
| A6 bridge | `bridgeNone`, `bridgeSpatial`, `bridgeChannel` |
| A7 recipe | `AdamWCos50` (2D paper's), `noDS` |
| A8 SSM | `dstate8`, `dstate32`, `expand1` |
| A9 why is Mamba bypassed? | `nativeInit`, `noInnerSkip`, `nativeInitNoInnerSkip` (finding 8) |

**Decided 2026-11-18: the first block is A1 + A9 + `dd` + Tiny-S**, as `scripts/cluster/runs/S1_mamba.txt`
(49 runs, open data only: Hippocampus 5-fold, LiTS and AMOS fold 0), followed by `mamba_knockout.py` on every PVM run.
Which paper we write (method paper vs. rigorous "the Mamba is bypassed" analysis paper) depends on it. The
original S1 (Gate A parity, Hippocampus nnU-Net, width pilot) follows as soon as ACDC/KiTS are on disk.

**E4. Main benchmark (P0).** `PVM3DFinal_S`, 5-fold × 7 datasets.

**E5. Pareto family.**
- P0: `PVM3DFinal_{T,B}` on ACDC, KiTS and AMOS.
- P2: the other four datasets.

**E6. Lightweight baselines.** §4.

**E7. Data efficiency (P1).** 10/25/50% of training cases, ACDC + KiTS fold 0. Trainers: `PVM3DFinal_S_frac*`,
`nnUNetTrainer_frac*`, `TinyUNet_matchS_frac*`.

**E8. AMOS CT→MRI (P1).** The `*_CTOnly` trainers, scored with `eval3d.py --modality-split amos`.

**E9. Analysis (P1, inference only).**
- Size-stratified failures on all datasets (port of `failure_analysis_3d.py`).
- 3D attention and Seg-Grad-CAM (port of `explainability.py`).
- TorchIO corruption sweep.

**E10. Deployment (P0, inference only).** `bench_infer3d.py` on 3 devices; CPU INT8 via the oracle swap.

---

## 7. Compute budget and schedule

| Block | Run file | Contents |
|---|---|---|
| Pre-S1 (laptop) | `laptop_pre.txt` | Hippocampus `PVM3D_Legacy` + `PVM3D_S`, 5-fold. About 12 s/epoch, so about 3.3 h per fold. Registrations, downloads and preprocessing in parallel. |
| S1-M | `S1_mamba.txt` | **First.** Is Mamba load-bearing? A1 + A9 + `dd` + Tiny-S on Hippocampus/LiTS/AMOS, then `mamba_knockout.py`. Also the throughput measurement (concurrency, `ABLATION_EPOCHS`). **Decides the paper's framing.** |
| S1 | `S1.txt` | Gate A; E2; Hippocampus nnU-Net; LightM-UNet on ACDC/Hippocampus (when ACDC/KiTS are available). **Gate A.** |
| S2 | `S2.txt` | A1–A5 + A7, fold 0 × 3 datasets. **Gate B: set `FINAL_DESIGN`.** |
| S3 | `S3.txt` | A6, A8; 5-fold A1/A3/A4; E4 on Hippocampus, ACDC, BTCV |
| S4 | `S4.txt` | E4 on KiTS, AMOS, LiTS, BraTS |
| S5 | `S5.txt` | E5 Pareto; Tiny-S on ACDC/Hippocampus |
| S6 | `S6.txt` | E6 P0 baselines on every dataset (the largest block; spills into S7 if throughput is low) |
| S7 | `S7.txt` | P1: SegFormer3D, SegMamba, E7, E8 |
| S8 | — | Inference-only work (E9, E10) on finished checkpoints; reruns |
| S9 | `P2.txt` | Buffer, then P2 |

**Levers if S1 shows fewer than about 4 concurrent runs or more than about 8 h per run, in order:**
1. `ABLATION_EPOCHS = 500`.
2. Drop P2.
3. Cut SegMamba to ACDC.
4. Run SegFormer3D on the 4 recommended datasets only.
5. Run E7 on ACDC only.
6. Run Tiny-T/B on ACDC only.

Each lever is a one-line change in a run file or `trainers_pvm3d.py`.

---

## 8. Paper output map

| Output | Contents | Produced by |
|---|---|---|
| Table 1 | 7 datasets × about 25 models, DSC/NSD (5-fold) | `stats3d.py` → `table_main.tex` |
| Table 2 | Efficiency: params, GMACs, VRAM, latency on 3 devices, GPU-hours | `flops3d.py`, `bench_infer3d.py` |
| Table 3 | Ablations A1–A8 with paired tests | `stats3d.py --ref nnUNetTrainerPVM3D_S_ref` |
| Fig. 1 | Architecture, including the DD-PVM scan diagram | — |
| Fig. 2 | Pareto front | `stats3d.py --efficiency` |
| Figs. 3–5 | Data efficiency, robustness / CT→MRI, qualitative + size bias + attention | E7–E9 |

Release: code, trainers, plans, splits, checkpoints, and per-case CSVs.

## 9. Risks and fallbacks

- **External baselines have not been executed yet.** Their wrappers are written against the authors' code at the
  pinned commits, but running that code was out of scope for the laptop session. On the cluster, run
  `fetch_baselines.sh`, then `pytest tests/test_external_baselines.py`, **before S1**.
- **AMOS modality.** Now derived from image intensities, because the id rule is ambiguous at 500 (finding 6).
  `verify` fails loudly unless it finds 300 CT / 60 MRI.
- **NFS data-loading speed is untested** (there is no scratch filesystem). The first S1 job's s/epoch shows whether
  it is a bottleneck.
- **PVM-3D-S more than 3 DSC below nnU-Net.** Lead with the Pareto front; B is the accuracy anchor.
- **Kernel or AMP issues on a new GPU.** `setup_env.sh` runs the GPU test suite on the node.
- **Gate A fails.** Spend S9 on nnU-Net/ResEnc-M reruns.
- **CPU latency claim (H6).** Finding 2: needs an optimized CPU scan, or rephrase the claim.
- **Registration delays.** Do Hippocampus/ACDC/LiTS/KiTS/AMOS first; BraTS/BTCV (Synapse) last.

## 10. Next steps

1. Register for Synapse (BraTS21, BTCV) and ACDC **today**. AMOS is open (Zenodo).
2. Laptop: `python scripts/cluster/pack_runs.py scripts/cluster/runs/laptop_pre.txt --capacity 2`
   (about 20 h; the laptop has about 1.6 GB of disk free, so keep the large datasets on the cluster).
3. Cluster, per `scripts/cluster/README.md`:
   - Clone, run `setup_env.sh`, submit `gpu_check.sbatch` on both GPU types.
   - `download_open_data.sh` (Hippocampus, LiTS, AMOS), then convert, `preprocess.sbatch`, `verify`.
   - With Rohan's go-ahead: `fetch_baselines.sh` + `pytest tests/test_external_baselines.py`, as a job.
4. First block: `sbatch scripts/cluster/slurm/train.sbatch scripts/cluster/runs/S1_mamba.txt 4`, then S1 once
   ACDC/KiTS are available (`sbatch ... runs/S1.txt 4`).
   - Runs whose dataset isn't converted and planned yet show as `nodata` and are skipped.
   - Resubmitting after ACDC/KiTS arrive picks them up. No special run file is needed.
