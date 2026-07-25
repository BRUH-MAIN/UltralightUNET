"""Training configuration for the MILK10k lightweight Mamba classifier.

Phase-2 novelty work. Deliberately mirrors the Phase-1 config style (a plain
class of attributes) so the surrounding train/engine code feels familiar, but the
task is classification, not segmentation.

Paths resolve relative to this file so the notebook works from any cwd.
"""

import os

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # milk10k/

# canonical class order == MILK10k GroundTruth column order
CLASSES = ['AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF', 'INF', 'MAL_OTH', 'MEL', 'NV', 'SCCKA', 'VASC']


class config:
    # ---- data
    data_root = os.path.join(_HERE, 'data')
    # the split manifest is version-controlled (it defines the reproducible split),
    # so it lives at the package root, outside the git-ignored data/ directory.
    manifest = os.path.join(_HERE, 'milk10k_manifest.csv')
    images_dir = os.path.join(_HERE, 'data', 'MILK10k_Training_Images')
    classes = CLASSES
    num_classes = len(CLASSES)

    # ---- input.  Modality selects what the model sees:
    #   'derm'  -> dermatoscopic only  (milestone 1, the single-modality baseline)
    #   'clin'  -> clinical only
    #   'both'  -> the paired multimodal model (the contribution)
    modality = 'derm'
    input_size = 256          # square RGB, matches the Phase-1 pipeline
    input_channels = 3

    # ---- imbalance.  The dataset runs 280x (BCC 48% .. MAL_OTH 0.2%), so macro-F1
    # is dominated by the rare classes. Balanced inverse-frequency CE is the default;
    # 'focal' is the experiment.
    loss = 'weighted_ce'      # 'weighted_ce' | 'focal' | 'ce'
    focal_gamma = 2.0
    class_weighting = True    # inverse-frequency weights computed from the TRAIN split

    # ---- optimisation.  AdamW + cosine, same family as Phase 1.
    seed = 42
    batch_size = 32           # classification: no dense output, so a larger batch fits
    epochs = 100
    num_workers = 0           # Windows-safe; raise on Linux/Kaggle
    opt = 'AdamW'
    lr = 1e-3
    weight_decay = 1e-2
    betas = (0.9, 0.999)
    eps = 1e-8
    amsgrad = False          # required by utils.get_optimizer (reused from Phase 1)
    sch = 'CosineAnnealingLR'
    # T_max must equal epochs for a single clean anneal to eta_min at the end.
    # CosineAnnealingLR is a cosine of period 2*T_max, so T_max < epochs makes the
    # LR climb back toward the maximum before the run ends -- e.g. T_max=50 over 100
    # epochs leaves the LR near its peak at epoch 100, so the best checkpoint gets
    # caught mid-cycle with no low-LR fine-tuning phase.
    T_max = epochs
    eta_min = 1e-5
    last_epoch = -1          # required by utils.get_scheduler
    amp = False

    # ---- model
    c_list = [8, 16, 24, 32, 48, 64]   # same channel schedule as UltraLight VM-UNet
    fusion = 'cross_mamba'    # only used when modality == 'both'; design finalised
                              # after the single-modality milestone

    # ---- bookkeeping
    print_interval = 20
    val_interval = 1          # cheap here; validate every epoch
    work_dir = os.path.join(_HERE, 'results', '')  # + run name at train time
    test_weights = ''
