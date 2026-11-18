"""Training configuration for the 3D PVM extension.

Mirrors configs/config_setting.py's structure and hyperparameter conventions
(AdamW lr 1e-3 / wd 1e-2, CosineAnnealingLR, seed 42, batch 8) so PVM-3D and
StandardUNet3D train under identical conditions -- same split, seed,
optimizer, schedule -- per docs/3d_extension_plan.md's "Next steps" #2. Switch
`network` between the two to run the like-for-like comparison the plan calls
for; nothing else in this file needs to change between the two runs.
"""

import os
from datetime import datetime

from utils3d import DiceCELoss3D

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class setting_config:
    # 'UltraLight_VM_UNet_3D' (proposed) or 'StandardUNet3D' (baseline) --
    # everything else below is shared so the two runs are a fair comparison.
    network = 'UltraLight_VM_UNet_3D'

    _MODEL_CLIST = {
        'UltraLight_VM_UNet_3D': [8, 16, 24, 32, 48],
        'StandardUNet3D': [16, 32, 64, 128, 256],
    }
    model_config = {
        'num_classes': 3,       # bg / anterior / posterior, MSD Task04 convention
        'input_channels': 1,    # single-channel MRI
        'c_list': _MODEL_CLIST[network],
        'split_att': 'fc',      # UltraLight_VM_UNet_3D only; ignored by StandardUNet3D
        'bridge': True,         # UltraLight_VM_UNet_3D only; ignored by StandardUNet3D
    }

    datasets = 'Hippocampus'
    data_path = os.path.join(_HERE, 'data', 'Task04_Hippocampus') + os.sep

    patch_size = 64
    criterion = DiceCELoss3D(num_classes=model_config['num_classes'])

    distributed = False
    local_rank = -1
    num_workers = 0
    seed = 42
    world_size = None
    rank = None
    amp = False
    batch_size = 8        # matches the 2D config's paper hyperparameter -- kept
                            # identical across both models for a fair comparison
    epochs = 300

    # dataprepare/Prepare_Hippocampus.py writes 203 train / 30 val / 30 test.
    # val/test loaders use drop_last=True, so a non-divisor silently discards
    # volumes -- train3d.py asserts this rather than trusting it (see
    # train.py's identical assertion for the 2D datasets).
    val_batch_size = 10
    test_batch_size = 1   # keep at 1: engine3d.test_one_epoch's save_mid_slices
                            # assumes a batch of one, same convention as utils.save_imgs

    work_dir = os.path.join(_HERE, 'results', '') + network + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20
    save_interval = 100

    opt = 'AdamW'
    lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 1e-2
    amsgrad = False

    sch = 'CosineAnnealingLR'
    T_max = 50
    eta_min = 0.00001
    last_epoch = -1
