"""Training configuration.

Differences from upstream's configs/config_setting.py:
  * data_path filled in, and resolved relative to this file so the notebook works
    regardless of the directory the kernel happens to start in.
  * val_batch_size / test_batch_size are settings rather than hardcoded 1s (see
    the comment below). Speed only; training is unaffected.

Every hyperparameter that touches the result is untouched: batch 8, 250 epochs,
AdamW lr 1e-3 / wd 1e-2, CosineAnnealingLR T_max=50 eta_min=1e-5, seed 42,
threshold 0.5, amp off, 256x256 input, c_list [8,16,24,32,48,64].
"""

import os

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from utils import *          # BceDiceLoss

from datetime import datetime

class setting_config:
    """
    the config of training setting.
    """
    network = 'UltraLight_VM_UNet' 
    model_config = {
        'num_classes': 1,
        'input_channels': 3,
        'c_list': [8,16,24,32,48,64],
        'split_att': 'fc',
        'bridge': True,
    }

    test_weights = ''

    datasets = 'PH2'
    # PATCH: data_path filled in. The loader concatenates raw strings
    # (path_Data + 'data_train.npy'), so the trailing separator is required.
    if datasets == 'ISIC2017':
        data_path = os.path.join(_HERE, 'data', 'ISIC2017') + os.sep
    elif datasets == 'ISIC2018':
        data_path = os.path.join(_HERE, 'data', 'ISIC2018') + os.sep
    elif datasets == 'HAM10000':
        data_path = os.path.join(_HERE, 'data', 'HAM10000') + os.sep
    elif datasets == 'PH2':
        data_path = os.path.join(_HERE, 'data', 'PH2') + os.sep
    else:
        raise Exception('datasets in not right!')

    criterion = BceDiceLoss()

    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 0
    seed = 42
    world_size = None
    rank = None
    amp = False
    batch_size = 8          # paper hyperparameter -- do not change for a replication run
    epochs = 250

    # Validation / test batch size. These affect SPEED ONLY: no gradients are taken
    # during either pass, so the training trajectory is completely unaffected.
    #
    # val_one_epoch aggregates with np.mean over per-batch losses, and BCELoss
    # reduces over every pixel while DiceLoss averages per sample -- so for
    # EQUAL-sized batches the mean of batch means equals the overall mean.
    # Measured over the 150 val images: batch 30 shifts the reported loss by 6e-5
    # on a loss of ~1.36 (fp32 reduction-order noise) and runs 20x faster. That is
    # ~25% off total wall clock, since validation was 6s of every 23s epoch.
    #
    # The one caveat worth stating: best-checkpoint selection compares val losses,
    # so in a near-exact tie (within 6e-5) a different epoch could win. That has not
    # been observed, and the effect is far below run-to-run variation.
    #
    # MUST divide the split size exactly -- the val/test loaders use drop_last=True,
    # so a non-divisor silently DISCARDS images (batch 8 would evaluate 144 of 150).
    # train.py asserts this rather than trusting it. Per-dataset because each
    # dataset's val split is a different size:
    #   ISIC2017  150  val = 2 * 3 * 5^2  -> 30
    #   ISIC2018  259  val = 7 * 37       -> 37
    #   HAM10000 1002  val = 2 * 3 * 167  -> 167 (only non-trivial divisor;
    #            167 is prime -- 6 batches, same "large divisor, few batches"
    #            convention as the other two). Pair count/split size printed by
    #            Prepare_HAM10000.py at prep time; update this if that changes.
    #   PH2        20  val = 2^2 * 5      -> 10 (largest non-trivial divisor, 2 batches)
    _VAL_BATCH_SIZE = {
        'ISIC2017': 30,
        'ISIC2018': 37,
        'HAM10000': 167,
        'PH2': 10,
    }
    val_batch_size = _VAL_BATCH_SIZE[datasets]
    test_batch_size = 1     # keep at 1: engine.test_one_epoch calls save_imgs, which
                            # does img.squeeze(0) and so assumes a batch of one

    work_dir = os.path.join(_HERE, 'results', '') + network + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20
    # PATCH: val_interval removed. engine.py's val_one_epoch used to gate the
    # full metric set (DSC/IoU/accuracy/sensitivity/specificity) to every
    # val_interval-th epoch; it now logs them every epoch (see engine.py), so
    # this had no remaining reader.
    save_interval = 100
    threshold = 0.5


    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
    if opt == 'Adadelta':
        lr = 0.01 # default: 1.0 – coefficient that scale delta before it is applied to the parameters
        rho = 0.9 # default: 0.9 – coefficient used for computing a running average of squared gradients
        eps = 1e-6 # default: 1e-6 – term added to the denominator to improve numerical stability 
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty) 
    elif opt == 'Adagrad':
        lr = 0.01 # default: 0.01 – learning rate
        lr_decay = 0 # default: 0 – learning rate decay
        eps = 1e-10 # default: 1e-10 – term added to the denominator to improve numerical stability
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Adam':
        lr = 0.001 # default: 1e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability 
        weight_decay = 0.0001 # default: 0 – weight decay (L2 penalty) 
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
    elif opt == 'AdamW':
        lr = 0.001 # default: 1e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 1e-2 # default: 1e-2 – weight decay coefficient
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond 
    elif opt == 'Adamax':
        lr = 2e-3 # default: 2e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0 # default: 0 – weight decay (L2 penalty) 
    elif opt == 'ASGD':
        lr = 0.01 # default: 1e-2 – learning rate 
        lambd = 1e-4 # default: 1e-4 – decay term
        alpha = 0.75 # default: 0.75 – power for eta update
        t0 = 1e6 # default: 1e6 – point at which to start averaging
        weight_decay = 0 # default: 0 – weight decay
    elif opt == 'RMSprop':
        lr = 1e-2 # default: 1e-2 – learning rate
        momentum = 0 # default: 0 – momentum factor
        alpha = 0.99 # default: 0.99 – smoothing constant
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        centered = False # default: False – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
        weight_decay = 0 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Rprop':
        lr = 1e-2 # default: 1e-2 – learning rate
        etas = (0.5, 1.2) # default: (0.5, 1.2) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors
        step_sizes = (1e-6, 50) # default: (1e-6, 50) – a pair of minimal and maximal allowed step sizes 
    elif opt == 'SGD':
        lr = 0.01 # – learning rate
        momentum = 0.9 # default: 0 – momentum factor 
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty) 
        dampening = 0 # default: 0 – dampening for momentum
        nesterov = False # default: False – enables Nesterov momentum 
    
    sch = 'CosineAnnealingLR'
    if sch == 'StepLR':
        step_size = epochs // 5 # – Period of learning rate decay.
        gamma = 0.5 # – Multiplicative factor of learning rate decay. Default: 0.1
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'MultiStepLR':
        milestones = [60, 120, 150] # – List of epoch indices. Must be increasing.
        gamma = 0.1 # – Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'ExponentialLR':
        gamma = 0.99 #  – Multiplicative factor of learning rate decay.
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'CosineAnnealingLR':
        T_max = 50 # – Maximum number of iterations. Cosine function period.
        eta_min = 0.00001 # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1.  
    elif sch == 'ReduceLROnPlateau':
        mode = 'min' # – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
        factor = 0.1 # – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
        patience = 10 # – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
        threshold = 0.0001 # – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
        threshold_mode = 'rel' # – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
        cooldown = 0 # – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
        min_lr = 0 # – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
        eps = 1e-08 # – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 50 # – Number of iterations for the first restart.
        T_mult = 2 # – A factor increases T_{i} after a restart. Default: 1.
        eta_min = 1e-6 # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1. 
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 20