"""Evaluate a trained checkpoint on the test split.

Differences from upstream's test.py:
  * torch.load(..., weights_only=False), same reason as train.py.
  * --weights and --work-dir command line arguments, so a checkpoint can be
    evaluated from the notebook without editing the config.
  * test batch size from the config.
  * the optimiser, scheduler, scaler and epoch counters upstream builds here are
    dropped. Evaluation takes no gradients and no steps, so none of them was ever
    read; the only one that matters is the criterion, for the reported loss.

Usage:
    python test.py --weights results/<run>/checkpoints/best-epoch116-loss0.2545.pth
"""

import torch
from torch.utils.data import DataLoader
from loader import *

from models.UltraLight_VM_UNet import UltraLight_VM_UNet
from engine import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")



def main(config, weights=None, work_dir=None):

    print('#----------Creating logger----------#')
    work_dir = work_dir or config.work_dir
    if not work_dir.endswith(os.sep) and not work_dir.endswith('/'):
        work_dir += os.sep
    # engine.test_one_epoch builds its save path from config.work_dir directly
    # (config.work_dir + 'outputs/'), so pointing only the local variable at the
    # requested directory would send prediction overlays to a fresh timestamped
    # folder that was never created.
    config.work_dir = work_dir
    log_dir = os.path.join(work_dir, 'log')
    checkpoint_dir = os.path.join(work_dir, 'checkpoints')
    resume_model = weights or config.test_weights
    if not resume_model or not os.path.exists(resume_model):
        raise SystemExit(f'checkpoint not found: {resume_model!r}\n'
                         'pass --weights <path/to/best-*.pth>')
    outputs = os.path.join(work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    logger = get_logger('test', log_dir)

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    set_seed(config.seed)
    torch.cuda.empty_cache()
    check_environment(logger)



    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config    
    model = UltraLight_VM_UNet(num_classes=model_cfg['num_classes'], 
                               input_channels=model_cfg['input_channels'], 
                               c_list=model_cfg['c_list'], 
                               split_att=model_cfg['split_att'], 
                               bridge=model_cfg['bridge'],)
    
    model = model.cuda()  # PATCH: single GPU, no DataParallel wrapper


    print('#----------Preparing dataset----------#')
    test_dataset = isic_loader(path_Data = config.data_path, train = False, Test = True)
    test_loader = DataLoader(test_dataset,
                                batch_size=config.test_batch_size,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)

    criterion = config.criterion

    print('#----------Testing----------#')
    best_weight = torch.load(resume_model, map_location=torch.device('cpu'),
                             weights_only=False)  # see train.py
    # A full training checkpoint (latest.pth) nests the weights; best.pth is the
    # bare state_dict. Accept either.
    if 'model_state_dict' in best_weight:
        best_weight = best_weight['model_state_dict']
    # Checkpoints written before the cal_params_flops fix in utils.py carry thop's
    # total_ops / total_params buffers, which a fresh model does not declare.
    # Dropping them makes those older checkpoints loadable.
    thop_keys = [k for k in best_weight if k.endswith(('total_ops', 'total_params'))]
    if thop_keys:
        print(f'stripping {len(thop_keys)} thop profiling buffers from the checkpoint')
        for k in thop_keys:
            del best_weight[k]
    model.load_state_dict(best_weight)
    test_one_epoch(
            test_loader,
            model,
            criterion,
            logger,
            config,
        )



if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', help='path to a .pth checkpoint')
    ap.add_argument('--work-dir', help='where to write log/ and outputs/ '
                                       '(default: a fresh timestamped dir)')
    args = ap.parse_args()
    main(setting_config, weights=args.weights, work_dir=args.work_dir)