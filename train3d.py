"""Train PVM-3D or StandardUNet3D on MSD Task04_Hippocampus.

Structurally mirrors train.py (same resume-from-checkpoint logic, same
best/latest checkpointing, same final test-on-best-checkpoint at the end).
Differs where the task differs: `config.network` selects between
UltraLight_VM_UNet_3D and StandardUNet3D so the two can be run back-to-back
under identical conditions (docs/3d_extension_plan.md's "Next steps" #2), and
the loader/engine/loss are the 3D multi-class versions (loader3d.py,
engine3d.py, utils3d.DiceCELoss3D) instead of the 2D binary ones.

    python train3d.py                                  # trains config.network
    python train3d.py --network StandardUNet3D          # override for the baseline run
"""

import argparse
import os
import warnings

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from loader3d import hippocampus_loader
from engine3d import train_one_epoch, val_one_epoch, test_one_epoch
from utils import set_seed, get_logger, log_config_info, get_optimizer, get_scheduler, check_environment
from utils3d import cal_params_flops3d
from configs.config_setting_3d import setting_config, _HERE

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")


def build_model(config):
    model_cfg = config.model_config
    if config.network == 'UltraLight_VM_UNet_3D':
        from models.UltraLight_VM_UNet_3D import UltraLight_VM_UNet_3D
        return UltraLight_VM_UNet_3D(num_classes=model_cfg['num_classes'],
                                      input_channels=model_cfg['input_channels'],
                                      c_list=model_cfg['c_list'],
                                      split_att=model_cfg['split_att'],
                                      bridge=model_cfg['bridge'])
    elif config.network == 'StandardUNet3D':
        from models.StandardUNet3D import StandardUNet3D
        return StandardUNet3D(num_classes=model_cfg['num_classes'],
                               input_channels=model_cfg['input_channels'],
                               c_list=model_cfg['c_list'])
    raise ValueError(f"unknown network {config.network!r}")


def main(config):
    print('#----------Creating logger----------#')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)

    logger = get_logger('train', log_dir)
    log_config_info(config, logger)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    torch.cuda.empty_cache()
    if config.network == 'UltraLight_VM_UNet_3D':
        check_environment(logger)  # only the PVM path touches the mamba_ssm kernel

    print('#----------Preparing dataset----------#')
    train_dataset = hippocampus_loader(path_Data=config.data_path, train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                               pin_memory=True, num_workers=config.num_workers)

    val_dataset = hippocampus_loader(path_Data=config.data_path, train=False)
    assert len(val_dataset) % config.val_batch_size == 0, (
        f'val_batch_size={config.val_batch_size} does not divide '
        f'{len(val_dataset)} val volumes; drop_last=True would discard '
        f'{len(val_dataset) % config.val_batch_size} of them')
    val_loader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,
                             pin_memory=True, num_workers=config.num_workers, drop_last=True)

    test_dataset = hippocampus_loader(path_Data=config.data_path, train=False, Test=True)
    assert len(test_dataset) % config.test_batch_size == 0, (
        f'test_batch_size={config.test_batch_size} does not divide '
        f'{len(test_dataset)} test volumes')
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False,
                              pin_memory=True, num_workers=config.num_workers, drop_last=True)

    print('#----------Preparing model----------#')
    model = build_model(config).cuda()
    cal_params_flops3d(model, config.patch_size, config.model_config['input_channels'], logger)

    print('#----------Preparing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']
        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()

        train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, logger, config, scaler=scaler)
        loss = val_one_epoch(val_loader, model, criterion, epoch, logger, config)

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save({
            'epoch': epoch,
            'min_loss': min_loss,
            'min_epoch': min_epoch,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(os.path.join(checkpoint_dir, 'best.pth'),
                                  map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(best_weight)
        test_one_epoch(test_loader, model, criterion, logger, config)
        os.rename(os.path.join(checkpoint_dir, 'best.pth'),
                  os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--network', choices=['UltraLight_VM_UNet_3D', 'StandardUNet3D'],
                     help='override configs/config_setting_3d.py:network')
    ap.add_argument('--epochs', type=int, help='override configs/config_setting_3d.py:epochs')
    args = ap.parse_args()

    config = setting_config
    if args.network and args.network != config.network:
        config.network = args.network
        config.model_config['c_list'] = config._MODEL_CLIST[args.network]
        from datetime import datetime
        config.work_dir = os.path.join(_HERE, 'results', '') + config.network + '_' + config.datasets \
            + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'
    if args.epochs:
        config.epochs = args.epochs

    main(config)
