"""Train UltraLight VM-UNet.

Differences from upstream's train.py, none of which change the result:
  * no DataParallel wrapper (see the comment where the model is built)
  * torch.load(..., weights_only=False) on the resume path. torch >= 2.6 flipped
    the default to True, which REJECTS this checkpoint: min_loss/loss are
    np.float64 (engine.py returns np.mean(...)), and numpy scalars are not in the
    default allowlist. Verified: loading such a checkpoint with weights_only=True
    raises UnpicklingError. These are our own files, so full unpickling is safe.
  * val/test batch size come from the config instead of being hardcoded to 1,
    with an assertion that they divide the split exactly -- the loaders use
    drop_last=True, so a non-divisor silently discards images.
  * a preflight that fails loudly when torch or the mamba_ssm kernel has no cubin
    for this GPU, rather than 20 minutes into a run.

No change to the model, the optimiser, the schedule, or any hyperparameter that
affects the result.
"""

import torch
from torch.cuda.amp import GradScaler
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


def main(config):

    print('#----------Creating logger----------#')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    logger = get_logger('train', log_dir)

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    set_seed(config.seed)
    torch.cuda.empty_cache()

    check_environment(logger)





    print('#----------Preparing dataset----------#')
    train_dataset = isic_loader(path_Data = config.data_path, train = True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = isic_loader(path_Data = config.data_path, train = False)
    # drop_last=True means a batch size that does not divide the split silently
    # discards the remainder. Fail instead of quietly evaluating fewer images.
    assert len(val_dataset) % config.val_batch_size == 0, (
        f'val_batch_size={config.val_batch_size} does not divide '
        f'{len(val_dataset)} val images; drop_last=True would discard '
        f'{len(val_dataset) % config.val_batch_size} of them')
    val_loader = DataLoader(val_dataset,
                                batch_size=config.val_batch_size,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)
    test_dataset = isic_loader(path_Data = config.data_path, train = False, Test = True)
    assert len(test_dataset) % config.test_batch_size == 0, (
        f'test_batch_size={config.test_batch_size} does not divide '
        f'{len(test_dataset)} test images')
    test_loader = DataLoader(test_dataset,
                                batch_size=config.test_batch_size,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)




    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config
    model = UltraLight_VM_UNet(num_classes=model_cfg['num_classes'], 
                               input_channels=model_cfg['input_channels'], 
                               c_list=model_cfg['c_list'], 
                               split_att=model_cfg['split_att'], 
                               bridge=model_cfg['bridge'],)
    
    # PATCH: single GPU, so the DataParallel wrapper is dropped and the '.module'
    # indirection below goes with it. Checkpoints are therefore written with plain
    # (un-prefixed) state_dict keys, same as upstream's model.module.state_dict().
    #
    # This matches the paper (a single V100). On a 2x T4 Kaggle session, run two
    # experiments side by side with CUDA_VISIBLE_DEVICES rather than splitting one
    # across both: at 0.049M params the bottleneck is kernel-launch overhead, which
    # DataParallel adds to rather than removes.
    model = model.cuda()

    cal_params_flops(model, 256, logger)






    print('#----------Prepareing loss, opt, sch and amp----------#')
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
        # weights_only=False: torch >= 2.6 defaults to True, which rejects the
        # np.float64 min_loss/loss in this checkpoint. Our own file, so safe.
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'),
                                weights_only=False)
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

        train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            scaler=scaler
        )

        loss = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )


        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save(
            {
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
        loss = test_one_epoch(
                test_loader,
                model,
                criterion,
                logger,
                config,
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )      


if __name__ == '__main__':
    config = setting_config
    main(config)