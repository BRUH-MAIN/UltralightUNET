import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import *

from models.UltraLight_VM_UNet import UltraLight_VM_UNet
from engine import *
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")


def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()





    print('#----------Preparing dataset----------#')
    train_dataset = isic_loader(path_Data = config.data_path, train = True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = isic_loader(path_Data = config.data_path, train = False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)
    test_dataset = isic_loader(path_Data = config.data_path, train = False, Test = True)
    test_loader = DataLoader(test_dataset,
                                batch_size=1,
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

    from models.UltraLight_VM_UNet import MAMBA_BACKEND
    log_info = f'mamba backend: {MAMBA_BACKEND}, device: {torch.cuda.get_device_name(0)}'
    print(log_info)
    logger.info(log_info)
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
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
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
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
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