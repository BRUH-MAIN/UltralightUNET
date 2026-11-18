"""Train/val/test loops for PVM-3D and StandardUNet3D.

Structurally mirrors engine.py (same train_one_epoch / val_one_epoch /
test_one_epoch three-way split, same epoch-level mean-loss logging, same
per-test-image dice logging for cross-referencing against saved overlays).
Differs where the task differs: multi-class Dice+CE loss and per-class DSC
(background/anterior/posterior) instead of the 2D loader's single-channel
BCE+Dice and binary confusion matrix.
"""

import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast

from utils3d import per_class_dice, save_mid_slices

CLASS_NAMES = ('background', 'anterior', 'posterior')


def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, logger, config, scaler=None):
    model.train()
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data
        images = images.cuda(non_blocking=True).float()
        targets = targets.cuda(non_blocking=True).long()
        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)

    log_info = f'train epoch: {epoch} done, mean loss: {np.mean(loss_list):.4f}'
    print(log_info)
    logger.info(log_info)
    scheduler.step()


def val_one_epoch(val_loader, model, criterion, epoch, logger, config):
    model.eval()
    loss_list = []
    dsc_list = []
    with torch.no_grad():
        for data in tqdm(val_loader):
            img, msk = data
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).long()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            dsc_list.append(per_class_dice(out, msk, num_classes=config.model_config['num_classes']))

    mean_dsc = np.mean(dsc_list, axis=0)
    dsc_str = ', '.join(f'{n}: {d:.4f}' for n, d in zip(CLASS_NAMES, mean_dsc))
    log_info = (f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, '
                f'mean_dsc(fg): {mean_dsc[1:].mean():.4f}, {dsc_str}')
    print(log_info)
    logger.info(log_info)

    return np.mean(loss_list)


def test_one_epoch(test_loader, model, criterion, logger, config, test_data_name=None, save_outputs=True):
    model.eval()
    loss_list = []
    dsc_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).long()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())

            img_dsc = per_class_dice(out, msk, num_classes=config.model_config['num_classes'])
            dsc_list.append(img_dsc)
            dsc_str = ', '.join(f'{n}: {d:.4f}' for n, d in zip(CLASS_NAMES, img_dsc))
            logger.info(f'test volume {i}: mean_dsc(fg): {img_dsc[1:].mean():.4f}, {dsc_str}')

            if save_outputs:
                pred = out.argmax(dim=1)
                save_mid_slices(img, msk, pred, i, config.work_dir + 'outputs/', test_data_name=test_data_name)

    mean_dsc = np.mean(dsc_list, axis=0)
    dsc_str = ', '.join(f'{n}: {d:.4f}' for n, d in zip(CLASS_NAMES, mean_dsc))
    if test_data_name is not None:
        log_info = f'test_datasets_name: {test_data_name}'
        print(log_info)
        logger.info(log_info)
    log_info = (f'test of best model, loss: {np.mean(loss_list):.4f}, '
                f'mean_dsc(fg): {mean_dsc[1:].mean():.4f}, {dsc_str}')
    print(log_info)
    logger.info(log_info)

    return np.mean(loss_list)
