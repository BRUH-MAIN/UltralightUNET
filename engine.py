import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
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

    # PATCH: one epoch-level summary, logged from loss_list which is already fully
    # populated at this point. print_interval-gated lines above give a running
    # mean as of whichever iter happened to hit the interval (e.g. iter 140 of
    # 157) -- this is the exact per-epoch mean, so plotting code doesn't have to
    # reconstruct one from unevenly-spaced samples.
    log_info = f'train epoch: {epoch} done, mean loss: {np.mean(loss_list):.4f}'
    print(log_info)
    logger.info(log_info)
    scheduler.step()


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    # PATCH: always compute and log the full metric set, not just every
    # config.val_interval-th epoch. val_interval gated LOGGING VERBOSITY only --
    # confusion_matrix() runs once per call on preds/gts already collected above
    # in memory (val_batch_size=30 worth of pixels), well under the per-epoch
    # cost of the forward passes themselves. Logging 30x more often is a few
    # extra lines of local text-file I/O, not a training-speed change.
    #
    # This does not change the return value below (np.mean(loss_list), computed
    # identically to what the old val_interval-gated branch also returned), so
    # train.py's `if loss < min_loss` checkpoint selection is unaffected.
    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)

    y_pre = np.where(preds>=config.threshold, 1, 0)
    y_true = np.where(gts>=0.5, 1, 0)

    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
            specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
    print(log_info)
    logger.info(log_info)

    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None,
                    save_outputs=True):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            if save_outputs:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

            # PATCH: per-image DSC, tagged with the same index `i` used for the
            # outputs/{i}.png overlay above, so failure cases can be
            # cross-referenced to their overlay image. test_batch_size is fixed
            # at 1 (configs/config_setting.py), so this loop already IS one
            # image per iteration -- this is the exact per-image score, applying
            # the same 2*inter/denom identity used for the pooled f1_or_dsc below
            # to one image's counts instead of all images' pooled counts. It
            # does not feed into loss_list/preds/gts, so the pooled metrics
            # computed after this loop are unaffected. logger.info only (no
            # print): this runs once per test image, and printing all of them
            # would flood stdout and fight the tqdm bar above.
            img_pred = np.where(out >= config.threshold, 1, 0)
            img_true = np.where(msk >= 0.5, 1, 0)
            img_inter = np.logical_and(img_pred, img_true).sum()
            img_denom = img_pred.sum() + img_true.sum()
            img_dice = float(2 * img_inter) / float(img_denom) if img_denom != 0 else 0
            logger.info(f'test image {i}: dice: {img_dice:.4f}')

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)
