"""Train / val / test loops and metrics for MILK10k classification.

The challenge metric is macro-F1 (unweighted mean of per-class F1) at a 0.5
threshold. Because the dataset is extreme (BCC 48% .. MAL_OTH 0.2%), macro-F1 is
dominated by the rare classes, so we always report per-class F1 alongside it --
hiding the breakdown behind a single number would hide where the model actually
fails.

Metrics are pooled over the whole split (all lesions), matching how the Phase-1
segmentation engine pooled over all pixels.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score)
from tqdm import tqdm


def _forward(model, batch, device):
    """Model takes a dict of modality tensors; returns logits (B, num_classes)."""
    x = {k: v.to(device, non_blocking=True).float()
         for k, v in batch.items() if k in ('derm', 'clin')}
    return model(x)


def train_one_epoch(loader, model, criterion, optimizer, scheduler, epoch, logger, config, scaler=None):
    model.train()
    device = next(model.parameters()).device
    losses = []
    for it, batch in enumerate(loader):
        y = batch['label'].to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = _forward(model, batch, device)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if it % config.print_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            msg = f'train: epoch {epoch}, iter:{it}, loss: {np.mean(losses):.4f}, lr: {lr}'
            print(msg)
            logger.info(msg)
    scheduler.step()
    return np.mean(losses)


@torch.no_grad()
def evaluate(loader, model, criterion, config, split='val'):
    """Returns (loss, metrics dict) with macro-F1, per-class F1, balanced acc."""
    model.eval()
    device = next(model.parameters()).device
    losses, all_logits, all_y = [], [], []
    for batch in tqdm(loader, leave=False):
        y = batch['label'].to(device, non_blocking=True)
        logits = _forward(model, batch, device)
        losses.append(criterion(logits, y).item())
        all_logits.append(logits.cpu())
        all_y.append(y.cpu())

    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y).numpy()
    y_pred = logits.argmax(1).numpy()
    n = config.num_classes

    per_class = f1_score(y_true, y_pred, labels=list(range(n)), average=None, zero_division=0)
    metrics = {
        'loss': float(np.mean(losses)),
        'macro_f1': float(f1_score(y_true, y_pred, labels=list(range(n)), average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, labels=list(range(n)), average='weighted', zero_division=0)),
        'balanced_acc': float(balanced_accuracy_score(y_true, y_pred)),
        'accuracy': float((y_true == y_pred).mean()),
        'per_class_f1': {config.classes[i]: float(per_class[i]) for i in range(n)},
        'confusion': confusion_matrix(y_true, y_pred, labels=list(range(n))).tolist(),
    }
    return metrics['loss'], metrics


def log_metrics(metrics, epoch, logger, split='val'):
    head = (f'{split} epoch {epoch}: loss {metrics["loss"]:.4f}  '
            f'macro-F1 {metrics["macro_f1"]:.4f}  bal-acc {metrics["balanced_acc"]:.4f}  '
            f'acc {metrics["accuracy"]:.4f}')
    print(head)
    logger.info(head)
    pc = '  '.join(f'{k}:{v:.2f}' for k, v in metrics['per_class_f1'].items())
    logger.info(f'{split} per-class F1: {pc}')
    return metrics['macro_f1']


# ---- losses ---------------------------------------------------------------

class FocalLoss(torch.nn.Module):
    """Multi-class focal loss. gamma=0 reduces to (optionally weighted) CE."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('weight', weight if weight is not None else None)

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        logpt = logp.gather(1, target[:, None]).squeeze(1)
        pt = p.gather(1, target[:, None]).squeeze(1)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def build_criterion(config, class_weights=None):
    w = class_weights if config.class_weighting else None
    if config.loss == 'focal':
        return FocalLoss(gamma=config.focal_gamma, weight=w)
    if config.loss == 'weighted_ce':
        return torch.nn.CrossEntropyLoss(weight=w)
    return torch.nn.CrossEntropyLoss()


def compute_class_weights(manifest_df, config, device):
    """Per-class loss weights from the TRAIN split, normalised to mean 1.

    ``config.weight_scheme`` selects how aggressively rare classes are up-weighted:

      'inverse'       w_i propto 1/n_i         -- full inverse frequency (280x spread
                                                  here). Crushes the majority class:
                                                  BCC errors become nearly free, so
                                                  the model stops predicting BCC.
      'sqrt_inverse'  w_i propto 1/sqrt(n_i)   -- gentler (~17x spread).
      'effective_num' Cui et al. 2019          -- w_i propto (1-beta)/(1-beta^n_i);
                                                  interpolates between uniform (beta=0)
                                                  and inverse (beta->1). The principled
                                                  default for extreme imbalance.
    """
    tr = manifest_df[manifest_df.split == 'train']
    counts = tr['label_idx'].value_counts().reindex(range(config.num_classes)).fillna(0).values
    counts = np.clip(counts, 1, None).astype(np.float64)

    scheme = getattr(config, 'weight_scheme', 'inverse')
    if scheme == 'effective_num':
        beta = getattr(config, 'weight_beta', 0.999)
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    elif scheme == 'sqrt_inverse':
        w = 1.0 / np.sqrt(counts)
    elif scheme == 'inverse':
        w = 1.0 / counts
    else:
        raise ValueError(f'unknown weight_scheme {scheme!r}')

    w = w / w.mean()   # mean 1, so the loss scale is comparable across schemes
    return torch.tensor(w, dtype=torch.float32, device=device)
