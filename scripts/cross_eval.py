"""Cross-dataset generalization sweep.

For every (train_dataset, eval_dataset) pair among {ISIC2017, ISIC2018, PH2,
HAM10000}, loads the checkpoint trained on train_dataset and evaluates it --
zero-shot, no fine-tuning -- on eval_dataset's test split. Reuses
engine.test_one_epoch for the actual forward pass and metric computation, the
same function test.py calls for every in-domain result in
results/COMPARISON.md, so the numbers are directly comparable to those.

The diagonal (train_dataset == eval_dataset) should reproduce each dataset's
already-recorded in-domain DSC from COMPARISON.md -- that's the built-in sanity
check that this script's plumbing matches test.py's, not an independent result.

Prediction overlays (save_imgs, one .png per test image) are skipped here --
test.py already wrote them for the in-domain runs, and this sweep evaluates
4 models x 4 datasets x however many test images each has, which would
otherwise write ~12,000 mostly-redundant PNGs. engine.test_one_epoch's new
save_outputs=False keyword controls this; the flag is additive and defaults
to True everywhere else, so test.py/train.py are unaffected.

Usage:
    python scripts/cross_eval.py
    python scripts/cross_eval.py --datasets ISIC2017 ISIC2018   # subset
"""

import argparse
import glob
import json
import os
import re
import sys
import warnings

import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # configs/engine/loader/models are top-level, not in scripts/

from configs.config_setting import setting_config
from engine import test_one_epoch
from loader import isic_loader
from models.UltraLight_VM_UNet import UltraLight_VM_UNet
from utils import get_logger

DATA_ROOT = os.path.join(REPO_ROOT, "data")
RESULTS_ROOT = os.path.join(REPO_ROOT, "results")
OUT_DIR = os.path.join(RESULTS_ROOT, "cross_eval")

DATASETS = ["ISIC2017", "ISIC2018", "PH2", "HAM10000"]

LOG_RE = re.compile(
    r"test of best model, loss: ([\d.]+),miou: ([\d.]+), f1_or_dsc: ([\d.]+), "
    r"accuracy: ([\d.]+), *specificity: ([\d.]+), sensitivity: ([\d.]+)"
)


def find_best_checkpoint(dataset):
    """Most recent run's best-*.pth for `dataset`, by directory mtime.

    NOT by sorting the dirname string: run dirs are named e.g.
    "..._Wednesday_19_August_2026_...", and a weekday-first string sort is not
    chronological (Wednesday < Monday alphabetically, regardless of which
    happened first) -- an earlier version of this function got a lucky
    coincidence out of that, then broke silently for ISIC2017's two runs."""
    run_dirs = glob.glob(os.path.join(RESULTS_ROOT, f"UltraLight_VM_UNet_{dataset}_*"))
    if not run_dirs:
        raise SystemExit(f"no results/UltraLight_VM_UNet_{dataset}_* run directory found -- "
                         f"train on {dataset} first")
    run_dirs.sort(key=os.path.getmtime)
    if len(run_dirs) > 1:
        print(f"  NOTE: {len(run_dirs)} run directories found for {dataset}; using the most "
             f"recent. All candidates:")
        for d in run_dirs:
            print(f"    {os.path.basename(d)}")
    run_dir = run_dirs[-1]
    ckpts = glob.glob(os.path.join(run_dir, "checkpoints", "best-*.pth"))
    if not ckpts:
        raise SystemExit(f"no best-*.pth checkpoint in {run_dir}/checkpoints/")
    return ckpts[0], run_dir


def load_model(ckpt_path, model_cfg):
    model = UltraLight_VM_UNet(num_classes=model_cfg['num_classes'],
                               input_channels=model_cfg['input_channels'],
                               c_list=model_cfg['c_list'],
                               split_att=model_cfg['split_att'],
                               bridge=model_cfg['bridge'])
    model = model.cuda()
    # Same checkpoint-loading logic as test.py: accept either a bare state_dict
    # (best-*.pth) or a full training checkpoint (latest.pth), and strip thop's
    # profiling buffers if present.
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    thop_keys = [k for k in state if k.endswith(('total_ops', 'total_params'))]
    for k in thop_keys:
        del state[k]
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_pair(model, train_ds, eval_ds, config):
    data_path = os.path.join(DATA_ROOT, eval_ds) + os.sep
    test_dataset = isic_loader(path_Data=data_path, train=False, Test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             pin_memory=True, num_workers=config.num_workers, drop_last=True)

    pair_dir = os.path.join(OUT_DIR, f"{train_ds}_on_{eval_ds}")
    log_dir = os.path.join(pair_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    # config is the shared setting_config class -- mutated in place per pair,
    # same pattern test.py/train.py already use. work_dir only matters here for
    # save_imgs, which is disabled below, but test_one_epoch still reads it.
    config.work_dir = pair_dir + os.sep
    config.datasets = eval_ds

    logger_name = f"{train_ds}_on_{eval_ds}"
    logger = get_logger(logger_name, log_dir)

    test_one_epoch(test_loader, model, config.criterion, logger, config,
                   test_data_name=eval_ds, save_outputs=False)

    log_path = os.path.join(log_dir, f"{logger_name}.info.log")
    with open(log_path) as f:
        text = f.read()
    m = LOG_RE.search(text)
    if not m:
        raise RuntimeError(f"could not parse metrics from {log_path}")
    loss, miou, dsc, acc, sp, se = (float(g) for g in m.groups())
    return dict(dsc=dsc, miou=miou, acc=acc, sp=sp, se=se, loss=loss, n_test=len(test_dataset))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS)
    args = ap.parse_args()
    datasets = args.datasets

    config = setting_config
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    for train_ds in datasets:
        ckpt_path, run_dir = find_best_checkpoint(train_ds)
        print(f"\n=== source model: {train_ds}  ({os.path.relpath(ckpt_path, REPO_ROOT)}) ===")
        model = load_model(ckpt_path, config.model_config)

        for eval_ds in datasets:
            print(f"  -> evaluating on {eval_ds}...", end=" ", flush=True)
            r = evaluate_pair(model, train_ds, eval_ds, config)
            results[(train_ds, eval_ds)] = r
            tag = "(in-domain)" if train_ds == eval_ds else ""
            print(f"DSC={r['dsc']:.4f}  mIoU={r['miou']:.4f}  n={r['n_test']} {tag}")

        del model
        torch.cuda.empty_cache()

    # Matrix printout + CSV/JSON export.
    print("\n=== DSC matrix (rows: trained on, cols: evaluated on) ===")
    header = "trained\\eval".ljust(14) + "".join(d.ljust(12) for d in datasets)
    print(header)
    for train_ds in datasets:
        row = train_ds.ljust(14)
        for eval_ds in datasets:
            row += f"{results[(train_ds, eval_ds)]['dsc']:.4f}".ljust(12)
        print(row)

    csv_path = os.path.join(OUT_DIR, "cross_eval_matrix.csv")
    with open(csv_path, "w") as f:
        f.write("trained_on,evaluated_on,dsc,miou,accuracy,specificity,sensitivity,loss,n_test\n")
        for train_ds in datasets:
            for eval_ds in datasets:
                r = results[(train_ds, eval_ds)]
                f.write(f"{train_ds},{eval_ds},{r['dsc']:.6f},{r['miou']:.6f},{r['acc']:.6f},"
                       f"{r['sp']:.6f},{r['se']:.6f},{r['loss']:.6f},{r['n_test']}\n")

    json_path = os.path.join(OUT_DIR, "cross_eval_matrix.json")
    with open(json_path, "w") as f:
        json.dump({f"{k[0]}__{k[1]}": v for k, v in results.items()}, f, indent=2)

    print(f"\nwrote {csv_path}\nwrote {json_path}")


if __name__ == "__main__":
    main()
