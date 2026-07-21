# Running the training on Kaggle (2 × T4)

The local RTX 3050 is for script development, tests, and smoke runs. The 250-epoch runs happen
here.

> **Copy these cells literally.** There are no placeholders to fill in. An earlier version of this
> file used `<user>` as a stand-in, which bash reads as *redirect stdin from a file named `user`* —
> so `git clone https://github.com/<user>/...` silently ran no clone and reported
> `user: No such file or directory`.

## Notebook settings

- Accelerator: **GPU T4 x2**
- Internet: **on**
- Add-ons → Secrets: add **`HF_TOKEN`** (the dataset repo is private)

## Cell 1 — bootstrap

```python
import os, subprocess
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

REPO = "https://github.com/BRUH-MAIN/UltralightUNET.git"
DEST = "/kaggle/working/repo"

subprocess.run(["rm", "-rf", DEST], check=True)
subprocess.run(["git", "clone", "--depth", "1", REPO, DEST], check=True)
os.chdir(DEST)
subprocess.run(["pip", "install", "-q", "thop", "huggingface_hub", "pytest"], check=True)

print("cwd:", os.getcwd())
print("files:", sorted(os.listdir())[:12])
```

`check=True` matters: it raises on a failed clone instead of letting later cells run in the wrong
directory, which is exactly how the `<user>` bug stayed invisible.

## Cell 2 — data

```python
!python scripts/hf_data.py pull --dataset ISIC2017
```

Downloads the six prepared `.npy` files (524 MB) into `data/ISIC2017/`. The repo defaults to
`RohanRamesh/ultralight-vmunet-data`, so there is nothing to pass. Preprocessing and the
train/val/test split were done once locally — this notebook never re-derives them.

## Cell 3 — sanity check before burning hours

```python
!python -m pytest tests/ -q
```

Expect **19 passed**. This verifies the pure-PyTorch selective scan still matches the reference
implementation and that the batched PVM layer matches upstream's four-call form, on this machine's
torch build. If it fails here, stop — do not train.

## Cell 4 — train

```python
!python train.py
```

Prints the parameter count (**must be 49,457**) and GFLOPs at startup, then trains 250 epochs and
evaluates the best checkpoint on the test split automatically. Roughly 3.3 h based on local
measurement; check the first epoch's timing against that.

Results land in `results/UltraLight_VM_UNet_ISIC2017_<timestamp>/`:
`log/train.info.log` (metrics), `checkpoints/` (weights), `outputs/` (per-image overlays).

## Using both GPUs

Run two independent experiments concurrently rather than splitting one across both. The model is
0.049 M parameters and its selective scan issues many very small CUDA kernels, so the bottleneck is
launch overhead on the CPU — `DataParallel` re-replicates the module every step and runs replicas
in GIL-contending threads, adding to exactly the cost that dominates. It would also halve the
per-GPU batch from the paper's 8 to 4.

```python
import subprocess
env0 = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
env1 = {**os.environ, "CUDA_VISIBLE_DEVICES": "1"}
p0 = subprocess.Popen(["python", "train.py"], env=env0,
                      stdout=open("/kaggle/working/isic2017.log", "w"), stderr=subprocess.STDOUT)
# edit configs/config_setting.py:datasets to 'ISIC2018' before launching the second
```

Each run then matches the paper's single-V100 setup.

## Getting results out

`/kaggle/working` is wiped when the session dies. Before it ends:

```python
!tail -40 results/*/log/train.info.log
!zip -qr /kaggle/working/results.zip results/
```

Download `results.zip`, or write it to a Kaggle Dataset output.

## Session limits

Kaggle caps a GPU session at 12 h and 30 GPU-hours/week. `train.py` writes
`checkpoints/latest.pth` every epoch and resumes from it automatically, so a run that outlives a
session can continue if you restore the previous `results/<run>/` directory first — keep it in a
Kaggle Dataset between sessions.

## If something breaks

Kaggle ships newer torch/timm/numpy than the pinned local versions. Two things to watch:

- `from timm.models.layers import trunc_normal_` is a deprecated shim in timm 1.x. It still works,
  but if it ever errors, change it to `from timm.layers import trunc_normal_` in
  `models/UltraLight_VM_UNet.py`.
- torch ≥ 2.6 defaults `torch.load(weights_only=True)`. The checkpoints here hold only tensors and
  plain numbers so they load fine, but a `_pickle.UnpicklingError` on resume would mean passing
  `weights_only=False` in `train.py`.

Cell 3 catches both before any training time is spent.
