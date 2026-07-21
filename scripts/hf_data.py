"""Move the prepared .npy splits between this machine, HuggingFace, and Kaggle.

The prepared arrays are the reproducibility boundary: preprocessing and the
train/val/test split happen exactly once, locally, and both the RTX 3050 debug
runs and the Kaggle T4 training runs consume the identical bytes. Nothing
re-derives the split on the training machine.

    # once, after running dataprepare/Prepare_ISIC2017.py
    python scripts/hf_data.py push --dataset ISIC2017 --repo <user>/ultralight-vmunet-data

    # on any machine, including a Kaggle notebook
    python scripts/hf_data.py pull --dataset ISIC2017 --repo <user>/ultralight-vmunet-data

Auth: huggingface_hub picks up HF_TOKEN from the environment or the token saved
by `huggingface-cli login`. For a private repo on Kaggle, add HF_TOKEN as a
notebook secret; for a public repo, `pull` needs no token at all.
"""

import argparse
import os
import sys

SPLIT_FILES = [f"{kind}_{split}.npy"
               for kind in ("data", "mask")
               for split in ("train", "val", "test")]


def _require_hub():
    try:
        from huggingface_hub import HfApi, snapshot_download  # noqa: F401
    except ImportError:
        sys.exit("huggingface_hub is not installed.\n  pip install huggingface_hub")
    from huggingface_hub import HfApi, snapshot_download
    return HfApi, snapshot_download


def push(args):
    HfApi, _ = _require_hub()
    local = os.path.join("data", args.dataset)

    missing = [f for f in SPLIT_FILES if not os.path.exists(os.path.join(local, f))]
    if missing:
        sys.exit(f"missing in {local}: {', '.join(missing)}\n"
                 f"Run: python dataprepare/Prepare_{args.dataset}.py")

    total = sum(os.path.getsize(os.path.join(local, f)) for f in SPLIT_FILES)
    print(f"uploading {len(SPLIT_FILES)} files ({total / 1e6:.0f} MB) "
          f"from {local} -> {args.repo}:{args.dataset}/")
    print(f"visibility: {'private' if args.private else 'PUBLIC'}")

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=local,
        path_in_repo=args.dataset,
        allow_patterns=SPLIT_FILES,
        commit_message=f"{args.dataset}: prepared 256x256 uint8 splits",
    )
    print(f"done -> https://huggingface.co/datasets/{args.repo}/tree/main/{args.dataset}")


def pull(args):
    _, snapshot_download = _require_hub()
    local = os.path.join("data", args.dataset)

    have = [f for f in SPLIT_FILES if os.path.exists(os.path.join(local, f))]
    if len(have) == len(SPLIT_FILES) and not args.force:
        print(f"{local} already complete ({len(have)} files); use --force to re-download")
        return

    print(f"downloading {args.repo}:{args.dataset}/ -> {local}")
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=[f"{args.dataset}/*.npy"],
        local_dir="data_hf_tmp",
    )
    os.makedirs(local, exist_ok=True)
    src = os.path.join("data_hf_tmp", args.dataset)
    for f in SPLIT_FILES:
        s = os.path.join(src, f)
        if not os.path.exists(s):
            sys.exit(f"{f} missing from the repo; was push run for {args.dataset}?")
        os.replace(s, os.path.join(local, f))
    print(f"{local}/ ready ({len(SPLIT_FILES)} files)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn in (("push", push), ("pull", pull)):
        p = sub.add_parser(name)
        p.add_argument("--dataset", default="ISIC2017", choices=["ISIC2017", "ISIC2018", "PH2"])
        p.add_argument("--repo", required=True, help="e.g. yourname/ultralight-vmunet-data")
        p.set_defaults(fn=fn)

    sub.choices["push"].add_argument(
        "--public", dest="private", action="store_false", default=True,
        help="publish the repo publicly (default: private). ISIC data is CC-BY-NC; "
             "check the challenge terms before making a derived copy public.")
    sub.choices["pull"].add_argument("--force", action="store_true")

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
