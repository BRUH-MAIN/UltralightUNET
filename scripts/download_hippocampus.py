"""Download and extract MSD Task04_Hippocampus for the 3D PVM extension.

The official Medical Segmentation Decathlon archive is Google-Drive-hosted
(PH2-style manual friction); this pulls the same archive from a public
Hugging Face mirror instead, which needs no account or token for a public repo.

    python scripts/download_hippocampus.py

Downloads data/Task04_Hippocampus.tar (28 MB) and extracts it to
data/Task04_Hippocampus/{imagesTr,labelsTr,imagesTs}/, skipping either step if
already done. Next: python dataprepare/Prepare_Hippocampus.py
"""

import argparse
import os
import sys
import tarfile

REPO = "Novel-BioMedAI/Medical_Segmentation_Decathlon"
FILENAME = "Task04_Hippocampus.tar"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="directory to download/extract into")
    ap.add_argument("--keep-tar", action="store_true", help="keep the .tar archive after extraction")
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    extracted = os.path.join(args.root, "Task04_Hippocampus")
    images_dir = os.path.join(extracted, "imagesTr")
    labels_dir = os.path.join(extracted, "labelsTr")

    if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
        n_img = len([f for f in os.listdir(images_dir) if f.endswith(".nii.gz")])
        n_lbl = len([f for f in os.listdir(labels_dir) if f.endswith(".nii.gz")])
        if n_img and n_img == n_lbl:
            print(f"{extracted}/ already has {n_img} image/label pairs, nothing to do")
            return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is not installed.\n  pip install huggingface_hub")

    print(f"downloading {FILENAME} from {REPO} (28 MB)")
    tar_path = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=FILENAME,
                                local_dir=args.root)

    print(f"extracting -> {extracted}/")
    with tarfile.open(tar_path) as tf:
        tf.extractall(args.root)

    if not args.keep_tar:
        os.remove(tar_path)

    n_img = len([f for f in os.listdir(images_dir) if f.endswith(".nii.gz")])
    n_lbl = len([f for f in os.listdir(labels_dir) if f.endswith(".nii.gz")])
    print(f"{extracted}/: {n_img} images, {n_lbl} labels")
    if n_img != n_lbl:
        print("WARNING: image/label count mismatch", file=sys.stderr)
    print("Next: python dataprepare/Prepare_Hippocampus.py")


if __name__ == "__main__":
    main()
