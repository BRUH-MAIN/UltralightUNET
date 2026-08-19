"""Install the mamba_ssm / causal_conv1d wheels that match this environment.

PyPI carries only sdists for these two. Installing from source compiles a few
hundred CUDA template instantiations against a local CUDA >= 12.8 toolkit -- tens
of minutes, and a toolkit this project otherwise does not need. Both projects
publish prebuilt wheels on their GitHub releases instead, named by CUDA major,
torch minor, C++ ABI, python tag and machine:

    mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl

There is no resolver for that -- pip cannot pick one, because the filename
encodes facts about the *installed* torch. This script reads them off the running
interpreter and installs the matching pair.

Blackwell (sm_120): take a cu12 or cu13 wheel, never cu11. mamba_ssm's setup.py
emits an sm_120 cubin only when the build ran against CUDA >= 12.8 -- its release
CI uses 11.8.0, 12.9.1 and 13.0.1, so the cu12 and cu13 wheels have it and the
cu11 wheel does not. There is no forward-compatible PTX to fall back on, so a
cu11 wheel does not run on this GPU at all. utils.check_environment catches that
by executing a kernel, but it is cheaper not to install the wrong one.

Usage:
    python scripts/install_mamba.py            # install both
    python scripts/install_mamba.py --print    # just show the URLs
"""

import argparse
import platform
import subprocess
import sys

import torch

MAMBA_VERSION = "2.3.2.post1"
CAUSAL_CONV1D_VERSION = "1.6.2.post1"

RELEASES = {
    "mamba_ssm": ("state-spaces/mamba", MAMBA_VERSION),
    "causal_conv1d": ("Dao-AILab/causal-conv1d", CAUSAL_CONV1D_VERSION),
}

# torch minors both projects publish wheels for, as of these releases. Outside this
# set there is no URL to build -- the request would 404 -- so say so plainly rather
# than let pip report a missing file.
SUPPORTED_TORCH = ("2.6", "2.7", "2.8", "2.9", "2.10")


def wheel_urls():
    if torch.version.cuda is None:
        raise SystemExit("this torch build has no CUDA; mamba_ssm is CUDA-only")

    cuda_major = torch.version.cuda.split(".")[0]
    if cuda_major == "11":
        raise SystemExit(
            "this torch is a CUDA 11 build; the cu11 wheels are compiled for "
            "sm_75/80/90 only and\ncannot run on Blackwell. Reinstall torch from "
            "the cu129 index -- see requirements.txt.")

    # "2.10.0+cu129" -> "2.10": the wheels are built per torch minor release.
    torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    if torch_mm not in SUPPORTED_TORCH:
        raise SystemExit(
            f"no prebuilt wheel for torch {torch_mm}: mamba_ssm {MAMBA_VERSION} "
            f"publishes {', '.join(SUPPORTED_TORCH)}.\nEither install one of those "
            f"torch versions, or compile for this one:\n"
            f"  python scripts/build_mamba.py\n"
            f"which supplies the CUDA toolkit from pip and builds for this GPU only.")

    abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    machine = platform.machine()

    urls = {}
    for name, (repo, version) in RELEASES.items():
        tag = (f"{name}-{version}+cu{cuda_major}torch{torch_mm}cxx11abi{abi}"
               f"-{py}-{py}-linux_{machine}.whl")
        urls[name] = f"https://github.com/{repo}/releases/download/v{version}/{tag}"
    return urls


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--print", dest="print_only", action="store_true",
                    help="print the URLs instead of installing")
    args = ap.parse_args()

    urls = wheel_urls()
    for name, url in urls.items():
        print(f"{name}: {url}")
    if args.print_only:
        return

    # --no-deps: these declare a torch requirement that can drag in a different
    # build of torch over the one already installed.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps",
                           *urls.values()])
    print("\ninstalled. verify with: python -m pytest tests/ -q")


if __name__ == "__main__":
    main()
