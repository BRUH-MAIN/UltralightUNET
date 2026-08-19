"""Compile mamba_ssm + causal_conv1d for this GPU, when no prebuilt wheel fits.

``scripts/install_mamba.py`` covers the easy case: the projects publish wheels for
torch 2.6-2.10, and one of those is almost always what you want. This is the
fallback for anything outside that -- a newer torch, an unusual python, or a
machine where the wheel refuses to import.

Two things it does that a bare ``pip install mamba-ssm`` does not:

  * **Supplies a CUDA toolkit.** Building these needs ``nvcc`` and the CUDA
    headers. A torch wheel carries the CUDA *runtime* libraries but no compiler,
    so a stock pip environment fails with "CUDA_HOME environment variable is not
    set". The ``nvidia-cuda-*`` pip packages provide the missing pieces; this
    installs them and assembles the prefix torch's build system expects. No
    system CUDA installation, and no root.

  * **Builds for this GPU only.** mamba's setup.py hardcodes gencode flags for
    every architecture it knows -- nine of them under CUDA 13, each one a full
    compile of every kernel. Restricting to the local compute capability is the
    difference between a few minutes and the better part of an hour, and a cubin
    for sm_90 is of no use on an sm_120 card. Pass --all-archs for a portable
    build.

Usage:
    python scripts/build_mamba.py             # build for this GPU
    python scripts/build_mamba.py --all-archs # every architecture upstream ships
    python scripts/build_mamba.py --jobs 4    # fewer parallel nvcc processes
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import torch

PACKAGES = [
    # (pypi name, import name, version, FORCE_BUILD env var)
    ("mamba-ssm", "mamba_ssm", "2.3.2.post1", "MAMBA_FORCE_BUILD"),
    ("causal-conv1d", "causal_conv1d", "1.6.2.post1", "CAUSAL_CONV1D_FORCE_BUILD"),
]

# nvcc, the CUDA headers, and CCCL (cub/thrust, which the scan kernels include).
#
# The names changed with CUDA 13: the components dropped the -cu12 suffix and
# moved to one merged prefix. The -cu13 names do still resolve on PyPI, but to
# empty 0.0.1 placeholder packages that install nothing and report success --
# so getting this wrong looks exactly like a working install until nvcc is
# missing later.
#
# nvvm and crt are nvcc's own front end and headers. They are pulled in as
# unpinned dependencies, so pip happily pairs a 13.0 ptxas with a 13.3 cicc --
# which then emits PTX ISA 9.3 that the older ptxas rejects outright
# ("Unsupported .version 9.3; current version is '9.0'"). Pin the whole set to
# torch's CUDA minor.
COMPONENTS = ["nvidia-cuda-nvcc", "nvidia-cuda-runtime", "nvidia-cuda-cccl",
              "nvidia-nvvm", "nvidia-cuda-crt"]

# The line every one of these setup.py files has directly after its gencode block.
# Anchoring on it means the patch either lands exactly where intended or fails.
ANCHOR = "    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI"


def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", *args])


def install_toolkit(cuda_version):
    """Install nvcc and the CUDA headers, as one internally consistent set.

    The minor version is deliberately NOT pinned to torch's. Matching the major is
    what matters for linking; matching the *host* toolchain matters just as much,
    and older CUDA minors do not build against recent glibc -- 13.0 fails on
    glibc >= 2.41 with "exception specification is incompatible with that of
    previous function rsqrt", because glibc grew a declaration that CUDA's
    math_functions.h only learned to stand down from later.
    """
    major = cuda_version.split(".")[0]
    suffix = "" if int(major) >= 13 else f"-cu{major}"
    pip("install", "--progress-bar", "off", "-U",
        *[f"{c}{suffix}=={major}.*" for c in COMPONENTS])


def assemble_cuda_home():
    """Build a CUDA_HOME out of whatever the nvidia-* wheels dropped in site-packages.

    CUDA 13's wheels use one merged prefix (nvidia/cu13/{bin,include,lib}); CUDA
    12's use one prefix per component (nvidia/cuda_nvcc/, nvidia/cuda_runtime/,
    ...). Rather than special-case them, symlink every prefix found into a single
    tree. Symlinks, so this costs nothing and never shadows the originals.
    """
    # sys.path order matters and is deliberate: the venv's own site-packages comes
    # before any inherited one, and the first prefix to claim a name wins below. So
    # a toolkit installed here shadows an older one from a parent environment.
    seen, roots = set(), []
    for entry in sys.path:
        if entry.endswith("site-packages") and entry not in seen:
            seen.add(entry)
            roots.extend(sorted((Path(entry) / "nvidia").glob("*")))
    if not roots:
        raise SystemExit("no nvidia-* packages found; run without --no-toolkit")

    # Rebuilt from scratch every run: upgrading a component leaves the previous
    # version's links dangling, and a dangling symlink is the kind of thing that
    # surfaces as an inscrutable compiler error much later.
    home = Path(sys.prefix) / "cuda"
    shutil.rmtree(home, ignore_errors=True)
    for sub in ("bin", "include", "lib64", "nvvm"):
        (home / sub).mkdir(parents=True, exist_ok=True)

    for root in roots:
        for src_name, dst_name in (("bin", "bin"), ("include", "include"),
                                   ("lib", "lib64"), ("lib64", "lib64"),
                                   ("nvvm", "nvvm")):   # nvvm = nvcc's device compiler
            src = root / src_name
            if not src.is_dir():
                continue
            for entry in src.iterdir():
                link = home / dst_name / entry.name
                if not link.is_symlink():
                    link.symlink_to(entry)

    # Extensions link with -lcudart, which needs the unversioned name; the wheels
    # ship only libcudart.so.NN.
    lib = home / "lib64"
    versioned = next(lib.glob("libcudart.so.*"), None)
    if versioned and not (lib / "libcudart.so").exists():
        (lib / "libcudart.so").symlink_to(versioned)

    nvcc = home / "bin" / "nvcc"
    if not nvcc.exists():
        raise SystemExit(f"no nvcc under {home}/bin -- toolkit install incomplete")
    print(f"CUDA_HOME = {home}")
    print(subprocess.run([str(nvcc), "-V"], capture_output=True, text=True)
          .stdout.strip().splitlines()[-1])
    return home


def fetch_sdist(name, version, dest):
    """Download and unpack the sdist straight from PyPI.

    Not `pip download`: that runs the package's own metadata hook first, which for
    these two imports torch and can fail for reasons that have nothing to do with
    fetching a tarball.
    """
    meta = json.load(urllib.request.urlopen(
        f"https://pypi.org/pypi/{name}/{version}/json"))
    url = next(f["url"] for f in meta["urls"] if f["filename"].endswith(".tar.gz"))
    print(f"fetching {url}")
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f"{name}.tar.gz"
    urllib.request.urlretrieve(url, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    return next(p for p in dest.iterdir() if p.is_dir())


def restrict_archs(setup_py, arch):
    """Replace the hardcoded gencode list with this GPU's architecture."""
    src = setup_py.read_text()
    if src.count(ANCHOR) != 1:
        raise SystemExit(f"{setup_py}: anchor line not found exactly once; "
                         "upstream's setup.py has changed, re-check the patch")
    patched = (f'    cc_flag = ["-gencode", "arch=compute_{arch},code=sm_{arch}"]'
               f"  # patched by scripts/build_mamba.py\n")
    setup_py.write_text(src.replace(ANCHOR, patched + ANCHOR))
    print(f"  building for sm_{arch} only")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all-archs", action="store_true",
                    help="build every architecture upstream ships (much slower)")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 4),
                    help="parallel nvcc processes (each wants ~2 GB of RAM)")
    ap.add_argument("--no-toolkit", action="store_true",
                    help="skip installing the nvidia-cuda-* packages")
    args = ap.parse_args()

    if torch.version.cuda is None:
        raise SystemExit("this torch build has no CUDA; mamba_ssm is CUDA-only")
    cc = torch.cuda.get_device_capability(0)
    arch = f"{cc[0]}{cc[1]}"
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) -> sm_{arch}\n")

    if not args.no_toolkit:
        install_toolkit(torch.version.cuda)
    cuda_home = assemble_cuda_home()

    # The venv's bin/ goes on PATH so torch's build system finds ninja: without it
    # setuptools falls back to compiling one .cu at a time and MAX_JOBS does
    # nothing, which is the difference between ~2 minutes and ~15.
    env = {**os.environ, "CUDA_HOME": str(cuda_home), "CUDA_PATH": str(cuda_home),
           "MAX_JOBS": str(args.jobs),
           "PATH": os.pathsep.join([f"{cuda_home}/bin",
                                    str(Path(sys.executable).parent),
                                    os.environ.get("PATH", "")])}

    with tempfile.TemporaryDirectory() as tmp:
        for name, module, version, force_build in PACKAGES:
            print(f"\n=== {name} {version} ===")
            src = fetch_sdist(name, version, Path(tmp) / name)
            if not args.all_archs:
                restrict_archs(src / "setup.py", arch)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--no-build-isolation",
                 "--no-deps", str(src)],
                env={**env, force_build: "TRUE"})

    print("\ninstalled. verify with: python -m pytest tests/ -q")


if __name__ == "__main__":
    main()
