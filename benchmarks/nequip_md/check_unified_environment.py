#!/usr/bin/env python3
"""Static preflight for the shared Python 3.12/Torch 2.11/CUDA 12.6 setup.

This deliberately does not claim that an AOTInductor artifact is usable.  The
artifact must still be compiled and smoke-tested on the target H100 because it
contains generated code and OpenEquivariance custom operations.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvcc_version(cuda_home: str | None) -> str | None:
    if not cuda_home:
        return None
    executable = Path(cuda_home) / "bin" / "nvcc"
    if not executable.is_file():
        return None
    try:
        output = subprocess.check_output(
            [str(executable), "--version"], text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.strip().splitlines()[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-accelerators",
        action="store_true",
        help="also require OpenEquivariance and the alchemiops neighbor list",
    )
    args = parser.parse_args()

    import os
    import torch
    import nequip

    failures: list[str] = []
    warnings: list[str] = []
    repo_root = Path(__file__).resolve().parents[2]
    nequip_import = Path(nequip.__file__).resolve()
    if repo_root not in nequip_import.parents:
        failures.append(
            "nequip is not imported from this checkout; run pip install -e . "
            f"(import resolved to {nequip_import})"
        )
    if sys.version_info[:2] != (3, 12):
        failures.append(f"expected Python 3.12, found {platform.python_version()}")
    if not torch.__version__.startswith("2.11.0"):
        failures.append(f"expected torch 2.11.0, found {torch.__version__}")
    if torch.version.cuda != "12.6":
        failures.append(f"expected torch CUDA 12.6, found {torch.version.cuda}")
    if not torch.cuda.is_available():
        failures.append("torch.cuda.is_available() is false")

    oeq_version = _distribution_version("openequivariance")
    alchemiops_version = _distribution_version("nvalchemi-toolkit-ops")
    if args.require_accelerators:
        if oeq_version is None:
            failures.append("openequivariance is not installed")
        else:
            try:
                import openequivariance  # noqa: F401
            except Exception as exc:
                failures.append(f"openequivariance import failed: {exc}")
        if alchemiops_version is None:
            failures.append("nvalchemi-toolkit-ops is not installed")
        else:
            try:
                from nvalchemiops.torch.neighbors import batch_cell_list  # noqa: F401
            except Exception as exc:
                failures.append(f"nvalchemiops torch API import failed: {exc}")

    cuda_home = os.environ.get("CUDA_HOME")
    nvcc = _nvcc_version(cuda_home)
    if args.require_accelerators and nvcc is None:
        failures.append("CUDA_HOME/bin/nvcc is unavailable")
    elif args.require_accelerators and "release 12.6" not in nvcc:
        failures.append(f"expected CUDA_HOME nvcc 12.6, found: {nvcc}")
    elif nvcc is None:
        warnings.append("CUDA_HOME/bin/nvcc not found; E0 can run but B0/B1 cannot compile")

    payload = {
        "status": "passed" if not failures else "failed",
        "scope": "static_preflight_not_runtime_compatibility_proof",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nequip": nequip.__version__,
        "nequip_import": str(nequip_import),
        "openequivariance": oeq_version,
        "nvalchemi_toolkit_ops": alchemiops_version,
        "cuda_home": cuda_home,
        "nvcc": nvcc,
        "failures": failures,
        "warnings": warnings,
        "required_runtime_followup": (
            "compile the AOTI artifact, then run E0/B0/B1 numerical smoke tests "
            "on the target H100"
        ),
    }
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
