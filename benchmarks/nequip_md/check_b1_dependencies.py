#!/usr/bin/env python
"""Fail fast when the B1 GPU neighbor-list dependency is unavailable."""

from __future__ import annotations

import importlib.metadata
import sys


def main() -> None:
    if sys.version_info < (3, 11):
        raise RuntimeError(
            "B1 requires the NVIDIA nvalchemi-toolkit-ops package, whose "
            "official releases require Python >= 3.11. "
            f"The active interpreter is Python {sys.version.split()[0]}."
        )

    try:
        package_version = importlib.metadata.version("nvalchemi-toolkit-ops")
        warp_version = importlib.metadata.version("warp-lang")
        import nvalchemiops  # noqa: F401
        from nvalchemiops.torch.neighbors import batch_cell_list  # noqa: F401
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise RuntimeError(
            "B1 requires nvalchemi-toolkit-ops with its PyTorch bindings. "
            "Install a pinned official release in the benchmark environment."
        ) from error

    print(
        "B1 dependency check passed: "
        f"python={sys.version.split()[0]}, "
        f"nvalchemi-toolkit-ops={package_version}, "
        f"warp-lang={warp_version}"
    )


if __name__ == "__main__":
    main()
