#!/usr/bin/env python
"""Fetch a nequip.net model into a stable, checksum-verified local path."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from nequip.model.saved_models.load_utils import _get_model_file_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_package(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Downloaded model does not exist: {path}")
    with zipfile.ZipFile(path) as archive:
        broken_member = archive.testzip()
    if broken_member is not None:
        raise RuntimeError(
            f"Downloaded model ZIP failed integrity checking at: {broken_member}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="nequip.net:mir-group/NequIP-OAM-L:0.1",
        help="Official nequip.net model identifier",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    target = args.output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    # NequIP verifies the repository response and maintains its own download
    # cache. Copy that exact artifact to a predictable path used by E0/E1/B0/B1.
    with _get_model_file_path(args.source) as resolved_source:
        source = Path(resolved_source).resolve()
        validate_package(source)
        source_hash = sha256(source)

        if target.is_file():
            try:
                validate_package(target)
                target_hash = sha256(target)
            except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                print(f"Replacing invalid local model package ({error}): {target}")
                _copy_atomically(source, target)
            else:
                if target_hash == source_hash:
                    print(f"Official model already current: {target}")
                else:
                    print(f"Replacing model whose checksum differs: {target}")
                    _copy_atomically(source, target)
        else:
            _copy_atomically(source, target)

    checksum_path = Path(f"{target}.sha256")
    checksum_path.write_text(f"{source_hash}  {target}\n", encoding="utf-8")
    print(f"Official model source: {args.source}")
    print(f"Official model path: {target}")
    print(f"Official model SHA256: {source_hash}")


def _copy_atomically(source: Path, target: Path) -> None:
    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".download",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copyfile(source, temporary_path)
        validate_package(temporary_path)
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
