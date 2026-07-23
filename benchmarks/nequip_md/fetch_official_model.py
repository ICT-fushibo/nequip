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
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the previously downloaded package without network access",
    )
    args = parser.parse_args()

    target = args.output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.verify_only:
        verify_local_package(target, args.source)
        return

    from nequip.model.saved_models.load_utils import _get_model_file_path

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
    source_path = Path(f"{target}.source")
    source_path.write_text(f"{args.source}\n", encoding="utf-8")
    print(f"Official model source: {args.source}")
    print(f"Official model path: {target}")
    print(f"Official model SHA256: {source_hash}")


def verify_local_package(target: Path, expected_source: str) -> None:
    checksum_path = Path(f"{target}.sha256")
    source_path = Path(f"{target}.source")
    for required_path in (target, checksum_path, source_path):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Official model prerequisite is missing: {required_path}\n"
                "Run fetch_official_model.py on an internet-connected login node."
            )

    recorded_source = source_path.read_text(encoding="utf-8").strip()
    if recorded_source != expected_source:
        raise RuntimeError(
            f"Model source mismatch: expected {expected_source!r}, "
            f"found {recorded_source!r}"
        )

    expected_hash = checksum_path.read_text(encoding="utf-8").split()[0]
    actual_hash = sha256(target)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Model checksum mismatch for {target}: "
            f"expected {expected_hash}, found {actual_hash}"
        )
    validate_package(target)
    print(f"Verified offline model source: {recorded_source}")
    print(f"Verified offline model path: {target}")
    print(f"Verified offline model SHA256: {actual_hash}")


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
