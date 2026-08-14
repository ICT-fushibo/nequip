"""Stable NequIP MD route for the shared acceleration benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from md_benchmark.md_route import (
    MDRunRequest,
    MDRunResult,
    configure_torch_baseline,
    run_ase_baseline,
    run_optimized_stage,
)


BASELINE_MODES = ("E0", "E1", "B0", "B1")


def _normalized_mode(backend: str) -> str:
    """Map generic outer-runner names onto the retained NequIP baselines."""
    return {"EAGER": "E0", "COMPILE": "E1"}.get(
        backend.upper(), backend.upper()
    )


def _compiled_model_path(options: dict[str, Any], mode: str) -> str:
    """Return the AOTI artifact without overloading the official model path.

    ``MDRunRequest.model_path`` is always the source ``.nequip.zip`` package for
    provenance and cross-mode comparisons.  AOTInductor artifacts are tied to
    the PyTorch/CUDA build that produced them, so B0/B1 receive their artifact
    separately through route options.
    """
    value = options.get("compiled_model_path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"NequIP {mode} requires route option 'compiled_model_path'; "
            "keep --model-path pointed at the official .nequip.zip package"
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"NequIP compiled model not found: {path}")
    if path.suffix != ".pt2":
        raise ValueError(
            f"NequIP {mode} requires an AOTInductor .pt2 artifact, got {path}"
        )
    return str(path)


def run_md(request: MDRunRequest) -> MDRunResult:
    if request.model != "nequip":
        raise ValueError(f"nequip.md_route does not own model {request.model!r}")
    if request.stage != "baseline":
        return run_optimized_stage(request, module_prefix="nequip.md_stages")

    from nequip.integrations.ase import NequIPCalculator

    # This baseline is the numerical reference.  Optimized stages must make
    # any reduced-precision policy explicit rather than inheriting a process
    # or framework default.
    configure_torch_baseline()

    # Keep the shared CLI's generic backend names useful while preserving the
    # E0/E1/B0/B1 names used by the existing NequIP benchmark.
    mode = _normalized_mode(request.backend)
    if mode in {"E0", "E1"}:
        if mode == "E1":
            # E1 is a torch.compile comparison, not a CUDA Graph stage.
            import torch

            torch._inductor.config.triton.cudagraphs = False
        calculator = NequIPCalculator._from_saved_model(
            model_path=request.model_path,
            device=request.config.device,
            compile_mode="eager" if mode == "E0" else "compile",
            allow_tf32=False,
            chemical_species_to_atom_type_map=True,
            neighborlist_backend="matscipy",
        )
    elif mode in {"B0", "B1"}:
        # Older AOTI packages do not persist their custom-op dependency, so it
        # must be registered before the artifact is loaded.
        try:
            import openequivariance  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"NequIP {mode} requires openequivariance"
            ) from exc
        if mode == "B0":
            neighborlist_backend = "matscipy"
        else:
            neighborlist_backend = "alchemiops"
        compiled_model_path = _compiled_model_path(request.options, mode)
        calculator = NequIPCalculator.from_compiled_model(
            compile_path=compiled_model_path,
            device=request.config.device,
            chemical_species_to_atom_type_map=True,
            neighborlist_backend=neighborlist_backend,
        )
    else:
        raise ValueError(
            "NequIP baseline backend must be eager/E0, compile/E1, B0, or B1"
        )
    metadata: dict[str, Any] = {
        "nequip_mode": mode,
        "source_model_path": str(Path(request.model_path).expanduser().resolve()),
        "tf32_enabled": False,
        "cuda_graph_enabled": False if mode == "E1" else None,
    }
    if mode in {"B0", "B1"}:
        metadata["compiled_model_path"] = compiled_model_path
        metadata["compiled_model_kind"] = "aotinductor_openequivariance"
    return run_ase_baseline(request, calculator, metadata=metadata)
