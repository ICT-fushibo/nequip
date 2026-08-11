#!/usr/bin/env python3
"""Fair ASE-MD benchmark for the E0/E1/B0/B1 NequIP baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys
import time
from typing import Any

# Make direct execution from a source checkout work even when NequIP has not
# been installed editable.  An installed package still takes the same path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if (REPO_ROOT / "nequip").is_dir():
    sys.path.insert(0, str(REPO_ROOT))

import ase
from ase import units
from ase.io import read
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
import nequip
import numpy as np
import torch


MODE_SPECS = {
    "E0": {
        "model_execution": "eager_e3nn",
        "neighborlist_backend": "matscipy",
    },
    "E1": {
        "model_execution": "torch_compile_e3nn",
        "neighborlist_backend": "matscipy",
    },
    "B0": {
        "model_execution": "aotinductor_openequivariance",
        "neighborlist_backend": "matscipy",
    },
    "B1": {
        "model_execution": "aotinductor_openequivariance",
        "neighborlist_backend": "alchemiops",
    },
}

ENERGY_CHECKPOINTS = (1, 50, 100, 1000)


def build_calculator(
    mode: str,
    model_package: str | None,
    compiled_model: str | None,
    device: str = "cuda",
):
    """Construct exactly one of the four baseline calculators."""
    from nequip.integrations.ase import NequIPCalculator

    mode = mode.upper()
    if mode not in MODE_SPECS:
        raise ValueError(f"Unknown mode {mode!r}; choose from {sorted(MODE_SPECS)}")

    spec = MODE_SPECS[mode]
    backend = spec["neighborlist_backend"]

    if mode in {"E0", "E1"}:
        if model_package is None:
            raise ValueError(f"{mode} requires --model-package")
        if mode == "E1":
            # E1 is a torch.compile baseline, not a CUDA Graph baseline.
            torch._inductor.config.triton.cudagraphs = False

        return NequIPCalculator._from_saved_model(
            model_path=model_package,
            device=device,
            compile_mode="eager" if mode == "E0" else "compile",
            allow_tf32=False,
            chemical_species_to_atom_type_map=True,
            neighborlist_backend=backend,
        )

    if compiled_model is None:
        raise ValueError(f"{mode} requires --compiled-model")

    # The compiled artifact contains OpenEquivariance custom operations.  The
    # library must be imported before the AOTI package is loaded.
    try:
        import openequivariance  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"{mode} requires openequivariance in the active environment"
        ) from exc

    return NequIPCalculator.from_compiled_model(
        compile_path=compiled_model,
        device=device,
        chemical_species_to_atom_type_map=True,
        neighborlist_backend=backend,
    )


def sha256_path(path_string: str | None) -> str | None:
    if path_string is None:
        return None
    path = Path(path_string)
    if not path.exists():
        return None

    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare_velocities(atoms, velocity_mode: str, temperature_k: float, seed: int):
    if velocity_mode == "maxwell":
        rng = np.random.RandomState(seed)
        MaxwellBoltzmannDistribution(
            atoms,
            temperature_K=temperature_k,
            force_temp=True,
            rng=rng,
        )
        Stationary(atoms)
    elif velocity_mode == "zero":
        atoms.set_momenta(np.zeros((len(atoms), 3), dtype=np.float64))
    elif velocity_mode == "keep":
        if not atoms.has("momenta"):
            raise ValueError(
                "--velocity-mode=keep requested, but the input structure has no momenta"
            )
    else:
        raise ValueError(f"Unsupported velocity mode: {velocity_mode}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_dynamics(
    atoms,
    *,
    ensemble: str,
    timestep_fs: float,
    temperature_k: float,
    taut_fs: float,
):
    common = {
        "atoms": atoms,
        "timestep": timestep_fs * units.fs,
        "logfile": None,
        "trajectory": None,
    }
    if ensemble == "nve":
        return VelocityVerlet(**common)
    if ensemble == "nvt":
        return NVTBerendsen(
            **common,
            temperature_K=temperature_k,
            taut=taut_fs * units.fs,
        )
    raise ValueError(f"Unsupported ensemble: {ensemble}")


def capture_atoms_state(atoms) -> dict[str, np.ndarray]:
    return {
        "cell": atoms.cell.array.copy(),
        "pbc": np.asarray(atoms.pbc, dtype=bool).copy(),
        "positions": atoms.get_positions().copy(),
        "momenta": atoms.get_momenta().copy(),
    }


def restore_atoms_state(atoms, state: dict[str, np.ndarray]) -> None:
    atoms.set_cell(state["cell"], scale_atoms=False)
    atoms.set_pbc(state["pbc"])
    atoms.set_positions(state["positions"])
    atoms.set_momenta(state["momenta"])
    if atoms.calc is not None:
        atoms.calc.reset()


def gpu_metadata() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "visible_device_index": 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "name": props.name,
        "compute_capability": [props.major, props.minor],
        "total_memory_bytes": props.total_memory,
        "torch_cuda_version": torch.version.cuda,
    }


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def load_numerical_validation(
    validation_path_string: str | None,
    mode: str,
    system_label: str,
    structure_path: Path,
    structure_index: int,
    timestep_fs: float,
    temperature_k: float,
    ensemble: str,
    thermostat: str,
    taut_fs: float,
    velocity_mode: str,
    seed: int,
    model_package: str | None,
    compiled_model: str | None,
) -> dict[str, Any]:
    if validation_path_string is None:
        return {
            "status": "not_run",
            "validation_passed": None,
            "reference_mode": "E0",
        }

    validation_path = Path(validation_path_string).expanduser().resolve()
    if not validation_path.is_file():
        raise FileNotFoundError(f"Validation result not found: {validation_path}")
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    if payload.get("system_label") != system_label:
        raise ValueError(
            f"Validation system label {payload.get('system_label')!r} does not "
            f"match benchmark label {system_label!r}"
        )
    if payload.get("reference_mode") != "E0":
        raise ValueError(f"Validation file does not use E0: {validation_path}")
    expected_values = {
        "structure_index": structure_index,
        "timestep_fs": timestep_fs,
        "temperature_initialization_k": temperature_k,
        "ensemble": ensemble,
        "thermostat": thermostat,
        "taut_fs": taut_fs,
        "velocity_mode": velocity_mode,
        "seed": seed,
    }
    for key, expected in expected_values.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Validation {key}={payload.get(key)!r} does not match "
                f"benchmark value {expected!r}: {validation_path}"
            )
    if Path(payload["structure"]).expanduser().resolve() != structure_path:
        raise ValueError(
            f"Validation structure does not match benchmark structure: {validation_path}"
        )
    artifact_expectations = {
        "model_package": model_package,
        "compiled_model": compiled_model,
    }
    for key, expected_path_string in artifact_expectations.items():
        if expected_path_string is None:
            continue
        expected_path = Path(expected_path_string).expanduser().resolve()
        actual_path = Path(payload[key]).expanduser().resolve()
        if actual_path != expected_path:
            raise ValueError(
                f"Validation {key} does not match benchmark artifact: "
                f"{validation_path}"
            )
    try:
        mode_result = payload["modes"][mode]
    except KeyError as exc:
        raise ValueError(
            f"Validation file has no result for mode {mode}: {validation_path}"
        ) from exc
    return {
        "status": mode_result["status"],
        "validation_passed": mode_result["validation_passed"],
        "reference_mode": "E0",
        "source": str(validation_path),
        "completed_through_step": payload.get("completed_through_step"),
        "energy_error_definition": payload.get("energy_error_definition"),
        "required_abs_error_lt_ev": payload.get("required_abs_error_lt_ev"),
        "checkpoint_potential_energy_abs_error_ev": mode_result.get(
            "checkpoint_potential_energy_abs_error_ev"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=sorted(MODE_SPECS))
    parser.add_argument("--structure", required=True, help="ASE-readable structure")
    parser.add_argument("--structure-index", type=int, default=-1)
    parser.add_argument("--system-label", default=None)
    parser.add_argument("--model-package", default=None)
    parser.add_argument("--compiled-model", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--ensemble", choices=("nve", "nvt"), default="nvt")
    parser.add_argument("--thermostat", choices=("none", "berendsen"), default="berendsen")
    parser.add_argument("--taut-fs", type=float, default=100.0)
    parser.add_argument(
        "--velocity-mode",
        choices=("maxwell", "keep", "zero"),
        default="maxwell",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--output", default=None, help="JSON result path")
    parser.add_argument(
        "--validation-result",
        default=None,
        help="trajectory-validation JSON to embed in this benchmark result",
    )
    parser.add_argument(
        "--model-sha256",
        default=None,
        help="precomputed hash, avoiding filesystem I/O during parallel runs",
    )
    parser.add_argument(
        "--skip-model-hash",
        action="store_true",
        help="skip hashing the model artifact before the benchmark",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps cannot be negative")
    if args.timestep_fs <= 0:
        raise ValueError("--timestep-fs must be positive")
    if args.temperature_k <= 0:
        raise ValueError("--temperature-k must be positive")
    if args.taut_fs <= 0:
        raise ValueError("--taut-fs must be positive")
    if args.ensemble == "nvt" and args.thermostat != "berendsen":
        raise ValueError("NVT requires --thermostat=berendsen")
    if args.ensemble == "nve" and args.thermostat != "none":
        raise ValueError("NVE requires --thermostat=none")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    structure_path = Path(args.structure).expanduser().resolve()
    if not structure_path.is_file():
        raise FileNotFoundError(f"Structure not found: {structure_path}")

    model_for_mode = (
        args.model_package if args.mode in {"E0", "E1"} else args.compiled_model
    )
    if model_for_mode is None:
        required = "--model-package" if args.mode in {"E0", "E1"} else "--compiled-model"
        raise ValueError(f"{args.mode} requires {required}")
    if not Path(model_for_mode).expanduser().exists():
        raise FileNotFoundError(f"Model artifact not found: {model_for_mode}")

    atoms = read(structure_path, index=args.structure_index)
    if len(atoms) == 0:
        raise ValueError(f"Structure contains no atoms: {structure_path}")
    if np.any(atoms.get_masses() <= 0):
        raise ValueError("All atoms must have positive masses for MD")

    seed_everything(args.seed)
    prepare_velocities(
        atoms,
        velocity_mode=args.velocity_mode,
        temperature_k=args.temperature_k,
        seed=args.seed,
    )
    initial_state = capture_atoms_state(atoms)

    calculator_build_start = time.perf_counter()
    calculator = build_calculator(
        mode=args.mode,
        model_package=args.model_package,
        compiled_model=args.compiled_model,
        device=args.device,
    )
    calculator_build_seconds = time.perf_counter() - calculator_build_start
    atoms.calc = calculator

    # The first force call triggers E1 torch.compile and any OpenEquivariance JIT
    # initialization.  It is deliberately outside both warmup and timed regions.
    first_evaluation_start = time.perf_counter()
    initial_energy_ev = float(atoms.get_potential_energy())
    initial_forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    torch.cuda.synchronize()
    first_evaluation_seconds = time.perf_counter() - first_evaluation_start

    warmup_start = time.perf_counter()
    if args.warmup_steps:
        warmup_dynamics = build_dynamics(
            atoms,
            ensemble=args.ensemble,
            timestep_fs=args.timestep_fs,
            temperature_k=args.temperature_k,
            taut_fs=args.taut_fs,
        )
        warmup_dynamics.run(args.warmup_steps)
        torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_start

    # Warmup must not change the initial state used by the measured trajectory.
    restore_atoms_state(atoms, initial_state)
    dynamics = build_dynamics(
        atoms,
        ensemble=args.ensemble,
        timestep_fs=args.timestep_fs,
        temperature_k=args.temperature_k,
        taut_fs=args.taut_fs,
    )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    checkpoint_energies: dict[str, dict[str, float]] = {}
    completed_steps = 0
    for checkpoint in (step for step in ENERGY_CHECKPOINTS if step <= args.steps):
        dynamics.run(checkpoint - completed_steps)
        potential_energy = float(atoms.get_potential_energy())
        kinetic_energy = float(atoms.get_kinetic_energy())
        checkpoint_energies[str(checkpoint)] = {
            "potential_energy_ev": potential_energy,
            "kinetic_energy_ev": kinetic_energy,
            "total_md_energy_ev": potential_energy + kinetic_energy,
            "temperature_k": float(atoms.get_temperature()),
        }
        completed_steps = checkpoint
    if completed_steps < args.steps:
        dynamics.run(args.steps - completed_steps)
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start

    final_energy_ev = float(atoms.get_potential_energy())
    final_temperature_k = float(atoms.get_temperature())
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())

    seconds_per_step = elapsed_seconds / args.steps
    steps_per_second = args.steps / elapsed_seconds
    ns_per_day = steps_per_second * args.timestep_fs * 86400.0 / 1.0e6

    selected_model_hash = args.model_sha256
    if selected_model_hash is None and not args.skip_model_hash:
        selected_model_hash = sha256_path(str(Path(model_for_mode).expanduser()))
    structure_hash = sha256_path(str(structure_path))

    system_label = args.system_label or structure_path.stem
    numerical_validation = load_numerical_validation(
        validation_path_string=args.validation_result,
        mode=args.mode,
        system_label=system_label,
        structure_path=structure_path,
        structure_index=args.structure_index,
        timestep_fs=args.timestep_fs,
        temperature_k=args.temperature_k,
        ensemble=args.ensemble,
        thermostat=args.thermostat,
        taut_fs=args.taut_fs,
        velocity_mode=args.velocity_mode,
        seed=args.seed,
        model_package=args.model_package,
        compiled_model=args.compiled_model,
    )

    result = {
        "schema_version": 2,
        "mode": args.mode,
        **MODE_SPECS[args.mode],
        "system_label": system_label,
        "structure": str(structure_path),
        "structure_index": args.structure_index,
        "n_atoms": len(atoms),
        "chemical_formula": atoms.get_chemical_formula(),
        "pbc": np.asarray(atoms.pbc, dtype=bool).tolist(),
        "model_package": args.model_package,
        "compiled_model": args.compiled_model,
        "selected_model_sha256": selected_model_hash,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "warmup_seconds": warmup_seconds,
        "timestep_fs": args.timestep_fs,
        "temperature_initialization_k": args.temperature_k,
        "temperature_target_k": args.temperature_k,
        "temperature_K": args.temperature_k,
        "ensemble": args.ensemble,
        "thermostat": args.thermostat,
        "taut_fs": args.taut_fs,
        "velocity_mode": args.velocity_mode,
        "seed": args.seed,
        "repeat": args.repeat,
        "calculator_build_seconds": calculator_build_seconds,
        "first_evaluation_seconds": first_evaluation_seconds,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_step": seconds_per_step,
        "milliseconds_per_step": seconds_per_step * 1000.0,
        "steps_per_second": steps_per_second,
        "ns_per_day": ns_per_day,
        "initial_energy_ev": finite_or_none(initial_energy_ev),
        "initial_force_max_abs_ev_per_a": finite_or_none(
            float(np.abs(initial_forces).max())
        ),
        "initial_force_rms_ev_per_a": finite_or_none(
            float(np.sqrt(np.mean(initial_forces**2)))
        ),
        "final_energy_ev": finite_or_none(final_energy_ev),
        "final_temperature_k": finite_or_none(final_temperature_k),
        "checkpoint_energies": checkpoint_energies,
        "structure_sha256": structure_hash,
        "energy_reference_role": "baseline" if args.mode == "E0" else "candidate",
        "peak_cuda_memory_allocated_bytes": peak_allocated,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
        "numerical_validation": numerical_validation,
        "gpu": gpu_metadata(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "ase_version": ase.__version__,
        "nequip_version": nequip.__version__,
        "nequip_git_revision": git_revision(),
    }
    for step in ENERGY_CHECKPOINTS:
        checkpoint = checkpoint_energies.get(str(step))
        result[f"energy_step_{step}_eV"] = (
            checkpoint["potential_energy_ev"] if checkpoint is not None else None
        )

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(rendered + "\n", encoding="utf-8")
        temporary_path.replace(output_path)
    print(rendered, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
