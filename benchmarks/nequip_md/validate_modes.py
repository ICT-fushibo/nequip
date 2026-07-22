#!/usr/bin/env python3
"""Validate optimized MD trajectories against the E0 numerical reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ase import units
from ase.io import read
from ase.md.verlet import VelocityVerlet
import numpy as np
import torch

from benchmark_md import MODE_SPECS, build_calculator, prepare_velocities


CHECKPOINTS = (1, 50, 100, 1000)
REQUIRED_ABS_ERROR_EV = {1: 1.0e-8, 50: 1.0e-6}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--structure-index", type=int, default=-1)
    parser.add_argument("--system-label", default=None)
    parser.add_argument("--model-package", required=True)
    parser.add_argument("--compiled-model", required=True)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument(
        "--velocity-mode",
        choices=("maxwell", "keep", "zero"),
        default="maxwell",
    )
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def evaluate_trajectory(
    mode: str,
    initial_atoms,
    model_package: str,
    compiled_model: str,
    timestep_fs: float,
) -> dict:
    """Run all 1000 steps even when an early checkpoint later fails."""
    atoms = initial_atoms.copy()
    atoms.calc = build_calculator(
        mode=mode,
        model_package=model_package,
        compiled_model=compiled_model,
        device="cuda",
    )

    # This step-0 evaluation performs E1 compilation and OE/AOTI initialization
    # before the trajectory starts. It also provides force/stress diagnostics.
    initial_potential_energy = float(atoms.get_potential_energy())
    initial_forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    initial_stress = np.asarray(atoms.get_stress(voigt=False), dtype=np.float64)
    torch.cuda.synchronize()

    dynamics = VelocityVerlet(
        atoms,
        timestep=timestep_fs * units.fs,
        logfile=None,
        trajectory=None,
    )
    checkpoint_values = {}
    previous_step = 0
    for checkpoint in CHECKPOINTS:
        dynamics.run(checkpoint - previous_step)
        torch.cuda.synchronize()
        potential_energy = float(atoms.get_potential_energy())
        kinetic_energy = float(atoms.get_kinetic_energy())
        checkpoint_values[str(checkpoint)] = {
            "potential_energy_ev": potential_energy,
            "kinetic_energy_ev": kinetic_energy,
            "total_md_energy_ev": potential_energy + kinetic_energy,
        }
        previous_step = checkpoint

    return {
        "initial_potential_energy_ev": initial_potential_energy,
        "initial_forces_ev_per_a": initial_forces,
        "initial_stress_ev_per_a3": initial_stress,
        "checkpoints": checkpoint_values,
    }


def finite_trajectory(result: dict) -> bool:
    values = [result["initial_potential_energy_ev"]]
    values.extend(result["initial_forces_ev_per_a"].ravel())
    values.extend(result["initial_stress_ev_per_a3"].ravel())
    for checkpoint in result["checkpoints"].values():
        values.extend(checkpoint.values())
    return bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())


def main() -> int:
    args = parse_args()
    if args.timestep_fs <= 0:
        raise ValueError("--timestep-fs must be positive")
    structure_path = Path(args.structure).expanduser().resolve()
    atoms = read(structure_path, index=args.structure_index)
    prepare_velocities(
        atoms,
        velocity_mode=args.velocity_mode,
        temperature_k=args.temperature_k,
        seed=args.seed,
    )

    predictions = {}
    for mode in MODE_SPECS:
        print(f"Running {mode} numerical trajectory through step 1000", flush=True)
        predictions[mode] = evaluate_trajectory(
            mode=mode,
            initial_atoms=atoms,
            model_package=args.model_package,
            compiled_model=args.compiled_model,
            timestep_fs=args.timestep_fs,
        )

    reference = predictions["E0"]
    reference_finite = finite_trajectory(reference)
    mode_results = {
        "E0": {
            "status": "reference" if reference_finite else "reference_nonfinite",
            "validation_passed": reference_finite,
            "finite": reference_finite,
            "checkpoint_potential_energy_ev": {
                step: values["potential_energy_ev"]
                for step, values in reference["checkpoints"].items()
            },
            "checkpoint_total_md_energy_ev": {
                step: values["total_md_energy_ev"]
                for step, values in reference["checkpoints"].items()
            },
            "checkpoint_potential_energy_abs_error_ev": {
                str(step): 0.0 for step in CHECKPOINTS
            },
        }
    }

    all_passed = reference_finite
    for mode in ("E1", "B0", "B1"):
        current = predictions[mode]
        finite = finite_trajectory(current)
        checkpoint_details = {}
        potential_errors = {}
        total_md_errors = {}
        required_checks_passed = finite

        for step in CHECKPOINTS:
            key = str(step)
            reference_potential = reference["checkpoints"][key]["potential_energy_ev"]
            current_potential = current["checkpoints"][key]["potential_energy_ev"]
            potential_error = abs(current_potential - reference_potential)
            total_md_error = abs(
                current["checkpoints"][key]["total_md_energy_ev"]
                - reference["checkpoints"][key]["total_md_energy_ev"]
            )
            threshold = REQUIRED_ABS_ERROR_EV.get(step)
            # The requirement says "less than", so equality does not pass.
            checkpoint_passed = threshold is None or potential_error < threshold
            if threshold is not None:
                required_checks_passed = required_checks_passed and checkpoint_passed
            potential_errors[key] = potential_error
            total_md_errors[key] = total_md_error
            checkpoint_details[key] = {
                "reference_potential_energy_ev": reference_potential,
                "optimized_potential_energy_ev": current_potential,
                "potential_energy_abs_error_ev": potential_error,
                "total_md_energy_abs_error_ev": total_md_error,
                "required_abs_error_lt_ev": threshold,
                "required_checkpoint": threshold is not None,
                "passed": checkpoint_passed,
            }

        initial_force_difference = np.abs(
            current["initial_forces_ev_per_a"]
            - reference["initial_forces_ev_per_a"]
        )
        initial_stress_difference = np.abs(
            current["initial_stress_ev_per_a3"]
            - reference["initial_stress_ev_per_a3"]
        )
        all_passed = all_passed and required_checks_passed
        mode_results[mode] = {
            "status": "passed" if required_checks_passed else "failed",
            "validation_passed": required_checks_passed,
            "finite": finite,
            "checkpoint_potential_energy_ev": {
                step: values["potential_energy_ev"]
                for step, values in current["checkpoints"].items()
            },
            "checkpoint_total_md_energy_ev": {
                step: values["total_md_energy_ev"]
                for step, values in current["checkpoints"].items()
            },
            "checkpoint_potential_energy_abs_error_ev": potential_errors,
            "checkpoint_total_md_energy_abs_error_ev": total_md_errors,
            "checkpoint_details": checkpoint_details,
            "step0_potential_energy_abs_error_ev": abs(
                current["initial_potential_energy_ev"]
                - reference["initial_potential_energy_ev"]
            ),
            "step0_force_max_abs_error_ev_per_a": float(
                initial_force_difference.max()
            ),
            "step0_force_rms_error_ev_per_a": float(
                np.sqrt(np.mean(initial_force_difference**2))
            ),
            "step0_stress_max_abs_error_ev_per_a3": float(
                initial_stress_difference.max()
            ),
        }

    result = {
        "schema_version": 2,
        "reference_mode": "E0",
        "system_label": args.system_label or structure_path.stem,
        "structure": str(structure_path),
        "structure_index": args.structure_index,
        "n_atoms": len(atoms),
        "model_package": str(Path(args.model_package).expanduser().resolve()),
        "compiled_model": str(Path(args.compiled_model).expanduser().resolve()),
        "timestep_fs": args.timestep_fs,
        "temperature_initialization_k": args.temperature_k,
        "velocity_mode": args.velocity_mode,
        "seed": args.seed,
        "energy_error_definition": (
            "absolute difference in total model potential energy; unit eV; "
            "not normalized by atom count"
        ),
        "trajectory_comparison": (
            "each mode runs an independent velocity-Verlet trajectory from the "
            "identical initial positions and momenta"
        ),
        "checkpoints": list(CHECKPOINTS),
        "required_abs_error_lt_ev": {
            str(step): threshold for step, threshold in REQUIRED_ABS_ERROR_EV.items()
        },
        "modes": mode_results,
        "all_optimized_modes_passed": all_passed,
        "completed_through_step": CHECKPOINTS[-1],
    }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(rendered + "\n", encoding="utf-8")
        temporary_path.replace(output_path)
    print(rendered)
    if not all_passed:
        print(
            "Numerical validation failed for at least one optimized mode; "
            "all modes still completed 1000 steps and results were saved.",
            file=sys.stderr,
        )
    # A numerical threshold failure is data, not a runtime failure. Returning
    # zero lets the full benchmark pipeline continue and mark affected results.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
