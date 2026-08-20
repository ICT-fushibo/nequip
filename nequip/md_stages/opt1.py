"""NequIP Opt1: eager model inference in a GPU-resident NVT loop.

The scientific model is the same eager e3nn model loaded from the official
``.nequip.zip`` package used by E0.  Opt1 changes only execution residency:
positions, momenta, forces, thermostat state, model inference, and the
AlchemiOps neighbor list remain on one CUDA device.  It deliberately does not
enable AOTInductor, OpenEquivariance, ``torch.compile``, CUDA Graphs, TF32, or
model-specific fusion.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import ase.io
import numpy as np
import torch
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from torch import Tensor

from md_benchmark.md_route import (
    MDObservation,
    MDRunRequest,
    MDRunResult,
    validate_result,
)


OPT1_POLICY = {
    "gpu_resident": True,
    "model_compile": False,
    "aotinductor": False,
    "open_equivariance": False,
    "cuda_graph": False,
    "model_specific_fusion": False,
    "tf32": False,
    "neighbor_list_backend": "alchemiops",
}

_FOURTH_ORDER_COEFFS = (
    1.0 / (2.0 - 2.0 ** (1.0 / 3.0)),
    -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0)),
    1.0 / (2.0 - 2.0 ** (1.0 / 3.0)),
)


@dataclass
class ModelOutput:
    energy: Tensor
    forces: Tensor
    stress: Tensor | None


class Evaluator(Protocol):
    def __call__(self, positions: Tensor) -> ModelOutput: ...


class EagerNequIPTorchSimEvaluator:
    """Persistent eager NequIP/TorchSim evaluator with a CUDA neighbor list."""

    def __init__(
        self,
        atoms: Atoms,
        model_path: str,
        *,
        device: torch.device,
        require_stress: bool,
    ) -> None:
        try:
            import torch_sim as ts
        except ImportError as exc:
            raise RuntimeError(
                "NequIP Opt1 requires torch-sim-atomistic; install the same "
                "TorchSim version used by the unified md_opt environment"
            ) from exc

        try:
            import nvalchemiops  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "NequIP Opt1 requires nvalchemiops for its CUDA-resident "
                "AlchemiOps neighbor list"
            ) from exc

        from nequip.integrations.torchsim import NequIPTorchSimCalc

        atomic_numbers = torch.as_tensor(
            atoms.get_atomic_numbers(), dtype=torch.long, device=device
        )
        system_idx = torch.zeros(len(atoms), dtype=torch.long, device=device)
        try:
            calculator = NequIPTorchSimCalc.from_saved_model(
                model_path=model_path,
                device=device,
                chemical_species_to_atom_type_map=True,
                allow_tf32=False,
                compile_mode="eager",
                neighborlist_backend="alchemiops",
                atomic_numbers=atomic_numbers,
                system_idx=system_idx,
            )
        except Exception as exc:
            raise RuntimeError(
                "NequIP Opt1 could not load the official .nequip.zip package "
                "as an eager TorchSim model in this PyTorch runtime. Opt1 "
                "will not substitute an AOTI/OpenEquivariance .pt2 artifact; "
                "inspect the chained loader error or use a compatible NequIP "
                "saved-package loader."
            ) from exc

        _assert_plain_eager_model(calculator.model)
        calculator.compute_forces = True
        calculator.compute_stress = require_stress
        self.calculator = calculator
        self.device = device
        self.num_atoms = len(atoms)
        self.require_stress = require_stress

        # TorchSim creates the immutable cell/PBC/system-index tensors once.
        # Atomic numbers live in the calculator so its forward path can reuse
        # them without a per-step torch.equal synchronization.
        self.sim_state = ts.io.atoms_to_state(
            [atoms], device=device, dtype=torch.float64
        )
        self.sim_state.atomic_numbers = None
        calculator.set_static_geometry(
            self.sim_state.row_vector_cell, self.sim_state.pbc
        )

        model_dtype = getattr(calculator.model, "model_dtype", None)
        self.model_dtype = (
            str(model_dtype).removeprefix("torch.")
            if isinstance(model_dtype, torch.dtype)
            else str(model_dtype)
        )
        self.parameter_dtypes = sorted(
            {
                str(parameter.dtype).removeprefix("torch.")
                for parameter in calculator.model.parameters()
            }
        )

    def __call__(self, positions: Tensor) -> ModelOutput:
        if positions.device != self.device or positions.dtype != torch.float64:
            raise ValueError(
                "NequIP Opt1 positions must be FP64 tensors on the selected "
                "CUDA device"
            )
        if positions.shape != (self.num_atoms, 3):
            raise ValueError(
                f"Expected positions shape {(self.num_atoms, 3)}, "
                f"got {tuple(positions.shape)}"
            )

        # Rebind a CUDA tensor; no copy or host conversion occurs here.
        self.sim_state.positions = positions
        outputs = self.calculator(self.sim_state)
        missing = [name for name in ("energy", "forces") if name not in outputs]
        if missing:
            raise RuntimeError(f"NequIP eager model omitted outputs {missing}")
        stress = outputs.get("stress") if self.require_stress else None
        if stress is not None:
            stress = stress.reshape(-1, 3, 3)[0].detach().to(torch.float64)
        return ModelOutput(
            energy=outputs["energy"].reshape(-1)[0].detach().to(torch.float64),
            forces=outputs["forces"].reshape(self.num_atoms, 3).detach().to(
                torch.float64
            ),
            stress=stress,
        )


def _assert_plain_eager_model(model: torch.nn.Module) -> None:
    """Reject accelerators that belong to later optimization stages."""
    if isinstance(model, torch.jit.ScriptModule) or hasattr(model, "_orig_mod"):
        raise RuntimeError("NequIP Opt1 requires a regular eager nn.Module")
    forbidden_modules = []
    for module in model.modules():
        owner = type(module).__module__.lower()
        custom_ops = {
            str(name).lower()
            for name in getattr(module, "_nequip_custom_ops_libs", ())
        }
        if (
            "openequivariance" in owner
            or "cuequivariance" in owner
            or "openequivariance" in custom_ops
            or "cuequivariance_torch" in custom_ops
        ):
            forbidden_modules.append(f"{owner}.{type(module).__name__}")
    if forbidden_modules:
        raise RuntimeError(
            "NequIP Opt1 forbids accelerated equivariance backends; found "
            + ", ".join(sorted(set(forbidden_modules)))
        )


@dataclass
class GPUMDState:
    """Mutable FP64 MD state whose tensors remain on one CUDA device."""

    positions: Tensor
    momenta: Tensor
    output: ModelOutput | None = None


class BerendsenIntegrator:
    """CUDA port of unconstrained ASE 3.29 ``NVTBerendsen``."""

    name = "berendsen"

    def __init__(
        self,
        masses: Tensor,
        *,
        timestep_fs: float,
        temperature_k: float,
        thermostat_time_fs: float,
        degrees_of_freedom: int,
    ) -> None:
        self.masses = masses.reshape(-1, 1)
        self.dt = float(timestep_fs) * units.fs
        self.target_temperature = float(temperature_k)
        self.taut = float(thermostat_time_fs) * units.fs
        self.degrees_of_freedom = int(degrees_of_freedom)
        if self.degrees_of_freedom <= 0:
            raise ValueError("degrees of freedom must be positive")

    def kinetic_energy(self, momenta: Tensor) -> Tensor:
        return (0.5 * momenta.square() / self.masses).sum()

    def step(self, state: GPUMDState, evaluator: Evaluator) -> None:
        temperature = (
            2.0
            * self.kinetic_energy(state.momenta)
            / (self.degrees_of_freedom * units.kB)
        ).clamp_min(1.0e-12)
        scale = torch.sqrt(
            1.0
            + (self.target_temperature / temperature - 1.0)
            * (self.dt / self.taut)
        ).clamp(min=0.9, max=1.1)
        momenta = state.momenta * scale
        _ensure_evaluated(state, evaluator)
        assert state.output is not None
        momenta = momenta + 0.5 * self.dt * state.output.forces
        # ASE NVTBerendsen defaults to fixcm=True.
        momenta = momenta - momenta.sum(dim=0, keepdim=True) / float(
            momenta.shape[0]
        )
        positions = state.positions + self.dt * momenta / self.masses
        output = evaluator(positions)
        momenta = momenta + 0.5 * self.dt * output.forces
        state.positions = positions
        state.momenta = momenta
        state.output = output


class NoseHooverChainIntegrator:
    """FP64 CUDA port of ASE 3.29 NHC (tchain=3, tloop=1)."""

    name = "nose_hoover_chain"

    def __init__(
        self,
        masses: Tensor,
        *,
        timestep_fs: float,
        temperature_k: float,
        thermostat_time_fs: float,
        chain_length: int = 3,
        chain_loops: int = 1,
    ) -> None:
        if chain_length < 1 or chain_loops < 1:
            raise ValueError("Nose-Hoover chain length/loops must be positive")
        self.masses = masses.reshape(-1, 1)
        self.num_atoms = int(self.masses.numel())
        self.dt = float(timestep_fs) * units.fs
        self.kT = float(temperature_k) * units.kB
        self.tdamp = float(thermostat_time_fs) * units.fs
        self.chain_length = int(chain_length)
        self.chain_loops = int(chain_loops)
        self.Q = self.masses.new_full((self.chain_length,), self.kT * self.tdamp**2)
        self.Q[0] *= 3.0 * self.num_atoms
        self.eta = self.masses.new_zeros(self.chain_length)
        self.p_eta = self.masses.new_zeros(self.chain_length)

    def kinetic_energy(self, momenta: Tensor) -> Tensor:
        return (0.5 * momenta.square() / self.masses).sum()

    def _integrate_p_eta_j(
        self, momenta: Tensor, j: int, delta2: float, delta4: float
    ) -> None:
        if j < self.chain_length - 1:
            self.p_eta[j] *= torch.exp(
                -delta4 * self.p_eta[j + 1] / self.Q[j + 1]
            )
        if j == 0:
            g_j = (
                momenta.square() / self.masses
            ).sum() - 3.0 * self.num_atoms * self.kT
        else:
            g_j = self.p_eta[j - 1].square() / self.Q[j - 1] - self.kT
        self.p_eta[j] += delta2 * g_j
        if j < self.chain_length - 1:
            self.p_eta[j] *= torch.exp(
                -delta4 * self.p_eta[j + 1] / self.Q[j + 1]
            )

    def _integrate_loop(self, momenta: Tensor, delta: float) -> Tensor:
        delta2, delta4 = delta / 2.0, delta / 4.0
        for j in reversed(range(self.chain_length)):
            self._integrate_p_eta_j(momenta, j, delta2, delta4)
        self.eta += delta * self.p_eta / self.Q
        momenta = momenta * torch.exp(-delta * self.p_eta[0] / self.Q[0])
        for j in range(self.chain_length):
            self._integrate_p_eta_j(momenta, j, delta2, delta4)
        return momenta

    def _integrate_chain(self, momenta: Tensor, delta: float) -> Tensor:
        for _ in range(self.chain_loops):
            for coefficient in _FOURTH_ORDER_COEFFS:
                momenta = self._integrate_loop(
                    momenta, coefficient * delta / self.chain_loops
                )
        return momenta

    def step(self, state: GPUMDState, evaluator: Evaluator) -> None:
        dt2 = self.dt / 2.0
        momenta = self._integrate_chain(state.momenta, dt2)
        _ensure_evaluated(state, evaluator)
        assert state.output is not None
        momenta = momenta + dt2 * state.output.forces
        positions = state.positions + self.dt * momenta / self.masses
        output = evaluator(positions)
        momenta = momenta + dt2 * output.forces
        momenta = self._integrate_chain(momenta, dt2)
        state.positions = positions
        state.momenta = momenta
        state.output = output


def _ensure_evaluated(state: GPUMDState, evaluator: Evaluator) -> None:
    if state.output is None:
        state.output = evaluator(state.positions)


def _build_integrator(request: MDRunRequest, masses: Tensor):
    config = request.config
    if config.integrator == "berendsen":
        return BerendsenIntegrator(
            masses,
            timestep_fs=config.timestep_fs,
            temperature_k=config.temperature_k,
            thermostat_time_fs=config.thermostat_time_fs,
            degrees_of_freedom=request.atoms.get_number_of_degrees_of_freedom(),
        )
    if config.integrator == "nose_hoover_chain":
        return NoseHooverChainIntegrator(
            masses,
            timestep_fs=config.timestep_fs,
            temperature_k=config.temperature_k,
            thermostat_time_fs=config.thermostat_time_fs,
        )
    raise ValueError(f"NequIP Opt1 does not support {config.integrator!r}")


def _validate_request(request: MDRunRequest) -> Path:
    if request.model != "nequip" or request.stage != "opt1":
        raise ValueError(
            "nequip.md_stages.opt1 owns nequip/opt1, got "
            f"{request.model}/{request.stage}"
        )
    if request.backend != "gpu-resident":
        raise ValueError("NequIP opt1 backend must be 'gpu-resident'")
    if request.config.device.split(":", maxsplit=1)[0] != "cuda":
        raise ValueError("NequIP Opt1 is CUDA-only; CPU fallback is forbidden")
    if request.config.dtype != "float64":
        raise ValueError("NequIP Opt1 requires --dtype float64 for the MD state")
    if request.atoms.constraints:
        raise NotImplementedError("NequIP Opt1 does not ignore ASE constraints")
    if len(request.atoms) < 2:
        raise ValueError("NVT MD requires at least two atoms")
    path = Path(request.model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not path.name.endswith(".nequip.zip"):
        raise ValueError(
            "NequIP Opt1 requires the official .nequip.zip saved package; "
            ".pt2/AOTI artifacts belong to later or retained control modes"
        )
    if request.options.get("compiled_model_path"):
        raise ValueError("NequIP Opt1 forbids compiled_model_path")
    if request.options.get("neighborlist_backend", "alchemiops") != "alchemiops":
        raise ValueError("NequIP Opt1 requires the CUDA AlchemiOps neighbor list")
    for key in ("compile", "cuda_graph", "model_specific_fusion", "allow_tf32"):
        if request.options.get(key):
            raise ValueError(f"NequIP Opt1 forbids route option {key!r}")
    return path


def _configure_precision() -> None:
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise RuntimeError(
            "NequIP Opt1 forbids TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1"
        )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _configure_output(request: MDRunRequest) -> tuple[Path | None, Path | None]:
    config = request.config
    if config.collect_trajectory and config.record_interval < 1:
        raise ValueError("collect_trajectory requires record_interval >= 1")
    if request.output_path is None:
        return None, None
    if not config.collect_trajectory:
        raise ValueError("output_path requires collect_trajectory=True")
    target = Path(request.output_path).expanduser().resolve()
    partial = target.with_name(f"{target.stem}.part.extxyz")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not request.options.get("overwrite", False):
        raise FileExistsError(f"Refusing to overwrite {target}")
    for stale in (target, partial):
        if stale.exists():
            stale.unlink()
    return target, partial


def _require_output(state: GPUMDState) -> ModelOutput:
    if state.output is None:
        raise RuntimeError("MD state has not been evaluated")
    return state.output


def _frame(
    template: Atoms,
    state: GPUMDState,
    *,
    step: int,
    require_stress: bool,
) -> Atoms:
    output = _require_output(state)
    if require_stress and output.stress is None:
        raise RuntimeError("Matbench trajectory requires NequIP stress output")
    frame = template.copy()
    frame.set_positions(state.positions.detach().cpu().numpy())
    frame.set_momenta(state.momenta.detach().cpu().numpy())
    results: dict[str, Any] = {
        "energy": float(output.energy.item()),
        "forces": output.forces.detach().cpu().numpy(),
    }
    if output.stress is not None:
        results["stress"] = output.stress.detach().cpu().numpy()
    frame.info["md_step"] = step
    frame.calc = SinglePointCalculator(frame, **results)
    return frame


def _observation(
    state: GPUMDState, *, step: int, masses: Tensor
) -> MDObservation:
    output = _require_output(state)
    kinetic = (0.5 * state.momenta.square() / masses.reshape(-1, 1)).sum()
    return MDObservation(
        step=step,
        potential_energy_ev=float(output.energy.item()),
        kinetic_energy_ev=float(kinetic.item()),
        forces_ev_per_a=output.forces.detach().cpu().numpy().copy(),
        positions_a=state.positions.detach().cpu().numpy().copy(),
    )


def _validate_finite(state: GPUMDState) -> None:
    output = _require_output(state)
    tensors = {
        "positions": state.positions,
        "momenta": state.momenta,
        "energy": output.energy,
        "forces": output.forces,
    }
    if output.stress is not None:
        tensors["stress"] = output.stress
    invalid = [
        name for name, value in tensors.items() if not bool(torch.isfinite(value).all())
    ]
    if invalid:
        raise FloatingPointError(f"NequIP Opt1 final state has non-finite {invalid}")


def _distribution_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def run_md(request: MDRunRequest) -> MDRunResult:
    """Run NequIP Opt1 through the stable shared MD route."""

    model_path = _validate_request(request)
    if not torch.cuda.is_available():
        raise RuntimeError("NequIP Opt1 requested CUDA, but CUDA is unavailable")
    _configure_precision()
    device = torch.device(request.config.device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    config = request.config
    atoms = request.atoms.copy()
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=config.temperature_k,
        rng=np.random.default_rng(config.seed),
    )
    positions0 = torch.as_tensor(
        np.asarray(atoms.positions), dtype=torch.float64, device=device
    ).clone()
    momenta0 = torch.as_tensor(
        np.asarray(atoms.get_momenta()), dtype=torch.float64, device=device
    ).clone()
    masses = torch.as_tensor(
        np.asarray(atoms.get_masses()), dtype=torch.float64, device=device
    ).clone()
    evaluator = EagerNequIPTorchSimEvaluator(
        atoms,
        str(model_path),
        device=device,
        require_stress=config.collect_trajectory,
    )

    # Warmup owns disposable state and thermostat variables.  Production is
    # reconstructed from the exact initial positions/momenta, matching the ASE
    # baseline's restore-and-rebuild behavior.
    if config.warmup_steps:
        warm_state = GPUMDState(positions0.clone(), momenta0.clone())
        warm_integrator = _build_integrator(request, masses)
        for _ in range(config.warmup_steps):
            warm_integrator.step(warm_state, evaluator)
        torch.cuda.synchronize(device)

    state = GPUMDState(positions0.clone(), momenta0.clone())
    integrator = _build_integrator(request, masses)
    target_path, partial_path = _configure_output(request)
    trajectory: list[Atoms] | None = (
        [] if config.collect_trajectory and target_path is None else None
    )
    observations: list[MDObservation] = []
    observation_steps = set(config.observation_steps)

    def record_frame(step: int) -> None:
        frame = _frame(atoms, state, step=step, require_stress=True)
        if partial_path is not None:
            ase.io.write(partial_path, frame, append=True, format="extxyz")
        else:
            assert trajectory is not None
            trajectory.append(frame)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    _ensure_evaluated(state, evaluator)
    # ASE Dynamics observers include the initial frame at nsteps=0.
    if config.collect_trajectory:
        record_frame(0)
    for step in range(1, config.steps + 1):
        integrator.step(state, evaluator)
        if config.collect_statistics and step in observation_steps:
            observations.append(_observation(state, step=step, masses=masses))
        if config.collect_trajectory and step % config.record_interval == 0:
            record_frame(step)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1.0e9
    _validate_finite(state)

    if target_path is not None:
        assert partial_path is not None
        os.replace(partial_path, target_path)
    final_atoms = _frame(
        atoms, state, step=config.steps, require_stress=False
    )
    result = MDRunResult(
        model=request.model,
        stage=request.stage,
        completed_steps=config.steps,
        elapsed_s=elapsed,
        peak_cuda_memory_gb=peak_memory_gb,
        final_atoms=final_atoms,
        observations=observations,
        trajectory=trajectory,
        trajectory_path=str(target_path) if target_path is not None else None,
        metadata={
            "engine": "nequip_torchsim_gpu_resident_eager",
            "backend": request.backend,
            "model_path": str(model_path),
            "source_model_kind": "nequip_saved_package",
            "source_loader": "NequIPTorchSimCalc.from_saved_model",
            "torch_sim_version": _distribution_version("torch-sim-atomistic"),
            "nvalchemiops_version": _distribution_version(
                "nvalchemi-toolkit-ops"
            ),
            "gpu_resident": True,
            "md_state_device": str(device),
            "md_state_dtype": "float64",
            "checkpoint_model_dtype": evaluator.model_dtype,
            "checkpoint_parameter_dtypes": evaluator.parameter_dtypes,
            "neighbor_list": "nequip.alchemiops_batch_cell_list",
            "neighbor_list_device": "cuda",
            "integrator": config.integrator,
            "integrator_implementation": "nequip.md_stages.opt1",
            "warmup_steps": config.warmup_steps,
            "warmup_state_restored": True,
            "hot_loop_cpu_or_numpy_transfer": False,
            "reporting_transfer_interval": (
                config.record_interval if config.collect_trajectory else None
            ),
            "trajectory_includes_step_zero": config.collect_trajectory,
            "stress_requested": config.collect_trajectory,
            # The released eager model's ForceStressOutput still differentiates
            # the energy with respect to strain while computing forces.  Opt1
            # safely avoids requesting/collecting stress, but deliberately does
            # not rewrite the scientific model graph into a force-only variant.
            "model_force_stress_graph_rewritten": False,
            **OPT1_POLICY,
        },
    )
    validate_result(request, result)
    return result


__all__ = [
    "BerendsenIntegrator",
    "EagerNequIPTorchSimEvaluator",
    "GPUMDState",
    "ModelOutput",
    "NoseHooverChainIntegrator",
    "OPT1_POLICY",
    "run_md",
]
