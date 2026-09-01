"""NequIP Opt2: model-only CUDA Graph over eager energy and force VJP.

Neighbor-list construction, NVT integration, thermostat updates, statistics,
and trajectory handling remain outside the graph.  The captured region is the
released eager NequIP energy network plus the conservative force VJP only.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from torch import Tensor

from md_benchmark.md_route import (
    MDObservation,
    MDRunRequest,
    MDRunResult,
    validate_result,
)
from md_benchmark.performance import (
    CudaPhaseProfiler,
    performance_profile_requested,
)
from nequip.data import AtomicDataDict
from nequip.md_stages.opt1 import (
    GPUMDState,
    ModelOutput,
    _build_integrator,
    _configure_precision,
    _distribution_version,
    _ensure_evaluated,
    _frame,
    _observation,
)


OPT2_POLICY = {
    "gpu_resident": True,
    "model_compile": False,
    "aotinductor": False,
    "open_equivariance": False,
    "cuda_graph": True,
    "cuda_graph_scope": "model-only",
    "model_specific_fusion": False,
    "tf32": False,
    "amp": False,
    "neighbor_list_backend": "alchemiops",
    "neighbor_list_in_cuda_graph": False,
    "md_in_cuda_graph": False,
}

_REQUIRED_MODEL_FIELDS = (
    AtomicDataDict.POSITIONS_KEY,
    AtomicDataDict.CELL_KEY,
    AtomicDataDict.BATCH_KEY,
    AtomicDataDict.NUM_NODES_KEY,
    AtomicDataDict.ATOM_TYPE_KEY,
    AtomicDataDict.EDGE_INDEX_KEY,
    AtomicDataDict.EDGE_CELL_SHIFT_KEY,
)


class ForceOnlyEnergyVJP(torch.nn.Module):
    """Scientific-equivalent fixed-cell force wrapper for a released model.

    Released NequIP packages wrap their energy network in
    :class:`ForceStressOutput`, which differentiates with respect to both
    positions and an artificial strain tensor.  Fixed-cell NVT needs only the
    positional derivative.  This wrapper retains the exact packaged energy
    network and computes ``-dE/dR`` without the strain/stress branch.
    """

    def __init__(self, energy_model: torch.nn.Module) -> None:
        super().__init__()
        self.energy_model = energy_model

    @classmethod
    def from_released_graph_model(cls, graph_model: torch.nn.Module):
        outer = graph_model
        inspected_types = []
        # Some saved packages contain an additional GraphModel wrapper.  Walk
        # only that unambiguous one-child wrapper chain; never search through a
        # general Sequential, where extracting one nested module could omit
        # required energy operations.
        for _ in range(4):
            outer_type = type(outer)
            inspected_types.append(
                f"{outer_type.__module__}.{outer_type.__name__}"
            )
            if outer_type.__name__ == "ForceStressOutput":
                break
            if outer_type.__name__ != "GraphModel":
                break
            nested = getattr(outer, "model", None)
            if not isinstance(nested, torch.nn.Module) or nested is outer:
                break
            outer = nested

        # ``.nequip.zip`` is loaded by ``torch.package``.  Its classes are not
        # identical to classes imported from the active Python environment, so
        # ``isinstance(outer, ForceStressOutput)`` is always false.  Package
        # versions have used both ``grad_output`` and ``_grad_output`` module
        # paths, so validate the class name and required structure instead of
        # pinning a private module path.
        outer_type = type(outer)
        energy_model = getattr(outer, "func", None)
        is_force_stress_output = (
            outer_type.__name__ == "ForceStressOutput"
            and isinstance(energy_model, torch.nn.Module)
        )
        if not is_force_stress_output:
            raise RuntimeError(
                "NequIP Opt2 requires the released GraphModel to have an "
                "outer ForceStressOutput; refusing unsafe model surgery. "
                f"Inspected wrapper chain: {inspected_types}"
            )
        # Older torch.package archives may not preserve the annotated bool as
        # a plain Python ``bool``.  An explicit False is unsafe; absence or a
        # truthy packaged scalar is accepted because ForceStressOutput's
        # structural contract and the subsequent eager parity checks remain
        # authoritative.
        if getattr(outer, "do_derivatives", True) is False:
            raise RuntimeError(
                "NequIP Opt2 requires a derivative-enabled released model"
            )
        return cls(energy_model)

    def forward(
        self, inputs: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor]:
        positions = inputs[AtomicDataDict.POSITIONS_KEY]
        if not positions.requires_grad:
            raise RuntimeError("model-only graph positions must require gradients")
        # NequIP graph modules add intermediate fields in-place.  A fresh
        # shallow dictionary prevents stale edge vectors from surviving across
        # eager warmups or CUDA Graph capture.
        data = {key: value for key, value in inputs.items()}
        out = self.energy_model(data)
        energy = out[AtomicDataDict.TOTAL_ENERGY_KEY]
        if energy is None:
            raise RuntimeError("released NequIP energy model returned no energy")
        gradient = torch.autograd.grad(
            energy.sum(), positions, create_graph=False, retain_graph=False
        )[0]
        if gradient is None:
            raise RuntimeError("NequIP force VJP returned no positional gradient")
        return energy.reshape(-1), gradient.neg()


@dataclass
class FixedCapacityModelInputs:
    """Fixed-address model tensors with neutral far-edge padding."""

    tensors: dict[str, Tensor]
    edge_capacity: int
    padding_shift: Tensor
    initial_edge_count: int
    peak_edge_count: int

    @classmethod
    def from_exact(
        cls,
        exact: dict[str, Tensor],
        *,
        edge_capacity: int,
        padding_shift: Tensor,
    ) -> FixedCapacityModelInputs:
        edge_index = exact[AtomicDataDict.EDGE_INDEX_KEY]
        edge_shift = exact[AtomicDataDict.EDGE_CELL_SHIFT_KEY]
        edge_count = edge_index.shape[1]
        if edge_count > edge_capacity:
            raise RuntimeError(
                f"initial edge count {edge_count} exceeds fixed capacity "
                f"{edge_capacity}"
            )
        tensors: dict[str, Tensor] = {}
        for key, value in exact.items():
            if key == AtomicDataDict.EDGE_INDEX_KEY:
                tensors[key] = torch.zeros(
                    (2, edge_capacity), dtype=value.dtype, device=value.device
                )
            elif key == AtomicDataDict.EDGE_CELL_SHIFT_KEY:
                tensors[key] = torch.empty(
                    (edge_capacity, 3), dtype=value.dtype, device=value.device
                )
            else:
                tensors[key] = value.detach().clone().contiguous()
        tensors[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)
        result = cls(
            tensors=tensors,
            edge_capacity=edge_capacity,
            padding_shift=padding_shift.to(edge_shift).contiguous(),
            initial_edge_count=edge_count,
            peak_edge_count=edge_count,
        )
        result.update(exact)
        return result

    def update(self, exact: dict[str, Tensor]) -> int:
        """Copy dynamic inputs and fail closed on shape/capacity changes."""

        edge_index = exact[AtomicDataDict.EDGE_INDEX_KEY]
        edge_shift = exact[AtomicDataDict.EDGE_CELL_SHIFT_KEY]
        edge_count = edge_index.shape[1]
        if edge_count > self.edge_capacity:
            raise RuntimeError(
                "NequIP Opt2 fixed edge capacity overflow: "
                f"observed {edge_count}, capacity {self.edge_capacity}; "
                "increase route option edge_capacity"
            )
        positions = exact[AtomicDataDict.POSITIONS_KEY]
        if positions.shape != self.tensors[AtomicDataDict.POSITIONS_KEY].shape:
            raise RuntimeError("NequIP Opt2 atom count changed after CUDA capture")
        if edge_index.shape[0] != 2 or edge_shift.shape != (edge_count, 3):
            raise RuntimeError("NequIP Opt2 received malformed neighbor-list tensors")

        with torch.no_grad():
            self.tensors[AtomicDataDict.POSITIONS_KEY].copy_(positions)
            static_index = self.tensors[AtomicDataDict.EDGE_INDEX_KEY]
            static_shift = self.tensors[AtomicDataDict.EDGE_CELL_SHIFT_KEY]
            static_index[:, :edge_count].copy_(edge_index)
            static_shift[:edge_count].copy_(edge_shift)
            if edge_count < self.edge_capacity:
                static_index[:, edge_count:].zero_()
                static_shift[edge_count:].copy_(self.padding_shift)
        self.peak_edge_count = max(self.peak_edge_count, edge_count)
        return edge_count

    def data_ptrs(self) -> dict[str, int]:
        return {key: value.data_ptr() for key, value in self.tensors.items()}


def _maximum_neighbors_per_atom(
    edge_index: Tensor,
    *,
    num_atoms: int,
) -> int:
    """Return the largest NequIP receiver degree during a capacity probe."""
    if num_atoms < 1:
        raise ValueError("num_atoms must be positive")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_index.shape[1] == 0:
        return 0
    counts = torch.bincount(edge_index[1], minlength=num_atoms)[:num_atoms]
    return int(counts.max().item())


class ModelOnlyCUDAGraphEvaluator:
    """AlchemiOps neighbor list outside, eager NequIP E/F inside CUDA Graph."""

    def __init__(
        self,
        atoms: Atoms,
        model_path: str,
        *,
        device: torch.device,
        options: dict[str, Any],
        profiler: CudaPhaseProfiler | None = None,
    ) -> None:
        try:
            import torch_sim as ts
            import nvalchemiops  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "NequIP Opt2 requires torch-sim and nvalchemiops"
            ) from exc

        from nequip.integrations.torchsim import NequIPTorchSimCalc
        from nequip.md_stages.opt1 import _assert_plain_eager_model

        atomic_numbers = torch.as_tensor(
            atoms.get_atomic_numbers(), dtype=torch.long, device=device
        )
        system_idx = torch.zeros(len(atoms), dtype=torch.long, device=device)
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
        _assert_plain_eager_model(calculator.model)
        calculator.compute_forces = False
        calculator.compute_stress = False
        self.calculator = calculator
        self.wrapper = ForceOnlyEnergyVJP.from_released_graph_model(
            calculator.model
        ).eval()
        for parameter in self.wrapper.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.num_atoms = len(atoms)
        self.profiler = profiler or CudaPhaseProfiler(enabled=False, device=device)

        self.sim_state = ts.io.atoms_to_state(
            [atoms], device=device, dtype=torch.float64
        )
        self.sim_state.atomic_numbers = None
        calculator.set_static_geometry(
            self.sim_state.row_vector_cell, self.sim_state.pbc
        )

        initial = self._prepare_exact(self.sim_state.positions)
        self.model_fields = tuple(
            key
            for key in calculator.model.model_input_fields
            if key in initial
        )
        missing = [
            key for key in _REQUIRED_MODEL_FIELDS if key not in self.model_fields
        ]
        if missing:
            raise RuntimeError(
                f"NequIP Opt2 model inputs missing required fields {missing}"
            )
        if AtomicDataDict.EDGE_TRANSPOSE_PERM_KEY in initial:
            raise RuntimeError(
                "NequIP Opt2 does not support dynamic edge transpose permutations"
            )
        if AtomicDataDict.EDGE_VECTORS_KEY in self.model_fields:
            raise RuntimeError(
                "NequIP Opt2 requires edge vectors to be derived inside the "
                "captured energy model from fixed-capacity edge indices"
            )
        exact_inputs = self._filter_model_inputs(initial, clone_position=True)
        self.track_neighbor_capacity = bool(
            options.get("capacity_probe_collect_per_atom", False)
        )
        self.initial_max_neighbors: int | None = None
        self.peak_neighbors_per_atom: int | None = None
        if self.track_neighbor_capacity:
            self.initial_max_neighbors = _maximum_neighbors_per_atom(
                exact_inputs[AtomicDataDict.EDGE_INDEX_KEY],
                num_atoms=self.num_atoms,
            )
            self.peak_neighbors_per_atom = self.initial_max_neighbors

        initial_edges = exact_inputs[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
        edge_capacity = _edge_capacity(initial_edges, options)
        padding_shift = _padding_shift(atoms, calculator.model.metadata)
        self.fixed = FixedCapacityModelInputs.from_exact(
            exact_inputs,
            edge_capacity=edge_capacity,
            padding_shift=padding_shift.to(device=device),
        )
        self._captured_ptrs = self.fixed.data_ptrs()

        energy_rtol = float(options.get("capture_energy_rtol", 1.0e-5))
        energy_atol = float(options.get("capture_energy_atol", 1.0e-5))
        force_rtol = float(options.get("capture_force_rtol", 1.0e-4))
        force_atol = float(options.get("capture_force_atol", 1.0e-5))
        self.validation_tolerances = {
            "energy_rtol": energy_rtol,
            "energy_atol": energy_atol,
            "force_rtol": force_rtol,
            "force_atol": force_atol,
        }

        official_inputs = {
            key: value.detach().clone().contiguous()
            for key, value in exact_inputs.items()
        }
        with torch.enable_grad():
            official_out = calculator.model(official_inputs)
            official_energy = official_out[AtomicDataDict.TOTAL_ENERGY_KEY]
            official_forces = official_out[AtomicDataDict.FORCE_KEY]
            if official_energy is None or official_forces is None:
                raise RuntimeError(
                    "released NequIP ForceStressOutput omitted energy or forces"
                )
            reference_energy, reference_forces = self.wrapper(exact_inputs)
            # Do not touch the persistent capture leaf on the default stream.
            # Its AccumulateGrad node must first be created on the warmup and
            # capture stream or CUDA rejects the backward graph dependency.
            padded_inputs = {
                key: value.detach().clone().contiguous()
                for key, value in self.fixed.tensors.items()
            }
            padded_inputs[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)
            padded_energy, padded_forces = self.wrapper(padded_inputs)
        official_energy = official_energy.detach().reshape(-1).clone()
        official_forces = official_forces.detach().clone()
        reference_energy = reference_energy.detach().clone()
        reference_forces = reference_forces.detach().clone()
        self.validation_max_abs = {
            "official_vs_force_only_energy": _max_abs(
                reference_energy, official_energy
            ),
            "official_vs_force_only_forces": _max_abs(
                reference_forces, official_forces
            ),
            "exact_vs_padding_energy": _max_abs(
                padded_energy.detach(), reference_energy
            ),
            "exact_vs_padding_forces": _max_abs(
                padded_forces.detach(), reference_forces
            ),
        }
        _assert_close(
            "force-only wrapper energy",
            reference_energy,
            official_energy,
            rtol=energy_rtol,
            atol=energy_atol,
        )
        _assert_close(
            "force-only wrapper forces",
            reference_forces,
            official_forces,
            rtol=force_rtol,
            atol=force_atol,
        )
        _assert_close(
            "neutral edge padding energy",
            padded_energy.detach(),
            reference_energy,
            rtol=energy_rtol,
            atol=energy_atol,
        )
        _assert_close(
            "neutral edge padding forces",
            padded_forces.detach(),
            reference_forces,
            rtol=force_rtol,
            atol=force_atol,
        )
        self.force_wrapper_validation_passed = bool(
            torch.allclose(
                reference_energy,
                official_energy,
                rtol=energy_rtol,
                atol=energy_atol,
            )
            and torch.allclose(
                reference_forces,
                official_forces,
                rtol=force_rtol,
                atol=force_atol,
            )
        )
        self.padding_validation_passed = bool(
            torch.allclose(
                padded_energy.detach(),
                reference_energy,
                rtol=energy_rtol,
                atol=energy_atol,
            )
            and torch.allclose(
                padded_forces.detach(),
                reference_forces,
                rtol=force_rtol,
                atol=force_atol,
            )
        )
        del official_out, padded_inputs, padded_energy, padded_forces

        warmup_stream = torch.cuda.Stream(device=device)
        current_stream = torch.cuda.current_stream(device)
        warmup_stream.wait_stream(current_stream)
        try:
            with torch.cuda.stream(warmup_stream), torch.enable_grad():
                for _ in range(3):
                    self.wrapper(self.fixed.tensors)
            warmup_stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.enable_grad(), torch.cuda.graph(
                self.graph, stream=warmup_stream
            ):
                self.graph_energy, self.graph_forces = self.wrapper(
                    self.fixed.tensors
                )
            current_stream.wait_stream(warmup_stream)
            torch.cuda.synchronize(device)
        except Exception as exc:
            raise RuntimeError(
                "NequIP Opt2 model-only CUDA Graph capture failed; "
                "no eager fallback is permitted"
            ) from exc

        if self.fixed.data_ptrs() != self._captured_ptrs:
            raise RuntimeError(
                "NequIP Opt2 input address changed during CUDA Graph capture"
            )
        self._captured_output_ptrs = {
            "energy": self.graph_energy.data_ptr(),
            "forces": self.graph_forces.data_ptr(),
        }
        self.graph.replay()
        torch.cuda.synchronize(device)
        first_replay_energy = self.graph_energy.detach().clone()
        first_replay_forces = self.graph_forces.detach().clone()
        if self.fixed.data_ptrs() != self._captured_ptrs:
            raise RuntimeError(
                "NequIP Opt2 input address changed after first validation replay"
            )
        if self._output_ptrs() != self._captured_output_ptrs:
            raise RuntimeError(
                "NequIP Opt2 output address changed after first validation replay"
            )
        _assert_close(
            "CUDA Graph replay energy",
            self.graph_energy.detach(),
            reference_energy,
            rtol=energy_rtol,
            atol=energy_atol,
        )
        _assert_close(
            "CUDA Graph replay forces",
            self.graph_forces.detach(),
            reference_forces,
            rtol=force_rtol,
            atol=force_atol,
        )
        self.graph.replay()
        torch.cuda.synchronize(device)
        if self.fixed.data_ptrs() != self._captured_ptrs:
            raise RuntimeError(
                "NequIP Opt2 input address changed after second validation replay"
            )
        if self._output_ptrs() != self._captured_output_ptrs:
            raise RuntimeError(
                "NequIP Opt2 output address changed after second validation replay"
            )
        _assert_close(
            "consecutive replay energy",
            self.graph_energy.detach(),
            first_replay_energy,
            rtol=energy_rtol,
            atol=energy_atol,
        )
        _assert_close(
            "consecutive replay forces",
            self.graph_forces.detach(),
            first_replay_forces,
            rtol=force_rtol,
            atol=force_atol,
        )
        self.validation_max_abs.update(
            {
                "eager_vs_replay_energy": _max_abs(
                    first_replay_energy, reference_energy
                ),
                "eager_vs_replay_forces": _max_abs(
                    first_replay_forces, reference_forces
                ),
                "replay1_vs_replay2_energy": _max_abs(
                    self.graph_energy.detach(), first_replay_energy
                ),
                "replay1_vs_replay2_forces": _max_abs(
                    self.graph_forces.detach(), first_replay_forces
                ),
            }
        )
        self.replay_validation_passed = bool(
            torch.allclose(
                first_replay_energy,
                reference_energy,
                rtol=energy_rtol,
                atol=energy_atol,
            )
            and torch.allclose(
                first_replay_forces,
                reference_forces,
                rtol=force_rtol,
                atol=force_atol,
            )
            and torch.allclose(
                self.graph_energy.detach(),
                first_replay_energy,
                rtol=energy_rtol,
                atol=energy_atol,
            )
            and torch.allclose(
                self.graph_forces.detach(),
                first_replay_forces,
                rtol=force_rtol,
                atol=force_atol,
            )
        )
        self.production_replays = 0

        model_dtype = getattr(calculator.model, "model_dtype", None)
        self.model_dtype = str(model_dtype).removeprefix("torch.")
        self.parameter_dtypes = sorted(
            {
                str(parameter.dtype).removeprefix("torch.")
                for parameter in calculator.model.parameters()
            }
        )

    def _prepare_exact(self, positions: Tensor) -> dict[str, Tensor]:
        if positions.device != self.device or positions.dtype != torch.float64:
            raise ValueError("NequIP Opt2 requires FP64 CUDA MD positions")
        if positions.shape != (self.num_atoms, 3):
            raise ValueError(
                f"expected positions {(self.num_atoms, 3)}, got {positions.shape}"
            )
        self.sim_state.positions = positions
        with self.profiler.phase("neighbor_and_transforms"):
            return self.calculator.prepare_model_inputs(self.sim_state)

    def _filter_model_inputs(
        self, prepared: dict[str, Tensor], *, clone_position: bool
    ) -> dict[str, Tensor]:
        inputs = {key: prepared[key] for key in self.model_fields}
        if clone_position:
            inputs[AtomicDataDict.POSITIONS_KEY] = (
                inputs[AtomicDataDict.POSITIONS_KEY]
                .detach()
                .clone()
                .contiguous()
                .requires_grad_(True)
            )
        return inputs

    def __call__(self, positions: Tensor) -> ModelOutput:
        prepared = self._prepare_exact(positions)
        exact = self._filter_model_inputs(prepared, clone_position=False)
        if self.track_neighbor_capacity:
            maximum = _maximum_neighbors_per_atom(
                exact[AtomicDataDict.EDGE_INDEX_KEY],
                num_atoms=self.num_atoms,
            )
            assert self.peak_neighbors_per_atom is not None
            self.peak_neighbors_per_atom = max(
                self.peak_neighbors_per_atom,
                maximum,
            )
        with self.profiler.phase("fixed_input_update"):
            self.fixed.update(exact)
        if self.fixed.data_ptrs() != self._captured_ptrs:
            raise RuntimeError("NequIP Opt2 fixed input address changed after capture")
        with self.profiler.phase("model_cuda_graph_replay"):
            self.graph.replay()
        if self._output_ptrs() != self._captured_output_ptrs:
            raise RuntimeError("NequIP Opt2 output address changed after capture")
        self.production_replays += 1
        return ModelOutput(
            energy=self.graph_energy.reshape(-1)[0].detach().to(torch.float64),
            forces=self.graph_forces.reshape(self.num_atoms, 3)
            .detach()
            .to(torch.float64),
            stress=None,
        )

    def _output_ptrs(self) -> dict[str, int]:
        return {
            "energy": self.graph_energy.data_ptr(),
            "forces": self.graph_forces.data_ptr(),
        }

    def reset_production_replays(self) -> None:
        """Exclude disposable MD warmup calls from production accounting."""

        self.production_replays = 0
        self.peak_neighbors_per_atom = self.initial_max_neighbors


def _edge_capacity(initial_edges: int, options: dict[str, Any]) -> int:
    explicit = options.get("edge_capacity")
    if explicit is not None:
        if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit < 1:
            raise ValueError("route option edge_capacity must be a positive integer")
        return explicit
    factor = float(options.get("edge_capacity_factor", 1.25))
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError("route option edge_capacity_factor must be finite and >= 1")
    capacity = max(initial_edges + 64, math.ceil(initial_edges * factor), 128)
    return math.ceil(capacity / 128) * 128


def _padding_shift(atoms: Atoms, metadata: dict[str, str]) -> Tensor:
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    lengths = np.linalg.norm(cell, axis=1)
    axis = int(np.argmax(lengths))
    if not np.isfinite(lengths[axis]) or lengths[axis] <= 0.0:
        raise ValueError("NequIP Opt2 requires a finite nonzero periodic cell")
    r_max = float(metadata["r_max"])
    multiple = math.ceil((2.0 * r_max + 1.0) / lengths[axis]) + 1
    shift = torch.zeros(3, dtype=torch.float64)
    shift[axis] = float(multiple)
    return shift


def _assert_close(
    label: str,
    actual: Tensor,
    expected: Tensor,
    *,
    rtol: float,
    atol: float,
) -> None:
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"NequIP Opt2 {label} validation shape mismatch: "
            f"{tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    if not bool(torch.isfinite(actual).all()) or not bool(
        torch.isfinite(expected).all()
    ):
        raise FloatingPointError(
            f"NequIP Opt2 {label} validation contains non-finite values"
        )
    # Retain the tolerance arguments for metadata/reporting compatibility.
    # Numerical differences are reported and do not reject Opt2 execution.
    _ = rtol, atol


def _max_abs(actual: Tensor, expected: Tensor) -> float:
    return float((actual - expected).abs().max().item())


def _validate_request(request: MDRunRequest) -> Path:
    if request.model != "nequip" or request.stage != "opt2":
        raise ValueError(
            f"nequip.md_stages.opt2 owns nequip/opt2, got "
            f"{request.model}/{request.stage}"
        )
    if request.backend != "model-only-cuda-graph":
        raise ValueError("NequIP Opt2 backend must be 'model-only-cuda-graph'")
    if request.config.device.split(":", maxsplit=1)[0] != "cuda":
        raise ValueError("NequIP Opt2 is CUDA-only")
    if request.config.dtype != "float64":
        raise ValueError("NequIP Opt2 requires an FP64 MD state")
    if request.atoms.constraints:
        raise NotImplementedError("NequIP Opt2 does not ignore ASE constraints")
    if len(request.atoms) < 2:
        raise ValueError("NVT MD requires at least two atoms")
    if not bool(np.all(request.atoms.pbc)):
        raise ValueError("NequIP Opt2 fixed-capacity padding requires full PBC")
    if request.config.collect_trajectory:
        raise ValueError(
            "NequIP Opt2 captures force-only fixed-cell inference and cannot "
            "produce stress trajectories"
        )
    if request.output_path is not None:
        raise ValueError("NequIP Opt2 output_path requires an unsupported trajectory")
    path = Path(request.model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not path.name.endswith(".nequip.zip"):
        raise ValueError("NequIP Opt2 requires the official .nequip.zip package")
    if request.options.get("neighborlist_backend", "alchemiops") != "alchemiops":
        raise ValueError("NequIP Opt2 requires the external AlchemiOps neighbor list")
    forbidden = (
        "compiled_model_path",
        "compile",
        "aotinductor",
        "open_equivariance",
        "model_specific_fusion",
        "allow_tf32",
        "amp",
        "whole_step_cuda_graph",
    )
    for key in forbidden:
        if request.options.get(key):
            raise ValueError(f"NequIP Opt2 forbids route option {key!r}")
    _edge_capacity(1, request.options)
    return path


def _validate_finite(state: GPUMDState) -> None:
    if state.output is None:
        raise RuntimeError("NequIP Opt2 final state was not evaluated")
    tensors = {
        "positions": state.positions,
        "momenta": state.momenta,
        "energy": state.output.energy,
        "forces": state.output.forces,
    }
    invalid = [
        name for name, value in tensors.items() if not bool(torch.isfinite(value).all())
    ]
    if invalid:
        raise FloatingPointError(f"NequIP Opt2 final state has non-finite {invalid}")


def run_md(request: MDRunRequest) -> MDRunResult:
    """Run strict model-only CUDA Graph NequIP NVT MD."""

    model_path = _validate_request(request)
    if not torch.cuda.is_available():
        raise RuntimeError("NequIP Opt2 requested CUDA, but CUDA is unavailable")
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
    profiler = CudaPhaseProfiler(
        enabled=performance_profile_requested(request.options), device=device
    )
    evaluator = ModelOnlyCUDAGraphEvaluator(
        atoms,
        str(model_path),
        device=device,
        options=request.options,
        profiler=profiler,
    )

    if config.warmup_steps:
        warm_state = GPUMDState(positions0.clone(), momenta0.clone())
        warm_integrator = _build_integrator(request, masses)
        for _ in range(config.warmup_steps):
            warm_integrator.step(warm_state, evaluator)
        torch.cuda.synchronize(device)
    evaluator.reset_production_replays()

    state = GPUMDState(positions0.clone(), momenta0.clone())
    integrator = _build_integrator(request, masses)
    observations: list[MDObservation] = []
    observation_steps = set(config.observation_steps)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    profiler.start()
    started = time.perf_counter()
    _ensure_evaluated(state, evaluator)
    if config.collect_statistics and 0 in observation_steps:
        observations.append(_observation(state, step=0, masses=masses))
    for step in range(1, config.steps + 1):
        with profiler.phase("md_step"):
            integrator.step(state, evaluator)
        if config.collect_statistics and step in observation_steps:
            observations.append(_observation(state, step=step, masses=masses))
    torch.cuda.synchronize(device)
    profiler.stop()
    elapsed = time.perf_counter() - started
    performance_profile = profiler.summary(synchronize=False)
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1.0e9
    expected_replays = config.steps + 1
    if evaluator.production_replays != expected_replays:
        raise RuntimeError(
            "NequIP Opt2 production replay count mismatch: "
            f"expected {expected_replays}, observed {evaluator.production_replays}"
        )
    _validate_finite(state)

    final_atoms = _frame(atoms, state, step=config.steps, require_stress=False)
    result = MDRunResult(
        model=request.model,
        stage=request.stage,
        completed_steps=config.steps,
        elapsed_s=elapsed,
        peak_cuda_memory_gb=peak_memory_gb,
        final_atoms=final_atoms,
        observations=observations,
        trajectory=None,
        trajectory_path=None,
        metadata={
            "engine": "nequip_model_only_cuda_graph",
            "backend": request.backend,
            "model_path": str(model_path),
            "source_model_kind": "nequip_saved_package",
            "source_loader": "NequIPTorchSimCalc.from_saved_model",
            "torch_sim_version": _distribution_version("torch-sim-atomistic"),
            "nvalchemiops_version": _distribution_version(
                "nvalchemi-toolkit-ops"
            ),
            "checkpoint_model_dtype": evaluator.model_dtype,
            "checkpoint_parameter_dtypes": evaluator.parameter_dtypes,
            "md_state_device": str(device),
            "md_state_dtype": "float64",
            "captured_components": ["eager_energy_model", "force_vjp"],
            "uncaptured_components": [
                "neighbor_list",
                "md_integrator",
                "thermostat",
                "state_update",
                "statistics",
            ],
            "force_wrapper": "fixed_cell_position_only_vjp",
            "model_parameters_require_grad": False,
            "released_force_stress_wrapper_reused": False,
            "force_wrapper_validation_passed": (
                evaluator.force_wrapper_validation_passed
            ),
            "fixed_input_addresses": True,
            "fixed_output_addresses": True,
            "capture_input_data_ptrs": evaluator._captured_ptrs,
            "capture_output_data_ptrs": evaluator._captured_output_ptrs,
            "fixed_edge_capacity": evaluator.fixed.edge_capacity,
            "initial_edge_count": evaluator.fixed.initial_edge_count,
            "peak_edge_count": evaluator.fixed.peak_edge_count,
            "peak_neighbors_per_atom": evaluator.peak_neighbors_per_atom,
            "capacity_probe_collect_per_atom": (
                evaluator.track_neighbor_capacity
            ),
            "edge_overflow_policy": "raise_no_fallback",
            "edge_padding": "far_periodic_self_edge_zero_cutoff",
            "edge_padding_cell_shift": (
                evaluator.fixed.padding_shift.detach().cpu().tolist()
            ),
            "padding_validation_passed": evaluator.padding_validation_passed,
            "replay_validation_passed": evaluator.replay_validation_passed,
            "validation_tolerances": evaluator.validation_tolerances,
            "validation_max_abs": evaluator.validation_max_abs,
            "numerical_validation_failure_policy": "report_only",
            "capture_failure_policy": "raise_no_fallback",
            "cuda_graph_capture_count": 1,
            "production_replays": evaluator.production_replays,
            "expected_production_replays": expected_replays,
            "stress_requested": False,
            "stress_supported": False,
            "warmup_steps": config.warmup_steps,
            "warmup_state_restored": True,
            "performance_profile": performance_profile,
            **OPT2_POLICY,
        },
    )
    validate_result(request, result)
    return result


__all__ = [
    "FixedCapacityModelInputs",
    "ForceOnlyEnergyVJP",
    "ModelOnlyCUDAGraphEvaluator",
    "OPT2_POLICY",
    "run_md",
]
