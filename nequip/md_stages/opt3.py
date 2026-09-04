"""NequIP Opt3: one whole-step CUDA Graph for fixed-cell NVT MD.

The captured graph contains the fixed-shape PBC neighbor builder, the released
eager NequIP energy model, its conservative positional VJP, the selected NVT
integrator, thermostat state, and persistent MD-state updates.  It deliberately
does not enable compilation, OpenEquivariance, model-specific fusion, graph
buckets, or transactional rollback.

AlchemiOps' public neighbor-list API returns ragged newly allocated tensors and
therefore cannot safely be replayed from a CUDA Graph when the edge count
changes.  Opt3 instead freezes AlchemiOps' single-periodic-system semantics into
a capture-safe candidate enumeration.  The exact AlchemiOps path remains the
scientific reference used by the mandatory pre-capture parity check.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from torch import Tensor

from md_benchmark.md_route import (
    MDObservation,
    MDRunRequest,
    MDRunResult,
    validate_result,
)
from md_benchmark.neighbor_utils import (
    capacities_from_counts,
    displacement_exceeds_skin,
    make_slot_layout,
    normalize_neighbor_capacities,
    select_skin_candidates,
)
from md_benchmark.performance import (
    CudaPhaseProfiler,
    performance_profile_requested,
)
from nequip.data import AtomicDataDict
from nequip.md_stages.opt1 import (
    BerendsenIntegrator,
    GPUMDState,
    ModelOutput,
    NoseHooverChainIntegrator,
    _build_integrator,
    _configure_precision,
    _distribution_version,
    _frame,
    _observation,
)
from nequip.md_stages.opt2 import (
    ForceOnlyEnergyVJP,
    _assert_close,
    _max_abs,
)


OPT3_POLICY = {
    "gpu_resident": True,
    "model_compile": False,
    "aotinductor": False,
    "open_equivariance": False,
    "cuda_graph": True,
    "cuda_graph_scope": "whole-step",
    "model_specific_fusion": False,
    "tf32": False,
    "amp": False,
    "neighbor_list_backend": "alchemiops-fixed-shape-semantic",
    "neighbor_list_in_cuda_graph": True,
    "md_in_cuda_graph": True,
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


def _pbc_repetitions(cell: Tensor, cutoff: float, pbc: Tensor) -> tuple[int, int, int]:
    """Return the complete image range required by a spherical cutoff."""

    cell64 = cell.detach().to(device="cpu", dtype=torch.float64).reshape(3, 3)
    pbc_cpu = pbc.detach().to(device="cpu", dtype=torch.bool).reshape(3)
    cross_a2a3 = torch.cross(cell64[1], cell64[2], dim=0)
    volume = torch.dot(cell64[0], cross_a2a3)
    if not bool(torch.isfinite(volume)) or float(volume.abs()) == 0.0:
        raise ValueError("Cannot enumerate PBC images for a singular cell")
    reciprocal = (
        cross_a2a3,
        torch.cross(cell64[2], cell64[0], dim=0),
        torch.cross(cell64[0], cell64[1], dim=0),
    )
    repetitions = []
    for axis in range(3):
        if bool(pbc_cpu[axis]):
            inverse_plane_distance = torch.linalg.vector_norm(reciprocal[axis] / volume)
            repetitions.append(int(torch.ceil(cutoff * inverse_plane_distance).item()))
        else:
            repetitions.append(0)
    return tuple(repetitions)  # type: ignore[return-value]


def _round_neighbor_capacity(maximum: int, *, factor: float, slot_step: int) -> int:
    if maximum < 1:
        raise ValueError("maximum neighbor count must be positive")
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError("neighbor capacity factor must be finite and >= 1")
    if slot_step < 1:
        raise ValueError("neighbor capacity slot step must be positive")
    required = max(maximum + 1, math.ceil(maximum * factor))
    return math.ceil(required / slot_step) * slot_step


def _neighbors_per_atom(
    exact_edge_index: Tensor,
    *,
    num_atoms: int,
    options: dict[str, Any],
) -> tuple[int, int | None]:
    """Resolve one eSEN-style CAP without applying probe headroom twice."""

    edge_count = int(exact_edge_index.shape[1])
    if edge_count < 1:
        raise ValueError("NequIP Opt3 requires at least one real neighbor edge")
    counts = torch.bincount(exact_edge_index[1], minlength=num_atoms)[:num_atoms]
    maximum = int(counts.max().item())
    slot_step = int(options.get("neighbor_capacity_slot_step", 8))
    if slot_step < 1:
        raise ValueError("neighbor capacity slot step must be positive")
    factor = float(options.get("edge_capacity_factor", 1.10))
    initial_safe = _round_neighbor_capacity(
        maximum,
        factor=factor,
        slot_step=slot_step,
    )

    requested_total = options.get("edge_capacity")
    total_floor = 0
    if requested_total is not None:
        if (
            isinstance(requested_total, bool)
            or not isinstance(requested_total, int)
            or requested_total < edge_count
        ):
            raise ValueError(
                "route option edge_capacity must be an integer no smaller than "
                "the initial real edge count"
            )
        # The trajectory probe has already applied its total-edge headroom.
        # Convert and align exactly once; an extra slot bucket here would apply
        # a second guard (for example 88 -> 96 on the bulk-Cu investigation).
        per_centre = math.ceil(requested_total / num_atoms)
        total_floor = math.ceil(per_centre / slot_step) * slot_step

    explicit = options.get("neighbors_per_atom")
    if explicit is not None:
        if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit < 1:
            raise ValueError("route option neighbors_per_atom must be positive")
        if explicit < maximum:
            raise ValueError(
                "route option neighbors_per_atom is smaller than the initial "
                f"receiver degree: {explicit} < {maximum}"
            )
        return explicit, requested_total

    return max(total_floor, initial_safe), requested_total


class FixedShapeAlchemiNeighborBuilder:
    """Capture-safe, fixed-shape form of single-system AlchemiOps semantics.

    Every real centre owns ``neighbors_per_atom`` slots.  Missing slots become
    far periodic self-edges distributed over all real atom indices.  Because
    the released model has no reserved dummy chemical species, this is safer
    than adding Z=0 nodes: cutoff functions make these edges exactly inactive,
    and no extra node can enter the energy readout or force output.
    """

    def __init__(
        self,
        *,
        num_atoms: int,
        cell: Tensor,
        pbc: Tensor,
        cutoff: float,
        neighbors_per_atom: int,
        neighbor_capacities: list[int] | Tensor | None = None,
        output_edge_index: Tensor,
        output_edge_shift: Tensor,
        verlet_skin: float = 0.0,
        verlet_candidate_capacity: int | None = None,
    ) -> None:
        if num_atoms < 2:
            raise ValueError("fixed neighbor builder requires at least two atoms")
        if cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        if neighbors_per_atom < 1:
            raise ValueError("neighbors_per_atom must be positive")
        self.num_atoms = int(num_atoms)
        self.cutoff = float(cutoff)
        capacities = normalize_neighbor_capacities(
            neighbor_capacities,
            num_atoms=num_atoms,
            default=int(neighbors_per_atom),
        )
        (
            self.slot_centres,
            self.slot_ranks,
            self.selection_indices,
            self.neighbors_per_atom,
            self.edge_capacity,
        ) = make_slot_layout(capacities, device=cell.device)
        self.neighbor_capacities = torch.as_tensor(
            capacities, dtype=torch.long, device=cell.device
        )
        if verlet_skin < 0:
            raise ValueError("verlet_skin must be non-negative")
        self.verlet_skin = float(verlet_skin)
        self.verlet_candidate_capacity = verlet_candidate_capacity
        self.skin_candidate_ids: Tensor | None = None
        self.skin_candidate_mask: Tensor | None = None
        self.skin_reference_positions: Tensor | None = None
        self.skin_misses = torch.zeros((), dtype=torch.long, device=cell.device)
        self.skin_rebuilds = 0
        self.device = cell.device
        self.cell = cell.detach().reshape(3, 3).contiguous()
        self.inverse_cell = torch.linalg.inv(self.cell).contiguous()
        self.pbc = pbc.detach().to(device=cell.device, dtype=torch.bool).reshape(3)
        self.repetitions = _pbc_repetitions(cell, cutoff + self.verlet_skin, pbc)

        axes = [
            torch.arange(
                -repeat,
                repeat + 1,
                dtype=self.cell.dtype,
                device=self.device,
            )
            for repeat in self.repetitions
        ]
        unit_shifts = torch.cartesian_prod(*axes).reshape(-1, 3)
        self.unit_cell_shifts = unit_shifts.contiguous()
        self.num_cells = int(unit_shifts.shape[0])
        self.candidates_per_centre = self.num_atoms * self.num_cells
        if self.neighbors_per_atom > self.candidates_per_centre:
            raise ValueError(
                "neighbors_per_atom exceeds the complete PBC candidate count"
            )

        self.candidate_sources = torch.arange(
            self.num_atoms, dtype=torch.long, device=self.device
        ).repeat_interleave(self.num_cells)
        self.candidate_shifts = self.unit_cell_shifts.repeat(self.num_atoms, 1)
        self.candidate_ids = torch.arange(
            self.candidates_per_centre,
            dtype=torch.long,
            device=self.device,
        ).reshape(1, -1)
        self.slot_centres = self.slot_centres

        if output_edge_index.shape != (2, self.edge_capacity):
            raise ValueError("fixed edge-index output has the wrong shape")
        if output_edge_shift.shape != (self.edge_capacity, 3):
            raise ValueError("fixed edge-shift output has the wrong shape")
        self.edge_index = output_edge_index
        self.edge_shift = output_edge_shift

        # Distributed self-edge sinks avoid concentrating padded scatter writes
        # on atom zero.  They do not introduce dummy nodes into the model.
        slots = torch.arange(self.edge_capacity, dtype=torch.long, device=self.device)
        self.sink_indices = (slots + self.slot_centres).remainder(self.num_atoms)
        lengths = torch.linalg.vector_norm(self.cell, dim=1)
        axis = int(torch.argmax(lengths).item())
        axis_length = float(lengths[axis].item())
        if not math.isfinite(axis_length) or axis_length <= 0.0:
            raise ValueError("Cannot construct sink padding for an invalid cell")
        far_shift = max(2, math.ceil((self.cutoff + 1.0) / axis_length) + 1)
        self.padding_shifts = self.edge_shift.new_zeros(self.edge_capacity, 3)
        self.padding_shifts[:, axis] = float(far_shift)
        self.active_mask = torch.zeros(
            self.edge_capacity, dtype=torch.bool, device=self.device
        )

        self.build_calls = torch.zeros((), dtype=torch.long, device=self.device)
        self.capacity_misses = torch.zeros((), dtype=torch.long, device=self.device)
        self.first_overflow_step = torch.full(
            (), -1, dtype=torch.long, device=self.device
        )
        self.current_real_edges = torch.zeros((), dtype=torch.long, device=self.device)
        self.minimum_real_edges = torch.full(
            (), self.edge_capacity, dtype=torch.long, device=self.device
        )
        self.maximum_real_edges = torch.zeros((), dtype=torch.long, device=self.device)
        self.maximum_neighbors = torch.zeros((), dtype=torch.long, device=self.device)
        self.maximum_capacity_excess = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.maximum_neighbors_by_atom = torch.zeros(
            self.num_atoms, dtype=torch.long, device=self.device
        )

    @torch.no_grad()
    def reset_stats(self) -> None:
        self.build_calls.zero_()
        self.capacity_misses.zero_()
        self.first_overflow_step.fill_(-1)
        self.current_real_edges.zero_()
        self.minimum_real_edges.fill_(self.edge_capacity)
        self.maximum_real_edges.zero_()
        self.maximum_neighbors.zero_()
        self.maximum_capacity_excess.zero_()
        self.maximum_neighbors_by_atom.zero_()
        self.skin_misses.zero_()
        self.skin_rebuilds = 0

    @torch.no_grad()
    def initialize_skin(self, positions: Tensor) -> None:
        if self.verlet_skin <= 0:
            return
        requested = self.verlet_candidate_capacity
        slots = max(self.neighbors_per_atom, int(requested)) if requested is not None else max(
            self.neighbors_per_atom * 2, self.neighbors_per_atom + 32
        )
        slots = min(slots, self.candidates_per_centre)
        selected, counts, selected_valid = select_skin_candidates(
            positions,
            self.candidate_sources,
            -self.candidate_shifts,
            self.cell,
            cutoff=self.cutoff + self.verlet_skin,
            slots_per_atom=slots,
        )
        torch._assert_async(
            (counts <= slots).all(),
            "NequIP Opt3 Verlet candidate capacity is smaller than the "
            "cutoff+skin candidate count",
        )
        if self.skin_candidate_ids is None:
            self.skin_candidate_ids = selected
            self.skin_candidate_mask = selected_valid
            self.skin_reference_positions = positions.detach().clone()
        else:
            if self.skin_candidate_ids.shape != selected.shape:
                raise RuntimeError("Verlet candidate shape changed during rebuild")
            self.skin_candidate_ids.copy_(selected)
            assert self.skin_candidate_mask is not None
            self.skin_candidate_mask.copy_(selected_valid)
            assert self.skin_reference_positions is not None
            self.skin_reference_positions.copy_(positions)
        self.verlet_candidate_capacity = slots
        self.skin_rebuilds += 1

    @torch.no_grad()
    def build(
        self, positions: Tensor, *, step: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if positions.shape != (self.num_atoms, 3):
            raise ValueError("fixed neighbor builder received the wrong atom shape")
        if positions.device != self.device:
            raise ValueError(
                "fixed neighbor builder received positions on wrong device"
            )

        if self.skin_candidate_ids is not None:
            assert self.skin_reference_positions is not None
            assert self.skin_candidate_mask is not None
            skin_miss = displacement_exceeds_skin(
                positions,
                self.skin_reference_positions,
                self.verlet_skin,
                self.cell,
                self.pbc,
                self.inverse_cell,
            )
            self.skin_misses.add_(skin_miss.to(torch.long))
            torch._assert_async(
                ~skin_miss,
                "NequIP Opt3 Verlet skin exhausted; rebuild the candidate list",
            )
            cached = self.skin_candidate_ids.reshape(-1)
            source_ids = self.candidate_sources.index_select(0, cached).reshape(
                self.num_atoms, -1
            )
            candidate_shifts = self.candidate_shifts.index_select(
                0, cached
            ).reshape(self.num_atoms, -1, 3)
            candidate_width = int(source_ids.shape[1])
            shift_vectors = torch.mm(
                candidate_shifts.reshape(-1, 3).to(dtype=positions.dtype),
                self.cell.to(dtype=positions.dtype),
            ).reshape(self.num_atoms, candidate_width, 3)
            source_positions = positions.index_select(
                0, source_ids.reshape(-1)
            ).reshape(self.num_atoms, candidate_width, 3)
            candidate_ids = torch.arange(
                candidate_width, dtype=torch.long, device=self.device
            ).reshape(1, -1).expand(self.num_atoms, -1)
            vectors = positions.unsqueeze(1) - source_positions + shift_vectors
            valid_candidates = self.skin_candidate_mask
        else:
            source_positions = positions.index_select(0, self.candidate_sources)
            shift_vectors = torch.mm(
                self.candidate_shifts.to(dtype=positions.dtype),
                self.cell.to(dtype=positions.dtype),
            )
            candidate_ids = self.candidate_ids.expand(self.num_atoms, -1)
            candidate_shifts = self.candidate_shifts
            vectors = (
                positions.unsqueeze(1)
                - source_positions.unsqueeze(0)
                + shift_vectors.unsqueeze(0)
            )
            valid_candidates = torch.ones_like(candidate_ids, dtype=torch.bool)
        candidate_width = int(candidate_ids.shape[1])
        # NequIP convention: r_ij = r_j - r_i + shift @ cell for
        # edge_index=(i, j).
        distance_sqr = vectors.square().sum(dim=-1)
        valid = (
            valid_candidates
            & (distance_sqr <= self.cutoff * self.cutoff)
            & (distance_sqr > 1.0e-8)
        )
        counts = valid.sum(dim=1)
        ordered = torch.where(
            valid,
            candidate_ids,
            torch.full_like(candidate_ids, candidate_width),
        )
        selected_matrix = torch.topk(
            ordered,
            k=self.neighbors_per_atom,
            dim=1,
            largest=False,
            sorted=True,
        ).values
        selected_valid_matrix = selected_matrix < candidate_width
        safe = selected_matrix.clamp_max(candidate_width - 1)
        if self.skin_candidate_ids is not None:
            selected_sources = torch.gather(source_ids, 1, safe)
            selected_shifts = torch.gather(
                candidate_shifts,
                1,
                safe.unsqueeze(-1).expand(-1, -1, 3),
            )
        else:
            selected_sources = self.candidate_sources.index_select(
                0, safe.reshape(-1)
            ).reshape(self.num_atoms, -1)
            selected_shifts = self.candidate_shifts.index_select(
                0, safe.reshape(-1)
            ).reshape(self.num_atoms, -1, 3)
        selected_valid = selected_valid_matrix.reshape(-1).index_select(
            0, self.selection_indices
        )
        sources = selected_sources.reshape(-1).index_select(
            0, self.selection_indices
        )
        shifts = selected_shifts.reshape(-1, 3).index_select(
            0, self.selection_indices
        )
        self.edge_index[0].copy_(
            torch.where(selected_valid, sources, self.sink_indices)
        )
        self.edge_index[1].copy_(
            torch.where(selected_valid, self.slot_centres, self.sink_indices)
        )
        self.edge_shift.copy_(
            torch.where(
                selected_valid.unsqueeze(1),
                shifts.to(dtype=self.edge_shift.dtype),
                self.padding_shifts,
            )
        )
        self.active_mask.copy_(selected_valid)

        real_edges = selected_valid.sum()
        maximum = counts.max()
        excess = torch.clamp_min(
            (counts - self.neighbor_capacities).max(), 0
        )
        overflow = excess > 0
        call_step = self.build_calls if step is None else step
        self.current_real_edges.copy_(real_edges)
        self.minimum_real_edges.copy_(
            torch.minimum(self.minimum_real_edges, real_edges)
        )
        self.maximum_real_edges.copy_(
            torch.maximum(self.maximum_real_edges, real_edges)
        )
        self.maximum_neighbors.copy_(torch.maximum(self.maximum_neighbors, maximum))
        self.maximum_neighbors_by_atom.copy_(
            torch.maximum(self.maximum_neighbors_by_atom, counts)
        )
        self.maximum_capacity_excess.copy_(
            torch.maximum(self.maximum_capacity_excess, excess)
        )
        self.capacity_misses.add_(overflow.to(torch.long))
        first = (self.first_overflow_step < 0) & overflow
        self.first_overflow_step.copy_(
            torch.where(first, call_step, self.first_overflow_step)
        )
        self.build_calls.add_(1)
        return self.edge_index, self.edge_shift

    def raise_for_overflow(self) -> None:
        misses = int(self.capacity_misses.item())
        if misses:
            required = int(self.maximum_neighbors.item())
            raise RuntimeError(
                "NequIP Opt3 per-centre CAP overflow: "
                f"required {required}, capacity {self.neighbors_per_atom}; "
                "increase neighbors_per_atom or the probe-derived edge_capacity"
            )

    def stats(self) -> dict[str, Any]:
        calls = int(self.build_calls.item())
        minimum = int(self.minimum_real_edges.item()) if calls else None
        maximum = int(self.maximum_real_edges.item()) if calls else None
        misses = int(self.capacity_misses.item())
        first = int(self.first_overflow_step.item())
        return {
            "fixed_builder_build_calls": calls,
            "fixed_builder_capacity_misses": misses,
            "fixed_builder_first_overflow_step": first if first >= 0 else None,
            "fixed_builder_edge_capacity": self.edge_capacity,
            "fixed_builder_neighbors_per_atom": self.neighbors_per_atom,
            "fixed_builder_neighbor_capacities": self.neighbor_capacities.detach()
            .to(device="cpu")
            .tolist(),
            "fixed_builder_capacity_policy": (
                "per-atom-cap"
                if self.neighbor_capacities.unique().numel() > 1
                else "uniform-cap"
            ),
            "fixed_builder_min_real_edges": minimum,
            "fixed_builder_max_real_edges": maximum,
            "fixed_builder_max_padding_fraction": (
                None
                if minimum is None
                else (self.edge_capacity - minimum) / self.edge_capacity
            ),
            "fixed_builder_max_neighbors": int(self.maximum_neighbors.item()),
            "fixed_builder_maximum_neighbors_by_atom": self.maximum_neighbors_by_atom.detach()
            .to(device="cpu")
            .tolist(),
            "fixed_builder_max_capacity_excess": int(
                self.maximum_capacity_excess.item()
            ),
            "fixed_builder_verlet_skin": self.verlet_skin,
            "fixed_builder_verlet_candidate_capacity": self.verlet_candidate_capacity,
            "fixed_builder_verlet_skin_misses": int(self.skin_misses.item()),
            "fixed_builder_verlet_rebuilds": self.skin_rebuilds,
            "fixed_builder_verlet_enabled": self.skin_candidate_ids is not None,
            "fixed_builder_active_candidate_slots": self.num_atoms
            * (
                int(self.skin_candidate_ids.shape[1])
                if self.skin_candidate_ids is not None
                else self.candidates_per_centre
            ),
            "fixed_builder_candidate_reduction_fraction": (
                0.0
                if self.skin_candidate_ids is None
                else 1.0
                - int(self.skin_candidate_ids.shape[1]) / self.candidates_per_centre
            ),
            "fixed_builder_candidate_universe_size": (
                self.num_atoms * self.candidates_per_centre
            ),
            "fixed_builder_candidates_per_atom": self.candidates_per_centre,
            "fixed_builder_num_pbc_cells": self.num_cells,
            "fixed_builder_pbc_repetitions": list(self.repetitions),
        }


def _edge_signature(
    edge_index: Tensor, edge_shift: Tensor, active: Tensor | None = None
) -> list[tuple[int, int, float, float, float]]:
    """Canonical host signature used only for the pre-capture topology check."""

    if active is not None:
        edge_index = edge_index[:, active]
        edge_shift = edge_shift[active]
    index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    shift_cpu = edge_shift.detach().to(device="cpu", dtype=torch.float64)
    return sorted(
        (
            int(index_cpu[0, edge]),
            int(index_cpu[1, edge]),
            round(float(shift_cpu[edge, 0]), 10),
            round(float(shift_cpu[edge, 1]), 10),
            round(float(shift_cpu[edge, 2]), 10),
        )
        for edge in range(index_cpu.shape[1])
    )


@dataclass
class _InitialState:
    positions: Tensor
    momenta: Tensor


class WholeStepCUDAGraphMD:
    """Own one captured initial-evaluation/NVT-step graph."""

    def __init__(
        self,
        atoms: Atoms,
        model_path: str,
        integrator: BerendsenIntegrator | NoseHooverChainIntegrator,
        initial_state: _InitialState,
        *,
        device: torch.device,
        options: dict[str, Any],
        neighbor_capacities: list[int] | Tensor | None = None,
    ) -> None:
        try:
            import torch_sim as ts
            import nvalchemiops  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "NequIP Opt3 requires torch-sim and nvalchemiops"
            ) from exc

        from nequip.integrations.torchsim import NequIPTorchSimCalc
        from nequip.md_stages.opt1 import _assert_plain_eager_model
        enable_cueq = bool(options.get("_opt4_enable_cueq", False))
        self.cueq_enabled = enable_cueq

        self.device = device
        self.num_atoms = len(atoms)
        self.integrator = integrator
        self.capture_warmup = int(options.get("capture_warmup", 3))
        if self.capture_warmup < 0:
            raise ValueError("capture_warmup must be non-negative")
        self.verlet_rebuild_interval = int(
            options.get("verlet_rebuild_interval", 0)
        )
        if self.verlet_rebuild_interval < 0:
            raise ValueError("verlet_rebuild_interval must be non-negative")
        atomic_numbers = torch.as_tensor(
            atoms.get_atomic_numbers(), dtype=torch.long, device=device
        )
        system_idx = torch.zeros(self.num_atoms, dtype=torch.long, device=device)
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
        if enable_cueq:
            try:
                from nequip.nn._tp_scatter_base import TensorProductScatter

                TensorProductScatter.enable_CuEquivariance(calculator.model)
            except Exception as exc:
                raise RuntimeError(
                    "NequIP Opt4 cuEquivariance fusion could not be enabled"
                ) from exc
        else:
            _assert_plain_eager_model(calculator.model)
        calculator.compute_forces = False
        calculator.compute_stress = False
        self.calculator = calculator
        self.wrapper = ForceOnlyEnergyVJP.from_released_graph_model(
            calculator.model
        ).eval()
        for parameter in self.wrapper.parameters():
            parameter.requires_grad_(False)

        sim_state = ts.io.atoms_to_state([atoms], device=device, dtype=torch.float64)
        sim_state.atomic_numbers = None
        calculator.set_static_geometry(sim_state.row_vector_cell, sim_state.pbc)
        sim_state.positions = initial_state.positions
        exact = calculator.prepare_model_inputs(sim_state)
        self.model_fields = tuple(
            key for key in calculator.model.model_input_fields if key in exact
        )
        missing = [
            key for key in _REQUIRED_MODEL_FIELDS if key not in self.model_fields
        ]
        if missing:
            raise RuntimeError(f"NequIP Opt3 model inputs missing fields {missing}")
        if AtomicDataDict.EDGE_TRANSPOSE_PERM_KEY in exact:
            raise RuntimeError(
                "NequIP Opt3 does not support dynamic edge transpose permutations"
            )
        if AtomicDataDict.EDGE_VECTORS_KEY in self.model_fields:
            raise RuntimeError(
                "NequIP Opt3 requires edge vectors to be derived inside capture"
            )

        exact_inputs = {key: exact[key] for key in self.model_fields}
        exact_edge_index = exact_inputs[AtomicDataDict.EDGE_INDEX_KEY]
        self.initial_maximum_neighbors = int(
            torch.bincount(exact_edge_index[1], minlength=self.num_atoms).max().item()
        )
        neighbors_per_atom, requested_total = _neighbors_per_atom(
            exact_edge_index, num_atoms=self.num_atoms, options=options
        )
        initial_counts = torch.bincount(
            exact_edge_index[1], minlength=self.num_atoms
        )[: self.num_atoms]
        if neighbor_capacities is None and options.get("per_atom_cap", False):
            capacities = capacities_from_counts(
                initial_counts,
                factor=float(options.get("edge_capacity_factor", 1.10)),
                headroom=1,
                alignment=int(options.get("neighbor_capacity_slot_step", 8)),
            )
        else:
            capacities = normalize_neighbor_capacities(
                neighbor_capacities,
                num_atoms=self.num_atoms,
                default=neighbors_per_atom,
            )
        initial_excess = torch.clamp_min(
            initial_counts
            - torch.as_tensor(capacities, dtype=torch.long, device=device),
            0,
        )
        if bool(initial_excess.max().item() > 0):
            raise RuntimeError(
                "NequIP Opt3 per-centre capacity vector is smaller than the "
                "initial graph"
            )
        self.neighbor_capacities = capacities
        if neighbor_capacities is None and options.get("per_atom_cap", False):
            self.capacity_source = "initial-per-atom-cap-vector"
        neighbors_per_atom = max(capacities)
        self.requested_total_edge_capacity = requested_total
        if options.get("neighbors_per_atom") is not None:
            self.capacity_source = (
                "trajectory-total-and-per-atom-probe"
                if requested_total is not None
                else "explicit-per-atom"
            )
        else:
            self.capacity_source = (
                "total-edge-plus-initial-per-atom"
                if requested_total is not None
                else "initial-per-atom-auto"
            )
        edge_capacity = int(sum(capacities))
        self.static_inputs: dict[str, Tensor] = {}
        for key, value in exact_inputs.items():
            if key == AtomicDataDict.POSITIONS_KEY:
                self.static_inputs[key] = (
                    value.detach().clone().contiguous().requires_grad_(True)
                )
            elif key == AtomicDataDict.EDGE_INDEX_KEY:
                self.static_inputs[key] = torch.empty(
                    2, edge_capacity, dtype=value.dtype, device=device
                )
            elif key == AtomicDataDict.EDGE_CELL_SHIFT_KEY:
                self.static_inputs[key] = torch.empty(
                    edge_capacity, 3, dtype=value.dtype, device=device
                )
            else:
                self.static_inputs[key] = value.detach().clone().contiguous()
        self.model_positions = self.static_inputs[AtomicDataDict.POSITIONS_KEY]
        builder_pbc = sim_state.pbc
        if isinstance(builder_pbc, bool):
            builder_pbc = torch.full((3,), builder_pbc, dtype=torch.bool, device=device)
        self.builder = FixedShapeAlchemiNeighborBuilder(
            num_atoms=self.num_atoms,
            cell=self.static_inputs[AtomicDataDict.CELL_KEY].reshape(3, 3),
            pbc=builder_pbc,
            cutoff=float(calculator.model.metadata["r_max"]),
            neighbors_per_atom=neighbors_per_atom,
            neighbor_capacities=capacities,
            output_edge_index=self.static_inputs[AtomicDataDict.EDGE_INDEX_KEY],
            output_edge_shift=self.static_inputs[AtomicDataDict.EDGE_CELL_SHIFT_KEY],
            verlet_skin=float(options.get("verlet_skin", 0.0)),
            verlet_candidate_capacity=options.get("verlet_candidate_capacity"),
        )
        self.builder.initialize_skin(initial_state.positions)
        if self.builder.verlet_skin <= 0:
            self.verlet_rebuild_interval = 0
        self.builder.build(initial_state.positions)
        official_signature = _edge_signature(
            exact_inputs[AtomicDataDict.EDGE_INDEX_KEY],
            exact_inputs[AtomicDataDict.EDGE_CELL_SHIFT_KEY],
        )
        fixed_signature = _edge_signature(
            self.builder.edge_index,
            self.builder.edge_shift,
            self.builder.active_mask,
        )
        if fixed_signature != official_signature:
            official_set = set(official_signature)
            fixed_set = set(fixed_signature)
            raise RuntimeError(
                "NequIP Opt3 fixed builder does not match the initial "
                "AlchemiOps topology; refusing whole-step capture. "
                f"official_edges={len(official_signature)}, "
                f"fixed_edges={len(fixed_signature)}, "
                f"missing_sample={next(iter(official_set - fixed_set), None)}, "
                f"extra_sample={next(iter(fixed_set - official_set), None)}"
            )

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
        exact_validation = {
            key: value.detach().clone().contiguous()
            for key, value in exact_inputs.items()
        }
        exact_validation[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)
        fixed_validation = {
            key: value.detach().clone().contiguous()
            for key, value in self.static_inputs.items()
        }
        fixed_validation[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)
        with torch.enable_grad():
            exact_energy, exact_forces = self.wrapper(exact_validation)
            fixed_energy, fixed_forces = self.wrapper(fixed_validation)
        self.validation_max_abs = {
            "alchemiops_vs_fixed_builder_energy": _max_abs(
                fixed_energy.detach(), exact_energy.detach()
            ),
            "alchemiops_vs_fixed_builder_forces": _max_abs(
                fixed_forces.detach(), exact_forces.detach()
            ),
        }
        _assert_close(
            "fixed-builder energy",
            fixed_energy.detach(),
            exact_energy.detach(),
            rtol=energy_rtol,
            atol=energy_atol,
        )
        _assert_close(
            "fixed-builder forces",
            fixed_forces.detach(),
            exact_forces.detach(),
            rtol=force_rtol,
            atol=force_atol,
        )
        self.fixed_builder_validation_passed = bool(
            torch.allclose(
                fixed_energy.detach(),
                exact_energy.detach(),
                rtol=energy_rtol,
                atol=energy_atol,
            )
            and torch.allclose(
                fixed_forces.detach(),
                exact_forces.detach(),
                rtol=force_rtol,
                atol=force_atol,
            )
        )

        self.positions = initial_state.positions.detach().clone()
        self.momenta = initial_state.momenta.detach().clone()
        self.forces = torch.zeros_like(self.positions)
        self.energy = torch.zeros((), dtype=torch.float64, device=device)
        self.advance = torch.zeros((), dtype=torch.float64, device=device)
        self.step_counter = torch.zeros((), dtype=torch.long, device=device)
        if isinstance(integrator, NoseHooverChainIntegrator):
            self.thermostat_eta = integrator.eta.detach().clone()
            self.thermostat_p_eta = integrator.p_eta.detach().clone()
            self.initial_thermostat_eta = self.thermostat_eta.clone()
            self.initial_thermostat_p_eta = self.thermostat_p_eta.clone()
        else:
            self.thermostat_eta = None
            self.thermostat_p_eta = None
            self.initial_thermostat_eta = None
            self.initial_thermostat_p_eta = None

        self.graph: torch.cuda.CUDAGraph | None = None
        self.capture_count = 0
        self.production_replays = 0
        self.total_replays = 0
        self.output_addresses_stable = False
        self.capture_wall_time_s = 0.0
        self.one_step_validation_passed = False
        self.one_step_validation_tolerances: dict[str, float] = {}
        self.one_step_validation_max_abs: dict[str, float] = {}
        self.one_step_validation_warning: str | None = None
        model_dtype = getattr(calculator.model, "model_dtype", None)
        self.model_dtype = str(model_dtype).removeprefix("torch.")
        self.parameter_dtypes = sorted(
            {
                str(parameter.dtype).removeprefix("torch.")
                for parameter in calculator.model.parameters()
            }
        )

    @torch.no_grad()
    def restore_(self, initial: _InitialState) -> None:
        self.positions.copy_(initial.positions)
        self.momenta.copy_(initial.momenta)
        self.forces.zero_()
        self.energy.zero_()
        self.advance.zero_()
        self.step_counter.zero_()
        if self.thermostat_eta is not None:
            assert self.initial_thermostat_eta is not None
            assert self.thermostat_p_eta is not None
            assert self.initial_thermostat_p_eta is not None
            self.thermostat_eta.copy_(self.initial_thermostat_eta)
            self.thermostat_p_eta.copy_(self.initial_thermostat_p_eta)

    def _nhc_integrate(
        self,
        momenta: Tensor,
        eta: Tensor,
        p_eta: Tensor,
        delta: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        integrator = self.integrator
        assert isinstance(integrator, NoseHooverChainIntegrator)
        p = momenta
        eta_out = eta
        p_eta_out = p_eta
        for _ in range(integrator.chain_loops):
            for coefficient in (
                1.0 / (2.0 - 2.0 ** (1.0 / 3.0)),
                -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0)),
                1.0 / (2.0 - 2.0 ** (1.0 / 3.0)),
            ):
                sub_delta = coefficient * delta / integrator.chain_loops
                delta2, delta4 = sub_delta / 2.0, sub_delta / 4.0
                values = [p_eta_out[j] for j in range(integrator.chain_length)]
                for j in reversed(range(integrator.chain_length)):
                    if j < integrator.chain_length - 1:
                        values[j] = values[j] * torch.exp(
                            -delta4 * values[j + 1] / integrator.Q[j + 1]
                        )
                    if j == 0:
                        g_j = (p.square() / integrator.masses).sum() - (
                            3.0 * integrator.num_atoms * integrator.kT
                        )
                    else:
                        g_j = values[j - 1].square() / integrator.Q[j - 1] - (
                            integrator.kT
                        )
                    values[j] = values[j] + delta2 * g_j
                    if j < integrator.chain_length - 1:
                        values[j] = values[j] * torch.exp(
                            -delta4 * values[j + 1] / integrator.Q[j + 1]
                        )
                stacked = torch.stack(values)
                eta_out = eta_out + sub_delta * stacked / integrator.Q
                p = p * torch.exp(-sub_delta * values[0] / integrator.Q[0])
                for j in range(integrator.chain_length):
                    if j < integrator.chain_length - 1:
                        values[j] = values[j] * torch.exp(
                            -delta4 * values[j + 1] / integrator.Q[j + 1]
                        )
                    if j == 0:
                        g_j = (p.square() / integrator.masses).sum() - (
                            3.0 * integrator.num_atoms * integrator.kT
                        )
                    else:
                        g_j = values[j - 1].square() / integrator.Q[j - 1] - (
                            integrator.kT
                        )
                    values[j] = values[j] + delta2 * g_j
                    if j < integrator.chain_length - 1:
                        values[j] = values[j] * torch.exp(
                            -delta4 * values[j + 1] / integrator.Q[j + 1]
                        )
                p_eta_out = torch.stack(values)
        return p, eta_out, p_eta_out

    def _proposal(self) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        old_momenta = self.momenta
        if isinstance(self.integrator, BerendsenIntegrator):
            integrator = self.integrator
            kinetic = (0.5 * old_momenta.square() / integrator.masses).sum()
            temperature = (
                2.0 * kinetic / (integrator.degrees_of_freedom * units.kB)
            ).clamp_min(1.0e-12)
            scale = torch.sqrt(
                1.0
                + (integrator.target_temperature / temperature - 1.0)
                * (integrator.dt / integrator.taut)
            ).clamp(min=0.9, max=1.1)
            half = old_momenta * scale + 0.5 * integrator.dt * self.forces
            half = half - half.sum(dim=0, keepdim=True) / float(self.num_atoms)
            advanced = self.positions + integrator.dt * half / integrator.masses
            return half, advanced, None, None

        integrator = self.integrator
        assert isinstance(integrator, NoseHooverChainIntegrator)
        assert self.thermostat_eta is not None
        assert self.thermostat_p_eta is not None
        half, eta_half, p_eta_half = self._nhc_integrate(
            old_momenta,
            self.thermostat_eta,
            self.thermostat_p_eta,
            integrator.dt / 2.0,
        )
        half = half + 0.5 * integrator.dt * self.forces
        advanced = self.positions + integrator.dt * half / integrator.masses
        return half, advanced, eta_half, p_eta_half

    def _graph_body(self) -> None:
        with torch.no_grad():
            old_momenta = self.momenta
            old_eta = self.thermostat_eta
            old_p_eta = self.thermostat_p_eta
            half, advanced_positions, eta_half, p_eta_half = self._proposal()
            evaluation_positions = self.positions + self.advance * (
                advanced_positions - self.positions
            )
            self.model_positions.copy_(evaluation_positions)
            graph_step = self.step_counter + self.advance.to(torch.long)
            self.builder.build(self.model_positions, step=graph_step)

        model_energy, model_forces = self.wrapper(self.static_inputs)

        with torch.no_grad():
            forces = model_forces.to(dtype=self.positions.dtype)
            if isinstance(self.integrator, BerendsenIntegrator):
                advanced_momenta = half + 0.5 * self.integrator.dt * forces
                eta_final = None
                p_eta_final = None
            else:
                assert eta_half is not None and p_eta_half is not None
                half = half + 0.5 * self.integrator.dt * forces
                advanced_momenta, eta_final, p_eta_final = self._nhc_integrate(
                    half,
                    eta_half,
                    p_eta_half,
                    self.integrator.dt / 2.0,
                )
            final_momenta = old_momenta + self.advance * (
                advanced_momenta - old_momenta
            )
            self.positions.copy_(evaluation_positions)
            self.momenta.copy_(final_momenta)
            self.forces.copy_(forces)
            self.energy.copy_(model_energy.reshape(-1)[0].to(torch.float64))
            if eta_final is not None:
                assert old_eta is not None and old_p_eta is not None
                assert self.thermostat_eta is not None
                assert self.thermostat_p_eta is not None
                self.thermostat_eta.copy_(
                    old_eta + self.advance * (eta_final - old_eta)
                )
                self.thermostat_p_eta.copy_(
                    old_p_eta + self.advance * (p_eta_final - old_p_eta)
                )
            self.step_counter.add_(self.advance.to(torch.long))

    def capture(self, initial: _InitialState) -> None:
        if self.graph is not None:
            raise RuntimeError("NequIP Opt3 graph has already been captured")
        current = torch.cuda.current_stream(self.device)
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(current)
        with torch.cuda.stream(side), torch.enable_grad():
            self.restore_(initial)
            self.advance.fill_(1.0)
            for _ in range(self.capture_warmup):
                self._graph_body()
            self.restore_(initial)
            self.advance.fill_(1.0)
            self.builder.reset_stats()
        current.wait_stream(side)
        torch.cuda.synchronize(self.device)

        graph = torch.cuda.CUDAGraph()
        started = time.perf_counter()
        try:
            with torch.enable_grad(), torch.cuda.graph(graph, stream=side):
                self._graph_body()
            torch.cuda.synchronize(self.device)
        except Exception as exc:
            raise RuntimeError(
                "NequIP Opt3 whole-step CUDA Graph capture failed; "
                "no eager fallback is permitted"
            ) from exc
        self.capture_wall_time_s = time.perf_counter() - started
        self.graph = graph
        self.capture_count = 1
        addresses = self._addresses()
        self.restore_(initial)
        self.builder.reset_stats()
        torch.cuda.synchronize(self.device)
        self.output_addresses_stable = addresses == self._addresses()
        if not self.output_addresses_stable:
            raise RuntimeError("NequIP Opt3 persistent state address changed")

    def _addresses(self) -> dict[str, int]:
        addresses = {
            "positions": self.positions.data_ptr(),
            "momenta": self.momenta.data_ptr(),
            "forces": self.forces.data_ptr(),
            "energy": self.energy.data_ptr(),
            "model_positions": self.model_positions.data_ptr(),
            "edge_index": self.builder.edge_index.data_ptr(),
            "edge_shift": self.builder.edge_shift.data_ptr(),
        }
        if self.thermostat_eta is not None:
            addresses["thermostat_eta"] = self.thermostat_eta.data_ptr()
            assert self.thermostat_p_eta is not None
            addresses["thermostat_p_eta"] = self.thermostat_p_eta.data_ptr()
        return addresses

    @torch.no_grad()
    def _state_snapshot(self) -> dict[str, Tensor]:
        snapshot = {
            "positions": self.positions.clone(),
            "momenta": self.momenta.clone(),
            "forces": self.forces.clone(),
            "energy": self.energy.clone(),
        }
        if self.thermostat_eta is not None:
            snapshot["thermostat_eta"] = self.thermostat_eta.clone()
            assert self.thermostat_p_eta is not None
            snapshot["thermostat_p_eta"] = self.thermostat_p_eta.clone()
        return snapshot

    def validate_one_step(
        self, initial: _InitialState, *, options: dict[str, Any]
    ) -> None:
        """Compare eager and captured initial-evaluation plus one MD step.

        Non-finite state, capacity overflow, and address changes are structural
        failures.  Finite numerical differences are recorded and warned about,
        matching the benchmark's report-only numerical policy.
        """

        if self.graph is None:
            raise RuntimeError("capture must complete before one-step validation")
        addresses = self._addresses()

        self.restore_(initial)
        self.builder.reset_stats()
        with torch.enable_grad():
            self.advance.zero_()
            self._graph_body()
            self.advance.fill_(1.0)
            self._graph_body()
        torch.cuda.synchronize(self.device)
        self.builder.raise_for_overflow()
        eager = self._state_snapshot()

        self.restore_(initial)
        self.builder.reset_stats()
        self.advance.zero_()
        self.graph.replay()
        self.advance.fill_(1.0)
        self.graph.replay()
        torch.cuda.synchronize(self.device)
        self.builder.raise_for_overflow()
        replay = self._state_snapshot()

        if self._addresses() != addresses:
            raise RuntimeError(
                "NequIP Opt3 persistent address changed during one-step validation"
            )
        for owner, state in (("eager", eager), ("graph", replay)):
            invalid = [
                name
                for name, value in state.items()
                if not bool(torch.isfinite(value).all())
            ]
            if invalid:
                raise FloatingPointError(
                    f"NequIP Opt3 {owner} one-step state has non-finite {invalid}"
                )

        state_rtol = float(options.get("one_step_state_rtol", 1.0e-10))
        state_atol = float(options.get("one_step_state_atol", 1.0e-10))
        force_rtol = float(options.get("capture_force_rtol", 1.0e-4))
        force_atol = float(options.get("capture_force_atol", 1.0e-5))
        energy_rtol = float(options.get("capture_energy_rtol", 1.0e-5))
        energy_atol = float(options.get("capture_energy_atol", 1.0e-5))
        self.one_step_validation_tolerances = {
            "state_rtol": state_rtol,
            "state_atol": state_atol,
            "force_rtol": force_rtol,
            "force_atol": force_atol,
            "energy_rtol": energy_rtol,
            "energy_atol": energy_atol,
        }
        self.one_step_validation_max_abs = {
            name: _max_abs(replay[name], eager[name]) for name in eager
        }
        passed = True
        for name in eager:
            if name == "forces":
                rtol, atol = force_rtol, force_atol
            elif name == "energy":
                rtol, atol = energy_rtol, energy_atol
            else:
                rtol, atol = state_rtol, state_atol
            passed = passed and bool(
                torch.allclose(replay[name], eager[name], rtol=rtol, atol=atol)
            )
        self.one_step_validation_passed = passed
        if not passed:
            detail = ", ".join(
                f"{name}={difference:.6g}"
                for name, difference in self.one_step_validation_max_abs.items()
            )
            self.one_step_validation_warning = (
                "NequIP Opt3 eager-vs-Graph one-step numerical tolerance "
                f"exceeded ({detail}); continuing under report-only policy"
            )
            warnings.warn(self.one_step_validation_warning, stacklevel=2)

        self.restore_(initial)
        self.builder.reset_stats()

    def reset_production(self, initial: _InitialState) -> None:
        if self.graph is None:
            raise RuntimeError("capture must complete before production")
        self.restore_(initial)
        self.builder.reset_stats()
        self.builder.initialize_skin(self.positions)
        self.production_replays = 0

    def evaluate_initial(self) -> ModelOutput:
        if self.graph is None:
            raise RuntimeError("capture must complete before replay")
        self.advance.zero_()
        self.graph.replay()
        self.advance.fill_(1.0)
        self.production_replays += 1
        self.total_replays += 1
        return self.output_view()

    def step(self) -> ModelOutput:
        if self.graph is None:
            raise RuntimeError("capture must complete before replay")
        if (
            self.verlet_rebuild_interval
            and self.production_replays % self.verlet_rebuild_interval == 0
        ):
            self.builder.initialize_skin(self.positions)
        self.graph.replay()
        self.production_replays += 1
        self.total_replays += 1
        return self.output_view()

    def output_view(self) -> ModelOutput:
        return ModelOutput(energy=self.energy, forces=self.forces, stress=None)

    def raise_for_overflow(self) -> None:
        self.builder.raise_for_overflow()


def _validate_request(request: MDRunRequest) -> Path:
    if request.model != "nequip" or request.stage != "opt3":
        raise ValueError(
            f"nequip.md_stages.opt3 owns nequip/opt3, got "
            f"{request.model}/{request.stage}"
        )
    if request.backend != "whole-step-cuda-graph":
        raise ValueError("NequIP Opt3 backend must be 'whole-step-cuda-graph'")
    if request.config.device.split(":", maxsplit=1)[0] != "cuda":
        raise ValueError("NequIP Opt3 is CUDA-only")
    if request.config.dtype != "float64":
        raise ValueError("NequIP Opt3 requires an FP64 MD state")
    if request.config.integrator not in {"berendsen", "nose_hoover_chain"}:
        raise ValueError("NequIP Opt3 supports Berendsen and Nose-Hoover chain")
    if request.atoms.constraints:
        raise NotImplementedError("NequIP Opt3 does not ignore ASE constraints")
    if len(request.atoms) < 2:
        raise ValueError("NVT MD requires at least two atoms")
    if not bool(np.all(request.atoms.pbc)):
        raise ValueError("NequIP Opt3 fixed builder requires full PBC")
    if request.config.collect_trajectory:
        raise ValueError(
            "NequIP Opt3 captures fixed-cell force-only inference and cannot "
            "produce stress trajectories"
        )
    if request.output_path is not None:
        raise ValueError("NequIP Opt3 output_path requires unsupported trajectory")
    path = Path(request.model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not path.name.endswith(".nequip.zip"):
        raise ValueError("NequIP Opt3 requires the official .nequip.zip package")
    if request.options.get("neighborlist_backend", "alchemiops") != "alchemiops":
        raise ValueError("NequIP Opt3 requires AlchemiOps reference semantics")
    for key in (
        "compiled_model_path",
        "compile",
        "aotinductor",
        "open_equivariance",
        "model_specific_fusion",
        "allow_tf32",
        "amp",
        "graph_buckets",
        "transactional_recovery",
    ):
        if request.options.get(key):
            raise ValueError(f"NequIP Opt3 forbids route option {key!r}")
    return path


def _validate_finite(state: GPUMDState) -> None:
    if state.output is None:
        raise RuntimeError("NequIP Opt3 final state was not evaluated")
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
        raise FloatingPointError(f"NequIP Opt3 final state has non-finite {invalid}")


def run_md(request: MDRunRequest) -> MDRunResult:
    """Run strict whole-step CUDA Graph NequIP NVT MD."""

    model_path = _validate_request(request)
    if not torch.cuda.is_available():
        raise RuntimeError("NequIP Opt3 requested CUDA, but CUDA is unavailable")
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
    initial = _InitialState(
        positions=torch.as_tensor(
            np.asarray(atoms.positions), dtype=torch.float64, device=device
        ).clone(),
        momenta=torch.as_tensor(
            np.asarray(atoms.get_momenta()), dtype=torch.float64, device=device
        ).clone(),
    )
    masses = torch.as_tensor(
        np.asarray(atoms.get_masses()), dtype=torch.float64, device=device
    ).clone()
    integrator = _build_integrator(request, masses)
    if not isinstance(integrator, (BerendsenIntegrator, NoseHooverChainIntegrator)):
        raise TypeError("NequIP Opt3 received an unsupported integrator")
    profiler = CudaPhaseProfiler(
        enabled=performance_profile_requested(request.options), device=device
    )
    engine = WholeStepCUDAGraphMD(
        atoms,
        str(model_path),
        integrator,
        initial,
        device=device,
        options=request.options,
        neighbor_capacities=request.options.get("neighbor_capacities"),
    )
    engine.capture(initial)
    engine.validate_one_step(initial, options=request.options)
    engine.reset_production(initial)
    state = GPUMDState(engine.positions, engine.momenta)
    observations: list[MDObservation] = []
    observation_steps = set(config.observation_steps)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    profiler.start()
    started = time.perf_counter()
    with profiler.phase("whole_step_cuda_graph_replay"):
        state.output = engine.evaluate_initial()
    if config.collect_statistics and 0 in observation_steps:
        engine.raise_for_overflow()
        observations.append(_observation(state, step=0, masses=masses))
    for step in range(1, config.steps + 1):
        with profiler.phase("whole_step_cuda_graph_replay"):
            state.output = engine.step()
        if config.collect_statistics and step in observation_steps:
            # Observation already synchronizes energy/forces to the host.  Read
            # the device overflow telemetry at the same synchronization point.
            engine.raise_for_overflow()
            observations.append(_observation(state, step=step, masses=masses))
    torch.cuda.synchronize(device)
    engine.raise_for_overflow()
    profiler.stop()
    elapsed = time.perf_counter() - started
    performance_profile = profiler.summary(synchronize=False)
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1.0e9
    expected_replays = config.steps + 1
    if engine.production_replays != expected_replays:
        raise RuntimeError(
            "NequIP Opt3 production replay count mismatch: "
            f"expected {expected_replays}, observed {engine.production_replays}"
        )
    _validate_finite(state)
    builder_stats = engine.builder.stats()

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
            "engine": "nequip_whole_step_cuda_graph",
            "backend": request.backend,
            "model_path": str(model_path),
            "source_model_kind": "nequip_saved_package",
            "source_loader": "NequIPTorchSimCalc.from_saved_model",
            "torch_sim_version": _distribution_version("torch-sim-atomistic"),
            "nvalchemiops_version": _distribution_version("nvalchemi-toolkit-ops"),
            "checkpoint_model_dtype": engine.model_dtype,
            "checkpoint_parameter_dtypes": engine.parameter_dtypes,
            "md_state_device": str(device),
            "md_state_dtype": "float64",
            "captured_components": [
                "fixed_shape_pbc_neighbor_builder",
                "eager_energy_model",
                "force_vjp",
                "md_integrator",
                "thermostat_state",
                "persistent_state_update",
            ],
            "uncaptured_components": ["statistics", "overflow_host_check"],
            "integrator": config.integrator,
            "force_wrapper": "fixed_cell_position_only_vjp",
            "model_parameters_require_grad": False,
            "fixed_input_addresses": True,
            "fixed_output_addresses": True,
            "capture_state_data_ptrs": engine._addresses(),
            "requested_total_edge_capacity": (engine.requested_total_edge_capacity),
            "fixed_edge_capacity": engine.builder.edge_capacity,
            "neighbors_per_atom": engine.builder.neighbors_per_atom,
            "initial_maximum_neighbors": engine.initial_maximum_neighbors,
            "edge_capacity_source": engine.capacity_source,
            "capacity_total_to_per_atom_guard_slots": 0,
            "edge_capacity_policy": (
                "esen_cap_per_atom"
                if len(set(engine.neighbor_capacities)) > 1
                else "esen_cap_uniform_per_centre"
            ),
            "neighbor_capacities": engine.neighbor_capacities,
            "edge_overflow_policy": "device_detect_raise_at_sync_no_fallback",
            "edge_padding": "distributed_far_periodic_self_edge_zero_cutoff",
            "sink_padding": "distributed_far_periodic_self_edge_zero_cutoff",
            "sink_padding_uses_extra_atoms": False,
            "fixed_builder_validation_passed": (engine.fixed_builder_validation_passed),
            "one_step_eager_vs_graph_validation_passed": (
                engine.one_step_validation_passed
            ),
            "one_step_eager_vs_graph_validation_tolerances": (
                engine.one_step_validation_tolerances
            ),
            "one_step_eager_vs_graph_validation_max_abs": (
                engine.one_step_validation_max_abs
            ),
            "one_step_eager_vs_graph_validation_warning": (
                engine.one_step_validation_warning
            ),
            "validation_tolerances": engine.validation_tolerances,
            "validation_max_abs": engine.validation_max_abs,
            "numerical_validation_failure_policy": "report_only",
            "capture_failure_policy": "raise_no_fallback",
            "cuda_graph_capture_count": engine.capture_count,
            "capture_count": engine.capture_count,
            "graph_capture_scope": "whole-md-step",
            "production_replays": engine.production_replays,
            "cuda_graph_production_replays": engine.production_replays,
            "expected_production_replays": expected_replays,
            "cuda_graph_replay_output_addresses_stable": (
                engine.output_addresses_stable
            ),
            "cuda_graph_capture_wall_time_s": engine.capture_wall_time_s,
            "stress_requested": False,
            "stress_supported": False,
            "warmup_steps": config.warmup_steps,
            "capture_warmup": engine.capture_warmup,
            "verlet_rebuild_interval": engine.verlet_rebuild_interval,
            "warmup_state_restored": True,
            "transactional_recovery": False,
            "transaction_rollback": False,
            "neighbor_list_inside_cuda_graph": True,
            "cuda_graph_neighbor_build_inside": True,
            "capacity_overflow_count": int(
                builder_stats["fixed_builder_capacity_misses"]
            ),
            "graph_bucket_count": 1,
            "performance_profile": performance_profile,
            **builder_stats,
            **OPT3_POLICY,
            "cuequivariance": bool(engine.cueq_enabled),
            "tensor_product_accelerator": (
                "cuequivariance" if engine.cueq_enabled else None
            ),
            "model_specific_fusion": bool(engine.cueq_enabled),
        },
    )
    validate_result(request, result)
    return result


__all__ = [
    "FixedShapeAlchemiNeighborBuilder",
    "OPT3_POLICY",
    "WholeStepCUDAGraphMD",
    "run_md",
]
