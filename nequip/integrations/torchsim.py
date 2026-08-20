"""Wrapper for NequIP framework models in torch-sim"""

import torch

import torch_sim as ts
from torch_sim.models.interface import ModelInterface

from nequip.data import AtomicDataDict
from nequip.data._nl import NEIGHBORLIST_BACKEND_ALCHEMIOPS

from .mixins import _IntegrationLoaderMixin

from collections.abc import Callable
from typing import Union, List, Dict, Optional
from pathlib import Path


class NequIPTorchSimCalc(_IntegrationLoaderMixin, ModelInterface):
    """NequIP framework torch-sim calculator.

    This torch-sim calculator is compatible with models from the NequIP framework,
    including NequIP and Allegro models.

    The recommended way to use this calculator is with a compiled model, i.e.
    ``nequip-compile`` the model and load it into the calculator with
    ``NequIPTorchSimCalc.from_compiled_model(...)``.

    Args:
        model (:class:`torch.nn.Module`): a model in the NequIP framework
        device (str or :class:`torch.device`): device for model to evaluate on,
            e.g. ``"cpu"`` or ``"cuda"`` (default: ``"cpu"``)
        transforms (List[Callable]): list of data transforms
        atomic_numbers (:class:`torch.Tensor` or None): atomic numbers with shape
            ``[n_atoms]``. If provided at initialization, cannot be provided
            again during forward pass
        system_idx (:class:`torch.Tensor` or None): batch indices with shape ``[n_atoms]``
            indicating which system each atom belongs to. If not provided
            with ``atomic_numbers``, all atoms are assumed to be in the same system
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: Union[str, torch.device] = "cpu",
        transforms: List[Callable] = [],
        atomic_numbers: torch.Tensor | None = None,
        system_idx: torch.Tensor | None = None,
        model_requires_contiguous_inputs: bool = False,
    ) -> None:
        """Initialize the NequIP torch-sim calculator.

        .. note::
            This is a low-level initializer. Users should typically use
            ``from_compiled_model`` instead.
        """
        super().__init__()

        # === set up ModelInterface attributes ===
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        self._dtype = torch.float64  # default dtype for calculations
        self._compute_forces = True
        self._compute_stress = True
        self._memory_scales_with = "n_atoms_x_density"
        self.model_requires_contiguous_inputs = model_requires_contiguous_inputs
        self._static_cell: torch.Tensor | None = None
        self._static_pbc: torch.Tensor | None = None

        if not isinstance(model, torch.nn.Module):
            raise TypeError("Invalid model type. Must be a torch.nn.Module.")
        self.model = model.to(self._device)

        # move transforms to device (they are torch.nn.Module's)
        self.transforms = [t.to(self._device) for t in transforms]

        # store flag to track if atomic numbers were provided at init
        self.atomic_numbers_in_init = atomic_numbers is not None
        self.n_systems = 1

        # set up batch information if atomic numbers are provided
        if atomic_numbers is not None:
            if system_idx is None:
                # if batch is not provided, assume all atoms belong to same system
                system_idx = torch.zeros(
                    len(atomic_numbers), dtype=torch.long, device=self._device
                )

            self.setup_from_system_idx(atomic_numbers, system_idx)
            self._cache_static_topology_transforms()

    @ModelInterface.compute_forces.setter
    def compute_forces(self, value: bool) -> None:
        """Set whether to compute forces."""
        self._compute_forces = value

    @ModelInterface.compute_stress.setter
    def compute_stress(self, value: bool) -> None:
        """Set whether to compute stress."""
        self._compute_stress = value

    @classmethod
    def _get_aoti_compile_target(cls) -> Dict:
        from nequip.scripts._compile_utils import COMPILE_TARGET_DICT, AOTI_BATCH_TARGET

        return COMPILE_TARGET_DICT[AOTI_BATCH_TARGET]

    @classmethod
    def from_compiled_model(
        cls,
        compile_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        chemical_species_to_atom_type_map: Optional[Union[Dict[str, str], bool]] = None,
        neighborlist_backend: str = NEIGHBORLIST_BACKEND_ALCHEMIOPS,
        **kwargs,
    ):
        # AOTI inputs are specialized on their compile-time strides.  Eager
        # models accept regular strided tensors, so only compiled artifacts
        # need the defensive final contiguous pass in ``forward``.
        kwargs.setdefault("model_requires_contiguous_inputs", True)
        return super().from_compiled_model(
            compile_path=compile_path,
            device=device,
            chemical_species_to_atom_type_map=chemical_species_to_atom_type_map,
            neighborlist_backend=neighborlist_backend,
            **kwargs,
        )

    def setup_from_system_idx(
        self, atomic_numbers: torch.Tensor, system_idx: torch.Tensor
    ) -> None:
        """Set up internal state from atomic numbers and system indices.

        Args:
            atomic_numbers (:class:`torch.Tensor`): atomic numbers with shape ``[n_atoms]``.
            system_idx (:class:`torch.Tensor`): system indices with shape ``[n_atoms]``.
        """
        self.atomic_numbers = atomic_numbers.contiguous()
        self.system_idx = system_idx.contiguous()

        # determine number of systems and atoms per system
        self.n_systems = system_idx.max().item() + 1
        self.total_atoms = atomic_numbers.shape[0]
        self.num_nodes = torch.bincount(
            self.system_idx, minlength=self.n_systems
        ).contiguous()

    def _cache_static_topology_transforms(self) -> None:
        """Apply immutable chemical-species mapping once at initialization."""

        from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper

        static_data = {
            AtomicDataDict.ATOMIC_NUMBERS_KEY: self.atomic_numbers,
        }
        dynamic_transforms = []
        self.atom_types: torch.Tensor | None = None
        for transform in self.transforms:
            if isinstance(transform, ChemicalSpeciesToAtomTypeMapper):
                static_data = transform(static_data)
                self.atom_types = static_data[
                    AtomicDataDict.ATOM_TYPE_KEY
                ].contiguous()
            else:
                dynamic_transforms.append(transform)
        self.transforms = dynamic_transforms

    def set_static_geometry(
        self, row_vector_cell: torch.Tensor, pbc: bool | torch.Tensor
    ) -> None:
        """Cache fixed-cell geometry for an NVT trajectory.

        The caller opts into this cache and is responsible for calling this
        method again if the cell or periodic-boundary flags change.  This is
        appropriate for fixed-cell NVT MD and avoids materializing ``cell.mT``
        and expanded stride-zero PBC views on every force evaluation.
        """

        if tuple(row_vector_cell.shape) not in {
            (3, 3),
            (self.n_systems, 3, 3),
        }:
            raise ValueError(
                "Expected row-vector cell shape [3, 3] or "
                f"[{self.n_systems}, 3, 3], got {tuple(row_vector_cell.shape)}"
            )
        self._static_cell = row_vector_cell.reshape(-1, 3, 3).contiguous()
        self._static_pbc = self._batch_pbc(pbc).contiguous()

    def _batch_pbc(self, pbc: bool | torch.Tensor) -> torch.Tensor:
        """Return PBC flags with shape ``[n_systems, 3]``."""

        if isinstance(pbc, bool):
            return torch.full(
                (self.n_systems, 3), pbc, dtype=torch.bool, device=self._device
            )
        if pbc.ndim == 1 and tuple(pbc.shape) == (3,):
            return pbc.unsqueeze(0).expand(self.n_systems, 3)
        if pbc.ndim == 2 and tuple(pbc.shape) == (self.n_systems, 3):
            return pbc
        raise ValueError(
            f"Expected pbc shape [3] or [{self.n_systems}, 3], "
            f"got {tuple(pbc.shape)}"
        )

    def forward(self, state: ts.SimState) -> dict[str, torch.Tensor]:  # noqa: C901
        """Compute energies, forces, and stresses.

        Args:
            state (:class:`~torch_sim.SimState`): state object containing positions, cell,
                and system information.

        Returns:
            dict[str, :class:`torch.Tensor`]: computed properties (``"energy"``, ``"forces"``, ``"stress"``).
        """
        sim_state = state

        # handle input validation for atomic numbers
        if sim_state.atomic_numbers is None and not self.atomic_numbers_in_init:
            raise ValueError(
                "Atomic numbers must be provided in either the constructor or forward."
            )
        if sim_state.atomic_numbers is not None and self.atomic_numbers_in_init:
            raise ValueError(
                "Atomic numbers cannot be provided in both the constructor and forward."
            )

        # Use immutable topology supplied at initialization without mutating
        # SimState.  The previous implementation assigned into SimState and
        # always forwarded ``sim_state.atomic_numbers`` even when atomic
        # numbers were supplied to the constructor.  Apart from being
        # inconsistent, the fallback made persistent GPU-resident states
        # impossible to use without a per-forward ``torch.equal`` host sync.
        if self.atomic_numbers_in_init:
            # The Opt1 NVT route has immutable topology.  Always use the
            # constructor-owned contiguous tensors rather than equivalent
            # SimState views, so no comparison, bincount, or materialization
            # is performed in the MD hot loop.
            atomic_numbers = self.atomic_numbers
            system_idx = self.system_idx
            num_nodes = self.num_nodes
        else:
            atomic_numbers = sim_state.atomic_numbers
            system_idx = sim_state.system_idx
            if atomic_numbers is None:
                raise ValueError(
                    "Atomic numbers must be provided if not set during initialization"
                )
            if system_idx is None:
                raise ValueError(
                    "System indices must be provided if not set during initialization"
                )

        # update batch information if new atomic numbers are provided
        if (
            atomic_numbers is not None
            and not self.atomic_numbers_in_init
            and atomic_numbers is not getattr(self, "atomic_numbers", None)
            and not torch.equal(
                atomic_numbers,
                getattr(self, "atomic_numbers", torch.zeros(0, device=self._device)),
            )
        ):
            self.setup_from_system_idx(atomic_numbers, system_idx)
        if not self.atomic_numbers_in_init:
            num_nodes = self.num_nodes

        # === prepare raw dict ===
        # convert PBC to tensor with shape [n_systems, 3] for batched data
        if self._static_cell is None:
            cell = sim_state.row_vector_cell
            pbc_tensor = self._batch_pbc(sim_state.pbc)
        else:
            assert self._static_pbc is not None
            cell = self._static_cell
            pbc_tensor = self._static_pbc

        data: dict[str, torch.Tensor] = {
            AtomicDataDict.POSITIONS_KEY: sim_state.positions,
            AtomicDataDict.CELL_KEY: cell,
            AtomicDataDict.PBC_KEY: pbc_tensor,
            AtomicDataDict.BATCH_KEY: system_idx,
            AtomicDataDict.NUM_NODES_KEY: num_nodes,
            AtomicDataDict.ATOMIC_NUMBERS_KEY: atomic_numbers,
        }
        if self.atomic_numbers_in_init and self.atom_types is not None:
            data[AtomicDataDict.ATOM_TYPE_KEY] = self.atom_types

        # === apply transforms ===
        for t in self.transforms:
            data = t(data)

        # AOTI reads inputs using compile-time strides.  Preserve the old
        # defensive behavior for compiled calculators, but do not blanket-
        # materialize an eager model's whole data dictionary every MD step.
        if self.model_requires_contiguous_inputs:
            data = {
                k: (v.contiguous() if torch.is_tensor(v) else v)
                for k, v in data.items()
            }

        # === run model ===
        out = self.model(data)

        # === collect outputs ===
        results: dict[str, torch.Tensor] = {}

        energy = out[AtomicDataDict.TOTAL_ENERGY_KEY]
        if energy is not None:
            results["energy"] = energy.view(-1).detach()
        else:
            results["energy"] = torch.zeros(self.n_systems, device=self._device)

        if self.compute_forces:
            forces = out[AtomicDataDict.FORCE_KEY]
            if forces is not None:
                results["forces"] = forces.detach()

        if self.compute_stress:
            stress = out[AtomicDataDict.STRESS_KEY]
            if stress is not None:
                results["stress"] = stress.detach()

        self.save_extra_outputs(out, results)

        return results

    def save_extra_outputs(
        self, out: dict[str, torch.Tensor], results: dict[str, torch.Tensor]
    ) -> None:
        # subclasses can implement this method to process extra outputs without code duplication
        pass
