"""Contract and ASE-alignment tests for NequIP GPU-resident Opt1."""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.nvtberendsen import NVTBerendsen

from nequip.md_stages.opt1 import (
    OPT1_POLICY,
    BerendsenIntegrator,
    EagerNequIPTorchSimEvaluator,
    GPUMDState,
    ModelOutput,
    NoseHooverChainIntegrator,
    _assert_plain_eager_model,
    _frame,
    _validate_request,
)


class _ConstantForceCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, forces: np.ndarray) -> None:
        super().__init__()
        self.forces = forces

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": self.forces.copy()}


class _ConstantEvaluator:
    def __init__(self, forces: torch.Tensor) -> None:
        self.forces = forces

    def __call__(self, positions: torch.Tensor) -> ModelOutput:
        return ModelOutput(
            energy=positions.new_zeros(()),
            forces=self.forces.to(positions),
            stress=None,
        )


@pytest.mark.parametrize("integrator_name", ["berendsen", "nose_hoover_chain"])
def test_gpu_integrators_match_one_ase_329_step_on_cpu(integrator_name: str):
    positions = np.array([[0.1, 0.2, 0.3], [1.1, 0.7, 0.4]])
    momenta = np.array([[0.22, -0.13, 0.31], [-0.19, 0.17, -0.28]])
    forces = np.array([[0.03, -0.02, 0.01], [-0.04, 0.02, -0.01]])
    masses = np.array([12.0, 16.0])
    atoms = Atoms("CO", positions=positions, masses=masses)
    atoms.set_momenta(momenta)
    atoms.calc = _ConstantForceCalculator(forces)

    if integrator_name == "berendsen":
        ase_md = NVTBerendsen(
            atoms,
            timestep=units.fs,
            temperature_K=300.0,
            taut=100.0 * units.fs,
            fixcm=True,
        )
        gpu_integrator = BerendsenIntegrator(
            torch.tensor(masses, dtype=torch.float64),
            timestep_fs=1.0,
            temperature_k=300.0,
            thermostat_time_fs=100.0,
            degrees_of_freedom=atoms.get_number_of_degrees_of_freedom(),
        )
    else:
        ase_md = NoseHooverChainNVT(
            atoms,
            timestep=units.fs,
            temperature_K=300.0,
            tdamp=100.0 * units.fs,
        )
        gpu_integrator = NoseHooverChainIntegrator(
            torch.tensor(masses, dtype=torch.float64),
            timestep_fs=1.0,
            temperature_k=300.0,
            thermostat_time_fs=100.0,
        )

    ase_md.run(1)
    state = GPUMDState(
        positions=torch.tensor(positions, dtype=torch.float64),
        momenta=torch.tensor(momenta, dtype=torch.float64),
    )
    gpu_integrator.step(
        state, _ConstantEvaluator(torch.tensor(forces, dtype=torch.float64))
    )
    np.testing.assert_allclose(
        state.positions.numpy(), atoms.positions, rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(
        state.momenta.numpy(), atoms.get_momenta(), rtol=1e-13, atol=1e-13
    )


def test_opt1_policy_excludes_later_accelerators() -> None:
    assert OPT1_POLICY == {
        "gpu_resident": True,
        "model_compile": False,
        "aotinductor": False,
        "open_equivariance": False,
        "cuda_graph": False,
        "model_specific_fusion": False,
        "tf32": False,
        "neighbor_list_backend": "alchemiops",
    }


def test_opt1_rejects_accelerated_equivariance_module() -> None:
    class Accelerated(torch.nn.Module):
        _nequip_custom_ops_libs = ("openequivariance",)

    with pytest.raises(RuntimeError, match="accelerated equivariance"):
        _assert_plain_eager_model(Accelerated())


def test_hot_loop_has_no_host_transfers_or_later_stage_calls() -> None:
    functions = (
        EagerNequIPTorchSimEvaluator.__call__,
        BerendsenIntegrator.step,
        NoseHooverChainIntegrator.step,
        NoseHooverChainIntegrator._integrate_chain,
        NoseHooverChainIntegrator._integrate_loop,
        NoseHooverChainIntegrator._integrate_p_eta_j,
    )
    forbidden = {"cpu", "numpy", "item", "compile", "from_compiled_model"}
    for function in functions:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        assert not attributes & forbidden


def test_alchemiops_nominal_cell_selection_has_no_host_predicate() -> None:
    from nequip.data._nl import alchemiops_batch_cell_list

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(alchemiops_batch_cell_list))
    )
    for conditional in (node for node in ast.walk(tree) if isinstance(node, ast.If)):
        predicate_attributes = {
            node.attr
            for node in ast.walk(conditional.test)
            if isinstance(node, ast.Attribute)
        }
        assert "any" not in predicate_attributes


def test_opt1_rejects_non_gpu_resident_backend_before_model_load() -> None:
    request = SimpleNamespace(
        model="nequip",
        stage="opt1",
        backend="E0",
        config=SimpleNamespace(device="cuda:0", dtype="float64"),
        atoms=Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]]),
        model_path="missing.nequip.zip",
        options={},
    )
    with pytest.raises(ValueError, match="gpu-resident"):
        _validate_request(request)


def test_opt1_rejects_compiled_artifact(tmp_path) -> None:
    artifact = tmp_path / "model.pt2"
    artifact.write_bytes(b"not-a-model")
    request = SimpleNamespace(
        model="nequip",
        stage="opt1",
        backend="gpu-resident",
        config=SimpleNamespace(device="cuda:0", dtype="float64"),
        atoms=Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]]),
        model_path=str(artifact),
        options={},
    )
    with pytest.raises(ValueError, match="official .nequip.zip"):
        _validate_request(request)


def test_route_keeps_baseline_modes_and_dispatches_opt1(monkeypatch) -> None:
    from nequip import md_route

    assert md_route.BASELINE_MODES == ("E0", "E1", "B0", "B1")
    sentinel = object()
    monkeypatch.setattr(md_route, "run_optimized_stage", lambda *args, **kw: sentinel)
    request = SimpleNamespace(model="nequip", stage="opt1")
    assert md_route.run_md(request) is sentinel


def test_matbench_frame_contains_step_zero_energy_forces_and_stress() -> None:
    atoms = Atoms(
        "H2",
        positions=[[0, 0, 0], [0, 0, 0.7]],
        cell=[5, 5, 5],
        pbc=True,
    )
    state = GPUMDState(
        positions=torch.tensor(atoms.positions, dtype=torch.float64),
        momenta=torch.zeros(2, 3, dtype=torch.float64),
        output=ModelOutput(
            energy=torch.tensor(-1.25, dtype=torch.float64),
            forces=torch.ones(2, 3, dtype=torch.float64),
            stress=torch.eye(3, dtype=torch.float64),
        ),
    )
    frame = _frame(atoms, state, step=0, require_stress=True)
    assert frame.info["md_step"] == 0
    assert frame.get_potential_energy() == -1.25
    np.testing.assert_allclose(frame.get_forces(), np.ones((2, 3)))
    np.testing.assert_allclose(frame.get_stress(voigt=False), np.eye(3))


def test_torchsim_persistent_topology_does_not_mutate_state() -> None:
    ts = pytest.importorskip("torch_sim")
    from nequip.data import AtomicDataDict
    from nequip.integrations.torchsim import NequIPTorchSimCalc

    class FakeModel(torch.nn.Module):
        def forward(self, data):
            assert data[AtomicDataDict.ATOMIC_NUMBERS_KEY] is not None
            assert data[AtomicDataDict.BATCH_KEY] is not None
            n_atoms = data[AtomicDataDict.POSITIONS_KEY].shape[0]
            return {
                AtomicDataDict.TOTAL_ENERGY_KEY: torch.zeros(1),
                AtomicDataDict.FORCE_KEY: torch.zeros(n_atoms, 3),
                AtomicDataDict.STRESS_KEY: torch.zeros(1, 3, 3),
            }

    atomic_numbers = torch.tensor([1, 1], dtype=torch.long)
    system_idx = torch.zeros(2, dtype=torch.long)
    calc = NequIPTorchSimCalc(
        FakeModel(),
        atomic_numbers=atomic_numbers,
        system_idx=system_idx,
    )
    atoms = Atoms(
        "H2",
        positions=[[0, 0, 0], [0, 0, 0.7]],
        cell=[5, 5, 5],
        pbc=True,
    )
    state = ts.io.atoms_to_state([atoms], "cpu", dtype=torch.float32)
    original_system_idx = state.system_idx
    state.atomic_numbers = None
    output = calc(state)
    assert output["forces"].shape == (2, 3)
    assert state.atomic_numbers is None
    assert state.system_idx is original_system_idx
