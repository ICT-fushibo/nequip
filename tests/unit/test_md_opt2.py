"""Contract tests for strict NequIP model-only CUDA Graph Opt2."""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch
from ase import Atoms

from nequip.data import AtomicDataDict
from nequip.md_stages.opt2 import (
    OPT2_POLICY,
    FixedCapacityModelInputs,
    ForceOnlyEnergyVJP,
    ModelOnlyCUDAGraphEvaluator,
    _maximum_neighbors_per_atom,
    _edge_capacity,
    _max_abs,
    _validate_request,
    run_md,
)


class _QuadraticEnergy(torch.nn.Module):
    def forward(self, data):
        positions = data[AtomicDataDict.POSITIONS_KEY]
        return {
            AtomicDataDict.TOTAL_ENERGY_KEY: positions.square().sum().reshape(1)
        }


def _exact_inputs(edge_count: int = 2) -> dict[str, torch.Tensor]:
    return {
        AtomicDataDict.POSITIONS_KEY: torch.tensor(
            [[0.2, -0.1, 0.3], [0.7, 0.4, -0.2]], dtype=torch.float64
        ),
        AtomicDataDict.CELL_KEY: torch.eye(3, dtype=torch.float64).reshape(1, 3, 3),
        AtomicDataDict.BATCH_KEY: torch.zeros(2, dtype=torch.long),
        AtomicDataDict.NUM_NODES_KEY: torch.tensor([2], dtype=torch.long),
        AtomicDataDict.ATOM_TYPE_KEY: torch.zeros(2, dtype=torch.long),
        AtomicDataDict.EDGE_INDEX_KEY: torch.zeros(2, edge_count, dtype=torch.long),
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: torch.zeros(
            edge_count, 3, dtype=torch.float64
        ),
    }


def test_opt2_policy_is_strictly_model_only() -> None:
    assert OPT2_POLICY["cuda_graph"] is True
    assert OPT2_POLICY["cuda_graph_scope"] == "model-only"
    assert OPT2_POLICY["neighbor_list_in_cuda_graph"] is False
    assert OPT2_POLICY["md_in_cuda_graph"] is False
    assert OPT2_POLICY["model_compile"] is False
    assert OPT2_POLICY["aotinductor"] is False
    assert OPT2_POLICY["open_equivariance"] is False
    assert OPT2_POLICY["model_specific_fusion"] is False
    assert OPT2_POLICY["tf32"] is False
    assert OPT2_POLICY["amp"] is False


def test_route_preserves_baselines_and_dispatches_opt2(monkeypatch) -> None:
    from nequip import md_route

    sentinel = object()
    monkeypatch.setattr(md_route, "run_optimized_stage", lambda *args, **kw: sentinel)
    request = SimpleNamespace(model="nequip", stage="opt2")
    assert md_route.BASELINE_MODES == ("E0", "E1", "B0", "B1")
    assert md_route.run_md(request) is sentinel


def test_force_only_wrapper_returns_conservative_vjp() -> None:
    wrapper = ForceOnlyEnergyVJP(_QuadraticEnergy())
    inputs = _exact_inputs()
    inputs[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)
    energy, forces = wrapper(inputs)
    torch.testing.assert_close(
        energy, inputs[AtomicDataDict.POSITIONS_KEY].square().sum().reshape(1)
    )
    torch.testing.assert_close(
        forces, -2.0 * inputs[AtomicDataDict.POSITIONS_KEY]
    )


def test_force_only_wrapper_accepts_torch_package_class_identity() -> None:
    def init(self):
        torch.nn.Module.__init__(self)
        self.func = _QuadraticEnergy()
        self.do_derivatives = True

    packaged_type = type(
        "ForceStressOutput",
        (torch.nn.Module,),
        {
            "__module__": "torch_package_0.nequip.nn._grad_output",
            "__init__": init,
        },
    )

    class GraphModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = packaged_type()

    graph_model = GraphModel()
    wrapper = ForceOnlyEnergyVJP.from_released_graph_model(graph_model)
    assert isinstance(wrapper.energy_model, _QuadraticEnergy)


def test_force_only_wrapper_accepts_packaged_non_bool_derivative_flag() -> None:
    def init(self):
        torch.nn.Module.__init__(self)
        self.func = _QuadraticEnergy()
        self.do_derivatives = 1

    packaged_type = type(
        "ForceStressOutput",
        (torch.nn.Module,),
        {
            "__module__": "torch_package_0.nequip.nn.grad_output",
            "__init__": init,
        },
    )
    wrapper = ForceOnlyEnergyVJP.from_released_graph_model(packaged_type())
    assert isinstance(wrapper.energy_model, _QuadraticEnergy)


def test_fixed_capacity_inputs_keep_addresses_and_pad_far_edges() -> None:
    exact = _exact_inputs(edge_count=2)
    fixed = FixedCapacityModelInputs.from_exact(
        exact,
        edge_capacity=4,
        padding_shift=torch.tensor([7.0, 0.0, 0.0]),
    )
    pointers = fixed.data_ptrs()
    assert fixed.tensors[AtomicDataDict.POSITIONS_KEY].requires_grad
    torch.testing.assert_close(
        fixed.tensors[AtomicDataDict.EDGE_CELL_SHIFT_KEY][2:],
        torch.tensor([[7.0, 0.0, 0.0], [7.0, 0.0, 0.0]], dtype=torch.float64),
    )

    moved = _exact_inputs(edge_count=3)
    moved[AtomicDataDict.POSITIONS_KEY].add_(1.0)
    fixed.update(moved)
    assert fixed.data_ptrs() == pointers
    assert fixed.peak_edge_count == 3
    torch.testing.assert_close(
        fixed.tensors[AtomicDataDict.POSITIONS_KEY],
        moved[AtomicDataDict.POSITIONS_KEY],
    )


def test_fixed_capacity_overflow_has_no_fallback() -> None:
    fixed = FixedCapacityModelInputs.from_exact(
        _exact_inputs(edge_count=2),
        edge_capacity=2,
        padding_shift=torch.tensor([7.0, 0.0, 0.0]),
    )
    with pytest.raises(RuntimeError, match="capacity overflow"):
        fixed.update(_exact_inputs(edge_count=3))


def test_edge_capacity_validation_and_rounding() -> None:
    assert _edge_capacity(100, {"edge_capacity": 256}) == 256
    assert _edge_capacity(100, {"edge_capacity_factor": 1.25}) == 256
    with pytest.raises(ValueError, match="positive integer"):
        _edge_capacity(100, {"edge_capacity": 0})
    with pytest.raises(ValueError, match=">= 1"):
        _edge_capacity(100, {"edge_capacity_factor": 0.5})


def test_probe_maximum_neighbors_uses_nequip_receiver_axis() -> None:
    edge_index = torch.tensor(
        [[1, 2, 0, 2, 0, 1], [0, 0, 1, 1, 2, 2]], dtype=torch.long
    )
    assert _maximum_neighbors_per_atom(edge_index, num_atoms=3) == 2


def _request(tmp_path, **overrides):
    model_path = tmp_path / "model.nequip.zip"
    model_path.write_bytes(b"placeholder")
    values = {
        "model": "nequip",
        "stage": "opt2",
        "backend": "model-only-cuda-graph",
        "model_path": str(model_path),
        "atoms": Atoms(
            "H2",
            positions=[[0, 0, 0], [0, 0, 0.7]],
            cell=[5, 5, 5],
            pbc=True,
        ),
        "config": SimpleNamespace(
            device="cuda:0",
            dtype="float64",
            collect_trajectory=False,
        ),
        "options": {},
        "output_path": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_opt2_validation_accepts_strict_contract(tmp_path) -> None:
    path = _validate_request(_request(tmp_path))
    assert path.name == "model.nequip.zip"


def test_opt2_validation_rejects_wrong_backend_and_trajectory(tmp_path) -> None:
    with pytest.raises(ValueError, match="model-only-cuda-graph"):
        _validate_request(_request(tmp_path, backend="gpu-resident"))
    request = _request(tmp_path)
    request.config.collect_trajectory = True
    with pytest.raises(ValueError, match="cannot produce stress trajectories"):
        _validate_request(request)


@pytest.mark.parametrize(
    "forbidden",
    [
        "compiled_model_path",
        "compile",
        "aotinductor",
        "open_equivariance",
        "model_specific_fusion",
        "allow_tf32",
        "amp",
        "whole_step_cuda_graph",
    ],
)
def test_opt2_validation_rejects_later_or_orthogonal_accelerators(
    tmp_path, forbidden: str
) -> None:
    request = _request(tmp_path)
    request.options[forbidden] = "enabled"
    with pytest.raises(ValueError, match="forbids route option"):
        _validate_request(request)


def test_evaluator_source_keeps_neighbor_build_outside_replay() -> None:
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(ModelOnlyCUDAGraphEvaluator.__call__))
    )
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "_prepare_exact" in calls
    assert "update" in calls
    assert "replay" in calls
    source = textwrap.dedent(inspect.getsource(ModelOnlyCUDAGraphEvaluator.__call__))
    assert source.index("_prepare_exact") < source.index("replay")
    assert source.index("fixed.update") < source.index("replay")
    assert "_output_ptrs" in source
    assert "production_replays += 1" in source


def test_replay_counter_reset_and_max_abs() -> None:
    evaluator = object.__new__(ModelOnlyCUDAGraphEvaluator)
    evaluator.production_replays = 9
    evaluator.initial_max_neighbors = 7
    evaluator.peak_neighbors_per_atom = 9
    evaluator.reset_production_replays()
    assert evaluator.production_replays == 0
    assert evaluator.peak_neighbors_per_atom == evaluator.initial_max_neighbors
    assert _max_abs(torch.tensor([1.0, 4.0]), torch.tensor([2.0, 1.0])) == 3.0


def test_run_requires_one_initial_plus_one_replay_per_step() -> None:
    source = textwrap.dedent(inspect.getsource(run_md))
    assert "expected_replays = config.steps + 1" in source
    assert "production replay count mismatch" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_force_vjp_replays_from_fixed_cuda_address() -> None:
    device = torch.device("cuda:0")
    wrapper = ForceOnlyEnergyVJP(_QuadraticEnergy().to(device))
    inputs = {
        key: value.to(device) for key, value in _exact_inputs().items()
    }
    inputs[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)

    warmup = torch.cuda.Stream(device=device)
    with torch.cuda.stream(warmup), torch.enable_grad():
        wrapper(inputs)
    warmup.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.enable_grad(), torch.cuda.graph(graph):
        energy, forces = wrapper(inputs)
    input_ptr = inputs[AtomicDataDict.POSITIONS_KEY].data_ptr()
    output_ptrs = (energy.data_ptr(), forces.data_ptr())

    with torch.no_grad():
        inputs[AtomicDataDict.POSITIONS_KEY].copy_(
            torch.tensor(
                [[1.0, 2.0, 3.0], [-1.0, 0.5, 0.25]],
                dtype=torch.float64,
                device=device,
            )
        )
    graph.replay()
    torch.cuda.synchronize(device)
    first_energy = energy.clone()
    first_forces = forces.clone()
    graph.replay()
    torch.cuda.synchronize(device)
    assert inputs[AtomicDataDict.POSITIONS_KEY].data_ptr() == input_ptr
    assert (energy.data_ptr(), forces.data_ptr()) == output_ptrs
    expected_energy = inputs[AtomicDataDict.POSITIONS_KEY].square().sum().reshape(1)
    torch.testing.assert_close(energy, expected_energy)
    torch.testing.assert_close(
        forces, -2.0 * inputs[AtomicDataDict.POSITIONS_KEY]
    )
    torch.testing.assert_close(energy, first_energy)
    torch.testing.assert_close(forces, first_forces)
