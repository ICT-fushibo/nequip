"""Contract tests for NequIP whole-step CUDA Graph Opt3."""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch
from ase import Atoms

from nequip.md_stages.opt3 import (
    OPT3_POLICY,
    FixedShapeAlchemiNeighborBuilder,
    WholeStepCUDAGraphMD,
    _neighbors_per_atom,
    _validate_request,
    run_md,
)
from nequip.md_stages.opt1 import NoseHooverChainIntegrator


def test_opt3_policy_is_strict_whole_step() -> None:
    assert OPT3_POLICY["cuda_graph"] is True
    assert OPT3_POLICY["cuda_graph_scope"] == "whole-step"
    assert OPT3_POLICY["neighbor_list_in_cuda_graph"] is True
    assert OPT3_POLICY["md_in_cuda_graph"] is True
    assert OPT3_POLICY["model_compile"] is False
    assert OPT3_POLICY["aotinductor"] is False
    assert OPT3_POLICY["open_equivariance"] is False
    assert OPT3_POLICY["model_specific_fusion"] is False
    assert OPT3_POLICY["tf32"] is False
    assert OPT3_POLICY["amp"] is False


def test_route_preserves_earlier_stages_and_dispatches_opt3(monkeypatch) -> None:
    from nequip import md_route

    sentinel = object()
    monkeypatch.setattr(md_route, "run_optimized_stage", lambda *args, **kw: sentinel)
    request = SimpleNamespace(model="nequip", stage="opt3")
    assert md_route.run_md(request) is sentinel


def _builder(
    positions: torch.Tensor, *, cutoff: float, capacity: int
) -> FixedShapeAlchemiNeighborBuilder:
    edge_capacity = positions.shape[0] * capacity
    return FixedShapeAlchemiNeighborBuilder(
        num_atoms=positions.shape[0],
        cell=torch.eye(3, dtype=torch.float64) * 5.0,
        pbc=torch.ones(3, dtype=torch.bool),
        cutoff=cutoff,
        neighbors_per_atom=capacity,
        output_edge_index=torch.empty(2, edge_capacity, dtype=torch.long),
        output_edge_shift=torch.empty(edge_capacity, 3, dtype=torch.float64),
    )


def test_fixed_builder_uses_distributed_far_self_edge_padding() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64)
    builder = _builder(positions, cutoff=1.0, capacity=2)
    edge_index, edge_shift = builder.build(positions)
    assert edge_index.shape == (2, 4)
    assert edge_shift.shape == (4, 3)
    assert int(builder.current_real_edges) == 2
    padded = edge_shift.abs().sum(dim=1) > 0
    assert int(padded.sum()) == 2
    torch.testing.assert_close(edge_index[0, padded], edge_index[1, padded])
    # Sink indices rotate instead of concentrating every inactive edge at zero.
    assert torch.unique(edge_index[0, padded]).numel() == 2
    builder.raise_for_overflow()


def test_fixed_builder_device_telemetry_rejects_per_centre_overflow() -> None:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 0.7, 0.0]],
        dtype=torch.float64,
    )
    builder = _builder(positions, cutoff=1.1, capacity=1)
    builder.build(positions, step=torch.tensor(7))
    assert int(builder.capacity_misses) == 1
    assert int(builder.first_overflow_step) == 7
    with pytest.raises(RuntimeError, match="per-centre CAP overflow"):
        builder.raise_for_overflow()


def test_total_probe_capacity_maps_to_per_centre_cap() -> None:
    edge_index = torch.tensor(
        [[1, 2, 0, 2, 0, 1], [0, 0, 1, 1, 2, 2]], dtype=torch.long
    )
    capacity, requested = _neighbors_per_atom(
        edge_index,
        num_atoms=3,
        options={"edge_capacity": 9, "neighbor_capacity_slot_step": 1},
    )
    assert capacity == 3
    assert requested == 9
    explicit, requested = _neighbors_per_atom(
        edge_index, num_atoms=3, options={"neighbors_per_atom": 7}
    )
    assert explicit == 7
    assert requested is None


def test_initial_capacity_applies_only_one_guard_bucket() -> None:
    edge_index = torch.stack(
        [
            torch.arange(80, dtype=torch.long) % 2,
            torch.zeros(80, dtype=torch.long),
        ]
    )
    capacity, requested = _neighbors_per_atom(
        edge_index,
        num_atoms=2,
        options={"edge_capacity_factor": 1.10, "neighbor_capacity_slot_step": 8},
    )
    assert capacity == 88
    assert requested is None


def test_pure_nhc_update_matches_opt1_integrator() -> None:
    masses = torch.tensor([63.546, 63.546], dtype=torch.float64)
    captured = NoseHooverChainIntegrator(
        masses,
        timestep_fs=0.25,
        temperature_k=1000.0,
        thermostat_time_fs=25.0,
    )
    reference = NoseHooverChainIntegrator(
        masses,
        timestep_fs=0.25,
        temperature_k=1000.0,
        thermostat_time_fs=25.0,
    )
    momenta = torch.tensor([[0.1, -0.2, 0.3], [-0.2, 0.15, -0.05]], dtype=torch.float64)
    owner = object.__new__(WholeStepCUDAGraphMD)
    owner.integrator = captured
    actual_p, actual_eta, actual_p_eta = owner._nhc_integrate(
        momenta, captured.eta, captured.p_eta, captured.dt / 2.0
    )
    expected_p = reference._integrate_chain(momenta, reference.dt / 2.0)
    torch.testing.assert_close(actual_p, expected_p)
    torch.testing.assert_close(actual_eta, reference.eta)
    torch.testing.assert_close(actual_p_eta, reference.p_eta)


def _request(tmp_path, **overrides):
    model_path = tmp_path / "model.nequip.zip"
    model_path.write_bytes(b"placeholder")
    values = {
        "model": "nequip",
        "stage": "opt3",
        "backend": "whole-step-cuda-graph",
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
            integrator="nose_hoover_chain",
            collect_trajectory=False,
        ),
        "options": {},
        "output_path": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_opt3_validation_accepts_nhc_and_berendsen(tmp_path) -> None:
    assert _validate_request(_request(tmp_path)).name == "model.nequip.zip"
    request = _request(tmp_path)
    request.config.integrator = "berendsen"
    assert _validate_request(request).name == "model.nequip.zip"


def test_opt3_validation_rejects_wrong_backend_and_trajectory(tmp_path) -> None:
    with pytest.raises(ValueError, match="whole-step-cuda-graph"):
        _validate_request(_request(tmp_path, backend="model-only-cuda-graph"))
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
        "graph_buckets",
        "transactional_recovery",
    ],
)
def test_opt3_rejects_orthogonal_or_deferred_accelerators(
    tmp_path, forbidden: str
) -> None:
    request = _request(tmp_path)
    request.options[forbidden] = "enabled"
    with pytest.raises(ValueError, match="forbids route option"):
        _validate_request(request)


def test_graph_body_contains_builder_model_integrator_and_state_updates() -> None:
    source = textwrap.dedent(inspect.getsource(WholeStepCUDAGraphMD._graph_body))
    tree = ast.parse(source)
    attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "build" in attributes
    assert "copy_" in attributes
    assert "_proposal" in attributes
    assert "_nhc_integrate" in attributes
    assert "wrapper" in source


def test_capture_is_followed_by_full_one_step_validation() -> None:
    validation = textwrap.dedent(
        inspect.getsource(WholeStepCUDAGraphMD.validate_one_step)
    )
    snapshot = textwrap.dedent(
        inspect.getsource(WholeStepCUDAGraphMD._state_snapshot)
    )
    assert validation.count("self._graph_body()") == 2
    assert validation.count("self.graph.replay()") == 2
    for field in (
        "positions",
        "momenta",
        "forces",
        "energy",
        "thermostat_eta",
        "thermostat_p_eta",
    ):
        assert field in snapshot
    assert validation.count("self._state_snapshot()") == 2
    assert "torch.isfinite" in validation
    assert "warnings.warn" in validation

    run_source = textwrap.dedent(inspect.getsource(run_md))
    assert run_source.index("engine.capture(initial)") < run_source.index(
        "engine.validate_one_step"
    )


def test_run_requires_one_initial_plus_one_replay_per_step() -> None:
    source = textwrap.dedent(inspect.getsource(run_md))
    assert "expected_replays = config.steps + 1" in source
    assert "production replay count mismatch" in source
    assert "engine.raise_for_overflow()" in source
