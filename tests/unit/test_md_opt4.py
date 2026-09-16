"""CPU contract checks for the NequIP Opt4 route."""

import pytest
import torch
from torch import nn

from nequip.md_stages import opt4
from nequip.md_stages.opt4_fusion import (
    _FastEqTensorProductScatter,
    _Uniform1DTPScatter,
)


def test_opt4_rejects_other_route() -> None:
    with pytest.raises(ValueError, match="NequIP Opt4 route"):
        opt4.run_md(type("Request", (), {"model": "dpa4", "stage": "opt4"})())


class _TP(nn.Module):
    def forward(self, x, edge_attr, edge_weight):
        return x * edge_attr + edge_weight


class _NativeScatter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tp = _TP()

    def forward(self, x, edge_attr, edge_weight, edge_dst, edge_src):
        values = self.tp(x.index_select(0, edge_src), edge_attr, edge_weight)
        out = values.new_zeros((x.shape[0], values.shape[-1]))
        return out.index_add(0, edge_dst, values)


def test_fasteq_scatter_uses_live_destination_indices() -> None:
    """Distributed sink slots must not be reduced with a cached row layout."""

    native = _NativeScatter()
    cached_rows = torch.tensor([0, 0, 1, 1, 2, 2])
    live_dst = torch.tensor([2, 1, 0, 2, 1, 0])
    live_src = torch.tensor([0, 1, 2, 0, 1, 2])
    boundary = _Uniform1DTPScatter(native.tp, cached_rows, rows=3)
    wrapped = _FastEqTensorProductScatter(native, boundary, edge_capacity=6)

    x_native = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    edge_native = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    weight_native = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    x_fused = x_native.detach().clone().requires_grad_(True)
    edge_fused = edge_native.detach().clone().requires_grad_(True)
    weight_fused = weight_native.detach().clone().requires_grad_(True)

    expected = native(
        x_native, edge_native, weight_native, live_dst, live_src
    )
    actual = wrapped(x_fused, edge_fused, weight_fused, live_dst, live_src)
    torch.testing.assert_close(actual, expected)

    probe = torch.cos(torch.arange(expected.numel(), dtype=expected.dtype)).view_as(
        expected
    )
    expected_grads = torch.autograd.grad(
        (expected * probe).sum(), (x_native, edge_native, weight_native)
    )
    actual_grads = torch.autograd.grad(
        (actual * probe).sum(), (x_fused, edge_fused, weight_fused)
    )
    for got, want in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got, want)
