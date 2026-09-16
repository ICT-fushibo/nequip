"""NequIP frozen TP-v1 control plus a FastEq full TP-scatter boundary.

Algorithmic adaptation of FastEq commit 40ba40e72bee769d74a869bb4a4ba820ee1c55c0
(MIT); the integration repository carries the complete third-party notice.
"""
from __future__ import annotations

from torch import nn

from md_benchmark.opt4_fx import CheckedRegion, install_tp_regions
from md_benchmark.opt4_registry import fixed_csr_layout, record


class _Uniform1DTPScatter(nn.Module):
    """Source gather + original TP instructions + live destination reduction.

    The fixed-capacity builder guarantees the edge tensor shape, but padding
    slots use distributed sink destinations.  Consequently ``edge_dst`` is a
    live graph input and cannot be replaced by the capacity-slot row layout.
    """

    def __init__(self, tp, edge_rows, rows: int) -> None:
        super().__init__()
        object.__setattr__(self, "_tp", tp)
        self.register_buffer("edge_rows", edge_rows, persistent=False)
        self.rows = int(rows)

    def set_layout(self, edge_rows, rows: int) -> None:
        self.edge_rows = edge_rows
        self.rows = int(rows)

    def forward(self, x, edge_attr, edge_weight, edge_dst, edge_src):
        edge_features = self._tp(x.index_select(0, edge_src), edge_attr, edge_weight)
        out = edge_features.new_zeros((self.rows, edge_features.shape[-1]))
        out.index_add_(0, edge_dst, edge_features)
        return out


class _FastEqTensorProductScatter(nn.Module):
    """Instance-local wrapper; compact setup graphs retain released behavior."""

    def __init__(self, original, region, edge_capacity: int) -> None:
        super().__init__()
        self.original = original
        self._opt4_fasteq_uniform1d = region
        self._opt4_edge_capacity = int(edge_capacity)

    def forward(self, x, edge_attr, edge_weight, edge_dst, edge_src):
        if edge_src.shape[0] != self._opt4_edge_capacity:
            return self.original(x, edge_attr, edge_weight, edge_dst, edge_src)
        return self._opt4_fasteq_uniform1d(
            x, edge_attr, edge_weight, edge_dst, edge_src
        )


def _layout(options, parameter):
    row_ptr, edge_rows, _max_row = fixed_csr_layout(options, parameter)
    return edge_rows, int(row_ptr.shape[0] - 1)


def refresh(model, options) -> None:
    edge_rows, rows = _layout(options, next(model.parameters()))
    for module in model.modules():
        region = getattr(module, "_opt4_fasteq_uniform1d", None)
        if isinstance(region, CheckedRegion):
            module._opt4_edge_capacity = int(edge_rows.numel())
            region.reference.set_layout(edge_rows, rows)
            region.signatures.clear()


def install(model, passes, report, options):
    # This pair is the already validated NequIP TP-v1 control.  The candidate
    # keeps those regions everywhere except inside the full TP-scatter boundary.
    tp_paths = tuple(
        path
        for path, module in model.named_modules()
        if type(module).__name__ == "TensorProductScatter"
    )

    def control_scope(path: str) -> bool:
        lower = path.lower()
        selected = any(
            token in lower
            for token in (
                "tp",
                "tensor_product",
                "self_connection",
                "selfconnection",
                "sc.",
            )
        )
        if "fasteq_uniform1d_tp_scatter" not in passes:
            return selected
        return selected and not any(
            path == prefix or path.startswith(prefix + ".") for prefix in tp_paths
        )

    install_tp_regions(model, passes, report, control_scope)
    if "fasteq_uniform1d_tp_scatter" not in passes:
        return

    edge_rows, rows = _layout(options, next(model.parameters()))
    modules = []
    for path, module in list(model.named_modules()):
        if type(module).__name__ != "TensorProductScatter":
            continue
        detail = {
            "module": path,
            "validated_shapes": 0,
            "benchmark_requested": report.get("benchmark_boundaries", False),
        }
        boundary = _Uniform1DTPScatter(module.tp, edge_rows, rows)
        region = CheckedRegion(boundary, detail)
        parent_path, _, leaf = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(
            parent,
            leaf,
            _FastEqTensorProductScatter(module, region, edge_rows.numel()),
        )
        modules.append(detail)
    record(
        report,
        "fasteq_uniform1d_tp_scatter",
        len(modules),
        "inductor-triton-full-tp-scatter-aot-vjp",
        modules=modules,
        base_configuration=["tp_pointwise_reduce", "tp_layout_pack"],
        fused_boundaries=[
            "source-gather",
            "tp-instruction-chain",
            "live-destination-reduce",
        ],
        topology="fixed-shape-live-edge-dst",
        nested_fx_tp_regions=False,
        gemm="original-e3nn",
        backward="aot-compiled-complete-input-vjp",
        replay_runtime_compile=False,
    )
