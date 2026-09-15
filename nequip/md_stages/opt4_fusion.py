"""NequIP Opt4: frozen TP-v1 passes plus incremental fixed CSR scatter."""
from __future__ import annotations

from torch import nn

from md_benchmark.opt4_fx import CheckedRegion, install_tp_regions
from md_benchmark.opt4_ops import csr_segment_sum
from md_benchmark.opt4_registry import fixed_csr_layout, record


class _NativeScatter(nn.Module):
    def __init__(self, edge_rows, rows):
        super().__init__()
        self.register_buffer("edge_rows", edge_rows, persistent=False)
        self.rows = int(rows)

    def forward(self, values):
        out = values.new_zeros((self.rows, *values.shape[1:]))
        out.index_add_(0, self.edge_rows, values)
        return out


class _FixedCSR(nn.Module):
    def __init__(self, row_ptr, edge_rows, max_row):
        super().__init__()
        self.register_buffer("row_ptr", row_ptr, persistent=False)
        self.register_buffer("edge_rows", edge_rows, persistent=False)
        self.max_row = int(max_row)

    def set_layout(self, row_ptr, edge_rows, max_row):
        self.row_ptr = row_ptr
        self.edge_rows = edge_rows
        self.max_row = int(max_row)

    def forward(self, values):
        return csr_segment_sum(
            values.contiguous(), self.row_ptr, self.edge_rows, self.max_row
        )


class _TensorProductScatterCSR(nn.Module):
    """Instance-local adapter; the packaged checkpoint class is untouched."""

    def __init__(self, original, region):
        super().__init__()
        self.original = original
        self._opt4_tp_scatter_csr = region

    def forward(self, x, edge_attr, edge_weight, edge_dst, edge_src):
        del edge_dst
        edge_features = self.original.tp(x[edge_src], edge_attr, edge_weight)
        return self._opt4_tp_scatter_csr(edge_features)


def refresh(model, options):
    row_ptr, edge_rows, max_row = fixed_csr_layout(
        options, next(model.parameters())
    )
    for module in model.modules():
        region = getattr(module, "_opt4_tp_scatter_csr", None)
        if isinstance(region, CheckedRegion):
            region.reference.edge_rows = edge_rows
            region.reference.rows = row_ptr.shape[0] - 1
            region.compiled.set_layout(row_ptr, edge_rows, max_row)
            region.signatures.clear()


def install(model, passes, report, options):
    # This exact pair is the already validated NequIP TP-v1 control.  Do not
    # refactor its graph partitioning or numerical order in this iteration.
    install_tp_regions(
        model,
        passes,
        report,
        lambda path: any(
            token in path.lower()
            for token in ("tp", "tensor_product", "self_connection", "selfconnection", "sc.")
        ),
    )
    if "tp_scatter_csr" not in passes:
        return
    row_ptr, edge_rows, max_row = fixed_csr_layout(
        options, next(model.parameters())
    )
    modules = []
    for path, module in list(model.named_modules()):
        if type(module).__name__ != "TensorProductScatter":
            continue
        detail = {
            "module": path,
            "validated_shapes": 0,
            "benchmark_requested": report.get("benchmark_boundaries", False),
        }
        region = CheckedRegion(
            _NativeScatter(edge_rows, row_ptr.shape[0] - 1),
            detail,
            _FixedCSR(row_ptr, edge_rows, max_row),
        )
        parent_path, _, leaf = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, leaf, _TensorProductScatterCSR(module, region))
        modules.append(detail)
    record(
        report,
        "tp_scatter_csr",
        len(modules),
        "triton-fixed-csr-explicit-vjp",
        modules=modules,
        base_configuration=["tp_pointwise_reduce", "tp_layout_pack"],
        fused_boundaries=["tp-output-destination-reduce"],
        tensor_product="frozen-nequip-tp-v1",
        backward_recomputes_reference=False,
        fusion_scope="forward-and-backward",
    )
