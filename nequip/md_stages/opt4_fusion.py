"""NequIP-owned TP and self-connection selection, including packaged classes."""
from md_benchmark.opt4_fx import install_tp_regions


def install(model, passes, report):
    # Structural path matching also works for torch.package's class namespaces.
    install_tp_regions(model, passes, report,
        lambda path: any(x in path.lower() for x in ("tp", "tensor_product", "self_connection", "selfconnection", "sc.")))
