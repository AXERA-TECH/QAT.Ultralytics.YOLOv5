"""Minimal reproduction: ONNX initializer dedup causes cross-epoch structural inconsistency.

The QAT export path is:
    torch.onnx.export(model, ..., dynamo=True) -> onnx_program.optimize()
    onnxslim.slim(model)

onnx_program.optimize() runs onnxscript.optimizer.optimize_ir(), which includes
onnx_ir.passes.common.DeduplicateInitializersPass().  onnxslim.slim() also runs
that pass internally.  The pass merges initializers whose (dtype, shape, tobytes)
are identical, so:
    * epoch with conv1_scale == conv3_scale  -> 1 initializer (merged)
    * epoch with conv1_scale != conv3_scale  -> 2 initializers
The saved models then differ in initializer count AND in every val_N name that
follows the merged/kept tensors.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import onnx
import torch
import torch.nn as nn


@dataclasses.dataclass
class ExportResult:
    label: str
    raw: onnx.ModelProto
    optimized: onnx.ModelProto
    slimmed: onnx.ModelProto


def _weight_branch_model(same: bool) -> nn.Module:
    """Two independent linear branches with separate weights.

    Branch A keeps the same weight in both variants; branch B shares A's weight
    when ``same_values=True`` (bit-identical) and differs otherwise.
    """

    class TwoBranches(nn.Module):
        def __init__(self, same: bool):
            super().__init__()
            self.wa = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
            if same:
                self.wb = nn.Parameter(self.wa.detach().clone())
            else:
                self.wb = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 5.0]]))
            self.suffix = nn.Parameter(torch.tensor([10.0, 20.0]))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            ya = torch.nn.functional.linear(x, self.wa)
            yb = torch.nn.functional.linear(x, self.wb)
            return (ya + yb) * self.suffix

    return TwoBranches(same)


def _qdq_branch_model(same: bool) -> nn.Module:
    """Two independent fake-quant branches (Q/DQ pattern, as in QAT export).

    Branch A always uses scale = 0.5; branch B uses 0.5 when ``same_scale``
    and 0.7 otherwise.  In the ONNX graph these become QuantizeLinear /
    DequantizeLinear nodes whose scale/zero_point initializers may be merged.
    """

    class TwoQdqBranches(nn.Module):
        def __init__(self, same: bool):
            super().__init__()
            self.register_buffer("scale_a", torch.tensor([0.5]))
            self.register_buffer("scale_b", torch.tensor([0.5 if same else 0.7]))
            self.register_buffer("zp", torch.tensor([0], dtype=torch.int32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qmin, qmax, dtype = -128, 127, torch.int8
            qa = torch.ops.quantized_decomposed.quantize_per_tensor(
                x, self.scale_a, self.zp, qmin, qmax, dtype
            )
            da = torch.ops.quantized_decomposed.dequantize_per_tensor(
                qa, self.scale_a, self.zp, qmin, qmax, dtype
            )
            qb = torch.ops.quantized_decomposed.quantize_per_tensor(
                x, self.scale_b, self.zp, qmin, qmax, dtype
            )
            db = torch.ops.quantized_decomposed.dequantize_per_tensor(
                qb, self.scale_b, self.zp, qmin, qmax, dtype
            )
            return da + db

    return TwoQdqBranches(same)


def export_variant(model: nn.Module, label: str) -> ExportResult:
    model.eval()
    x = torch.randn(1, 2)
    with torch.no_grad():
        onnx_program = torch.onnx.export(model, (x,), dynamo=True)
        raw_proto = onnx_program.model_proto

        onnx_program.optimize()
        optimized_proto = onnx_program.model_proto

        slimmed_proto = onnxslim_slim(optimized_proto)

    return ExportResult(label=label, raw=raw_proto, optimized=optimized_proto, slimmed=slimmed_proto)


def onnxslim_slim(model: onnx.ModelProto) -> onnx.ModelProto:
    from onnxslim import slim

    return slim(model)


def summarize(model: onnx.ModelProto) -> dict[str, object]:
    init_names = [i.name for i in model.graph.initializer]
    qdq = [
        (n.op_type, [inp for inp in n.input])
        for n in model.graph.node
        if n.op_type in ("QuantizeLinear", "DequantizeLinear")
    ]
    return {
        "initializer_count": len(init_names),
        "initializer_names": init_names,
        "qdq_nodes": qdq,
        "topology": [n.op_type for n in model.graph.node],
        "inputs": [i.name for i in model.graph.input],
        "outputs": [o.name for o in model.graph.output],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdq", action="store_true", help="use Q/DQ-style model")
    args = parser.parse_args()

    builder = _qdq_branch_model if args.qdq else _weight_branch_model
    kind = "QDQ-branches" if args.qdq else "weight-branches"

    results = [
        export_variant(builder(same=True), f"{kind}: values EQUAL (epoch A)"),
        export_variant(builder(same=False), f"{kind}: values DIFFER (epoch B)"),
    ]

    for r in results:
        print(f"\n=== {r.label} ===")
        for stage, proto in (
            ("raw (dynamo export)", r.raw),
            ("after onnx_program.optimize()", r.optimized),
            ("after onnxslim.slim()", r.slimmed),
        ):
            s = summarize(proto)
            print(f"[{stage}] initializers={s['initializer_count']} names={s['initializer_names']}")
            for op_type, inputs in s["qdq_nodes"]:
                print(f"    {op_type} inputs={inputs}")

    print("\n=== comparison ===")
    eq, diff = results
    for stage in ("raw", "optimized", "slimmed"):
        e, d = summarize(getattr(eq, stage)), summarize(getattr(diff, stage))
        print(
            f"{stage:>8}: equal-epoch={e['initializer_count']} initializers, "
            f"differ-epoch={d['initializer_count']} initializers -> "
            f"count delta={d['initializer_count'] - e['initializer_count']}"
        )
        common = set(e["initializer_names"]) & set(d["initializer_names"])
        print(f"{stage:>8}: same names={len(common)}/{len(e['initializer_names'])}, "
              f"shifted/removed={len(set(e['initializer_names']) ^ set(d['initializer_names']))}")
        topology_same = (
            e["topology"] == d["topology"]
            and e["inputs"] == d["inputs"]
            and e["outputs"] == d["outputs"]
        )
        print(f"{stage:>8}: graph topology identical = {topology_same}")

    out_dir = Path("repro_output")
    out_dir.mkdir(exist_ok=True)
    for r in results:
        tag = "epochA" if "EQUAL" in r.label else "epochB"
        for stage, proto in (
            ("raw", r.raw),
            ("optimized", r.optimized),
            ("slimmed", r.slimmed),
        ):
            onnx.save(proto, out_dir / f"weights_{tag}_{stage}.onnx")
    print(f"\nmodels saved under {out_dir}/")


if __name__ == "__main__":
    sys.exit(main())
