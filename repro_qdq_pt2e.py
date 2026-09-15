"""Faithful QAT-style reproduction using the real pt2e pipeline.

Mirrors train_qat_gpus.py:
    prepare_pt2e -> convert_pt2e -> torch.onnx.export(..., dynamo=True)
    onnx_program.optimize()  (runs DeduplicateInitializersPass)
    onnxslim.slim(...)       (runs DeduplicateInitializersPass again)

Two QAT epochs are simulated by hand-setting the activation scales of two
Conv/Linear layers:
    * epoch A: layer1.scale == layer2.scale (0.5)  -> dedup merges the scales
    * epoch B: layer1.scale == 0.5, layer2.scale == 0.7 -> no merge
Everything else in the graph stays identical.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_qat_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)


@dataclasses.dataclass
class ExportResult:
    label: str
    raw: onnx.ModelProto
    optimized: onnx.ModelProto
    slimmed: onnx.ModelProto


class TwoConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 2, 1)
        self.conv2 = nn.Conv2d(2, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv1(x)) + torch.relu(self.conv2(x))


def build_quantized_model(same: bool) -> torch.export.ExportedProgram:
    model = TwoConvModel().eval()
    example = torch.randn(2, 2, 4, 4)
    ep = torch.export.export_for_training(model, (example,)).module()
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=False)
    )
    prepared = prepare_qat_pt2e(ep, quantizer)
    torch.ao.quantization.move_exported_model_to_eval(prepared)
    with torch.no_grad():
        prepared(example)

    # Simulate the "same scale at epoch A / different at epoch B" state.
    # Force the conv1/conv2 output-activation scales.
    # scale = max_val / 127 for this symmetric configuration, so
    #   same=True  -> both scales == 1/127          -> initializers merge
    #   same=False -> scales == 1/127 and 1.4/127  -> initializers stay
    prepared.activation_post_process_2.min_val = torch.tensor(0.0)
    prepared.activation_post_process_2.max_val = torch.tensor(1.0)
    prepared.activation_post_process_4.min_val = torch.tensor(0.0)
    prepared.activation_post_process_4.max_val = torch.tensor(1.0 if same else 1.4)
    quantized = convert_pt2e(prepared)
    return quantized


def export_variant(same: bool, label: str) -> ExportResult:
    ep = build_quantized_model(same)
    x = torch.randn(2, 2, 4, 4)
    onnx_program = torch.onnx.export(ep, (x,), dynamo=True)

    raw = onnx_program.model_proto
    onnx_program.optimize()
    optimized = onnx_program.model_proto

    from onnxslim import slim

    slimmed = slim(optimized)
    return ExportResult(label=label, raw=raw, optimized=optimized, slimmed=slimmed)


def summarize(model: onnx.ModelProto) -> dict[str, object]:
    names = [i.name for i in model.graph.initializer]
    qdq = [
        (n.op_type, list(n.input))
        for n in model.graph.node
        if n.op_type in ("QuantizeLinear", "DequantizeLinear")
    ]
    return {
        "count": len(names),
        "names": names,
        "qdq": qdq,
        "topology": [n.op_type for n in model.graph.node],
        "inputs": [i.name for i in model.graph.input],
        "outputs": [o.name for o in model.graph.output],
    }


def main() -> int:
    results = [
        export_variant(True, "epoch A: both activation scales == 1/127"),
        export_variant(False, "epoch B: activation scales == 1/127 and 1.4/127"),
    ]

    for r in results:
        print(f"\n=== {r.label} ===")
        for stage, proto in (
            ("raw dynamo export", r.raw),
            ("after optimize()", r.optimized),
            ("after slim()", r.slimmed),
        ):
            s = summarize(proto)
            print(f"[{stage}] initializers={s['count']}: {s['names']}")
            for op, inputs in s["qdq"]:
                print(f"    {op} -> {inputs}")

    print("\n=== cross-epoch comparison ===")
    a, b = results
    for stage in ("raw", "optimized", "slimmed"):
        sa, sb = summarize(getattr(a, stage)), summarize(getattr(b, stage))
        topology_same = (
            sa["topology"] == sb["topology"]
            and sa["inputs"] == sb["inputs"]
            and sa["outputs"] == sb["outputs"]
        )
        print(
            f"{stage:>8}: epochA={sa['count']} initializers, "
            f"epochB={sb['count']} initializers, delta={sb['count'] - sa['count']}"
        )
        print(f"{stage:>8}: name sets differ by {len(set(sa['names']) ^ set(sb['names']))} entries")
        print(f"{stage:>8}: graph topology identical = {topology_same}")
        print(f"         epochA names: {sa['names']}")
        print(f"         epochB names: {sb['names']}")

    out_dir = Path("repro_output")
    out_dir.mkdir(exist_ok=True)
    for r in results:
        tag = "epochA" if "epoch A" in r.label else "epochB"
        for stage, proto in (
            ("raw", r.raw),
            ("optimized", r.optimized),
            ("slimmed", r.slimmed),
        ):
            onnx.save(proto, out_dir / f"qdq_{tag}_{stage}.onnx")
    print(f"\nmodels saved under {out_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
