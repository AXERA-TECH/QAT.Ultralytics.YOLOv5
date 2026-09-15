"""Numeric consistency check between PyTorch and ONNX Runtime inference.

Runs the same fixed inputs through
  * a PyTorch checkpoint (``.pt``, Detect layer kept in train mode so it emits
    raw per-scale conv outputs like the QAT training graph, BN in eval mode),
  * an ONNX model in ONNX Runtime with ALL graph optimizations disabled
    (``GraphOptimizationLevel.ORT_DISABLE_ALL``),
then decodes both raw outputs with the shared ``decode_bbox_hardcode`` helper
and reports element-wise differences.

Usage:
    python compare_pt_ort.py --pt yolov5s.pt --onnx yolov5s.onnx
    python compare_pt_ort.py --pt yolov5s.pt --onnx yolov5s_qat.onnx --atol 5e-2 --rtol 5e-2
"""

import argparse

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt", required=True, help="PyTorch checkpoint path")
    parser.add_argument("--onnx", required=True, help="ONNX model path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--num", type=int, default=4, help="number of random inputs")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--qat", action="store_true",
                        help="--pt is a QAT state_dict (prepared FX graph): rebuild the "
                             "prepare_qat_pt2e model from models/yolov5s.yaml + config.json "
                             "and load it, with observers disabled to match export semantics")
    opt = parser.parse_args()

    device = torch.device(opt.device)
    rng = np.random.default_rng(opt.seed)

    from models.common import decode_bbox_hardcode

    if opt.qat:
        from copy import deepcopy

        from models.yolo import Model
        from torch.ao.quantization import disable_observer
        from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e

        from utils.ax_quantizer_lsq import AXQuantizer, load_config

        base = Model("models/yolov5s.yaml", ch=3, nc=80).to(device)
        dynamic_shapes = {"x": {0: torch.export.Dim.AUTO, 2: torch.export.Dim.AUTO, 3: torch.export.Dim.AUTO}}
        ep = torch.export.export_for_training(
            deepcopy(base), (torch.rand(2, 3, opt.imgsz, opt.imgsz, device=device),),
            dynamic_shapes=dynamic_shapes)
        global_config, regional_configs = load_config("./config.json")
        quantizer = AXQuantizer()
        quantizer.set_global(global_config)
        quantizer.set_regional(regional_configs)
        pt_model = prepare_qat_pt2e(ep.module(), quantizer)
        torch.ao.quantization.move_exported_model_to_eval(pt_model)
        torch.ao.quantization.allow_exported_model_train_eval(pt_model)
        state = torch.load(opt.pt, map_location=device, weights_only=False)
        state = state.get("ema") or state.get("model") or state
        pt_model.load_state_dict(state)
        pt_model.apply(disable_observer)  # match convert_pt2e export semantics
        pt_model.eval()
    else:
        from models.experimental import attempt_load

        pt_model = attempt_load(opt.pt, device=device, fuse=False)
        pt_model.eval()

    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(opt.onnx, session_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Training-graph exports emit raw per-scale conv outputs (ndim==5 each);
    # inference exports emit a single decoded tensor. Match the PT side to it.
    probe = rng.random((1, 3, opt.imgsz, opt.imgsz), dtype=np.float32)
    probe_out = session.run(None, {input_name: probe})
    raw_outputs = all(o.ndim == 5 for o in probe_out)
    if raw_outputs and not opt.qat:
        pt_model.model[-1].train()  # Detect emits raw per-scale outputs

    def decode(pt_out, ort_out):
        if raw_outputs:
            pt_dec = decode_bbox_hardcode(pt_out, device=device)
            ort_dec = decode_bbox_hardcode(ort_out, device=device)
            return pt_dec, ort_dec
        pt_dec = pt_out[0] if isinstance(pt_out, (list, tuple)) else pt_out
        return pt_dec, torch.from_numpy(ort_out[0]).to(device)

    ok, max_abs = True, 0.0
    for k in range(opt.num):
        x = rng.random((opt.batch, 3, opt.imgsz, opt.imgsz), dtype=np.float32)
        with torch.no_grad():
            pt_out = pt_model(torch.from_numpy(x).to(device))
        ort_out = session.run(None, {input_name: x})
        pt_dec, ort_dec = decode(pt_out, ort_out)
        diff = float((pt_dec - ort_dec).abs().max())
        max_abs = max(max_abs, diff)
        close = torch.allclose(pt_dec, ort_dec, atol=opt.atol, rtol=opt.rtol)
        ok &= close
        print(f"[{k}] max_abs_diff={diff:.3g} allclose(atol={opt.atol}, rtol={opt.rtol})={close}")

    print(f"overall max_abs_diff={max_abs:.3g} {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
