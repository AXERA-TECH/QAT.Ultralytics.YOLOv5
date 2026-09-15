"""Compare QAT ONNX exports across epochs / runs.

Checks the three symptoms from the issue:
    1. initializer count consistency
    2. initializer name consistency (val_N numbering)
    3. Q/DQ shared-reference consistency
plus graph topology equality.
"""

from __future__ import annotations

import glob
import os
import sys

import onnx


def summarize(path: str) -> dict[str, object]:
    m = onnx.load(path)
    names = [i.name for i in m.graph.initializer]
    qdq = [
        (n.op_type, list(n.input))
        for n in m.graph.node
        if n.op_type in ("QuantizeLinear", "DequantizeLinear")
    ]
    return {
        "path": path,
        "count": len(names),
        "names": names,
        "qdq": qdq,
        "topology": [n.op_type for n in m.graph.node],
    }


def compare(paths: list[str]) -> int:
    rows = [summarize(p) for p in paths]
    base = rows[0]
    print(f"{'file':<72} {'init':>4}  names==  topo==")
    for r in rows:
        print(
            f"{os.path.basename(r['path']):<72} {r['count']:>4}  "
            f"{str(r['names'] == base['names']):>7}  {str(r['topology'] == base['topology']):>7}"
        )

    counts = {r["count"] for r in rows}
    names_ok = all(r["names"] == base["names"] for r in rows)
    topo_ok = all(r["topology"] == base["topology"] for r in rows)
    qdq_ok = all(r["qdq"] == base["qdq"] for r in rows)

    print(
        f"\ninitializer counts: {sorted(counts)} -> {'CONSISTENT' if len(counts) == 1 else 'INCONSISTENT'}"
    )
    print(f"initializer names identical: {names_ok}")
    print(f"graph topology identical:    {topo_ok}")
    print(f"Q/DQ references identical:   {qdq_ok}")

    if not names_ok:
        print("\nname differences vs first file:")
        base_names = set(base["names"])
        for r in rows[1:]:
            diff = set(r["names"]) ^ base_names
            if diff:
                print(f"  {os.path.basename(r['path'])}: {sorted(diff)}")
    return 0 if (len(counts) == 1 and names_ok and topo_ok and qdq_ok) else 1


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        print("usage: python compare_qat_onnx.py <a.onnx> <b.onnx> ...")
        sys.exit(2)
    sys.exit(compare(paths))
