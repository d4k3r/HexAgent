#!/usr/bin/env python3
"""Bit-exact parent/initial-tensor audit for one autonomous generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def fingerprint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state") or payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"model state is absent: {path}")
    h = hashlib.sha256(); tensors = 0; parameters = 0
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous(); raw = value.numpy().tobytes(order="C")
        h.update(json.dumps({"name": name, "shape": list(value.shape), "dtype": str(value.dtype), "nbytes": len(raw)},
                            sort_keys=True, separators=(",", ":")).encode())
        h.update(b"\0"); h.update(raw); tensors += 1; parameters += value.numel()
    return {"tensor_fingerprint": h.hexdigest(), "tensors": tensors, "parameters": parameters}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--initial", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = {"parent": fingerprint(args.parent.resolve())}
    for path in args.initial:
        item = path.resolve(); values[str(item)] = fingerprint(item)
    all_fingerprints = {item["tensor_fingerprint"] for item in values.values()}
    shapes = {(item["tensors"], item["parameters"]) for item in values.values()}
    report = {"schema": "autonomous-candidate-initialization-audit-v1", "fingerprints": values,
              "passed": len(all_fingerprints) == 1 and len(shapes) == 1}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
