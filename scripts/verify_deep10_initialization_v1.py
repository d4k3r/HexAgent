#!/usr/bin/env python3
"""Verify DEEP10, CONTROL, and Champion-2 initial tensors are identical."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from verify_control_deep5_initialization_v1 import fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--control-initial", type=Path, required=True)
    parser.add_argument("--deep10-initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = {"parent": fingerprint(args.parent), "control_initial": fingerprint(args.control_initial), "deep10_initial": fingerprint(args.deep10_initial)}
    result = {"schema": "deep10-initialization-audit-v1", "fingerprints": values, "passed": len(set(values.values())) == 1}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(partial, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
