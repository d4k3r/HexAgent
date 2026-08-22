#!/usr/bin/env python3
"""Estimate the frozen PUCT/FPU grid runtime from shared-backend calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SIMULATIONS = {128: 3_145_728, 512: 12_582_912, 2048: 50_331_648}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs", type=int, default=12,
                        help="number of c_puct/FPU configs at each budget")
    parser.add_argument("--setup-seconds-per-config", type=float, default=0.0,
                        help="optional measured model/session/IO overhead")
    args = parser.parse_args()
    if args.configs <= 0 or args.setup_seconds_per_config < 0:
        raise SystemExit("invalid estimator parameters")

    rates = {}
    calibration = {}
    for budget in (128, 512, 2048):
        path = args.calibration_root / f"c1.5-zero-r0-v{budget}-32pos" / "telemetry.json"
        if not path.exists():
            raise SystemExit(f"missing calibration telemetry: {path}")
        item = json.loads(path.read_text())
        rate = float(item.get("simulations_per_second", 0.0))
        if rate <= 0:
            raise SystemExit(f"invalid calibration rate in {path}")
        rates[budget] = rate
        calibration[budget] = {"path": str(path), "telemetry": item}

    components = {
        str(budget): {
            "requested_simulations": simulations,
            "measured_simulations_per_second": rates[budget],
            "core_seconds": simulations / rates[budget],
            "configs": args.configs,
        }
        for budget, simulations in SIMULATIONS.items()
    }
    core_seconds = sum(item["core_seconds"] for item in components.values())
    setup_seconds = args.configs * 3 * args.setup_seconds_per_config
    expected = core_seconds + setup_seconds
    output = {
        "schema": "hex-puct-fpu-runtime-estimate-v1",
        "calibration_root": str(args.calibration_root.resolve()),
        "configs_per_budget": args.configs,
        "requested_simulations": sum(SIMULATIONS.values()),
        "calibration": calibration,
        "components": components,
        "setup_seconds_per_config": args.setup_seconds_per_config,
        "setup_seconds_total": setup_seconds,
        "core_seconds": core_seconds,
        "expected_seconds": expected,
        "expected_hours": expected / 3600.0,
        "range_seconds": {"low": expected * 0.9, "high": expected * 1.15},
        "range_hours": {"low": expected * 0.9 / 3600.0, "high": expected * 1.15 / 3600.0},
        "note": "Range is an operational estimate, not a strength or correctness result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
