#!/usr/bin/env python3
"""Frozen cross-language golden harness for Stage-1 deterministic PUCT."""
from __future__ import annotations
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hex_reconstruction.board import BLACK, WHITE, BOARD_AREA, HexBoard
from hex_reconstruction.puct import DeterministicPUCT, Evaluation, SearchConfig

BUDGETS = (1, 4, 8, 32, 128)
BANK = (
    ("empty_black", [], [], BLACK),
    ("early_white", [0, 12], [1, 13], WHITE),
    ("mid_black", [0,12,24,36,48,60,72,84,96,108], [1,13,25,37,49,61,73,85,97], BLACK),
    ("late_white", [2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92], [1,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105], WHITE),
    ("forced_literal_win", [0,11,22,33,44,55,66,77,88,99], [1,12,23,34,45,56,67,78,89,100], BLACK),
    ("transpose_black", [0,12,25,37], [1,13,24,36], BLACK),
    ("colour_transpose_white", [11,13,24,36], [0,12,35,47], WHITE),
)

def signature(board: HexBoard) -> int:
    value = 17 if board.side_to_move == BLACK else 29
    for action, cell in enumerate(board.cells):
        value = (value + (action + 1) * (1 if cell == BLACK else 2 if cell == WHITE else 0)) % 1_000_003
    return value

class FakeEvaluator:
    def __init__(self) -> None: self.trace: list[int] = []
    def evaluate(self, board: HexBoard) -> Evaluation:
        state = signature(board); self.trace.append(state)
        return Evaluation([((action * 17 + state * 31) % 37 - 18) / 8.0 for action in range(BOARD_AREA)], ((state % 13) - 6) / 7.0)

def close(actual: float | None, expected: float | None, label: str) -> None:
    if actual is None or expected is None:
        assert actual is expected, label; return
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=2e-14), f"{label}: {actual} != {expected}"

def main() -> None:
    executable = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build" / "cpp-puct-stage1" / "hex_puct_parity_runner"
    actual = json.loads(subprocess.check_output([str(executable)], text=True))["cases"]
    expected = []
    for name, black, white, side in BANK:
        board = HexBoard.from_setup(black=black, white=white, side_to_move=side)
        for budget in BUDGETS:
            evaluator = FakeEvaluator(); result = DeterministicPUCT(evaluator, SearchConfig(budget)).search(board)
            q = [value / visits if visits else 0.0 for value, visits in zip(result.raw_value_sums, result.raw_visits)]
            expected.append({"name": name, "budget": budget, "legal_mask": [int(x) for x in board.legal_mask()], "trace": evaluator.trace, "terminal_winner": board.literal_winner(), "selected_action": result.selected_action, "root_visits": result.root_visits, "raw_visits": list(result.raw_visits), "priors": list(result.priors), "raw_value_sums": list(result.raw_value_sums), "q": q, "root_value": result.root_value})
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got["name"] == want["name"] and got["budget"] == want["budget"]
        for key in ("legal_mask", "trace", "terminal_winner", "selected_action", "root_visits", "raw_visits"):
            assert got[key] == want[key], f"{want['name']} budget {want['budget']} {key} mismatch"
        for key in ("priors", "raw_value_sums", "q"):
            for i, (a, b) in enumerate(zip(got[key], want[key])): close(a, b, f"{want['name']} {want['budget']} {key}[{i}]")
        close(got["root_value"], want["root_value"], f"{want['name']} {want['budget']} root_value")
    print(f"C++/Python deterministic PUCT parity passed: {len(expected)} cases ({len(BANK)} positions x {len(BUDGETS)} budgets).")

if __name__ == "__main__": main()
