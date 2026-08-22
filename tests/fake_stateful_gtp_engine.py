#!/usr/bin/env python3
"""Stateful 11x11 fake KataHex used for synchronization contract tests."""

from __future__ import annotations

import os
import sys


SIZE = 11
board = [None] * (SIZE * SIZE)
history: list[tuple[str, int]] = []
next_player = "black"


def opponent(player: str) -> str:
    return "white" if player == "black" else "black"


def parse(move: str) -> int:
    return (int(move[1:]) - 1) * SIZE + ord(move[0].lower()) - ord("a")


def coord(action: int) -> str:
    return f"{chr(ord('a') + action % SIZE)}{action // SIZE + 1}"


def respond(command_id: str, payload: str = "", *, success: bool = True) -> None:
    marker = "=" if success else "?"
    suffix = f" {payload}" if payload else ""
    sys.stdout.write(f"{marker}{command_id}{suffix}\n\n")
    sys.stdout.flush()


def showboard() -> str:
    lines = ["MoveNum: %d HASH: fake" % len(history), "  " + " ".join(chr(65 + i) for i in range(SIZE))]
    for row in range(SIZE):
        cells = []
        for column in range(SIZE):
            value = board[row * SIZE + column]
            cells.append("X" if value == "black" else "O" if value == "white" else ".")
        lines.append(" " * row + f"{row + 1:2d} " + " ".join(cells))
    lines.append(f"Next player: {next_player.title()}")
    lines.append('Rules: {"koRule":"SIMPLE"}')
    return "\n".join(lines)


for wire in sys.stdin:
    command_id, command = wire.strip().split(" ", 1)
    pieces = command.split()
    verb = pieces[0]
    if verb == "boardsize":
        respond(command_id)
    elif verb == "clear_board":
        board[:] = [None] * (SIZE * SIZE)
        history.clear()
        next_player = "black"
        respond(command_id)
    elif verb == "play":
        player, move = pieces[1], pieces[2].lower()
        action = parse(move)
        if player != next_player or board[action] is not None:
            respond(command_id, "illegal move", success=False)
            continue
        board[action] = player
        history.append((player, action))
        next_player = opponent(player)
        respond(command_id)
    elif verb == "showboard":
        respond(command_id, showboard())
    elif verb == "undo":
        if not history:
            respond(command_id, "cannot undo", success=False)
            continue
        player, action = history.pop()
        board[action] = None
        next_player = player
        respond(command_id)
    elif verb == "kata-genmove_analyze":
        player = pieces[1]
        has_avoid = "avoid" in pieces
        if player != next_player:
            respond(command_id, "wrong player", success=False)
            continue
        physical = next(action for action, value in enumerate(board) if value is None)
        move = coord(physical)
        if not has_avoid:
            respond(
                command_id,
                "info move pass visits 3 utility 1 winrate 1 prior 0.8 lcb 0.9 utilityLcb 0.8 pv pass\nplay pass",
            )
            continue
        board[physical] = player
        history.append((player, physical))
        next_player = opponent(player)
        respond(
            command_id,
            "info move pass visits 0 utility 1 winrate 1 prior 0.8 lcb 0.9 utilityLcb 0.8 pv pass "
            f"info move {move} visits 4 utility 0.9 winrate 0.95 prior 0.1 lcb 0.7 utilityLcb 0.6 pv {move}\n"
            f"play {move}",
        )
    elif verb == "error":
        respond(command_id, "deliberate error", success=False)
    elif verb == "crash":
        os._exit(23)
    elif verb == "quit":
        respond(command_id)
        break
    else:
        respond(command_id)
