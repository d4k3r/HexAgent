"""KataHex/local-board synchronization and University pie-rule metadata."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .board import BLACK, BOARD_AREA, BOARD_SIZE, WHITE, HexBoard, action_to_gtp, opponent
from .gtp import CompletedAnalysis, GTPClient, MoveKind


class SynchronizationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EngineBoardSnapshot:
    cells: tuple[str | None, ...]
    side_to_move: str


_ROW = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")
_NEXT = re.compile(r"(?mi)^Next player:\s+(Black|White)\s*$")


def parse_showboard(payload: str) -> EngineBoardSnapshot:
    """Parse KataHex ``showboard``, including last-move numeric suffixes."""

    rows: dict[int, list[str | None]] = {}
    for line in payload.splitlines():
        match = _ROW.match(line)
        if match is None:
            continue
        row = int(match.group(1))
        if not 1 <= row <= BOARD_SIZE:
            continue
        # KataHex suppresses separators after history-marked cells, yielding
        # strings such as ``X X1X3X O``. Digits are annotations, so parse the
        # eleven actual board symbols independently of whitespace.
        tokens = re.findall(r"[XO.@]", match.group(2))
        if len(tokens) != BOARD_SIZE:
            continue
        parsed: list[str | None] = []
        for token in tokens:
            symbol = token
            if symbol == "X":
                parsed.append(BLACK)
            elif symbol == "O":
                parsed.append(WHITE)
            elif symbol in (".", "@"):
                parsed.append(None)
            else:
                raise SynchronizationError(f"unexpected showboard cell token {token!r}")
        rows[row - 1] = parsed
    if len(rows) != BOARD_SIZE:
        raise SynchronizationError("showboard response did not contain 11 parseable rows")
    next_match = _NEXT.search(payload)
    if next_match is None:
        raise SynchronizationError("showboard response omitted next player")
    side = BLACK if next_match.group(1).lower() == BLACK else WHITE
    cells = tuple(cell for row in range(BOARD_SIZE) for cell in rows[row])
    if len(cells) != BOARD_AREA:
        raise SynchronizationError("showboard response had wrong board area")
    return EngineBoardSnapshot(cells, side)


@dataclass(slots=True)
class UniversityOpening:
    """Controller-to-colour state for the University's pie rule.

    A swap changes player ownership only. The first physical black/red stone
    remains in place, no coordinate is transposed, and the board remains white
    to move. This mirrors ``HexAgent/src/Game.py`` at the preserved framework
    commit recorded in the provenance manifest.
    """

    first_player: str = "player1"
    second_player: str = "player2"
    swap_applied: bool = False

    @property
    def controller_to_color(self) -> dict[str, str]:
        if self.swap_applied:
            return {self.first_player: WHITE, self.second_player: BLACK}
        return {self.first_player: BLACK, self.second_player: WHITE}

    def apply_swap(self, board: HexBoard) -> None:
        if self.swap_applied:
            raise ValueError("swap has already been applied")
        if board.ply != 1 or board.side_to_move != WHITE:
            raise ValueError("University swap is legal only after the first black move")
        self.swap_applied = True


class SynchronizedKataHexSession:
    """Own one mirrored local board and one KataHex GTP board."""

    def __init__(self, client: GTPClient) -> None:
        self.client = client
        self.board = HexBoard()
        self.history: list[tuple[str, int]] = []
        self.initialized = False

    def initialize(self) -> None:
        self.client.command(f"boardsize {BOARD_SIZE}")
        self.client.command("clear_board")
        self.board = HexBoard()
        self.history.clear()
        self.initialized = True
        self.assert_synchronized()

    def play_physical(self, action: int) -> None:
        self._require_initialized()
        if not self.board.is_legal(action):
            raise SynchronizationError(f"local action is illegal: {action}")
        player = self.board.side_to_move
        self.client.command(f"play {player} {action_to_gtp(action)}")
        self.board.play(action)
        self.history.append((player, action))
        self.assert_synchronized()

    def accept_engine_played(self, analysis: CompletedAnalysis) -> int:
        """Mirror the final ``play`` from completed genmove analysis locally."""

        self._require_initialized()
        if analysis.chosen_move_kind is not MoveKind.PHYSICAL:
            raise SynchronizationError(
                f"engine played non-physical move {analysis.chosen_move!r}"
            )
        from .board import gtp_to_action

        action = gtp_to_action(analysis.chosen_move)
        if not self.board.is_legal(action):
            raise SynchronizationError(f"engine played locally illegal action {analysis.chosen_move}")
        player = self.board.side_to_move
        self.board.play(action)
        self.history.append((player, action))
        self.assert_synchronized()
        return action

    def accept_or_replace_engine_played(
        self,
        analysis: CompletedAnalysis,
        selected_action: int,
    ) -> bool:
        """Accept KataHex's auto-play or atomically replace it through GTP.

        ``kata-genmove_analyze`` always plays its own final selection.  A
        controller that deliberately selects another physical root action
        must undo that engine move before replaying the controller's action.
        The local board remains at the pre-search state until this method
        succeeds.  The return value reports whether replacement was needed.
        """

        self._require_initialized()
        if analysis.chosen_move_kind is not MoveKind.PHYSICAL:
            raise SynchronizationError(
                f"cannot replace non-physical engine move {analysis.chosen_move!r}"
            )
        from .board import gtp_to_action

        engine_action = gtp_to_action(analysis.chosen_move)
        if not self.board.is_legal(selected_action):
            raise SynchronizationError(f"controller-selected action is illegal: {selected_action}")
        if selected_action == engine_action:
            self.accept_engine_played(analysis)
            return False

        player = self.board.side_to_move
        self.client.command("undo")
        self.client.command(f"play {player} {action_to_gtp(selected_action)}")
        self.board.play(selected_action)
        self.history.append((player, selected_action))
        self.assert_synchronized()
        return True

    def accept_engine_action(self, action: int) -> None:
        """Mirror an already-played engine action selected by a strategy."""

        self._require_initialized()
        if not self.board.is_legal(action):
            raise SynchronizationError(f"engine action is locally illegal: {action}")
        player = self.board.side_to_move
        self.board.play(action)
        self.history.append((player, action))
        self.assert_synchronized()

    def reset_and_replay(self, moves: Iterable[tuple[str, int]]) -> None:
        self._require_initialized()
        replay = list(moves)
        self.client.command("clear_board")
        self.board = HexBoard()
        self.history.clear()
        for color, action in replay:
            if color != self.board.side_to_move:
                raise SynchronizationError("replay sequence is not alternating")
            if not self.board.is_legal(action):
                raise SynchronizationError("replay contains illegal physical action")
            self.client.command(f"play {color} {action_to_gtp(action)}")
            self.board.play(action)
            self.history.append((color, action))
        self.assert_synchronized()

    def undo(self) -> tuple[str, int]:
        self._require_initialized()
        if not self.history:
            raise SynchronizationError("cannot undo an empty history")
        self.client.command("undo")
        removed = self.history.pop()
        rebuilt = HexBoard()
        for color, action in self.history:
            if color != rebuilt.side_to_move:
                raise SynchronizationError("stored history is not alternating")
            rebuilt.play(action)
        self.board = rebuilt
        self.assert_synchronized()
        return removed

    def replace_client_and_replay(self, replacement: GTPClient) -> None:
        """Recover after a clean restart/crash using authoritative local history."""

        replay = tuple(self.history)
        self.client = replacement
        self.initialized = False
        self.initialize()
        self.reset_and_replay(replay)

    def assert_synchronized(self) -> EngineBoardSnapshot:
        self._require_initialized()
        snapshot = parse_showboard(self.client.command("showboard").payload)
        if snapshot.cells != tuple(self.board.cells):
            raise SynchronizationError("KataHex/local board cell divergence")
        if snapshot.side_to_move != self.board.side_to_move:
            raise SynchronizationError(
                "KataHex/local side-to-move divergence: "
                f"engine={snapshot.side_to_move} local={self.board.side_to_move}"
            )
        return snapshot

    def _require_initialized(self) -> None:
        if not self.initialized:
            raise SynchronizationError("session is not initialized")
