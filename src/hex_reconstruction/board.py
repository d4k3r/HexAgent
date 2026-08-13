"""11x11 physical Hex board with literal and KataHex virtual terminals.

Black/red connects top to bottom. White/blue connects left to right. The
virtual detector is a compact behavior-preserving port of
selinger/katahex@41a65784:cpp/game/gamelogic.cpp, recorded in the provenance
manifest. It is intentionally separate from the literal DSU.
"""

from __future__ import annotations

import heapq
from typing import Iterable


BOARD_SIZE = 11
BOARD_AREA = BOARD_SIZE * BOARD_SIZE
BLACK = "black"
WHITE = "white"
COLORS = (BLACK, WHITE)
V_START = BOARD_AREA
V_END = BOARD_AREA + 1
NEIGHBORS = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))
DIRECT_DX = (0, 1, 1, 0, -1, -1, 0)
DIRECT_DY = (-1, -1, 0, 1, 1, 0, -1)
JUMP_DX = (1, 2, 1, -1, -2, -1)
JUMP_DY = (-2, -1, 1, 2, 1, -1)


def opponent(player: str) -> str:
    if player == BLACK:
        return WHITE
    if player == WHITE:
        return BLACK
    raise ValueError(f"invalid player: {player}")


def action_to_gtp(action: int) -> str:
    if not 0 <= action < BOARD_AREA:
        raise ValueError("physical action must be in [0,120]")
    row, column = divmod(action, BOARD_SIZE)
    return f"{chr(ord('a') + column)}{row + 1}"


def gtp_to_action(move: str) -> int:
    normalized = move.strip().lower()
    if normalized in ("pass", "resign", "swap"):
        raise ValueError(f"{normalized} is game control, not a physical action")
    if len(normalized) < 2 or not ("a" <= normalized[0] <= "k"):
        raise ValueError(f"invalid 11x11 coordinate: {move}")
    try:
        row = int(normalized[1:]) - 1
    except ValueError as error:
        raise ValueError(f"invalid 11x11 coordinate: {move}") from error
    column = ord(normalized[0]) - ord("a")
    if not 0 <= row < BOARD_SIZE:
        raise ValueError(f"invalid 11x11 coordinate: {move}")
    return row * BOARD_SIZE + column


class DSU:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def unite(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1

    def connected(self, left: int, right: int) -> bool:
        return self.find(left) == self.find(right)


class HexBoard:
    cells: list[str | None]
    side_to_move: str = BLACK
    last_move: int | None = None
    ply: int = 0

    def __init__(self, side_to_move: str = BLACK) -> None:
        if side_to_move not in COLORS:
            raise ValueError("side_to_move must be black or white")
        self.cells = [None] * BOARD_AREA
        self.side_to_move = side_to_move
        self.last_move = None
        self.ply = 0
        self._dsu = {BLACK: DSU(BOARD_AREA + 2), WHITE: DSU(BOARD_AREA + 2)}

    def copy(self) -> "HexBoard":
        clone = HexBoard(self.side_to_move)
        for action, color in enumerate(self.cells):
            if color is not None:
                clone._place(action, color)
        clone.last_move = self.last_move
        clone.ply = self.ply
        return clone

    @classmethod
    def from_moves(cls, moves: Iterable[tuple[str, str]]) -> "HexBoard":
        board = cls()
        for color, move in moves:
            if color != board.side_to_move:
                raise ValueError(f"fixture is not alternating at {color} {move}")
            board.play(gtp_to_action(move))
        return board

    @classmethod
    def from_setup(
        cls,
        black: Iterable[int] = (),
        white: Iterable[int] = (),
        *,
        side_to_move: str = BLACK,
        last_move: int | None = None,
    ) -> "HexBoard":
        board = cls(side_to_move)
        for action in black:
            board._place(action, BLACK)
        for action in white:
            board._place(action, WHITE)
        board.ply = sum(cell is not None for cell in board.cells)
        board.last_move = last_move
        return board

    @staticmethod
    def _coordinates(action: int) -> tuple[int, int]:
        return action % BOARD_SIZE, action // BOARD_SIZE

    @staticmethod
    def _action(x: int, y: int) -> int:
        return y * BOARD_SIZE + x

    def _place(self, action: int, color: str) -> None:
        if color not in COLORS:
            raise ValueError("invalid color")
        if not 0 <= action < BOARD_AREA:
            raise ValueError("action outside board")
        if self.cells[action] is not None:
            raise ValueError(f"occupied action: {action}")
        self.cells[action] = color
        x, y = self._coordinates(action)
        dsu = self._dsu[color]
        if color == BLACK:
            if y == 0:
                dsu.unite(action, V_START)
            if y == BOARD_SIZE - 1:
                dsu.unite(action, V_END)
        else:
            if x == 0:
                dsu.unite(action, V_START)
            if x == BOARD_SIZE - 1:
                dsu.unite(action, V_END)
        for dx, dy in NEIGHBORS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                neighbor = self._action(nx, ny)
                if self.cells[neighbor] == color:
                    dsu.unite(action, neighbor)

    def play(self, action: int) -> None:
        if self.literal_winner() is not None:
            raise ValueError("cannot play after literal terminal")
        color = self.side_to_move
        self._place(action, color)
        self.last_move = action
        self.ply += 1
        self.side_to_move = opponent(color)

    def is_legal(self, action: int) -> bool:
        return 0 <= action < BOARD_AREA and self.cells[action] is None and self.literal_winner() is None

    def legal_actions(self) -> list[int]:
        if self.literal_winner() is not None:
            return []
        return [action for action, cell in enumerate(self.cells) if cell is None]

    def legal_mask(self) -> list[bool]:
        terminal = self.literal_winner() is not None
        return [not terminal and cell is None for cell in self.cells]

    def literal_winner(self) -> str | None:
        for color in COLORS:
            if self._dsu[color].connected(V_START, V_END):
                return color
        return None

    def virtual_winner(self) -> str | None:
        for color in COLORS:
            if self._check_connection(color, include_jump=True):
                return color
        return None

    def _check_connection(self, player: str, *, include_jump: bool) -> bool:
        # KataHex always searches the y axis and transposes white.
        if player == BLACK:
            buffer = [
                [1 if self.cells[self._action(x, y)] == player else 2 if self.cells[self._action(x, y)] else 0 for x in range(BOARD_SIZE)]
                for y in range(BOARD_SIZE)
            ]
        else:
            buffer = [
                [1 if self.cells[self._action(y, x)] == player else 2 if self.cells[self._action(y, x)] else 0 for x in range(BOARD_SIZE)]
                for y in range(BOARD_SIZE)
            ]

        def get(x: int, y: int) -> int:
            if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
                return 2
            return buffer[y][x]

        def visit(x0: int, y0: int) -> bool:
            buffer[y0][x0] = 3
            for direction in range(6):
                x = x0 + DIRECT_DX[direction]
                y = y0 + DIRECT_DY[direction]
                color = get(x, y)
                if color == 4:
                    return True
                if color == 1 and visit(x, y):
                    return True
            if include_jump:
                for direction in range(6):
                    x = x0 + JUMP_DX[direction]
                    y = y0 + JUMP_DY[direction]
                    color = get(x, y)
                    gap1 = (x0 + DIRECT_DX[direction], y0 + DIRECT_DY[direction])
                    gap2 = (x0 + DIRECT_DX[direction + 1], y0 + DIRECT_DY[direction + 1])
                    if color == 4:
                        if get(*gap1) == 0 and get(*gap2) == 0:
                            return True
                    elif color == 1 and get(*gap1) == 0 and get(*gap2) == 0:
                        buffer[gap1[1]][gap1[0]] = 2
                        buffer[gap2[1]][gap2[0]] = 2
                        if visit(x, y):
                            return True
            return False

        for x in range(BOARD_SIZE):
            if buffer[BOARD_SIZE - 1][x] == 1:
                buffer[BOARD_SIZE - 1][x] = 4
        if include_jump:
            for x in range(1, BOARD_SIZE):
                if buffer[BOARD_SIZE - 2][x] == 1 and buffer[BOARD_SIZE - 1][x] == 0 and buffer[BOARD_SIZE - 1][x - 1] == 0:
                    buffer[BOARD_SIZE - 2][x] = 4

        for x in range(BOARD_SIZE):
            if buffer[0][x] == 1 and visit(x, 0):
                return True
        if include_jump:
            for x in range(BOARD_SIZE - 1):
                if buffer[1][x] == 1 and buffer[0][x] == 0 and buffer[0][x + 1] == 0 and visit(x, 1):
                    return True
        return False

    def feature_planes(self) -> list[list[int]]:
        red = [int(cell == BLACK) for cell in self.cells]
        blue = [int(cell == WHITE) for cell in self.cells]
        turn = [int(self.side_to_move == BLACK)] * BOARD_AREA
        last_move = [int(action == self.last_move) for action in range(BOARD_AREA)]
        conn_start: list[int] = []
        conn_end: list[int] = []
        for action, color in enumerate(self.cells):
            if color is None:
                conn_start.append(0)
                conn_end.append(0)
            else:
                dsu = self._dsu[color]
                conn_start.append(int(dsu.connected(action, V_START)))
                conn_end.append(int(dsu.connected(action, V_END)))
        return [red, blue, turn, last_move, conn_start, conn_end]

    def shortest_connection_path(self, player: str) -> list[int]:
        """Return a deterministic minimum-empty-cell physical connection path."""

        if player not in COLORS:
            raise ValueError("invalid player")
        blocked = opponent(player)
        starts = [self._action(x, 0) for x in range(BOARD_SIZE)] if player == BLACK else [self._action(0, y) for y in range(BOARD_SIZE)]

        distances = [10**9] * BOARD_AREA
        previous: list[int | None] = [None] * BOARD_AREA
        heap: list[tuple[int, int]] = []
        for action in starts:
            if self.cells[action] == blocked:
                continue
            cost = 0 if self.cells[action] == player else 1
            distances[action] = cost
            heapq.heappush(heap, (cost, action))

        target: int | None = None
        while heap:
            distance, action = heapq.heappop(heap)
            if distance != distances[action]:
                continue
            x, y = self._coordinates(action)
            if (player == BLACK and y == BOARD_SIZE - 1) or (player == WHITE and x == BOARD_SIZE - 1):
                target = action
                break
            for dx, dy in NEIGHBORS:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE):
                    continue
                neighbor = self._action(nx, ny)
                if self.cells[neighbor] == blocked:
                    continue
                new_distance = distance + (0 if self.cells[neighbor] == player else 1)
                # Strict improvement is essential. Re-parenting equal-cost
                # zero-weight stone edges can make two predecessors point at
                # each other and create a cycle during path reconstruction.
                # Heap ordering by action already makes the chosen path
                # deterministic.
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = action
                    heapq.heappush(heap, (new_distance, neighbor))

        if target is None:
            return []
        path: list[int] = []
        seen: set[int] = set()
        cursor: int | None = target
        while cursor is not None:
            if cursor in seen:
                raise RuntimeError("cycle in shortest-path predecessor chain")
            seen.add(cursor)
            path.append(cursor)
            cursor = previous[cursor]
        path.reverse()
        return path
