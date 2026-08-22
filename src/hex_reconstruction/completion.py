"""Pass-free physical completion strategy interface.

These moves are trajectory mechanics, not policy supervision. The included
strategy is intentionally a deterministic validation bootstrap, not a claim
that an arbitrary non-pass move preserves a KataHex virtual connection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import TYPE_CHECKING

from .board import HexBoard, gtp_to_action, opponent
from .gtp import AnalysisCandidate, CompletedAnalysis, GTPClient, MoveKind, parse_completed_analysis

if TYPE_CHECKING:
    from .sync import SynchronizedKataHexSession


KATAHEX_PASS_FORBIDDEN_VERSION = "katahex-pass-forbidden-completion-v1"


class CompletionKind(str, Enum):
    DETERMINISTIC_SHORTEST_PATH = "deterministic_shortest_path"
    GROUP49_PUCT = "group49_puct"
    KATAHEX_PASS_FORBIDDEN = "katahex_pass_forbidden"
    CLASSICAL_MCTS = "classical_mcts"


class CompletionStrategy(ABC):
    kind: CompletionKind

    @abstractmethod
    def choose_action(self, board: HexBoard, *, virtual_winner: str) -> int:
        """Return one legal physical action. Pass/resign/swap are impossible."""


@dataclass(frozen=True, slots=True)
class KataHexCompletionDiagnostic:
    side_to_move: str
    chosen_move: str
    candidates: tuple[AnalysisCandidate, ...]
    raw_response: str


class KataHexPassForbiddenCompletion(CompletionStrategy):
    """Completed KataHex analysis with pass prohibited at the root.

    ``kata-genmove_analyze`` plays the final choice on the engine board. The
    caller must apply the returned physical action to its local board exactly
    once and then perform a synchronization check.
    """

    kind = CompletionKind.KATAHEX_PASS_FORBIDDEN
    mechanism_version = KATAHEX_PASS_FORBIDDEN_VERSION

    def __init__(
        self,
        client: GTPClient,
        *,
        report_interval_centiseconds: int = 10000,
        timeout: float = 120.0,
        synchronized_session: "SynchronizedKataHexSession | None" = None,
    ) -> None:
        self.client = client
        self.report_interval_centiseconds = report_interval_centiseconds
        self.timeout = timeout
        if synchronized_session is not None and synchronized_session.client is not client:
            raise ValueError("synchronized session must own the supplied GTP client")
        self.synchronized_session = synchronized_session
        self.diagnostics: list[KataHexCompletionDiagnostic] = []
        self.engine_played_last_action = False

    def choose_action(self, board: HexBoard, *, virtual_winner: str) -> int:
        del virtual_winner
        player = board.side_to_move
        command = " ".join(
            (
                "kata-genmove_analyze",
                player,
                str(self.report_interval_centiseconds),
                "avoid",
                player,
                "pass",
                "1",
                "pvVisits",
                "true",
                "pvEdgeVisits",
                "true",
            )
        )
        analysis: CompletedAnalysis = parse_completed_analysis(
            self.client.command(command, timeout=self.timeout)
        )
        self.diagnostics.append(
            KataHexCompletionDiagnostic(
                side_to_move=player,
                chosen_move=analysis.chosen_move,
                candidates=analysis.candidates,
                raw_response=analysis.raw_response,
            )
        )
        self.engine_played_last_action = True
        if analysis.chosen_move_kind is not MoveKind.PHYSICAL:
            raise RuntimeError(
                "pass-forbidden KataHex returned game-control move "
                f"{analysis.chosen_move!r}"
            )
        action = gtp_to_action(analysis.chosen_move)
        if not board.is_legal(action):
            raise RuntimeError(f"pass-forbidden KataHex returned illegal move {analysis.chosen_move}")
        if not any(candidate.move == analysis.chosen_move for candidate in analysis.candidates):
            raise RuntimeError("KataHex final play is absent from completed candidate records")
        return action

    def apply_selected_action(self, board: HexBoard, action: int) -> None:
        if self.synchronized_session is None:
            board.play(action)
            return
        if board is not self.synchronized_session.board:
            raise RuntimeError("completion board is not the synchronized session board")
        self.synchronized_session.accept_engine_action(action)


class _XorShift64:
    """Group 49's historical xorshift sequence, including uint64 wrapping."""

    _MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = seed & self._MASK

    def next(self) -> int:
        value = self.state
        value ^= (value << 13) & self._MASK
        value ^= value >> 7
        value ^= (value << 17) & self._MASK
        self.state = value & self._MASK
        return self.state

    def range(self, maximum: int) -> int:
        if maximum <= 0:
            raise ValueError("maximum must be positive")
        return self.next() % maximum


@dataclass(slots=True)
class _MCTSEdge:
    move: int
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    child: "_MCTSNode | None" = None


@dataclass(slots=True)
class _MCTSNode:
    player_just_moved: str | None
    expanded: bool = False
    children: list[_MCTSEdge] = field(default_factory=list)


class ClassicalLiteralMCTSCompletion(CompletionStrategy):
    """Small behavioral port of Group 49's stable classical MCTS refactor.

    Provenance: ``HexAgent`` commit 3537835ac86b36e3a25ea53f70f5d66fcafc9c76,
    ``agents/Group49/src/MCTS.h`` blob d42800a0776d1e114f93a884801759607906961f.
    The original uses uniform PUCT priors and random literal-terminal rollouts.
    This completion-only port deliberately exposes neither visits nor policy
    targets to the training-data contract.
    """

    kind = CompletionKind.CLASSICAL_MCTS

    def __init__(self, *, iterations: int = 256, seed: int = 0xCAFEBABE, cpuct: float = 1.0) -> None:
        if iterations < 2:
            raise ValueError("iterations must be at least 2")
        self.iterations = iterations
        self.seed = seed
        self.cpuct = cpuct
        self._rng = _XorShift64(seed)

    def choose_action(self, board: HexBoard, *, virtual_winner: str) -> int:
        del virtual_winner
        if board.literal_winner() is not None:
            raise RuntimeError("cannot search a literal-terminal board")
        root = _MCTSNode(player_just_moved=opponent(board.side_to_move))
        for _ in range(self.iterations):
            position = board.copy()
            node = root
            path: list[_MCTSEdge] = []

            while node.expanded and position.literal_winner() is None:
                edge = self._select(node)
                position.play(edge.move)
                if edge.child is None:
                    edge.child = _MCTSNode(player_just_moved=opponent(position.side_to_move))
                path.append(edge)
                node = edge.child

            winner = position.literal_winner()
            if winner is None:
                self._expand(node, position)
                winner = self._rollout(position)
            self._backpropagate(path, winner)

        if not root.expanded:
            self._expand(root, board)
        if not root.children:
            raise RuntimeError("no legal completion action before literal terminal")
        # Historical code keeps the first ascending-index move on visit ties.
        best = max(root.children, key=lambda edge: edge.visits)
        if not board.is_legal(best.move):
            raise RuntimeError("classical MCTS selected an illegal move")
        return best.move

    def _select(self, node: _MCTSNode) -> _MCTSEdge:
        total_visits = sum(edge.visits for edge in node.children)
        root = math.sqrt(max(1.0, float(total_visits)))
        best_edge: _MCTSEdge | None = None
        best_score = -math.inf
        for edge in node.children:
            quality = edge.value_sum / edge.visits if edge.visits else 0.0
            exploration = self.cpuct * edge.prior * root / (1 + edge.visits)
            score = quality + exploration
            if score > best_score:
                best_score = score
                best_edge = edge
        assert best_edge is not None
        return best_edge

    @staticmethod
    def _expand(node: _MCTSNode, board: HexBoard) -> None:
        legal = board.legal_actions()
        prior = 1.0 / len(legal) if legal else 0.0
        node.children = [_MCTSEdge(move=move, prior=prior) for move in legal]
        node.expanded = True

    def _rollout(self, board: HexBoard) -> str:
        winner = board.literal_winner()
        while winner is None:
            legal = board.legal_actions()
            if not legal:
                raise RuntimeError("full Hex board has no literal winner")
            board.play(legal[self._rng.range(len(legal))])
            winner = board.literal_winner()
        return winner

    @staticmethod
    def _backpropagate(path: list[_MCTSEdge], winner: str) -> None:
        for edge in reversed(path):
            edge.visits += 1
            assert edge.child is not None and edge.child.player_just_moved is not None
            if winner == edge.child.player_just_moved:
                edge.value_sum += 1.0


class DeterministicShortestPathCompletion(CompletionStrategy):
    """Validation-only heuristic that advances each side's shortest path.

    It always consumes a legal cell and therefore reaches a literal terminal
    on a finite board. It is not a certified virtual-connection realizer; any
    disagreement between the virtual and literal winners is quarantined.
    """

    kind = CompletionKind.DETERMINISTIC_SHORTEST_PATH

    def choose_action(self, board: HexBoard, *, virtual_winner: str) -> int:
        del virtual_winner  # retained by the interface for future witness-aware strategies
        path = board.shortest_connection_path(board.side_to_move)
        for action in path:
            if board.is_legal(action):
                return action
        legal = board.legal_actions()
        if not legal:
            raise RuntimeError("no legal completion action before literal terminal")
        return legal[0]
