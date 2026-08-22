"""Deterministic single-threaded PUCT reference search for physical 11x11 Hex.

The evaluator value is always from the *side-to-move in the evaluated board*
perspective.  Edge values are stored from their parent state's side-to-move
perspective.  That convention makes alternating-sign backup explicit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

from .board import BOARD_AREA, HexBoard, opponent


@dataclass(frozen=True)
class Evaluation:
    """Raw 121 physical-action logits and side-to-move scalar value in [-1, 1]."""

    policy_logits: Sequence[float]
    value: float


class Evaluator(Protocol):
    def evaluate(self, position: HexBoard) -> Evaluation: ...


@dataclass
class Edge:
    action: int
    prior: float
    child: "Node | None" = None
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class Node:
    """A state node. `visits` is the number of completed simulations through it."""

    to_play: str
    visits: int = 0
    edges: list[Edge] = field(default_factory=list)
    expansion_value: float = 0.0


@dataclass(frozen=True)
class SearchConfig:
    simulations: int
    c_puct: float = 1.5
    fpu_mode: str = "zero"
    fpu_reduction: float = 0.0

    def __post_init__(self) -> None:
        if self.simulations < 0:
            raise ValueError("simulations must be nonnegative")
        if not math.isfinite(self.c_puct) or self.c_puct < 0:
            raise ValueError("c_puct must be nonnegative")
        if self.fpu_mode not in {"zero", "parent_value_reduced"}:
            raise ValueError("unsupported fpu_mode")
        if not math.isfinite(self.fpu_reduction) or self.fpu_reduction < 0 or (self.fpu_mode == "zero" and self.fpu_reduction != 0):
            raise ValueError("invalid fpu_reduction")


@dataclass(frozen=True)
class SearchResult:
    selected_action: int | None
    root_visits: int
    raw_visits: tuple[int, ...]
    policy: tuple[float, ...]
    root_value: float | None
    priors: tuple[float, ...]
    raw_value_sums: tuple[float, ...] = ()
    evaluations: int = 0
    max_depth: int = 0


def compute_fpu_q(mode: str, parent_value: float, reduction: float, visited_prior_mass: float) -> float:
    if mode == "zero":
        return 0.0
    if mode != "parent_value_reduced" or reduction < 0:
        raise ValueError("invalid FPU configuration")
    raw = parent_value - reduction * math.sqrt(max(0.0, visited_prior_mass))
    return max(-1.0, min(1.0, raw))


def _terminal_value(position: HexBoard) -> float:
    """Value from `position.side_to_move`; legal Hex terminal states normally give -1."""

    winner = position.literal_winner()
    if winner is None:
        raise ValueError("terminal value requested for non-terminal position")
    return 1.0 if winner == position.side_to_move else -1.0


def _legal_priors(position: HexBoard, logits: Sequence[float]) -> list[Edge]:
    if len(logits) != BOARD_AREA:
        raise ValueError(f"evaluator must return {BOARD_AREA} logits")
    legal = position.legal_actions()
    if not legal:
        return []
    selected = [float(logits[action]) for action in legal]
    if not all(math.isfinite(value) for value in selected):
        raise ValueError("evaluator logits for legal actions must be finite")
    maximum = max(selected)
    weights = [math.exp(value - maximum) for value in selected]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("unable to normalize legal policy priors")
    return [Edge(action=action, prior=weight / total) for action, weight in zip(legal, weights)]


class DeterministicPUCT:
    """Reference PUCT: one thread, no virtual loss, no pruning, no batching.

    Selection score is ``Q + c_puct * P * sqrt(max(1, N_parent)) / (1+N_edge)``.
    Equal scores are resolved by smallest physical action index.  Root action
    selection is maximum raw edge visits, then smallest physical action index.
    """

    def __init__(self, evaluator: Evaluator, config: SearchConfig) -> None:
        self.evaluator = evaluator
        self.config = config

    def _expand(self, node: Node, position: HexBoard) -> float:
        evaluation = self.evaluator.evaluate(position)
        if not math.isfinite(evaluation.value) or not -1.000001 <= evaluation.value <= 1.000001:
            raise ValueError("evaluator value must be finite and within [-1,1]")
        node.expansion_value = float(evaluation.value)
        node.edges = _legal_priors(position, evaluation.policy_logits)
        return float(evaluation.value)

    def _fpu_q(self, node: Node) -> float:
        if self.config.fpu_mode == "zero":
            return 0.0
        visited_prior = sum(edge.prior for edge in node.edges if edge.visits > 0)
        return compute_fpu_q(self.config.fpu_mode, node.expansion_value, self.config.fpu_reduction, visited_prior)

    def _select(self, node: Node) -> Edge:
        if not node.edges:
            raise RuntimeError("cannot select from unexpanded node")
        scale = self.config.c_puct * math.sqrt(max(1, node.visits))
        fpu = self._fpu_q(node)
        return max(node.edges, key=lambda edge: ((edge.q if edge.visits else fpu) + scale * edge.prior / (1 + edge.visits), -edge.action))

    def search(self, source: HexBoard) -> SearchResult:
        """Search a copy; `source` is never changed."""

        position = source.copy()
        if position.literal_winner() is not None:
            zero = tuple(0 for _ in range(BOARD_AREA))
            return SearchResult(None, 0, zero, tuple(0.0 for _ in zero), _terminal_value(position), tuple(0.0 for _ in zero), tuple(0.0 for _ in zero))

        root = Node(to_play=position.side_to_move)
        self._expand(root, position)
        evaluations = 1
        max_depth = 0
        for _ in range(self.config.simulations):
            current = source.copy()
            node = root
            node_path = [root]
            edge_path: list[Edge] = []

            while node.edges and current.literal_winner() is None:
                edge = self._select(node)
                current.play(edge.action)
                edge_path.append(edge)
                if edge.child is None:
                    edge.child = Node(to_play=current.side_to_move)
                node = edge.child
                node_path.append(node)
                if node.visits == 0 or not node.edges:
                    break

            if current.literal_winner() is not None:
                leaf_value = _terminal_value(current)
            else:
                leaf_value = self._expand(node, current)
                evaluations += 1
            max_depth = max(max_depth, len(edge_path))

            # `leaf_value` is for current.to_play. Each parent is its opponent.
            for path_node in node_path:
                path_node.visits += 1
            value = leaf_value
            for edge in reversed(edge_path):
                value = -value
                edge.visits += 1
                edge.value_sum += value

        raw = [0] * BOARD_AREA
        priors = [0.0] * BOARD_AREA
        raw_value_sums = [0.0] * BOARD_AREA
        for edge in root.edges:
            raw[edge.action] = edge.visits
            priors[edge.action] = edge.prior
            raw_value_sums[edge.action] = edge.value_sum
        total = sum(raw)
        policy = [count / total if total else 0.0 for count in raw]
        selected = min((edge.action for edge in root.edges if edge.visits == max((candidate.visits for candidate in root.edges), default=0)), default=None)
        root_value = sum(edge.value_sum for edge in root.edges) / total if total else None
        return SearchResult(selected, total, tuple(raw), tuple(policy), root_value, tuple(priors), tuple(raw_value_sums), evaluations, max_depth)


class TorchStudentEvaluator:
    """Direct PyTorch bridge; avoids an unqualified historical ONNX/ORT boundary."""

    def __init__(self, checkpoint: Path, *, device: str = "cpu") -> None:
        import torch
        from .student_training import Group49Student

        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        architecture = payload["config"]["architecture"]
        self.model = Group49Student(channels=architecture["channels"], blocks=architecture["residual_blocks"]).to(device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.device = torch.device(device)
        self.checkpoint = Path(checkpoint)
        self.epoch = int(payload.get("epoch", -1))

    def evaluate(self, position: HexBoard) -> Evaluation:
        import torch

        state = torch.tensor(position.feature_planes(), dtype=torch.float32, device=self.device).reshape(1, 6, 11, 11)
        with torch.no_grad():
            logits, value = self.model(state)
        return Evaluation(tuple(float(item) for item in logits[0].detach().cpu()), float(value[0].item()))


class OutputAblationEvaluator:
    """Diagnostic output-level policy/value source mixer.

    It never combines network activations: the full 121 logits come from one
    qualified evaluator and the scalar side-to-move value from another.
    """

    def __init__(self, policy_evaluator: Evaluator, value_evaluator: Evaluator, *, label: str) -> None:
        self.policy_evaluator = policy_evaluator
        self.value_evaluator = value_evaluator
        self.label = label

    def evaluate(self, position: HexBoard) -> Evaluation:
        policy = self.policy_evaluator.evaluate(position)
        value = self.value_evaluator.evaluate(position)
        return Evaluation(policy.policy_logits, value.value)
