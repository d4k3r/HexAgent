"""Strict semantic validation for schema version 1."""

from __future__ import annotations

import math

from .schema import (
    BOARD_AREA,
    COLORS,
    PLANE_NAMES,
    POSITION_STATUSES,
    SCHEMA_VERSION,
    SOURCES,
    VALUE_PERSPECTIVES,
    TrainingExample,
)


class ValidationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate_example(example: TrainingExample, *, finalized: bool = True) -> None:
    _require(example.schema_version == SCHEMA_VERSION, "unsupported schema_version")
    _require(bool(example.game_id), "game_id is required")
    _require(0 <= example.ply <= BOARD_AREA, "ply must be in [0, 121]")

    state = example.state
    _require(tuple(state.plane_names) == PLANE_NAMES, "unexpected plane order")
    _require(tuple(state.shape) == (6, 11, 11), "state shape must be [6,11,11]")
    _require(len(state.planes) == 6, "state must contain six planes")
    for index, plane in enumerate(state.planes):
        _require(len(plane) == BOARD_AREA, f"plane {index} must contain 121 cells")
        _require(all(cell in (0, 1) for cell in plane), f"plane {index} must be binary")
    _require(state.side_to_move in COLORS, "invalid side_to_move")
    _require(all(not (red and blue) for red, blue in zip(state.planes[0], state.planes[1])), "red and blue planes overlap")
    expected_turn = 1 if state.side_to_move == "black" else 0
    _require(all(cell == expected_turn for cell in state.planes[2]), "turn plane disagrees with side_to_move")
    _require(sum(state.planes[3]) <= 1, "last_move plane must be empty or one-hot")

    policy = example.policy
    _require(len(policy.legal_mask) == BOARD_AREA, "legal_mask must contain 121 entries")
    _require(all(type(item) is bool for item in policy.legal_mask), "legal_mask must be boolean")
    _require(math.isfinite(policy.weight) and policy.weight >= 0, "invalid policy weight")
    if example.position_status != "literal_terminal":
        expected_legal = [not (red or blue) for red, blue in zip(state.planes[0], state.planes[1])]
        _require(policy.legal_mask == expected_legal, "legal_mask disagrees with board occupancy")

    if policy.pi is None:
        _require(policy.weight == 0, "pi=null requires policy weight 0")
        _require(policy.target_kind is None, "pi=null requires target_kind=null")
    else:
        _require(len(policy.pi) == BOARD_AREA, "pi must contain 121 entries")
        _require(policy.weight > 0, "supervised pi requires positive policy weight")
        _require(policy.target_kind == "mcts_visits", "soft policy must be an MCTS visit target")
        _require(all(math.isfinite(p) and p >= 0 for p in policy.pi), "pi must be finite and nonnegative")
        _require(math.isclose(sum(policy.pi), 1.0, rel_tol=0, abs_tol=1e-6), "pi must sum to 1")
        for action, (probability, legal) in enumerate(zip(policy.pi, policy.legal_mask)):
            _require(legal or abs(probability) <= 1e-12, f"pi has mass on illegal action {action}")

    if policy.raw_visit_counts is not None:
        counts = policy.raw_visit_counts
        _require(len(counts) == BOARD_AREA, "raw_visit_counts must contain 121 entries")
        _require(all(type(count) is int and count >= 0 for count in counts), "visit counts must be nonnegative integers")
        for action, (count, legal) in enumerate(zip(counts, policy.legal_mask)):
            _require(legal or count == 0, f"visit count on illegal action {action}")
        if policy.pi is not None:
            total = sum(counts)
            _require(total > 0, "supervised visit counts must have positive total")
            for action, (probability, count) in enumerate(zip(policy.pi, counts)):
                expected = count / total
                _require(math.isclose(probability, expected, rel_tol=0, abs_tol=1e-6), f"pi does not normalize visits at action {action}")
    elif policy.pi is not None:
        raise ValidationError("MCTS pi requires raw_visit_counts")

    chosen = example.transition.chosen_action
    if chosen is not None:
        _require(type(chosen) is int and 0 <= chosen < BOARD_AREA, "chosen_action must be a physical action in [0,120]")
        _require(policy.legal_mask[chosen], "chosen_action must be legal in the recorded pre-move state")
    _require(example.transition.control_event in (None, "swap"), "invalid control event")
    _require(not (chosen is not None and example.transition.control_event is not None), "physical and control actions are mutually exclusive")

    value = example.value
    if finalized:
        _require(value.z in (-1.0, 1.0), "finalized z must be -1 or 1")
    else:
        _require(value.z is None or value.z in (-1.0, 1.0), "z must be null, -1, or 1")
    _require(value.z_perspective == "side_to_move", "z perspective must be side_to_move")
    _require(math.isfinite(value.weight) and value.weight >= 0, "invalid value weight")
    if value.teacher_root_value is None:
        _require(value.teacher_value_type is None, "missing teacher value requires null value type")
        _require(value.teacher_value_perspective is None, "missing teacher value requires null perspective")
    else:
        _require(math.isfinite(value.teacher_root_value), "teacher_root_value must be finite")
        _require(bool(value.teacher_value_type), "teacher value type is required")
        _require(value.teacher_value_perspective in VALUE_PERSPECTIVES, "invalid teacher value perspective")

    _require(example.source in SOURCES, "invalid source")
    _require(example.position_status in POSITION_STATUSES, "invalid position status")
    terminal = example.terminal
    _require(terminal.virtual_winner in COLORS | {None}, "invalid virtual winner")
    _require(terminal.literal_winner in COLORS | {None}, "invalid literal winner")

    if example.position_status == "normal":
        _require(terminal.virtual_winner is None and terminal.literal_winner is None, "normal position cannot already be terminal")
    elif example.position_status == "virtual_terminal":
        _require(terminal.virtual_winner is not None, "virtual_terminal requires virtual_winner")
        _require(terminal.literal_winner is None, "virtual_terminal must precede literal terminal")
        _require(example.source == "completion", "virtual-terminal rows must be completion rows")
    else:
        _require(terminal.literal_winner is not None, "literal_terminal requires literal_winner")
        _require(chosen is None, "literal terminal has no chosen action")
        _require(policy.pi is None and policy.weight == 0, "literal terminal has no policy target")
        _require(not any(policy.legal_mask), "literal terminal has no legal actions")

    if example.source == "completion":
        _require(policy.pi is None and policy.weight == 0, "initial completion rows are policy-unsupervised")
    if example.source != "katahex_teacher":
        _require(value.teacher_root_value is None, "teacher value is only valid for KataHex teacher rows")

    provenance = example.provenance
    required_strings = {
        "generator_version": provenance.generator_version,
        "generator_commit": provenance.generator_commit,
        "engine_repository": provenance.engine_repository,
        "engine_commit": provenance.engine_commit,
        "model_filename": provenance.model_filename,
        "model_sha256": provenance.model_sha256,
        "model_release": provenance.model_release,
        "config_sha256": provenance.config_sha256,
        "raw_log_reference": provenance.raw_log_reference,
    }
    for name, value_string in required_strings.items():
        _require(isinstance(value_string, str) and bool(value_string.strip()), f"provenance.{name} is required")
    _require(isinstance(provenance.search_settings, dict), "search_settings must be an object")
    _require(isinstance(provenance.seed, (str, int)), "seed is required")
