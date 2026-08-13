"""Foundational components for the Group 49 Hex reconstruction."""

from .board import BOARD_AREA, BOARD_SIZE, HexBoard
from .schema import TrainingExample
from .validation import ValidationError, validate_example

__all__ = [
    "BOARD_AREA",
    "BOARD_SIZE",
    "HexBoard",
    "TrainingExample",
    "ValidationError",
    "validate_example",
]

