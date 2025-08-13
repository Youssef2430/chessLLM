"""
UI package for Chess LLM Benchmark.

This package contains user interface components for displaying real-time
chess game information, statistics, and results using Rich terminal UI.
"""

from .dashboard import Dashboard
from .board import (
    ChessBoardRenderer,
    BoardTheme,
    PieceStyle,
    BoardColors,
    render_board_simple,
    render_robot_battle
)

__all__ = [
    "Dashboard",
    "ChessBoardRenderer",
    "BoardTheme",
    "PieceStyle",
    "BoardColors",
    "render_board_simple",
    "render_robot_battle",
]
