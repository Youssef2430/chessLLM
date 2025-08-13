"""
Chess LLM Benchmark - A tool for testing LLMs with chess games and assessing their ELOs.

This package provides a framework for running chess games between Large Language Models
and Stockfish at various ELO ratings, creating a ladder system to evaluate LLM
chess-playing capabilities.
"""

__version__ = "0.2.0"
__author__ = "Chess LLM Bench Team"
__license__ = "MIT"

# Core imports
from .core.models import BotSpec, GameRecord, LadderStats, LiveState
from .core.engine import ChessEngine
from .core.game import GameRunner
from .llm.client import LLMClient
from .ui.dashboard import Dashboard
from .cli import main

__all__ = [
    "BotSpec",
    "GameRecord",
    "LadderStats",
    "LiveState",
    "ChessEngine",
    "GameRunner",
    "LLMClient",
    "Dashboard",
    "main",
]
