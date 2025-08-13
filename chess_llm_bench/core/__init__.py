"""
Core package for Chess LLM Benchmark.

This package contains the fundamental components for running chess games
between LLMs and engines, including data models, engine management,
and game orchestration.
"""

from .models import (
    BotSpec,
    GameRecord,
    LadderStats,
    LiveState,
    Config,
    BenchmarkResult
)

from .engine import (
    ChessEngine,
    EngineError,
    autodetect_stockfish,
    get_friendly_stockfish_hint
)

from .game import (
    GameRunner,
    LadderRunner
)

__all__ = [
    # Data models
    "BotSpec",
    "GameRecord",
    "LadderStats",
    "LiveState",
    "Config",
    "BenchmarkResult",

    # Engine components
    "ChessEngine",
    "EngineError",
    "autodetect_stockfish",
    "get_friendly_stockfish_hint",

    # Game components
    "GameRunner",
    "LadderRunner",
]
