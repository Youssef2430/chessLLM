"""Chess benchmarks with explicit outcomes, move analysis and usage telemetry.

Opponent UCI_Elo settings do not establish an LLM Elo rating.
"""

__version__ = "0.4.0"
__author__ = "Chess LLM Bench Team"
__license__ = "MIT"

# Core imports
from .core.models import BotSpec, GameRecord, LadderStats, LiveState
from .core.engine import ChessEngine
from .core.human_engine import (
    HumanLikeEngine,
    MaiaEngine,
    LeelaEngine,
    HumanStockfishEngine,
)
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
    "HumanLikeEngine",
    "MaiaEngine",
    "LeelaEngine",
    "HumanStockfishEngine",
    "GameRunner",
    "LLMClient",
    "Dashboard",
    "main",
]
