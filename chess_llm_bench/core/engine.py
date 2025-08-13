"""
Chess engine management for the Chess LLM Benchmark.

This module provides an abstraction layer for interacting with UCI chess engines,
primarily Stockfish, with support for ELO-limited play and robust error handling.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import logging

import chess
import chess.engine as chess_engine

from .models import Config

logger = logging.getLogger(__name__)


class EngineError(Exception):
    """Custom exception for engine-related errors."""
    pass


class ChessEngine:
    """
    Manages a UCI chess engine (typically Stockfish) for ELO-limited play.

    This class handles engine lifecycle, configuration, and move generation
    with proper error handling and resource cleanup.
    """

    def __init__(self, engine_path: str, config: Config):
        """
        Initialize the chess engine manager.

        Args:
            engine_path: Path to the UCI engine executable
            config: Global configuration object
        """
        self.engine_path = engine_path
        self.config = config
        self._engine: Optional[chess_engine.SimpleEngine] = None
        self._current_elo: Optional[int] = None
        self._is_running = False

    async def start(self) -> None:
        """Start the UCI engine."""
        if self._is_running:
            return

        try:
            self._engine = chess_engine.SimpleEngine.popen_uci(self.engine_path)
            self._is_running = True
            logger.info(f"Started engine: {self.engine_path}")

            # Get engine info
            engine_name = self._engine.id.get("name", "Unknown Engine")
            logger.info(f"Engine name: {engine_name}")

        except Exception as e:
            raise EngineError(f"Failed to start engine at {self.engine_path}: {e}")

    async def stop(self) -> None:
        """Stop the UCI engine and clean up resources."""
        if not self._is_running or not self._engine:
            return

        try:
            self._engine.quit()
            logger.info("Engine stopped successfully")
        except Exception as e:
            logger.warning(f"Error stopping engine: {e}")
        finally:
            self._engine = None
            self._current_elo = None
            self._is_running = False

    async def configure_elo(self, elo: int) -> None:
        """
        Configure the engine to play at a specific ELO rating.

        Args:
            elo: Target ELO rating
        """
        if not self._engine:
            raise EngineError("Engine not started")

        if self._current_elo == elo:
            return  # Already configured for this ELO

        try:
            # Try UCI_LimitStrength first (Stockfish standard)
            await asyncio.to_thread(
                self._engine.configure,
                {"UCI_LimitStrength": True, "UCI_Elo": elo}
            )
            self._current_elo = elo
            logger.debug(f"Configured engine for ELO {elo} using UCI_LimitStrength")

        except chess_engine.EngineError:
            # Fallback to Skill Level for engines that don't support UCI_Elo
            try:
                skill_level = max(0, min(20, (elo - 1000) // 50))
                await asyncio.to_thread(
                    self._engine.configure,
                    {"Skill Level": skill_level}
                )
                self._current_elo = elo
                logger.debug(f"Configured engine for ELO {elo} using Skill Level {skill_level}")

            except Exception as e:
                logger.warning(f"Could not configure engine strength: {e}")
                # Continue without strength limitation

    async def get_move(self, board: chess.Board) -> chess.Move:
        """
        Get the engine's move for the given position.

        Args:
            board: Current chess position

        Returns:
            The engine's chosen move

        Raises:
            EngineError: If the engine fails to provide a move
        """
        if not self._engine:
            raise EngineError("Engine not started")

        if board.is_game_over():
            raise EngineError("Cannot get move for finished game")

        try:
            result = await asyncio.to_thread(
                self._engine.play,
                board,
                chess_engine.Limit(time=self.config.think_time)
            )

            if not result.move:
                raise EngineError("Engine returned no move")

            return result.move

        except Exception as e:
            raise EngineError(f"Engine move generation failed: {e}")

    async def analyze_position(self, board: chess.Board, depth: int = 10) -> Dict[str, Any]:
        """
        Analyze a chess position and return evaluation info.

        Args:
            board: Position to analyze
            depth: Search depth

        Returns:
            Dictionary with analysis results (score, best move, etc.)
        """
        if not self._engine:
            raise EngineError("Engine not started")

        try:
            info = await asyncio.to_thread(
                self._engine.analyse,
                board,
                chess_engine.Limit(depth=depth)
            )

            result = {
                "score": info.get("score"),
                "depth": info.get("depth"),
                "nodes": info.get("nodes"),
                "time": info.get("time"),
                "pv": info.get("pv", []),
            }

            if result["pv"]:
                result["best_move"] = result["pv"][0]

            return result

        except Exception as e:
            raise EngineError(f"Position analysis failed: {e}")

    @property
    def is_running(self) -> bool:
        """True if the engine is currently running."""
        return self._is_running and self._engine is not None

    @property
    def current_elo(self) -> Optional[int]:
        """Current ELO configuration, if any."""
        return self._current_elo

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.stop()


def autodetect_stockfish(cli_path: Optional[str] = None) -> Optional[str]:
    """
    Auto-detect Stockfish installation path.

    Search order:
    1. Explicit CLI path argument
    2. STOCKFISH_PATH environment variable
    3. System PATH lookup
    4. Common installation directories

    Args:
        cli_path: Explicitly provided path (highest priority)

    Returns:
        Path to Stockfish executable if found, None otherwise
    """
    # 1. CLI argument
    if cli_path and Path(cli_path).exists():
        return cli_path

    # 2. Environment variable
    env_path = os.getenv("STOCKFISH_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # 3. System PATH
    which_path = shutil.which("stockfish")
    if which_path:
        return which_path

    # 4. Common installation paths
    common_paths = [
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
        "C:/Program Files/Stockfish/stockfish.exe",
        "C:/stockfish/stockfish.exe",
    ]

    for path in common_paths:
        if Path(path).exists():
            return path

    return None


def validate_engine(engine_path: str) -> bool:
    """
    Validate that the given path points to a working UCI engine.

    Args:
        engine_path: Path to engine executable

    Returns:
        True if the engine is valid and responds to UCI commands
    """
    try:
        # Quick test: run engine with "uci" command
        result = subprocess.run(
            [engine_path],
            input="uci\nquit\n",
            text=True,
            capture_output=True,
            timeout=5
        )

        # Check if engine responds with "uciok"
        return "uciok" in result.stdout.lower()

    except Exception as e:
        logger.debug(f"Engine validation failed for {engine_path}: {e}")
        return False


def get_engine_info(engine_path: str) -> Dict[str, str]:
    """
    Get basic information about a UCI engine.

    Args:
        engine_path: Path to engine executable

    Returns:
        Dictionary with engine name, author, and other info
    """
    info = {}

    try:
        result = subprocess.run(
            [engine_path],
            input="uci\nquit\n",
            text=True,
            capture_output=True,
            timeout=5
        )

        for line in result.stdout.split('\n'):
            if line.startswith("id name "):
                info["name"] = line[8:].strip()
            elif line.startswith("id author "):
                info["author"] = line[10:].strip()
            elif line.startswith("option name "):
                # Parse engine options (for advanced configuration)
                if "options" not in info:
                    info["options"] = []
                info["options"].append(line[12:].strip())

    except Exception as e:
        logger.debug(f"Failed to get engine info for {engine_path}: {e}")

    return info


def get_friendly_stockfish_hint() -> str:
    """
    Get a user-friendly message about how to install Stockfish.

    Returns:
        Formatted installation instructions
    """
    return (
        "Stockfish not found. Install it and try again:\n"
        "• macOS:    brew install stockfish\n"
        "• Ubuntu:   sudo apt-get install stockfish\n"
        "• Windows:  choco install stockfish\n"
        "• Manual:   Download from https://stockfishchess.org/\n"
        "\nOr set environment variable: export STOCKFISH_PATH=/path/to/stockfish"
    )
