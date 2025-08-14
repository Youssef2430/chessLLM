"""
Adaptive chess engine management for low-ELO games.

This module provides an abstraction layer that dynamically switches between different chess
engines based on requested ELO levels, ensuring more realistic low-ELO play by utilizing
specialized engines better suited for different rating ranges.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, Union

import chess

from .engine import ChessEngine, autodetect_stockfish, validate_engine as engine_validate
from .human_engine import HumanLikeEngine, get_best_human_engine
from .models import Config

logger = logging.getLogger(__name__)


class AdaptiveEngineError(Exception):
    """Exception raised for errors in the AdaptiveEngine."""
    pass


class AdaptiveEngine:
    """
    A chess engine manager that dynamically selects the appropriate engine based on ELO.

    This class can switch between different engines to provide more realistic play at
    various ELO levels, particularly at the lower end of the spectrum where engines like
    Stockfish and Maia may not accurately simulate beginner play.
    """

    # ELO thresholds for engine selection
    STOCKFISH_MIN_ELO = 1300
    MAIA_MIN_ELO = 600  # Maia's 1100 model will be used for sub-1100 ratings

    # List of low-ELO capable engines in order of preference
    # These engines are better suited for sub-1100 ELO simulation
    LOW_ELO_ENGINES = ["texel", "madchess", "toga", "glaurung", "fruit", "crafty"]

    def __init__(self, config: Config):
        """
        Initialize the adaptive engine manager.

        Args:
            config: Global configuration object
        """
        self.config = config

        # Initialize engine instances
        self._primary_engine: Optional[Union[ChessEngine, HumanLikeEngine]] = None
        self._low_elo_engine: Optional[ChessEngine] = None
        self._current_engine: Optional[Union[ChessEngine, HumanLikeEngine]] = None

        # State tracking
        self._current_elo: Optional[int] = None
        self._is_running = False
        self._human_engine_type: Optional[str] = None

        # Engine paths
        self._stockfish_path: Optional[str] = None
        self._human_engine_info: Optional[Tuple[str, str]] = None
        self._low_elo_engine_path: Optional[str] = None
        self._low_elo_engine_type: Optional[str] = None

    async def start(self) -> None:
        """Start the engine manager and initialize available engines."""
        if self._is_running:
            return

        # 1. Detect and initialize primary engine (Stockfish or human-like)
        await self._initialize_primary_engine()

        # 2. Detect and initialize low-ELO engine
        await self._initialize_low_elo_engine()

        if not self._primary_engine:
            raise AdaptiveEngineError("Failed to initialize any chess engines")

        self._is_running = True
        logger.info("Adaptive engine manager started successfully")

    async def _initialize_primary_engine(self) -> None:
        """Initialize the primary chess engine (Stockfish or human-like)."""
        # First, try to use human-like engines if configured
        if self.config.use_human_engine:
            # Auto-detect best human engine
            human_engine_info = get_best_human_engine()
            if human_engine_info:
                engine_type, engine_path = human_engine_info
                self._human_engine_type = engine_type
                self._human_engine_info = human_engine_info

                # Import create_human_engine here to avoid circular imports
                from .human_engine import create_human_engine

                try:
                    engine = create_human_engine(engine_type, engine_path, self.config)
                    await engine.start()
                    self._primary_engine = engine
                    logger.info(f"Initialized primary human-like engine: {engine_type} at {engine_path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to initialize human-like engine: {e}")

        # Fall back to Stockfish
        stockfish_path = autodetect_stockfish(self.config.stockfish_path)
        if stockfish_path:
            self._stockfish_path = stockfish_path
            try:
                engine = ChessEngine(stockfish_path, self.config)
                await engine.start()
                self._primary_engine = engine
                logger.info(f"Initialized primary Stockfish engine at {stockfish_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize Stockfish engine: {e}")
        else:
            logger.error("No chess engines found. Please install Stockfish.")
            raise AdaptiveEngineError("No chess engines found")

    async def _initialize_low_elo_engine(self) -> None:
        """Initialize a chess engine specifically for low-ELO play."""
        for engine_name in self.LOW_ELO_ENGINES:
            engine_path = self._autodetect_engine(engine_name)
            if engine_path:
                try:
                    # Verify it's a valid UCI engine before using it
                    if engine_validate(engine_path):
                        engine = ChessEngine(engine_path, self.config)
                        await engine.start()
                        self._low_elo_engine = engine
                        self._low_elo_engine_path = engine_path
                        self._low_elo_engine_type = engine_name
                        logger.info(f"Initialized low-ELO engine: {engine_name} at {engine_path}")
                        return
                except Exception as e:
                    logger.warning(f"Failed to initialize {engine_name} engine: {e}")

        logger.info("No specialized low-ELO engines found, will use primary engine for all ratings")
        logger.debug(get_low_elo_engine_installation_hint())

    async def stop(self) -> None:
        """Stop all engines and clean up resources."""
        if not self._is_running:
            return

        # Stop the primary engine
        if self._primary_engine:
            try:
                await self._primary_engine.stop()
                logger.info("Primary engine stopped successfully")
            except Exception as e:
                logger.warning(f"Error stopping primary engine: {e}")

        # Stop the low-ELO engine if different from primary
        if self._low_elo_engine and self._low_elo_engine is not self._primary_engine:
            try:
                await self._low_elo_engine.stop()
                logger.info("Low-ELO engine stopped successfully")
            except Exception as e:
                logger.warning(f"Error stopping low-ELO engine: {e}")

        # Reset state
        self._current_engine = None
        self._current_elo = None
        self._is_running = False

    async def configure_elo(self, elo: int) -> None:
        """
        Configure the appropriate engine for the requested ELO rating.

        This method selects the appropriate engine based on the ELO rating and
        configures it accordingly.

        Args:
            elo: Target ELO rating
        """
        if not self._is_running:
            raise AdaptiveEngineError("Engine manager not started")

        # Warn about ELO values below absolute minimum
        MIN_ELO = 600
        if elo < MIN_ELO:
            logger.warning(f"⚠️  Requested ELO {elo} is below absolute minimum of {MIN_ELO}")
            logger.warning(f"   Adaptive engine will attempt configuration but gameplay quality may be poor")
            logger.warning(f"   Recommend using ELO {MIN_ELO} or higher for reliable chess benchmarking")

        # Skip if already configured for this ELO
        if self._current_elo == elo and self._current_engine is not None:
            return

        # 1. Determine which engine to use based on ELO threshold
        engine_to_use = self._primary_engine

        # Use low ELO engine if available and appropriate for sub-1100 ratings
        if self._low_elo_engine and elo < self._get_min_elo_threshold():
            engine_to_use = self._low_elo_engine
            min_threshold = self._get_min_elo_threshold()
            logger.info(f"Using specialized low-ELO engine ({self._low_elo_engine_type}) for ELO {elo} (below threshold {min_threshold})")
        else:
            engine_type = self._human_engine_type if self._human_engine_type else "stockfish"
            min_threshold = self._get_min_elo_threshold()
            if elo < min_threshold:
                logger.info(f"Using primary engine ({engine_type}) for sub-{min_threshold} ELO {elo} - consider installing specialized low-ELO engines for more realistic play")
            else:
                logger.info(f"Using primary engine ({engine_type}) for ELO {elo}")

        # 2. Configure the selected engine
        if not engine_to_use:
            raise AdaptiveEngineError("No engine available")

        await engine_to_use.configure_elo(elo)
        self._current_engine = engine_to_use
        self._current_elo = elo

    def _get_min_elo_threshold(self) -> int:
        """
        Get the minimum ELO threshold based on the primary engine type.

        Returns:
            Minimum ELO threshold for the primary engine
        """
        if isinstance(self._primary_engine, HumanLikeEngine):
            if self._human_engine_type == "maia":
                return self.MAIA_MIN_ELO
            # Other human-like engines would have their own thresholds here

        # Default to Stockfish threshold
        return self.STOCKFISH_MIN_ELO

    async def get_move(self, board: chess.Board) -> chess.Move:
        """
        Get a move from the currently configured engine.

        Args:
            board: Current chess position

        Returns:
            The engine's chosen move
        """
        if not self._current_engine:
            raise AdaptiveEngineError("No engine configured")

        return await self._current_engine.get_move(board)

    def _autodetect_engine(self, engine_name: str) -> Optional[str]:
        """
        Auto-detect a specific chess engine installation.

        Args:
            engine_name: Name of the engine to detect

        Returns:
            Path to the engine executable if found, None otherwise
        """
        # Check environment variable first
        env_var = f"{engine_name.upper()}_PATH"
        env_path = os.getenv(env_var)
        if env_path and Path(env_path).exists():
            return env_path

        # Check system path
        which_path = shutil.which(engine_name)
        if which_path:
            return which_path

        # Check common installation paths
        common_paths = [
            f"/usr/local/bin/{engine_name}",
            f"/usr/bin/{engine_name}",
            f"/opt/homebrew/bin/{engine_name}",
            f"C:/Program Files/{engine_name.capitalize()}/{engine_name}.exe",
            f"C:/{engine_name}/{engine_name}.exe",
            # Additional common locations
            f"/usr/games/{engine_name}",
            f"{os.path.expanduser('~')}/.local/bin/{engine_name}",
            f"{os.path.expanduser('~')}/chess_engines/{engine_name}/{engine_name}"
        ]

        for path in common_paths:
            if Path(path).exists():
                return path

        return None

    @property
    def is_running(self) -> bool:
        """True if the engine manager is currently running."""
        return self._is_running

    @property
    def current_elo(self) -> Optional[int]:
        """Current ELO configuration, if any."""
        return self._current_elo

    @property
    def current_engine_type(self) -> str:
        """Type of engine currently in use."""
        if not self._current_engine:
            return "none"

        if self._current_engine is self._low_elo_engine and self._low_elo_engine_type:
            return self._low_elo_engine_type

        if isinstance(self._current_engine, HumanLikeEngine):
            return self._current_engine.engine_type

        return "stockfish"

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.stop()


def validate_engine(engine_path: str) -> bool:
    """
    Validate that the given path points to a working UCI engine.

    Args:
        engine_path: Path to engine executable

    Returns:
        True if the engine is valid and responds to UCI commands
    """
    try:
        # Run the engine with a timeout to verify it's a valid UCI engine
        import subprocess
        process = subprocess.run(
            [engine_path],
            input="uci\nquit\n",
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return "uciok" in process.stdout.lower()
    except Exception:
        return False


def get_low_elo_engine_installation_hint() -> str:
    """
    Get a user-friendly message about how to install low-ELO engines.

    Returns:
        Formatted installation instructions
    """
    return (
        "No specialized sub-1100 ELO engines found. To enable better beginner-level play (ELO < 1100), install one of these engines:\n"
        "\n"
        "🎯 RECOMMENDED for sub-1100 ELO simulation:\n"
        "\n"
        "1. Texel - Excellent for lower ELO levels (600-1100)\n"
        "   • Linux:   sudo apt-get install texel\n"
        "   • macOS:   brew install texel\n"
        "   • Windows: Download from http://www.open-chess.org/viewtopic.php?f=5&t=3070\n"
        "\n"
        "2. MadChess - Good low-ELO simulation with realistic blunders\n"
        "   • All platforms: Download from https://github.com/kevingreenheck/madchess\n"
        "\n"
        "3. Fruit - Fast engine with good weak play simulation\n"
        "   • Linux:   sudo apt-get install fruit\n"
        "   • macOS:   brew install fruit\n"
        "   • Windows: Download from https://github.com/fathoms/Fruit-reloaded\n"
        "\n"
        "4. Crafty - Classic engine with excellent beginner-level play\n"
        "   • Linux:   sudo apt-get install crafty\n"
        "   • macOS:   brew install crafty\n"
        "   • Windows: Download from https://www.craftychess.com/\n"
        "\n"
        "5. Toga - Another good option for lower ratings\n"
        "   • All platforms: Download from http://www.mediafire.com/file/3jwz4sbgqt2dwd2/TogaII40.zip\n"
        "\n"
        "💡 Without these engines, the system will use Maia-1100 or Stockfish skill levels for sub-1100 ELOs,\n"
        "   which may not be as realistic for true beginner play.\n"
        "\n"
        "After installing, you can specify paths using environment variables:\n"
        "export TEXEL_PATH=/path/to/texel\n"
        "export MADCHESS_PATH=/path/to/madchess\n"
        "export FRUIT_PATH=/path/to/fruit\n"
        "export CRAFTY_PATH=/path/to/crafty\n"
        "export TOGA_PATH=/path/to/toga\n"
    )
