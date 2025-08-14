"""
Human-like chess engine management for the Chess LLM Benchmark.

This module provides human-like chess opponents that play more naturally than
traditional engines like Stockfish. Includes support for Maia Chess Engine,
Leela Chess Zero, and human-like configurations.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict
import logging
import random

import chess
import chess.engine as chess_engine

from .models import Config

logger = logging.getLogger(__name__)


class HumanEngineError(Exception):
    """Custom exception for human-like engine-related errors."""
    pass


class HumanLikeEngine:
    """
    Base class for human-like chess engines.

    Provides a common interface for engines that play more human-like chess,
    including neural network based engines and modified traditional engines.
    """

    def __init__(self, engine_path: str, config: Config, engine_type: str = "maia"):
        """
        Initialize the human-like chess engine manager.

        Args:
            engine_path: Path to the engine executable
            config: Global configuration object
            engine_type: Type of engine ("maia", "lczero", "human_stockfish")
        """
        self.engine_path = engine_path
        self.config = config
        self.engine_type = engine_type.lower()
        self._engine: Optional[chess_engine.SimpleEngine] = None
        self._current_elo: Optional[int] = None
        self._is_running = False

    async def start(self) -> None:
        """Start the human-like engine."""
        if self._is_running:
            return

        try:
            self._engine = chess_engine.SimpleEngine.popen_uci(self.engine_path)
            self._is_running = True
            logger.info(f"Started human-like engine ({self.engine_type}): {self.engine_path}")

            # Get engine info
            engine_name = self._engine.id.get("name", "Unknown Engine")
            logger.info(f"Human-like engine name: {engine_name}")

            # Apply engine-specific initialization
            await self._initialize_engine_specific()

        except Exception as e:
            raise HumanEngineError(f"Failed to start human-like engine at {self.engine_path}: {e}")

    async def stop(self) -> None:
        """Stop the human-like engine and clean up resources."""
        if not self._is_running or not self._engine:
            return

        try:
            self._engine.quit()
            logger.info("Human-like engine stopped successfully")
        except Exception as e:
            logger.warning(f"Error stopping human-like engine: {e}")
        finally:
            self._engine = None
            self._current_elo = None
            self._is_running = False

    async def configure_elo(self, elo: int) -> None:
        """
        Configure the engine to play at a human-like level for specific ELO rating.

        Args:
            elo: Target ELO rating for human-like play
        """
        if not self._engine:
            raise HumanEngineError("Engine not started")

        # Warn about ELO values below absolute minimum
        MIN_ELO = 600
        if elo < MIN_ELO:
            logger.warning(f"⚠️  Requested ELO {elo} is below absolute minimum of {MIN_ELO}")
            logger.warning(f"   Human-like engine will attempt configuration but may not provide realistic gameplay")
            logger.warning(f"   Recommend using ELO {MIN_ELO} or higher for authentic human-like chess behavior")

        if self._current_elo == elo:
            return  # Already configured for this ELO

        await self._configure_engine_specific_elo(elo)
        self._current_elo = elo
        logger.debug(f"Configured human-like engine for ELO {elo}")

    async def get_move(self, board: chess.Board) -> chess.Move:
        """
        Get a human-like move for the given position.

        Args:
            board: Current chess position

        Returns:
            The engine's chosen move with human-like characteristics

        Raises:
            HumanEngineError: If the engine fails to provide a move
        """
        if not self._engine:
            raise HumanEngineError("Engine not started")

        if board.is_game_over():
            raise HumanEngineError("Cannot get move for finished game")

        try:
            # Get base move from engine
            base_move = await self._get_engine_move(board)

            # Apply human-like modifications
            final_move = await self._apply_human_like_modifications(board, base_move)

            return final_move

        except Exception as e:
            raise HumanEngineError(f"Human-like engine move generation failed: {e}")

    async def _initialize_engine_specific(self) -> None:
        """Initialize engine-specific settings. Override in subclasses."""
        pass

    async def _configure_engine_specific_elo(self, elo: int) -> None:
        """Configure engine-specific ELO settings. Override in subclasses."""
        if self.engine_type == "maia":
            await self._configure_maia_elo(elo)
        elif self.engine_type == "lczero":
            await self._configure_lczero_elo(elo)
        elif self.engine_type == "human_stockfish":
            await self._configure_human_stockfish_elo(elo)
        else:
            logger.warning(f"Unknown engine type for ELO configuration: {self.engine_type}")

    async def _get_engine_move(self, board: chess.Board) -> chess.Move:
        """Get base move from the underlying engine."""
        try:
            if not self._engine:
                raise HumanEngineError("Engine not started")

            # Use fixed time per move for consistent strength
            move_time = self.config.think_time
            logger.debug(f"Human engine thinking time: {move_time:.3f}s")

            result = await asyncio.to_thread(
                self._engine.play,
                board,
                chess_engine.Limit(time=move_time)
            )

            if not result.move:
                raise HumanEngineError("Engine returned no move")

            logger.debug(f"Human engine selected move: {result.move.uci()}")
            return result.move

        except Exception as e:
            raise HumanEngineError(f"Engine move generation failed: {e}")

    async def _apply_human_like_modifications(self, board: chess.Board, base_move: chess.Move) -> chess.Move:
        """
        Apply human-like modifications to the base move.

        This includes:
        - Occasional blunders at lower ELOs
        - Opening book variations
        - Endgame simplifications
        - Time pressure simulation
        - Move variation for more human-like play
        """
        if not self._current_elo:
            return base_move

        # Calculate modification probability based on ELO
        # Higher probability for lower ELOs to simulate human inconsistency
        modification_prob = max(0.1, (1800 - self._current_elo) / 1800) * 0.3

        # Add some randomness even for high ELOs to avoid robotic play
        if self._current_elo > 1800:
            modification_prob = 0.05  # 5% chance even for strong players

        # Occasionally make a suboptimal move for human-likeness
        if random.random() < modification_prob:
            return await self._get_human_like_alternative(board)

        return base_move

    async def _get_human_like_alternative(self, board: chess.Board) -> chess.Move:
        """Get a human-like alternative move (slightly suboptimal)."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise HumanEngineError("No legal moves available")

        # For very low ELO simulation, sometimes pick a random legal move
        if self._current_elo and self._current_elo < 800:
            # 50% chance of random move for very weak players
            if random.random() < 0.5:
                return random.choice(legal_moves)

        # For low to medium ELO, get multiple moves and pick from top choices
        try:
            if not self._engine:
                return random.choice(legal_moves)

            # Use longer analysis time for better move alternatives
            multipv_result = await asyncio.to_thread(
                self._engine.analyse,
                board,
                chess_engine.Limit(time=0.3),
                multipv=min(5, len(legal_moves))
            )

            if multipv_result and len(multipv_result) > 1:
                # Weight selection based on ELO
                if self._current_elo and self._current_elo < 1200:
                    # Lower ELO: more likely to pick suboptimal moves
                    choices = [0, 1, 2, 3, 4][:min(len(multipv_result), 5)]
                    weights = [0.4, 0.3, 0.2, 0.08, 0.02][:len(choices)]
                    choice_idx = random.choices(choices, weights=weights)[0]
                else:
                    # Higher ELO: mostly pick from top 3 moves
                    choice_idx = random.choice([0, 1, 2][:min(len(multipv_result), 3)])

                pv = multipv_result[choice_idx].get("pv", [])
                if pv:
                    return pv[0]
        except Exception as e:
            logger.debug(f"Failed to get alternative move: {e}")

        # Fallback: pick a reasonable move from legal moves
        # Prioritize captures, checks, and central moves for more human-like play
        good_moves = []
        for move in legal_moves:
            if board.is_capture(move) or board.gives_check(move):
                good_moves.append(move)
            elif move.to_square in [chess.E4, chess.E5, chess.D4, chess.D5]:  # Central squares
                good_moves.append(move)

        if good_moves:
            return random.choice(good_moves)
        else:
            return random.choice(legal_moves)

    async def _configure_maia_elo(self, elo: int) -> None:
        """Configure Maia engine for specific ELO."""
        # Maia models are trained for specific rating ranges
        # Map ELO to closest Maia model
        maia_models = {
            1100: "maia-1100",
            1200: "maia-1200",
            1300: "maia-1300",
            1400: "maia-1400",
            1500: "maia-1500",
            1600: "maia-1600",
            1700: "maia-1700",
            1800: "maia-1800",
            1900: "maia-1900"
        }

        # For sub-1100 ELOs, use the lowest available Maia model (1100)
        # as it's the best approximation for beginner play
        if elo < 1100:
            model_name = "maia-1100"
            logger.debug(f"Using maia-1100 model for sub-1100 ELO {elo}")
        else:
            # Find closest Maia model for ELO >= 1100
            closest_elo = min(maia_models.keys(), key=lambda x: abs(x - elo))
            model_name = maia_models[closest_elo]

        try:
            if not self._engine:
                logger.warning("Engine not started for Maia configuration")
                return

            await asyncio.to_thread(
                self._engine.configure,
                {"Maia_Model": model_name}
            )
            logger.debug(f"Configured Maia with model {model_name} for ELO {elo}")
        except chess_engine.EngineError as e:
            logger.warning(f"Could not configure Maia model: {e}")

    async def _configure_lczero_elo(self, elo: int) -> None:
        """Configure Leela Chess Zero for human-like play at specific ELO."""
        # Configure LCZero for more human-like play
        # Reduce search nodes and add noise for lower ELOs
        nodes = max(100, min(10000, elo))  # Scale nodes with ELO
        temperature = max(0.1, (1800 - elo) / 1000)  # More random at lower ELOs

        try:
            config_options = {
                "Nodes": nodes,
                "Temperature": temperature,
                "TempDecayMoves": 30,
                "Noise": True if elo < 1500 else False
            }

            if not self._engine:
                logger.warning("Engine not started for LCZero configuration")
                return

            await asyncio.to_thread(self._engine.configure, config_options)
            logger.debug(f"Configured LCZero for ELO {elo}: nodes={nodes}, temp={temperature}")
        except chess_engine.EngineError as e:
            logger.warning(f"Could not configure LCZero: {e}")

    async def _configure_human_stockfish_elo(self, elo: int) -> None:
        """Configure Stockfish with human-like parameters for any ELO level."""
        # Use Stockfish but with human-like settings optimized for different ELO ranges
        try:
            if not self._engine:
                logger.warning("Engine not started for human Stockfish configuration")
                return

            # Stockfish UCI_Elo has minimum value (usually 1320)
            # For lower ELOs, use Skill Level with additional human-like parameters
            if elo >= 1320:
                config_options = {
                    "UCI_LimitStrength": True,
                    "UCI_Elo": elo,
                    # Add some randomness for more human-like play
                    "MultiPV": 1,
                    "Contempt": max(-50, min(50, (elo - 1500) // 20))
                }
            else:
                # Enhanced skill level mapping for sub-1100 ELOs
                # More granular mapping for better differentiation at low levels
                if elo < 600:
                    skill_level = 0
                elif elo < 800:
                    skill_level = max(1, (elo - 500) // 50)  # 1-6 range
                elif elo < 1100:
                    skill_level = max(6, (elo - 600) // 40)  # 6-12 range
                else:  # 1100-1319
                    skill_level = max(12, min(20, (elo - 900) // 30))  # 12-20 range

                config_options = {
                    "UCI_LimitStrength": False,
                    "Skill Level": skill_level,
                    # Additional parameters for more human-like weak play
                    "Contempt": -20,  # Slightly pessimistic for beginners
                    "MultiPV": 1,
                    # Reduce search depth for lower skill levels
                    "Depth": max(1, min(15, skill_level + 3))
                }

            await asyncio.to_thread(self._engine.configure, config_options)
            skill_info = f"skill level {config_options.get('Skill Level', 'N/A')}" if elo < 1320 else f"UCI_Elo {elo}"
            logger.debug(f"Configured human-like Stockfish for ELO {elo} using {skill_info}: {config_options}")
        except chess_engine.EngineError as e:
            logger.warning(f"Could not configure human-like Stockfish: {e}")

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


def autodetect_human_engines() -> Dict[str, Optional[str]]:
    """
    Auto-detect available human-like chess engines.

    Returns:
        Dictionary mapping engine type to path, None if not found
    """
    engines = {}

    # Check for Maia
    maia_paths = [
        "maia",
        "/usr/local/bin/maia",
        "/opt/homebrew/bin/maia",
        shutil.which("maia")
    ]

    for path in maia_paths:
        if path and Path(path).exists():
            engines["maia"] = path
            break
    else:
        engines["maia"] = None

    # Check for LCZero
    lczero_paths = [
        "lc0",
        "leela-chess-zero",
        "/usr/local/bin/lc0",
        "/opt/homebrew/bin/lc0",
        shutil.which("lc0"),
        shutil.which("leela-chess-zero")
    ]

    for path in lczero_paths:
        if path and Path(path).exists():
            engines["lczero"] = path
            break
    else:
        engines["lczero"] = None

    # Stockfish is always available for human-like mode if it exists
    stockfish_path = shutil.which("stockfish")
    if stockfish_path:
        engines["human_stockfish"] = stockfish_path
    else:
        engines["human_stockfish"] = None

    return engines


def get_best_human_engine(preferred_type: Optional[str] = None) -> Optional[tuple[str, str]]:
    """
    Get the best available human-like engine.

    Args:
        preferred_type: Preferred engine type ("maia", "lczero", "human_stockfish")

    Returns:
        Tuple of (engine_type, path) for the best available engine,
        or None if no human-like engines are available.
    """
    available = autodetect_human_engines()

    # If a preferred type is specified and available, use it
    if preferred_type and preferred_type in available:
        return preferred_type, available[preferred_type]

    # Priority order: Maia > LCZero > Human Stockfish
    for engine_type in ["maia", "lczero", "human_stockfish"]:
        engine_path = available.get(engine_type)
        if engine_path:
            return engine_type, engine_path

    return None


def validate_human_engine(engine_path: str) -> bool:
    """
    Validate that the given path points to a working human-like UCI engine.

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
        logger.debug(f"Human engine validation failed for {engine_path}: {e}")
        return False


def get_human_engine_installation_hint() -> str:
    """
    Get a user-friendly message about how to install human-like engines.

    Returns:
        Formatted installation instructions
    """
    return (
        "Human-like engines not found. Install one and try again:\n\n"
        "🧠 MAIA (Recommended - Most Human-like):\n"
        "• Download from: https://github.com/CSSLab/maia-chess\n"
        "• Follow installation instructions for your platform\n\n"
        "♟️  LEELA CHESS ZERO:\n"
        "• macOS:    brew install lc0\n"
        "• Ubuntu:   sudo apt-get install lc0\n"
        "• Windows:  Download from https://lczero.org/\n\n"
        "🤖 HUMAN-LIKE STOCKFISH:\n"
        "• Uses regular Stockfish with human-like settings\n"
        "• macOS:    brew install stockfish\n"
        "• Ubuntu:   sudo apt-get install stockfish\n"
        "• Windows:  choco install stockfish\n\n"
        "Or set environment variable: export MAIA_PATH=/path/to/maia"
    )


class MaiaEngine(HumanLikeEngine):
    """
    Maia Chess Engine - specifically trained to play like humans.

    Maia is a neural network trained on human games to play chess
    more like a human player at different rating levels.
    """

    def __init__(self, engine_path: str, config: Config):
        """Initialize Maia engine."""
        super().__init__(engine_path, config, "maia")

    async def _initialize_engine_specific(self) -> None:
        """Initialize Maia-specific settings."""
        try:
            # Configure Maia for human-like play
            if not self._engine:
                logger.warning("Engine not started for Maia initialization")
                return

            await asyncio.to_thread(
                self._engine.configure,
                {
                    "Hash": min(512, 128),  # Moderate hash size
                    "Threads": 1,           # Single thread for consistency
                    "MultiPV": 1            # Single best move
                }
            )
            logger.debug("Initialized Maia-specific settings")
        except Exception as e:
            logger.warning(f"Could not apply Maia-specific settings: {e}")


class LeelaEngine(HumanLikeEngine):
    """
    Leela Chess Zero configured for human-like play.
    """

    def __init__(self, engine_path: str, config: Config):
        """Initialize Leela Chess Zero engine."""
        super().__init__(engine_path, config, "lczero")

    async def _initialize_engine_specific(self) -> None:
        """Initialize LCZero-specific settings for human-like play."""
        try:
            # Configure LCZero for more human-like behavior
            if not self._engine:
                logger.warning("Engine not started for LCZero initialization")
                return

            await asyncio.to_thread(
                self._engine.configure,
                {
                    "Threads": 1,
                    "NNCacheSize": 200000,
                    "MinibatchSize": 8,
                    "MaxPrefetch": 32
                }
            )
            logger.debug("Initialized LCZero-specific settings")
        except Exception as e:
            logger.warning(f"Could not apply LCZero-specific settings: {e}")


class HumanStockfishEngine(HumanLikeEngine):
    """
    Stockfish configured to play more human-like chess.
    """

    def __init__(self, engine_path: str, config: Config):
        """Initialize human-like Stockfish engine."""
        super().__init__(engine_path, config, "human_stockfish")

    async def _initialize_engine_specific(self) -> None:
        """Initialize human-like Stockfish settings."""
        try:
            # Configure Stockfish for more human-like play
            if not self._engine:
                logger.warning("Engine not started for human Stockfish initialization")
                return

            # Only use options that are commonly supported across Stockfish versions
            basic_options = {
                "Hash": 64,           # Smaller hash for faster/less perfect play
                "Threads": 1,         # Single thread
            }

            await asyncio.to_thread(self._engine.configure, basic_options)
            logger.debug("Initialized human-like Stockfish settings")
        except Exception as e:
            logger.warning(f"Could not apply human-like Stockfish settings: {e}")


def create_human_engine(engine_type: str, engine_path: str, config: Config) -> HumanLikeEngine:
    """
    Factory function to create the appropriate human-like engine.

    Args:
        engine_type: Type of engine ("maia", "lczero", "human_stockfish")
        engine_path: Path to engine executable
        config: Global configuration

    Returns:
        Configured human-like engine instance
    """
    engine_type = engine_type.lower()

    if engine_type == "maia":
        return MaiaEngine(engine_path, config)
    elif engine_type == "lczero":
        return LeelaEngine(engine_path, config)
    elif engine_type == "human_stockfish":
        return HumanStockfishEngine(engine_path, config)
    else:
        raise HumanEngineError(f"Unknown human engine type: {engine_type}")
