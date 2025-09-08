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
from typing import Optional, Dict, Any, Union
import logging
import json
import time

import chess
import chess.engine as chess_engine

from .models import Config

# Forward reference for type hints
AdaptiveEngine = None  # Will be imported when needed to avoid circular imports

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
        self._effective_elo: Optional[int] = None
        self._engine_name: str = "Unknown"
        self._engine_version: str = "Unknown"
        self._engine_options: Dict[str, Any] = {}
        self._supported_elo_range: Optional[tuple] = None
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
            self._engine_name = self._engine.id.get("name", "Unknown Engine")
            self._engine_version = extract_engine_version(self._engine_name)
            logger.info(f"Engine name: {self._engine_name}")

            # Get engine capabilities and supported options
            await self._detect_engine_capabilities()

            # Log engine details to the report file
            await self._log_engine_report()

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

        Raises:
            EngineError: If the requested ELO is outside the supported range
        """
        if not self._engine:
            raise EngineError("Engine not started")

        # Warn about ELO values below absolute minimum
        MIN_ELO = 600
        if elo < MIN_ELO:
            logger.warning(f"⚠️  Requested ELO {elo} is below absolute minimum of {MIN_ELO}")
            logger.warning(f"   Engine will attempt to configure at ELO {elo} but results may be unpredictable")
            logger.warning(f"   Consider using ELO {MIN_ELO} or higher for reliable chess gameplay")

        if self._current_elo == elo:
            return  # Already configured for this ELO

        # Check if the requested ELO is in supported range
        # For sub-1100 ELOs, we'll use skill levels, so be more permissive
        if self._supported_elo_range:
            min_elo, max_elo = self._supported_elo_range
            if elo > max_elo:
                raise EngineError(
                    f"Requested ELO {elo} is above maximum supported {max_elo} "
                    f"for engine {self._engine_name}"
                )
            elif elo < min_elo and elo < 400:
                # Only reject extremely low ELOs that are unrealistic
                raise EngineError(
                    f"Requested ELO {elo} is below minimum realistic rating "
                    f"for engine {self._engine_name}"
                )
            elif elo < min_elo:
                logger.info(f"ELO {elo} is below native UCI_Elo range ({min_elo}-{max_elo}), will use skill levels")

        effective_elo = elo
        try:
            # Try UCI_LimitStrength first (Stockfish standard)
            await asyncio.to_thread(
                self._engine.configure,
                {"UCI_LimitStrength": True, "UCI_Elo": elo}
            )
            self._current_elo = elo
            self._effective_elo = elo
            logger.debug(f"Configured engine for ELO {elo} using UCI_LimitStrength")

        except chess_engine.EngineError:
            # Fallback to Skill Level for engines that don't support UCI_Elo or for sub-1100 ELOs
            try:
                # Enhanced skill level mapping for better sub-1100 ELO differentiation
                if elo < 600:
                    skill_level = 0
                elif elo < 800:
                    skill_level = max(1, (elo - 500) // 50)  # 1-6 range
                elif elo < 1100:
                    skill_level = max(6, (elo - 600) // 40)  # 6-12 range
                else:  # 1100+ range
                    skill_level = max(12, min(20, (elo - 900) // 30))  # 12-20 range
                await asyncio.to_thread(
                    self._engine.configure,
                    {"Skill Level": skill_level}
                )
                self._current_elo = elo
                # Approximate the effective ELO based on skill level using reverse mapping
                if skill_level == 0:
                    self._effective_elo = 500  # Very low ELO
                elif skill_level <= 6:
                    self._effective_elo = 500 + skill_level * 50  # 550-800 range
                elif skill_level <= 12:
                    self._effective_elo = 600 + (skill_level - 6) * 40  # 640-840 range, adjusted to ~800-1100
                else:
                    self._effective_elo = 900 + (skill_level - 12) * 30  # 900+ range
                effective_elo = self._effective_elo
                logger.debug(f"Configured engine for ELO {elo} using Skill Level {skill_level} (effective: ~{effective_elo})")

            except Exception as e:
                logger.warning(f"Could not configure engine strength: {e}")
                # Continue without strength limitation
                self._current_elo = elo
                self._effective_elo = None

        # Update engine report with the newly configured ELO
        await self._update_engine_report(requested_elo=elo, effective_elo=effective_elo)

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
            # Use fixed time per move for consistent strength
            move_time = self.config.think_time
            logger.debug(f"Engine thinking time: {move_time:.3f}s")

            result = await asyncio.to_thread(
                self._engine.play,
                board,
                chess_engine.Limit(time=move_time)
            )

            if not result.move:
                raise EngineError("Engine returned no move")

            logger.debug(f"Engine selected move: {result.move.uci()}")
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

    @property
    def effective_elo(self) -> Optional[int]:
        """Effective ELO strength after configuration, if known."""
        return self._effective_elo

    @property
    def engine_name(self) -> str:
        """Name of the engine being used."""
        return self._engine_name

    @property
    def engine_type(self) -> str:
        """Type classification of the engine (stockfish, maia, etc.)."""
        name_lower = self._engine_name.lower()
        if "stockfish" in name_lower:
            return "stockfish"
        elif "maia" in name_lower:
            return "maia"
        elif "lc0" in name_lower or "leela" in name_lower:
            return "lczero"
        elif "texel" in name_lower:
            return "texel"
        elif "madchess" in name_lower:
            return "madchess"
        elif "toga" in name_lower:
            return "toga"
        return "unknown"

    async def _detect_engine_capabilities(self) -> None:
        """
        Detect engine capabilities including supported ELO ranges.
        """
        if not self._engine:
            return

        try:
            # Get engine options
            options = self._engine.options
            self._engine_options = {name: option for name, option in options.items()}

            # Determine supported ELO range
            if "UCI_LimitStrength" in options and "UCI_Elo" in options:
                uci_elo = options["UCI_Elo"]
                if hasattr(uci_elo, "min") and hasattr(uci_elo, "max"):
                    self._supported_elo_range = (uci_elo.min, uci_elo.max)
                    logger.info(f"Engine supports ELO range: {uci_elo.min}-{uci_elo.max}")
            elif "Skill Level" in options:
                # Approximate ELO range based on skill level
                skill_option = options["Skill Level"]
                if hasattr(skill_option, "min") and hasattr(skill_option, "max"):
                    min_elo = 1000 + skill_option.min * 50
                    max_elo = 1000 + skill_option.max * 50
                    self._supported_elo_range = (min_elo, max_elo)
                    logger.info(f"Engine supports approximate ELO range: {min_elo}-{max_elo} (via Skill Level)")

            # For engine-specific capabilities
            if "maia" in self._engine_name.lower():
                # Maia models are trained on specific ELO bands
                maia_level = extract_maia_level(self._engine_name)
                if maia_level:
                    band_min = maia_level - 100
                    band_max = maia_level + 100
                    self._supported_elo_range = (band_min, band_max)
                    logger.info(f"Maia model trained on {maia_level} ELO band (effective range: {band_min}-{band_max})")

        except Exception as e:
            logger.warning(f"Failed to detect engine capabilities: {e}")

    async def _log_engine_report(self) -> None:
        """
        Log detailed engine information to a report file.
        """
        try:
            # Create report data
            report = {
                "engine": {
                    "name": self._engine_name,
                    "version": self._engine_version,
                    "type": self.engine_type,
                    "path": self.engine_path,
                    "supported_elo_range": list(self._supported_elo_range) if self._supported_elo_range else None,
                    "options": [{
                        "name": name,
                        "type": getattr(option, "type", "unknown"),
                        "default": getattr(option, "default", None),
                        "min": getattr(option, "min", None) if hasattr(option, "min") else None,
                        "max": getattr(option, "max", None) if hasattr(option, "max") else None,
                    } for name, option in self._engine_options.items()]
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "configurations": []
            }

            # Ensure the meta directory exists
            meta_dir = Path(self.config.output_dir) / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)

            # Save the report
            report_file = meta_dir / "opponent_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Engine report saved to {report_file}")

        except Exception as e:
            logger.warning(f"Failed to save engine report: {e}")

    async def _update_engine_report(self, requested_elo: int, effective_elo: Optional[int] = None) -> None:
        """
        Update the engine report with new ELO configuration.

        Args:
            requested_elo: The ELO rating that was requested
            effective_elo: The actual effective ELO (may be different from requested)
        """
        try:
            meta_dir = Path(self.config.output_dir) / "meta"
            report_file = meta_dir / "opponent_report.json"

            if not report_file.exists():
                await self._log_engine_report()

            # Load existing report
            with open(report_file, 'r') as f:
                report = json.load(f)

            # Add new configuration
            config_entry = {
                "requested_elo": requested_elo,
                "effective_elo": effective_elo or requested_elo,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "options_used": {}
            }

            # Add key configuration options used
            if "UCI_LimitStrength" in self._engine_options:
                config_entry["options_used"]["UCI_LimitStrength"] = True
            if "UCI_Elo" in self._engine_options:
                config_entry["options_used"]["UCI_Elo"] = requested_elo
            if "Skill Level" in self._engine_options and effective_elo and effective_elo != requested_elo:
                skill_level = max(0, min(20, (requested_elo - 1000) // 50))
                config_entry["options_used"]["Skill Level"] = skill_level

            # Add the configuration to the report
            report["configurations"].append(config_entry)

            # Save the updated report
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to update engine report: {e}")

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


def create_engine(config: Config, engine_path: Optional[str] = None,
                  engine_type: Optional[str] = None) -> Union[ChessEngine, 'AdaptiveEngine']:
    """
    Factory function to create the appropriate chess engine based on configuration.

    This function creates either a standard ChessEngine, a human-like engine, or
    an AdaptiveEngine that can switch between engines based on ELO rating.

    Args:
        config: Global configuration
        engine_path: Optional explicit path to engine executable
        engine_type: Optional engine type override (stockfish, maia, texel, etc.)

    Returns:
        Configured chess engine instance
    """
    # Import here to avoid circular imports
    from .adaptive_engine import AdaptiveEngine

    # If specific opponent type is specified, use that
    if engine_type:
        if engine_type == "stockfish":
            stockfish_path = engine_path or autodetect_stockfish(config.stockfish_path)
            if not stockfish_path:
                raise EngineError("Stockfish not found. Please install it or specify the path.")
            return ChessEngine(stockfish_path, config)
        elif engine_type in ("maia", "lczero", "human_stockfish"):
            from .human_engine import get_best_human_engine, create_human_engine
            # Either use provided path or auto-detect
            if engine_path:
                if not Path(engine_path).exists():
                    raise EngineError(f"{engine_type.capitalize()} engine not found at specified path: {engine_path}")
                return create_human_engine(engine_type, engine_path, config)
            else:
                best_engine = get_best_human_engine(preferred_type=engine_type)
                if best_engine:
                    detected_type, detected_path = best_engine
                    if not detected_path:
                        raise EngineError(f"{engine_type.capitalize()} engine detected but path is invalid")
                    return create_human_engine(detected_type, detected_path, config)
                raise EngineError(f"{engine_type.capitalize()} engine not found. Please install it or specify the path.")
        elif engine_type in ("texel", "madchess", "toga"):
            # Try to find the specified low-ELO engine
            detected_path = autodetect_engine(engine_type)
            if detected_path and Path(detected_path).exists():
                return ChessEngine(detected_path, config)
            raise EngineError(f"{engine_type.capitalize()} engine not found. Please install it or specify the path.")
        else:
            logger.warning(f"Unknown engine type '{engine_type}', falling back to standard behavior")

    # If adaptive_elo_engines is enabled, create an adaptive engine
    if getattr(config, 'adaptive_elo_engines', True):
        try:
            engine = AdaptiveEngine(config)
            return engine
        except Exception as e:
            logger.warning(f"Failed to create adaptive engine: {e}. Falling back to standard engine.")

    # Otherwise create a standard engine (either human-like or stockfish)
    if config.use_human_engine:
        from .human_engine import get_best_human_engine, create_human_engine

        # Auto-detect best human engine
        best_engine = get_best_human_engine()
        if best_engine:
            engine_type, detected_path = best_engine
            path_to_use = engine_path or detected_path
            if not path_to_use:
                raise EngineError("Human engine detected but path is invalid")
            return create_human_engine(engine_type, path_to_use, config)

    # Fallback to regular Stockfish
    stockfish_path = engine_path or autodetect_stockfish(config.stockfish_path)
    if not stockfish_path:
        raise EngineError("Stockfish not found. Please install it or specify the path.")

    if not Path(stockfish_path).exists():
        raise EngineError(f"Stockfish not found at path: {stockfish_path}")

    return ChessEngine(stockfish_path, config)


def autodetect_engine(engine_name: str) -> Optional[str]:
    """
    Auto-detect installation path for a specific chess engine.

    Args:
        engine_name: Name of the engine to detect

    Returns:
        Path to the engine executable if found, None otherwise
    """
    # Check environment variable
    env_var = f"{engine_name.upper()}_PATH"
    env_path = os.getenv(env_var)
    if env_path and Path(env_path).exists():
        return env_path

    # Check in system PATH
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
        f"/usr/games/{engine_name}",
        f"{os.path.expanduser('~')}/.local/bin/{engine_name}",
        f"{os.path.expanduser('~')}/chess_engines/{engine_name}/{engine_name}"
    ]

    for path in common_paths:
        if Path(path).exists():
            return path

    return None


def extract_engine_version(engine_name: str) -> str:
    """
    Extract version information from engine name.

    Args:
        engine_name: Full engine name string

    Returns:
        Version string if found, empty string otherwise
    """
    import re

    # Common version patterns
    patterns = [
        r'(\d+\.\d+\.\d+)',  # Format: X.Y.Z
        r'(\d+\.\d+)',       # Format: X.Y
        r'v(\d+\.\d+\.\d+)', # Format: vX.Y.Z
        r'v(\d+\.\d+)',      # Format: vX.Y
        r'(\d+)'             # Just a number
    ]

    for pattern in patterns:
        match = re.search(pattern, engine_name)
        if match:
            return match.group(1)

    return "Unknown"


def extract_maia_level(engine_name: str) -> Optional[int]:
    """
    Extract the ELO level from a Maia engine name.

    Args:
        engine_name: Full engine name string

    Returns:
        ELO level if found, None otherwise
    """
    import re

    # Patterns for Maia model names (e.g., "maia-1900")
    patterns = [
        r'maia[_-](\d{3,4})',  # Format: maia-1900 or maia_1900
        r'(\d{3,4})'           # Just look for a 3-4 digit number
    ]

    for pattern in patterns:
        match = re.search(pattern, engine_name.lower())
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass

    return None


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
