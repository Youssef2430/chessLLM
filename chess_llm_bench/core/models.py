"""
Core data models for the Chess LLM Benchmark.

This module defines the fundamental data structures used throughout the application
for representing bots, games, statistics, and UI state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import chess


@dataclass
class BotSpec:
    """Specification for a chess bot/LLM configuration."""

    provider: str  # "openai", "anthropic", "random", etc.
    model: str     # Model name (empty for random)
    name: str      # Display name for the bot

    def __post_init__(self):
        """Validate bot specification after initialization."""
        if not self.provider:
            raise ValueError("Provider cannot be empty")
        if not self.name:
            raise ValueError("Bot name cannot be empty")
        self.provider = self.provider.lower()

    def __str__(self) -> str:
        return f"{self.name} ({self.provider}:{self.model})" if self.model else f"{self.name} ({self.provider})"


@dataclass
class GameRecord:
    """Record of a completed chess game."""

    elo: int                    # Engine ELO rating for this game
    color_llm_white: bool      # True if LLM played as white
    result: str                # "1-0", "0-1", "1/2-1/2"
    ply_count: int            # Number of half-moves in the game
    path: Path                # Path to saved PGN file
    timestamp: datetime = field(default_factory=datetime.utcnow)
    game_duration: float = 0.0  # Total game duration in seconds including API response times

    @property
    def llm_won(self) -> bool:
        """True if the LLM won this game."""
        if self.result == "1-0":
            return self.color_llm_white
        elif self.result == "0-1":
            return not self.color_llm_white
        return False

    @property
    def llm_lost(self) -> bool:
        """True if the LLM lost this game."""
        if self.result == "1-0":
            return not self.color_llm_white
        elif self.result == "0-1":
            return self.color_llm_white
        return False

    @property
    def is_draw(self) -> bool:
        """True if the game was a draw."""
        return self.result == "1/2-1/2"


@dataclass
class LadderStats:
    """Statistics for a bot's performance on the ELO ladder."""

    max_elo_reached: int = 0
    games: List[GameRecord] = field(default_factory=list)
    losses: int = 0
    draws: int = 0
    wins: int = 0

    # Timing and move quality stats
    total_move_time: float = 0.0
    total_illegal_moves: int = 0
    total_game_duration: float = 0.0  # Total duration of all games in seconds

    @property
    def total_games(self) -> int:
        """Total number of games played."""
        return len(self.games)

    @property
    def win_rate(self) -> float:
        """Win percentage (0.0 to 1.0)."""
        if self.total_games == 0:
            return 0.0
        return self.wins / self.total_games

    @property
    def draw_rate(self) -> float:
        """Draw percentage (0.0 to 1.0)."""
        if self.total_games == 0:
            return 0.0
        return self.draws / self.total_games

    @property
    def loss_rate(self) -> float:
        """Loss percentage (0.0 to 1.0)."""
        if self.total_games == 0:
            return 0.0
        return self.losses / self.total_games

    def add_game(self, game: GameRecord) -> None:
        """Add a game record and update statistics."""
        self.games.append(game)
        self.max_elo_reached = max(self.max_elo_reached, game.elo)

        if game.llm_won:
            self.wins += 1
        elif game.llm_lost:
            self.losses += 1
        else:
            self.draws += 1

        # Add game duration statistics
        self.add_game_duration(game.game_duration)

    def reset(self) -> None:
        """Reset all statistics."""
        self.max_elo_reached = 0
        self.games.clear()
        self.losses = 0
        self.draws = 0
        self.wins = 0
        self.total_move_time = 0.0
        self.total_illegal_moves = 0
        self.total_game_duration = 0.0

    @property
    def average_move_time(self) -> float:
        """Average time per move across all games."""
        total_moves = sum(game.ply_count for game in self.games)
        if total_moves == 0:
            return 0.0
        return self.total_move_time / total_moves

    def add_timing_stats(self, total_time: float, illegal_moves: int) -> None:
        """Add timing and move quality statistics from a completed game."""
        self.total_move_time += total_time
        self.total_illegal_moves += illegal_moves

    def add_game_duration(self, duration: float) -> None:
        """Add game duration statistics from a completed game."""
        self.total_game_duration += duration

    @property
    def average_game_duration(self) -> float:
        """Average duration per game."""
        if self.total_games == 0:
            return 0.0
        return self.total_game_duration / self.total_games


@dataclass
class LiveState:
    """Live state for UI rendering during game execution."""

    title: str
    ladder: List[int] = field(default_factory=list)  # ELO rungs climbed
    current_elo: int = 0
    status: str = "waiting"
    board_ascii: str = ""
    last_move_uci: str = ""
    color_llm_white: bool = True
    moves_made: int = 0
    final_result: Optional[str] = None
    error_message: Optional[str] = None

    # Timing and move tracking
    total_move_time: float = 0.0  # Total time spent generating moves
    average_move_time: float = 0.0  # Average time per move
    illegal_move_attempts: int = 0  # Number of illegal moves attempted
    game_duration: float = 0.0  # Total game duration in seconds

    # Beautiful chess board support
    _chess_board: Optional[chess.Board] = field(default=None, init=False)
    _moves_played: List[chess.Move] = field(default_factory=list, init=False)

    @property
    def current_rung(self) -> int:
        """Current rung number (1-indexed) on the ladder."""
        return len(self.ladder)

    @property
    def ladder_display(self) -> str:
        """Formatted string showing the ladder progression."""
        return " → ".join(map(str, self.ladder)) if self.ladder else "—"

    @property
    def is_finished(self) -> bool:
        """True if the bot has finished its ladder run."""
        return self.final_result is not None or self.error_message is not None

    @property
    def color_display(self) -> str:
        """Human-readable color assignment."""
        return "White" if self.color_llm_white else "Black"

    def update_board_state(self, board_ascii: str, last_move: str, moves_made: int) -> None:
        """Update the current board visualization state."""
        self.board_ascii = board_ascii
        self.last_move_uci = last_move
        self.moves_made = moves_made

    def set_error(self, error: str) -> None:
        """Set an error state for this bot."""
        self.error_message = error
        self.status = f"error: {error}"


@dataclass
class Config:
    """Configuration settings for the chess LLM benchmark."""

    # Bot and game settings
    bots: str = "random::bot1,random::bot2"
    stockfish_path: Optional[str] = None
    opponent_type: Optional[str] = None  # "stockfish", "maia", "texel", "madchess"

    # Human-like engine settings
    use_human_engine: bool = False
    human_engine_type: str = "maia"  # "maia", "lczero", "human_stockfish"
    human_engine_path: Optional[str] = None
    human_engine_fallback: bool = True  # Fall back to stockfish if human engine fails

    # Adaptive engine settings
    adaptive_elo_engines: bool = True  # Use specialized engines for different ELO ranges

    # ELO ladder settings
    start_elo: int = 600
    elo_step: int = 100
    max_elo: int = 2400

    # Game settings
    think_time: float = 0.3
    max_plies: int = 300
    escalate_on: str = "always"  # "always" or "on_win"

    # LLM settings
    llm_timeout: float = 20.0
    llm_temperature: float = 0.0

    # Output settings
    output_dir: str = "runs"
    save_pgn: bool = True

    # UI settings
    refresh_rate: int = 6  # Hz

    # Budget tracking settings
    budget_limit: Optional[float] = None
    show_costs: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Config:
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class BenchmarkResult:
    """Final results of a complete benchmark run."""

    run_id: str
    timestamp: datetime
    config: Config
    bot_results: Dict[str, LadderStats]
    output_dir: Path

    @property
    def total_games(self) -> int:
        """Total games played across all bots."""
        return sum(stats.total_games for stats in self.bot_results.values())

    @property
    def best_bot(self) -> Optional[str]:
        """Name of the bot that reached the highest ELO."""
        if not self.bot_results:
            return None
        return max(self.bot_results.keys(), key=lambda name: self.bot_results[name].max_elo_reached)

    @property
    def best_elo(self) -> int:
        """Highest ELO reached by any bot."""
        if not self.bot_results:
            return 0
        return max(stats.max_elo_reached for stats in self.bot_results.values())
