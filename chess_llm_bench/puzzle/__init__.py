"""
Chess Puzzle Solving Framework

This module provides a comprehensive framework for testing LLMs on various chess puzzles:
- Tactics: Mate-in-N and tactical motifs with single best moves
- Endgames: Theoretical endgame positions with tablebase verification
- Blunder-avoidance: Critical positions where major evaluation drops exist
- Gamelets: Opening positions with forced sequences requiring precise evaluation

The puzzle system tracks detailed performance metrics and provides
granular analysis of LLM chess understanding across different domains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import chess


class PuzzleType(Enum):
    """Types of chess puzzles supported by the framework."""

    TACTIC = "tactic"           # Mate-in-N, pins, forks, skewers, etc.
    ENDGAME = "endgame"         # Theoretical endgame positions
    BLUNDER_AVOID = "blunder"   # Avoid major evaluation drops
    GAMELET = "gamelet"         # Opening sequences with forced continuations


class TacticMotif(Enum):
    """Common tactical motifs and themes."""

    # Mate patterns
    MATE_IN_1 = "mate_in_1"
    MATE_IN_2 = "mate_in_2"
    MATE_IN_3 = "mate_in_3"

    # Basic tactics
    FORK = "fork"
    PIN = "pin"
    SKEWER = "skewer"
    DISCOVERED_ATTACK = "discovered_attack"
    DOUBLE_ATTACK = "double_attack"

    # Advanced tactics
    DEFLECTION = "deflection"
    DECOY = "decoy"
    CLEARANCE = "clearance"
    INTERFERENCE = "interference"
    ZUGZWANG = "zugzwang"

    # Sacrificial themes
    SACRIFICE = "sacrifice"
    GREEK_GIFT = "greek_gift"
    SMOTHERED_MATE = "smothered_mate"


class EndgameType(Enum):
    """Standard endgame classifications."""

    # King + Queen
    KQK = "KQK"           # King + Queen vs King
    KQKP = "KQKP"         # King + Queen vs King + Pawn

    # King + Rook
    KRK = "KRK"           # King + Rook vs King
    KRKP = "KRKP"         # King + Rook vs King + Pawn
    KRRKR = "KRRKR"       # King + 2 Rooks vs King + Rook

    # King + Bishops/Knights
    KBBK = "KBBK"         # King + 2 Bishops vs King
    KBNK = "KBNK"         # King + Bishop + Knight vs King
    KNNK = "KNNK"         # King + 2 Knights vs King

    # Pawn endgames
    KPK = "KPK"           # King + Pawn vs King
    KPPKP = "KPPKP"       # King + 2 Pawns vs King + Pawn

    # Mixed endgames
    KRBKR = "KRBKR"       # King + Rook + Bishop vs King + Rook
    KRNKR = "KRNKR"       # King + Rook + Knight vs King + Rook


@dataclass
class PuzzlePosition:
    """A chess puzzle position with metadata."""

    fen: str                                    # Board position
    puzzle_type: PuzzleType                     # Type of puzzle
    difficulty: int                             # Difficulty rating (1-10)
    best_move: str                              # Best UCI move
    evaluation: Optional[float] = None          # Engine evaluation in centipawns
    mate_in: Optional[int] = None              # Mate in N moves (if applicable)

    # Metadata
    title: str = ""                            # Human-readable title
    description: str = ""                      # Puzzle description
    source: str = ""                           # Source database/book
    tags: List[str] = field(default_factory=list)  # Additional tags

    # Type-specific data
    tactic_motif: Optional[TacticMotif] = None      # For tactical puzzles
    endgame_type: Optional[EndgameType] = None      # For endgame puzzles
    blunder_threshold: Optional[int] = None         # CP drop threshold for blunders
    opening_name: Optional[str] = None              # Opening name for gamelets

    def __post_init__(self):
        """Validate puzzle position after initialization."""
        try:
            board = chess.Board(self.fen)
        except ValueError as e:
            raise ValueError(f"Invalid FEN: {self.fen} - {e}")

        if not self.best_move:
            raise ValueError("Best move cannot be empty")

        # Validate move format (accept both UCI and algebraic notation)
        if len(self.best_move) < 2:
            raise ValueError(f"Invalid move format: {self.best_move}")

        # Try to validate the move is reasonable (basic format check)
        move_str = self.best_move.strip()
        if not move_str or not any(c.isalnum() for c in move_str):
            raise ValueError(f"Invalid move format: {self.best_move}")


@dataclass
class PuzzleAttempt:
    """Record of an LLM's attempt at solving a puzzle."""

    puzzle_id: str                          # Unique puzzle identifier
    llm_move: str                           # LLM's chosen move (UCI)
    response_time: float                    # Time taken to respond (seconds)
    is_correct: bool                        # Whether move matches best_move
    evaluation_delta: Optional[float] = None # Evaluation difference from best move
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Additional attempt metadata
    raw_response: str = ""                  # Full LLM response
    reasoning: str = ""                     # Extracted reasoning (if any)
    confidence: Optional[float] = None      # LLM confidence score (if available)
    illegal_attempts: int = 0               # Number of illegal moves tried first


@dataclass
class PuzzleStats:
    """Statistics for puzzle solving performance."""

    # Overall performance
    total_attempts: int = 0
    correct_solutions: int = 0
    total_response_time: float = 0.0
    total_illegal_moves: int = 0

    # Performance by puzzle type
    tactic_stats: Dict[TacticMotif, int] = field(default_factory=dict)
    endgame_stats: Dict[EndgameType, int] = field(default_factory=dict)
    difficulty_stats: Dict[int, int] = field(default_factory=dict)

    # Timing statistics
    fastest_solve: float = float('inf')
    slowest_solve: float = 0.0

    @property
    def success_rate(self) -> float:
        """Overall success rate (0.0 to 1.0)."""
        if self.total_attempts == 0:
            return 0.0
        return self.correct_solutions / self.total_attempts

    @property
    def average_response_time(self) -> float:
        """Average time per puzzle attempt."""
        if self.total_attempts == 0:
            return 0.0
        return self.total_response_time / self.total_attempts

    def add_attempt(self, attempt: PuzzleAttempt) -> None:
        """Add a puzzle attempt and update statistics."""
        self.total_attempts += 1
        self.total_response_time += attempt.response_time
        self.total_illegal_moves += attempt.illegal_attempts

        if attempt.is_correct:
            self.correct_solutions += 1

        # Update timing records
        self.fastest_solve = min(self.fastest_solve, attempt.response_time)
        self.slowest_solve = max(self.slowest_solve, attempt.response_time)

    def get_motif_success_rate(self, motif: TacticMotif) -> float:
        """Get success rate for a specific tactical motif."""
        if motif not in self.tactic_stats:
            return 0.0

        total = sum(1 for k in self.tactic_stats.keys() if k == motif)
        correct = self.tactic_stats.get(motif, 0)
        return correct / total if total > 0 else 0.0

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_attempts = 0
        self.correct_solutions = 0
        self.total_response_time = 0.0
        self.total_illegal_moves = 0
        self.tactic_stats.clear()
        self.endgame_stats.clear()
        self.difficulty_stats.clear()
        self.fastest_solve = float('inf')
        self.slowest_solve = 0.0


@dataclass
class PuzzleSet:
    """A collection of related puzzles."""

    name: str                               # Set name
    description: str                        # Set description
    puzzles: List[PuzzlePosition] = field(default_factory=list)
    puzzle_type: Optional[PuzzleType] = None # Primary puzzle type (if homogeneous)
    difficulty_range: Tuple[int, int] = (1, 10)  # Min/max difficulty
    source: str = ""                        # Original source

    def add_puzzle(self, puzzle: PuzzlePosition) -> None:
        """Add a puzzle to this set."""
        self.puzzles.append(puzzle)

    def filter_by_difficulty(self, min_diff: int, max_diff: int) -> List[PuzzlePosition]:
        """Get puzzles within difficulty range."""
        return [p for p in self.puzzles if min_diff <= p.difficulty <= max_diff]

    def filter_by_type(self, puzzle_type: PuzzleType) -> List[PuzzlePosition]:
        """Get puzzles of specific type."""
        return [p for p in self.puzzles if p.puzzle_type == puzzle_type]

    def shuffle(self) -> None:
        """Randomly shuffle puzzle order."""
        import random
        random.shuffle(self.puzzles)


# Export main classes and enums for easy importing
__all__ = [
    'PuzzleType',
    'TacticMotif',
    'EndgameType',
    'PuzzlePosition',
    'PuzzleAttempt',
    'PuzzleStats',
    'PuzzleSet'
]
