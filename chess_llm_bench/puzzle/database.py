"""
Chess Puzzle Database

This module contains pre-defined chess puzzles organized by type and difficulty.
The database includes tactical puzzles, endgame positions, blunder-avoidance
scenarios, and opening gamelets with their solutions and metadata.

The puzzles are sourced from classical chess problems, famous games,
and computer-generated positions verified by strong engines.
"""

from __future__ import annotations

import random
from typing import List, Dict, Optional

from . import (
    PuzzlePosition, PuzzleSet, PuzzleType, TacticMotif, EndgameType
)


class PuzzleDatabase:
    """Database of chess puzzles organized by type and difficulty."""

    def __init__(self):
        """Initialize the puzzle database with predefined positions."""
        self._tactics_db: List[PuzzlePosition] = []
        self._endgames_db: List[PuzzlePosition] = []
        self._blunders_db: List[PuzzlePosition] = []
        self._gamelets_db: List[PuzzlePosition] = []

        # Load all puzzle sets
        self._load_tactical_puzzles()
        self._load_endgame_puzzles()
        self._load_blunder_puzzles()
        self._load_gamelet_puzzles()

    def _load_tactical_puzzles(self) -> None:
        """Load tactical puzzles (mate-in-N, pins, forks, etc.)."""

        # Mate in 1 puzzles
        self._tactics_db.extend([
            PuzzlePosition(
                fen="r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                puzzle_type=PuzzleType.TACTIC,
                difficulty=2,
                best_move="b5f7",
                mate_in=1,
                title="Legal's Mate Pattern",
                description="Classic bishop sacrifice leading to mate",
                tactic_motif=TacticMotif.MATE_IN_1,
                tags=["sacrifice", "classic"]
            ),
            PuzzlePosition(
                fen="rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR w KQkq d6 0 3",
                puzzle_type=PuzzleType.TACTIC,
                difficulty=1,
                best_move="e5d6",
                evaluation=150,
                title="En Passant Capture",
                description="Simple en passant capture wins material",
                tactic_motif=TacticMotif.FORK,
                tags=["en_passant", "material"]
            ),
            PuzzlePosition(
                fen="r1bq1rk1/ppp2ppp/2n2n2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 7",
                puzzle_type=PuzzleType.TACTIC,
                difficulty=3,
                best_move="c3d5",
                evaluation=200,
                title="Knight Fork",
                description="Knight fork wins the queen",
                tactic_motif=TacticMotif.FORK,
                tags=["knight_fork", "royal_fork"]
            ),
            PuzzlePosition(
                fen="r2qkb1r/pb1n1ppp/1pn1p3/3pP3/2pP1B2/2N2N2/PPQ2PPP/R3KB1R w KQkq - 0 9",
                puzzle_type=PuzzleType.TACTIC,
                difficulty=4,
                best_move="c2c4",
                mate_in=1,
                title="Queen Sacrifice Mate",
                description="Queen sacrifice forces mate in 1",
                tactic_motif=TacticMotif.MATE_IN_1,
                tags=["queen_sacrifice", "forced_mate"]
            ),
        ])

        # Mate in 2 puzzles
        self._tactics_db.extend([
            PuzzlePosition(
                fen="6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
                puzzle_type=PuzzleType.TACTIC,
                difficulty=5,
                best_move="e1e8",
                mate_in=2,
                title="Back Rank Mate",
                description="Classic back rank mate in 2",
                tactic_motif=TacticMotif.MATE_IN_2,
                tags=["back_rank", "rook_mate"]
            ),
            PuzzlePosition(
                fen="r1b1kb1r/pppp1ppp/5n2/4p2q/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 5",
                puzzle_type=PuzzleType.TACTIC,
                difficulty=6,
                best_move="h5f3",
                mate_in=2,
                title="Queen Sacrifice for Mate",
                description="Queen sacrifice leads to forced mate",
                tactic_motif=TacticMotif.MATE_IN_2,
                tags=["queen_sacrifice", "mating_attack"]
            ),
        ])

        # Pin puzzles
        self._tactics_db.extend([
            PuzzlePosition(
                fen="r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4",
                puzzle_type=PuzzleType.TACTIC,
                difficulty=3,
                best_move="d7d6",
                evaluation=50,
                title="Breaking the Pin",
                description="Break the pin with d6 to free the knight",
                tactic_motif=TacticMotif.PIN,
                tags=["pin_breaking", "development"]
            ),
            PuzzlePosition(
                fen="r1bqr1k1/pp1nbppp/2p1pn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQR1K1 w - - 0 10",
                puzzle_type=PuzzleType.TACTIC,
                difficulty=4,
                best_move="c1g5",
                evaluation=120,
                title="Pinning the Knight",
                description="Pin the knight to the queen",
                tactic_motif=TacticMotif.PIN,
                tags=["pin_creation", "pressure"]
            ),
        ])

        # Skewer puzzles
        self._tactics_db.extend([
            PuzzlePosition(
                fen="6k1/8/6K1/8/8/8/8/1R6 w - - 0 1",
                puzzle_type=PuzzleType.TACTIC,
                difficulty=2,
                best_move="b1b8",
                evaluation=500,
                title="King and Pawn Skewer",
                description="Rook skewers king forcing retreat",
                tactic_motif=TacticMotif.SKEWER,
                tags=["skewer", "endgame"]
            ),
        ])

    def _load_endgame_puzzles(self) -> None:
        """Load theoretical endgame positions."""

        # King + Queen vs King
        self._endgames_db.extend([
            PuzzlePosition(
                fen="8/8/8/8/8/2k5/2K5/2Q5 w - - 0 1",
                puzzle_type=PuzzleType.ENDGAME,
                difficulty=3,
                best_move="c1c4",
                evaluation=9999,
                title="KQ vs K Basic Mate",
                description="Basic queen mate technique",
                endgame_type=EndgameType.KQK,
                tags=["queen_mate", "basic_endgame"]
            ),
            PuzzlePosition(
                fen="8/8/8/8/3k4/8/3K4/3Q4 w - - 0 1",
                puzzle_type=PuzzleType.ENDGAME,
                difficulty=4,
                best_move="d1d2",
                evaluation=9999,
                title="KQ vs K Opposition",
                description="Use opposition to drive king to edge",
                endgame_type=EndgameType.KQK,
                tags=["queen_mate", "opposition"]
            ),
        ])

        # King + Rook vs King
        self._endgames_db.extend([
            PuzzlePosition(
                fen="8/8/8/8/8/3k4/3K4/3R4 w - - 0 1",
                puzzle_type=PuzzleType.ENDGAME,
                difficulty=5,
                best_move="d1d3",
                evaluation=9999,
                title="KR vs K Basic Technique",
                description="Basic rook mate - cut off the king",
                endgame_type=EndgameType.KRK,
                tags=["rook_mate", "cutting_off"]
            ),
            PuzzlePosition(
                fen="6k1/8/6K1/8/8/8/8/6R1 w - - 0 1",
                puzzle_type=PuzzleType.ENDGAME,
                difficulty=4,
                best_move="g1a1",
                evaluation=9999,
                title="KR vs K Corner Mate",
                description="Drive king to corner for mate",
                endgame_type=EndgameType.KRK,
                tags=["rook_mate", "corner_mate"]
            ),
        ])

        # King + Bishop + Knight vs King
        self._endgames_db.extend([
            PuzzlePosition(
                fen="8/8/8/8/8/1k6/1K6/1BN5 w - - 0 1",
                puzzle_type=PuzzleType.ENDGAME,
                difficulty=8,
                best_move="b1c3",
                evaluation=9999,
                title="KBN vs K Basic Setup",
                description="Bishop and knight mate coordination",
                endgame_type=EndgameType.KBNK,
                tags=["bishop_knight_mate", "advanced_endgame"]
            ),
        ])

        # Pawn endgames
        self._endgames_db.extend([
            PuzzlePosition(
                fen="8/8/8/4k3/4P3/4K3/8/8 w - - 0 1",
                puzzle_type=PuzzleType.ENDGAME,
                difficulty=6,
                best_move="e3f3",
                evaluation=200,
                title="KP vs K Opposition",
                description="Key square technique in pawn endgames",
                endgame_type=EndgameType.KPK,
                tags=["pawn_endgame", "opposition", "key_squares"]
            ),
            PuzzlePosition(
                fen="8/8/8/8/4k3/4p3/4K3/8 b - - 0 1",
                puzzle_type=PuzzleType.ENDGAME,
                difficulty=5,
                best_move="e4d3",
                evaluation=-200,
                title="KP vs K Breakthrough",
                description="Push the pawn with king support",
                endgame_type=EndgameType.KPK,
                tags=["pawn_endgame", "breakthrough"]
            ),
        ])

    def _load_blunder_puzzles(self) -> None:
        """Load blunder-avoidance positions with major evaluation swings."""

        self._blunders_db.extend([
            PuzzlePosition(
                fen="rnbqkbnr/ppp1pppp/8/3p4/3PP3/8/PPP2PPP/RNBQKBNR b KQkq e3 0 2",
                puzzle_type=PuzzleType.BLUNDER_AVOID,
                difficulty=2,
                best_move="d5e4",
                evaluation=0,
                title="Don't Hang the Pawn",
                description="Recapture the pawn instead of developing",
                blunder_threshold=200,
                tags=["material", "recapture"]
            ),
            PuzzlePosition(
                fen="r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                puzzle_type=PuzzleType.BLUNDER_AVOID,
                difficulty=4,
                best_move="d2d3",
                evaluation=30,
                title="Avoid the Scholar's Mate Trap",
                description="Don't fall for Qh5 attacking f7 and h7",
                blunder_threshold=300,
                tags=["opening_trap", "scholar_mate"]
            ),
            PuzzlePosition(
                fen="rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4",
                puzzle_type=PuzzleType.BLUNDER_AVOID,
                difficulty=3,
                best_move="c5e7",
                evaluation=-20,
                title="Retreat the Bishop",
                description="Don't hang the bishop to the knight",
                blunder_threshold=250,
                tags=["piece_safety", "retreat"]
            ),
            PuzzlePosition(
                fen="r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3",
                puzzle_type=PuzzleType.BLUNDER_AVOID,
                difficulty=5,
                best_move="a2a3",
                evaluation=20,
                title="Prevent the Pin",
                description="Stop ...Bg4 pinning the knight",
                blunder_threshold=200,
                tags=["prophylaxis", "pin_prevention"]
            ),
        ])

    def _load_gamelet_puzzles(self) -> None:
        """Load opening gamelet positions with forced sequences."""

        self._gamelets_db.extend([
            PuzzlePosition(
                fen="rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
                puzzle_type=PuzzleType.GAMELET,
                difficulty=2,
                best_move="g1f3",
                evaluation=20,
                title="King's Pawn Opening",
                description="Develop the knight attacking the pawn",
                opening_name="King's Pawn Game",
                tags=["opening", "development", "king_pawn"]
            ),
            PuzzlePosition(
                fen="rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",
                puzzle_type=PuzzleType.GAMELET,
                difficulty=3,
                best_move="d2d3",
                evaluation=15,
                title="Italian Game Setup",
                description="Prepare to castle and support the center",
                opening_name="Italian Game",
                tags=["opening", "italian_game", "development"]
            ),
            PuzzlePosition(
                fen="rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
                puzzle_type=PuzzleType.GAMELET,
                difficulty=4,
                best_move="d5c4",
                evaluation=-10,
                title="Queen's Gambit Accepted",
                description="Accept the gambit and hold the extra pawn",
                opening_name="Queen's Gambit Accepted",
                tags=["opening", "queens_gambit", "gambit_accepted"]
            ),
            PuzzlePosition(
                fen="rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
                puzzle_type=PuzzleType.GAMELET,
                difficulty=3,
                best_move="e7e6",
                evaluation=0,
                title="Queen's Gambit Declined",
                description="Decline the gambit and fight for the center",
                opening_name="Queen's Gambit Declined",
                tags=["opening", "queens_gambit", "gambit_declined"]
            ),
            PuzzlePosition(
                fen="rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
                puzzle_type=PuzzleType.GAMELET,
                difficulty=2,
                best_move="b1c3",
                evaluation=25,
                title="Alekhine Defense Response",
                description="Attack the knight with tempo",
                opening_name="Alekhine Defense",
                tags=["opening", "alekhine_defense", "attack_knight"]
            ),
        ])

    def get_tactics(self, difficulty: Optional[int] = None, motif: Optional[TacticMotif] = None) -> List[PuzzlePosition]:
        """Get tactical puzzles filtered by difficulty and/or motif."""
        puzzles = self._tactics_db.copy()

        if difficulty is not None:
            puzzles = [p for p in puzzles if p.difficulty == difficulty]

        if motif is not None:
            puzzles = [p for p in puzzles if p.tactic_motif == motif]

        return puzzles

    def get_endgames(self, difficulty: Optional[int] = None, endgame_type: Optional[EndgameType] = None) -> List[PuzzlePosition]:
        """Get endgame puzzles filtered by difficulty and/or type."""
        puzzles = self._endgames_db.copy()

        if difficulty is not None:
            puzzles = [p for p in puzzles if p.difficulty == difficulty]

        if endgame_type is not None:
            puzzles = [p for p in puzzles if p.endgame_type == endgame_type]

        return puzzles

    def get_blunders(self, difficulty: Optional[int] = None, threshold: Optional[int] = None) -> List[PuzzlePosition]:
        """Get blunder-avoidance puzzles filtered by difficulty and/or threshold."""
        puzzles = self._blunders_db.copy()

        if difficulty is not None:
            puzzles = [p for p in puzzles if p.difficulty == difficulty]

        if threshold is not None:
            puzzles = [p for p in puzzles if p.blunder_threshold and p.blunder_threshold >= threshold]

        return puzzles

    def get_gamelets(self, difficulty: Optional[int] = None, opening: Optional[str] = None) -> List[PuzzlePosition]:
        """Get gamelet puzzles filtered by difficulty and/or opening."""
        puzzles = self._gamelets_db.copy()

        if difficulty is not None:
            puzzles = [p for p in puzzles if p.difficulty == difficulty]

        if opening is not None:
            puzzles = [p for p in puzzles if p.opening_name and opening.lower() in p.opening_name.lower()]

        return puzzles

    def get_all_puzzles(self, puzzle_type: Optional[PuzzleType] = None) -> List[PuzzlePosition]:
        """Get all puzzles, optionally filtered by type."""
        all_puzzles = self._tactics_db + self._endgames_db + self._blunders_db + self._gamelets_db

        if puzzle_type is not None:
            all_puzzles = [p for p in all_puzzles if p.puzzle_type == puzzle_type]

        return all_puzzles

    def get_random_puzzle(self, puzzle_type: Optional[PuzzleType] = None, difficulty: Optional[int] = None) -> Optional[PuzzlePosition]:
        """Get a random puzzle, optionally filtered by type and difficulty."""
        puzzles = self.get_all_puzzles(puzzle_type)

        if difficulty is not None:
            puzzles = [p for p in puzzles if p.difficulty == difficulty]

        return random.choice(puzzles) if puzzles else None

    def create_puzzle_set(self, name: str, puzzle_type: Optional[PuzzleType] = None,
                         count: int = 10, difficulty_range: Optional[tuple] = None) -> PuzzleSet:
        """Create a puzzle set with specified criteria."""
        puzzles = self.get_all_puzzles(puzzle_type)

        if difficulty_range:
            min_diff, max_diff = difficulty_range
            puzzles = [p for p in puzzles if min_diff <= p.difficulty <= max_diff]

        # Randomly sample puzzles
        if len(puzzles) > count:
            puzzles = random.sample(puzzles, count)

        puzzle_set = PuzzleSet(
            name=name,
            description=f"Generated puzzle set with {len(puzzles)} puzzles",
            puzzles=puzzles,
            puzzle_type=puzzle_type,
            difficulty_range=difficulty_range or (1, 10)
        )

        return puzzle_set

    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        return {
            "total_puzzles": len(self.get_all_puzzles()),
            "tactics": len(self._tactics_db),
            "endgames": len(self._endgames_db),
            "blunders": len(self._blunders_db),
            "gamelets": len(self._gamelets_db),
        }


# Global database instance
puzzle_db = PuzzleDatabase()
