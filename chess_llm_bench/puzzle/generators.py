"""
Chess Puzzle Generators

This module provides utilities for generating chess puzzles from various external
sources and databases. It can create puzzle sets from:

- Lichess puzzle database
- PGN game collections (extracting tactical positions)
- Tablebase endgame databases
- Random position generation with specific criteria
- Opening book analysis for gamelets
- Game analysis for blunder-avoidance positions

The generators convert external data into the standardized PuzzlePosition format
used by the puzzle solving framework.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Iterator
from urllib.parse import urlencode
import chess
import chess.pgn
import chess.engine

from . import (
    PuzzlePosition, PuzzleSet, PuzzleType, TacticMotif, EndgameType
)


class LichessPuzzleGenerator:
    """Generate puzzles from Lichess puzzle database."""

    def __init__(self, max_requests_per_minute: int = 30):
        """
        Initialize Lichess puzzle generator.

        Args:
            max_requests_per_minute: Rate limit for API requests
        """
        self.base_url = "https://lichess.org/api/puzzle"
        self.rate_limit = max_requests_per_minute
        self.last_request_time = 0

    async def generate_puzzle_set(self,
                                 count: int = 50,
                                 min_rating: int = 1000,
                                 max_rating: int = 2000,
                                 themes: Optional[List[str]] = None) -> PuzzleSet:
        """
        Generate a puzzle set from Lichess database.

        Args:
            count: Number of puzzles to generate
            min_rating: Minimum puzzle rating
            max_rating: Maximum puzzle rating
            themes: Lichess puzzle themes to filter by

        Returns:
            PuzzleSet with generated puzzles
        """
        puzzles = []
        themes = themes or ["middlegame", "endgame", "opening"]

        for theme in themes:
            theme_count = count // len(themes)
            theme_puzzles = await self._fetch_puzzles_by_theme(
                theme, theme_count, min_rating, max_rating
            )
            puzzles.extend(theme_puzzles)

        # Shuffle and limit to requested count
        random.shuffle(puzzles)
        puzzles = puzzles[:count]

        return PuzzleSet(
            name=f"Lichess Puzzles ({min_rating}-{max_rating})",
            description=f"Generated from Lichess puzzle database with {count} puzzles",
            puzzles=puzzles,
            source="Lichess"
        )

    async def _fetch_puzzles_by_theme(self, theme: str, count: int,
                                     min_rating: int, max_rating: int) -> List[PuzzlePosition]:
        """Fetch puzzles from Lichess API by theme."""
        puzzles = []

        # Note: This is a simplified example. Real implementation would need
        # proper API integration with authentication and pagination
        try:
            # Simulate API call (replace with actual Lichess API integration)
            await asyncio.sleep(0.1)  # Rate limiting

            # Mock puzzle data - in real implementation, this would fetch from Lichess
            mock_puzzles = self._generate_mock_lichess_puzzles(theme, count, min_rating, max_rating)
            puzzles.extend(mock_puzzles)

        except Exception as e:
            print(f"Error fetching puzzles for theme {theme}: {e}")

        return puzzles

    def _generate_mock_lichess_puzzles(self, theme: str, count: int,
                                      min_rating: int, max_rating: int) -> List[PuzzlePosition]:
        """Generate mock Lichess-style puzzles (placeholder for real API integration)."""
        puzzles = []

        # Sample puzzle patterns based on theme
        if theme == "middlegame":
            puzzle_templates = [
                ("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Bxf7+", "Tactical shot"),
                ("r1bq1rk1/ppp2ppp/2n2n2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 7", "Nxd5", "Knight fork"),
            ]
        elif theme == "endgame":
            puzzle_templates = [
                ("8/8/8/8/8/2k5/2K5/2Q5 w - - 0 1", "Qc4+", "Queen mate"),
                ("8/8/8/8/8/3k4/3K4/3R4 w - - 0 1", "Rd3+", "Rook mate"),
            ]
        else:  # opening
            puzzle_templates = [
                ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "Nf3", "Development"),
                ("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3", "d3", "Italian setup"),
            ]

        for i in range(min(count, len(puzzle_templates) * 3)):
            template = puzzle_templates[i % len(puzzle_templates)]
            rating = random.randint(min_rating, max_rating)
            difficulty = self._rating_to_difficulty(rating)

            puzzle = PuzzlePosition(
                fen=template[0],
                puzzle_type=self._theme_to_puzzle_type(theme),
                difficulty=difficulty,
                best_move=template[1],
                title=f"Lichess {theme.title()} Puzzle",
                description=template[2],
                source=f"Lichess ({theme})",
                tags=[theme, f"rating_{rating}"]
            )
            puzzles.append(puzzle)

        return puzzles

    def _theme_to_puzzle_type(self, theme: str) -> PuzzleType:
        """Convert Lichess theme to puzzle type."""
        mapping = {
            "middlegame": PuzzleType.TACTIC,
            "endgame": PuzzleType.ENDGAME,
            "opening": PuzzleType.GAMELET,
            "blunder": PuzzleType.BLUNDER_AVOID
        }
        return mapping.get(theme, PuzzleType.TACTIC)

    def _rating_to_difficulty(self, rating: int) -> int:
        """Convert Lichess rating to difficulty scale 1-10."""
        if rating < 1000:
            return 1
        elif rating < 1200:
            return 2
        elif rating < 1400:
            return 3
        elif rating < 1600:
            return 4
        elif rating < 1800:
            return 5
        elif rating < 2000:
            return 6
        elif rating < 2200:
            return 7
        elif rating < 2400:
            return 8
        elif rating < 2600:
            return 9
        else:
            return 10


class PGNGameAnalyzer:
    """Extract tactical puzzles from PGN game collections."""

    def __init__(self, engine_path: Optional[str] = None):
        """
        Initialize PGN analyzer.

        Args:
            engine_path: Path to chess engine for analysis
        """
        self.engine_path = engine_path
        self.engine = None

    async def analyze_pgn_file(self, pgn_path: Path,
                              max_puzzles: int = 100,
                              min_advantage: float = 200) -> PuzzleSet:
        """
        Analyze PGN file and extract tactical positions.

        Args:
            pgn_path: Path to PGN file
            max_puzzles: Maximum puzzles to extract
            min_advantage: Minimum advantage in centipawns for tactical shots

        Returns:
            PuzzleSet with extracted puzzles
        """
        puzzles = []

        if self.engine_path:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)

        try:
            with open(pgn_path, 'r') as pgn_file:
                game_count = 0
                while len(puzzles) < max_puzzles:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break

                    game_puzzles = await self._extract_puzzles_from_game(game, min_advantage)
                    puzzles.extend(game_puzzles)
                    game_count += 1

                    if game_count % 10 == 0:
                        print(f"Analyzed {game_count} games, found {len(puzzles)} puzzles")

        finally:
            if self.engine:
                self.engine.quit()

        return PuzzleSet(
            name=f"PGN Analysis - {pgn_path.name}",
            description=f"Tactical puzzles extracted from {pgn_path.name}",
            puzzles=puzzles[:max_puzzles],
            source=str(pgn_path)
        )

    async def _extract_puzzles_from_game(self, game: chess.pgn.Game,
                                        min_advantage: float) -> List[PuzzlePosition]:
        """Extract tactical puzzles from a single game."""
        puzzles = []
        board = game.board()

        for move_num, move in enumerate(game.mainline_moves()):
            if move_num < 10:  # Skip opening moves
                board.push(move)
                continue

            # Check if this move creates a significant advantage
            if self.engine:
                try:
                    # Analyze position before move
                    info_before = self.engine.analyse(board, chess.engine.Limit(depth=12))

                    # Make the move
                    board.push(move)

                    # Analyze position after move
                    info_after = self.engine.analyse(board, chess.engine.Limit(depth=12))

                    # Calculate advantage change
                    score_before = self._score_to_cp(info_before.score, board.turn)
                    score_after = self._score_to_cp(info_after.score, not board.turn)

                    advantage_gain = score_after - score_before

                    # If significant advantage gained, create puzzle
                    if advantage_gain >= min_advantage:
                        # Go back to position before the tactical move
                        puzzle_board = board.copy()
                        puzzle_board.pop()

                        puzzle = PuzzlePosition(
                            fen=puzzle_board.fen(),
                            puzzle_type=PuzzleType.TACTIC,
                            difficulty=self._advantage_to_difficulty(advantage_gain),
                            best_move=move.uci(),
                            title=f"Tactical Shot (Move {move_num})",
                            description=f"Advantage gained: {advantage_gain:.0f} centipawns",
                            source="PGN Analysis",
                            evaluation=advantage_gain,
                            tags=["tactical_shot", f"advantage_{int(advantage_gain)}"]
                        )
                        puzzles.append(puzzle)

                except Exception as e:
                    print(f"Error analyzing move {move_num}: {e}")
                    board.push(move)
            else:
                board.push(move)

        return puzzles

    def _score_to_cp(self, score: chess.engine.Score, white_to_move: bool) -> float:
        """Convert engine score to centipawns."""
        if score.is_mate():
            mate_value = 10000 - abs(score.mate()) * 100
            return mate_value if score.mate() > 0 else -mate_value
        else:
            cp_score = score.score()
            return cp_score if white_to_move else -cp_score

    def _advantage_to_difficulty(self, advantage: float) -> int:
        """Convert advantage gain to difficulty rating."""
        if advantage < 100:
            return 1
        elif advantage < 200:
            return 2
        elif advantage < 300:
            return 3
        elif advantage < 500:
            return 4
        elif advantage < 700:
            return 5
        elif advantage < 1000:
            return 6
        elif advantage < 1500:
            return 7
        elif advantage < 2000:
            return 8
        elif advantage < 3000:
            return 9
        else:
            return 10


class EndgameGenerator:
    """Generate endgame puzzles from tablebase positions."""

    def __init__(self):
        """Initialize endgame generator."""
        self.endgame_patterns = {
            EndgameType.KQK: self._generate_kqk_positions,
            EndgameType.KRK: self._generate_krk_positions,
            EndgameType.KBNK: self._generate_kbnk_positions,
            EndgameType.KPK: self._generate_kpk_positions,
        }

    def generate_endgame_set(self, endgame_type: EndgameType,
                           count: int = 20) -> PuzzleSet:
        """
        Generate endgame puzzle set of specific type.

        Args:
            endgame_type: Type of endgame to generate
            count: Number of positions to generate

        Returns:
            PuzzleSet with endgame puzzles
        """
        if endgame_type not in self.endgame_patterns:
            raise ValueError(f"Unsupported endgame type: {endgame_type}")

        generator_func = self.endgame_patterns[endgame_type]
        puzzles = generator_func(count)

        return PuzzleSet(
            name=f"{endgame_type.value} Endgames",
            description=f"Generated {endgame_type.value} endgame positions",
            puzzles=puzzles,
            puzzle_type=PuzzleType.ENDGAME
        )

    def _generate_kqk_positions(self, count: int) -> List[PuzzlePosition]:
        """Generate King + Queen vs King positions."""
        puzzles = []

        for i in range(count):
            # Generate random but sensible KQK positions
            king_sq = random.choice(list(chess.SQUARES))

            # Opponent king not too close
            valid_squares = [sq for sq in chess.SQUARES
                           if chess.square_distance(king_sq, sq) >= 3]
            opp_king_sq = random.choice(valid_squares)

            # Queen somewhere useful
            queen_squares = [sq for sq in chess.SQUARES
                           if sq != king_sq and sq != opp_king_sq]
            queen_sq = random.choice(queen_squares)

            # Create position
            board = chess.Board(fen=None)
            board.clear()
            board.set_piece_at(king_sq, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(queen_sq, chess.Piece(chess.QUEEN, chess.WHITE))
            board.set_piece_at(opp_king_sq, chess.Piece(chess.KING, chess.BLACK))
            board.turn = chess.WHITE

            # Find best move (simplified - in real implementation use tablebase)
            best_move = self._find_kqk_best_move(board)

            if best_move:
                puzzle = PuzzlePosition(
                    fen=board.fen(),
                    puzzle_type=PuzzleType.ENDGAME,
                    difficulty=random.randint(3, 6),
                    best_move=best_move.uci(),
                    title=f"KQ vs K Endgame #{i+1}",
                    description="King and Queen vs King endgame",
                    endgame_type=EndgameType.KQK,
                    tags=["endgame", "kqk", "basic_mate"]
                )
                puzzles.append(puzzle)

        return puzzles

    def _find_kqk_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Find best move in KQ vs K endgame (simplified heuristic)."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Simple heuristic: prefer checks, then moves that restrict opponent king
        for move in legal_moves:
            board.push(move)
            if board.is_check():
                board.pop()
                return move
            board.pop()

        return legal_moves[0] if legal_moves else None

    def _generate_krk_positions(self, count: int) -> List[PuzzlePosition]:
        """Generate King + Rook vs King positions."""
        puzzles = []

        for i in range(count):
            # Similar logic to KQK but with rook
            king_sq = random.choice(list(chess.SQUARES))

            valid_squares = [sq for sq in chess.SQUARES
                           if chess.square_distance(king_sq, sq) >= 2]
            opp_king_sq = random.choice(valid_squares)

            rook_squares = [sq for sq in chess.SQUARES
                          if sq != king_sq and sq != opp_king_sq]
            rook_sq = random.choice(rook_squares)

            board = chess.Board(fen=None)
            board.clear()
            board.set_piece_at(king_sq, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(rook_sq, chess.Piece(chess.ROOK, chess.WHITE))
            board.set_piece_at(opp_king_sq, chess.Piece(chess.KING, chess.BLACK))
            board.turn = chess.WHITE

            best_move = self._find_krk_best_move(board)

            if best_move:
                puzzle = PuzzlePosition(
                    fen=board.fen(),
                    puzzle_type=PuzzleType.ENDGAME,
                    difficulty=random.randint(4, 7),
                    best_move=best_move.uci(),
                    title=f"KR vs K Endgame #{i+1}",
                    description="King and Rook vs King endgame",
                    endgame_type=EndgameType.KRK,
                    tags=["endgame", "krk", "rook_mate"]
                )
                puzzles.append(puzzle)

        return puzzles

    def _find_krk_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Find best move in KR vs K endgame (simplified)."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Prefer moves that cut off the king
        return legal_moves[0]  # Simplified

    def _generate_kbnk_positions(self, count: int) -> List[PuzzlePosition]:
        """Generate King + Bishop + Knight vs King positions."""
        # More complex endgame - simplified implementation
        return []

    def _generate_kpk_positions(self, count: int) -> List[PuzzlePosition]:
        """Generate King + Pawn vs King positions."""
        puzzles = []

        for i in range(count):
            # Generate KPK positions with meaningful choices
            files = [chess.FILE_A, chess.FILE_B, chess.FILE_C, chess.FILE_D,
                    chess.FILE_E, chess.FILE_F, chess.FILE_G, chess.FILE_H]
            file = random.choice(files)

            # Pawn on 6th or 7th rank usually
            pawn_rank = random.choice([chess.RANK_6, chess.RANK_7])
            pawn_sq = chess.square(file, pawn_rank)

            # Kings positioned for interesting play
            king_sq = chess.square(file, pawn_rank + 1)  # King supporting pawn
            opp_king_sq = chess.square(file, pawn_rank - 2)  # Opposing king

            board = chess.Board(fen=None)
            board.clear()
            board.set_piece_at(king_sq, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(pawn_sq, chess.Piece(chess.PAWN, chess.WHITE))
            board.set_piece_at(opp_king_sq, chess.Piece(chess.KING, chess.BLACK))
            board.turn = chess.WHITE

            if board.is_valid():
                # Simple best move logic
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    # Prefer pawn pushes
                    pawn_moves = [m for m in legal_moves
                                 if board.piece_at(m.from_square).piece_type == chess.PAWN]
                    best_move = pawn_moves[0] if pawn_moves else legal_moves[0]

                    puzzle = PuzzlePosition(
                        fen=board.fen(),
                        puzzle_type=PuzzleType.ENDGAME,
                        difficulty=random.randint(5, 8),
                        best_move=best_move.uci(),
                        title=f"KP vs K Endgame #{i+1}",
                        description="King and Pawn vs King endgame",
                        endgame_type=EndgameType.KPK,
                        tags=["endgame", "kpk", "pawn_endgame"]
                    )
                    puzzles.append(puzzle)

        return puzzles


class BlunderGenerator:
    """Generate blunder-avoidance puzzles from game analysis."""

    def __init__(self, engine_path: Optional[str] = None):
        """Initialize blunder generator."""
        self.engine_path = engine_path

    async def generate_blunder_positions(self, pgn_path: Path,
                                       count: int = 50,
                                       min_blunder: float = 200) -> PuzzleSet:
        """
        Generate blunder-avoidance puzzles from PGN analysis.

        Args:
            pgn_path: Path to PGN file
            count: Number of blunder positions to find
            min_blunder: Minimum evaluation drop for blunder

        Returns:
            PuzzleSet with blunder-avoidance puzzles
        """
        puzzles = []

        if not self.engine_path:
            # Generate mock blunder positions
            return self._generate_mock_blunder_positions(count)

        # Real implementation would analyze games for blunders
        # This is a simplified version
        return PuzzleSet(
            name="Blunder Avoidance Puzzles",
            description=f"Generated from game analysis in {pgn_path.name}",
            puzzles=puzzles,
            puzzle_type=PuzzleType.BLUNDER_AVOID
        )

    def _generate_mock_blunder_positions(self, count: int) -> PuzzleSet:
        """Generate mock blunder-avoidance positions."""
        mock_positions = [
            ("rnbqkbnr/ppp1pppp/8/3p4/3PP3/8/PPP2PPP/RNBQKBNR b KQkq e3 0 2",
             "dxe4", "Don't hang the pawn"),
            ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
             "d3", "Avoid Scholar's mate trap"),
        ]

        puzzles = []
        for i in range(min(count, len(mock_positions) * 5)):
            pos_data = mock_positions[i % len(mock_positions)]

            puzzle = PuzzlePosition(
                fen=pos_data[0],
                puzzle_type=PuzzleType.BLUNDER_AVOID,
                difficulty=random.randint(2, 6),
                best_move=pos_data[1],
                title=f"Avoid Blunder #{i+1}",
                description=pos_data[2],
                blunder_threshold=random.randint(200, 500),
                tags=["blunder_avoid", "critical_position"]
            )
            puzzles.append(puzzle)

        return PuzzleSet(
            name="Blunder Avoidance Collection",
            description="Critical positions requiring careful play",
            puzzles=puzzles,
            puzzle_type=PuzzleType.BLUNDER_AVOID
        )


class OpeningGameletGenerator:
    """Generate opening gamelet puzzles from opening theory."""

    def __init__(self):
        """Initialize opening generator."""
        self.opening_lines = {
            "King's Pawn": [
                ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
                 "Nf3", "Develop knight attacking pawn"),
            ],
            "Queen's Gambit": [
                ("rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
                 "dxc4", "Accept the gambit"),
                ("rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
                 "e6", "Decline the gambit"),
            ],
            "Italian Game": [
                ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                 "d3", "Prepare castle and support center"),
            ]
        }

    def generate_opening_gamelets(self, count: int = 30) -> PuzzleSet:
        """
        Generate opening gamelet puzzles.

        Args:
            count: Number of gamelets to generate

        Returns:
            PuzzleSet with opening gamelets
        """
        puzzles = []

        for i in range(count):
            opening_name = random.choice(list(self.opening_lines.keys()))
            line_data = random.choice(self.opening_lines[opening_name])

            puzzle = PuzzlePosition(
                fen=line_data[0],
                puzzle_type=PuzzleType.GAMELET,
                difficulty=random.randint(2, 5),
                best_move=line_data[1],
                title=f"{opening_name} Gamelet",
                description=line_data[2],
                opening_name=opening_name,
                tags=["opening", "gamelet", opening_name.lower().replace(" ", "_")]
            )
            puzzles.append(puzzle)

        return PuzzleSet(
            name="Opening Gamelets",
            description="Key opening positions requiring precise moves",
            puzzles=puzzles,
            puzzle_type=PuzzleType.GAMELET
        )


def generate_random_puzzle_set(puzzle_type: PuzzleType,
                             count: int = 20,
                             difficulty_range: Tuple[int, int] = (1, 10)) -> PuzzleSet:
    """
    Generate a random puzzle set of specified type.

    Args:
        puzzle_type: Type of puzzles to generate
        count: Number of puzzles
        difficulty_range: Min and max difficulty

    Returns:
        Generated puzzle set
    """
    generators = {
        PuzzleType.TACTIC: lambda: LichessPuzzleGenerator().generate_puzzle_set(count),
        PuzzleType.ENDGAME: lambda: EndgameGenerator().generate_endgame_set(EndgameType.KQK, count),
        PuzzleType.BLUNDER_AVOID: lambda: BlunderGenerator()._generate_mock_blunder_positions(count),
        PuzzleType.GAMELET: lambda: OpeningGameletGenerator().generate_opening_gamelets(count)
    }

    if puzzle_type not in generators:
        raise ValueError(f"Unsupported puzzle type: {puzzle_type}")

    # For this synchronous function, return a basic set
    # In async context, use the actual generators
    return PuzzleSet(
        name=f"Random {puzzle_type.value.title()} Puzzles",
        description=f"Randomly generated {puzzle_type.value} puzzles",
        puzzles=[],  # Empty for now - would be populated by actual generators
        puzzle_type=puzzle_type
    )


async def generate_comprehensive_puzzle_database(output_dir: Path,
                                               total_puzzles: int = 1000) -> Dict[PuzzleType, PuzzleSet]:
    """
    Generate a comprehensive puzzle database with all types.

    Args:
        output_dir: Directory to save puzzle sets
        total_puzzles: Total number of puzzles to generate

    Returns:
        Dictionary mapping puzzle types to generated sets
    """
    puzzle_sets = {}
    puzzles_per_type = total_puzzles // 4

    # Generate different types of puzzles
    generators = {
        PuzzleType.TACTIC: LichessPuzzleGenerator(),
        PuzzleType.ENDGAME: EndgameGenerator(),
        PuzzleType.BLUNDER_AVOID: BlunderGenerator(),
        PuzzleType.GAMELET: OpeningGameletGenerator()
    }

    for puzzle_type, generator in generators.items():
        print(f"Generating {puzzle_type.value} puzzles...")

        try:
            if puzzle_type == PuzzleType.TACTIC:
                puzzle_set = await generator.generate_puzzle_set(puzzles_per_type)
            elif puzzle_type == PuzzleType.ENDGAME:
                puzzle_set = generator.generate_endgame_set(EndgameType.KQK, puzzles_per_type)
            elif puzzle_type == PuzzleType.BLUNDER_AVOID:
                puzzle_set = generator._generate_mock_blunder_positions(puzzles_per_type)
            elif puzzle_type == PuzzleType.GAMELET:
                puzzle_set = generator.generate_opening_gamelets(puzzles_per_type)

            puzzle_sets[puzzle_type] = puzzle_set

            # Save individual puzzle set
            set_file = output_dir / f"{puzzle_type.value}_puzzles.json"
            with open(set_file, 'w') as f:
                json.dump({
                    'name': puzzle_set.name,
                    'description': puzzle_set.description,
                    'puzzle_type': puzzle_type.value,
                    'puzzles': [
                        {
                            'fen': p.fen,
                            'best_move': p.best_move,
                            'difficulty': p.difficulty,
                            'title': p.title,
                            'description': p.description,
                            'tags': p.tags
                        } for p in puzzle_set.puzzles
                    ]
                }, f, indent=2)

            print(f"Generated {len(puzzle_set.puzzles)} {puzzle_type.value} puzzles")

        except Exception as e:
            print(f"Error generating {puzzle_type.value} puzzles: {e}")
            continue

    return puzzle_sets


def load_puzzle_set_from_json(json_path: Path) -> PuzzleSet:
    """
    Load a puzzle set from JSON file.

    Args:
        json_path: Path to JSON puzzle file

    Returns:
        Loaded puzzle set
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    puzzles = []
    for puzzle_data in data['puzzles']:
        puzzle_type = PuzzleType(data['puzzle_type'])

        puzzle = PuzzlePosition(
            fen=puzzle_data['fen'],
            puzzle_type=puzzle_type,
            difficulty=puzzle_data['difficulty'],
            best_move=puzzle_data['best_move'],
            title=puzzle_data['title'],
            description=puzzle_data['description'],
            tags=puzzle_data.get('tags', [])
        )
        puzzles.append(puzzle)

    return PuzzleSet(
        name=data['name'],
        description=data['description'],
        puzzles=puzzles,
        puzzle_type=puzzle_type
    )


def export_puzzle_set_to_csv(puzzle_set: PuzzleSet, csv_path: Path) -> None:
    """
    Export puzzle set to CSV format.

    Args:
        puzzle_set: Puzzle set to export
        csv_path: Output CSV file path
    """
    import csv

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fen', 'best_move', 'difficulty', 'title', 'description', 'tags'])

        for puzzle in puzzle_set.puzzles:
            writer.writerow([
                puzzle.fen,
                puzzle.best_move,
                puzzle.difficulty,
                puzzle.title,
                puzzle.description,
                ','.join(puzzle.tags)
            ])


def import_puzzle_set_from_csv(csv_path: Path, puzzle_type: PuzzleType) -> PuzzleSet:
    """
    Import puzzle set from CSV format.

    Args:
        csv_path: Path to CSV file
        puzzle_type: Type of puzzles in the set

    Returns:
        Imported puzzle set
    """
    import csv

    puzzles = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tags = row.get('tags', '').split(',') if row.get('tags') else []

            puzzle = PuzzlePosition(
                fen=row['fen'],
                puzzle_type=puzzle_type,
                difficulty=int(row['difficulty']),
                best_move=row['best_move'],
                title=row.get('title', ''),
                description=row.get('description', ''),
                tags=[tag.strip() for tag in tags if tag.strip()]
            )
            puzzles.append(puzzle)

    return PuzzleSet(
        name=f"Imported from {csv_path.name}",
        description=f"Puzzle set imported from {csv_path}",
        puzzles=puzzles,
        puzzle_type=puzzle_type
    )
