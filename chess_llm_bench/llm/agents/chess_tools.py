"""
Chess analysis tools for agent-based chess playing.

This module provides various tools that agents can use to analyze chess positions
and make decisions without relying on chess engines.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import chess
import chess.pgn
from enum import Enum


class MoveCategory(Enum):
    """Categories for classifying chess moves."""
    CAPTURE = "capture"
    CHECK = "check"
    CASTLE = "castle"
    PAWN_ADVANCE = "pawn_advance"
    PIECE_DEVELOPMENT = "piece_development"
    CENTER_CONTROL = "center_control"
    KING_SAFETY = "king_safety"
    TACTICAL = "tactical"
    DEFENSIVE = "defensive"
    ATTACKING = "attacking"


@dataclass
class MoveAnalysis:
    """Analysis result for a single move."""
    move: chess.Move
    san: str
    categories: List[MoveCategory]
    score: float  # Heuristic evaluation score
    threats: List[str]
    defends: List[str]
    explanation: str


class ChessAnalysisTools:
    """Collection of chess analysis tools for agents."""

    # Piece values for material evaluation
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # King has no material value
    }

    # Central squares for position evaluation
    CENTER_SQUARES = [
        chess.E4, chess.D4, chess.E5, chess.D5,  # Inner center
        chess.C3, chess.C4, chess.C5, chess.C6,  # Extended center
        chess.D3, chess.D6, chess.E3, chess.E6,
        chess.F3, chess.F4, chess.F5, chess.F6
    ]

    # Piece-square tables for positional evaluation (simplified)
    PAWN_TABLE = [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ]

    KNIGHT_TABLE = [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ]

    def __init__(self, board: chess.Board):
        """Initialize analysis tools with a chess board."""
        self.board = board

    def get_board_state(self) -> Dict[str, Any]:
        """Get comprehensive board state information."""
        return {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn else "black",
            "move_number": self.board.fullmove_number,
            "halfmove_clock": self.board.halfmove_clock,
            "is_check": self.board.is_check(),
            "is_checkmate": self.board.is_checkmate(),
            "is_stalemate": self.board.is_stalemate(),
            "is_insufficient_material": self.board.is_insufficient_material(),
            "can_claim_draw": self.board.can_claim_draw(),
            "legal_moves_count": self.board.legal_moves.count(),
            "pieces": self._get_piece_positions(),
            "material_balance": self.evaluate_material(),
            "position_evaluation": self.evaluate_position()
        }

    def _get_piece_positions(self) -> Dict[str, List[str]]:
        """Get positions of all pieces on the board."""
        positions = {
            "white": {"pawns": [], "knights": [], "bishops": [], "rooks": [], "queens": [], "king": []},
            "black": {"pawns": [], "knights": [], "bishops": [], "rooks": [], "queens": [], "king": []}
        }

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                color = "white" if piece.color else "black"
                square_name = chess.square_name(square)

                if piece.piece_type == chess.PAWN:
                    positions[color]["pawns"].append(square_name)
                elif piece.piece_type == chess.KNIGHT:
                    positions[color]["knights"].append(square_name)
                elif piece.piece_type == chess.BISHOP:
                    positions[color]["bishops"].append(square_name)
                elif piece.piece_type == chess.ROOK:
                    positions[color]["rooks"].append(square_name)
                elif piece.piece_type == chess.QUEEN:
                    positions[color]["queens"].append(square_name)
                elif piece.piece_type == chess.KING:
                    positions[color]["king"].append(square_name)

        return positions

    def evaluate_material(self) -> Dict[str, Any]:
        """Evaluate material balance on the board."""
        white_material = 0
        black_material = 0
        piece_count = {"white": {}, "black": {}}

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = self.PIECE_VALUES[piece.piece_type]
                color = "white" if piece.color else "black"
                piece_name = piece.symbol().lower()

                if piece.color:
                    white_material += value
                else:
                    black_material += value

                piece_count[color][piece_name] = piece_count[color].get(piece_name, 0) + 1

        return {
            "white_material": white_material,
            "black_material": black_material,
            "material_difference": white_material - black_material,
            "piece_count": piece_count,
            "material_advantage": "white" if white_material > black_material else "black" if black_material > white_material else "equal"
        }

    def evaluate_position(self) -> Dict[str, Any]:
        """Evaluate positional factors of the current position."""
        evaluation = {
            "center_control": self._evaluate_center_control(),
            "piece_activity": self._evaluate_piece_activity(),
            "king_safety": self._evaluate_king_safety(),
            "pawn_structure": self._evaluate_pawn_structure(),
            "development": self._evaluate_development()
        }

        # Calculate overall position score
        total_score = sum([
            evaluation["center_control"]["score"],
            evaluation["piece_activity"]["score"],
            evaluation["king_safety"]["score"],
            evaluation["pawn_structure"]["score"],
            evaluation["development"]["score"]
        ])

        evaluation["total_score"] = total_score
        evaluation["position_advantage"] = "white" if total_score > 0 else "black" if total_score < 0 else "equal"

        return evaluation

    def _evaluate_center_control(self) -> Dict[str, Any]:
        """Evaluate control of the center squares."""
        white_control = 0
        black_control = 0

        for square in self.CENTER_SQUARES:
            # Check if square is occupied
            piece = self.board.piece_at(square)
            if piece:
                if piece.color:
                    white_control += 2
                else:
                    black_control += 2

            # Check if square is attacked
            white_attackers = len(self.board.attackers(chess.WHITE, square))
            black_attackers = len(self.board.attackers(chess.BLACK, square))

            white_control += white_attackers
            black_control += black_attackers

        score = white_control - black_control
        return {
            "white_control": white_control,
            "black_control": black_control,
            "score": score,
            "evaluation": "white controls center" if score > 2 else "black controls center" if score < -2 else "balanced"
        }

    def _evaluate_piece_activity(self) -> Dict[str, Any]:
        """Evaluate piece mobility and activity."""
        original_turn = self.board.turn

        # Count legal moves for white
        self.board.turn = chess.WHITE
        white_mobility = self.board.legal_moves.count()

        # Count legal moves for black
        self.board.turn = chess.BLACK
        black_mobility = self.board.legal_moves.count()

        # Restore original turn
        self.board.turn = original_turn

        score = (white_mobility - black_mobility) * 0.1

        return {
            "white_mobility": white_mobility,
            "black_mobility": black_mobility,
            "score": score,
            "evaluation": "white more active" if score > 2 else "black more active" if score < -2 else "balanced"
        }

    def _evaluate_king_safety(self) -> Dict[str, Any]:
        """Evaluate king safety for both sides."""
        white_king_square = self.board.king(chess.WHITE)
        black_king_square = self.board.king(chess.BLACK)

        white_safety = 0
        black_safety = 0

        if white_king_square:
            # Check if king is castled
            if white_king_square in [chess.G1, chess.C1]:
                white_safety += 3
            # Count pawn shield
            white_safety += self._count_pawn_shield(white_king_square, chess.WHITE)
            # Penalty for exposed king
            white_safety -= len(self.board.attackers(chess.BLACK, white_king_square)) * 2

        if black_king_square:
            # Check if king is castled
            if black_king_square in [chess.G8, chess.C8]:
                black_safety += 3
            # Count pawn shield
            black_safety += self._count_pawn_shield(black_king_square, chess.BLACK)
            # Penalty for exposed king
            black_safety -= len(self.board.attackers(chess.WHITE, black_king_square)) * 2

        score = white_safety - black_safety

        return {
            "white_king_safety": white_safety,
            "black_king_safety": black_safety,
            "score": score,
            "evaluation": "white king safer" if score > 2 else "black king safer" if score < -2 else "both kings safe"
        }

    def _count_pawn_shield(self, king_square: int, color: chess.Color) -> int:
        """Count pawns protecting the king."""
        shield_count = 0
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)

        # Define shield squares based on king position
        if color == chess.WHITE:
            shield_ranks = [king_rank + 1, king_rank + 2] if king_rank < 6 else [king_rank + 1]
        else:
            shield_ranks = [king_rank - 1, king_rank - 2] if king_rank > 1 else [king_rank - 1]

        for rank in shield_ranks:
            for file_offset in [-1, 0, 1]:
                file = king_file + file_offset
                if 0 <= file <= 7 and 0 <= rank <= 7:
                    square = chess.square(file, rank)
                    piece = self.board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        shield_count += 1

        return shield_count

    def _evaluate_pawn_structure(self) -> Dict[str, Any]:
        """Evaluate pawn structure quality."""
        white_doubled = 0
        black_doubled = 0
        white_isolated = 0
        black_isolated = 0
        white_passed = 0
        black_passed = 0

        # Check each file for pawn structure features
        for file in range(8):
            white_pawns_on_file = []
            black_pawns_on_file = []

            for rank in range(8):
                square = chess.square(file, rank)
                piece = self.board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    if piece.color:
                        white_pawns_on_file.append(rank)
                    else:
                        black_pawns_on_file.append(rank)

            # Count doubled pawns
            if len(white_pawns_on_file) > 1:
                white_doubled += len(white_pawns_on_file) - 1
            if len(black_pawns_on_file) > 1:
                black_doubled += len(black_pawns_on_file) - 1

            # Check for isolated pawns (no pawns on adjacent files)
            adjacent_files = [f for f in [file - 1, file + 1] if 0 <= f <= 7]

            if white_pawns_on_file:
                has_support = False
                for adj_file in adjacent_files:
                    for rank in range(8):
                        square = chess.square(adj_file, rank)
                        piece = self.board.piece_at(square)
                        if piece and piece.piece_type == chess.PAWN and piece.color:
                            has_support = True
                            break
                if not has_support:
                    white_isolated += len(white_pawns_on_file)

            if black_pawns_on_file:
                has_support = False
                for adj_file in adjacent_files:
                    for rank in range(8):
                        square = chess.square(adj_file, rank)
                        piece = self.board.piece_at(square)
                        if piece and piece.piece_type == chess.PAWN and not piece.color:
                            has_support = True
                            break
                if not has_support:
                    black_isolated += len(black_pawns_on_file)

        # Calculate score (penalties for doubled and isolated pawns)
        white_structure_score = -(white_doubled * 0.5 + white_isolated * 0.3)
        black_structure_score = -(black_doubled * 0.5 + black_isolated * 0.3)
        score = white_structure_score - black_structure_score

        return {
            "white_doubled_pawns": white_doubled,
            "black_doubled_pawns": black_doubled,
            "white_isolated_pawns": white_isolated,
            "black_isolated_pawns": black_isolated,
            "score": score,
            "evaluation": "white structure better" if score > 1 else "black structure better" if score < -1 else "balanced"
        }

    def _evaluate_development(self) -> Dict[str, Any]:
        """Evaluate piece development in the opening."""
        if self.board.fullmove_number > 15:
            return {"score": 0, "evaluation": "middlegame/endgame"}

        white_developed = 0
        black_developed = 0

        # Check knight development
        if self.board.piece_at(chess.B1) != chess.Piece(chess.KNIGHT, chess.WHITE):
            white_developed += 1
        if self.board.piece_at(chess.G1) != chess.Piece(chess.KNIGHT, chess.WHITE):
            white_developed += 1
        if self.board.piece_at(chess.B8) != chess.Piece(chess.KNIGHT, chess.BLACK):
            black_developed += 1
        if self.board.piece_at(chess.G8) != chess.Piece(chess.KNIGHT, chess.BLACK):
            black_developed += 1

        # Check bishop development
        if self.board.piece_at(chess.C1) != chess.Piece(chess.BISHOP, chess.WHITE):
            white_developed += 1
        if self.board.piece_at(chess.F1) != chess.Piece(chess.BISHOP, chess.WHITE):
            white_developed += 1
        if self.board.piece_at(chess.C8) != chess.Piece(chess.BISHOP, chess.BLACK):
            black_developed += 1
        if self.board.piece_at(chess.F8) != chess.Piece(chess.BISHOP, chess.BLACK):
            black_developed += 1

        # Check castling
        if self.board.has_kingside_castling_rights(chess.WHITE) or self.board.has_queenside_castling_rights(chess.WHITE):
            white_developed -= 1  # Penalty for not castling yet
        if self.board.has_kingside_castling_rights(chess.BLACK) or self.board.has_queenside_castling_rights(chess.BLACK):
            black_developed -= 1

        score = white_developed - black_developed

        return {
            "white_development": white_developed,
            "black_development": black_developed,
            "score": score,
            "evaluation": "white better developed" if score > 1 else "black better developed" if score < -1 else "equal development"
        }

    def get_legal_moves(self) -> List[Dict[str, Any]]:
        """Get all legal moves with detailed information."""
        moves = []

        for move in self.board.legal_moves:
            move_info = {
                "uci": move.uci(),
                "san": self.board.san(move),
                "from_square": chess.square_name(move.from_square),
                "to_square": chess.square_name(move.to_square),
                "piece": self.board.piece_at(move.from_square).symbol() if self.board.piece_at(move.from_square) else None,
                "is_capture": self.board.is_capture(move),
                "is_check": self._gives_check(move),
                "is_castle": self.board.is_castling(move),
                "categories": self._categorize_move(move)
            }
            moves.append(move_info)

        return moves

    def _gives_check(self, move: chess.Move) -> bool:
        """Check if a move gives check."""
        self.board.push(move)
        is_check = self.board.is_check()
        self.board.pop()
        return is_check

    def _categorize_move(self, move: chess.Move) -> List[str]:
        """Categorize a move based on its characteristics."""
        categories = []

        piece = self.board.piece_at(move.from_square)
        if not piece:
            return categories

        # Capture moves
        if self.board.is_capture(move):
            categories.append(MoveCategory.CAPTURE.value)

        # Check moves
        if self._gives_check(move):
            categories.append(MoveCategory.CHECK.value)

        # Castling
        if self.board.is_castling(move):
            categories.append(MoveCategory.CASTLE.value)
            categories.append(MoveCategory.KING_SAFETY.value)

        # Pawn moves
        if piece.piece_type == chess.PAWN:
            categories.append(MoveCategory.PAWN_ADVANCE.value)
            # Check if it's a passed pawn
            if self._is_passed_pawn(move):
                categories.append(MoveCategory.ATTACKING.value)

        # Center control
        if move.to_square in self.CENTER_SQUARES:
            categories.append(MoveCategory.CENTER_CONTROL.value)

        # Development moves (in opening)
        if self.board.fullmove_number <= 10:
            if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                if chess.square_rank(move.from_square) in [0, 7]:  # Moving from back rank
                    categories.append(MoveCategory.PIECE_DEVELOPMENT.value)

        # Defensive moves
        if self._is_defensive_move(move):
            categories.append(MoveCategory.DEFENSIVE.value)

        # Attacking moves
        if self._is_attacking_move(move):
            categories.append(MoveCategory.ATTACKING.value)

        return categories

    def _is_passed_pawn(self, move: chess.Move) -> bool:
        """Check if a pawn move creates or advances a passed pawn."""
        piece = self.board.piece_at(move.from_square)
        if not piece or piece.piece_type != chess.PAWN:
            return False

        file = chess.square_file(move.to_square)
        rank = chess.square_rank(move.to_square)

        # Check if there are enemy pawns that can block this pawn
        enemy_color = not piece.color

        if piece.color == chess.WHITE:
            # Check ranks ahead
            for check_rank in range(rank + 1, 8):
                for file_offset in [-1, 0, 1]:
                    check_file = file + file_offset
                    if 0 <= check_file <= 7:
                        square = chess.square(check_file, check_rank)
                        enemy_piece = self.board.piece_at(square)
                        if enemy_piece and enemy_piece.piece_type == chess.PAWN and enemy_piece.color == enemy_color:
                            return False
        else:
            # Check ranks behind (for black pawns moving down)
            for check_rank in range(rank - 1, -1, -1):
                for file_offset in [-1, 0, 1]:
                    check_file = file + file_offset
                    if 0 <= check_file <= 7:
                        square = chess.square(check_file, check_rank)
                        enemy_piece = self.board.piece_at(square)
                        if enemy_piece and enemy_piece.piece_type == chess.PAWN and enemy_piece.color == enemy_color:
                            return False

        return True

    def _is_defensive_move(self, move: chess.Move) -> bool:
        """Check if a move is primarily defensive."""
        # Move defends a piece under attack
        self.board.push(move)

        # Check if we're moving a piece that was under attack
        original_attackers = self.board.attackers(not self.board.turn, move.from_square)

        # Check if the move blocks an attack on a valuable piece
        is_defensive = len(original_attackers) > 0

        self.board.pop()

        return is_defensive

    def _is_attacking_move(self, move: chess.Move) -> bool:
        """Check if a move creates threats or attacks."""
        self.board.push(move)

        # Check if the move attacks enemy pieces
        attacked_squares = list(self.board.attacks(move.to_square))
        valuable_attacks = False

        for square in attacked_squares:
            piece = self.board.piece_at(square)
            if piece and piece.color != self.board.turn:
                if piece.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                    valuable_attacks = True
                    break

        self.board.pop()

        return valuable_attacks or self._gives_check(move)

    def analyze_move(self, move: chess.Move) -> MoveAnalysis:
        """Provide detailed analysis of a specific move."""
        san = self.board.san(move)
        categories = [MoveCategory(cat) for cat in self._categorize_move(move)]

        # Calculate move score with more nuanced evaluation
        self.board.push(move)

        # Get position evaluation after the move
        position_eval = self.evaluate_position()
        material_eval = self.evaluate_material()

        # Base score from position and material
        score = position_eval["total_score"] + material_eval["material_difference"]

        # Add bonuses for specific move characteristics
        if self.board.is_checkmate():
            score += 1000  # Checkmate is the ultimate goal
        elif self.board.is_check():
            score += 5  # Giving check is often good

        # Check if this move wins material
        captured_piece = self.board.piece_at(move.to_square) if self.board.is_capture(move) else None
        if captured_piece:
            score += self.PIECE_VALUES[captured_piece.piece_type] * 2  # Double weight for captures

        # Find threats created by this move
        threats = []
        attacks = list(self.board.attacks(move.to_square))
        for square in attacks:
            piece = self.board.piece_at(square)
            if piece and piece.color != self.board.turn:
                threat_value = self.PIECE_VALUES[piece.piece_type]
                threats.append(f"Threatens {piece.symbol()} on {chess.square_name(square)}")
                score += threat_value * 0.3  # Bonus for creating threats

        # Find what this move defends
        defends = []
        for square in attacks:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                defends.append(f"Defends {piece.symbol()} on {chess.square_name(square)}")
                score += 0.5  # Small bonus for defending pieces

        self.board.pop()

        # Generate explanation
        explanation = self._generate_move_explanation(move, categories, threats, defends)

        return MoveAnalysis(
            move=move,
            san=san,
            categories=categories,
            score=score,
            threats=threats,
            defends=defends,
            explanation=explanation
        )

    def _generate_move_explanation(self, move: chess.Move, categories: List[MoveCategory],
                                  threats: List[str], defends: List[str]) -> str:
        """Generate a human-readable explanation of a move."""
        explanations = []

        piece = self.board.piece_at(move.from_square)
        if not piece:
            return "Invalid move"

        piece_name = chess.piece_name(piece.piece_type).capitalize()

        # Basic move description
        if MoveCategory.CASTLE in categories:
            explanations.append("Castles for king safety")
        elif MoveCategory.CAPTURE in categories:
            captured = self.board.piece_at(move.to_square)
            if captured:
                explanations.append(f"{piece_name} captures {chess.piece_name(captured.piece_type)}")
        else:
            explanations.append(f"{piece_name} moves to {chess.square_name(move.to_square)}")

        # Add category-based explanations
        if MoveCategory.CHECK in categories:
            explanations.append("Gives check")
        if MoveCategory.CENTER_CONTROL in categories:
            explanations.append("Controls the center")
        if MoveCategory.PIECE_DEVELOPMENT in categories:
            explanations.append("Develops piece")
        if MoveCategory.ATTACKING in categories and MoveCategory.CHECK not in categories:
            explanations.append("Creates attacking chances")
        if MoveCategory.DEFENSIVE in categories:
            explanations.append("Defensive move")

        # Add threat information
        if threats:
            explanations.append(f"Creates threats: {', '.join(threats[:2])}")

        return ". ".join(explanations)

    def get_all_move_analyses(self) -> List[MoveAnalysis]:
        """Analyze all legal moves and return them sorted by score."""
        analyzed_moves = []

        for move in self.board.legal_moves:
            analysis = self.analyze_move(move)
            analyzed_moves.append(analysis)

        # Sort by score (higher is better)
        analyzed_moves.sort(key=lambda x: x.score, reverse=True)

        # If we're in check, mark moves that escape check
        if self.board.is_check():
            for analysis in analyzed_moves:
                self.board.push(analysis.move)
                if not self.board.is_check():
                    # Add a bonus to check-escaping moves
                    analysis.score += 10  # High priority for escaping check
                self.board.pop()

        return analyzed_moves

    def evaluate_endgame(self) -> Dict[str, Any]:
        """Special evaluation for endgame positions."""
        material = self.evaluate_material()
        total_material = material["white_material"] + material["black_material"]

        # Check if we're in endgame (low material)
        is_endgame = total_material <= 13  # Queen + Rook or less

        if not is_endgame:
            return {"is_endgame": False}

        endgame_eval = {
            "is_endgame": True,
            "phase": "endgame" if total_material <= 8 else "late_middlegame",
            "king_activity": self._evaluate_king_activity_endgame(),
            "pawn_advancement": self._evaluate_pawn_advancement(),
            "piece_coordination": self._evaluate_piece_coordination()
        }

        return endgame_eval

    def _evaluate_king_activity_endgame(self) -> Dict[str, Any]:
        """Evaluate king activity in the endgame."""
        white_king = self.board.king(chess.WHITE)
        black_king = self.board.king(chess.BLACK)

        white_centralization = 0
        black_centralization = 0

        if white_king:
            # King centralization is good in endgame
            file = chess.square_file(white_king)
            rank = chess.square_rank(white_king)
            white_centralization = 4 - abs(file - 3.5) + 4 - abs(rank - 3.5)

        if black_king:
            file = chess.square_file(black_king)
            rank = chess.square_rank(black_king)
            black_centralization = 4 - abs(file - 3.5) + 4 - abs(rank - 3.5)

        return {
            "white_king_activity": white_centralization,
            "black_king_activity": black_centralization,
            "score": white_centralization - black_centralization
        }

    def _evaluate_pawn_advancement(self) -> Dict[str, Any]:
        """Evaluate pawn advancement in endgame."""
        white_advancement = 0
        black_advancement = 0

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                rank = chess.square_rank(square)
                if piece.color == chess.WHITE:
                    # White pawns advance upward (higher rank = more advanced)
                    white_advancement += rank
                else:
                    # Black pawns advance downward (lower rank = more advanced)
                    black_advancement += (7 - rank)

        return {
            "white_advancement": white_advancement,
            "black_advancement": black_advancement,
            "score": white_advancement - black_advancement
        }

    def _evaluate_piece_coordination(self) -> Dict[str, Any]:
        """Evaluate piece coordination and cooperation."""
        white_coordination = 0
        black_coordination = 0

        # Check how many pieces defend each other
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                defenders = len(self.board.attackers(piece.color, square))
                if piece.color == chess.WHITE:
                    white_coordination += defenders
                else:
                    black_coordination += defenders

        return {
            "white_coordination": white_coordination,
            "black_coordination": black_coordination,
            "score": (white_coordination - black_coordination) * 0.1
        }
