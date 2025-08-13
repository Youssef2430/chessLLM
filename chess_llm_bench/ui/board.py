"""
Beautiful chess board visualization for the Chess LLM Benchmark.

This module provides stunning ASCII chess board rendering with Unicode pieces,
colors, move highlighting, and various display themes for the terminal interface.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional, Dict, List, Tuple, Set
from dataclasses import dataclass

import chess
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.align import Align
from rich.table import Table
from rich.box import ROUNDED


class BoardTheme(Enum):
    """Available chess board themes."""
    CLASSIC = "classic"
    MODERN = "modern"
    MINIMAL = "minimal"
    UNICODE = "unicode"
    ASCII = "ascii"


class PieceStyle(Enum):
    """Chess piece display styles."""
    UNICODE = "unicode"
    LETTERS = "letters"
    SYMBOLS = "symbols"


@dataclass
class BoardColors:
    """Color scheme for chess board rendering."""
    white_square: str = "white"
    black_square: str = "grey23"
    white_piece: str = "bright_white"
    black_piece: str = "grey0"
    highlight_move: str = "yellow"
    highlight_check: str = "red"
    highlight_last_move: str = "green"
    border: str = "cyan"
    coordinates: str = "dim white"


class ChessBoardRenderer:
    """
    Beautiful chess board renderer with Unicode pieces and colors.

    Provides various rendering styles, themes, and highlighting options
    for displaying chess positions in the terminal.
    """

    # Unicode chess pieces
    UNICODE_PIECES = {
        chess.PAWN: {"white": "♙", "black": "♟"},
        chess.ROOK: {"white": "♖", "black": "♜"},
        chess.KNIGHT: {"white": "♘", "black": "♞"},
        chess.BISHOP: {"white": "♗", "black": "♝"},
        chess.QUEEN: {"white": "♕", "black": "♛"},
        chess.KING: {"white": "♔", "black": "♚"},
    }

    # Alternative Unicode pieces (filled/outline)
    UNICODE_PIECES_ALT = {
        chess.PAWN: {"white": "♟", "black": "♙"},
        chess.ROOK: {"white": "♜", "black": "♖"},
        chess.KNIGHT: {"white": "♞", "black": "♘"},
        chess.BISHOP: {"white": "♝", "black": "♗"},
        chess.QUEEN: {"white": "♛", "black": "♕"},
        chess.KING: {"white": "♚", "black": "♔"},
    }

    # Letter notation
    LETTER_PIECES = {
        chess.PAWN: {"white": "P", "black": "p"},
        chess.ROOK: {"white": "R", "black": "r"},
        chess.KNIGHT: {"white": "N", "black": "n"},
        chess.BISHOP: {"white": "B", "black": "b"},
        chess.QUEEN: {"white": "Q", "black": "q"},
        chess.KING: {"white": "K", "black": "k"},
    }

    def __init__(
        self,
        theme: BoardTheme = BoardTheme.UNICODE,
        piece_style: PieceStyle = PieceStyle.UNICODE,
        colors: Optional[BoardColors] = None,
        flip_board: bool = False,
        show_coordinates: bool = True,
    ):
        """
        Initialize the chess board renderer.

        Args:
            theme: Board display theme
            piece_style: Style for chess pieces
            colors: Color scheme for the board
            flip_board: If True, display from black's perspective
            show_coordinates: Whether to show file/rank labels
        """
        self.theme = theme
        self.piece_style = piece_style
        self.colors = colors or BoardColors()
        self.flip_board = flip_board
        self.show_coordinates = show_coordinates

        # Select piece set based on style
        if piece_style == PieceStyle.UNICODE:
            self.pieces = self.UNICODE_PIECES
        elif piece_style == PieceStyle.LETTERS:
            self.pieces = self.LETTER_PIECES
        else:
            self.pieces = self.UNICODE_PIECES_ALT

    def render_board(
        self,
        board: chess.Board,
        last_move: Optional[chess.Move] = None,
        highlighted_squares: Optional[Set[chess.Square]] = None,
        console: Optional[Console] = None,
    ) -> Panel:
        """
        Render a beautiful chess board as a Rich Panel.

        Args:
            board: Chess position to render
            last_move: Last move to highlight
            highlighted_squares: Additional squares to highlight
            console: Rich console for rendering

        Returns:
            Rich Panel containing the rendered board
        """
        if console is None:
            console = Console()

        highlighted_squares = highlighted_squares or set()

        # Add last move squares to highlights
        if last_move:
            highlighted_squares.add(last_move.from_square)
            highlighted_squares.add(last_move.to_square)

        # Create the board table
        table = Table.grid(padding=0)

        # Add columns for coordinates + 8 files + coordinates
        if self.show_coordinates:
            table.add_column(justify="center", width=2)  # Rank numbers
        for _ in range(8):
            table.add_column(justify="center", width=3)  # Board squares
        if self.show_coordinates:
            table.add_column(justify="center", width=2)  # Rank numbers

        # File labels (a-h or h-a if flipped)
        if self.show_coordinates:
            files = "abcdefgh"
            if self.flip_board:
                files = files[::-1]

            file_row = ["  "] if self.show_coordinates else []
            for file_char in files:
                file_row.append(Text(file_char, style=self.colors.coordinates))
            if self.show_coordinates:
                file_row.append("  ")
            table.add_row(*file_row)

        # Board rows
        ranks = range(8, 0, -1) if not self.flip_board else range(1, 9)

        for rank in ranks:
            row_parts = []

            # Rank number (left side)
            if self.show_coordinates:
                row_parts.append(Text(str(rank), style=self.colors.coordinates))

            # Board squares
            files = range(8) if not self.flip_board else range(7, -1, -1)
            for file in files:
                square = chess.square(file, rank - 1)
                square_text = self._render_square(board, square, highlighted_squares)
                row_parts.append(square_text)

            # Rank number (right side)
            if self.show_coordinates:
                row_parts.append(Text(str(rank), style=self.colors.coordinates))

            table.add_row(*row_parts)

        # File labels (bottom)
        if self.show_coordinates:
            files = "abcdefgh"
            if self.flip_board:
                files = files[::-1]

            file_row = ["  "] if self.show_coordinates else []
            for file_char in files:
                file_row.append(Text(file_char, style=self.colors.coordinates))
            if self.show_coordinates:
                file_row.append("  ")
            table.add_row(*file_row)

        # Create title with game info
        title = self._create_board_title(board, last_move)

        # Wrap in panel with theme-specific styling
        return Panel(
            Align.center(table),
            title=title,
            border_style=self.colors.border,
            box=ROUNDED,
            padding=(0, 1),
        )

    def _render_square(
        self,
        board: chess.Board,
        square: chess.Square,
        highlighted_squares: Set[chess.Square],
    ) -> Text:
        """Render a single chess square with piece and background."""
        piece = board.piece_at(square)

        # Determine square colors
        is_light_square = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1

        if square in highlighted_squares:
            bg_color = self.colors.highlight_last_move
        elif is_light_square:
            bg_color = self.colors.white_square
        else:
            bg_color = self.colors.black_square

        # Get piece character
        if piece:
            piece_char = self._get_piece_char(piece)
            piece_color = self.colors.white_piece if piece.color == chess.WHITE else self.colors.black_piece
        else:
            piece_char = " "
            piece_color = "white"

        # Add spacing for visual appeal
        content = f" {piece_char} "

        return Text(content, style=f"{piece_color} on {bg_color}")

    def _get_piece_char(self, piece: chess.Piece) -> str:
        """Get the character representation of a chess piece."""
        color_key = "white" if piece.color == chess.WHITE else "black"
        return self.pieces[piece.piece_type][color_key]

    def _create_board_title(self, board: chess.Board, last_move: Optional[chess.Move]) -> str:
        """Create an informative title for the board."""
        turn = "White" if board.turn == chess.WHITE else "Black"
        move_num = (board.ply() // 2) + 1

        title_parts = [f"Move {move_num}"]

        if last_move:
            try:
                # Try to get SAN notation, but use UCI as fallback
                move_san = last_move.uci()  # Use UCI as default
                # Note: We can't use board.san() here because the move was already played
            except:
                move_san = last_move.uci()
            title_parts.append(f"Last: {move_san}")

        title_parts.append(f"To play: {turn}")

        # Add game status
        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
            title_parts.append(f"Checkmate - {winner} wins!")
        elif board.is_stalemate():
            title_parts.append("Stalemate")
        elif board.is_check():
            title_parts.append("Check!")
        elif board.is_insufficient_material():
            title_parts.append("Insufficient material")

        return " | ".join(title_parts)

    def render_move_list(
        self,
        board: chess.Board,
        moves: List[chess.Move],
        max_moves: int = 10,
    ) -> Panel:
        """
        Render a list of recent moves in algebraic notation.

        Args:
            board: Current chess position
            moves: List of moves to display
            max_moves: Maximum number of moves to show

        Returns:
            Rich Panel with move list
        """
        if not moves:
            return Panel("No moves yet", title="📜 Move History", border_style=self.colors.border)

        # Create a temporary board to generate SAN notation
        temp_board = chess.Board()
        move_texts = []

        # Show recent moves
        recent_moves = moves[-max_moves:] if len(moves) > max_moves else moves
        start_move = len(moves) - len(recent_moves) + 1

        for i, move in enumerate(recent_moves):
            move_num = start_move + i

            if temp_board.ply() < len(moves):
                try:
                    # Try to get the move from the current position
                    san = temp_board.san(move) if move in temp_board.legal_moves else move.uci()
                    temp_board.push(move)
                except:
                    san = move.uci()
            else:
                san = move.uci()

            # Format as "42. e4" or "42... e5"
            if move_num % 2 == 1:  # White move
                move_text = f"{(move_num + 1) // 2}. {san}"
            else:  # Black move
                if i == 0:  # First move in display and it's black
                    move_text = f"{move_num // 2}... {san}"
                else:
                    move_text = san

            move_texts.append(move_text)

        # Join moves in pairs for compact display
        formatted_moves = []
        i = 0
        while i < len(move_texts):
            if i + 1 < len(move_texts) and not move_texts[i].endswith('...'):
                # Pair white and black moves
                formatted_moves.append(f"{move_texts[i]} {move_texts[i + 1]}")
                i += 2
            else:
                formatted_moves.append(move_texts[i])
                i += 1

        content = "\n".join(formatted_moves)

        return Panel(
            content,
            title="📜 Recent Moves",
            border_style=self.colors.border,
            padding=(0, 1),
        )

    def render_game_info(
        self,
        white_name: str,
        black_name: str,
        result: Optional[str] = None,
        engine_elo: Optional[int] = None,
    ) -> Panel:
        """
        Render game information panel.

        Args:
            white_name: Name of white player
            black_name: Name of black player
            result: Game result if finished
            engine_elo: Engine ELO rating

        Returns:
            Rich Panel with game information
        """
        info_lines = []

        # Players
        info_lines.append(f"⚪ White: {white_name}")
        info_lines.append(f"⚫ Black: {black_name}")

        # Engine ELO
        if engine_elo:
            info_lines.append(f"🎯 Engine ELO: {engine_elo}")

        # Result
        if result:
            result_emoji = {
                "1-0": "🏆 White wins",
                "0-1": "🏆 Black wins",
                "1/2-1/2": "🤝 Draw",
                "*": "🎮 In progress"
            }.get(result, f"Result: {result}")
            info_lines.append(result_emoji)

        content = "\n".join(info_lines)

        return Panel(
            content,
            title="ℹ️ Game Info",
            border_style=self.colors.border,
            padding=(0, 1),
        )

    def create_robot_vs_robot_layout(
        self,
        board: chess.Board,
        white_bot: str,
        black_bot: str,
        last_move: Optional[chess.Move] = None,
        engine_elo: Optional[int] = None,
        moves: Optional[List[chess.Move]] = None,
    ) -> Panel:
        """
        Create a beautiful robot vs robot game layout.

        Args:
            board: Current chess position
            white_bot: Name of white bot
            black_bot: Name of black bot
            last_move: Last move played
            engine_elo: Engine ELO rating
            moves: List of all moves played

        Returns:
            Rich Panel with complete game layout
        """
        moves = moves or []

        # Main chess board
        board_panel = self.render_board(board, last_move)

        # Game info
        info_panel = self.render_game_info(
            f"🤖 {white_bot}",
            f"🤖 {black_bot}",
            board.result() if board.is_game_over() else None,
            engine_elo,
        )

        # Move history
        move_panel = self.render_move_list(board, moves)

        # Create layout table
        layout = Table.grid(expand=True)
        layout.add_column(ratio=3)  # Board
        layout.add_column(ratio=1)  # Side panel

        # Side panel with info and moves
        from rich.console import Group
        side_panel = Group(info_panel, move_panel)

        layout.add_row(board_panel, side_panel)

        # Status message
        status_msg = self._get_game_status_message(board, last_move)
        status_text = Text(status_msg, justify="center", style="bold")

        # Create main content with layout and status
        main_content = Group(layout, status_text)

        return Panel(
            main_content,
            title="🤖 Robot Chess Battle 🤖",
            title_align="center",
            border_style="bright_magenta",
            box=ROUNDED,
            padding=(1, 2),
        )

    def _get_game_status_message(self, board: chess.Board, last_move: Optional[chess.Move]) -> str:
        """Get current game status message."""
        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
            return f"🏆 Checkmate! {winner} wins!"
        elif board.is_stalemate():
            return "🤝 Stalemate! It's a draw!"
        elif board.is_insufficient_material():
            return "🤝 Draw by insufficient material!"
        elif board.is_check():
            player = "White" if board.turn == chess.WHITE else "Black"
            return f"⚠️  {player} is in check!"
        elif last_move:
            move_san = board.san(last_move) if last_move in board.legal_moves else last_move.uci()
            player = "Black" if board.turn == chess.WHITE else "White"
            return f"Last move: {move_san} - {player} to play..."
        else:
            player = "White" if board.turn == chess.WHITE else "Black"
            return f"Game starting - {player} to play..."


# Convenience functions for easy usage
def render_board_simple(
    board: chess.Board,
    last_move: Optional[chess.Move] = None,
    theme: BoardTheme = BoardTheme.UNICODE,
) -> Panel:
    """Simple board rendering function."""
    renderer = ChessBoardRenderer(theme=theme)
    return renderer.render_board(board, last_move)


def render_robot_battle(
    board: chess.Board,
    white_bot: str,
    black_bot: str,
    last_move: Optional[chess.Move] = None,
    engine_elo: Optional[int] = None,
    moves: Optional[List[chess.Move]] = None,
) -> Panel:
    """Render a robot vs robot chess battle."""
    renderer = ChessBoardRenderer(
        theme=BoardTheme.UNICODE,
        piece_style=PieceStyle.UNICODE,
        show_coordinates=True,
    )
    return renderer.create_robot_vs_robot_layout(
        board, white_bot, black_bot, last_move, engine_elo, moves
    )
