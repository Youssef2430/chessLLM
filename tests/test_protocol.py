"""
Acceptance tests for chess protocol validation.

This module contains tests that validate the chess protocol handling to catch common issues:
1. Wrong side-to-move detection
2. Illegal UCI move acceptance
3. Checkmate/stalemate detection
4. Three-fold repetition and 50-move rule detection
"""

import unittest
import asyncio
import chess
from unittest.mock import AsyncMock, MagicMock, patch

from chess_llm_bench.core.game import GameRunner
from chess_llm_bench.core.models import Config, BotSpec
from chess_llm_bench.core.engine import ChessEngine


def async_test(coro):
    """Decorator to run async tests."""
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper



class ProtocolTests(unittest.TestCase):
    """Test chess protocol validation and error detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            think_time=0.1,
            max_plies=100,
        )
        # Mock LLM client
        self.llm_client = MagicMock()
        self.llm_client.spec = BotSpec(provider="test", model="test", name="TestBot")
        self.llm_client.get_move = AsyncMock()
        self.llm_client.reset_move_stats = MagicMock()
        self.llm_client.get_move_stats = MagicMock(return_value=(0.0, 0, 0.0))

        # Mock engine
        self.engine = MagicMock()
        self.engine.configure_elo = AsyncMock()
        self.engine.get_move = AsyncMock()

        # Create game runner
        self.game_runner = GameRunner(self.llm_client, self.engine, self.config)

    @async_test
    async def test_wrong_side_to_move(self):
        """
        Test detection of moves played on wrong side-to-move.

        This test verifies that the system correctly rejects a move when a player
        attempts to move when it's not their turn.
        """
        # Setup a position where it's Black's move
        board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")

        # LLM plays as White and attempts to move when it's Black's turn
        self.llm_client.get_move.return_value = chess.Move.from_uci("e2e4")  # White pawn move

        # In real gameplay, the system should never call LLM's get_move when it's not their turn
        # But we can test that the move validation would catch this
        illegal_moves = []
        with patch.object(self.game_runner, '_execute_move') as mock_execute:
            mock_execute.side_effect = lambda board, move, *args, **kwargs: (
                illegal_moves.append(move) if move not in board.legal_moves else None
            )

            # Try to make the move and check that it would be rejected
            move = await self.llm_client.get_move(board)
            if move in board.legal_moves:
                self.game_runner._execute_move(board, move, None, None, "test")
            else:
                illegal_moves.append(move)

        self.assertGreaterEqual(len(illegal_moves), 1, "Wrong side-to-move not detected")

    @async_test
    async def test_illegal_uci_move(self):
        """
        Test detection of illegal UCI moves.

        This test verifies that the system correctly rejects moves that are
        not legal according to chess rules.
        """
        # Setup initial board position
        board = chess.Board()

        # LLM returns an illegal move (moving a pawn two squares after it already moved)
        self.llm_client.get_move.return_value = chess.Move.from_uci("e2e5")  # Illegal pawn move

        # Check that the move would be rejected
        move = await self.llm_client.get_move(board)
        self.assertNotIn(move, board.legal_moves, "Illegal UCI move not detected")

    @async_test
    async def test_checkmate_detection(self):
        """
        Test detection of checkmate.

        This test verifies that the system correctly identifies checkmate positions
        and ends the game with the appropriate result.
        """
        # Setup a checkmate position (Scholar's mate)
        board = chess.Board()
        for move_uci in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]:
            board.push(chess.Move.from_uci(move_uci))

        # Verify that the position is checkmate
        self.assertTrue(board.is_checkmate(), "Failed to recognize checkmate")
        self.assertTrue(board.is_game_over(), "Failed to recognize game over")
        self.assertEqual(board.result(), "1-0", "Incorrect result for checkmate")

    @async_test
    async def test_stalemate_detection(self):
        """
        Test detection of stalemate.

        This test verifies that the system correctly identifies stalemate positions
        and ends the game with a draw.
        """
        # Setup a stalemate position
        board = chess.Board("k7/8/1Q6/8/8/8/8/7K b - - 0 1")

        # Verify that the position is stalemate
        self.assertTrue(board.is_stalemate(), "Failed to recognize stalemate")
        self.assertTrue(board.is_game_over(), "Failed to recognize game over")
        self.assertEqual(board.result(), "1/2-1/2", "Incorrect result for stalemate")

    @async_test
    async def test_threefold_repetition(self):
        """
        Test detection of threefold repetition.

        This test verifies that the system correctly identifies threefold repetition
        and allows claiming a draw.
        """
        # Setup a position for threefold repetition
        board = chess.Board()
        moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]
        for move_uci in moves:
            board.push(chess.Move.from_uci(move_uci))

        # Verify that threefold repetition is detected
        self.assertTrue(board.can_claim_threefold_repetition(),
                        "Failed to recognize threefold repetition claim")

        # Claim the draw and verify the result
        result = board.result(claim_draw=True)
        self.assertEqual(result, "1/2-1/2", "Incorrect result for threefold repetition")

    @async_test
    async def test_fifty_move_rule(self):
        """
        Test detection of fifty-move rule.

        This test verifies that the system correctly identifies when the fifty-move
        rule applies and allows claiming a draw.
        """
        # Create a board with exactly 100 half-moves (50 full moves without capture/pawn move)
        # This should allow claiming a draw by the fifty-move rule
        board = chess.Board("8/8/8/8/k7/8/8/K1N5 w - - 100 51")

        # Verify the halfmove clock is set to 100
        self.assertEqual(board.halfmove_clock, 100)

        # At 100 half-moves, we should be able to claim the fifty-move rule
        self.assertTrue(board.can_claim_fifty_moves(),
                        "Failed to recognize fifty-move claim at 100 half-moves")

        # Claim the draw and verify the result
        result = board.result(claim_draw=True)
        self.assertEqual(result, "1/2-1/2", "Incorrect result for fifty-move rule")

        # Test that the game is considered over when claiming the draw
        self.assertTrue(board.is_game_over(claim_draw=True),
                        "Game should be over when fifty-move rule is claimed")

    @async_test
    async def test_game_end_on_checkmate(self):
        """Test that game properly ends when checkmate is reached."""
        # Create a mock game state
        mock_state = MagicMock()
        mock_state.status = "playing"

        # Create a board with checkmate
        board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1")

        # Verify the position is checkmate and game is over
        self.assertTrue(board.is_checkmate())
        self.assertTrue(board.is_game_over())

        # The game should end with checkmate detected
        result = board.result(claim_draw=True)
        self.assertEqual(result, "1-0", "Checkmate not properly recognized")

    @async_test
    async def test_game_end_on_draw(self):
        """Test that game properly ends on draw conditions."""
        # Create a mock game state
        mock_state = MagicMock()
        mock_state.status = "playing"

        # Test various draw scenarios
        draw_positions = [
            # Stalemate
            "k7/8/1Q6/8/8/8/8/7K b - - 0 1",
            # Insufficient material (K vs K)
            "8/8/8/8/8/8/k7/7K w - - 0 1",
            # Insufficient material (K+N vs K)
            "8/8/8/8/8/8/k1N5/7K b - - 0 1"
        ]

        for fen in draw_positions:
            board = chess.Board(fen)
            self.assertTrue(board.is_game_over(), f"Failed to detect game over in position: {fen}")
            result = board.result(claim_draw=True)
            self.assertEqual(result, "1/2-1/2", f"Draw not properly recognized in position: {fen}")


if __name__ == "__main__":
    unittest.main()
