"""
Unit tests for chess engine functionality.

Tests the ChessEngine class and related utility functions to ensure proper
UCI engine management, ELO configuration, and error handling.
"""

import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import os
import tempfile
from pathlib import Path

import chess

from chess_llm_bench.core.engine import (
    ChessEngine,
    EngineError,
    autodetect_stockfish,
    validate_engine,
    get_engine_info,
    get_friendly_stockfish_hint
)
from chess_llm_bench.core.models import Config


class ChessEngineTests(unittest.TestCase):
    """Test ChessEngine class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.engine_path = "/fake/stockfish"

    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = ChessEngine(self.engine_path, self.config)
        self.assertEqual(engine.engine_path, self.engine_path)
        self.assertEqual(engine.config, self.config)
        self.assertFalse(engine.is_running)
        self.assertIsNone(engine.current_elo)

    @patch('chess_llm_bench.core.engine.chess_engine.SimpleEngine.popen_uci')
    async def test_engine_start_success(self, mock_popen):
        """Test successful engine start."""
        mock_engine = Mock()
        mock_engine.id = {"name": "Stockfish 15"}
        mock_popen.return_value = mock_engine

        engine = ChessEngine(self.engine_path, self.config)
        await engine.start()

        self.assertTrue(engine.is_running)
        mock_popen.assert_called_once_with(self.engine_path)

    @patch('chess_llm_bench.core.engine.chess_engine.SimpleEngine.popen_uci')
    async def test_engine_start_failure(self, mock_popen):
        """Test engine start failure."""
        mock_popen.side_effect = Exception("Engine not found")

        engine = ChessEngine(self.engine_path, self.config)

        with self.assertRaises(EngineError) as context:
            await engine.start()

        self.assertIn("Failed to start engine", str(context.exception))
        self.assertFalse(engine.is_running)

    @patch('chess_llm_bench.core.engine.chess_engine.SimpleEngine.popen_uci')
    async def test_engine_stop(self, mock_popen):
        """Test engine stop."""
        mock_engine = Mock()
        mock_engine.id = {"name": "Stockfish"}
        mock_popen.return_value = mock_engine

        engine = ChessEngine(self.engine_path, self.config)
        await engine.start()
        await engine.stop()

        self.assertFalse(engine.is_running)
        mock_engine.quit.assert_called_once()

    @patch('chess_llm_bench.core.engine.chess_engine.SimpleEngine.popen_uci')
    async def test_engine_stop_with_error(self, mock_popen):
        """Test engine stop with quit error."""
        mock_engine = Mock()
        mock_engine.id = {"name": "Stockfish"}
        mock_engine.quit.side_effect = Exception("Quit failed")
        mock_popen.return_value = mock_engine

        engine = ChessEngine(self.engine_path, self.config)
        await engine.start()

        # Should not raise exception, just log warning
        await engine.stop()
        self.assertFalse(engine.is_running)

    @patch('chess_llm_bench.core.engine.chess_engine.SimpleEngine.popen_uci')
    @patch('chess_llm_bench.core.engine.asyncio.to_thread')
    async def test_configure_elo_success(self, mock_to_thread, mock_popen):
        """Test successful ELO configuration."""
        mock_engine = Mock()
        mock_engine.id = {"name": "Stockfish"}
        mock_popen.return_value = mock_engine
        mock_to_thread.return_value = None

        engine = ChessEngine(self.engine_path, self.config)
        await engine.start()
        await engine.configure_elo(1200)

        self.assertEqual(engine.current_elo, 1200)
        mock_to_thread.assert_called()

    @patch('chess_llm_bench.core.engine.chess_engine.SimpleEngine.popen_uci')
    @patch('chess_llm_bench.core.engine.asyncio.to_thread')
    async def test_configure_elo_fallback(self, mock_to_thread, mock_popen):
        """Test ELO configuration fallback to skill level."""
        mock_engine = Mock()
        mock_engine.id = {"name": "Stockfish"}
        mock_popen.return_value = mock_engine

        # First call fails (UCI_LimitStrength), second succeeds (Skill Level)
        mock_to_thread.side_effect = [Exception("UCI_Elo not supported"), None]

        engine = ChessEngine(self.engine_path, self.config)
        await engine.start()
        await engine.configure_elo(1200)

        self.assertEqual(engine.current_elo, 1200)
        self.assertEqual(mock_to_thread.call_count, 2)

    @patch('chess_llm_bench.core.engine.chess_engine.SimpleEngine.popen_uci')
    @patch('chess_llm_bench.core.engine.asyncio.to_thread')
    async def test_configure_elo_no_change(self, mock_to_thread, mock_popen):
        """Test ELO configuration when already at target ELO."""
        mock_engine = Mock()
        mock_engine.id = {"name": "Stockfish"}
        mock_popen.return_value = mock_engine

        engine = ChessEngine(self.engine_path, self.config)
        await engine.start()
        engine._current_elo = 1200

        await engine.configure_elo(1200)

        # Should not call configure since ELO unchanged
        mock_to_thread.assert_not_called()

    async def test_configure_elo_engine_not_started(self):
        """Test ELO configuration when engine not started."""
        engine = ChessEngine(self.engine_path, self.config)

        with self.assertRaises(EngineError) as context:
            await engine.configure_elo(1200)

        self.assertIn("Engine not started", str(context.exception))

    @patch('chess_llm_bench.core.engine.chess_engine.SimpleEngine.popen_uci')
    @patch('chess_llm_bench.core.engine.asyncio.to_thread')
    async def test_get_move_success(self, mock_to_thread, mock_popen):
        """Test successful move generation."""
        mock_engine = Mock()
        mock_engine.id = {"name": "Stockfish"}
        mock_popen.return_value = mock_engine

        mock_result = Mock()
        mock_result.move = chess.Move.from_uci("e2e4")
        mock_to_thread.return_value = mock_result

        engine = ChessEngine(self.engine_path, self.config)
        await engine.start()

        board = chess.Board()
        move = await engine.get_move(board)

        self.assertEqual(move.uci(), "e2e4")
        mock_to_thread.assert_called_once()

    async def test_get_move_engine_not_started(self):
        """Test move generation when engine not started."""
        engine = ChessEngine(self.engine_path, self.config)
        board = chess.Board()

        with self.assertRaises(EngineError) as context:
            await engine.get_move(board)

        self.assertIn("Engine not started", str(context.exception))

    @patch('chess_llm_bench.core.engine.chess_engine.SimpleEngine.popen_uci')
    async def test_get_move_game_over(self, mock_popen):
        """Test move generation on finished game."""
        mock_engine = Mock()
        mock_engine.id = {"name": "Stockfish"}
        mock_popen.return_value = mock_engine

        engine = ChessEngine(self.engine_path, self.config)
        await engine.start()

        # Create a finished game
        board = chess.Board()
        board.push(chess.Move.from_uci("f2f3"))
        board.push(chess.Move.from_uci("e7e5"))
        board.push(chess.Move.from_uci("g2g4"))
        board.push(chess.Move.from_uci("d8h4"))  # Checkmate

        with self.assertRaises(EngineError) as context:
            await engine.get_move(board)

        self.assertIn("Cannot get move for finished game", str(context.exception))

    @patch('chess_llm_bench.core.engine.chess_engine.SimpleEngine.popen_uci')
    @patch('chess_llm_bench.core.engine.asyncio.to_thread')
    async def test_get_move_engine_error(self, mock_to_thread, mock_popen):
        """Test move generation with engine error."""
        mock_engine = Mock()
        mock_engine.id = {"name": "Stockfish"}
        mock_popen.return_value = mock_engine
        mock_to_thread.side_effect = Exception("Engine crashed")

        engine = ChessEngine(self.engine_path, self.config)
        await engine.start()

        board = chess.Board()

        with self.assertRaises(EngineError) as context:
            await engine.get_move(board)

        self.assertIn("Engine move generation failed", str(context.exception))

    @patch('chess_llm_bench.core.engine.chess_engine.SimpleEngine.popen_uci')
    async def test_context_manager(self, mock_popen):
        """Test engine as async context manager."""
        mock_engine = Mock()
        mock_engine.id = {"name": "Stockfish"}
        mock_popen.return_value = mock_engine

        engine = ChessEngine(self.engine_path, self.config)

        async with engine as ctx_engine:
            self.assertIs(ctx_engine, engine)
            self.assertTrue(engine.is_running)

        self.assertFalse(engine.is_running)
        mock_engine.quit.assert_called_once()


class StockfishDetectionTests(unittest.TestCase):
    """Test Stockfish auto-detection functionality."""

    def test_autodetect_explicit_path(self):
        """Test detection with explicit valid path."""
        with tempfile.NamedTemporaryFile() as temp_file:
            result = autodetect_stockfish(temp_file.name)
            self.assertEqual(result, temp_file.name)

    def test_autodetect_explicit_invalid_path(self):
        """Test detection with explicit invalid path."""
        result = autodetect_stockfish("/nonexistent/path")
        # Should fall back to other methods
        self.assertNotEqual(result, "/nonexistent/path")

    @patch.dict(os.environ, {"STOCKFISH_PATH": "/env/stockfish"})
    @patch('pathlib.Path.exists')
    def test_autodetect_env_variable(self, mock_exists):
        """Test detection via environment variable."""
        mock_exists.return_value = True
        result = autodetect_stockfish()
        self.assertEqual(result, "/env/stockfish")

    @patch('shutil.which')
    def test_autodetect_system_path(self, mock_which):
        """Test detection via system PATH."""
        mock_which.return_value = "/usr/bin/stockfish"
        result = autodetect_stockfish()
        self.assertEqual(result, "/usr/bin/stockfish")

    @patch('pathlib.Path.exists')
    def test_autodetect_common_paths(self, mock_exists):
        """Test detection via common installation paths."""
        # Mock the first common path to exist
        def exists_side_effect(self):
            return str(self) == "/usr/local/bin/stockfish"

        mock_exists.side_effect = exists_side_effect

        result = autodetect_stockfish()
        self.assertEqual(result, "/usr/local/bin/stockfish")

    @patch('shutil.which')
    @patch('pathlib.Path.exists')
    def test_autodetect_not_found(self, mock_exists, mock_which):
        """Test detection when Stockfish not found anywhere."""
        mock_exists.return_value = False
        mock_which.return_value = None

        result = autodetect_stockfish()
        self.assertIsNone(result)


class EngineValidationTests(unittest.TestCase):
    """Test engine validation functionality."""

    @patch('subprocess.run')
    def test_validate_engine_success(self, mock_run):
        """Test successful engine validation."""
        mock_result = Mock()
        mock_result.stdout = "id name Stockfish 15\nuciok\nreadyok"
        mock_run.return_value = mock_result

        result = validate_engine("/path/to/stockfish")
        self.assertTrue(result)

        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(args[0][0], "/path/to/stockfish")
        self.assertEqual(kwargs["input"], "uci\nquit\n")

    @patch('subprocess.run')
    def test_validate_engine_no_uciok(self, mock_run):
        """Test engine validation without uciok response."""
        mock_result = Mock()
        mock_result.stdout = "id name BadEngine\nreadyok"
        mock_run.return_value = mock_result

        result = validate_engine("/path/to/badengine")
        self.assertFalse(result)

    @patch('subprocess.run')
    def test_validate_engine_subprocess_error(self, mock_run):
        """Test engine validation with subprocess error."""
        mock_run.side_effect = Exception("Process failed")

        result = validate_engine("/path/to/nonexistent")
        self.assertFalse(result)

    @patch('subprocess.run')
    def test_validate_engine_timeout(self, mock_run):
        """Test engine validation timeout."""
        mock_run.side_effect = Exception("Timeout")

        result = validate_engine("/path/to/slow_engine")
        self.assertFalse(result)


class EngineInfoTests(unittest.TestCase):
    """Test engine information extraction."""

    @patch('subprocess.run')
    def test_get_engine_info_success(self, mock_run):
        """Test successful engine info extraction."""
        mock_result = Mock()
        mock_result.stdout = (
            "id name Stockfish 15\n"
            "id author T. Romstad, M. Costalba, J. Kiiski, G. Linscott\n"
            "option name Hash type spin default 16 min 1 max 33554432\n"
            "option name Threads type spin default 1 min 1 max 1024\n"
            "uciok\n"
        )
        mock_run.return_value = mock_result

        info = get_engine_info("/path/to/stockfish")

        self.assertEqual(info["name"], "Stockfish 15")
        self.assertEqual(info["author"], "T. Romstad, M. Costalba, J. Kiiski, G. Linscott")
        self.assertIn("options", info)
        self.assertEqual(len(info["options"]), 2)
        self.assertIn("Hash type spin", info["options"][0])

    @patch('subprocess.run')
    def test_get_engine_info_minimal(self, mock_run):
        """Test engine info with minimal response."""
        mock_result = Mock()
        mock_result.stdout = "uciok\n"
        mock_run.return_value = mock_result

        info = get_engine_info("/path/to/minimal_engine")

        # Should return empty dict for missing info
        self.assertNotIn("name", info)
        self.assertNotIn("author", info)

    @patch('subprocess.run')
    def test_get_engine_info_error(self, mock_run):
        """Test engine info extraction with error."""
        mock_run.side_effect = Exception("Process failed")

        info = get_engine_info("/path/to/nonexistent")
        self.assertEqual(info, {})


class UtilityTests(unittest.TestCase):
    """Test utility functions."""

    def test_friendly_stockfish_hint(self):
        """Test Stockfish installation hint message."""
        hint = get_friendly_stockfish_hint()
        self.assertIn("Stockfish not found", hint)
        self.assertIn("macOS", hint)
        self.assertIn("Ubuntu", hint)
        self.assertIn("Windows", hint)
        self.assertIn("STOCKFISH_PATH", hint)


if __name__ == "__main__":
    unittest.main()
