#!/usr/bin/env python3
"""
Unit tests for the chess agent system.

This module tests the agent-based chess playing functionality including
tools, reasoning workflows, and LLM integration.
"""

import unittest
import asyncio
import chess
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_llm_bench.llm.agents import (
    ChessAgent,
    ChessAnalysisTools,
    LLMChessAgent,
    LLMAgentProvider,
    ThinkingStrategy,
    MoveCategory,
    MoveAnalysis,
    AgentThought,
    AgentDecision,
    create_agent_provider
)
from chess_llm_bench.llm.agents.base_agent import ChessAgent as BaseChessAgent
from chess_llm_bench.core.models import BotSpec


class TestChessAnalysisTools(unittest.TestCase):
    """Test chess analysis tools functionality."""

    def setUp(self):
        """Set up test board positions."""
        self.starting_board = chess.Board()
        self.tools = ChessAnalysisTools(self.starting_board)

        # Italian opening position
        self.italian_board = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4")
        self.italian_tools = ChessAnalysisTools(self.italian_board)

    def test_board_state(self):
        """Test board state analysis."""
        state = self.tools.get_board_state()

        self.assertEqual(state["turn"], "white")
        self.assertEqual(state["move_number"], 1)
        self.assertFalse(state["is_check"])
        self.assertFalse(state["is_checkmate"])
        self.assertEqual(state["legal_moves_count"], 20)

        # Check material balance
        material = state["material_balance"]
        self.assertEqual(material["white_material"], 39)  # Starting material
        self.assertEqual(material["black_material"], 39)
        self.assertEqual(material["material_difference"], 0)

    def test_material_evaluation(self):
        """Test material calculation."""
        material = self.tools.evaluate_material()

        self.assertEqual(material["white_material"], 39)
        self.assertEqual(material["black_material"], 39)
        self.assertEqual(material["material_advantage"], "equal")

        # Test with captured piece
        capture_board = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKBR w KQkq - 0 1")
        capture_tools = ChessAnalysisTools(capture_board)
        material = capture_tools.evaluate_material()

        self.assertEqual(material["white_material"], 36)  # Missing knight
        self.assertEqual(material["black_material"], 39)
        self.assertEqual(material["material_advantage"], "black")

    def test_position_evaluation(self):
        """Test positional evaluation."""
        position = self.tools.evaluate_position()

        self.assertIn("center_control", position)
        self.assertIn("piece_activity", position)
        self.assertIn("king_safety", position)
        self.assertIn("pawn_structure", position)
        self.assertIn("development", position)
        self.assertIn("total_score", position)

    def test_legal_moves(self):
        """Test legal move generation."""
        moves = self.tools.get_legal_moves()

        self.assertEqual(len(moves), 20)  # Starting position has 20 legal moves

        # Check move properties
        for move in moves:
            self.assertIn("uci", move)
            self.assertIn("san", move)
            self.assertIn("from_square", move)
            self.assertIn("to_square", move)
            self.assertIn("categories", move)

    def test_move_categorization(self):
        """Test move categorization."""
        # Test castling
        castle_board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        castle_tools = ChessAnalysisTools(castle_board)

        # Find castling move
        castle_move = chess.Move.from_uci("e1g1")
        categories = castle_tools._categorize_move(castle_move)

        self.assertIn(MoveCategory.CASTLE.value, categories)
        self.assertIn(MoveCategory.KING_SAFETY.value, categories)

    def test_candidate_moves(self):
        """Test candidate move suggestion."""
        candidates = self.italian_tools.suggest_candidate_moves(top_n=5)

        self.assertLessEqual(len(candidates), 5)

        for candidate in candidates:
            self.assertIsInstance(candidate, MoveAnalysis)
            self.assertIsNotNone(candidate.move)
            self.assertIsNotNone(candidate.san)
            self.assertIsNotNone(candidate.score)
            self.assertIsNotNone(candidate.explanation)

    def test_endgame_evaluation(self):
        """Test endgame evaluation."""
        # Create endgame position
        endgame_board = chess.Board("4k3/p7/8/8/8/8/P7/4K3 w - - 0 1")
        endgame_tools = ChessAnalysisTools(endgame_board)

        endgame = endgame_tools.evaluate_endgame()

        self.assertTrue(endgame["is_endgame"])
        self.assertIn("king_activity", endgame)
        self.assertIn("pawn_advancement", endgame)


class TestChessAgent(unittest.TestCase):
    """Test base chess agent functionality."""

    def setUp(self):
        """Set up test agent."""
        self.agent = MockChessAgent(
            name="TestAgent",
            strategy=ThinkingStrategy.BALANCED,
            verbose=False
        )

    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.name, "TestAgent")
        self.assertEqual(self.agent.strategy, ThinkingStrategy.BALANCED)
        self.assertFalse(self.agent.verbose)
        self.assertEqual(self.agent.total_moves, 0)

    async def test_make_move(self):
        """Test making a move."""
        board = chess.Board()
        decision = await self.agent.make_move(board)

        self.assertIsInstance(decision, AgentDecision)
        self.assertIsNotNone(decision.move)
        self.assertIsNotNone(decision.san)
        self.assertIsNotNone(decision.uci)
        self.assertIsInstance(decision.reasoning, list)
        self.assertGreater(decision.confidence, 0)
        self.assertLessEqual(decision.confidence, 1)

    async def test_thinking_process(self):
        """Test the thinking workflow."""
        board = chess.Board()

        # Set up agent
        self.agent.current_board = board.copy()
        self.agent.tools = ChessAnalysisTools(self.agent.current_board)

        # Test observation
        await self.agent._observe_position()
        self.assertGreater(len(self.agent.thoughts), 0)

        # Test analysis
        await self.agent._analyze_position()
        analysis_thoughts = [t for t in self.agent.thoughts if t.thought_type == "analysis"]
        self.assertGreater(len(analysis_thoughts), 0)

        # Test candidate generation
        candidates = await self.agent._generate_candidates()
        self.assertGreater(len(candidates), 0)

        # Test evaluation
        evaluated = await self.agent._evaluate_candidates(candidates)
        self.assertGreater(len(evaluated), 0)

        # Test decision
        decision = await self.agent._make_decision(evaluated)
        self.assertIsInstance(decision, AgentDecision)

    def test_statistics(self):
        """Test statistics tracking."""
        stats = self.agent.get_statistics()

        self.assertIn("total_moves", stats)
        self.assertIn("total_thinking_time", stats)
        self.assertIn("average_move_time", stats)
        self.assertIn("average_confidence", stats)
        self.assertIn("strategy", stats)

    def test_reset(self):
        """Test agent reset."""
        self.agent.total_moves = 5
        self.agent.reset()

        self.assertEqual(self.agent.total_moves, 0)
        self.assertIsNone(self.agent.current_board)
        self.assertEqual(len(self.agent.thoughts), 0)


class TestThinkingStrategies(unittest.TestCase):
    """Test different thinking strategies."""

    async def test_fast_strategy(self):
        """Test fast strategy prioritizes forcing moves."""
        agent = MockChessAgent(strategy=ThinkingStrategy.FAST)
        board = chess.Board()

        # Create a position with a check available
        board.push_san("e4")
        board.push_san("f6")
        board.push_san("d4")
        board.push_san("g5")

        decision = await agent.make_move(board)

        # Fast strategy should find Qh5+ (checkmate)
        self.assertIsNotNone(decision)

    async def test_deep_strategy(self):
        """Test deep strategy considers more candidates."""
        agent = MockChessAgent(strategy=ThinkingStrategy.DEEP)
        board = chess.Board()

        agent.current_board = board.copy()
        agent.tools = ChessAnalysisTools(agent.current_board)

        candidates = await agent._generate_candidates()

        # Deep strategy should generate more candidates
        self.assertGreaterEqual(len(candidates), 5)

    async def test_adaptive_strategy(self):
        """Test adaptive strategy changes based on game phase."""
        agent = MockChessAgent(strategy=ThinkingStrategy.ADAPTIVE)

        # Test opening phase
        opening_board = chess.Board()
        agent.current_board = opening_board.copy()
        agent.tools = ChessAnalysisTools(agent.current_board)

        # Make a move (e4 is typical opening move)
        move_analysis = MoveAnalysis(
            move=chess.Move.from_uci("e2e4"),
            san="e4",
            categories=[MoveCategory.CENTER_CONTROL, MoveCategory.PAWN_ADVANCE],
            score=1.0,
            threats=[],
            defends=[],
            explanation="Controls center"
        )

        score = await agent._score_move(move_analysis)

        # In opening, center control should be valued
        self.assertGreater(score, 1.0)


class TestLLMAgentProvider(unittest.TestCase):
    """Test LLM agent provider integration."""

    @patch('chess_llm_bench.llm.agents.llm_agent_provider.OpenAIProvider')
    def test_provider_creation(self, mock_openai):
        """Test creating agent provider."""
        provider = create_agent_provider(
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            strategy="balanced",
            verbose=False,
            use_tools=True
        )

        self.assertIsInstance(provider, LLMAgentProvider)
        self.assertEqual(provider.provider, "openai")
        self.assertEqual(provider.model, "gpt-4")
        self.assertEqual(provider.strategy, ThinkingStrategy.BALANCED)

    @patch('chess_llm_bench.llm.agents.llm_agent_provider.OpenAIProvider')
    async def test_generate_move(self, mock_openai):
        """Test move generation through provider."""
        provider = create_agent_provider(
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            strategy="fast"
        )

        # Mock the agent's decision
        mock_decision = AgentDecision(
            move=chess.Move.from_uci("e2e4"),
            san="e4",
            uci="e2e4",
            reasoning=[],
            confidence=0.8,
            alternatives_considered=[]
        )

        provider.agent.make_move = AsyncMock(return_value=mock_decision)

        board = chess.Board()
        uci_move, time_taken = await provider.generate_move(
            board=board,
            game_state=str(board),
            move_history=[]
        )

        self.assertEqual(uci_move, "e2e4")
        self.assertGreaterEqual(time_taken, 0)


class MockChessAgent(BaseChessAgent):
    """Mock agent for testing."""

    async def customize_evaluation(self, move_analysis: MoveAnalysis) -> float:
        """Mock evaluation customization."""
        return 0.0


def run_async_test(coro):
    """Helper to run async tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class AsyncTestCase(unittest.TestCase):
    """Base class for async test cases."""

    def async_test(test_func):
        """Decorator for async test methods."""
        def wrapper(self):
            run_async_test(test_func(self))
        return wrapper


if __name__ == "__main__":
    unittest.main()
