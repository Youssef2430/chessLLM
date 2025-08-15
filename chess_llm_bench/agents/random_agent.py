"""
Random Player Agent

This module provides a RandomPlayerAgent that plays random legal moves.
It's useful for baseline testing, debugging, and providing a simple opponent
for benchmarking other chess agents.
"""

from __future__ import annotations

import random
import logging
from typing import Optional

import chess
from autogen import LLMConfig

from .base import ChessAgent, ChessAgentConfig
from ..core.models import BotSpec

logger = logging.getLogger(__name__)


class RandomPlayerAgent(ChessAgent):
    """
    Chess agent that plays random legal moves.

    This agent provides a simple baseline for testing and benchmarking.
    It quickly selects random legal moves without any strategic consideration,
    making it useful for:
    - Baseline performance comparisons
    - Debugging game infrastructure
    - Fast testing scenarios
    - Demonstrating minimum viable chess play
    """

    def __init__(
        self,
        name: str = "RandomBot",
        bot_spec: Optional[BotSpec] = None,
        **kwargs
    ):
        """
        Initialize the random player agent.

        Args:
            name: Name of the random agent
            bot_spec: Bot specification (will create default if None)
            **kwargs: Additional arguments passed to parent
        """
        # Create default bot spec if none provided
        if bot_spec is None:
            bot_spec = BotSpec(
                provider="random",
                model="baseline",
                name=name
            )

        # Create configuration for random agent
        config = ChessAgentConfig(
            name=name,
            bot_spec=bot_spec,
            temperature=0.0,  # Not used for random moves
            timeout_seconds=1.0,  # Very fast
            max_thinking_time=0.1,  # Minimal thinking time
            max_retry_attempts=1,  # Random moves are always legal
            enable_fallback=False,  # No fallback needed
        )

        # Initialize parent with no LLM config needed
        super().__init__(
            config=config,
            llm_config=None,  # No LLM needed for random moves
            system_message=self._create_random_system_message(name),
            **kwargs
        )

        # Random state for reproducible testing if needed
        self._random_state = random.Random()

        logger.info(f"Initialized RandomPlayerAgent: {name}")

    def _create_random_system_message(self, name: str) -> str:
        """Create system message for random player."""
        return f"""You are {name}, a chess agent that plays random legal moves.

Your strategy is simple:
1. Look at all legal moves in the current position
2. Select one at random
3. Play it immediately

You don't analyze positions or consider strategy - you're designed to provide
a baseline for testing other chess engines and agents. You play fast and
randomly, which makes you useful for:
- Quick testing and debugging
- Baseline performance measurements
- Demonstrating basic game functionality

Despite playing randomly, you always play legal moves and follow chess rules.
"""

    async def generate_move(
        self,
        board: chess.Board,
        time_limit: Optional[float] = None,
        opponent_last_move: Optional[str] = None,
        **kwargs
    ) -> chess.Move:
        """
        Generate a random legal move.

        Args:
            board: Current chess board position
            time_limit: Ignored for random moves (always fast)
            opponent_last_move: Ignored for random strategy
            **kwargs: Additional parameters (ignored)

        Returns:
            A randomly selected legal move

        Raises:
            ValueError: If no legal moves are available
        """
        # Get all legal moves
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            raise ValueError("No legal moves available - game should be over")

        # Select random move
        selected_move = self._random_state.choice(legal_moves)

        logger.debug(f"RandomPlayerAgent selected move: {selected_move}")

        return selected_move

    def set_random_seed(self, seed: int) -> None:
        """
        Set random seed for reproducible testing.

        Args:
            seed: Random seed value
        """
        self._random_state.seed(seed)
        logger.info(f"Set random seed to {seed} for {self.name}")

    def get_move_count_estimate(self, board: chess.Board) -> int:
        """
        Get estimate of available moves (for testing/debugging).

        Args:
            board: Chess board to analyze

        Returns:
            Number of legal moves available
        """
        return len(list(board.legal_moves))

    def __str__(self) -> str:
        """String representation of random agent."""
        return f"RandomPlayerAgent({self.name})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"RandomPlayerAgent(name='{self.name}', "
            f"moves_played={self._move_stats['move_count']})"
        )


def create_random_agent(name: str = "RandomBot") -> RandomPlayerAgent:
    """
    Convenience function to create a random player agent.

    Args:
        name: Name for the random agent

    Returns:
        Configured RandomPlayerAgent
    """
    return RandomPlayerAgent(name=name)


def create_seeded_random_agent(name: str = "SeededRandomBot", seed: int = 42) -> RandomPlayerAgent:
    """
    Create a random agent with a specific seed for reproducible testing.

    Args:
        name: Name for the random agent
        seed: Random seed for reproducible moves

    Returns:
        Configured RandomPlayerAgent with set seed
    """
    agent = RandomPlayerAgent(name=name)
    agent.set_random_seed(seed)
    return agent
