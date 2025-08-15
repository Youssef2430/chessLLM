"""
Base Chess Agent Implementation

This module provides the foundational ChessAgent class that integrates with AG2
(AutoGen) framework for multi-agent chess playing. All chess agents inherit
from this base class to ensure consistent behavior and statistics tracking.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import chess
from autogen import ConversableAgent, LLMConfig

from ..core.models import BotSpec, LiveState
from ..llm.client import LLMProviderError

logger = logging.getLogger(__name__)


@dataclass
class ChessAgentConfig:
    """Configuration for chess agents."""

    # Basic agent configuration
    name: str
    bot_spec: BotSpec

    # Chess-specific settings
    temperature: float = 0.7
    timeout_seconds: float = 30.0
    max_thinking_time: float = 10.0

    # AG2 specific settings
    max_consecutive_auto_reply: int = 1
    human_input_mode: str = "NEVER"
    code_execution_config: bool = False

    # Move generation settings
    enable_fallback: bool = True
    max_retry_attempts: int = 3

    # Statistics tracking
    track_move_stats: bool = True

    def to_ag2_config(self) -> Dict[str, Any]:
        """Convert to AG2 ConversableAgent configuration."""
        return {
            "name": self.name,
            "max_consecutive_auto_reply": self.max_consecutive_auto_reply,
            "human_input_mode": self.human_input_mode,
            "code_execution_config": self.code_execution_config,
        }


class ChessAgent(ConversableAgent, ABC):
    """
    Base class for all chess-playing agents using AG2 framework.

    This class provides the foundation for chess agents that can:
    - Generate moves using various strategies (LLM, random, etc.)
    - Track move statistics and performance
    - Integrate with AG2's multi-agent communication
    - Maintain compatibility with existing ELO ladder system
    """

    def __init__(
        self,
        config: ChessAgentConfig,
        llm_config: Optional[LLMConfig] = None,
        system_message: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize chess agent with AG2 integration.

        Args:
            config: Chess agent configuration
            llm_config: AG2 LLM configuration (if applicable)
            system_message: Custom system message for the agent
            **kwargs: Additional AG2 ConversableAgent arguments
        """
        self.chess_config = config
        self.bot_spec = config.bot_spec

        # Merge AG2 config with any additional kwargs
        ag2_config = config.to_ag2_config()
        ag2_config.update(kwargs)

        # Set default system message if none provided
        if system_message is None:
            system_message = self._create_default_system_message()

        # Initialize AG2 ConversableAgent
        super().__init__(
            llm_config=llm_config,
            system_message=system_message,
            **ag2_config
        )

        # Chess-specific state
        self._current_board: Optional[chess.Board] = None
        self._game_history: List[chess.Move] = []

        # Move statistics (maintaining compatibility with existing system)
        self._move_stats = {
            "total_time": 0.0,
            "move_count": 0,
            "illegal_attempts": 0,
            "timeout_count": 0,
            "error_count": 0,
        }

        logger.info(f"Initialized chess agent: {self.name} ({config.bot_spec.provider})")

    def _create_default_system_message(self) -> str:
        """Create default system message for chess playing."""
        return f"""You are {self.name}, a chess-playing AI agent.

Your primary task is to analyze chess positions and suggest the best moves.
You should:
1. Carefully analyze the current board position
2. Consider tactical and strategic factors
3. Provide moves in standard algebraic notation (SAN) or UCI format
4. Think step by step through your reasoning
5. Aim to play strong, competitive chess

You are participating in a chess engine benchmark where your moves will be
evaluated against various ELO-rated opponents. Play your best chess!

Current configuration:
- Provider: {self.bot_spec.provider}
- Model: {self.bot_spec.model}
- Temperature: {self.chess_config.temperature}
"""

    @abstractmethod
    async def generate_move(
        self,
        board: chess.Board,
        time_limit: Optional[float] = None,
        opponent_last_move: Optional[str] = None,
        **kwargs
    ) -> chess.Move:
        """
        Generate a chess move for the given position.

        This is the core method that each agent type must implement.

        Args:
            board: Current chess board position
            time_limit: Maximum time to spend generating move
            opponent_last_move: The opponent's last move (if any)
            **kwargs: Additional move generation parameters

        Returns:
            A legal chess move

        Raises:
            LLMProviderError: If move generation fails
            TimeoutError: If time limit is exceeded
        """
        pass

    async def pick_move(
        self,
        board: chess.Board,
        temperature: Optional[float] = None,
        timeout_s: Optional[float] = None,
        opponent_move: Optional[str] = None,
        state: Optional[LiveState] = None,
    ) -> chess.Move:
        """
        Pick a move with statistics tracking and error handling.

        This method maintains compatibility with the existing LLMClient.pick_move
        interface while adding AG2 agent capabilities.

        Args:
            board: Current chess board position
            temperature: Move generation temperature (if supported)
            timeout_s: Timeout in seconds
            opponent_move: Opponent's last move
            state: Live state for UI updates

        Returns:
            A legal chess move
        """
        start_time = time.time()
        move_found = False
        move = None

        # Update current board state
        self._current_board = board.copy()

        # Use provided timeout or default
        time_limit = timeout_s if timeout_s is not None else self.chess_config.timeout_seconds

        # Update temperature if provided
        if temperature is not None:
            # Store original temperature
            original_temp = self.chess_config.temperature
            self.chess_config.temperature = temperature

        try:
            # Update live state if provided
            if state:
                state.status = f"{state.status} ({self.name} thinking...)"

            # Attempt move generation with retries
            for attempt in range(self.chess_config.max_retry_attempts):
                try:
                    move = await self.generate_move(
                        board=board,
                        time_limit=time_limit,
                        opponent_last_move=opponent_move,
                        temperature=temperature,
                    )

                    # Validate move is legal
                    if move in board.legal_moves:
                        move_found = True
                        break
                    else:
                        self._move_stats["illegal_attempts"] += 1
                        logger.warning(f"Illegal move generated: {move}")

                except Exception as e:
                    logger.warning(f"Move generation attempt {attempt + 1} failed: {e}")
                    self._move_stats["error_count"] += 1

                    if attempt == self.chess_config.max_retry_attempts - 1:
                        raise

            # Fallback to random move if needed
            if not move_found and self.chess_config.enable_fallback:
                logger.warning("Falling back to random move")
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    move = legal_moves[0]  # Simple fallback - could be randomized
                    move_found = True

            if not move_found:
                raise LLMProviderError("Failed to generate legal move after all attempts")

        except asyncio.TimeoutError:
            self._move_stats["timeout_count"] += 1
            raise
        except Exception as e:
            self._move_stats["error_count"] += 1
            raise
        finally:
            # Restore original temperature if it was changed
            if temperature is not None:
                self.chess_config.temperature = original_temp

            # Update statistics
            elapsed_time = time.time() - start_time
            if self.chess_config.track_move_stats:
                self._record_move_stats(elapsed_time)

        return move

    def _record_move_stats(self, elapsed_time: float) -> None:
        """Record move generation statistics."""
        self._move_stats["total_time"] += elapsed_time
        self._move_stats["move_count"] += 1

    def get_move_stats(self) -> tuple[float, int, float]:
        """
        Get move statistics in format compatible with existing system.

        Returns:
            Tuple of (total_time, illegal_attempts, avg_time)
        """
        stats = self._move_stats
        avg_time = (
            stats["total_time"] / stats["move_count"]
            if stats["move_count"] > 0 else 0.0
        )
        return stats["total_time"], stats["illegal_attempts"], avg_time

    def reset_move_stats(self) -> None:
        """Reset move statistics for a new game."""
        self._move_stats = {
            "total_time": 0.0,
            "move_count": 0,
            "illegal_attempts": 0,
            "timeout_count": 0,
            "error_count": 0,
        }

    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed statistics including new metrics."""
        stats = self._move_stats.copy()
        if stats["move_count"] > 0:
            stats["avg_time"] = stats["total_time"] / stats["move_count"]
        else:
            stats["avg_time"] = 0.0
        return stats

    def update_game_state(self, board: chess.Board, last_move: Optional[chess.Move] = None) -> None:
        """Update agent's game state awareness."""
        self._current_board = board.copy()
        if last_move and last_move not in self._game_history:
            self._game_history.append(last_move)

    def get_current_board(self) -> Optional[chess.Board]:
        """Get current board state."""
        return self._current_board.copy() if self._current_board else None

    def get_game_history(self) -> List[chess.Move]:
        """Get game move history."""
        return self._game_history.copy()

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"ChessAgent({self.name}, {self.bot_spec.provider}:{self.bot_spec.model})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"ChessAgent(name='{self.name}', "
            f"provider='{self.bot_spec.provider}', "
            f"model='{self.bot_spec.model}', "
            f"moves_played={self._move_stats['move_count']})"
        )
