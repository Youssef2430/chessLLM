"""
AG2-based Game Runner

This module provides AG2-based game runners that use chess agents for playing
games instead of direct LLM client calls. It maintains full compatibility with
the existing ELO ladder system while leveraging AG2's multi-agent framework.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import chess

from .models import Config, GameRecord, LiveState, LadderStats, BotSpec
from .engine import ChessEngine
from .human_engine import HumanLikeEngine
from .adaptive_engine import AdaptiveEngine
from ..agents import (
    GameAgent,
    ChessAgent,
    create_agent,
    AgentCreationError
)

logger = logging.getLogger(__name__)


class AG2GameRunner:
    """
    AG2-based game runner that coordinates chess games using chess agents.

    This class replaces the traditional GameRunner by using AG2 chess agents
    and a GameAgent coordinator instead of direct LLM client calls. It maintains
    full compatibility with the existing interface and statistics tracking.
    """

    def __init__(
        self,
        config: Config,
        engine: Union[ChessEngine, HumanLikeEngine, AdaptiveEngine]
    ):
        """
        Initialize the AG2 game runner.

        Args:
            config: Global configuration object
            engine: Chess engine for opponent moves
        """
        self.config = config
        self.engine = engine

        # Create the game coordinator agent
        self.game_agent = GameAgent(config, engine, name="GameCoordinator")

        # Agent cache to reuse agents across games
        self._agent_cache: Dict[str, ChessAgent] = {}

        # Game statistics
        self._game_counter = 0
        self._total_games_time = 0.0

        logger.info("Initialized AG2GameRunner")

    async def play_game(
        self,
        bot_spec: BotSpec,
        elo: int,
        output_dir: Path,
        state: LiveState,
        llm_plays_white: Optional[bool] = None,
        opening_moves: Optional[List[str]] = None
    ) -> GameRecord:
        """
        Play a single chess game between an agent and engine at specified ELO.

        Args:
            bot_spec: Bot specification for creating the chess agent
            elo: Engine ELO rating for this game
            output_dir: Directory to save PGN files
            state: Live state object for UI updates
            llm_plays_white: Force color assignment (None for alternating)
            opening_moves: List of UCI moves to use as opening (None for no opening)

        Returns:
            GameRecord with game results and metadata

        Raises:
            Exception: If game cannot be completed due to critical errors
        """
        start_time = time.time()

        try:
            # Get or create chess agent
            chess_agent = await self._get_chess_agent(bot_spec)

            # Use GameAgent to coordinate the game
            game_record = await self.game_agent.coordinate_game(
                llm_agent=chess_agent,
                elo=elo,
                output_dir=output_dir,
                state=state,
                llm_plays_white=llm_plays_white,
                opening_moves=opening_moves
            )

            # Update statistics
            self._game_counter += 1
            game_duration = time.time() - start_time
            self._total_games_time += game_duration

            logger.info(
                f"Game {self._game_counter} completed: "
                f"{chess_agent.name} vs Engine({elo}) = {game_record.result} "
                f"in {game_duration:.1f}s"
            )

            return game_record

        except Exception as e:
            logger.error(f"AG2 game failed: {e}")
            raise

    async def _get_chess_agent(self, bot_spec: BotSpec) -> ChessAgent:
        """
        Get or create a chess agent for the bot specification.

        Uses caching to reuse agents across games for better performance.
        """
        # Create cache key
        cache_key = f"{bot_spec.provider}:{bot_spec.model}:{bot_spec.name}"

        # Check cache first
        if cache_key in self._agent_cache:
            agent = self._agent_cache[cache_key]
            # Reset agent statistics for new game
            agent.reset_move_stats()
            return agent

        # Create new agent
        try:
            agent = create_agent(
                bot_spec=bot_spec,
                temperature=self.config.llm_temperature,
                timeout_seconds=self.config.llm_timeout
            )

            # Cache the agent
            self._agent_cache[cache_key] = agent

            logger.info(f"Created and cached new agent: {agent.name}")
            return agent

        except AgentCreationError as e:
            logger.error(f"Failed to create agent for {bot_spec}: {e}")
            raise

    def get_game_stats(self) -> Dict[str, any]:
        """Get comprehensive game statistics."""
        return {
            "games_played": self._game_counter,
            "total_time": self._total_games_time,
            "avg_game_time": (
                self._total_games_time / self._game_counter
                if self._game_counter > 0 else 0.0
            ),
            "cached_agents": len(self._agent_cache),
            "game_agent_stats": self.game_agent.get_game_stats(),
        }

    def reset_stats(self) -> None:
        """Reset game runner statistics."""
        self._game_counter = 0
        self._total_games_time = 0.0
        self.game_agent.reset_game_stats()

        # Reset cached agent statistics
        for agent in self._agent_cache.values():
            agent.reset_move_stats()

    def clear_agent_cache(self) -> None:
        """Clear the agent cache."""
        self._agent_cache.clear()
        logger.info("Cleared agent cache")

    def __str__(self) -> str:
        """String representation of game runner."""
        return f"AG2GameRunner(games={self._game_counter}, agents={len(self._agent_cache)})"


class AG2LadderRunner:
    """
    AG2-based ladder runner for ELO progression testing.

    This class manages the ELO ladder progression using AG2 chess agents,
    maintaining full compatibility with the existing ladder system.
    """

    def __init__(
        self,
        config: Config,
        engine: Union[ChessEngine, HumanLikeEngine, AdaptiveEngine]
    ):
        """
        Initialize the AG2 ladder runner.

        Args:
            config: Global configuration object
            engine: Chess engine for games
        """
        self.config = config
        self.engine = engine
        self.game_runner = AG2GameRunner(config, engine)

        # Ladder state
        self._current_elo = config.start_elo
        self._wins_at_current_elo = 0
        self._total_games = 0

        logger.info(f"Initialized AG2LadderRunner starting at ELO {self._current_elo}")

    async def run_ladder(
        self,
        bot_spec: BotSpec,
        output_dir: Path,
        state: LiveState
    ) -> LadderStats:
        """
        Run the complete ELO ladder for a bot specification.

        Args:
            bot_spec: Bot specification for the chess agent
            output_dir: Directory for saving game files
            state: Live state for UI updates

        Returns:
            Complete ladder statistics
        """
        logger.info(f"Starting ladder run for {bot_spec.name}")

        # Initialize ladder state
        ladder_start_time = time.time()
        games_played = []
        self._current_elo = self.config.start_elo
        self._wins_at_current_elo = 0
        self._total_games = 0

        # Initialize live state
        state.ladder = []
        state.current_bot = bot_spec.name
        state.games_played = 0
        state.ladder_complete = False

        try:
            # Run ladder games
            while (
                self._current_elo <= self.config.max_elo and
                self._total_games < self.config.max_games
            ):
                # Play game at current ELO
                game_record = await self._play_ladder_game(
                    bot_spec, output_dir, state
                )

                games_played.append(game_record)
                self._total_games += 1

                # Update live state
                state.games_played = self._total_games
                state.ladder.append({
                    "elo": game_record.elo,
                    "result": game_record.result,
                    "moves": game_record.moves,
                    "game_duration": game_record.game_duration
                })

                # Check for ELO advancement
                if self._should_advance(game_record):
                    self._advance_elo()
                else:
                    # Reset wins counter on loss/draw
                    if game_record.result != "win":
                        self._wins_at_current_elo = 0

                # Brief pause between games
                await asyncio.sleep(0.1)

            # Finalize ladder
            total_duration = time.time() - ladder_start_time
            state.ladder_complete = True

            # Calculate final statistics
            wins = len([g for g in games_played if g.result == "win"])
            losses = len([g for g in games_played if g.result == "loss"])
            draws = len([g for g in games_played if g.result == "draw"])

            effective_elo = self.get_effective_elo()

            ladder_stats = LadderStats(
                bot_name=bot_spec.name,
                games_played=len(games_played),
                wins=wins,
                losses=losses,
                draws=draws,
                final_elo=self._current_elo,
                effective_elo=effective_elo,
                total_duration=total_duration,
                games=games_played
            )

            logger.info(
                f"Ladder complete for {bot_spec.name}: "
                f"{wins}W-{losses}L-{draws}D, "
                f"Final ELO: {self._current_elo}, "
                f"Effective ELO: {effective_elo}"
            )

            return ladder_stats

        except Exception as e:
            logger.error(f"Ladder run failed for {bot_spec.name}: {e}")
            raise

    async def _play_ladder_game(
        self,
        bot_spec: BotSpec,
        output_dir: Path,
        state: LiveState
    ) -> GameRecord:
        """Play a single ladder game."""
        return await self.game_runner.play_game(
            bot_spec=bot_spec,
            elo=self._current_elo,
            output_dir=output_dir,
            state=state,
            llm_plays_white=None,  # Alternate colors
            opening_moves=None
        )

    def _should_advance(self, game_record: GameRecord) -> bool:
        """
        Determine if the bot should advance to the next ELO level.

        Args:
            game_record: Latest game result

        Returns:
            True if bot should advance to next ELO
        """
        if game_record.result == "win":
            self._wins_at_current_elo += 1
            return self._wins_at_current_elo >= self.config.wins_required
        else:
            return False

    def _advance_elo(self) -> None:
        """Advance to the next ELO level."""
        old_elo = self._current_elo
        self._current_elo += self.config.elo_step
        self._wins_at_current_elo = 0

        logger.info(f"Advanced ELO: {old_elo} -> {self._current_elo}")

    def get_effective_elo(self) -> int:
        """
        Calculate effective ELO based on current progress.

        Returns:
            Effective ELO rating considering partial progress
        """
        if self._wins_at_current_elo == 0:
            # No wins at current level, effective ELO is previous level
            return max(self.config.start_elo, self._current_elo - self.config.elo_step)
        else:
            # Partial progress at current level
            progress_ratio = self._wins_at_current_elo / self.config.wins_required
            previous_elo = max(self.config.start_elo, self._current_elo - self.config.elo_step)
            return int(previous_elo + (self._current_elo - previous_elo) * progress_ratio)

    def get_ladder_stats(self) -> Dict[str, any]:
        """Get current ladder statistics."""
        return {
            "current_elo": self._current_elo,
            "wins_at_current_elo": self._wins_at_current_elo,
            "total_games": self._total_games,
            "effective_elo": self.get_effective_elo(),
            "game_runner_stats": self.game_runner.get_game_stats(),
        }

    def reset_ladder(self) -> None:
        """Reset ladder to initial state."""
        self._current_elo = self.config.start_elo
        self._wins_at_current_elo = 0
        self._total_games = 0
        self.game_runner.reset_stats()

    def __str__(self) -> str:
        """String representation of ladder runner."""
        return (
            f"AG2LadderRunner(ELO={self._current_elo}, "
            f"wins={self._wins_at_current_elo}, "
            f"games={self._total_games})"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"AG2LadderRunner(current_elo={self._current_elo}, "
            f"wins_at_current_elo={self._wins_at_current_elo}, "
            f"total_games={self._total_games}, "
            f"effective_elo={self.get_effective_elo()})"
        )
