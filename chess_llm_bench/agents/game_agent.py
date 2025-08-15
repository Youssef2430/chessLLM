"""
Game Management Agent

This module provides the GameAgent class that coordinates chess games between
different types of chess agents using the AG2 framework. It handles game flow,
move validation, statistics tracking, and integration with the existing
chess engine and ELO ladder systems.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
import chess.pgn
from autogen import ConversableAgent, LLMConfig

from .base import ChessAgent, ChessAgentConfig
from ..core.models import BotSpec, LiveState, GameRecord, Config
from ..core.engine import ChessEngine
from ..core.human_engine import HumanLikeEngine
from ..core.adaptive_engine import AdaptiveEngine

logger = logging.getLogger(__name__)


class GameAgent(ConversableAgent):
    """
    Game management agent that coordinates chess games between players.

    This agent acts as a game coordinator, managing the flow between two
    chess-playing agents while maintaining compatibility with the existing
    ELO ladder and statistics systems.
    """

    def __init__(
        self,
        config: Config,
        engine: Union[ChessEngine, HumanLikeEngine, AdaptiveEngine],
        name: str = "GameCoordinator",
        **kwargs
    ):
        """
        Initialize the game management agent.

        Args:
            config: Global configuration object
            engine: Chess engine for opponent moves
            name: Name of the game agent
            **kwargs: Additional AG2 arguments
        """
        # Initialize AG2 ConversableAgent with minimal LLM requirements
        super().__init__(
            name=name,
            system_message=self._create_system_message(),
            llm_config=None,  # Game agent doesn't need LLM capabilities
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            **kwargs
        )

        self.config = config
        self.engine = engine
        self._is_human_engine = isinstance(engine, (HumanLikeEngine, AdaptiveEngine))
        self._game_counter = 0

        # Game state
        self._current_game: Optional[chess.Board] = None
        self._game_pgn: Optional[chess.pgn.Game] = None
        self._pgn_node: Optional[chess.pgn.GameNode] = None
        self._game_start_time: Optional[float] = None

        # Player tracking
        self._player_white: Optional[ChessAgent] = None
        self._player_black: Optional[ChessAgent] = None
        self._llm_agent: Optional[ChessAgent] = None

        logger.info(f"Initialized GameAgent: {name}")

    def _create_system_message(self) -> str:
        """Create system message for the game agent."""
        return """You are a chess game coordinator responsible for managing games between different chess agents.

Your responsibilities include:
1. Coordinating move exchanges between players
2. Validating move legality
3. Tracking game progress and statistics
4. Managing game flow and timing
5. Interfacing with chess engines and UI systems

You ensure fair play and accurate game recording while maintaining
compatibility with ELO rating systems and performance benchmarks.
"""

    async def coordinate_game(
        self,
        llm_agent: ChessAgent,
        elo: int,
        output_dir,
        state: LiveState,
        llm_plays_white: Optional[bool] = None,
        opening_moves: Optional[List[str]] = None
    ) -> GameRecord:
        """
        Coordinate a complete chess game between LLM agent and engine.

        Args:
            llm_agent: The LLM-based chess agent
            elo: Engine ELO rating for this game
            output_dir: Directory to save PGN files
            state: Live state object for UI updates
            llm_plays_white: Force color assignment (None for alternating)
            opening_moves: List of UCI moves for opening

        Returns:
            GameRecord with complete game results
        """
        self._game_counter += 1
        self._game_start_time = time.time()
        self._llm_agent = llm_agent

        # Determine colors
        if llm_plays_white is None:
            llm_white = len(state.ladder) % 2 == 0
        else:
            llm_white = llm_plays_white

        # Set up players
        if llm_white:
            self._player_white = llm_agent
            self._player_black = None  # Engine
        else:
            self._player_white = None  # Engine
            self._player_black = llm_agent

        # Configure engine for target ELO
        await self.engine.configure_elo(elo)

        # Initialize game
        self._current_game = chess.Board()
        self._game_pgn = self._create_pgn_header(elo, llm_white)
        self._pgn_node = self._game_pgn

        # Apply opening moves
        if opening_moves:
            await self._apply_opening_moves(opening_moves, state)

        # Update initial state
        self._update_live_state(state, elo, llm_white)

        # Reset agent statistics
        llm_agent.reset_move_stats()

        logger.info(f"Starting game {self._game_counter}: {llm_agent.name} vs Engine({elo}), "
                   f"LLM plays {'White' if llm_white else 'Black'}")

        try:
            # Main game loop
            await self._play_game_loop(state, elo)

            # Finalize game
            return await self._finalize_game(output_dir, elo, state)

        except Exception as e:
            logger.error(f"Game coordination failed: {e}")
            raise

    async def _play_game_loop(self, state: LiveState, elo: int) -> None:
        """Execute the main game loop."""
        while not self._current_game.is_game_over() and self._current_game.ply() < self.config.max_plies:
            current_player_is_llm = self._is_llm_turn()

            if current_player_is_llm:
                move = await self._get_llm_move(state)
                player_name = self._llm_agent.name
            else:
                move = await self._get_engine_move(state, elo)
                player_name = f"SF{elo}" if not self._is_human_engine else self._get_engine_name()

            # Execute and record move
            await self._execute_move(move, player_name, state)

            # Brief pause for UI updates
            await asyncio.sleep(0.01)

    def _is_llm_turn(self) -> bool:
        """Determine if it's the LLM agent's turn."""
        if self._current_game.turn == chess.WHITE:
            return self._player_white is not None
        else:
            return self._player_black is not None

    async def _get_llm_move(self, state: LiveState) -> chess.Move:
        """Get move from LLM agent with error handling."""
        try:
            current_agent = self._player_white if self._current_game.turn == chess.WHITE else self._player_black
            state.status = f"vs {state.current_elo} ({current_agent.name} thinking...)"

            opponent_move = state.last_move_uci if state.last_move_uci else None
            move = await current_agent.pick_move(
                self._current_game,
                temperature=self.config.llm_temperature,
                timeout_s=self.config.llm_timeout,
                opponent_move=opponent_move,
                state=state
            )
            return move
        except Exception as e:
            logger.warning(f"LLM move failed: {e}, falling back to random")
            # Fallback to random legal move
            legal_moves = list(self._current_game.legal_moves)
            if not legal_moves:
                raise Exception("No legal moves available")

            # Track as illegal attempt for statistics
            if hasattr(self._llm_agent, '_move_stats'):
                self._llm_agent._move_stats['illegal_attempts'] += 1

            return legal_moves[0]

    async def _get_engine_move(self, state: LiveState, elo: int) -> chess.Move:
        """Get move from chess engine."""
        if self._is_human_engine:
            engine_name = self._get_engine_name()
            state.status = f"vs {elo} ({engine_name} thinking...)"
        else:
            state.status = f"vs {elo} (Stockfish thinking...)"

        return await self.engine.get_move(self._current_game)

    def _get_engine_name(self) -> str:
        """Get friendly name for the current engine."""
        if hasattr(self.engine, 'current_engine_type'):
            return self.engine.current_engine_type.title()
        elif hasattr(self.engine, 'engine_type'):
            return self.engine.engine_type.title()
        else:
            return "Human Engine"

    async def _execute_move(self, move: chess.Move, player_name: str, state: LiveState) -> None:
        """Execute move and update game state."""
        # Update board
        self._current_game.push(move)

        # Update PGN
        self._pgn_node = self._pgn_node.add_variation(move)

        # Update live state
        state.moves_made = self._current_game.ply()
        state.last_move_uci = move.uci()
        state.board_ascii = str(self._current_game)

        # Store for beautiful rendering
        state._chess_board = self._current_game.copy()
        if not hasattr(state, '_moves_played'):
            state._moves_played = []
        state._moves_played.append(move)

        logger.debug(f"Move executed: {player_name} played {move}")

    async def _apply_opening_moves(self, opening_moves: List[str], state: LiveState) -> None:
        """Apply predefined opening moves."""
        logger.info(f"Applying {len(opening_moves)} opening moves")
        for uci in opening_moves:
            try:
                move = chess.Move.from_uci(uci)
                if move in self._current_game.legal_moves:
                    await self._execute_move(move, "Opening", state)
                else:
                    logger.warning(f"Illegal opening move {uci}, skipping remaining opening")
                    break
            except Exception as e:
                logger.warning(f"Failed to apply opening move {uci}: {e}")
                break

    def _update_live_state(self, state: LiveState, elo: int, llm_white: bool) -> None:
        """Update live state with initial game information."""
        state.current_elo = elo
        state.color_llm_white = llm_white
        state.status = f"vs {elo} (starting...)"
        state.moves_made = 0
        state.last_move_uci = ""
        state.board_ascii = str(self._current_game)
        state.final_result = None
        state._chess_board = self._current_game.copy()
        state._moves_played = []

    def _create_pgn_header(self, elo: int, llm_white: bool) -> chess.pgn.Game:
        """Create PGN header for the game."""
        game = chess.pgn.Game()

        # Set player names
        if llm_white:
            game.headers["White"] = self._llm_agent.name
            game.headers["Black"] = f"Stockfish_{elo}" if not self._is_human_engine else f"Engine_{elo}"
        else:
            game.headers["White"] = f"Stockfish_{elo}" if not self._is_human_engine else f"Engine_{elo}"
            game.headers["Black"] = self._llm_agent.name

        # Add metadata
        game.headers["Event"] = "Chess LLM Benchmark"
        game.headers["Site"] = "Local"
        game.headers["Date"] = time.strftime("%Y.%m.%d")
        game.headers["Round"] = str(self._game_counter)
        game.headers["ECO"] = "?"
        game.headers["Result"] = "*"

        # Add configuration details
        game.headers["LLM_Provider"] = self._llm_agent.bot_spec.provider
        game.headers["LLM_Model"] = self._llm_agent.bot_spec.model
        game.headers["Engine_ELO"] = str(elo)
        game.headers["Temperature"] = str(self.config.llm_temperature)

        return game

    async def _finalize_game(self, output_dir, elo: int, state: LiveState) -> GameRecord:
        """Finalize game and create game record."""
        # Determine result
        result = self._current_game.result(claim_draw=True)
        self._game_pgn.headers["Result"] = result

        # Handle timeout/max-ply situations
        if self._current_game.ply() >= self.config.max_plies and not self._current_game.is_game_over():
            result = "1/2-1/2"
            self._game_pgn.headers["Result"] = result
            self._game_pgn.headers["Termination"] = "Maximum moves reached"

        # Update final state
        state.final_result = result
        state.status = f"Game finished: {result}"

        # Save PGN
        pgn_path = await self._save_pgn(output_dir, elo)

        # Collect statistics
        total_time, illegal_attempts, avg_time = self._llm_agent.get_move_stats()
        game_duration = time.time() - self._game_start_time

        # Determine LLM result
        llm_white = state.color_llm_white
        if result == "1-0":
            llm_result = "win" if llm_white else "loss"
        elif result == "0-1":
            llm_result = "loss" if llm_white else "win"
        else:
            llm_result = "draw"

        # Create game record
        return GameRecord(
            elo=elo,
            result=llm_result,
            moves=self._current_game.ply(),
            game_duration=game_duration,
            avg_move_time=avg_time,
            illegal_moves=illegal_attempts,
            pgn_path=str(pgn_path) if pgn_path else None,
            termination=self._current_game.outcome().termination.name if self._current_game.outcome() else "unknown"
        )

    async def _save_pgn(self, output_dir, elo: int) -> Optional[any]:
        """Save game PGN to file."""
        try:
            from pathlib import Path

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"game_{self._game_counter}_{self._llm_agent.name}_vs_{elo}_{timestamp}.pgn"
            pgn_path = output_path / filename

            with open(pgn_path, 'w') as f:
                print(self._game_pgn, file=f)

            logger.info(f"Saved PGN: {pgn_path}")
            return pgn_path

        except Exception as e:
            logger.error(f"Failed to save PGN: {e}")
            return None

    def get_game_stats(self) -> Dict[str, Any]:
        """Get comprehensive game statistics."""
        return {
            "games_coordinated": self._game_counter,
            "current_game_moves": self._current_game.ply() if self._current_game else 0,
            "is_game_active": self._current_game is not None and not self._current_game.is_game_over(),
            "engine_type": "human" if self._is_human_engine else "stockfish",
        }

    def reset_game_stats(self) -> None:
        """Reset game coordination statistics."""
        self._game_counter = 0
        self._current_game = None
        self._game_pgn = None
        self._pgn_node = None
        self._game_start_time = None

    def __str__(self) -> str:
        """String representation."""
        return f"GameAgent(games={self._game_counter})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"GameAgent(name='{self.name}', games_coordinated={self._game_counter})"
