"""
Game runner module for managing chess games between LLMs and engines.

This module handles the orchestration of individual chess games, including
move generation, game state management, PGN creation, and result determination.
"""

from __future__ import annotations

import asyncio
import logging
import time
import random
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .adaptive_engine import AdaptiveEngine

import chess
import chess.pgn as chess_pgn

from .models import Config, GameRecord, LiveState, LadderStats
from .engine import ChessEngine
from .human_engine import HumanLikeEngine

# Import the AdaptiveEngine class at runtime
AdaptiveEngine = None
try:
    from .adaptive_engine import AdaptiveEngine
except ImportError:
    pass  # Type will be checked via isinstance, so this is safe
from ..llm.client import LLMClient, InvalidMoveError, LLMProviderError
from .budget import BudgetExceeded

logger = logging.getLogger(__name__)


class GameRunner:
    """
    Manages individual chess games between an LLM and a chess engine.

    This class orchestrates the game flow, handles move generation from both
    sides, manages game state, and produces game records with PGN files.
    Supports both traditional engines (Stockfish) and human-like engines (Maia, LCZero).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        engine: Union[ChessEngine, HumanLikeEngine, Any],
        config: Config,
    ):
        """
        Initialize the game runner.

        Args:
            llm_client: LLM client for move generation
            engine: Chess engine, human-like engine or adaptive engine for opponent moves
            config: Global configuration
        """
        self.llm = llm_client
        self.engine = engine
        self.config = config
        self._game_counter = 0
        self.random = random.Random(config.seed)
        self._is_human_engine = isinstance(engine, HumanLikeEngine) or (
            AdaptiveEngine and isinstance(engine, AdaptiveEngine)
        )

    async def play_game(
        self,
        elo: int,
        output_dir: Path,
        state: LiveState,
        llm_plays_white: Optional[bool] = None,
        opening_moves: Optional[List[str]] = None,
        opening_name: str = "",
    ) -> GameRecord:
        """
        Play a single chess game between the LLM and engine at specified ELO.

        Args:
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
        self._game_counter += 1

        # Start tracking game duration
        game_start_time = time.monotonic()

        # Determine colors (alternate by default)
        if llm_plays_white is None:
            llm_white = len(state.ladder) % 2 == 0
        else:
            llm_white = llm_plays_white

        # Configure engine for target ELO
        if self.config.fixed_opponent_elo != 0:
            await self.engine.configure_elo(elo)
        game_seed = self.config.seed + self._game_counter - 1
        self.random.seed(game_seed)
        self.llm.provider.random.seed(game_seed + 1_000_000)
        self.llm.provider.usage_context = {
            "game_id": str(self._game_counter),
            "elo_level": elo,
        }
        state.error_message = None

        # Initialize board and game state
        board = chess.Board()
        game_pgn = self._create_pgn_header(elo, llm_white)
        pgn_node = game_pgn

        for uci in opening_moves or []:
            move = board.parse_uci(uci)  # Invalid fixtures must fail visibly.
            board.push(move)
            pgn_node = pgn_node.add_variation(move)
        game_pgn.headers["Opening"] = opening_name
        game_pgn.headers["Seed"] = str(game_seed)
        game_pgn.headers["Protocol"] = self.config.protocol_version
        game_pgn.headers["Mode"] = (
            "tool-assisted" if self.config.use_agent else "prompt"
        )
        game_pgn.headers["MaxOutputTokens"] = str(self.config.max_output_tokens)
        game_pgn.headers["ReasoningEffort"] = self.config.reasoning_effort

        # Update live state
        state.current_elo = elo
        state.color_llm_white = llm_white
        state.status = f"vs {elo} (starting...)"
        state.moves_made = board.ply()
        state.last_move_uci = board.peek().uci() if board.move_stack else ""
        state.board_ascii = str(board)
        state.final_result = None

        # Store chess board and moves for beautiful rendering
        state._chess_board = board.copy()
        state._moves_played = list(board.move_stack)

        # Reset LLM move statistics for this game
        self.llm.reset_move_stats()

        logger.info(
            f"Starting game: {self.llm.spec.name} vs Stockfish({elo}), "
            f"LLM plays {'White' if llm_white else 'Black'}"
        )

        result = None
        termination = "normal"
        error = None
        llm_moves = 0
        try:
            # Main game loop
            while (
                not board.is_game_over(claim_draw=True)
                and board.ply() < self.config.max_plies
            ):
                current_player_is_llm = (board.turn == chess.WHITE and llm_white) or (
                    board.turn == chess.BLACK and not llm_white
                )

                if current_player_is_llm:
                    self.llm.provider.usage_context["move_number"] = board.ply() + 1
                    try:
                        move = await self._get_llm_move(board, state)
                        llm_moves += 1
                    except InvalidMoveError as exc:
                        result = "0-1" if llm_white else "1-0"
                        termination, error = "invalid_move", str(exc)
                        break
                    except (BudgetExceeded, LLMProviderError) as exc:
                        result = "*"
                        termination = (
                            "budget_exhausted"
                            if isinstance(exc, BudgetExceeded)
                            else "provider_error"
                        )
                        error = str(exc)
                        break
                    player_name = self.llm.spec.name
                else:
                    try:
                        move = await self._get_engine_move(board, state, elo)
                        if move not in board.legal_moves:
                            raise ValueError("Engine returned illegal move")
                    except Exception as exc:
                        result, termination, error = "*", "engine_error", str(exc)
                        break
                    player_name = f"SF{elo}"

                # Execute move
                self._execute_move(board, move, pgn_node, state, player_name)
                pgn_node = pgn_node.add_variation(move)

                # Brief pause for UI updates
                await asyncio.sleep(0.01)

            # Determine final result
            result = result or board.result(claim_draw=True)
            if termination == "normal" and board.outcome(claim_draw=True):
                termination = board.outcome(claim_draw=True).termination.name.lower()
            game_pgn.headers["Result"] = result

            # Handle timeout/max-ply situations
            if (
                result == "*"
                and termination == "normal"
                and board.ply() >= self.config.max_plies
            ):
                result = "*"  # Truncation is not a chess draw.
                termination = "max_plies"
                game_pgn.headers["Result"] = result
                game_pgn.headers["Termination"] = "Maximum moves reached"

            # Save PGN and create record
            game_pgn.headers["Termination"] = termination

            # Collect move statistics from LLM
            total_time, illegal_attempts, avg_time = self.llm.get_move_stats()

            # Calculate total game duration
            game_duration = time.monotonic() - game_start_time

            # Update live state with timing info
            state.total_move_time = total_time
            state.average_move_time = avg_time
            state.illegal_move_attempts = illegal_attempts
            state.game_duration = game_duration
            state.final_result = result
            state.status = f"finished {result} vs {elo}"

            logger.info(
                f"Game completed: {result} in {board.ply()} plies, "
                f"avg move time: {avg_time:.2f}s, illegal attempts: {illegal_attempts}, "
                f"total game time: {game_duration:.2f}s"
            )

            # Add game duration to PGN headers
            game_pgn.headers["GameDuration"] = f"{game_duration:.2f}s"
            pgn_path = await self._save_pgn(game_pgn, output_dir, elo)

            return GameRecord(
                elo=elo,
                color_llm_white=llm_white,
                result=result,
                ply_count=board.ply(),
                path=pgn_path,
                timestamp=datetime.now(timezone.utc),
                game_duration=game_duration,
                termination=termination,
                llm_moves=llm_moves,
                llm_requests=self.llm._move_count,
                opening=opening_name,
                seed=game_seed,
                error=error,
            )

        except Exception as e:
            logger.error(f"Game failed: {e}")
            state.set_error(f"Game failed: {str(e)}")
            raise

    async def _get_llm_move(self, board: chess.Board, state: LiveState) -> chess.Move:
        """Get a move from the LLM with proper error handling."""
        state.status = f"vs {state.current_elo} ({self.llm.spec.name} thinking...)"
        return await self.llm.pick_move(
            board,
            temperature=self.config.llm_temperature,
            timeout_s=self.config.llm_timeout,
        )

    async def _get_engine_move(
        self, board: chess.Board, state: LiveState, elo: int
    ) -> chess.Move:
        """Get a move from the chess engine or human-like engine."""
        # Special handling for random opponent (ELO 0)
        if self.config.fixed_opponent_elo == 0:
            state.status = f"vs Random (Random thinking...)"
            return self._get_random_move(board)

        if self._is_human_engine:
            if AdaptiveEngine and isinstance(self.engine, AdaptiveEngine):
                engine_name = self.engine.current_engine_type
                state.status = f"vs {elo} ({engine_name.title()} thinking...)"
            else:
                engine_name = getattr(self.engine, "engine_type", "Human Engine")
                state.status = f"vs {elo} ({engine_name.title()} thinking...)"
        else:
            state.status = f"vs {elo} (Stockfish thinking...)"
        return await self.engine.get_move(board)

    def _get_random_move(self, board: chess.Board) -> chess.Move:
        """Get a completely random legal move."""
        import random

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise Exception("No legal moves available")
        return self.random.choice(legal_moves)

    def _execute_move(
        self,
        board: chess.Board,
        move: chess.Move,
        pgn_node: chess_pgn.GameNode,
        state: LiveState,
        player_name: str,
    ) -> None:
        """Execute a move and update game state."""
        if move not in board.legal_moves:
            raise ValueError(f"Illegal move: {move}")
        move_uci = move.uci()
        board.push(move)

        # Update live state
        state.board_ascii = str(board)
        state.moves_made = board.ply()
        state.last_move_uci = move_uci
        state.status = f"vs {state.current_elo} (last: {player_name} {move_uci})"

        # Update chess board and moves for beautiful rendering
        if hasattr(state, "_chess_board"):
            state._chess_board = board.copy()
        if hasattr(state, "_moves_played"):
            state._moves_played.append(move)

        logger.debug(f"Move executed: {player_name} played {move.uci()}")

    def _create_pgn_header(self, elo: int, llm_white: bool) -> chess_pgn.Game:
        """Create PGN game with proper headers."""
        game = chess_pgn.Game()
        game.headers["Event"] = "LLM Chess ELO Ladder"
        game.headers["Site"] = "Chess LLM Benchmark"
        game.headers["Date"] = datetime.now(timezone.utc).strftime("%Y.%m.%d")
        game.headers["Round"] = str(self._game_counter)

        # Determine engine name and type
        if self.config.fixed_opponent_elo == 0:  # Random opponent
            engine_name = "Random"
            engine_type_header = "Random"
        elif self._is_human_engine:
            if AdaptiveEngine and isinstance(self.engine, AdaptiveEngine):
                engine_type = self.engine.current_engine_type
                engine_name = f"{engine_type.title()}({elo})"
                engine_type_header = "Adaptive Engine"
            else:
                engine_type = getattr(self.engine, "engine_type", "Human Engine")
                engine_name = f"{engine_type.title()}({elo})"
                engine_type_header = "Human-like Engine"
        else:
            engine_name = f"Stockfish({elo})"
            engine_type_header = "Engine"

        if llm_white:
            game.headers["White"] = self.llm.spec.name
            game.headers["Black"] = engine_name
            game.headers["WhiteType"] = "LLM"
            game.headers["BlackType"] = engine_type_header
        else:
            game.headers["White"] = engine_name
            game.headers["Black"] = self.llm.spec.name
            game.headers["WhiteType"] = engine_type_header
            game.headers["BlackType"] = "LLM"

        # Add metadata
        game.headers["LLM_Provider"] = self.llm.spec.provider
        game.headers["LLM_Model"] = self.llm.spec.model
        game.headers["Engine_ELO"] = str(elo)
        game.headers["TimeControl"] = f"{self.config.think_time}s+0"

        if self._is_human_engine:
            if AdaptiveEngine and isinstance(self.engine, AdaptiveEngine):
                game.headers["Engine_Type"] = self.engine.current_engine_type
                game.headers["Adaptive_Engine"] = "True"
            else:
                game.headers["Engine_Type"] = getattr(
                    self.engine, "engine_type", "human"
                )

        return game

    async def _save_pgn(self, game: chess_pgn.Game, output_dir: Path, elo: int) -> Path:
        """Save PGN to file and return path."""
        if not self.config.save_pgn:
            # Return a dummy path if PGN saving is disabled
            return output_dir / "dummy.pgn"

        # Create bot-specific directory
        bot_dir = output_dir / self.llm.spec.name
        bot_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        timestamp = datetime.now(timezone.utc).strftime("%H%M%S")
        pgn_path = bot_dir / f"elo_{elo}_{timestamp}.pgn"

        # Ensure unique filename
        counter = 1
        while pgn_path.exists():
            pgn_path = bot_dir / f"elo_{elo}_{timestamp}_{counter}.pgn"
            counter += 1

        pgn_path.write_text(str(game) + "\n", encoding="utf-8")
        return pgn_path


class LadderRunner:
    """A bounded schedule of identical openings with colors reversed."""

    def __init__(self, game_runner: GameRunner, config: Config):
        self.game_runner = game_runner
        self.config = config

    async def run_ladder(
        self, output_dir, state, bot_stats, start_elo=None, max_elo=None, elo_step=None
    ):
        from .openings import OpeningBook

        book = OpeningBook(self.config.seed)
        elo = self.config.fixed_opponent_elo
        fixed = elo is not None
        if not fixed:
            elo = self.config.start_elo if start_elo is None else start_elo
        maximum = self.config.max_elo if max_elo is None else max_elo
        step = self.config.elo_step if elo_step is None else elo_step
        if step <= 0:
            raise ValueError("elo_step must be positive")
        games = []
        output_dir.mkdir(parents=True, exist_ok=True)
        for pair in range(self.config.max_games // 2):
            if not fixed and elo > maximum:
                break
            opening = book.get_random_opening()
            if elo not in state.ladder:
                state.ladder.append(elo)
            pair_games = []
            for white in (True, False):
                record = await self.game_runner.play_game(
                    elo=elo,
                    output_dir=output_dir,
                    state=state,
                    llm_plays_white=white,
                    opening_moves=opening[2],
                    opening_name=opening[1],
                )
                games.append(record)
                pair_games.append(record)
                bot_stats.add_game(record)
                total_time, illegal, _ = self.game_runner.llm.get_move_stats()
                bot_stats.add_timing_stats(total_time, illegal)
                # Append immediately: completed games survive a later interruption.
                with (output_dir / "games.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {"bot": self.game_runner.llm.spec.name, **asdict(record)},
                            default=str,
                        )
                        + "\n"
                    )
                if record.termination in {
                    "budget_exhausted",
                    "provider_error",
                    "engine_error",
                }:
                    state.set_error(record.error or record.termination)
                    return bot_stats.max_elo_reached, games
            if not fixed:
                if any(not game.completed for game in pair_games):
                    break
                if self.config.escalate_on == "on_win" and not any(
                    game.llm_won for game in pair_games
                ):
                    break
                elo += step
        state.status = "finished"
        return bot_stats.max_elo_reached, games

    def _should_advance(self, game_record):
        return game_record.completed and (
            self.config.escalate_on == "always" or game_record.llm_won
        )

    def get_effective_elo(self):
        return getattr(self.game_runner.engine, "effective_elo", None)
