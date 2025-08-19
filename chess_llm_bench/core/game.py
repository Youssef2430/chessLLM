"""
Game runner module for managing chess games between LLMs and engines.

This module handles the orchestration of individual chess games, including
move generation, game state management, PGN creation, and result determination.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
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
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


class GameRunner:
    """
    Manages individual chess games between an LLM and a chess engine.

    This class orchestrates the game flow, handles move generation from both
    sides, manages game state, and produces game records with PGN files.
    Supports both traditional engines (Stockfish) and human-like engines (Maia, LCZero).
    """

    def __init__(self, llm_client: LLMClient, engine: Union[ChessEngine, HumanLikeEngine, Any], config: Config):
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
        self._is_human_engine = isinstance(engine, HumanLikeEngine) or (AdaptiveEngine and isinstance(engine, AdaptiveEngine))

    async def play_game(
        self,
        elo: int,
        output_dir: Path,
        state: LiveState,
        llm_plays_white: Optional[bool] = None,
        opening_moves: Optional[List[str]] = None
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
        game_start_time = time.time()

        # Determine colors (alternate by default)
        if llm_plays_white is None:
            llm_white = len(state.ladder) % 2 == 0
        else:
            llm_white = llm_plays_white

        # Configure engine for target ELO
        await self.engine.configure_elo(elo)

        # Initialize board and game state
        board = chess.Board()
        game_pgn = self._create_pgn_header(elo, llm_white)
        pgn_node = game_pgn

        # Apply opening moves if provided
        if opening_moves:
            logger.info(f"Applying {len(opening_moves)} opening moves")
            for uci in opening_moves:
                try:
                    move = chess.Move.from_uci(uci)
                    if move in board.legal_moves:
                        player_name = "Opening"
                        self._execute_move(board, move, pgn_node, state, player_name)
                        pgn_node = pgn_node.add_variation(move)
                    else:
                        logger.warning(f"Illegal opening move {uci}, skipping remaining opening")
                        break
                except Exception as e:
                    logger.warning(f"Failed to apply opening move {uci}: {e}")
                    break

        # Update live state
        state.current_elo = elo
        state.color_llm_white = llm_white
        state.status = f"vs {elo} (starting...)"
        state.moves_made = 0
        state.last_move_uci = ""
        state.board_ascii = str(board)
        state.final_result = None

        # Store chess board and moves for beautiful rendering
        state._chess_board = board.copy()
        state._moves_played = []

        # Reset LLM move statistics for this game
        self.llm.reset_move_stats()

        logger.info(f"Starting game: {self.llm.spec.name} vs Stockfish({elo}), "
                   f"LLM plays {'White' if llm_white else 'Black'}")

        try:
            # Main game loop
            while not board.is_game_over() and board.ply() < self.config.max_plies:
                current_player_is_llm = (
                    (board.turn == chess.WHITE and llm_white) or
                    (board.turn == chess.BLACK and not llm_white)
                )

                if current_player_is_llm:
                    move = await self._get_llm_move(board, state)
                    player_name = self.llm.spec.name
                else:
                    move = await self._get_engine_move(board, state, elo)
                    player_name = f"SF{elo}"

                # Execute move
                self._execute_move(board, move, pgn_node, state, player_name)
                pgn_node = pgn_node.add_variation(move)

                # Brief pause for UI updates
                await asyncio.sleep(0.01)

            # Determine final result
            result = board.result(claim_draw=True)
            game_pgn.headers["Result"] = result

            # Handle timeout/max-ply situations
            if board.ply() >= self.config.max_plies and not board.is_game_over():
                result = "1/2-1/2"  # Draw by move limit
                game_pgn.headers["Result"] = result
                game_pgn.headers["Termination"] = "Maximum moves reached"

            # Save PGN and create record
            pgn_path = await self._save_pgn(game_pgn, output_dir, elo)

            # Collect move statistics from LLM
            total_time, illegal_attempts, avg_time = self.llm.get_move_stats()

            # Calculate total game duration
            game_duration = time.time() - game_start_time

            # Update live state with timing info
            state.total_move_time = total_time
            state.average_move_time = avg_time
            state.illegal_move_attempts = illegal_attempts
            state.game_duration = game_duration
            state.final_result = result
            state.status = f"finished {result} vs {elo}"

            logger.info(f"Game completed: {result} in {board.ply()} plies, "
                       f"avg move time: {avg_time:.2f}s, illegal attempts: {illegal_attempts}, "
                       f"total game time: {game_duration:.2f}s")

            # Add game duration to PGN headers
            game_pgn.headers["GameDuration"] = f"{game_duration:.2f}s"

            return GameRecord(
                elo=elo,
                color_llm_white=llm_white,
                result=result,
                ply_count=board.ply(),
                path=pgn_path,
                timestamp=datetime.utcnow(),
                game_duration=game_duration
            )

        except Exception as e:
            logger.error(f"Game failed: {e}")
            state.set_error(f"Game failed: {str(e)}")
            raise

    async def _get_llm_move(self, board: chess.Board, state: LiveState) -> chess.Move:
        """Get a move from the LLM with proper error handling."""
        try:
            state.status = f"vs {state.current_elo} ({self.llm.spec.name} thinking...)"
            # Get opponent's last move if available
            opponent_move = state.last_move_uci if state.last_move_uci else None
            move = await self.llm.pick_move(
                board,
                temperature=self.config.llm_temperature,
                timeout_s=self.config.llm_timeout,
                opponent_move=opponent_move
            )
            return move
        except Exception as e:
            logger.warning(f"LLM move failed: {e}, falling back to random")
            # Fallback to random legal move
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                raise Exception("No legal moves available")
            # Count this as an illegal move attempt
            self.llm._illegal_move_attempts += 1
            fallback_move = legal_moves[0]  # Simple fallback
            logger.info(f"Using fallback move: {fallback_move.uci()}")
            return fallback_move

    async def _get_engine_move(self, board: chess.Board, state: LiveState, elo: int) -> chess.Move:
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
                engine_name = getattr(self.engine, 'engine_type', 'Human Engine')
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
        return random.choice(legal_moves)

    def _execute_move(
        self,
        board: chess.Board,
        move: chess.Move,
        pgn_node: chess_pgn.GameNode,
        state: LiveState,
        player_name: str
    ) -> None:
        """Execute a move and update game state."""
        move_uci = move.uci()
        board.push(move)

        # Update live state
        state.board_ascii = str(board)
        state.moves_made = board.ply()
        state.last_move_uci = move_uci
        state.status = f"vs {state.current_elo} (last: {player_name} {move_uci})"

        # Update chess board and moves for beautiful rendering
        if hasattr(state, '_chess_board'):
            state._chess_board = board.copy()
        if hasattr(state, '_moves_played'):
            state._moves_played.append(move)

        logger.debug(f"Move executed: {player_name} played {move.uci()}")

    def _create_pgn_header(self, elo: int, llm_white: bool) -> chess_pgn.Game:
        """Create PGN game with proper headers."""
        game = chess_pgn.Game()
        game.headers["Event"] = "LLM Chess ELO Ladder"
        game.headers["Site"] = "Chess LLM Benchmark"
        game.headers["Date"] = datetime.utcnow().strftime("%Y.%m.%d")
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
                engine_type = getattr(self.engine, 'engine_type', 'Human Engine')
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
                game.headers["Engine_Type"] = getattr(self.engine, 'engine_type', 'human')

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
        timestamp = datetime.utcnow().strftime("%H%M%S")
        pgn_path = bot_dir / f"elo_{elo}_{timestamp}.pgn"

        # Ensure unique filename
        counter = 1
        while pgn_path.exists():
            pgn_path = bot_dir / f"elo_{elo}_{timestamp}_{counter}.pgn"
            counter += 1

        # Write PGN file
        try:
            with pgn_path.open("w", encoding="utf-8") as f:
                print(game, file=f)
            logger.debug(f"PGN saved to {pgn_path}")
        except Exception as e:
            logger.error(f"Failed to save PGN: {e}")
            # Don't fail the game just because PGN saving failed

        return pgn_path


class LadderRunner:
    """
    Manages a complete ELO ladder run for a single bot.

    This class orchestrates multiple games across increasing ELO ratings,
    implementing the ladder progression rules and statistics tracking.
    """

    def __init__(self, game_runner: GameRunner, config: Config):
        """
        Initialize the ladder runner.

        Args:
            game_runner: GameRunner instance for individual games
            config: Global configuration
        """
        self.game_runner = game_runner
        self.config = config

    async def run_ladder(
        self,
        output_dir: Path,
        state: LiveState,
        bot_stats: LadderStats,
        start_elo: Optional[int] = None,
        max_elo: Optional[int] = None,
        elo_step: Optional[int] = None
    ) -> Tuple[int, list[GameRecord]]:
        """
        Run a complete ladder for the bot.

        Args:
            output_dir: Output directory for game files
            state: Live state for UI updates
            bot_stats: Bot statistics for real-time updates
            start_elo: Starting ELO (uses config default if None)
            max_elo: Maximum ELO (uses config default if None)
            elo_step: ELO increment (uses config default if None)

        Returns:
            Tuple of (max_elo_reached, list_of_game_records)
        """
        # Check if we're in fixed ELO mode
        if self.config.fixed_opponent_elo is not None:
            return await self._run_fixed_elo_games(output_dir, state, bot_stats)
        # Use config defaults if not specified
        current_elo = start_elo or self.config.start_elo
        max_target_elo = max_elo or self.config.max_elo
        step = elo_step or self.config.elo_step

        games: list[GameRecord] = []
        losses_at_elo: Dict[int, int] = {}  # Track losses per ELO level

        # Initialize opening book
        from .openings import OpeningBook
        opening_book = OpeningBook()

        logger.info(f"Starting ladder run: {current_elo} → {max_target_elo} (step: {step})")

        while current_elo <= max_target_elo:
            # Add current ELO to ladder display only if it's the first attempt
            if current_elo not in losses_at_elo:
                state.ladder.append(current_elo)
                losses_at_elo[current_elo] = 0

            try:
                # Get a balanced pair of openings for both colors
                opening_white, opening_black = opening_book.get_balanced_pair()

                logger.info(f"Playing with opening: {opening_white[1]} (ECO {opening_white[0]}) as White")

                # Play first game at current ELO (LLM as white)
                game_record_white = await self.game_runner.play_game(
                    elo=current_elo,
                    output_dir=output_dir,
                    state=state,
                    llm_plays_white=True,
                    opening_moves=opening_white[2]
                )
                games.append(game_record_white)

                # Collect timing statistics from LLM
                total_time_white, illegal_attempts_white, _ = self.game_runner.llm.get_move_stats()

                # Update statistics in real-time
                bot_stats.add_game(game_record_white)
                bot_stats.add_timing_stats(total_time_white, illegal_attempts_white)

                # Reset state for the second game
                self.game_runner.llm.reset_move_stats()
                state.status = f"playing as Black at ELO {current_elo}..."

                logger.info(f"Playing with opening: {opening_black[1]} (ECO {opening_black[0]}) as Black")

                # Play second game with LLM as black
                game_record_black = await self.game_runner.play_game(
                    elo=current_elo,
                    output_dir=output_dir,
                    state=state,
                    llm_plays_white=False,
                    opening_moves=opening_black[2]
                )
                games.append(game_record_black)

                # Collect timing statistics for second game
                total_time_black, illegal_attempts_black, _ = self.game_runner.llm.get_move_stats()

                # Update statistics for second game
                bot_stats.add_game(game_record_black)
                bot_stats.add_timing_stats(total_time_black, illegal_attempts_black)

                # Count losses from both games
                losses_in_pair = 0
                if game_record_white.llm_lost:
                    losses_in_pair += 1
                if game_record_black.llm_lost:
                    losses_in_pair += 1

                # Add losses to the counter
                losses_at_elo[current_elo] += losses_in_pair

                # Check total losses at this ELO level
                if losses_at_elo[current_elo] >= 2:
                    logger.info(f"Ladder run ended at ELO {current_elo} (accumulated {losses_at_elo[current_elo]} losses). Average game time: {bot_stats.average_game_duration:.2f}s")
                    break
                elif losses_in_pair == 1 and losses_at_elo[current_elo] == 1:
                    # First loss - give second chance
                    logger.info(f"First loss at ELO {current_elo}, giving second chance")
                    state.status = f"retry at {current_elo} (1 loss)..."
                    await asyncio.sleep(0.5)
                    continue  # Stay at same ELO for retry

                # Determine if we advance based on combined results
                should_advance_white = self._should_advance(game_record_white)
                should_advance_black = self._should_advance(game_record_black)
                should_advance = should_advance_white or should_advance_black

                if should_advance:
                    # Advance to next ELO level
                    current_elo += step
                    state.status = "advancing to next level..."
                    await asyncio.sleep(0.5)  # Brief pause between games
                else:
                    # This handles cases where escalate_on is "on_win" and we drew
                    logger.info(f"Ladder run ended at ELO {current_elo} (result: {game_record.result}). Average game time: {bot_stats.average_game_duration:.2f}s")
                    break

            except Exception as e:
                logger.error(f"Ladder run failed at ELO {current_elo}: {e}")
                state.set_error(f"Failed at ELO {current_elo}: {str(e)}")
                break

        max_elo_reached = max(game.elo for game in games) if games else 0
        logger.info(f"Ladder run completed. Max ELO reached: {max_elo_reached}. Total game time: {bot_stats.total_game_duration:.2f}s, Average game time: {bot_stats.average_game_duration:.2f}s")

        return max_elo_reached, games

    async def _run_fixed_elo_games(
        self,
        output_dir: Path,
        state: LiveState,
        bot_stats: LadderStats,
        num_games: int = 10
    ) -> Tuple[int, list[GameRecord]]:
        """
        Run multiple games against a fixed opponent ELO instead of climbing a ladder.

        Args:
            output_dir: Output directory for game files
            state: Live state for UI updates
            bot_stats: Bot statistics for real-time updates
            num_games: Number of games to play (default: 10)

        Returns:
            Tuple of (fixed_elo, list_of_game_records)
        """
        fixed_elo = self.config.fixed_opponent_elo
        games: list[GameRecord] = []
        
        # Handle special case for random opponent
        if fixed_elo == 0:  # 0 means random moves
            opponent_name = "Random"
            state.ladder.append(0)  # Display "Random" in ladder
            logger.info(f"Starting fixed opponent run against Random opponent ({num_games} games)")
        else:
            opponent_name = f"ELO {fixed_elo}"
            state.ladder.append(fixed_elo)
            logger.info(f"Starting fixed opponent run against ELO {fixed_elo} ({num_games} games)")

        # Initialize opening book
        from .openings import OpeningBook
        opening_book = OpeningBook()

        for game_num in range(1, num_games + 1):
            try:
                # Alternate colors: LLM plays white on odd games, black on even games
                llm_plays_white = game_num % 2 == 1
                color_name = "White" if llm_plays_white else "Black"
                
                state.status = f"Game {game_num}/{num_games} as {color_name} vs {opponent_name}..."
                
                # Get balanced opening for the current color
                if llm_plays_white:
                    opening_white, opening_black = opening_book.get_balanced_pair()
                    opening_moves = opening_white[2]
                    opening_name = opening_white[1]
                    opening_eco = opening_white[0]
                else:
                    opening_white, opening_black = opening_book.get_balanced_pair()
                    opening_moves = opening_black[2]
                    opening_name = opening_black[1]
                    opening_eco = opening_black[0]

                logger.info(f"Game {game_num}: Playing with opening: {opening_name} (ECO {opening_eco}) as {color_name}")

                # Handle random opponent specially
                if fixed_elo == 0:
                    # For random opponent, we'll need to modify the game runner
                    # For now, use a very low ELO to get weak play
                    game_elo = 600  # Use minimum ELO as baseline
                else:
                    game_elo = fixed_elo

                # Play the game
                game_record = await self.game_runner.play_game(
                    elo=game_elo,
                    output_dir=output_dir,
                    state=state,
                    llm_plays_white=llm_plays_white,
                    opening_moves=opening_moves
                )
                
                # Override the ELO in the record to reflect the actual fixed ELO
                game_record.elo = fixed_elo
                games.append(game_record)

                # Collect timing statistics from LLM
                total_time, illegal_attempts, _ = self.game_runner.llm.get_move_stats()

                # Update statistics in real-time
                bot_stats.add_game(game_record)
                bot_stats.add_timing_stats(total_time, illegal_attempts)

                # Reset stats for next game
                self.game_runner.llm.reset_move_stats()

                # Brief pause between games
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Fixed ELO game {game_num} failed: {e}")
                state.set_error(f"Failed at game {game_num}: {str(e)}")
                break

        # Calculate results
        if games:
            wins = sum(1 for game in games if game.llm_won)
            draws = sum(1 for game in games if game.is_draw)
            losses = sum(1 for game in games if game.llm_lost)
            win_rate = wins / len(games) * 100
            
            logger.info(f"Fixed ELO run completed against {opponent_name}. "
                       f"Games: {len(games)}, Wins: {wins}, Draws: {draws}, Losses: {losses}, "
                       f"Win rate: {win_rate:.1f}%, "
                       f"Average game time: {bot_stats.average_game_duration:.2f}s")
        else:
            logger.warning("No games completed in fixed ELO run")

        return fixed_elo, games

    def _should_advance(self, game_record: GameRecord) -> bool:
        """
        Determine if the bot should advance to the next ELO level.

        Args:
            game_record: Record of the completed game

        Returns:
            True if bot should advance, False if ladder run should end
        """
        if self.config.escalate_on == "always":
            # Advance on any result (win, draw, or loss)
            return True
        elif self.config.escalate_on == "on_win":
            # Only advance on wins
            return game_record.llm_won
        else:
            # Default: advance on wins and draws, stop on losses
            return not game_record.llm_lost

    def get_effective_elo(self) -> Optional[int]:
        """
        Get the effective ELO of the current engine.

        Returns:
            The effective ELO of the engine, or None if not available
        """
        if hasattr(self.game_runner.engine, 'effective_elo') and self.game_runner.engine.effective_elo:
            return self.game_runner.engine.effective_elo
        return None
