"""
Enhanced Chess Puzzle Runner with Progress Callbacks

This module provides an enhanced puzzle runner that supports real-time progress
updates and callbacks for live dashboard integration. It builds upon the base
puzzle runner functionality while adding modern features like progress tracking,
status updates, and concurrent execution support.

Features:
- Real-time progress callbacks for live UI updates
- Detailed status tracking and reporting
- Enhanced error handling and recovery
- Support for concurrent puzzle solving
- Comprehensive timing and performance metrics
- Integration with modern CLI dashboard
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union, Callable, Any, TYPE_CHECKING
import traceback

if TYPE_CHECKING:
    from ..core.engine import ChessEngine
    from ..llm.client import LLMClient

import chess
import chess.engine

from . import (
    PuzzlePosition, PuzzleAttempt, PuzzleStats, PuzzleSet,
    PuzzleType, TacticMotif, EndgameType
)
from .database import puzzle_db
from ..core.models import BotSpec, Config
from ..core.engine import create_engine

logger = logging.getLogger(__name__)


class EnhancedPuzzleRunner:
    """
    Enhanced puzzle runner with real-time progress tracking and callbacks.

    This runner provides detailed progress updates, status callbacks, and
    enhanced error handling for modern UI integration.
    """

    def __init__(self, llm_client: LLMClient, engine: ChessEngine, config: Config,
                 progress_callback: Optional[Callable] = None):
        """
        Initialize the enhanced puzzle runner.

        Args:
            llm_client: LLM client for puzzle solving
            engine: Chess engine for move validation and evaluation
            config: Global configuration settings
            progress_callback: Optional callback for progress updates
        """
        self.llm = llm_client
        self.engine = engine
        self.config = config
        self.stats = PuzzleStats()
        self.progress_callback = progress_callback
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._cancelled = False

    async def run_enhanced_puzzle_session(self,
                                        puzzle_set: PuzzleSet,
                                        output_dir: Path,
                                        time_limit: float = 30.0) -> PuzzleStats:
        """
        Run an enhanced puzzle-solving session with progress updates.

        Args:
            puzzle_set: Set of puzzles to solve
            output_dir: Directory to save results
            time_limit: Maximum time per puzzle (seconds)

        Returns:
            PuzzleStats with detailed performance metrics
        """
        logger.info(f"Starting enhanced puzzle session: {puzzle_set.name} ({len(puzzle_set.puzzles)} puzzles)")

        # Reset stats and notify start
        self.stats.reset()
        await self._notify_progress({
            'status': 'session_started',
            'bot_name': self.llm.spec.name,
            'total_puzzles': len(puzzle_set.puzzles),
            'puzzle_set_name': puzzle_set.name
        })

        # Create output directory for this session
        session_dir = output_dir / f"puzzle_session_{self._session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        # Track session timing
        session_start = time.time()
        results: List[PuzzleAttempt] = []

        try:
            for i, puzzle in enumerate(puzzle_set.puzzles):
                if self._cancelled:
                    logger.info("Puzzle session cancelled by user")
                    break

                logger.info(f"Solving puzzle {i+1}/{len(puzzle_set.puzzles)}: {puzzle.title}")

                # Notify puzzle start
                await self._notify_progress({
                    'status': 'puzzle_started',
                    'puzzle_index': i + 1,
                    'puzzle_title': puzzle.title,
                    'puzzle_type': puzzle.puzzle_type,
                    'puzzle_difficulty': puzzle.difficulty,
                    'is_solving': True
                })

                try:
                    attempt = await self._solve_puzzle_enhanced(puzzle, time_limit, i + 1)
                    results.append(attempt)
                    self.stats.add_attempt(attempt)

                    # Notify puzzle completion
                    await self._notify_progress({
                        'status': 'puzzle_completed',
                        'completed_puzzle': True,
                        'puzzle_index': i + 1,
                        'is_correct': attempt.is_correct,
                        'response_time': attempt.response_time,
                        'puzzle_type': puzzle.puzzle_type,
                        'current_success_rate': self.stats.success_rate,
                        'is_solving': False
                    })

                    # Log progress
                    success_rate = self.stats.success_rate * 100
                    logger.info(f"Puzzle {i+1} {'✓' if attempt.is_correct else '✗'} "
                               f"(Success rate: {success_rate:.1f}%)")

                except Exception as e:
                    logger.error(f"Error solving puzzle {i+1}: {e}")

                    # Create failed attempt record
                    failed_attempt = PuzzleAttempt(
                        puzzle_id=f"puzzle_{i+1}",
                        llm_move="",
                        response_time=time_limit,
                        is_correct=False,
                        raw_response=f"Error: {str(e)}"
                    )
                    results.append(failed_attempt)
                    self.stats.add_attempt(failed_attempt)

                    # Notify error
                    await self._notify_progress({
                        'status': 'puzzle_error',
                        'error': str(e),
                        'puzzle_index': i + 1,
                        'is_solving': False
                    })

            session_duration = time.time() - session_start

            # Save detailed results
            await self._save_session_results(session_dir, puzzle_set, results, session_duration)

            # Notify session completion
            await self._notify_progress({
                'status': 'session_completed',
                'total_solved': self.stats.correct_solutions,
                'total_attempts': self.stats.total_attempts,
                'success_rate': self.stats.success_rate,
                'session_duration': session_duration,
                'is_solving': False
            })

            logger.info(f"Session complete: {self.stats.correct_solutions}/{self.stats.total_attempts} "
                       f"solved ({self.stats.success_rate*100:.1f}%) in {session_duration:.1f}s")

            return self.stats

        except Exception as e:
            await self._notify_progress({
                'status': 'session_error',
                'error': str(e),
                'is_solving': False
            })
            raise

    async def _solve_puzzle_enhanced(self, puzzle: PuzzlePosition, time_limit: float,
                                   puzzle_index: int) -> PuzzleAttempt:
        """
        Solve a single puzzle with enhanced progress tracking.

        Args:
            puzzle: Puzzle to solve
            time_limit: Maximum solving time
            puzzle_index: Current puzzle number

        Returns:
            PuzzleAttempt with results and metadata
        """
        start_time = time.time()
        board = chess.Board(puzzle.fen)

        # Create puzzle prompt
        prompt = self._create_puzzle_prompt(puzzle, board)

        # Notify solving start
        await self._notify_progress({
            'status': 'solving_puzzle',
            'puzzle_index': puzzle_index,
            'puzzle_type': puzzle.puzzle_type,
            'is_solving': True
        })

        # Track illegal move attempts
        illegal_attempts = 0
        llm_move = ""
        raw_response = ""

        try:
            # Get LLM response with timeout
            response = await asyncio.wait_for(
                self.llm.get_move(board, prompt),
                timeout=time_limit
            )

            raw_response = str(response)

            # Extract move from response
            llm_move = self._extract_move_from_response(response, board)

            # Validate move legality with progress updates
            while llm_move and not self._is_legal_move(llm_move, board):
                illegal_attempts += 1
                if illegal_attempts >= 3:  # Give up after 3 attempts
                    break

                # Notify illegal move attempt
                await self._notify_progress({
                    'status': 'illegal_move_attempt',
                    'illegal_move': llm_move,
                    'attempt_number': illegal_attempts,
                    'puzzle_index': puzzle_index
                })

                # Ask for correction
                correction_prompt = f"The move '{llm_move}' is illegal. Please provide a legal move in UCI format:"
                try:
                    response = await asyncio.wait_for(
                        self.llm.get_move(board, correction_prompt),
                        timeout=time_limit / 2
                    )
                    llm_move = self._extract_move_from_response(response, board)
                except asyncio.TimeoutError:
                    break

        except asyncio.TimeoutError:
            logger.warning(f"Puzzle solving timed out after {time_limit}s")
            llm_move = ""
            raw_response = "TIMEOUT"

            await self._notify_progress({
                'status': 'puzzle_timeout',
                'puzzle_index': puzzle_index,
                'timeout_duration': time_limit
            })

        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            llm_move = ""
            raw_response = f"ERROR: {str(e)}"

            await self._notify_progress({
                'status': 'llm_error',
                'error': str(e),
                'puzzle_index': puzzle_index
            })

        response_time = time.time() - start_time

        # Evaluate the move
        is_correct = self._evaluate_move(llm_move, puzzle)
        evaluation_delta = await self._calculate_evaluation_delta(llm_move, puzzle, board)

        # Extract reasoning if possible
        reasoning = self._extract_reasoning(raw_response)

        puzzle_id = f"{puzzle.puzzle_type.value}_{hash(puzzle.fen) % 10000}"

        # Create attempt record
        attempt = PuzzleAttempt(
            puzzle_id=puzzle_id,
            llm_move=llm_move,
            response_time=response_time,
            is_correct=is_correct,
            evaluation_delta=evaluation_delta,
            raw_response=raw_response,
            reasoning=reasoning,
            illegal_attempts=illegal_attempts
        )

        return attempt

    async def _notify_progress(self, progress_data: Dict[str, Any]) -> None:
        """Notify progress callback if available."""
        if self.progress_callback:
            try:
                if asyncio.iscoroutinefunction(self.progress_callback):
                    await self.progress_callback(progress_data)
                else:
                    self.progress_callback(progress_data)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def _create_puzzle_prompt(self, puzzle: PuzzlePosition, board: chess.Board) -> str:
        """Create an appropriate prompt for the puzzle type."""
        base_prompt = f"Position (FEN): {puzzle.fen}\n\n"
        base_prompt += f"Current board:\n{board}\n\n"

        if puzzle.puzzle_type == PuzzleType.TACTIC:
            if puzzle.mate_in:
                base_prompt += f"🎯 Find the best move that leads to mate in {puzzle.mate_in}.\n"
            else:
                base_prompt += "🎯 Find the best tactical move.\n"

            if puzzle.tactic_motif:
                motif_hint = puzzle.tactic_motif.value.replace('_', ' ').title()
                base_prompt += f"💡 Hint: Look for a {motif_hint}.\n"

        elif puzzle.puzzle_type == PuzzleType.ENDGAME:
            base_prompt += "♔ This is an endgame position. Find the best move.\n"
            if puzzle.endgame_type:
                endgame_hint = puzzle.endgame_type.value
                base_prompt += f"📚 Endgame type: {endgame_hint}\n"

        elif puzzle.puzzle_type == PuzzleType.BLUNDER_AVOID:
            base_prompt += "⚠️ This is a critical position. Find the move that avoids a major blunder.\n"
            base_prompt += "🚨 Be very careful - there are moves that lose significant material or position.\n"

        elif puzzle.puzzle_type == PuzzleType.GAMELET:
            base_prompt += "📖 This is an opening position. Find the best move.\n"
            if puzzle.opening_name:
                base_prompt += f"🎼 Opening: {puzzle.opening_name}\n"

        base_prompt += "\n🎲 Provide your answer as a single UCI move (e.g., 'e2e4', 'g1f3', 'e1g1').\n"
        base_prompt += "💭 You may include brief reasoning, but the move must be clearly stated.\n"

        return base_prompt

    def _extract_move_from_response(self, response: str, board: chess.Board) -> str:
        """Extract UCI move from LLM response."""
        import re

        response_str = str(response).strip()

        # Try to find UCI format moves (e.g., e2e4, g1f3)
        uci_pattern = r'\b[a-h][1-8][a-h][1-8][qrbn]?\b'
        uci_matches = re.findall(uci_pattern, response_str.lower())

        for move_str in uci_matches:
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    return move_str
            except ValueError:
                continue

        # Try to find algebraic notation and convert
        for move in board.legal_moves:
            move_san = board.san(move)
            move_uci = move.uci()

            # Check if the move appears in the response
            if (move_san.lower() in response_str.lower() or
                move_uci.lower() in response_str.lower()):
                return move_uci

        # If no valid move found, return empty string
        return ""

    def _is_legal_move(self, move_str: str, board: chess.Board) -> bool:
        """Check if a move string represents a legal move."""
        if not move_str:
            return False

        try:
            move = chess.Move.from_uci(move_str)
            return move in board.legal_moves
        except ValueError:
            return False

    def _evaluate_move(self, llm_move: str, puzzle: PuzzlePosition) -> bool:
        """Evaluate if the LLM's move is correct."""
        if not llm_move:
            return False

        # For most puzzles, exact move match is required
        return llm_move.lower() == puzzle.best_move.lower()

    async def _calculate_evaluation_delta(self, llm_move: str, puzzle: PuzzlePosition,
                                        board: chess.Board) -> Optional[float]:
        """Calculate evaluation difference between LLM move and best move."""
        if not llm_move or not self._is_legal_move(llm_move, board):
            return None

        try:
            # Configure engine for analysis
            await self.engine.configure_elo(2800)  # Use maximum strength for analysis

            # Get evaluation for best move
            best_board = board.copy()
            best_move = chess.Move.from_uci(puzzle.best_move)
            best_board.push(best_move)

            best_eval = await self._get_position_evaluation(best_board)

            # Get evaluation for LLM move
            llm_board = board.copy()
            llm_move_obj = chess.Move.from_uci(llm_move)
            llm_board.push(llm_move_obj)

            llm_eval = await self._get_position_evaluation(llm_board)

            if best_eval is not None and llm_eval is not None:
                # Return difference (positive means LLM move is worse)
                return best_eval - llm_eval

        except Exception as e:
            logger.warning(f"Error calculating evaluation delta: {e}")

        return None

    async def _get_position_evaluation(self, board: chess.Board) -> Optional[float]:
        """Get engine evaluation for a position."""
        try:
            # Use the engine to evaluate position
            result = await self.engine.engine.analyse(board, chess.engine.Limit(depth=15))

            if result.score:
                # Convert to centipawns from white's perspective
                score = result.score.white()
                if score.is_mate():
                    # Convert mate scores to large numbers
                    mate_score = score.mate()
                    return 10000 - abs(mate_score) * 100 if mate_score > 0 else -10000 + abs(mate_score) * 100
                else:
                    return float(score.score())

        except Exception as e:
            logger.warning(f"Error getting position evaluation: {e}")

        return None

    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from LLM response."""
        import re

        # Simple extraction - look for explanatory text
        lines = response.split('\n')
        reasoning_lines = []

        for line in lines:
            line = line.strip()
            # Skip lines that are just the move
            if (len(line) > 10 and
                not line.lower().startswith('move:') and
                not re.match(r'^[a-h][1-8][a-h][1-8]', line.lower())):
                reasoning_lines.append(line)

        return ' '.join(reasoning_lines).strip()

    async def _save_session_results(self, session_dir: Path, puzzle_set: PuzzleSet,
                                   results: List[PuzzleAttempt], session_duration: float) -> None:
        """Save detailed session results to files."""

        # Save session summary
        summary_file = session_dir / "session_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Enhanced Puzzle Session Summary\n")
            f.write(f"==============================\n\n")
            f.write(f"🤖 Bot: {self.llm.spec.name}\n")
            f.write(f"🧩 Puzzle Set: {puzzle_set.name}\n")
            f.write(f"📊 Total Puzzles: {len(puzzle_set.puzzles)}\n")
            f.write(f"⏱️  Session Duration: {session_duration:.1f}s\n")
            f.write(f"📅 Timestamp: {datetime.now().isoformat()}\n\n")

            f.write(f"🎯 Performance Summary:\n")
            f.write(f"  ✅ Correct Solutions: {self.stats.correct_solutions}\n")
            f.write(f"  📈 Total Attempts: {self.stats.total_attempts}\n")
            f.write(f"  🎪 Success Rate: {self.stats.success_rate*100:.1f}%\n")
            f.write(f"  ⚡ Average Response Time: {self.stats.average_response_time:.2f}s\n")
            f.write(f"  🚫 Total Illegal Moves: {self.stats.total_illegal_moves}\n")

            if self.stats.fastest_solve != float('inf'):
                f.write(f"  🏃 Fastest Solve: {self.stats.fastest_solve:.2f}s\n")
                f.write(f"  🐌 Slowest Solve: {self.stats.slowest_solve:.2f}s\n")

        # Save detailed results as CSV
        import csv
        results_file = session_dir / "detailed_results.csv"
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'puzzle_id', 'llm_move', 'is_correct', 'response_time',
                'evaluation_delta', 'illegal_attempts', 'reasoning'
            ])

            for attempt in results:
                writer.writerow([
                    attempt.puzzle_id,
                    attempt.llm_move,
                    attempt.is_correct,
                    f"{attempt.response_time:.2f}",
                    f"{attempt.evaluation_delta:.1f}" if attempt.evaluation_delta else "",
                    attempt.illegal_attempts,
                    attempt.reasoning[:100] + "..." if len(attempt.reasoning) > 100 else attempt.reasoning
                ])

        logger.info(f"Enhanced session results saved to {session_dir}")

    def cancel(self) -> None:
        """Cancel the current puzzle session."""
        self._cancelled = True


async def run_enhanced_puzzle_benchmark(bot_spec: BotSpec, config: Config, output_dir: Path,
                                      progress_callback: Optional[Callable] = None) -> Dict[str, PuzzleStats]:
    """
    Run an enhanced puzzle benchmark for a single bot with progress tracking.

    Args:
        bot_spec: Bot specification
        config: Configuration settings
        output_dir: Output directory for results
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary of puzzle set results
    """
    from ..llm.client import LLMClient

    # Create LLM client and engine
    llm_client = LLMClient(bot_spec)
    engine = create_engine(config)

    try:
        # Initialize engine
        await engine.start()

        # Create enhanced puzzle runner
        runner = EnhancedPuzzleRunner(llm_client, engine, config, progress_callback)

        # Define standard puzzle sets with enhanced categorization
        puzzle_sets = [
            ("Tactics - Easy", PuzzleType.TACTIC, 10, (1, 3)),
            ("Tactics - Medium", PuzzleType.TACTIC, 10, (4, 6)),
            ("Tactics - Hard", PuzzleType.TACTIC, 10, (7, 10)),
            ("Endgames - Basic", PuzzleType.ENDGAME, 8, (1, 5)),
            ("Endgames - Advanced", PuzzleType.ENDGAME, 8, (6, 10)),
            ("Blunder Avoidance", PuzzleType.BLUNDER_AVOID, 10, (1, 10)),
            ("Opening Gamelets", PuzzleType.GAMELET, 10, (1, 6)),
        ]

        results = {}
        total_sets = len(puzzle_sets)

        for i, (name, puzzle_type, count, difficulty_range) in enumerate(puzzle_sets):
            logger.info(f"Running enhanced puzzle set {i+1}/{total_sets}: {name}")

            # Create puzzle set
            puzzle_set = puzzle_db.create_puzzle_set(
                name=name,
                puzzle_type=puzzle_type,
                count=count,
                difficulty_range=difficulty_range
            )

            if not puzzle_set.puzzles:
                logger.warning(f"No puzzles found for set: {name}")
                continue

            # Run the enhanced puzzle session
            try:
                stats = await runner.run_enhanced_puzzle_session(puzzle_set, output_dir)
                results[name] = stats

                logger.info(f"Completed {name}: {stats.correct_solutions}/{stats.total_attempts} "
                           f"({stats.success_rate*100:.1f}%)")

            except Exception as e:
                logger.error(f"Error running puzzle set {name}: {e}")
                continue

        return results

    finally:
        # Cleanup
        await engine.stop()


# Convenience function for creating progress callbacks
def create_progress_callback(update_function: Callable) -> Callable:
    """
    Create a progress callback that wraps an update function.

    Args:
        update_function: Function to call with progress updates

    Returns:
        Async-compatible progress callback
    """
    async def progress_callback(progress_data: Dict[str, Any]) -> None:
        try:
            if asyncio.iscoroutinefunction(update_function):
                await update_function(progress_data)
            else:
                update_function(progress_data)
        except Exception as e:
            logger.warning(f"Progress callback error: {e}")

    return progress_callback
