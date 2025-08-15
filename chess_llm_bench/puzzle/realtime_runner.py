"""
Real-time Chess Puzzle Runner with Live Progress and Cost Tracking

This module provides real-time puzzle execution with live progress updates,
detailed cost tracking, and comprehensive performance monitoring. It integrates
with the existing budget tracking system and provides granular progress callbacks
for live UI updates.

Features:
- Real-time progress updates during puzzle solving
- Detailed cost breakdown per model and puzzle type
- Live timing and performance metrics
- Concurrent execution with live status updates
- Integration with budget tracking system
- Granular progress callbacks for UI
- Comprehensive error handling and recovery
- Performance analytics and trend tracking
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, Tuple
import traceback

import chess

from . import (
    PuzzlePosition, PuzzleAttempt, PuzzleStats, PuzzleSet,
    PuzzleType, TacticMotif, EndgameType
)
from .database import puzzle_db
from ..core.models import BotSpec, Config
from ..core.engine import create_engine
from ..core.budget import get_budget_tracker, record_llm_usage
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


class RealTimePuzzleStats:
    """Enhanced statistics with real-time updates and cost tracking."""

    def __init__(self, bot_name: str):
        self.bot_name = bot_name
        self.start_time = datetime.now()
        self.last_update = datetime.now()

        # Core puzzle stats
        self.total_puzzles = 0
        self.current_puzzle = 0
        self.solved_puzzles = 0
        self.failed_puzzles = 0

        # Timing stats
        self.total_time = 0.0
        self.avg_time = 0.0
        self.fastest_solve = float('inf')
        self.slowest_solve = 0.0

        # Cost tracking
        self.total_cost = 0.0
        self.cost_by_type = {ptype: 0.0 for ptype in PuzzleType}
        self.token_usage = {
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0
        }

        # Performance by puzzle type
        self.type_stats = {ptype: {'solved': 0, 'total': 0, 'time': 0.0, 'cost': 0.0}
                          for ptype in PuzzleType}

        # Current puzzle info
        self.current_puzzle_type = None
        self.current_puzzle_title = ""
        self.current_puzzle_difficulty = 0
        self.current_puzzle_start_time = None
        self.is_solving = False

        # Status and progress
        self.status = "🚀 Initializing..."
        self.progress = 0.0
        self.success_rate = 0.0
        self.error_count = 0
        self.illegal_moves = 0

        # Performance trends (for live charts)
        self.recent_times = []
        self.recent_success = []
        self.success_streak = 0
        self.best_streak = 0

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since start."""
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def puzzles_per_minute(self) -> float:
        """Calculate solving rate."""
        elapsed = self.elapsed_time / 60.0
        return self.current_puzzle / elapsed if elapsed > 0 else 0.0

    @property
    def cost_per_puzzle(self) -> float:
        """Calculate average cost per puzzle."""
        return self.total_cost / self.current_puzzle if self.current_puzzle > 0 else 0.0

    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency (success rate weighted by speed)."""
        speed_factor = min(2.0, self.puzzles_per_minute / 2.0)
        return self.success_rate * speed_factor

    def update_puzzle_start(self, puzzle: PuzzlePosition, index: int):
        """Update stats when starting a new puzzle."""
        self.current_puzzle = index
        self.current_puzzle_type = puzzle.puzzle_type
        self.current_puzzle_title = puzzle.title
        self.current_puzzle_difficulty = puzzle.difficulty
        self.current_puzzle_start_time = time.time()
        self.is_solving = True
        self.progress = (index - 1) / self.total_puzzles if self.total_puzzles > 0 else 0.0

        emoji = {
            PuzzleType.TACTIC: "⚔️",
            PuzzleType.ENDGAME: "♔",
            PuzzleType.BLUNDER_AVOID: "⚠️",
            PuzzleType.GAMELET: "📖"
        }.get(puzzle.puzzle_type, "🧩")

        self.status = f"{emoji} Solving: {puzzle.title[:25]}..."

    def update_puzzle_complete(self, attempt: PuzzleAttempt, cost: float, tokens: Dict[str, int]):
        """Update stats when puzzle is completed."""
        # Update basic stats
        if attempt.is_correct:
            self.solved_puzzles += 1
            self.success_streak += 1
            self.best_streak = max(self.best_streak, self.success_streak)
        else:
            self.failed_puzzles += 1
            self.success_streak = 0

        # Update timing
        self.total_time += attempt.response_time
        self.avg_time = self.total_time / self.current_puzzle
        self.fastest_solve = min(self.fastest_solve, attempt.response_time)
        self.slowest_solve = max(self.slowest_solve, attempt.response_time)

        # Update costs
        self.total_cost += cost
        if self.current_puzzle_type:
            self.cost_by_type[self.current_puzzle_type] += cost

        # Update token usage
        self.token_usage['total_tokens'] += tokens.get('total_tokens', 0)
        self.token_usage['prompt_tokens'] += tokens.get('prompt_tokens', 0)
        self.token_usage['completion_tokens'] += tokens.get('completion_tokens', 0)

        # Update type-specific stats
        if self.current_puzzle_type:
            type_stat = self.type_stats[self.current_puzzle_type]
            type_stat['total'] += 1
            type_stat['time'] += attempt.response_time
            type_stat['cost'] += cost
            if attempt.is_correct:
                type_stat['solved'] += 1

        # Update performance trends
        self.recent_times.append(attempt.response_time)
        self.recent_success.append(1.0 if attempt.is_correct else 0.0)
        if len(self.recent_times) > 20:  # Keep last 20 results
            self.recent_times.pop(0)
            self.recent_success.pop(0)

        # Update overall stats
        self.success_rate = self.solved_puzzles / self.current_puzzle
        self.progress = self.current_puzzle / self.total_puzzles if self.total_puzzles > 0 else 0.0
        self.illegal_moves += attempt.illegal_attempts
        self.is_solving = False

        # Update status
        result_emoji = "✅" if attempt.is_correct else "❌"
        type_emoji = {
            PuzzleType.TACTIC: "⚔️",
            PuzzleType.ENDGAME: "♔",
            PuzzleType.BLUNDER_AVOID: "⚠️",
            PuzzleType.GAMELET: "📖"
        }.get(self.current_puzzle_type, "🧩")

        self.status = f"{type_emoji} {result_emoji} #{self.current_puzzle} ({attempt.response_time:.1f}s, ${cost:.4f})"
        self.last_update = datetime.now()


class RealTimePuzzleRunner:
    """Real-time puzzle runner with live updates and cost tracking."""

    def __init__(self, bot_spec: BotSpec, config: Config,
                 progress_callback: Optional[Callable] = None):
        """Initialize the real-time runner."""
        self.bot_spec = bot_spec
        self.config = config
        self.progress_callback = progress_callback

        # Initialize components
        self.llm_client = None
        self.engine = None
        self.budget_tracker = get_budget_tracker()

        # Stats tracking
        self.stats = RealTimePuzzleStats(bot_spec.name)

        # Control flags
        self._cancelled = False
        self._paused = False

    async def initialize(self):
        """Initialize LLM client and engine."""
        try:
            # Create LLM client
            self.llm_client = LLMClient(self.bot_spec)

            # Create and start engine
            self.engine = create_engine(self.config)
            await self.engine.start()

            logger.info(f"Initialized real-time runner for {self.bot_spec.name}")

        except Exception as e:
            logger.error(f"Failed to initialize runner for {self.bot_spec.name}: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
        if self.engine:
            await self.engine.stop()

    async def run_puzzle_set(self, puzzle_set: PuzzleSet,
                           timeout: float = 30.0) -> Dict[str, PuzzleStats]:
        """Run a puzzle set with real-time updates."""
        try:
            await self.initialize()

            # Initialize stats
            self.stats.total_puzzles = len(puzzle_set.puzzles)

            # Notify start
            await self._notify_progress({
                'event': 'session_started',
                'bot_name': self.bot_spec.name,
                'total_puzzles': len(puzzle_set.puzzles),
                'puzzle_set': puzzle_set.name
            })

            # Group puzzles by type for organized results
            results_by_type = {}

            # Process each puzzle
            for i, puzzle in enumerate(puzzle_set.puzzles, 1):
                if self._cancelled:
                    break

                # Update puzzle start
                self.stats.update_puzzle_start(puzzle, i)

                await self._notify_progress({
                    'event': 'puzzle_started',
                    'puzzle_index': i,
                    'puzzle': {
                        'title': puzzle.title,
                        'type': puzzle.puzzle_type.value,
                        'difficulty': puzzle.difficulty
                    },
                    'stats': self._get_stats_dict()
                })

                # Solve puzzle
                try:
                    attempt, cost, tokens = await self._solve_puzzle_with_tracking(
                        puzzle, timeout, i
                    )

                    # Update stats
                    self.stats.update_puzzle_complete(attempt, cost, tokens)

                    # Group by type for results
                    puzzle_type_name = puzzle.puzzle_type.value
                    if puzzle_type_name not in results_by_type:
                        results_by_type[puzzle_type_name] = PuzzleStats()

                    results_by_type[puzzle_type_name].add_attempt(attempt)

                    # Notify progress
                    await self._notify_progress({
                        'event': 'puzzle_completed',
                        'puzzle_index': i,
                        'attempt': {
                            'is_correct': attempt.is_correct,
                            'response_time': attempt.response_time,
                            'cost': cost
                        },
                        'stats': self._get_stats_dict()
                    })

                except Exception as e:
                    logger.error(f"Error solving puzzle {i}: {e}")
                    self.stats.error_count += 1

                    await self._notify_progress({
                        'event': 'puzzle_error',
                        'puzzle_index': i,
                        'error': str(e),
                        'stats': self._get_stats_dict()
                    })

            # Notify completion
            await self._notify_progress({
                'event': 'session_completed',
                'stats': self._get_stats_dict(),
                'final_results': self._get_final_results()
            })

            return results_by_type

        finally:
            await self.cleanup()

    async def _solve_puzzle_with_tracking(self, puzzle: PuzzlePosition,
                                        timeout: float, index: int) -> Tuple[PuzzleAttempt, float, Dict[str, int]]:
        """Solve a puzzle with cost tracking."""
        start_time = time.time()

        # Track initial budget state
        initial_cost = self.budget_tracker.get_current_cost()
        initial_tokens = self.budget_tracker.get_token_usage()

        # Create puzzle prompt
        board = chess.Board(puzzle.fen)
        prompt = self._create_puzzle_prompt(puzzle, board)

        # Solve puzzle
        illegal_attempts = 0
        llm_move = ""
        raw_response = ""

        try:
            # Get LLM response
            response = await asyncio.wait_for(
                self.llm_client.get_move(board, prompt),
                timeout=timeout
            )

            raw_response = str(response)
            llm_move = self._extract_move_from_response(response, board)

            # Handle illegal moves
            while llm_move and not self._is_legal_move(llm_move, board):
                illegal_attempts += 1
                if illegal_attempts >= 3:
                    break

                correction_prompt = f"The move '{llm_move}' is illegal. Please provide a legal move in UCI format:"
                try:
                    response = await asyncio.wait_for(
                        self.llm_client.get_move(board, correction_prompt),
                        timeout=timeout / 2
                    )
                    llm_move = self._extract_move_from_response(response, board)
                except asyncio.TimeoutError:
                    break

        except asyncio.TimeoutError:
            logger.warning(f"Puzzle {index} timed out")
            raw_response = "TIMEOUT"
        except Exception as e:
            logger.error(f"Error solving puzzle {index}: {e}")
            raw_response = f"ERROR: {str(e)}"

        response_time = time.time() - start_time

        # Calculate costs and token usage
        final_cost = self.budget_tracker.get_current_cost()
        final_tokens = self.budget_tracker.get_token_usage()

        puzzle_cost = final_cost - initial_cost
        puzzle_tokens = {
            'total_tokens': final_tokens.get('total_tokens', 0) - initial_tokens.get('total_tokens', 0),
            'prompt_tokens': final_tokens.get('prompt_tokens', 0) - initial_tokens.get('prompt_tokens', 0),
            'completion_tokens': final_tokens.get('completion_tokens', 0) - initial_tokens.get('completion_tokens', 0)
        }

        # Evaluate move
        is_correct = self._evaluate_move(llm_move, puzzle)

        # Create attempt record
        attempt = PuzzleAttempt(
            puzzle_id=f"{puzzle.puzzle_type.value}_{index}",
            llm_move=llm_move,
            response_time=response_time,
            is_correct=is_correct,
            raw_response=raw_response,
            illegal_attempts=illegal_attempts
        )

        return attempt, puzzle_cost, puzzle_tokens

    def _create_puzzle_prompt(self, puzzle: PuzzlePosition, board: chess.Board) -> str:
        """Create puzzle prompt."""
        base_prompt = f"Position (FEN): {puzzle.fen}\n\n"
        base_prompt += f"Current board:\n{board}\n\n"

        if puzzle.puzzle_type == PuzzleType.TACTIC:
            if puzzle.mate_in:
                base_prompt += f"🎯 Find the best move that leads to mate in {puzzle.mate_in}.\n"
            else:
                base_prompt += "🎯 Find the best tactical move.\n"
        elif puzzle.puzzle_type == PuzzleType.ENDGAME:
            base_prompt += "♔ This is an endgame position. Find the best move.\n"
        elif puzzle.puzzle_type == PuzzleType.BLUNDER_AVOID:
            base_prompt += "⚠️ This is a critical position. Avoid major blunders.\n"
        elif puzzle.puzzle_type == PuzzleType.GAMELET:
            base_prompt += "📖 This is an opening position. Find the best move.\n"

        base_prompt += "\nProvide your answer as a single UCI move (e.g., 'e2e4', 'g1f3').\n"

        return base_prompt

    def _extract_move_from_response(self, response: str, board: chess.Board) -> str:
        """Extract UCI move from response."""
        import re

        response_str = str(response).strip()

        # Try UCI format
        uci_pattern = r'\b[a-h][1-8][a-h][1-8][qrbn]?\b'
        uci_matches = re.findall(uci_pattern, response_str.lower())

        for move_str in uci_matches:
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    return move_str
            except ValueError:
                continue

        return ""

    def _is_legal_move(self, move_str: str, board: chess.Board) -> bool:
        """Check if move is legal."""
        if not move_str:
            return False
        try:
            move = chess.Move.from_uci(move_str)
            return move in board.legal_moves
        except ValueError:
            return False

    def _evaluate_move(self, llm_move: str, puzzle: PuzzlePosition) -> bool:
        """Evaluate if move is correct."""
        if not llm_move:
            return False
        return llm_move.lower() == puzzle.best_move.lower()

    def _get_stats_dict(self) -> Dict[str, Any]:
        """Get current stats as dictionary."""
        return {
            'bot_name': self.stats.bot_name,
            'current_puzzle': self.stats.current_puzzle,
            'total_puzzles': self.stats.total_puzzles,
            'solved_puzzles': self.stats.solved_puzzles,
            'failed_puzzles': self.stats.failed_puzzles,
            'success_rate': self.stats.success_rate,
            'progress': self.stats.progress,
            'avg_time': self.stats.avg_time,
            'puzzles_per_minute': self.stats.puzzles_per_minute,
            'total_cost': self.stats.total_cost,
            'cost_per_puzzle': self.stats.cost_per_puzzle,
            'efficiency_score': self.stats.efficiency_score,
            'status': self.stats.status,
            'is_solving': self.stats.is_solving,
            'success_streak': self.stats.success_streak,
            'best_streak': self.stats.best_streak,
            'elapsed_time': self.stats.elapsed_time,
            'error_count': self.stats.error_count,
            'illegal_moves': self.stats.illegal_moves,
            'token_usage': self.stats.token_usage.copy(),
            'cost_by_type': {k.value: v for k, v in self.stats.cost_by_type.items()},
            'type_performance': {
                k.value: {
                    'solved': v['solved'],
                    'total': v['total'],
                    'success_rate': v['solved'] / v['total'] if v['total'] > 0 else 0.0,
                    'avg_time': v['time'] / v['total'] if v['total'] > 0 else 0.0,
                    'avg_cost': v['cost'] / v['total'] if v['total'] > 0 else 0.0
                }
                for k, v in self.stats.type_stats.items() if v['total'] > 0
            }
        }

    def _get_final_results(self) -> Dict[str, Any]:
        """Get final comprehensive results."""
        return {
            'summary': {
                'total_puzzles': self.stats.total_puzzles,
                'solved_puzzles': self.stats.solved_puzzles,
                'success_rate': self.stats.success_rate,
                'total_time': self.stats.total_time,
                'avg_time': self.stats.avg_time,
                'total_cost': self.stats.total_cost,
                'cost_per_puzzle': self.stats.cost_per_puzzle,
                'efficiency_score': self.stats.efficiency_score
            },
            'timing': {
                'fastest_solve': self.stats.fastest_solve if self.stats.fastest_solve != float('inf') else 0,
                'slowest_solve': self.stats.slowest_solve,
                'avg_time': self.stats.avg_time,
                'total_time': self.stats.total_time
            },
            'costs': {
                'total_cost': self.stats.total_cost,
                'cost_per_puzzle': self.stats.cost_per_puzzle,
                'cost_by_type': {k.value: v for k, v in self.stats.cost_by_type.items()},
                'token_usage': self.stats.token_usage.copy()
            },
            'performance': {
                'success_rate': self.stats.success_rate,
                'success_streak': self.stats.success_streak,
                'best_streak': self.stats.best_streak,
                'efficiency_score': self.stats.efficiency_score,
                'error_count': self.stats.error_count,
                'illegal_moves': self.stats.illegal_moves
            }
        }

    async def _notify_progress(self, progress_data: Dict[str, Any]):
        """Notify progress callback."""
        if self.progress_callback:
            try:
                if asyncio.iscoroutinefunction(self.progress_callback):
                    await self.progress_callback(progress_data)
                else:
                    self.progress_callback(progress_data)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def cancel(self):
        """Cancel the current run."""
        self._cancelled = True

    def pause(self):
        """Pause the current run."""
        self._paused = True

    def resume(self):
        """Resume the current run."""
        self._paused = False


async def run_realtime_puzzle_benchmark(
    bot_specs: List[BotSpec],
    config: Config,
    output_dir: Path,
    puzzle_types: List[PuzzleType],
    difficulty_range: Tuple[int, int] = (1, 10),
    puzzle_count: int = 10,
    timeout: float = 30.0,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run real-time puzzle benchmark with live updates and cost tracking.

    Args:
        bot_specs: List of bots to test
        config: Configuration
        output_dir: Output directory
        puzzle_types: Types of puzzles to include
        difficulty_range: Difficulty range
        puzzle_count: Puzzles per type
        timeout: Timeout per puzzle
        progress_callback: Callback for progress updates

    Returns:
        Comprehensive results with costs and performance data
    """
    # Create puzzle sets
    puzzle_sets = []
    for puzzle_type in puzzle_types:
        puzzle_set = puzzle_db.create_puzzle_set(
            name=f"{puzzle_type.value.title()} Puzzles",
            puzzle_type=puzzle_type,
            count=puzzle_count,
            difficulty_range=difficulty_range
        )
        puzzle_sets.extend(puzzle_set.puzzles)

    # Create combined puzzle set
    combined_set = PuzzleSet(
        name="Real-time Benchmark",
        description=f"Combined puzzle set with {len(puzzle_sets)} puzzles",
        puzzles=puzzle_sets
    )

    # Run benchmarks for all bots
    results = {}

    for bot_spec in bot_specs:
        try:
            runner = RealTimePuzzleRunner(bot_spec, config, progress_callback)
            bot_results = await runner.run_puzzle_set(combined_set, timeout)

            # Convert PuzzleStats to compatible format
            converted_results = {}
            for puzzle_type_name, puzzle_stats in bot_results.items():
                converted_results[puzzle_type_name] = puzzle_stats

            results[bot_spec.name] = {
                'puzzle_results': converted_results,
                'final_stats': runner._get_final_results()
            }
        except Exception as e:
            logger.error(f"Error running benchmark for {bot_spec.name}: {e}")
            # Create empty PuzzleStats for error case
            from . import PuzzleStats
            empty_stats = PuzzleStats()
            results[bot_spec.name] = {
                'puzzle_results': {'error': empty_stats},
                'final_stats': {
                    'summary': {
                        'total_puzzles': 0,
                        'solved_puzzles': 0,
                        'success_rate': 0.0,
                        'total_time': 0.0,
                        'avg_time': 0.0,
                        'total_cost': 0.0,
                        'cost_per_puzzle': 0.0,
                        'efficiency_score': 0.0
                    },
                    'costs': {
                        'total_cost': 0.0,
                        'cost_per_puzzle': 0.0,
                        'cost_by_type': {},
                        'token_usage': {}
                    }
                }
            }

    return results
