"""
Modern Chess Puzzle CLI with Concurrent Execution and Real-time Stats

This module provides a beautiful, modern command-line interface for chess puzzle
solving with concurrent model testing, live statistics, and cool visual elements.

Features:
- Concurrent execution of multiple LLM models
- Real-time progress tracking and statistics
- Beautiful terminal UI with animations
- Live leaderboard updates
- Performance metrics dashboard
- Cool visual effects and modern styling
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import math

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.layout import Layout
from rich.align import Align
from rich.text import Text
from rich.columns import Columns
from rich.rule import Rule
from rich.box import ROUNDED, DOUBLE, HEAVY
from rich.style import Style
from rich import box
from rich.console import Group
from rich.padding import Padding

from . import PuzzleType, PuzzleStats, PuzzleAttempt
from .runner import run_puzzle_benchmark
from .enhanced_runner import run_enhanced_puzzle_benchmark, create_progress_callback
from .database import puzzle_db
from ..core.models import BotSpec, Config


@dataclass
class LivePuzzleStats:
    """Live statistics for real-time display."""

    bot_name: str
    total_puzzles: int = 0
    solved_puzzles: int = 0
    current_puzzle: int = 0
    avg_response_time: float = 0.0
    fastest_solve: float = float('inf')
    slowest_solve: float = 0.0
    current_streak: int = 0
    best_streak: int = 0
    start_time: Optional[datetime] = None
    last_update: Optional[datetime] = None

    # Performance by type
    tactics_score: int = 0
    tactics_total: int = 0
    endgames_score: int = 0
    endgames_total: int = 0
    blunders_score: int = 0
    blunders_total: int = 0
    gamelets_score: int = 0
    gamelets_total: int = 0

    # Status and progress
    status: str = "🚀 Starting..."
    current_type: str = ""
    is_solving: bool = False
    error_count: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        return (self.solved_puzzles / self.total_puzzles * 100) if self.total_puzzles > 0 else 0.0

    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time in seconds."""
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0

    @property
    def puzzles_per_minute(self) -> float:
        """Calculate solving rate."""
        elapsed = self.elapsed_time
        return (self.current_puzzle / elapsed * 60) if elapsed > 0 else 0.0

    @property
    def eta_seconds(self) -> float:
        """Estimate time to completion."""
        rate = self.puzzles_per_minute
        remaining = self.total_puzzles - self.current_puzzle
        return (remaining / rate * 60) if rate > 0 else 0.0


class ModernPuzzleCLI:
    """Modern, beautiful CLI for chess puzzle solving with concurrent execution."""

    def __init__(self):
        """Initialize the modern CLI."""
        self.console = Console()
        self.live_stats: Dict[str, LivePuzzleStats] = {}
        self.leaderboard: List[Tuple[str, float]] = []
        self.start_time = None
        self.total_bots = 0
        self.completed_bots = 0

        # Visual elements
        self.puzzle_emojis = {
            PuzzleType.TACTIC: "⚔️",
            PuzzleType.ENDGAME: "♔",
            PuzzleType.BLUNDER_AVOID: "⚠️",
            PuzzleType.GAMELET: "📖"
        }

        self.difficulty_colors = {
            1: "green", 2: "green", 3: "yellow", 4: "yellow",
            5: "orange1", 6: "orange1", 7: "red", 8: "red",
            9: "magenta", 10: "bright_magenta"
        }

    def create_header(self) -> Panel:
        """Create the main header panel."""
        title_text = Text()
        title_text.append("🧩 ", style="bold magenta")
        title_text.append("CHESS PUZZLE BENCHMARK", style="bold cyan")
        title_text.append(" 🧩", style="bold magenta")

        subtitle = Text("Advanced LLM Chess Intelligence Assessment", style="italic dim")

        header_content = Align.center(Group(title_text, subtitle))

        return Panel(
            header_content,
            box=DOUBLE,
            style="cyan",
            padding=(1, 2)
        )

    def create_overview_panel(self) -> Panel:
        """Create overview statistics panel."""
        if not self.live_stats:
            return Panel("No active sessions", title="📊 Overview", box=ROUNDED)

        total_puzzles = sum(stats.current_puzzle for stats in self.live_stats.values())
        total_solved = sum(stats.solved_puzzles for stats in self.live_stats.values())
        avg_success = sum(stats.success_rate for stats in self.live_stats.values()) / len(self.live_stats)

        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        overview_table = Table(show_header=False, box=None, padding=(0, 1))
        overview_table.add_column(style="bold cyan")
        overview_table.add_column(style="bold white")

        overview_table.add_row("🎯 Total Puzzles:", f"{total_puzzles:,}")
        overview_table.add_row("✅ Solved:", f"{total_solved:,}")
        overview_table.add_row("📈 Avg Success:", f"{avg_success:.1f}%")
        overview_table.add_row("⏱️  Elapsed:", f"{elapsed/60:.1f}m")
        overview_table.add_row("🤖 Active Bots:", f"{len(self.live_stats)}")
        overview_table.add_row("✨ Completed:", f"{self.completed_bots}/{self.total_bots}")

        return Panel(
            overview_table,
            title="📊 [bold cyan]Overview[/bold cyan]",
            box=ROUNDED,
            style="bright_blue"
        )

    def create_leaderboard_panel(self) -> Panel:
        """Create live leaderboard panel."""
        if not self.live_stats:
            return Panel("No data yet", title="🏆 Leaderboard", box=ROUNDED)

        # Sort by success rate
        sorted_stats = sorted(
            self.live_stats.items(),
            key=lambda x: x[1].success_rate,
            reverse=True
        )

        leaderboard_table = Table(box=None, padding=(0, 1))
        leaderboard_table.add_column("Rank", style="bold yellow", width=4)
        leaderboard_table.add_column("Bot", style="bold cyan", width=15)
        leaderboard_table.add_column("Score", style="bold green", width=8)
        leaderboard_table.add_column("Rate", style="bold white", width=7)
        leaderboard_table.add_column("Streak", style="bold magenta", width=7)

        medals = ["🥇", "🥈", "🥉", "🏅", "🎖️"]

        for i, (bot_name, stats) in enumerate(sorted_stats[:5]):
            rank_icon = medals[i] if i < len(medals) else f"{i+1}"
            score_text = f"{stats.solved_puzzles}/{stats.total_puzzles}"
            rate_text = f"{stats.success_rate:.1f}%"
            streak_text = f"{stats.current_streak}🔥" if stats.current_streak > 0 else "-"

            leaderboard_table.add_row(
                rank_icon,
                bot_name[:13] + "..." if len(bot_name) > 15 else bot_name,
                score_text,
                rate_text,
                streak_text
            )

        return Panel(
            leaderboard_table,
            title="🏆 [bold yellow]Live Leaderboard[/bold yellow]",
            box=ROUNDED,
            style="yellow"
        )

    def create_bot_panels(self) -> List[Panel]:
        """Create individual bot status panels."""
        panels = []

        for bot_name, stats in self.live_stats.items():
            # Progress bar
            progress = stats.current_puzzle / stats.total_puzzles if stats.total_puzzles > 0 else 0
            progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))

            # Status indicators
            status_color = "green" if stats.is_solving else "yellow" if stats.current_puzzle > 0 else "red"
            status_icon = "🔄" if stats.is_solving else "✅" if stats.current_puzzle == stats.total_puzzles else "⏸️"

            # Performance breakdown
            performance_table = Table(show_header=False, box=None, padding=(0, 0))
            performance_table.add_column(width=8)
            performance_table.add_column(width=10)

            # Add puzzle type performance
            if stats.tactics_total > 0:
                tactics_rate = stats.tactics_score / stats.tactics_total * 100
                performance_table.add_row(
                    "⚔️ Tactics:",
                    f"[green]{stats.tactics_score}[/green]/[dim]{stats.tactics_total}[/dim] ({tactics_rate:.0f}%)"
                )

            if stats.endgames_total > 0:
                endgames_rate = stats.endgames_score / stats.endgames_total * 100
                performance_table.add_row(
                    "♔ Endgames:",
                    f"[green]{stats.endgames_score}[/green]/[dim]{stats.endgames_total}[/dim] ({endgames_rate:.0f}%)"
                )

            if stats.blunders_total > 0:
                blunders_rate = stats.blunders_score / stats.blunders_total * 100
                performance_table.add_row(
                    "⚠️ Blunders:",
                    f"[green]{stats.blunders_score}[/green]/[dim]{stats.blunders_total}[/dim] ({blunders_rate:.0f}%)"
                )

            if stats.gamelets_total > 0:
                gamelets_rate = stats.gamelets_score / stats.gamelets_total * 100
                performance_table.add_row(
                    "📖 Gamelets:",
                    f"[green]{stats.gamelets_score}[/green]/[dim]{stats.gamelets_total}[/dim] ({gamelets_rate:.0f}%)"
                )

            # Main stats
            main_stats = Table(show_header=False, box=None, padding=(0, 1))
            main_stats.add_column(width=12)
            main_stats.add_column()

            main_stats.add_row("Progress:", f"[{status_color}]{progress_bar}[/{status_color}] {stats.current_puzzle}/{stats.total_puzzles}")
            main_stats.add_row("Success Rate:", f"[bold green]{stats.success_rate:.1f}%[/bold green]")
            main_stats.add_row("Avg Time:", f"{stats.avg_response_time:.2f}s")

            if stats.fastest_solve != float('inf'):
                main_stats.add_row("Fastest:", f"⚡ {stats.fastest_solve:.2f}s")

            if stats.current_streak > 0:
                main_stats.add_row("Streak:", f"🔥 {stats.current_streak} ({stats.best_streak} best)")

            main_stats.add_row("Status:", f"{status_icon} [italic]{stats.status}[/italic]")

            # Combine into panel content
            panel_content = Group(
                main_stats,
                Rule(style="dim"),
                performance_table
            )

            # Panel styling based on performance
            if stats.success_rate >= 80:
                panel_style = "bright_green"
                border_box = ROUNDED
            elif stats.success_rate >= 60:
                panel_style = "yellow"
                border_box = ROUNDED
            elif stats.success_rate >= 40:
                panel_style = "bright_red"
                border_box = ROUNDED
            else:
                panel_style = "red"
                border_box = ROUNDED

            panel = Panel(
                panel_content,
                title=f"🤖 [bold]{bot_name}[/bold]",
                box=border_box,
                style=panel_style,
                width=35
            )

            panels.append(panel)

        return panels

    def create_live_dashboard(self) -> Layout:
        """Create the main live dashboard layout."""
        layout = Layout()

        # Main sections
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )

        # Header
        layout["header"].update(self.create_header())

        # Body split into left panel and main area
        layout["body"].split_row(
            Layout(name="sidebar", ratio=1),
            Layout(name="main", ratio=2)
        )

        # Sidebar with overview and leaderboard
        layout["sidebar"].split_column(
            Layout(name="overview"),
            Layout(name="leaderboard")
        )

        layout["overview"].update(self.create_overview_panel())
        layout["leaderboard"].update(self.create_leaderboard_panel())

        # Main area with bot panels
        bot_panels = self.create_bot_panels()
        if bot_panels:
            # Create grid of bot panels
            if len(bot_panels) <= 2:
                layout["main"].split_column(*[Layout() for _ in bot_panels])
                for i, panel in enumerate(bot_panels):
                    layout["main"].splitters[i].update(panel)
            elif len(bot_panels) <= 4:
                layout["main"].split_column(
                    Layout(name="top_row"),
                    Layout(name="bottom_row")
                )
                # Split top and bottom rows
                top_panels = bot_panels[:2]
                bottom_panels = bot_panels[2:]

                if len(top_panels) == 2:
                    layout["top_row"].split_row(*[Layout() for _ in top_panels])
                    for i, panel in enumerate(top_panels):
                        layout["top_row"].splitters[i].update(panel)
                else:
                    layout["top_row"].update(top_panels[0])

                if bottom_panels:
                    if len(bottom_panels) == 2:
                        layout["bottom_row"].split_row(*[Layout() for _ in bottom_panels])
                        for i, panel in enumerate(bottom_panels):
                            layout["bottom_row"].splitters[i].update(panel)
                    else:
                        layout["bottom_row"].update(bottom_panels[0])
            else:
                # For more than 4 bots, create a scrollable view
                columns = Columns(bot_panels, equal=True, expand=True)
                layout["main"].update(Panel(columns, title="🤖 Bot Status", box=ROUNDED))
        else:
            layout["main"].update(Panel("Initializing bots...", title="Status", box=ROUNDED))

        # Footer with tips
        tips = [
            "💡 Tip: Higher difficulty puzzles take longer to solve",
            "🎯 Tip: Tactical puzzles test pattern recognition",
            "♔ Tip: Endgame puzzles require theoretical knowledge",
            "⚠️ Tip: Blunder avoidance tests defensive skills",
            "📖 Tip: Gamelets test opening theory understanding"
        ]

        current_tip = tips[int(time.time() / 5) % len(tips)]  # Rotate every 5 seconds

        footer_content = Align.center(
            Text(current_tip, style="italic dim cyan")
        )

        layout["footer"].update(Panel(footer_content, box=ROUNDED, style="dim"))

        return layout

    async def update_bot_stats(self, bot_name: str, puzzle_result: Dict[str, Any]) -> None:
        """Update statistics for a specific bot."""
        if bot_name not in self.live_stats:
            return

        stats = self.live_stats[bot_name]
        stats.last_update = datetime.now()

        # Handle enhanced runner progress updates
        status = puzzle_result.get('status', '')

        if status == 'session_started':
            stats.total_puzzles = puzzle_result.get('total_puzzles', stats.total_puzzles)
            stats.status = f"🚀 Starting {puzzle_result.get('puzzle_set_name', 'puzzles')}..."
            stats.is_solving = True

        elif status == 'puzzle_started':
            puzzle_index = puzzle_result.get('puzzle_index', 0)
            puzzle_title = puzzle_result.get('puzzle_title', '')
            puzzle_type = puzzle_result.get('puzzle_type')
            stats.current_puzzle = puzzle_index
            stats.is_solving = True
            emoji = self.puzzle_emojis.get(puzzle_type, "🧩")
            stats.status = f"{emoji} Solving: {puzzle_title[:20]}..."

        elif status == 'puzzle_completed' or puzzle_result.get('completed_puzzle'):
            puzzle_index = puzzle_result.get('puzzle_index', stats.current_puzzle)
            response_time = puzzle_result.get('response_time', 0)
            is_correct = puzzle_result.get('is_correct', False)
            puzzle_type = puzzle_result.get('puzzle_type')

            stats.current_puzzle = puzzle_index

            # Update timing stats
            if response_time > 0:
                total_time = stats.avg_response_time * (stats.current_puzzle - 1) + response_time
                stats.avg_response_time = total_time / stats.current_puzzle
                stats.fastest_solve = min(stats.fastest_solve, response_time)
                stats.slowest_solve = max(stats.slowest_solve, response_time)

            # Update success tracking
            if is_correct:
                stats.solved_puzzles += 1
                stats.current_streak += 1
                stats.best_streak = max(stats.best_streak, stats.current_streak)
            else:
                stats.current_streak = 0

            # Update by puzzle type
            if puzzle_type == PuzzleType.TACTIC:
                stats.tactics_total += 1
                if is_correct:
                    stats.tactics_score += 1
            elif puzzle_type == PuzzleType.ENDGAME:
                stats.endgames_total += 1
                if is_correct:
                    stats.endgames_score += 1
            elif puzzle_type == PuzzleType.BLUNDER_AVOID:
                stats.blunders_total += 1
                if is_correct:
                    stats.blunders_score += 1
            elif puzzle_type == PuzzleType.GAMELET:
                stats.gamelets_total += 1
                if is_correct:
                    stats.gamelets_score += 1

            # Update status
            emoji = self.puzzle_emojis.get(puzzle_type, "🧩")
            result_emoji = "✅" if is_correct else "❌"
            stats.status = f"{emoji} {result_emoji} Completed #{puzzle_index}"
            stats.current_type = puzzle_type.value if puzzle_type else ""

        elif status == 'session_completed':
            stats.status = "✨ Session Complete!"
            stats.is_solving = False

        elif status == 'solving_puzzle':
            puzzle_type = puzzle_result.get('puzzle_type')
            emoji = self.puzzle_emojis.get(puzzle_type, "🧩")
            stats.status = f"{emoji} 🤔 Thinking..."
            stats.is_solving = True

        elif status == 'puzzle_timeout':
            stats.status = "⏰ Puzzle timed out"

        elif status == 'illegal_move_attempt':
            attempt_num = puzzle_result.get('attempt_number', 1)
            stats.status = f"🚫 Illegal move (attempt {attempt_num})"

        elif puzzle_result.get('status_update'):
            stats.status = puzzle_result['status_update']
            stats.is_solving = puzzle_result.get('is_solving', False)

        elif puzzle_result.get('error') or status in ['puzzle_error', 'session_error', 'llm_error']:
            stats.error_count += 1
            error_msg = puzzle_result.get('error', 'Unknown error')
            stats.status = f"❌ Error: {error_msg[:30]}..."
            stats.is_solving = False

    async def run_concurrent_benchmark(self, bot_specs: List[BotSpec],
                                     config: Config, output_dir: Path,
                                     puzzle_types: List[PuzzleType],
                                     difficulty_range: Tuple[int, int],
                                     puzzle_count: int) -> Dict[str, Dict[str, PuzzleStats]]:
        """Run puzzle benchmarks for multiple bots concurrently with live updates."""

        self.start_time = datetime.now()
        self.total_bots = len(bot_specs)
        self.completed_bots = 0

        # Initialize stats for all bots
        estimated_puzzles = len(puzzle_types) * puzzle_count
        for bot_spec in bot_specs:
            self.live_stats[bot_spec.name] = LivePuzzleStats(
                bot_name=bot_spec.name,
                total_puzzles=estimated_puzzles,
                start_time=datetime.now()
            )

        # Create semaphore to limit concurrent bots (avoid overwhelming the system)
        semaphore = asyncio.Semaphore(min(4, len(bot_specs)))

        async def run_single_bot(bot_spec: BotSpec) -> Tuple[str, Dict[str, PuzzleStats]]:
            """Run puzzle benchmark for a single bot with progress updates."""
            async with semaphore:
                try:
                    # Create progress callback for this bot
                    async def progress_update(progress_data: Dict[str, Any]) -> None:
                        await self.update_bot_stats(bot_spec.name, progress_data)

                    # Update status
                    await self.update_bot_stats(bot_spec.name, {
                        'status_update': '🚀 Initializing...',
                        'is_solving': True
                    })

                    # Run the enhanced benchmark with real-time progress updates
                    results = await run_enhanced_puzzle_benchmark(
                        bot_spec, config, output_dir,
                        progress_callback=progress_update
                    )

                    # Mark as completed
                    await self.update_bot_stats(bot_spec.name, {
                        'status_update': '✨ Completed!',
                        'is_solving': False
                    })

                    self.completed_bots += 1
                    return bot_spec.name, results

                except Exception as e:
                    await self.update_bot_stats(bot_spec.name, {
                        'error': str(e),
                        'is_solving': False
                    })
                    return bot_spec.name, {}

        # Start all bot benchmarks concurrently
        tasks = [asyncio.create_task(run_single_bot(bot_spec)) for bot_spec in bot_specs]

        # Run with live display
        results = {}
        try:
            with Live(self.create_live_dashboard(), refresh_per_second=2, console=self.console) as live:
                # Wait for all tasks to complete while updating display
                for completed_task in asyncio.as_completed(tasks):
                    try:
                        bot_name, bot_results = await completed_task
                        results[bot_name] = bot_results

                        # Update the live display
                        live.update(self.create_live_dashboard())
                    except Exception as e:
                        logger.error(f"Task failed: {e}")
                        continue

                # Show final results for a moment
                await asyncio.sleep(2)

        except Exception as e:
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for tasks to be cancelled
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        return results

    def show_final_results(self, results: Dict[str, Dict[str, PuzzleStats]]) -> None:
        """Display beautiful final results summary."""

        # Header
        self.console.print("\n")
        self.console.rule("🎉 [bold cyan]BENCHMARK COMPLETE[/bold cyan] 🎉", style="cyan")
        self.console.print("\n")

        if not results:
            self.console.print(Panel("No results to display", title="❌ Error", style="red"))
            return

        # Calculate overall statistics
        all_results = []
        for bot_name, bot_results in results.items():
            if isinstance(bot_results, dict):
                total_correct = sum(stats.correct_solutions for stats in bot_results.values())
                total_attempts = sum(stats.total_attempts for stats in bot_results.values())
                avg_time = sum(stats.average_response_time for stats in bot_results.values()) / len(bot_results) if bot_results else 0
                success_rate = total_correct / total_attempts if total_attempts > 0 else 0

                all_results.append({
                    'name': bot_name,
                    'success_rate': success_rate,
                    'total_correct': total_correct,
                    'total_attempts': total_attempts,
                    'avg_time': avg_time,
                    'results': bot_results
                })

        # Sort by success rate
        all_results.sort(key=lambda x: x['success_rate'], reverse=True)

        # Create final results table
        final_table = Table(box=HEAVY, title="🏆 Final Rankings", title_style="bold yellow")
        final_table.add_column("Rank", style="bold yellow", width=6)
        final_table.add_column("Bot Name", style="bold cyan", width=20)
        final_table.add_column("Success Rate", style="bold green", width=12)
        final_table.add_column("Solved/Total", style="bold white", width=12)
        final_table.add_column("Avg Time", style="bold blue", width=10)
        final_table.add_column("Performance", style="bold magenta", width=15)

        medals = ["🥇", "🥈", "🥉"]

        for i, result in enumerate(all_results):
            rank_display = medals[i] if i < 3 else f"#{i+1}"

            # Performance indicator
            if result['success_rate'] >= 0.8:
                perf_indicator = "🚀 Excellent"
                perf_style = "bold green"
            elif result['success_rate'] >= 0.6:
                perf_indicator = "✨ Good"
                perf_style = "bold yellow"
            elif result['success_rate'] >= 0.4:
                perf_indicator = "📈 Fair"
                perf_style = "bold orange1"
            else:
                perf_indicator = "💪 Needs Work"
                perf_style = "bold red"

            final_table.add_row(
                rank_display,
                result['name'],
                f"{result['success_rate']:.1%}",
                f"{result['total_correct']}/{result['total_attempts']}",
                f"{result['avg_time']:.2f}s",
                f"[{perf_style}]{perf_indicator}[/{perf_style}]"
            )

        self.console.print(Align.center(final_table))

        # Show detailed breakdown for top performer
        if all_results:
            best_bot = all_results[0]
            self.console.print("\n")

            detail_table = Table(title=f"🎯 Detailed Breakdown - {best_bot['name']}", box=ROUNDED)
            detail_table.add_column("Puzzle Type", style="bold cyan")
            detail_table.add_column("Performance", style="bold green")
            detail_table.add_column("Success Rate", style="bold yellow")
            detail_table.add_column("Avg Time", style="bold blue")

            for puzzle_set, stats in best_bot['results'].items():
                emoji = "⚔️" if "tactic" in puzzle_set.lower() else \
                       "♔" if "endgame" in puzzle_set.lower() else \
                       "⚠️" if "blunder" in puzzle_set.lower() else \
                       "📖" if "gamelet" in puzzle_set.lower() else "🧩"

                detail_table.add_row(
                    f"{emoji} {puzzle_set}",
                    f"{stats.correct_solutions}/{stats.total_attempts}",
                    f"{stats.success_rate:.1%}",
                    f"{stats.average_response_time:.2f}s"
                )

            self.console.print(Align.center(detail_table))

        # Fun facts and insights
        if len(all_results) > 1:
            best = all_results[0]
            worst = all_results[-1]

            self.console.print("\n")
            insights_panel = Panel(
                Group(
                    Text("🔍 Insights & Fun Facts", style="bold cyan"),
                    Text(""),
                    Text(f"🏆 Champion: {best['name']} with {best['success_rate']:.1%} success rate", style="green"),
                    Text(f"📈 Improvement opportunity: {worst['name']} could improve by {(best['success_rate'] - worst['success_rate']):.1%}", style="yellow"),
                    Text(f"⚡ Speed range: {min(r['avg_time'] for r in all_results):.2f}s - {max(r['avg_time'] for r in all_results):.2f}s", style="blue"),
                    Text(f"🎯 Total puzzles solved: {sum(r['total_correct'] for r in all_results):,}", style="magenta"),
                ),
                title="✨ Benchmark Insights",
                box=ROUNDED,
                style="bright_blue"
            )

            self.console.print(Align.center(insights_panel))

        # Final message
        self.console.print("\n")
        self.console.print(
            Align.center(
                Text("Thanks for using Chess Puzzle Benchmark! 🎯♔⚔️", style="bold bright_cyan")
            )
        )
        self.console.print("\n")


async def run_modern_puzzle_cli(bot_specs: List[BotSpec], config: Config,
                               output_dir: Path, puzzle_types: List[PuzzleType],
                               difficulty_range: Tuple[int, int] = (1, 10),
                               puzzle_count: int = 10) -> Dict[str, Dict[str, PuzzleStats]]:
    """
    Run the modern puzzle CLI with concurrent execution and beautiful visuals.

    Args:
        bot_specs: List of bot specifications to test
        config: Configuration for the benchmark
        output_dir: Output directory for results
        puzzle_types: Types of puzzles to include
        difficulty_range: Min and max difficulty levels
        puzzle_count: Number of puzzles per type

    Returns:
        Dictionary mapping bot names to their results
    """
    cli = ModernPuzzleCLI()

    # Show startup animation
    with cli.console.status("[bold cyan]Initializing Chess Puzzle Benchmark...", spinner="dots"):
        await asyncio.sleep(1)

    cli.console.clear()

    try:
        # Run concurrent benchmark with live updates
        results = await cli.run_concurrent_benchmark(
            bot_specs, config, output_dir, puzzle_types, difficulty_range, puzzle_count
        )

        # Show final results
        cli.show_final_results(results)

        return results

    except KeyboardInterrupt:
        cli.console.print("\n[bold yellow]⚠️  Benchmark interrupted by user[/bold yellow]")
        return {}
    except Exception as e:
        cli.console.print(f"\n[bold red]❌ Benchmark failed: {e}[/bold red]")
        return {}
