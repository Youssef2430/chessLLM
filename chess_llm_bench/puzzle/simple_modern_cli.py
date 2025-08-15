"""
Simplified Modern Chess Puzzle CLI

A streamlined, working implementation of the modern chess puzzle CLI
with concurrent execution, real-time stats, and beautiful visuals.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.layout import Layout
from rich.align import Align
from rich.text import Text
from rich.columns import Columns
from rich.box import ROUNDED, DOUBLE
from rich.style import Style

from . import PuzzleType, PuzzleStats
from .runner import run_puzzle_benchmark
from .database import puzzle_db
from ..core.models import BotSpec, Config


@dataclass
class SimpleBotStats:
    """Simplified bot statistics for display."""
    name: str
    status: str = "🚀 Starting..."
    total_puzzles: int = 0
    solved_puzzles: int = 0
    current_puzzle: int = 0
    success_rate: float = 0.0
    avg_time: float = 0.0
    is_active: bool = True
    start_time: Optional[datetime] = None

    def update_completion(self, results: Dict[str, PuzzleStats]):
        """Update stats when bot completes."""
        if results:
            self.total_puzzles = sum(stats.total_attempts for stats in results.values())
            self.solved_puzzles = sum(stats.correct_solutions for stats in results.values())
            self.current_puzzle = self.total_puzzles
            self.success_rate = self.solved_puzzles / self.total_puzzles if self.total_puzzles > 0 else 0.0
            self.avg_time = sum(stats.average_response_time for stats in results.values()) / len(results)
            self.status = "✅ Complete!"
            self.is_active = False


class SimplePuzzleCLI:
    """Simplified modern CLI for puzzle benchmarking."""

    def __init__(self):
        self.console = Console()
        self.bot_stats: Dict[str, SimpleBotStats] = {}
        self.start_time = None
        self.completed_bots = 0
        self.total_bots = 0

    def create_header(self) -> Panel:
        """Create header panel."""
        title = Text("🧩 CHESS PUZZLE BENCHMARK 🧩", style="bold cyan")
        subtitle = Text("Concurrent LLM Testing with Real-time Stats", style="italic dim")
        return Panel(
            Align.center(Text.assemble(title, "\n", subtitle)),
            box=DOUBLE,
            style="cyan"
        )

    def create_overview(self) -> Panel:
        """Create overview panel."""
        if not self.bot_stats:
            return Panel("Initializing...", title="📊 Overview")

        total_solved = sum(stats.solved_puzzles for stats in self.bot_stats.values())
        total_puzzles = sum(stats.total_puzzles for stats in self.bot_stats.values())
        avg_rate = sum(stats.success_rate for stats in self.bot_stats.values()) / len(self.bot_stats)

        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        table = Table(show_header=False, box=None)
        table.add_column(style="cyan", width=12)
        table.add_column(style="white")

        table.add_row("🎯 Puzzles:", f"{total_solved}/{total_puzzles}")
        table.add_row("📈 Avg Rate:", f"{avg_rate:.1%}")
        table.add_row("⏱️ Elapsed:", f"{elapsed/60:.1f}m")
        table.add_row("🤖 Complete:", f"{self.completed_bots}/{self.total_bots}")

        return Panel(table, title="📊 [bold cyan]Overview[/bold cyan]", box=ROUNDED)

    def create_leaderboard(self) -> Panel:
        """Create leaderboard panel."""
        if not self.bot_stats:
            return Panel("No data", title="🏆 Leaderboard")

        # Sort by success rate
        sorted_bots = sorted(
            self.bot_stats.items(),
            key=lambda x: x[1].success_rate,
            reverse=True
        )

        table = Table(box=None)
        table.add_column("Rank", width=4)
        table.add_column("Bot", width=12)
        table.add_column("Score", width=8)
        table.add_column("Rate", width=7)

        medals = ["🥇", "🥈", "🥉", "🏅"]

        for i, (bot_name, stats) in enumerate(sorted_bots[:4]):
            rank = medals[i] if i < len(medals) else f"{i+1}"
            score = f"{stats.solved_puzzles}/{stats.total_puzzles}"
            rate = f"{stats.success_rate:.1%}"

            table.add_row(rank, bot_name[:10], score, rate)

        return Panel(table, title="🏆 [bold yellow]Leaderboard[/bold yellow]", box=ROUNDED)

    def create_bot_panels(self) -> List[Panel]:
        """Create individual bot status panels."""
        panels = []

        for bot_name, stats in self.bot_stats.items():
            # Progress bar
            if stats.total_puzzles > 0:
                progress = stats.current_puzzle / stats.total_puzzles
                progress_bar = "█" * int(progress * 15) + "░" * (15 - int(progress * 15))
            else:
                progress_bar = "░" * 15

            # Status color based on performance
            if not stats.is_active:
                color = "green" if stats.success_rate > 0.7 else "yellow" if stats.success_rate > 0.4 else "red"
            else:
                color = "blue"

            content = Table(show_header=False, box=None, padding=(0, 1))
            content.add_column(width=10)
            content.add_column()

            content.add_row("Progress:", f"[{color}]{progress_bar}[/{color}] {stats.current_puzzle}/{stats.total_puzzles}")
            content.add_row("Success:", f"[bold green]{stats.success_rate:.1%}[/bold green]")
            if stats.avg_time > 0:
                content.add_row("Avg Time:", f"{stats.avg_time:.1f}s")
            content.add_row("Status:", f"[italic]{stats.status}[/italic]")

            panel = Panel(
                content,
                title=f"🤖 [bold]{bot_name}[/bold]",
                box=ROUNDED,
                style=color,
                width=30
            )
            panels.append(panel)

        return panels

    def create_dashboard(self) -> Layout:
        """Create main dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="body"),
            Layout(name="footer", size=2)
        )

        # Header
        layout["header"].update(self.create_header())

        # Body
        layout["body"].split_row(
            Layout(name="sidebar", ratio=1),
            Layout(name="main", ratio=2)
        )

        # Sidebar
        layout["sidebar"].split_column(
            Layout(name="overview"),
            Layout(name="leaderboard")
        )

        layout["overview"].update(self.create_overview())
        layout["leaderboard"].update(self.create_leaderboard())

        # Main area with bot panels
        bot_panels = self.create_bot_panels()
        if bot_panels:
            if len(bot_panels) == 1:
                layout["main"].update(bot_panels[0])
            elif len(bot_panels) == 2:
                layout["main"].split_column(
                    Layout(name="top_bot"),
                    Layout(name="bottom_bot")
                )
                layout["main"]["top_bot"].update(bot_panels[0])
                layout["main"]["bottom_bot"].update(bot_panels[1])
            else:
                columns = Columns(bot_panels, equal=True, expand=True)
                layout["main"].update(columns)
        else:
            layout["main"].update(Panel("Initializing bots...", box=ROUNDED))

        # Footer
        tips = [
            "💡 Bots are running concurrently for faster results",
            "🎯 Success rate updates in real-time",
            "⚡ Progress bars show completion status",
            "🏆 Leaderboard ranks by success rate"
        ]
        tip = tips[int(time.time() / 3) % len(tips)]
        layout["footer"].update(Panel(Align.center(Text(tip, style="dim cyan")), box=ROUNDED))

        return layout

    async def run_concurrent_benchmark(self, bot_specs: List[BotSpec], config: Config,
                                     output_dir: Path, puzzle_types: List[PuzzleType],
                                     difficulty_range: tuple, puzzle_count: int) -> Dict[str, Dict[str, PuzzleStats]]:
        """Run concurrent puzzle benchmark with live dashboard."""

        self.start_time = datetime.now()
        self.total_bots = len(bot_specs)
        self.completed_bots = 0

        # Initialize bot stats
        for bot_spec in bot_specs:
            self.bot_stats[bot_spec.name] = SimpleBotStats(
                name=bot_spec.name,
                start_time=datetime.now()
            )

        # Limit concurrent execution
        semaphore = asyncio.Semaphore(min(3, len(bot_specs)))

        async def run_single_bot(bot_spec: BotSpec) -> tuple:
            """Run benchmark for a single bot."""
            async with semaphore:
                try:
                    # Update status
                    self.bot_stats[bot_spec.name].status = "🔄 Running puzzles..."

                    # Run benchmark
                    results = await run_puzzle_benchmark(bot_spec, config, output_dir)

                    # Update completion stats
                    self.bot_stats[bot_spec.name].update_completion(results)
                    self.completed_bots += 1

                    return bot_spec.name, results

                except Exception as e:
                    self.bot_stats[bot_spec.name].status = f"❌ Error: {str(e)[:20]}..."
                    self.bot_stats[bot_spec.name].is_active = False
                    return bot_spec.name, {}

        # Create tasks
        tasks = [asyncio.create_task(run_single_bot(bot_spec)) for bot_spec in bot_specs]
        results = {}

        # Run with live display
        with Live(self.create_dashboard(), refresh_per_second=2, console=self.console) as live:
            try:
                # Wait for completion
                for task in asyncio.as_completed(tasks):
                    bot_name, bot_results = await task
                    results[bot_name] = bot_results
                    live.update(self.create_dashboard())

                # Show final state
                await asyncio.sleep(1)

            except Exception as e:
                # Cancel remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                raise

        return results

    def show_final_results(self, results: Dict[str, Dict[str, PuzzleStats]]) -> None:
        """Show final results summary."""
        self.console.print("\n")
        self.console.rule("🎉 [bold green]BENCHMARK COMPLETE[/bold green] 🎉", style="green")

        if not results:
            self.console.print("[red]No results to display[/red]")
            return

        # Create final table
        final_table = Table(title="🏆 Final Results", box=ROUNDED)
        final_table.add_column("Bot", style="cyan")
        final_table.add_column("Solved/Total", style="green")
        final_table.add_column("Success Rate", style="yellow")
        final_table.add_column("Avg Time", style="blue")
        final_table.add_column("Grade", style="magenta")

        # Calculate results
        all_results = []
        for bot_name, bot_results in results.items():
            if bot_results:
                total_correct = sum(stats.correct_solutions for stats in bot_results.values())
                total_attempts = sum(stats.total_attempts for stats in bot_results.values())
                avg_time = sum(stats.average_response_time for stats in bot_results.values()) / len(bot_results)
                success_rate = total_correct / total_attempts if total_attempts > 0 else 0

                # Grade calculation
                if success_rate >= 0.8:
                    grade = "🚀 Excellent"
                elif success_rate >= 0.6:
                    grade = "✨ Good"
                elif success_rate >= 0.4:
                    grade = "📈 Fair"
                else:
                    grade = "💪 Needs Work"

                final_table.add_row(
                    bot_name,
                    f"{total_correct}/{total_attempts}",
                    f"{success_rate:.1%}",
                    f"{avg_time:.1f}s",
                    grade
                )

                all_results.append((bot_name, success_rate, total_correct, total_attempts))

        self.console.print(Align.center(final_table))

        # Winner announcement
        if all_results:
            winner = max(all_results, key=lambda x: x[1])
            self.console.print(f"\n🏆 [bold yellow]Winner: {winner[0]} with {winner[1]:.1%} success rate![/bold yellow]")

        self.console.print("\n[bold cyan]Thank you for using Chess Puzzle Benchmark! 🎯[/bold cyan]\n")


async def run_simple_modern_cli(bot_specs: List[BotSpec], config: Config, output_dir: Path,
                               puzzle_types: List[PuzzleType], difficulty_range: tuple = (1, 10),
                               puzzle_count: int = 10) -> Dict[str, Dict[str, PuzzleStats]]:
    """
    Run the simplified modern CLI.

    Args:
        bot_specs: List of bot specifications
        config: Configuration
        output_dir: Output directory
        puzzle_types: Types of puzzles
        difficulty_range: Difficulty range tuple
        puzzle_count: Number of puzzles per type

    Returns:
        Results dictionary
    """
    cli = SimplePuzzleCLI()

    try:
        # Show startup
        with cli.console.status("[bold cyan]🚀 Initializing Chess Puzzle Benchmark...", spinner="dots"):
            await asyncio.sleep(0.5)

        cli.console.clear()

        # Run benchmark
        results = await cli.run_concurrent_benchmark(
            bot_specs, config, output_dir, puzzle_types, difficulty_range, puzzle_count
        )

        # Show final results
        cli.show_final_results(results)

        return results

    except KeyboardInterrupt:
        cli.console.print("\n[yellow]⚠️ Benchmark interrupted by user[/yellow]")
        return {}
    except Exception as e:
        cli.console.print(f"\n[red]❌ Benchmark failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {}
