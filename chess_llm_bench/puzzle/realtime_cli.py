"""
Real-time Chess Puzzle CLI with Live Progress and Cost Tracking

This module provides an advanced, real-time CLI interface for chess puzzle
benchmarking with live progress updates, detailed cost breakdowns, and
beautiful visual presentation.

Features:
- Real-time progress updates during puzzle solving
- Live cost tracking and budget monitoring
- Detailed cost breakdown per model and puzzle type
- Concurrent execution with live status updates
- Beautiful animated terminal interface
- Performance metrics and efficiency scoring
- Live charts and trend analysis
- Comprehensive final results with cost analysis
"""

import asyncio
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.layout import Layout
from rich.align import Align
from rich.text import Text
from rich.columns import Columns
from rich.box import ROUNDED, DOUBLE, HEAVY
from rich.style import Style
from rich.rule import Rule
from rich.padding import Padding
from rich.console import Group

from . import PuzzleType, PuzzleStats
from .simple_realtime_runner import run_simple_realtime_benchmark
from .database import puzzle_db
from ..core.models import BotSpec, Config
from ..core.budget import get_budget_tracker


@dataclass
class LiveBotStatus:
    """Live status tracking for each bot with detailed metrics."""
    name: str
    status: str = "🚀 Initializing..."

    # Progress tracking
    current_puzzle: int = 0
    total_puzzles: int = 0
    solved_puzzles: int = 0
    progress: float = 0.0

    # Performance metrics
    success_rate: float = 0.0
    avg_time: float = 0.0
    puzzles_per_minute: float = 0.0
    efficiency_score: float = 0.0

    # Cost tracking
    total_cost: float = 0.0
    cost_per_puzzle: float = 0.0
    cost_by_type: Dict[str, float] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=dict)

    # Current puzzle info
    current_puzzle_title: str = ""
    current_puzzle_type: str = ""
    current_puzzle_difficulty: int = 0
    is_solving: bool = False

    # Performance trends
    recent_times: deque = field(default_factory=lambda: deque(maxlen=10))
    recent_success: deque = field(default_factory=lambda: deque(maxlen=10))
    success_streak: int = 0
    best_streak: int = 0

    # Timing stats
    fastest_solve: float = float('inf')
    slowest_solve: float = 0.0

    # Error tracking
    error_count: int = 0
    illegal_moves: int = 0

    # Animation state
    last_update: datetime = field(default_factory=datetime.now)
    pulse_phase: float = 0.0


class RealTimeStatsCLI:
    """Advanced real-time CLI with live progress updates and cost tracking."""

    def __init__(self):
        self.console = Console()
        self.bot_status: Dict[str, LiveBotStatus] = {}
        self.start_time = None
        self.completed_bots = 0
        self.total_bots = 0
        self.global_stats = {
            'total_puzzles': 0,
            'total_solved': 0,
            'total_cost': 0.0,
            'total_tokens': 0,
            'avg_success_rate': 0.0
        }

        # Animation and visual state
        self.update_counter = 0
        self.sparkline_chars = "▁▂▃▄▅▆▇█"

        # Color schemes
        self.colors = {
            'primary': 'cyan',
            'secondary': 'magenta',
            'success': 'green',
            'warning': 'yellow',
            'error': 'red',
            'cost': 'bright_yellow',
            'performance': 'blue'
        }

    def create_animated_header(self) -> Panel:
        """Create animated header with live global stats."""
        current_time = time.time()
        pulse = (1 + math.sin(current_time * 2)) / 2

        # Animated title with pulse effect
        title_style = "bold bright_cyan" if pulse > 0.7 else "bold cyan"

        title_text = Text()
        title_text.append("🧩 ", style="bold magenta")
        title_text.append("REAL-TIME CHESS PUZZLE BENCHMARK", style=title_style)
        title_text.append(" 🧩", style="bold magenta")

        # Live subtitle with current stats
        if self.global_stats['total_puzzles'] > 0:
            subtitle = Text(
                f"💰 ${self.global_stats['total_cost']:.4f} • "
                f"🎯 {self.global_stats['total_solved']}/{self.global_stats['total_puzzles']} • "
                f"📈 {self.global_stats['avg_success_rate']:.1%} • "
                f"🤖 {len(self.bot_status)} bots",
                style="dim bright_yellow"
            )
        else:
            subtitle = Text("Advanced Real-time Performance Analytics", style="dim bright_yellow")

        # Elapsed time
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            elapsed_text = Text(f"⏱️ {elapsed/60:.1f}m elapsed", style="dim white")
        else:
            elapsed_text = Text("Preparing for battle...", style="dim white")

        header_content = Align.center(Group(title_text, subtitle, elapsed_text))

        return Panel(
            header_content,
            box=DOUBLE,
            style=self.colors['primary'],
            padding=(1, 2)
        )

    def create_global_stats_panel(self) -> Panel:
        """Create global statistics panel with cost breakdown."""
        stats_table = Table(show_header=False, box=None, padding=(0, 1))
        stats_table.add_column(style="bold cyan", width=12)
        stats_table.add_column(style="bold white", width=12)
        stats_table.add_column(style="dim", width=8)

        # Global metrics
        stats_table.add_row("🎯 Progress:", f"{self.global_stats['total_solved']}/{self.global_stats['total_puzzles']}",
                           f"({self.global_stats['avg_success_rate']:.1%})")

        # Cost metrics
        stats_table.add_row("💰 Total Cost:", f"${self.global_stats['total_cost']:.4f}", "USD")

        if self.global_stats['total_puzzles'] > 0:
            avg_cost = self.global_stats['total_cost'] / self.global_stats['total_puzzles']
            stats_table.add_row("💵 Avg/Puzzle:", f"${avg_cost:.4f}", "USD")

        # Token usage
        stats_table.add_row("🔤 Tokens:", f"{self.global_stats['total_tokens']:,}", "total")

        # Active bots
        active_bots = sum(1 for bot in self.bot_status.values() if bot.is_solving)
        stats_table.add_row("🤖 Active:", f"{active_bots}/{len(self.bot_status)}", "bots")

        # Completion status
        stats_table.add_row("✅ Complete:", f"{self.completed_bots}/{self.total_bots}", "bots")

        return Panel(
            stats_table,
            title="📊 [bold cyan]Global Stats[/bold cyan]",
            box=ROUNDED,
            style=self.colors['performance']
        )

    def create_cost_breakdown_panel(self) -> Panel:
        """Create detailed cost breakdown panel."""
        if not self.bot_status:
            return Panel("No cost data yet", title="💰 Cost Breakdown", box=ROUNDED)

        cost_table = Table(box=None, padding=(0, 1))
        cost_table.add_column("Bot", style="cyan", width=12)
        cost_table.add_column("Total", style="bright_yellow", width=8)
        cost_table.add_column("Per Puzzle", style="yellow", width=10)
        cost_table.add_column("Efficiency", style="green", width=10)

        # Sort bots by cost
        sorted_bots = sorted(
            self.bot_status.items(),
            key=lambda x: x[1].total_cost,
            reverse=True
        )

        for bot_name, status in sorted_bots[:6]:  # Show top 6
            if status.total_cost > 0:
                efficiency = status.efficiency_score
                efficiency_display = f"{efficiency:.2f}"

                cost_table.add_row(
                    bot_name[:10],
                    f"${status.total_cost:.4f}",
                    f"${status.cost_per_puzzle:.4f}",
                    efficiency_display
                )

        return Panel(
            cost_table,
            title="💰 [bold bright_yellow]Cost Analysis[/bold bright_yellow]",
            box=ROUNDED,
            style=self.colors['cost']
        )

    def create_live_leaderboard(self) -> Panel:
        """Create live leaderboard with performance rankings."""
        if not self.bot_status:
            return Panel("Waiting for data...", title="🏆 Leaderboard", box=ROUNDED)

        # Sort by efficiency score (combines success rate and speed)
        sorted_bots = sorted(
            self.bot_status.items(),
            key=lambda x: (x[1].efficiency_score, x[1].success_rate, -x[1].avg_time),
            reverse=True
        )

        leaderboard_table = Table(box=None, padding=(0, 1))
        leaderboard_table.add_column("Rank", width=4)
        leaderboard_table.add_column("Bot", width=10)
        leaderboard_table.add_column("Score", width=7)
        leaderboard_table.add_column("Speed", width=8)
        leaderboard_table.add_column("Streak", width=6)

        rankings = ["👑", "🥇", "🥈", "🥉", "🏅", "⭐"]

        for i, (bot_name, status) in enumerate(sorted_bots[:6]):
            rank = rankings[i] if i < len(rankings) else f"#{i+1}"

            # Animated highlighting for leader
            if i == 0 and status.current_puzzle > 0:
                bot_display = f"[bold bright_yellow]{bot_name[:8]}[/bold bright_yellow]"
            else:
                bot_display = bot_name[:8]

            score = f"{status.success_rate:.0%}"
            speed = f"{status.puzzles_per_minute:.1f}/m"
            streak = f"{status.success_streak}🔥" if status.success_streak > 0 else "-"

            leaderboard_table.add_row(rank, bot_display, score, speed, streak)

        return Panel(
            leaderboard_table,
            title="🏆 [bold yellow]Live Rankings[/bold yellow]",
            box=ROUNDED,
            style=self.colors['warning']
        )

    def create_bot_status_panel(self, bot_name: str, status: LiveBotStatus) -> Panel:
        """Create detailed bot status panel with real-time metrics."""
        # Update animation state
        current_time = time.time()
        status.pulse_phase = (current_time * 3) % (2 * math.pi)

        # Animated progress bar
        if status.total_puzzles > 0:
            progress = status.current_puzzle / status.total_puzzles
            bar_width = 16
            filled = int(progress * bar_width)

            # Add pulse effect for active solving
            if status.is_solving:
                pulse_char = "▓" if int(current_time * 4) % 2 else "▒"
                progress_bar = "█" * filled + pulse_char + "░" * (bar_width - filled - 1)
            else:
                progress_bar = "█" * filled + "░" * (bar_width - filled)
        else:
            progress_bar = "░" * 16

        # Status color based on performance
        if not status.is_solving and status.current_puzzle == status.total_puzzles:
            panel_style = "bright_green" if status.success_rate > 0.7 else "green"
        elif status.is_solving:
            panel_style = "bright_blue"
        elif status.error_count > 0:
            panel_style = "red"
        else:
            panel_style = "blue"

        # Main stats table
        main_table = Table(show_header=False, box=None, padding=(0, 1))
        main_table.add_column(width=11)
        main_table.add_column()

        # Progress with animated bar
        main_table.add_row("Progress:", f"[{panel_style}]{progress_bar}[/{panel_style}]")
        main_table.add_row("", f"{status.current_puzzle}/{status.total_puzzles}")

        # Performance metrics
        main_table.add_row("Success:", f"[bold green]{status.success_rate:.1%}[/bold green]")
        main_table.add_row("Speed:", f"[bold blue]{status.puzzles_per_minute:.1f}/min[/bold blue]")
        main_table.add_row("Avg Time:", f"{status.avg_time:.1f}s")

        # Cost information
        main_table.add_row("Total Cost:", f"[bold bright_yellow]${status.total_cost:.4f}[/bold bright_yellow]")
        if status.current_puzzle > 0:
            main_table.add_row("Per Puzzle:", f"[yellow]${status.cost_per_puzzle:.4f}[/yellow]")

        # Token usage
        if status.token_usage.get('total_tokens', 0) > 0:
            main_table.add_row("Tokens:", f"{status.token_usage['total_tokens']:,}")

        # Performance indicators
        if status.success_streak > 0:
            streak_display = f"🔥 {status.success_streak}"
            if status.success_streak >= 5:
                streak_display += " [bold red]HOT![/bold red]"
            main_table.add_row("Streak:", streak_display)

        # Efficiency score
        if status.efficiency_score > 0:
            main_table.add_row("Efficiency:", f"[bold magenta]{status.efficiency_score:.2f}[/bold magenta]")

        # Current puzzle info
        if status.is_solving and status.current_puzzle_title:
            current_info = f"🧩 {status.current_puzzle_title[:20]}..."
            main_table.add_row("Solving:", f"[italic]{current_info}[/italic]")

        # Recent performance sparkline
        if len(status.recent_success) > 1:
            sparkline = self._create_sparkline([float(x) for x in status.recent_success])
            main_table.add_row("Trend:", f"[dim]{sparkline}[/dim]")

        # Status with animation
        status_text = status.status
        if status.is_solving:
            spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            spinner = spinner_chars[int(current_time * 8) % len(spinner_chars)]
            status_text = f"{spinner} {status_text}"

        main_table.add_row("Status:", f"[italic]{status_text}[/italic]")

        return Panel(
            main_table,
            title=f"🤖 [bold]{bot_name}[/bold]",
            box=ROUNDED,
            style=panel_style,
            width=32
        )

    def _create_sparkline(self, values: List[float], width: int = 12) -> str:
        """Create a sparkline from values."""
        if not values or len(values) < 2:
            return "▄" * width

        # Normalize values to 0-7 range
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return "▄" * width

        # Sample values to fit width
        if len(values) > width:
            step = len(values) / width
            sampled = [values[int(i * step)] for i in range(width)]
        else:
            sampled = values + [values[-1]] * (width - len(values))

        # Convert to sparkline
        sparkline = ""
        for val in sampled:
            normalized = (val - min_val) / (max_val - min_val)
            char_index = min(7, int(normalized * 8))
            sparkline += self.sparkline_chars[char_index]

        return sparkline

    def create_dashboard(self) -> Layout:
        """Create the main real-time dashboard layout."""
        self.update_counter += 1

        layout = Layout()

        # Main sections
        layout.split_column(
            Layout(name="header", size=6),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )

        # Header
        layout["header"].update(self.create_animated_header())

        # Body layout
        layout["body"].split_row(
            Layout(name="sidebar", ratio=1),
            Layout(name="main", ratio=2)
        )

        # Sidebar with stats and leaderboard
        layout["sidebar"].split_column(
            Layout(name="global_stats"),
            Layout(name="cost_breakdown"),
            Layout(name="leaderboard")
        )

        layout["global_stats"].update(self.create_global_stats_panel())
        layout["cost_breakdown"].update(self.create_cost_breakdown_panel())
        layout["leaderboard"].update(self.create_live_leaderboard())

        # Main area with bot panels
        bot_panels = [
            self.create_bot_status_panel(name, status)
            for name, status in self.bot_status.items()
        ]

        if bot_panels:
            if len(bot_panels) == 1:
                layout["main"].update(bot_panels[0])
            elif len(bot_panels) == 2:
                layout["main"].split_column(
                    Layout(name="bot1"),
                    Layout(name="bot2")
                )
                layout["main"]["bot1"].update(bot_panels[0])
                layout["main"]["bot2"].update(bot_panels[1])
            else:
                # Use columns for multiple bots
                columns = Columns(bot_panels, equal=True, expand=True)
                layout["main"].update(columns)
        else:
            startup_panel = Panel(
                Align.center(Text("🚀 Initializing Real-time Analytics...", style="bold cyan")),
                box=ROUNDED
            )
            layout["main"].update(startup_panel)

        # Animated footer with tips
        tips = [
            "💰 Cost tracking shows real-time spending per model and puzzle type",
            "⚡ Efficiency score combines success rate with solving speed",
            "🔥 Streaks track consecutive correct solutions",
            "📈 Sparklines show recent performance trends",
            "🎯 Progress updates in real-time as puzzles are solved",
            "🏆 Leaderboard ranks by overall efficiency score",
            "🤖 Watch bots compete simultaneously with live updates",
            "💡 Budget limits help control API spending during testing"
        ]

        tip_index = (self.update_counter // 20) % len(tips)
        current_tip = tips[tip_index]

        # Add glow effect
        import math
        glow = (1 + math.sin(time.time() * 2)) / 2
        tip_style = "bright_cyan" if glow > 0.7 else "cyan"

        footer_content = Align.center(Text(current_tip, style=tip_style))
        layout["footer"].update(Panel(footer_content, box=ROUNDED, style="dim"))

        return layout

    async def update_bot_status(self, bot_name: str, progress_data: Dict[str, Any]):
        """Update bot status from progress data."""
        if bot_name not in self.bot_status:
            self.bot_status[bot_name] = LiveBotStatus(name=bot_name)

        status = self.bot_status[bot_name]
        event = progress_data.get('event', '')

        if event == 'session_started':
            status.total_puzzles = progress_data.get('total_puzzles', 0)
            status.status = f"🚀 Starting {progress_data.get('puzzle_set', 'puzzles')}..."

        elif event == 'puzzle_started':
            puzzle_info = progress_data.get('puzzle', {})
            status.current_puzzle = progress_data.get('puzzle_index', 0)
            status.current_puzzle_title = puzzle_info.get('title', '')
            status.current_puzzle_type = puzzle_info.get('type', '')
            status.current_puzzle_difficulty = puzzle_info.get('difficulty', 0)
            status.is_solving = True

            emoji = {"tactic": "⚔️", "endgame": "♔", "blunder": "⚠️", "gamelet": "📖"}.get(
                status.current_puzzle_type, "🧩"
            )
            status.status = f"{emoji} Solving: {status.current_puzzle_title[:25]}..."

        elif event == 'puzzle_completed':
            attempt_info = progress_data.get('attempt', {})
            stats_data = progress_data.get('stats', {})

            # Update from comprehensive stats
            status.current_puzzle = stats_data.get('current_puzzle', status.current_puzzle)
            status.solved_puzzles = stats_data.get('solved_puzzles', status.solved_puzzles)
            status.success_rate = stats_data.get('success_rate', status.success_rate)
            status.avg_time = stats_data.get('avg_time', status.avg_time)
            status.puzzles_per_minute = stats_data.get('puzzles_per_minute', status.puzzles_per_minute)
            status.total_cost = stats_data.get('total_cost', status.total_cost)
            status.cost_per_puzzle = stats_data.get('cost_per_puzzle', status.cost_per_puzzle)
            status.efficiency_score = stats_data.get('efficiency_score', status.efficiency_score)
            status.success_streak = stats_data.get('success_streak', status.success_streak)
            status.best_streak = stats_data.get('best_streak', status.best_streak)
            status.error_count = stats_data.get('error_count', status.error_count)
            status.illegal_moves = stats_data.get('illegal_moves', status.illegal_moves)
            status.token_usage = stats_data.get('token_usage', {})
            status.cost_by_type = stats_data.get('cost_by_type', {})
            status.is_solving = False

            # Update progress
            status.progress = status.current_puzzle / status.total_puzzles if status.total_puzzles > 0 else 0.0

            # Add to recent trends
            status.recent_times.append(attempt_info.get('response_time', 0))
            status.recent_success.append(1.0 if attempt_info.get('is_correct', False) else 0.0)

            # Update status
            result_emoji = "✅" if attempt_info.get('is_correct', False) else "❌"
            cost = attempt_info.get('cost', 0)
            time_taken = attempt_info.get('response_time', 0)

            type_emoji = {"tactic": "⚔️", "endgame": "♔", "blunder": "⚠️", "gamelet": "📖"}.get(
                status.current_puzzle_type, "🧩"
            )

            status.status = f"{type_emoji} {result_emoji} #{status.current_puzzle} ({time_taken:.1f}s, ${cost:.4f})"

        elif event == 'session_completed':
            status.is_solving = False
            status.status = "✨ Session Complete!"
            self.completed_bots += 1

        elif event == 'puzzle_error':
            status.error_count += 1
            error = progress_data.get('error', '')
            status.status = f"❌ Error: {error[:25]}..."
            status.is_solving = False

        # Update global stats
        self._update_global_stats()
        status.last_update = datetime.now()

    def _update_global_stats(self):
        """Update global statistics from all bots."""
        if not self.bot_status:
            return

        self.global_stats['total_puzzles'] = sum(s.total_puzzles for s in self.bot_status.values())
        self.global_stats['total_solved'] = sum(s.solved_puzzles for s in self.bot_status.values())
        self.global_stats['total_cost'] = sum(s.total_cost for s in self.bot_status.values())
        self.global_stats['total_tokens'] = sum(
            s.token_usage.get('total_tokens', 0) for s in self.bot_status.values()
        )

        # Calculate average success rate
        active_bots = [s for s in self.bot_status.values() if s.current_puzzle > 0]
        if active_bots:
            self.global_stats['avg_success_rate'] = sum(s.success_rate for s in active_bots) / len(active_bots)

    async def run_realtime_benchmark(self, bot_specs: List[BotSpec], config: Config,
                                   output_dir: Path, puzzle_types: List[PuzzleType],
                                   difficulty_range: Tuple[int, int],
                                   puzzle_count: int, timeout: float) -> Dict[str, Any]:
        """Run the real-time benchmark with live dashboard."""

        self.start_time = datetime.now()
        self.total_bots = len(bot_specs)
        self.completed_bots = 0

        # Initialize bot status
        for bot_spec in bot_specs:
            self.bot_status[bot_spec.name] = LiveBotStatus(name=bot_spec.name)

        # Create progress callback
        async def progress_callback(progress_data: Dict[str, Any]):
            bot_name = progress_data.get('stats', {}).get('bot_name') or progress_data.get('bot_name')
            if bot_name:
                await self.update_bot_status(bot_name, progress_data)

        # Run with live display
        results = {}

        with Live(self.create_dashboard(), refresh_per_second=4, console=self.console) as live:
            try:
                # Run real-time benchmark
                results = await run_simple_realtime_benchmark(
                    bot_specs=bot_specs,
                    config=config,
                    output_dir=output_dir,
                    puzzle_types=puzzle_types,
                    difficulty_range=difficulty_range,
                    puzzle_count=puzzle_count,
                    progress_callback=progress_callback
                )

                # Show final state for a moment
                await asyncio.sleep(2)

            except Exception as e:
                self.console.print(f"\n[red]Error during benchmark: {e}[/red]")
                raise

        return results

    def show_final_results(self, results: Dict[str, Any]):
        """Show comprehensive final results with cost analysis."""
        self.console.clear()
        self.console.print("")
        self.console.rule("🎉 [bold bright_green]REAL-TIME BENCHMARK COMPLETE[/bold bright_green] 🎉",
                         style="bright_green")
        self.console.print("")

        if not results:
            self.console.print(Panel("No results available", title="❌ Error", style="red"))
            return

        # Create comprehensive results table
        results_table = Table(box=HEAVY, title="🏆 Final Championship Results", title_style="bold bright_yellow")
        results_table.add_column("Rank", style="bold yellow", width=6)
        results_table.add_column("Champion", style="bold cyan", width=15)
        results_table.add_column("Success Rate", style="bold green", width=12)
        results_table.add_column("Total Cost", style="bold bright_yellow", width=10)
        results_table.add_column("Cost/Puzzle", style="bold yellow", width=12)
        results_table.add_column("Efficiency", style="bold magenta", width=10)
        results_table.add_column("Grade", style="bold bright_cyan", width=12)

        # Process and rank results
        processed_results = []
        for bot_name, bot_data in results.items():
            if 'final_stats' in bot_data and 'summary' in bot_data['final_stats']:
                stats = bot_data['final_stats']['summary']
                costs = bot_data['final_stats']['costs']

                processed_results.append({
                    'name': bot_name,
                    'success_rate': stats['success_rate'],
                    'total_cost': costs['total_cost'],
                    'cost_per_puzzle': costs['cost_per_puzzle'],
                    'efficiency': stats['efficiency_score']
                })

        # Sort by efficiency score
        processed_results.sort(key=lambda x: x['efficiency'], reverse=True)

        # Add to table
        rankings = ["👑 CHAMPION", "🥇 MASTER", "🥈 EXPERT", "🥉 SKILLED", "🏅 NOVICE", "⭐ BEGINNER"]

        for i, result in enumerate(processed_results):
            rank_display = rankings[i] if i < len(rankings) else f"#{i+1}"

            # Determine grade based on efficiency
            if result['efficiency'] >= 1.5:
                grade = "🚀 LEGENDARY"
            elif result['efficiency'] >= 1.2:
                grade = "⭐ EXCELLENT"
            elif result['efficiency'] >= 0.8:
                grade = "✨ GOOD"
            elif result['efficiency'] >= 0.5:
                grade = "📈 FAIR"
            else:
                grade = "💪 TRAINING"

            results_table.add_row(
                rank_display,
                result['name'],
                f"{result['success_rate']:.1%}",
                f"${result['total_cost']:.4f}",
                f"${result['cost_per_puzzle']:.4f}",
                f"{result['efficiency']:.2f}",
                grade
            )

        self.console.print(Align.center(results_table))

        # Cost analysis summary
        total_cost = sum(r['total_cost'] for r in processed_results)
        total_puzzles = sum(self.bot_status[r['name']].total_puzzles for r in processed_results)

        if processed_results:
            self.console.print("")
            cost_summary = Panel(
                Group(
                    Text("💰 Cost Analysis Summary", style="bold bright_yellow"),
                    Text(""),
                    Text(f"Total Spent: ${total_cost:.4f} USD", style="green"),
                    Text(f"Total Puzzles: {total_puzzles:,}", style="blue"),
                    Text(f"Average Cost per Puzzle: ${total_cost/total_puzzles:.4f}" if total_puzzles > 0 else "No puzzles completed", style="cyan"),
                    Text(f"Most Expensive: {max(processed_results, key=lambda x: x['total_cost'])['name']}", style="yellow"),
                    Text(f"Most Efficient: {max(processed_results, key=lambda x: x['efficiency'])['name']}", style="magenta"),
                ),
                title="💸 Financial Report",
                box=ROUNDED,
                style="bright_yellow"
            )
            self.console.print(Align.center(cost_summary))

        # Champion announcement
        if processed_results:
            winner = processed_results[0]
            self.console.print("")

            champion_panel = Panel(
                Align.center(Group(
                    Text(f"🏆 CHAMPION: {winner['name']} 🏆", style="bold bright_yellow"),
                    Text(""),
                    Text(f"Success Rate: {winner['success_rate']:.1%}", style="green"),
                    Text(f"Total Cost: ${winner['total_cost']:.4f}", style="bright_yellow"),
                    Text(f"Efficiency Score: {winner['efficiency']:.2f}", style="magenta"),
                    Text(""),
                    Text("🎉 REAL-TIME PUZZLE MASTER! 🎉", style="bold bright_gold")
                )),
                title="👑 VICTORY",
                box=DOUBLE,
                style="bright_yellow"
            )
            self.console.print(Align.center(champion_panel))

        # Final message
        self.console.print("")
        self.console.print(Align.center(
            Text("Thank you for the real-time chess puzzle battle! 🎯♔⚔️💰", style="bold bright_cyan")
        ))
        self.console.print("")


async def run_realtime_puzzle_cli(bot_specs: List[BotSpec], config: Config, output_dir: Path,
                                 puzzle_types: List[PuzzleType], difficulty_range: Tuple[int, int] = (1, 10),
                                 puzzle_count: int = 10, timeout: float = 30.0) -> Dict[str, Any]:
    """
    Run the real-time puzzle CLI with live updates and cost tracking.

    Args:
        bot_specs: List of bot specifications
        config: Configuration
        output_dir: Output directory
        puzzle_types: Types of puzzles
        difficulty_range: Difficulty range tuple
        puzzle_count: Number of puzzles per type
        timeout: Timeout per puzzle

    Returns:
        Results dictionary with comprehensive data
    """
    cli = RealTimeStatsCLI()

    try:
        # Epic startup sequence
        with cli.console.status("[bold bright_cyan]🚀 Launching Real-time Chess Analytics...", spinner="dots"):
            await asyncio.sleep(1)

        cli.console.clear()

        # Show startup banner
        cli.console.print("")
        cli.console.rule("⚡ [bold bright_cyan]REAL-TIME CHESS PUZZLE ANALYTICS[/bold bright_cyan] ⚡",
                        style="bright_cyan")
        cli.console.print("")

        startup_text = Text.assemble(
            ("🧩 Real-time puzzle solving with live cost tracking! 🧩\n", "bold cyan"),
            ("💰 Watch spending • 📊 Live metrics • ⚡ Instant updates\n", "dim"),
            ("\nPrepare for concurrent real-time action!", "yellow")
        )

        cli.console.print(Align.center(Panel(startup_text, box=ROUNDED, style="cyan")))
        cli.console.print("")

        # Brief pause for effect
        await asyncio.sleep(1)

        # Run the real-time benchmark
        results = await cli.run_realtime_benchmark(
            bot_specs, config, output_dir, puzzle_types, difficulty_range, puzzle_count, timeout
        )

        # Show comprehensive final results
        cli.show_final_results(results)

        return results

    except KeyboardInterrupt:
        cli.console.print("\n[yellow]⚠️ Real-time benchmark interrupted by user[/yellow]")
        return {}
    except Exception as e:
        cli.console.print(f"\n[red]❌ Real-time benchmark failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {}
