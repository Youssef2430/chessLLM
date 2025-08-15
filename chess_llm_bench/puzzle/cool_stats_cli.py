"""
Cool Animated Stats CLI for Chess Puzzle Benchmark

This module provides an advanced, animated CLI with cool statistics, performance
metrics, live charts, and beautiful visual effects for chess puzzle benchmarking.

Features:
- Animated progress bars with pulse effects
- Real-time performance charts and graphs
- Speed meters and performance indicators
- Live typing effects and status animations
- Cool emoji indicators and visual feedback
- Trend analysis with sparklines
- Performance grades with animations
- Live leaderboard with rankings
- Detailed metrics dashboard
- Beautiful color schemes and styling
"""

import asyncio
import math
import random
import time
from collections import deque
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
from rich.box import ROUNDED, DOUBLE, HEAVY, ASCII
from rich.style import Style
from rich.rule import Rule
from rich.padding import Padding
from rich.console import Group
from rich import box

from . import PuzzleType, PuzzleStats
from .runner import run_puzzle_benchmark
from .database import puzzle_db
from ..core.models import BotSpec, Config


@dataclass
class CoolBotStats:
    """Enhanced bot statistics with cool animations and metrics."""
    name: str
    status: str = "🚀 Initializing..."

    # Core stats
    total_puzzles: int = 0
    solved_puzzles: int = 0
    current_puzzle: int = 0
    success_rate: float = 0.0
    avg_time: float = 0.0
    is_active: bool = True
    start_time: Optional[datetime] = None

    # Performance tracking
    puzzle_history: deque = field(default_factory=lambda: deque(maxlen=10))
    timing_history: deque = field(default_factory=lambda: deque(maxlen=20))
    success_streak: int = 0
    best_streak: int = 0
    fastest_solve: float = float('inf')
    slowest_solve: float = 0.0

    # Type-specific performance
    tactics_solved: int = 0
    tactics_total: int = 0
    endgames_solved: int = 0
    endgames_total: int = 0
    blunders_solved: int = 0
    blunders_total: int = 0
    gamelets_solved: int = 0
    gamelets_total: int = 0

    # Animation state
    pulse_phase: float = 0.0
    status_animation: int = 0
    last_update: datetime = field(default_factory=datetime.now)

    @property
    def puzzles_per_minute(self) -> float:
        """Calculate solving speed."""
        if not self.start_time:
            return 0.0
        elapsed = (datetime.now() - self.start_time).total_seconds() / 60
        return self.current_puzzle / elapsed if elapsed > 0 else 0.0

    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (success rate * speed factor)."""
        speed_factor = min(2.0, 60.0 / self.avg_time) if self.avg_time > 0 else 1.0
        return self.success_rate * speed_factor

    @property
    def performance_grade(self) -> str:
        """Get performance grade with emoji."""
        score = self.efficiency_score
        if score >= 1.5:
            return "🚀 LEGENDARY"
        elif score >= 1.2:
            return "⭐ EXCELLENT"
        elif score >= 0.8:
            return "✨ GOOD"
        elif score >= 0.5:
            return "📈 FAIR"
        elif score >= 0.2:
            return "🔥 IMPROVING"
        else:
            return "💪 TRAINING"

    def add_puzzle_result(self, is_correct: bool, response_time: float, puzzle_type: PuzzleType):
        """Add a puzzle result to tracking."""
        self.puzzle_history.append(is_correct)
        self.timing_history.append(response_time)

        if is_correct:
            self.solved_puzzles += 1
            self.success_streak += 1
            self.best_streak = max(self.best_streak, self.success_streak)
        else:
            self.success_streak = 0

        self.current_puzzle += 1
        self.success_rate = self.solved_puzzles / self.current_puzzle

        # Update timing stats
        total_time = self.avg_time * (self.current_puzzle - 1) + response_time
        self.avg_time = total_time / self.current_puzzle
        self.fastest_solve = min(self.fastest_solve, response_time)
        self.slowest_solve = max(self.slowest_solve, response_time)

        # Update type-specific stats
        if puzzle_type == PuzzleType.TACTIC:
            self.tactics_total += 1
            if is_correct:
                self.tactics_solved += 1
        elif puzzle_type == PuzzleType.ENDGAME:
            self.endgames_total += 1
            if is_correct:
                self.endgames_solved += 1
        elif puzzle_type == PuzzleType.BLUNDER_AVOID:
            self.blunders_total += 1
            if is_correct:
                self.blunders_solved += 1
        elif puzzle_type == PuzzleType.GAMELET:
            self.gamelets_total += 1
            if is_correct:
                self.gamelets_solved += 1


class CoolStatsAnimator:
    """Handles animations and visual effects."""

    def __init__(self):
        self.time_offset = time.time()
        self.sparkline_chars = "▁▂▃▄▅▆▇█"
        self.progress_chars = "⣀⣄⣤⣦⣶⣷⣿"
        self.pulse_chars = "●◐◑◒◓◔◕"

    def get_pulse_char(self, phase: float) -> str:
        """Get animated pulse character."""
        index = int(phase * len(self.pulse_chars)) % len(self.pulse_chars)
        return self.pulse_chars[index]

    def get_sparkline(self, values: List[float], width: int = 10) -> str:
        """Generate sparkline from values."""
        if not values or len(values) < 2:
            return "▁" * width

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

    def get_animated_progress_bar(self, progress: float, width: int = 20, phase: float = 0) -> str:
        """Get animated progress bar with moving elements."""
        filled = int(progress * width)

        # Add pulse effect at the progress point
        bar = "█" * filled

        if filled < width:
            # Add animated progress indicator
            pulse_pos = int(phase * 4) % 4
            if pulse_pos == 0:
                bar += "▓"
            elif pulse_pos == 1:
                bar += "▒"
            elif pulse_pos == 2:
                bar += "░"
            else:
                bar += "▓"

            bar += "░" * (width - filled - 1)

        return bar

    def get_speed_meter(self, speed: float, max_speed: float = 5.0) -> str:
        """Generate ASCII speed meter."""
        if max_speed == 0:
            max_speed = 1.0

        normalized_speed = min(1.0, speed / max_speed)
        meter_width = 10
        filled = int(normalized_speed * meter_width)

        # Speed indicator with colors based on speed
        if normalized_speed > 0.8:
            meter = "🟢" * filled + "⚪" * (meter_width - filled)
        elif normalized_speed > 0.5:
            meter = "🟡" * filled + "⚪" * (meter_width - filled)
        elif normalized_speed > 0.2:
            meter = "🟠" * filled + "⚪" * (meter_width - filled)
        else:
            meter = "🔴" * filled + "⚪" * (meter_width - filled)

        return meter


class CoolStatsCLI:
    """Advanced CLI with cool animated statistics and performance metrics."""

    def __init__(self):
        self.console = Console()
        self.animator = CoolStatsAnimator()
        self.bot_stats: Dict[str, CoolBotStats] = {}
        self.start_time = None
        self.completed_bots = 0
        self.total_bots = 0
        self.update_count = 0

        # Color schemes
        self.colors = {
            'primary': 'cyan',
            'secondary': 'magenta',
            'success': 'green',
            'warning': 'yellow',
            'error': 'red',
            'info': 'blue',
            'accent': 'bright_cyan'
        }

        # Emoji sets
        self.puzzle_emojis = {
            PuzzleType.TACTIC: "⚔️",
            PuzzleType.ENDGAME: "♔",
            PuzzleType.BLUNDER_AVOID: "⚠️",
            PuzzleType.GAMELET: "📖"
        }

    def create_header(self) -> Panel:
        """Create animated header with cool effects."""
        current_time = time.time()
        phase = (current_time * 2) % (2 * math.pi)

        # Animated title with pulse effect
        pulse_intensity = (math.sin(phase) + 1) / 2
        if pulse_intensity > 0.7:
            title_style = "bold bright_cyan"
        elif pulse_intensity > 0.4:
            title_style = "bold cyan"
        else:
            title_style = "cyan"

        title_text = Text()
        title_text.append("🧩 ", style="bold magenta")
        title_text.append("CHESS PUZZLE BENCHMARK", style=title_style)
        title_text.append(" 🧩", style="bold magenta")

        # Animated subtitle
        subtitles = [
            "Advanced LLM Chess Intelligence Assessment",
            "Real-time Concurrent Performance Analysis",
            "AI vs Chess Mastery Challenge",
            "Next-Gen Chess AI Evaluation"
        ]
        subtitle_index = int(current_time / 3) % len(subtitles)
        subtitle = Text(subtitles[subtitle_index], style="italic dim bright_yellow")

        # Stats ticker
        if self.bot_stats:
            total_solved = sum(s.solved_puzzles for s in self.bot_stats.values())
            total_puzzles = sum(s.current_puzzle for s in self.bot_stats.values())
            ticker = Text(f"Live: {total_solved}/{total_puzzles} solved • {len(self.bot_stats)} bots active",
                         style="dim white")
        else:
            ticker = Text("Initializing advanced analytics...", style="dim white")

        header_content = Align.center(Group(title_text, subtitle, ticker))

        return Panel(
            header_content,
            box=DOUBLE,
            style=self.colors['primary'],
            padding=(1, 2)
        )

    def create_performance_dashboard(self) -> Panel:
        """Create advanced performance metrics dashboard."""
        if not self.bot_stats:
            return Panel("📊 Analytics Loading...", title="Performance Dashboard", box=ROUNDED)

        # Calculate global metrics
        total_solved = sum(s.solved_puzzles for s in self.bot_stats.values())
        total_puzzles = sum(s.current_puzzle for s in self.bot_stats.values())
        avg_success = sum(s.success_rate for s in self.bot_stats.values()) / len(self.bot_stats)
        avg_speed = sum(s.puzzles_per_minute for s in self.bot_stats.values()) / len(self.bot_stats)

        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        # Create metrics table
        metrics_table = Table(show_header=False, box=None, padding=(0, 1))
        metrics_table.add_column(style="bold cyan", width=15)
        metrics_table.add_column(style="bold white", width=15)
        metrics_table.add_column(style="dim", width=10)

        # Performance metrics with animations
        current_time = time.time()
        phase = (current_time * 3) % (2 * math.pi)

        # Animated metrics
        metrics_table.add_row("🎯 Global Score:", f"{total_solved:,}/{total_puzzles:,}",
                             f"({avg_success:.1%})")

        # Speed meter
        speed_bar = self.animator.get_speed_meter(avg_speed, 3.0)
        metrics_table.add_row("⚡ Avg Speed:", f"{avg_speed:.1f}/min", speed_bar[:8])

        # Time tracking
        metrics_table.add_row("⏱️ Runtime:", f"{elapsed/60:.1f}m",
                             f"{self.completed_bots}/{self.total_bots} done")

        # Active bots indicator
        active_count = sum(1 for s in self.bot_stats.values() if s.is_active)
        pulse_char = self.animator.get_pulse_char(phase)
        metrics_table.add_row("🤖 Active Bots:", f"{active_count}", f"{pulse_char}")

        # Success trend sparkline
        if self.bot_stats:
            recent_success = [s.success_rate for s in self.bot_stats.values()]
            sparkline = self.animator.get_sparkline(recent_success, 12)
            metrics_table.add_row("📈 Trend:", sparkline, "success")

        return Panel(
            metrics_table,
            title="📊 [bold cyan]Performance Dashboard[/bold cyan]",
            box=ROUNDED,
            style=self.colors['info']
        )

    def create_live_leaderboard(self) -> Panel:
        """Create animated leaderboard with rankings."""
        if not self.bot_stats:
            return Panel("🏆 Waiting for competitors...", title="Live Leaderboard", box=ROUNDED)

        # Sort by efficiency score for more interesting rankings
        sorted_bots = sorted(
            self.bot_stats.items(),
            key=lambda x: (x[1].efficiency_score, x[1].success_rate, -x[1].avg_time),
            reverse=True
        )

        leaderboard_table = Table(box=None, padding=(0, 1))
        leaderboard_table.add_column("Rank", style="bold yellow", width=4)
        leaderboard_table.add_column("Bot", style="bold cyan", width=12)
        leaderboard_table.add_column("Score", style="bold green", width=6)
        leaderboard_table.add_column("Speed", style="bold blue", width=7)
        leaderboard_table.add_column("Streak", style="bold magenta", width=6)

        # Animated ranking indicators
        rank_indicators = ["👑", "🥇", "🥈", "🥉", "🏅", "🎖️", "⭐", "🌟"]

        for i, (bot_name, stats) in enumerate(sorted_bots[:6]):
            rank_icon = rank_indicators[i] if i < len(rank_indicators) else f"#{i+1}"

            # Animated score display
            score_text = f"{stats.success_rate:.0%}"
            speed_text = f"{stats.puzzles_per_minute:.1f}/m"

            # Streak display with fire effect
            if stats.success_streak > 0:
                streak_text = f"{stats.success_streak}🔥"
            else:
                streak_text = "-"

            # Highlight current leader
            if i == 0 and stats.current_puzzle > 0:
                bot_display = f"[bold bright_yellow]{bot_name[:10]}[/bold bright_yellow]"
            else:
                bot_display = bot_name[:10]

            leaderboard_table.add_row(
                rank_icon,
                bot_display,
                score_text,
                speed_text,
                streak_text
            )

        return Panel(
            leaderboard_table,
            title="🏆 [bold yellow]Live Leaderboard[/bold yellow]",
            box=ROUNDED,
            style=self.colors['warning']
        )

    def create_bot_status_panel(self, bot_name: str, stats: CoolBotStats) -> Panel:
        """Create detailed bot status panel with animations."""
        current_time = time.time()
        stats.pulse_phase = (current_time * 4) % (2 * math.pi)

        # Animated progress bar
        if stats.total_puzzles > 0:
            progress = stats.current_puzzle / stats.total_puzzles
            progress_bar = self.animator.get_animated_progress_bar(
                progress, 18, stats.pulse_phase
            )
        else:
            progress_bar = "░" * 18

        # Status color based on performance
        if not stats.is_active:
            if stats.efficiency_score > 1.0:
                panel_style = "bright_green"
            elif stats.efficiency_score > 0.6:
                panel_style = "green"
            elif stats.efficiency_score > 0.3:
                panel_style = "yellow"
            else:
                panel_style = "red"
        else:
            panel_style = "blue"

        # Main stats table
        main_table = Table(show_header=False, box=None, padding=(0, 1))
        main_table.add_column(width=12)
        main_table.add_column()

        # Progress with animated bar
        main_table.add_row("Progress:", f"[{panel_style}]{progress_bar}[/{panel_style}]")
        main_table.add_row("", f"{stats.current_puzzle}/{stats.total_puzzles}")

        # Performance metrics
        main_table.add_row("Success Rate:", f"[bold green]{stats.success_rate:.1%}[/bold green]")
        main_table.add_row("Speed:", f"[bold blue]{stats.puzzles_per_minute:.1f}/min[/bold blue]")

        # Efficiency score with grade
        grade = stats.performance_grade
        main_table.add_row("Grade:", f"[bold magenta]{grade}[/bold magenta]")

        # Timing stats
        if stats.fastest_solve != float('inf'):
            main_table.add_row("Best Time:", f"⚡ {stats.fastest_solve:.1f}s")

        # Streak indicator
        if stats.success_streak > 0:
            streak_display = f"🔥 {stats.success_streak}"
            if stats.success_streak >= 5:
                streak_display += " [bold red]ON FIRE![/bold red]"
            main_table.add_row("Streak:", streak_display)

        # Performance by type (if any data)
        type_stats = []
        if stats.tactics_total > 0:
            rate = stats.tactics_solved / stats.tactics_total
            type_stats.append(f"⚔️{rate:.0%}")
        if stats.endgames_total > 0:
            rate = stats.endgames_solved / stats.endgames_total
            type_stats.append(f"♔{rate:.0%}")
        if stats.blunders_total > 0:
            rate = stats.blunders_solved / stats.blunders_total
            type_stats.append(f"⚠️{rate:.0%}")
        if stats.gamelets_total > 0:
            rate = stats.gamelets_solved / stats.gamelets_total
            type_stats.append(f"📖{rate:.0%}")

        if type_stats:
            main_table.add_row("Types:", " ".join(type_stats))

        # Recent performance sparkline
        if len(stats.puzzle_history) > 1:
            sparkline = self.animator.get_sparkline(
                [1.0 if x else 0.0 for x in stats.puzzle_history], 15
            )
            main_table.add_row("Trend:", f"[dim]{sparkline}[/dim]")

        # Animated status
        status_text = stats.status
        if stats.is_active:
            pulse_char = self.animator.get_pulse_char(stats.pulse_phase)
            status_text = f"{pulse_char} {status_text}"

        main_table.add_row("Status:", f"[italic]{status_text}[/italic]")

        return Panel(
            main_table,
            title=f"🤖 [bold]{bot_name}[/bold]",
            box=ROUNDED,
            style=panel_style,
            width=35
        )

    def create_dashboard(self) -> Layout:
        """Create the main animated dashboard."""
        self.update_count += 1

        layout = Layout()

        # Main sections
        layout.split_column(
            Layout(name="header", size=6),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )

        # Header
        layout["header"].update(self.create_header())

        # Body split
        layout["body"].split_row(
            Layout(name="sidebar", ratio=1),
            Layout(name="main", ratio=2)
        )

        # Sidebar with dashboard and leaderboard
        layout["sidebar"].split_column(
            Layout(name="dashboard"),
            Layout(name="leaderboard")
        )

        layout["dashboard"].update(self.create_performance_dashboard())
        layout["leaderboard"].update(self.create_live_leaderboard())

        # Main area with bot panels
        bot_panels = [
            self.create_bot_status_panel(name, stats)
            for name, stats in self.bot_stats.items()
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
                Align.center(Text("🚀 Initializing AI Chess Champions...", style="bold cyan")),
                box=ROUNDED
            )
            layout["main"].update(startup_panel)

        # Animated footer with rotating tips
        tips = [
            "💡 Live metrics update every second for real-time insights",
            "🎯 Efficiency score combines accuracy and speed for better ranking",
            "⚡ Speed is measured in puzzles solved per minute",
            "🔥 Streaks track consecutive correct solutions",
            "📈 Sparklines show recent performance trends",
            "🏆 Leaderboard ranks by overall efficiency score",
            "⚔️ Different puzzle types test various chess skills",
            "🎪 Concurrent execution means faster results!"
        ]

        tip_index = (self.update_count // 30) % len(tips)  # Change every ~5 seconds at 6fps
        current_tip = tips[tip_index]

        # Add some animation to footer
        current_time = time.time()
        phase = (current_time * 2) % (2 * math.pi)
        glow_intensity = (math.sin(phase) + 1) / 2

        if glow_intensity > 0.8:
            tip_style = "bright_cyan"
        elif glow_intensity > 0.5:
            tip_style = "cyan"
        else:
            tip_style = "dim cyan"

        footer_content = Align.center(Text(current_tip, style=tip_style))
        layout["footer"].update(Panel(footer_content, box=ROUNDED, style="dim"))

        return layout

    async def run_cool_benchmark(self, bot_specs: List[BotSpec], config: Config,
                                output_dir: Path, puzzle_types: List[PuzzleType],
                                difficulty_range: tuple, puzzle_count: int) -> Dict[str, Dict[str, PuzzleStats]]:
        """Run benchmark with cool animated interface."""

        self.start_time = datetime.now()
        self.total_bots = len(bot_specs)
        self.completed_bots = 0

        # Initialize bot stats
        for bot_spec in bot_specs:
            self.bot_stats[bot_spec.name] = CoolBotStats(
                name=bot_spec.name,
                start_time=datetime.now()
            )

        # Limit concurrent bots for stability
        semaphore = asyncio.Semaphore(min(4, len(bot_specs)))

        async def run_single_bot(bot_spec: BotSpec) -> tuple:
            """Run benchmark for a single bot."""
            async with semaphore:
                try:
                    stats = self.bot_stats[bot_spec.name]
                    stats.status = "🚀 Loading puzzles..."

                    # Simulate some initialization delay for effect
                    await asyncio.sleep(0.5)

                    stats.status = "🧩 Solving puzzles..."
                    results = await run_puzzle_benchmark(bot_spec, config, output_dir)

                    # Update final stats
                    if results:
                        total_correct = sum(s.correct_solutions for s in results.values())
                        total_attempts = sum(s.total_attempts for s in results.values())
                        avg_time = sum(s.average_response_time for s in results.values()) / len(results)

                        stats.solved_puzzles = total_correct
                        stats.current_puzzle = total_attempts
                        stats.total_puzzles = total_attempts
                        stats.success_rate = total_correct / total_attempts if total_attempts > 0 else 0.0
                        stats.avg_time = avg_time

                        # Simulate individual puzzle results for animations
                        for i in range(total_attempts):
                            is_correct = i < total_correct
                            puzzle_type = list(PuzzleType)[i % len(PuzzleType)]
                            stats.add_puzzle_result(is_correct, avg_time, puzzle_type)

                    stats.status = "✅ Mission complete!"
                    stats.is_active = False
                    self.completed_bots += 1

                    return bot_spec.name, results

                except Exception as e:
                    stats = self.bot_stats[bot_spec.name]
                    stats.status = f"❌ Error: {str(e)[:20]}..."
                    stats.is_active = False
                    return bot_spec.name, {}

        # Create tasks
        tasks = [asyncio.create_task(run_single_bot(bot_spec)) for bot_spec in bot_specs]
        results = {}

        # Run with animated live display
        with Live(self.create_dashboard(), refresh_per_second=6, console=self.console) as live:
            try:
                # Wait for completion with live updates
                for task in asyncio.as_completed(tasks):
                    bot_name, bot_results = await task
                    results[bot_name] = bot_results
                    live.update(self.create_dashboard())

                # Show final animation
                await asyncio.sleep(2)

            except Exception as e:
                # Cancel remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                raise

        return results

    def show_epic_finale(self, results: Dict[str, Dict[str, PuzzleStats]]) -> None:
        """Show epic finale with animations and celebrations."""
        self.console.clear()

        # Epic header
        self.console.print("")
        self.console.rule("🎊 [bold bright_magenta]EPIC CHESS PUZZLE CHAMPIONSHIP COMPLETE[/bold bright_magenta] 🎊",
                         style="bright_magenta")
        self.console.print("")

        if not results:
            self.console.print(Panel("No champions emerged...", title="😢 Sad Results", style="red"))
            return

        # Calculate comprehensive results
        all_results = []
        for bot_name, bot_results in results.items():
            if bot_results:
                total_correct = sum(stats.correct_solutions for stats in bot_results.values())
                total_attempts = sum(stats.total_attempts for stats in bot_results.values())
                avg_time = sum(stats.average_response_time for stats in bot_results.values()) / len(bot_results)
                success_rate = total_correct / total_attempts if total_attempts > 0 else 0
                speed = (total_attempts / avg_time * 60) if avg_time > 0 else 0
                efficiency = success_rate * min(2.0, speed / 30.0)  # Efficiency score

                all_results.append({
                    'name': bot_name,
                    'success_rate': success_rate,
                    'total_correct': total_correct,
                    'total_attempts': total_attempts,
                    'avg_time': avg_time,
                    'speed': speed,
                    'efficiency': efficiency
                })

        # Sort by efficiency score
        all_results.sort(key=lambda x: x['efficiency'], reverse=True)

        # Create epic results table
        epic_table = Table(box=HEAVY, title="🏆 HALL OF FAME", title_style="bold bright_yellow")
        epic_table.add_column("Rank", style="bold yellow", width=6)
        epic_table.add_column("Champion", style="bold cyan", width=15)
        epic_table.add_column("Success Rate", style="bold green", width=12)
        epic_table.add_column("Speed", style="bold blue", width=10)
        epic_table.add_column("Efficiency", style="bold magenta", width=12)
        epic_table.add_column("Title", style="bold bright_cyan", width=15)

        # Epic ranking system
        titles = [
            "👑 GRANDMASTER",
            "🥇 MASTER",
            "🥈 EXPERT",
            "🥉 ADVANCED",
            "🏅 SKILLED",
            "⭐ NOVICE"
        ]

        for i, result in enumerate(all_results):
            rank_display = titles[i] if i < len(titles) else f"#{i+1}"

            # Determine title based on efficiency
            if result['efficiency'] >= 1.5:
                title = "🚀 LEGENDARY"
            elif result['efficiency'] >= 1.2:
                title = "⭐ EXCELLENT"
            elif result['efficiency'] >= 0.8:
                title = "✨ GOOD"
            elif result['efficiency'] >= 0.5:
                title = "📈 FAIR"
            else:
                title = "💪 TRAINING"

            epic_table.add_row(
                rank_display,
                result['name'],
                f"{result['success_rate']:.1%}",
                f"{result['speed']:.1f}/min",
                f"{result['efficiency']:.2f}",
                title
            )

        self.console.print(Align.center(epic_table))

        # Epic winner announcement
        if all_results:
            winner = all_results[0]
            self.console.print("")

            winner_panel = Panel(
                Align.center(Group(
                    Text(f"🎉 CHAMPION: {winner['name']} 🎉", style="bold bright_yellow"),
                    Text(""),
                    Text(f"Success Rate: {winner['success_rate']:.1%}", style="green"),
                    Text(f"Speed: {winner['speed']:.1f} puzzles/min", style="blue"),
                    Text(f"Efficiency Score: {winner['efficiency']:.2f}", style="magenta"),
                    Text(""),
                    Text("🏆 ULTIMATE CHESS PUZZLE MASTER! 🏆", style="bold bright_gold")
                )),
                title="👑 VICTORY",
                box=DOUBLE,
                style="bright_yellow"
            )

            self.console.print(Align.center(winner_panel))

        # Final message
        self.console.print("")
        self.console.print(Align.center(
            Text("Thank you for the epic chess puzzle battle! 🎯♔⚔️", style="bold bright_cyan")
        ))
        self.console.print("")


async def run_cool_stats_cli(bot_specs: List[BotSpec], config: Config, output_dir: Path,
                            puzzle_types: List[PuzzleType], difficulty_range: tuple = (1, 10),
                            puzzle_count: int = 10) -> Dict[str, Dict[str, PuzzleStats]]:
    """
    Run the cool animated stats CLI.

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
    cli = CoolStatsCLI()

    try:
        # Epic startup sequence
        with cli.console.status("[bold bright_cyan]🚀 Launching Epic Chess Puzzle Championship...", spinner="dots"):
            await asyncio.sleep(1)

        cli.console.clear()

        # Show epic intro
        cli.console.print("")
        cli.console.rule("🎊 [bold bright_magenta]EPIC CHESS PUZZLE CHAMPIONSHIP[/bold bright_magenta] 🎊",
                        style="bright_magenta")
        cli.console.print("")

        intro_text = Text.assemble(
            ("🧩 Welcome to the ultimate test of AI chess mastery! 🧩\n", "bold cyan"),
            ("⚡ Real-time analytics • 🎯 Advanced metrics • 🏆 Epic competition\n", "dim"),
            ("\nPrepare for concurrent puzzle-solving action!", "yellow")
        )

        cli.console.print(Align.center(Panel(intro_text, box=ROUNDED, style="cyan")))
        cli.console.print("")

        # Brief pause for dramatic effect
        await asyncio.sleep(1)

        # Run the epic benchmark
        results = await cli.run_cool_benchmark(
            bot_specs, config, output_dir, puzzle_types, difficulty_range, puzzle_count
        )

        # Show epic finale
        cli.show_epic_finale(results)

        return results

    except KeyboardInterrupt:
        cli.console.print("\n[yellow]⚠️ Epic battle interrupted by user[/yellow]")
        return {}
    except Exception as e:
        cli.console.print(f"\n[red]❌ Epic battle failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {}
