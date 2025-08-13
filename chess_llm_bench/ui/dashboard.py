"""
UI dashboard module for real-time chess game visualization.

This module provides a Rich-based terminal UI for displaying live game states,
statistics, and ladder progression across multiple bots simultaneously.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from datetime import datetime

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.layout import Layout
from rich.align import Align

from ..core.models import LiveState, LadderStats, Config, BenchmarkResult
from ..core.budget import get_budget_tracker
from .board import ChessBoardRenderer, render_robot_battle, BoardTheme
import chess

logger = logging.getLogger(__name__)


class Dashboard:
    """
    Rich-based terminal dashboard for live chess game visualization.

    Displays real-time information about multiple bots playing chess games
    simultaneously, including board states, statistics, and ladder progression.
    """

    def __init__(self, console: Optional[Console] = None, config: Optional[Config] = None):
        """
        Initialize the dashboard.

        Args:
            console: Rich console instance (creates new if None)
            config: Configuration object for display settings
        """
        self.console = console or Console()
        self.config = config or Config()
        self._live: Optional[Live] = None
        self.board_renderer = ChessBoardRenderer(
            theme=BoardTheme.UNICODE,
            show_coordinates=True
        )

    def start_live_display(self) -> Live:
        """
        Start the live updating display.

        Returns:
            Live context manager for updates
        """
        if self._live is not None:
            return self._live

        # Start Live display with proper configuration for real-time updates
        self._live = Live(
            self._create_empty_display(),
            console=self.console,
            refresh_per_second=self.config.refresh_rate,
            auto_refresh=True
        )
        return self._live

    def stop_live_display(self) -> None:
        """Stop the live display if running."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception as e:
                logger.warning(f"Error stopping live display: {e}")
            finally:
                self._live = None

    def update_display(self, states: Dict[str, LiveState], stats: Dict[str, LadderStats]) -> None:
        """
        Update the live display with current states and statistics.

        Args:
            states: Current live states for each bot
            stats: Current statistics for each bot
        """
        if self._live is None:
            logger.warning("Dashboard update called but _live is None")
            return

        try:
            # Always update with appropriate content
            if states:
                content = self.render_dashboard(states, stats)
            else:
                content = self._create_empty_display()

            # Update the live display
            self._live.update(content)

        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")


    def render_dashboard(self, states: Dict[str, LiveState], stats: Dict[str, LadderStats]) -> Panel:
        """
        Render the complete dashboard layout.

        Args:
            states: Current live states for each bot
            stats: Current statistics for each bot

        Returns:
            Rich Panel with complete dashboard
        """
        if not states:
            return self._create_empty_display()

        # Create individual bot panels
        bot_panels = []
        for bot_name in sorted(states.keys()):
            state = states[bot_name]
            bot_stats = stats.get(bot_name)
            panel = self._render_bot_panel(bot_name, state, bot_stats)
            bot_panels.append(panel)

        # Create summary panel
        summary_panel = self._render_summary_panel(states, stats)

        # Arrange panels
        if len(bot_panels) == 1:
            main_content = bot_panels[0]
        elif len(bot_panels) == 2:
            main_content = Columns(bot_panels, equal=True, expand=True)
        else:
            # For more than 2 bots, arrange in rows
            rows = []
            for i in range(0, len(bot_panels), 2):
                row_panels = bot_panels[i:i+2]
                if len(row_panels) == 1:
                    rows.append(row_panels[0])
                else:
                    rows.append(Columns(row_panels, equal=True, expand=True))
            main_content = Group(*rows)

        # Combine with summary
        if len(states) > 1:
            full_content = Group(summary_panel, main_content)
        else:
            full_content = Group(main_content, summary_panel)

        return Panel(
            full_content,
            title="🏆 Chess LLM ELO Ladder Benchmark",
            title_align="center",
            border_style="magenta",
            padding=(1, 2)
        )

    def _render_bot_panel(
        self,
        bot_name: str,
        state: LiveState,
        stats: Optional[LadderStats]
    ) -> Panel:
        """Render an individual bot's status panel."""
        # Header with bot name and ladder
        header = Text(f"{bot_name}", style="bold cyan")
        ladder_display = f"Ladder: {state.ladder_display}"

        # Create main table
        table = Table.grid(expand=True)

        # Board display (left column) - Beautiful Unicode chess board
        if state.board_ascii and hasattr(state, '_chess_board') and state._chess_board is not None:
            # Use the beautiful chess board renderer
            try:
                last_move = chess.Move.from_uci(state.last_move_uci) if state.last_move_uci else None
            except:
                last_move = None

            board_panel = self.board_renderer.render_board(
                state._chess_board,
                last_move=last_move
            )

            # Apply border style based on state
            if state.error_message:
                board_panel.border_style = "red"
            elif "thinking" in state.status.lower():
                board_panel.border_style = "yellow"
            else:
                board_panel.border_style = "green"
        elif state.board_ascii:
            # Fallback to ASCII if chess board object not available
            board_text = Text(state.board_ascii, style="white")
            if state.error_message:
                board_style = "red"
            elif "thinking" in state.status.lower():
                board_style = "yellow"
            else:
                board_style = "green"
            board_panel = Panel(
                Align.center(board_text),
                border_style=board_style,
                padding=(0, 1)
            )
        else:
            board_panel = Panel(
                Align.center(Text("🎯 Waiting for game...", style="dim cyan")),
                border_style="blue",
                padding=(0, 1)
            )

        # Status information (right column)
        status_lines = []

        # Current game info
        status_lines.append(f"Status: {state.status}")
        if state.current_elo > 0:
            status_lines.append(f"Current ELO: {state.current_elo}")
        status_lines.append(f"Color: {state.color_display}")
        status_lines.append(f"Moves: {state.moves_made}")
        if state.last_move_uci:
            status_lines.append(f"Last move: {state.last_move_uci}")

        # Statistics
        if stats:
            status_lines.append("")  # Separator
            status_lines.append(f"Max ELO reached: {stats.max_elo_reached}")
            status_lines.append(f"Games played: {stats.total_games}")
            if stats.total_games > 0:
                status_lines.append(f"Win rate: {stats.win_rate:.1%}")
                status_lines.append(f"Wins: {stats.wins} | Draws: {stats.draws} | Losses: {stats.losses}")

            # Timing and move quality stats
            if stats.average_move_time > 0:
                status_lines.append(f"Avg move time: {stats.average_move_time:.2f}s")
            if stats.total_illegal_moves > 0:
                status_lines.append(f"Illegal moves: {stats.total_illegal_moves}")

        # Live timing stats during current game
        if hasattr(state, 'average_move_time') and state.average_move_time > 0:
            status_lines.append("")
            status_lines.append(f"Current game avg: {state.average_move_time:.2f}s")
            if hasattr(state, 'illegal_move_attempts') and state.illegal_move_attempts > 0:
                status_lines.append(f"Current illegal: {state.illegal_move_attempts}")

        # Error handling
        if state.error_message:
            status_lines.append("")
            status_lines.append(f"❌ Error: {state.error_message}")

        status_text = Text("\n".join(status_lines))

        # Combine board and status
        content_table = Table.grid(expand=True)
        content_table.add_column("board", ratio=3)
        content_table.add_column("status", ratio=2)
        content_table.add_row(board_panel, status_text)

        # Final result highlight
        border_style = "cyan"
        if state.is_finished:
            if state.error_message:
                border_style = "red"
            elif state.final_result:
                if "1-0" in state.final_result or "0-1" in state.final_result:
                    border_style = "green"
                else:
                    border_style = "yellow"

        return Panel(
            Group(
                header,
                Text(ladder_display, style="dim"),
                content_table
            ),
            title=bot_name,
            border_style=border_style,
            padding=(1, 1)
        )

    def _render_summary_panel(
        self,
        states: Dict[str, LiveState],
        stats: Dict[str, LadderStats]
    ) -> Panel:
        """Render a summary panel with overall statistics."""
        if len(states) <= 1:
            return self._render_single_bot_summary(states, stats)

        # Multi-bot summary table
        table = Table(expand=True)
        table.add_column("Bot", style="cyan", no_wrap=True)
        table.add_column("Status", style="white")
        table.add_column("Max ELO", style="green", justify="right")
        table.add_column("Games", style="blue", justify="right")
        table.add_column("Win Rate", style="yellow", justify="right")
        table.add_column("Record", style="dim", justify="center")
        table.add_column("Avg Time", style="magenta", justify="right")
        table.add_column("Illegal Moves", style="red", justify="right")

        for bot_name in sorted(states.keys()):
            state = states[bot_name]
            bot_stats = stats.get(bot_name)

            # Status with emoji
            if state.error_message:
                status = "❌ Error"
                status_style = "red"
            elif state.is_finished:
                status = "✅ Finished"
                status_style = "green"
            elif "thinking" in state.status.lower():
                status = "🤔 Thinking"
                status_style = "yellow"
            else:
                status = "🎮 Playing"
                status_style = "blue"

            # Statistics
            max_elo = str(bot_stats.max_elo_reached) if bot_stats else "0"
            games = str(bot_stats.total_games) if bot_stats else "0"
            win_rate = f"{bot_stats.win_rate:.1%}" if bot_stats and bot_stats.total_games > 0 else "—"
            record = f"{bot_stats.wins}W-{bot_stats.draws}D-{bot_stats.losses}L" if bot_stats else "—"

            # Timing and illegal move statistics
            avg_time = f"{bot_stats.average_move_time:.2f}s" if bot_stats and bot_stats.average_move_time > 0 else "—"
            illegal_moves = str(bot_stats.total_illegal_moves) if bot_stats else "0"

            table.add_row(
                bot_name,
                Text(status, style=status_style),
                max_elo,
                games,
                win_rate,
                record,
                avg_time,
                illegal_moves
            )

        # Add budget information if available
        budget_text = None
        try:
            budget_tracker = get_budget_tracker()
            if budget_tracker and budget_tracker.is_active and budget_tracker.get_current_cost() > 0:
                budget_info = budget_tracker.get_budget_info()
                if budget_info:
                    budget_text = Text()
                    budget_text.append("💰 Cost: ", style="dim")
                    budget_text.append(budget_info["total_cost"], style=budget_info.get("style", "green"))

                    if "budget_limit" in budget_info and budget_info["budget_limit"] != "Unlimited":
                        budget_text.append(" / ", style="dim")
                        budget_text.append(budget_info["budget_limit"], style="dim")
                        budget_text.append(f" ({budget_info.get('usage_percentage', '0%')})", style="dim")

                    if "api_calls" in budget_info:
                        budget_text.append(f"  |  📞 {budget_info['api_calls']} calls", style="dim")
        except Exception:
            # Ignore budget display errors to not break the dashboard
            pass

        # Combine table and budget info
        if budget_text:
            final_content = Group(table, budget_text)
        else:
            final_content = table

        return Panel(
            final_content,
            title="📊 Summary",
            border_style="blue",
            padding=(0, 1)
        )

    def _render_single_bot_summary(
        self,
        states: Dict[str, LiveState],
        stats: Dict[str, LadderStats]
    ) -> Panel:
        """Render summary for single bot runs."""
        if not states:
            return Panel("No active bots", title="Summary")

        bot_name = list(states.keys())[0]
        state = states[bot_name]
        bot_stats = stats.get(bot_name)

        summary_lines = []

        if state.is_finished:
            summary_lines.append("🎯 Run completed!")
        else:
            summary_lines.append("🎮 Game in progress...")

        if bot_stats:
            summary_lines.append(f"🏆 Best ELO: {bot_stats.max_elo_reached}")
            if bot_stats.total_games > 0:
                summary_lines.append(f"📊 Performance: {bot_stats.wins}W-{bot_stats.draws}D-{bot_stats.losses}L")

        timestamp = datetime.now().strftime("%H:%M:%S")
        summary_lines.append(f"⏰ {timestamp}")

        return Panel(
            "\n".join(summary_lines),
            title="📊 Summary",
            border_style="blue",
            padding=(0, 1)
        )

    def _create_empty_display(self) -> Panel:
        """Create an empty dashboard display."""
        return Panel(
            Align.center(
                Group(
                    Text("🏆 Chess LLM ELO Ladder", style="bold magenta", justify="center"),
                    Text("Initializing...", style="dim", justify="center")
                )
            ),
            border_style="magenta",
            padding=(2, 4)
        )

    def display_final_results(self, result: BenchmarkResult) -> None:
        """
        Display final benchmark results in a formatted summary.

        Args:
            result: Complete benchmark results
        """
        self.console.print("\n")

        # Main results panel
        results_table = Table(expand=True, show_header=True, header_style="bold cyan")
        results_table.add_column("Bot", style="cyan", no_wrap=True)
        results_table.add_column("Max ELO", style="green", justify="right")
        results_table.add_column("Games", style="blue", justify="right")
        results_table.add_column("Win Rate", style="yellow", justify="right")
        results_table.add_column("Record", style="white", justify="center")
        results_table.add_column("Avg Time", style="bright_magenta", justify="right")
        results_table.add_column("Illegal Moves", style="red", justify="right")
        results_table.add_column("Performance", style="magenta")

        for bot_name, stats in result.bot_results.items():
            win_rate = f"{stats.win_rate:.1%}" if stats.total_games > 0 else "—"
            record = f"{stats.wins}W-{stats.draws}D-{stats.losses}L"

            # Timing and illegal move statistics
            avg_time = f"{stats.average_move_time:.2f}s" if stats.average_move_time > 0 else "—"
            illegal_moves = str(stats.total_illegal_moves)

            # Performance assessment
            if stats.max_elo_reached >= 1800:
                performance = "🏆 Excellent"
                performance_style = "bold green"
            elif stats.max_elo_reached >= 1400:
                performance = "⭐ Good"
                performance_style = "green"
            elif stats.max_elo_reached >= 1000:
                performance = "👍 Fair"
                performance_style = "yellow"
            else:
                performance = "📚 Learning"
                performance_style = "red"

            results_table.add_row(
                bot_name,
                str(stats.max_elo_reached),
                str(stats.total_games),
                win_rate,
                record,
                avg_time,
                illegal_moves,
                Text(performance, style=performance_style)
            )

        # Summary info
        summary_lines = [
            f"🕐 Completed: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"🎮 Total games: {result.total_games}",
        ]

        if result.best_bot:
            summary_lines.append(f"🏆 Best bot: {result.best_bot} (ELO {result.best_elo})")

        # Add budget information if available
        budget_tracker = get_budget_tracker()
        if budget_tracker.is_active and budget_tracker.get_current_cost() > 0:
            summary_lines.append(f"💰 Total cost: ${budget_tracker.get_current_cost():.4f}")
            if budget_tracker.budget_limit:
                percentage = (budget_tracker.get_current_cost() / budget_tracker.budget_limit) * 100
                summary_lines.append(f"📊 Budget usage: {percentage:.1f}%")

        summary_lines.append(f"💾 Results saved to: {result.output_dir}")

        summary_text = "\n".join(summary_lines)

        # Final display
        final_panel = Panel(
            Group(
                results_table,
                Text(""),
                Text(summary_text, style="dim")
            ),
            title="🎯 Final Results",
            title_align="center",
            border_style="green",
            padding=(1, 2)
        )

        self.console.print(final_panel)

    def display_error(self, error: str, title: str = "Error") -> None:
        """Display an error message in a formatted panel."""
        error_panel = Panel(
            Text(error, style="red"),
            title=f"❌ {title}",
            border_style="red",
            padding=(1, 2)
        )
        self.console.print(error_panel)

    def display_info(self, message: str, title: str = "Info") -> None:
        """Display an info message in a formatted panel."""
        info_panel = Panel(
            Text(message, style="blue"),
            title=f"ℹ️ {title}",
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(info_panel)

    def display_success(self, message: str, title: str = "Success") -> None:
        """Display a success message in a formatted panel."""
        success_panel = Panel(
            Text(message, style="green"),
            title=f"✅ {title}",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(success_panel)

    def render_robot_battle(
        self,
        board: chess.Board,
        white_bot: str,
        black_bot: str,
        last_move: Optional[chess.Move] = None,
        engine_elo: Optional[int] = None,
        moves: Optional[List[chess.Move]] = None,
        status: str = ""
    ) -> Panel:
        """
        Render a beautiful robot vs robot chess battle.

        Args:
            board: Current chess position
            white_bot: Name of white bot
            black_bot: Name of black bot
            last_move: Last move played
            engine_elo: Engine ELO rating
            moves: List of moves played
            status: Current game status

        Returns:
            Rich Panel with robot battle display
        """
        return render_robot_battle(
            board=board,
            white_bot=white_bot,
            black_bot=black_bot,
            last_move=last_move,
            engine_elo=engine_elo,
            moves=moves
        )

    def display_robot_demo(
        self,
        board: chess.Board,
        white_bot: str,
        black_bot: str,
        last_move: Optional[chess.Move] = None,
        engine_elo: Optional[int] = None,
        moves: Optional[List[chess.Move]] = None,
        status: str = "Game in progress..."
    ) -> None:
        """
        Display a robot vs robot demo game.

        Args:
            board: Current chess position
            white_bot: Name of white bot
            black_bot: Name of black bot
            last_move: Last move played
            engine_elo: Engine ELO rating
            moves: List of moves played
            status: Current game status
        """
        battle_panel = self.render_robot_battle(
            board, white_bot, black_bot, last_move, engine_elo, moves, status
        )

        # Add status footer
        footer_text = Text(status, justify="center", style="bold cyan")

        complete_display = Table.grid()
        complete_display.add_row(battle_panel)
        complete_display.add_row(Panel(footer_text, border_style="dim"))

        self.console.print(complete_display)

    def __enter__(self):
        """Context manager entry."""
        return self.start_live_display()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_live_display()
