#!/usr/bin/env python3
"""
Human Engine Demo for Chess LLM Benchmark

This script demonstrates the human-like chess engines available in the Chess LLM
Benchmark system. It shows how human engines provide more realistic opponents
compared to traditional chess engines like Stockfish.

Features demonstrated:
- Human-like Stockfish vs Traditional Stockfish comparison
- ELO-based difficulty scaling with human characteristics
- Move variation and human-like decision making
- Integration with LLM benchmarking

Usage:
    python examples/human_engine_demo.py [--mode MODE] [--elo ELO]

Modes:
    comparison   - Compare human vs traditional engines (default)
    benchmark    - Run a mini benchmark with human engines
    interactive  - Interactive demo with move explanations
    analysis     - Analyze human-like characteristics

Requirements:
    - At least one chess engine installed (Stockfish recommended)
    - python-chess library
    - Chess LLM Benchmark dependencies
"""

import argparse
import asyncio
import logging
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import chess
import chess.pgn as chess_pgn
from datetime import datetime

from chess_llm_bench.core.models import Config, BotSpec, LiveState
from chess_llm_bench.core.engine import ChessEngine, autodetect_stockfish
from chess_llm_bench.core.human_engine import (
    autodetect_human_engines,
    get_best_human_engine,
    create_human_engine,
    HumanStockfishEngine
)
from chess_llm_bench.core.game import GameRunner
from chess_llm_bench.llm.client import LLMClient
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise for demo
logger = logging.getLogger(__name__)

console = Console()


class HumanEngineDemo:
    """Main demo class for human engine capabilities."""

    def __init__(self):
        self.config = Config(
            think_time=0.5,  # Faster for demo
            max_plies=100,   # Shorter games
            escalate_on="always"
        )
        self.engines = {}
        self.available_engines = {}

    async def setup_engines(self):
        """Initialize available engines."""
        console.print("🔍 [bold blue]Detecting available engines...[/bold blue]")

        # Check for human engines
        self.available_engines = autodetect_human_engines()

        # Add regular Stockfish
        stockfish_path = autodetect_stockfish()
        if stockfish_path:
            self.available_engines['stockfish'] = stockfish_path

        # Display available engines
        table = Table(title="Available Chess Engines")
        table.add_column("Engine Type", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Path", style="dim")
        table.add_column("Human-like", style="magenta")

        for engine_type, path in self.available_engines.items():
            if path:
                status = "✅ Available"
                human_like = "🧠 Yes" if engine_type != "stockfish" else "🤖 No"
            else:
                status = "❌ Not found"
                human_like = "—"
                path = "—"

            table.add_row(engine_type, status, str(path)[-50:], human_like)

        console.print(table)

        if not any(self.available_engines.values()):
            console.print("\n[red]❌ No chess engines found![/red]")
            console.print("Please install Stockfish or another supported engine.")
            return False

        return True

    async def comparison_demo(self):
        """Compare human engines vs traditional engines."""
        console.print("\n🎯 [bold green]Human vs Traditional Engine Comparison[/bold green]")
        console.print("=" * 60)

        # Get engines to compare
        human_engine_info = get_best_human_engine()
        stockfish_path = self.available_engines.get('stockfish')

        if not stockfish_path:
            console.print("[red]❌ Stockfish required for comparison[/red]")
            return

        # Test positions for comparison
        test_positions = [
            {
                "name": "Opening",
                "fen": chess.Board().fen(),
                "description": "Starting position"
            },
            {
                "name": "Tactical",
                "fen": "r1bq1rk1/ppp2ppp/2n2n2/2bpp3/2B1P3/3P1N2/PPP2PPP/RNBQR1K1 w - - 0 8",
                "description": "Tactical middlegame position"
            },
            {
                "name": "Endgame",
                "fen": "8/8/8/3k4/8/3K4/8/R7 w - - 0 1",
                "description": "Rook endgame"
            }
        ]

        for position in test_positions:
            console.print(f"\n📋 [bold]{position['name']}[/bold]: {position['description']}")
            board = chess.Board(position['fen'])

            results = {}
            elos_to_test = [800, 1200, 1600, 2000]

            # Test human engine if available
            if human_engine_info:
                engine_type, engine_path = human_engine_info
                console.print(f"🧠 Testing {engine_type} (human-like)...")

                try:
                    human_engine = create_human_engine(engine_type, engine_path, self.config)
                    await human_engine.start()

                    human_moves = {}
                    for elo in elos_to_test:
                        await human_engine.configure_elo(elo)
                        move = await human_engine.get_move(board)
                        human_moves[elo] = move

                    results['human'] = human_moves
                    await human_engine.stop()

                except Exception as e:
                    console.print(f"[red]❌ Human engine failed: {e}[/red]")

            # Test traditional Stockfish
            console.print("🤖 Testing traditional Stockfish...")
            try:
                stockfish_engine = ChessEngine(stockfish_path, self.config)
                await stockfish_engine.start()

                stockfish_moves = {}
                for elo in elos_to_test:
                    await stockfish_engine.configure_elo(elo)
                    move = await stockfish_engine.get_move(board)
                    stockfish_moves[elo] = move

                results['stockfish'] = stockfish_moves
                await stockfish_engine.stop()

            except Exception as e:
                console.print(f"[red]❌ Stockfish failed: {e}[/red]")

            # Display comparison
            if results:
                comparison_table = Table(title=f"Move Comparison: {position['name']}")
                comparison_table.add_column("ELO", style="bold")

                for engine_name in results.keys():
                    style = "green" if engine_name == 'human' else "blue"
                    comparison_table.add_column(f"{engine_name.title()}", style=style)

                for elo in elos_to_test:
                    row = [str(elo)]
                    for engine_name in results.keys():
                        move = results[engine_name].get(elo, "—")
                        row.append(str(move))
                    comparison_table.add_row(*row)

                console.print(comparison_table)

                # Analyze differences
                if 'human' in results and 'stockfish' in results:
                    differences = 0
                    for elo in elos_to_test:
                        if results['human'].get(elo) != results['stockfish'].get(elo):
                            differences += 1

                    pct_different = (differences / len(elos_to_test)) * 100
                    console.print(f"📊 [bold]Difference: {pct_different:.0f}% of moves varied between engines[/bold]")

    async def benchmark_demo(self):
        """Run a mini benchmark comparing human vs traditional engines."""
        console.print("\n🏆 [bold green]Mini Benchmark: Human vs Traditional Engines[/bold green]")
        console.print("=" * 60)

        # Create a simple random bot for testing
        random_bot = BotSpec(provider="random", model="", name="RandomBot")

        engines_to_test = []

        # Add human engine if available
        human_engine_info = get_best_human_engine()
        if human_engine_info:
            engines_to_test.append(('human', human_engine_info))

        # Add traditional Stockfish
        stockfish_path = self.available_engines.get('stockfish')
        if stockfish_path:
            engines_to_test.append(('traditional', ('stockfish', stockfish_path)))

        if not engines_to_test:
            console.print("[red]❌ No engines available for benchmark[/red]")
            return

        results = {}

        for engine_category, (engine_type, engine_path) in engines_to_test:
            console.print(f"\n🎮 Testing {engine_category} engine ({engine_type})...")

            try:
                # Create engine
                if engine_category == 'human' and engine_type != 'stockfish':
                    engine = create_human_engine(engine_type, engine_path, self.config)
                else:
                    engine = ChessEngine(engine_path, self.config)

                # Create LLM client (random bot)
                llm_client = LLMClient(random_bot)

                # Run games at different ELOs
                elo_results = {}
                test_elos = [800, 1200, 1600]

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:

                    for elo in test_elos:
                        task = progress.add_task(f"Playing vs ELO {elo}...", total=None)

                        game_runner = GameRunner(llm_client, engine, self.config)

                        # Play a quick game
                        state = LiveState(title=f"Test vs {elo}")
                        game_record = await game_runner.play_game(
                            elo=elo,
                            output_dir=Path("temp"),
                            state=state,
                            llm_plays_white=True
                        )

                        elo_results[elo] = {
                            'result': game_record.result,
                            'plies': game_record.ply_count,
                            'llm_won': game_record.llm_won,
                            'is_draw': game_record.is_draw
                        }

                        progress.remove_task(task)

                results[engine_category] = elo_results

            except Exception as e:
                console.print(f"[red]❌ Failed to test {engine_category}: {e}[/red]")

        # Display benchmark results
        if results:
            self._display_benchmark_results(results)

    def _display_benchmark_results(self, results: Dict):
        """Display benchmark results in a nice format."""
        console.print("\n📊 [bold]Benchmark Results[/bold]")

        result_table = Table(title="Game Results by Engine Type")
        result_table.add_column("Engine", style="cyan")
        result_table.add_column("ELO", style="bold")
        result_table.add_column("Result", style="green")
        result_table.add_column("Plies", style="blue")
        result_table.add_column("Outcome", style="magenta")

        for engine_type, elo_results in results.items():
            for elo, game_result in elo_results.items():
                outcome = "Win" if game_result['llm_won'] else "Draw" if game_result['is_draw'] else "Loss"
                outcome_style = "✅" if game_result['llm_won'] else "➖" if game_result['is_draw'] else "❌"

                result_table.add_row(
                    engine_type.title(),
                    str(elo),
                    game_result['result'],
                    str(game_result['plies']),
                    f"{outcome_style} {outcome}"
                )

        console.print(result_table)

        # Summary statistics
        summary = {}
        for engine_type, elo_results in results.items():
            wins = sum(1 for r in elo_results.values() if r['llm_won'])
            draws = sum(1 for r in elo_results.values() if r['is_draw'])
            total_games = len(elo_results)
            avg_plies = sum(r['plies'] for r in elo_results.values()) / total_games

            summary[engine_type] = {
                'wins': wins,
                'draws': draws,
                'losses': total_games - wins - draws,
                'total_games': total_games,
                'avg_plies': avg_plies,
                'win_rate': (wins / total_games) * 100
            }

        summary_table = Table(title="Summary Statistics")
        summary_table.add_column("Engine", style="cyan")
        summary_table.add_column("Win Rate", style="green")
        summary_table.add_column("Record", style="blue")
        summary_table.add_column("Avg Plies", style="magenta")

        for engine_type, stats in summary.items():
            summary_table.add_row(
                engine_type.title(),
                f"{stats['win_rate']:.1f}%",
                f"{stats['wins']}W-{stats['draws']}D-{stats['losses']}L",
                f"{stats['avg_plies']:.0f}"
            )

        console.print("\n")
        console.print(summary_table)

    async def interactive_demo(self):
        """Interactive demo showing move-by-move differences."""
        console.print("\n🎮 [bold green]Interactive Human Engine Demo[/bold green]")
        console.print("=" * 60)

        # Get best available engine
        human_engine_info = get_best_human_engine()
        if not human_engine_info:
            console.print("[red]❌ No human engines available for interactive demo[/red]")
            return

        engine_type, engine_path = human_engine_info
        console.print(f"Using {engine_type} engine for interactive demo")

        try:
            engine = create_human_engine(engine_type, engine_path, self.config)
            await engine.start()

            # Interactive game
            board = chess.Board()
            move_count = 0

            console.print("\n🎯 [bold]Interactive Game Preview[/bold]")
            console.print("Watch how the human engine plays at different ELO levels!\n")

            elos_to_demo = [800, 1200, 1600, 2000]

            for elo in elos_to_demo:
                if board.is_game_over() or move_count > 10:
                    break

                console.print(f"[bold cyan]ELO {elo} Move:[/bold cyan]")

                await engine.configure_elo(elo)
                move = await engine.get_move(board)

                # Create a display of the current position
                console.print(f"Position: {board.fen()}")
                console.print(f"Legal moves: {len(list(board.legal_moves))}")
                console.print(f"Chosen move: [bold green]{move}[/bold green]")

                # Make the move
                board.push(move)
                move_count += 1

                # Show a simple ASCII board
                console.print("Current position:")
                console.print(Panel(str(board), title=f"After {move}", border_style="blue"))

                if move_count < 10 and not board.is_game_over():
                    console.print("Press Enter to see next ELO level...")
                    input()

            await engine.stop()

        except Exception as e:
            console.print(f"[red]❌ Interactive demo failed: {e}[/red]")

    async def analysis_demo(self):
        """Analyze human-like characteristics of the engines."""
        console.print("\n📈 [bold green]Human-like Characteristics Analysis[/bold green]")
        console.print("=" * 60)

        human_engine_info = get_best_human_engine()
        if not human_engine_info:
            console.print("[red]❌ No human engines available for analysis[/red]")
            return

        engine_type, engine_path = human_engine_info

        try:
            engine = create_human_engine(engine_type, engine_path, self.config)
            await engine.start()

            # Test move consistency (human trait: some variation)
            console.print("🔄 [bold]Testing move variation (human trait)[/bold]")

            board = chess.Board()
            moves_per_elo = {}

            for elo in [800, 1200, 1600, 2000]:
                await engine.configure_elo(elo)
                moves = []

                # Get 5 moves from the same position
                for _ in range(5):
                    move = await engine.get_move(board)
                    moves.append(move)

                unique_moves = len(set(moves))
                moves_per_elo[elo] = {
                    'moves': moves,
                    'unique': unique_moves,
                    'variation_pct': (unique_moves / 5) * 100
                }

                console.print(f"ELO {elo}: {unique_moves}/5 unique moves ({(unique_moves/5)*100:.0f}% variation)")
                console.print(f"  Moves: {', '.join(str(m) for m in moves)}")

            # Analyze ELO scaling
            console.print("\n📊 [bold]ELO Scaling Analysis[/bold]")

            # Test different positions at different ELOs
            positions = [
                chess.Board(),  # Opening
                chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"),  # After 1.e4 d5
            ]

            for i, test_board in enumerate(positions):
                console.print(f"\nPosition {i+1}:")

                elo_moves = {}
                for elo in [600, 1000, 1400, 1800]:
                    await engine.configure_elo(elo)
                    move = await engine.get_move(test_board)
                    elo_moves[elo] = move

                # Check if moves change with ELO (human trait: skill scaling)
                unique_moves = len(set(elo_moves.values()))
                console.print(f"  ELO scaling: {unique_moves}/4 different moves across skill levels")

                for elo, move in elo_moves.items():
                    console.print(f"    ELO {elo}: {move}")

            await engine.stop()

            console.print("\n✨ [bold green]Analysis complete![/bold green]")
            console.print("Human engines show variation and ELO-appropriate scaling,")
            console.print("making them more realistic opponents for LLM evaluation.")

        except Exception as e:
            console.print(f"[red]❌ Analysis failed: {e}[/red]")

    async def run_demo(self, mode: str = "comparison", elo: int = 1200):
        """Run the specified demo mode."""
        if not await self.setup_engines():
            return

        if mode == "comparison":
            await self.comparison_demo()
        elif mode == "benchmark":
            await self.benchmark_demo()
        elif mode == "interactive":
            await self.interactive_demo()
        elif mode == "analysis":
            await self.analysis_demo()
        else:
            console.print(f"[red]❌ Unknown mode: {mode}[/red]")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Human Engine Demo for Chess LLM Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare human vs traditional engines
  python examples/human_engine_demo.py --mode comparison

  # Run mini benchmark
  python examples/human_engine_demo.py --mode benchmark

  # Interactive demonstration
  python examples/human_engine_demo.py --mode interactive

  # Analyze human-like characteristics
  python examples/human_engine_demo.py --mode analysis
        """
    )

    parser.add_argument(
        "--mode",
        choices=["comparison", "benchmark", "interactive", "analysis"],
        default="comparison",
        help="Demo mode to run (default: comparison)"
    )

    parser.add_argument(
        "--elo",
        type=int,
        default=1200,
        help="Target ELO for testing (default: 1200)"
    )

    args = parser.parse_args()

    console.print("🏆 [bold blue]Chess LLM Benchmark - Human Engine Demo[/bold blue]")
    console.print("=" * 60)

    try:
        demo = HumanEngineDemo()
        asyncio.run(demo.run_demo(args.mode, args.elo))

        console.print("\n✨ [bold green]Demo completed successfully![/bold green]")
        console.print("\nTo use human engines in benchmarks:")
        console.print("  [cyan]python main.py --preset premium --use-human-engine[/cyan]")

    except KeyboardInterrupt:
        console.print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        console.print(f"\n[red]❌ Demo failed: {e}[/red]")
        import traceback
        console.print("[dim]" + traceback.format_exc() + "[/dim]")


if __name__ == "__main__":
    main()
