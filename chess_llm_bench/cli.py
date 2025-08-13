"""
Command-line interface for the Chess LLM Benchmark.

This module provides the main entry point and argument parsing for the chess
LLM benchmark tool, coordinating all components to run ELO ladder tests.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import chess
from rich.console import Console

from .core.models import Config, BotSpec, LiveState, LadderStats, BenchmarkResult
from .core.engine import ChessEngine, autodetect_stockfish, get_friendly_stockfish_hint
from .core.game import GameRunner, LadderRunner
from .llm.client import LLMClient, parse_bot_spec
from .llm.models import PRESET_CONFIGS, format_bot_spec_string, print_available_models, get_premium_bot_lineup
from .ui.dashboard import Dashboard
from .ui.board import render_robot_battle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()


class BenchmarkOrchestrator:
    """
    Main orchestrator for running chess LLM benchmarks.

    Coordinates engines, LLM clients, game runners, and UI components
    to execute complete benchmark runs with multiple bots.
    """

    def __init__(self, config: Config):
        """Initialize the benchmark orchestrator."""
        self.config = config
        self.dashboard = Dashboard(console, config)

        # Runtime state
        self.bots: List[BotSpec] = []
        self.engines: Dict[str, ChessEngine] = {}
        self.clients: Dict[str, LLMClient] = {}
        self.states: Dict[str, LiveState] = {}
        self.stats: Dict[str, LadderStats] = {}

    async def run_benchmark(self) -> BenchmarkResult:
        """
        Run the complete benchmark with all configured bots.

        Returns:
            Complete benchmark results
        """
        # Parse and validate bot specifications
        try:
            self.bots = parse_bot_spec(self.config.bots)
            if not self.bots:
                raise ValueError("No valid bots specified")
        except Exception as e:
            raise RuntimeError(f"Invalid bot specification: {e}")

        # Validate Stockfish
        stockfish_path = autodetect_stockfish(self.config.stockfish_path)
        if not stockfish_path:
            raise RuntimeError(f"Stockfish not found.\n\n{get_friendly_stockfish_hint()}")

        logger.info(f"Using Stockfish: {stockfish_path}")

        # Initialize components
        await self._initialize_components(stockfish_path)

        # Create output directory
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting benchmark with {len(self.bots)} bots")
        logger.info(f"Output directory: {output_dir}")

        try:
            # Run benchmark with live dashboard
            with self.dashboard.start_live_display() as live_display:
                # Start ladder runs for all bots
                tasks = []
                for bot in self.bots:
                    task = asyncio.create_task(
                        self._run_bot_ladder(bot.name, output_dir)
                    )
                    tasks.append(task)

                # Update dashboard while games run
                while any(not task.done() for task in tasks):
                    self.dashboard.update_display(self.states, self.stats)
                    await asyncio.sleep(0.2)

                # Wait for all tasks to complete
                await asyncio.gather(*tasks, return_exceptions=True)

                # Final dashboard update
                self.dashboard.update_display(self.states, self.stats)

        finally:
            # Cleanup
            await self._cleanup_components()

        # Create final results
        result = BenchmarkResult(
            run_id=run_id,
            timestamp=datetime.utcnow(),
            config=self.config,
            bot_results=self.stats.copy(),
            output_dir=output_dir
        )

        # Display final results
        self.dashboard.display_final_results(result)

        return result

    async def _initialize_components(self, stockfish_path: str) -> None:
        """Initialize all components for the benchmark."""
        for bot in self.bots:
            # Initialize LLM client
            try:
                client = LLMClient(bot)
                self.clients[bot.name] = client
                logger.info(f"Initialized LLM client: {bot}")
            except Exception as e:
                logger.error(f"Failed to initialize client for {bot.name}: {e}")
                raise

            # Initialize chess engine (one per bot for isolation)
            try:
                engine = ChessEngine(stockfish_path, self.config)
                await engine.start()
                self.engines[bot.name] = engine
                logger.debug(f"Started engine for {bot.name}")
            except Exception as e:
                logger.error(f"Failed to start engine for {bot.name}: {e}")
                raise

            # Initialize state tracking
            self.states[bot.name] = LiveState(title=bot.name)
            self.stats[bot.name] = LadderStats()

    async def _run_bot_ladder(self, bot_name: str, output_dir: Path) -> None:
        """Run the complete ladder for a single bot."""
        try:
            client = self.clients[bot_name]
            engine = self.engines[bot_name]
            state = self.states[bot_name]
            bot_stats = self.stats[bot_name]

            # Create game and ladder runners
            game_runner = GameRunner(client, engine, self.config)
            ladder_runner = LadderRunner(game_runner, self.config)

            # Run the ladder
            max_elo, games = await ladder_runner.run_ladder(output_dir, state)

            # Update statistics
            for game in games:
                bot_stats.add_game(game)

            logger.info(f"{bot_name} completed ladder: max ELO {max_elo}")

        except Exception as e:
            logger.error(f"Ladder run failed for {bot_name}: {e}")
            self.states[bot_name].set_error(str(e))

    async def _cleanup_components(self) -> None:
        """Clean up all components."""
        for bot_name, engine in self.engines.items():
            try:
                await engine.stop()
                logger.debug(f"Stopped engine for {bot_name}")
            except Exception as e:
                logger.warning(f"Error stopping engine for {bot_name}: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="🏆 Chess LLM ELO Ladder Benchmark - Test LLMs with chess games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 Quick Start Examples:
  # Premium models (best from each provider)
  %(prog)s --preset premium

  # Budget-friendly models
  %(prog)s --preset budget

  # All OpenAI models
  %(prog)s --preset openai

  # Demo with random bots
  %(prog)s --demo

  # Custom bot lineup
  %(prog)s --bots "openai:gpt-4o:GPT-4o,anthropic:claude-3-5-sonnet-20241022:Claude-3.5-Sonnet"

  # Custom ELO range
  %(prog)s --preset budget --start-elo 800 --max-elo 1600 --elo-step 200

🤖 Bot specification format: "provider:model:name"
  • provider: openai, anthropic, gemini, random
  • model: exact model ID (use --list-models to see available)
  • name: display name for the bot

📋 Available presets: premium, budget, recommended, openai, anthropic, gemini
        """
    )

    # Bot configuration (mutually exclusive with preset)
    bot_group = parser.add_mutually_exclusive_group()
    bot_group.add_argument(
        "--bots",
        type=str,
        help="Comma-separated bot specs: provider:model:name"
    )
    bot_group.add_argument(
        "--preset",
        type=str,
        choices=list(PRESET_CONFIGS.keys()),
        help="Use a predefined set of bots (premium, budget, recommended, openai, anthropic, gemini)"
    )

    # Information commands
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit"
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List all available presets and exit"
    )

    # Engine configuration
    parser.add_argument(
        "--stockfish",
        type=str,
        help="Path to Stockfish executable (auto-detected if not specified)"
    )

    # ELO ladder settings
    parser.add_argument(
        "--start-elo",
        type=int,
        default=600,
        help="Starting ELO rating (default: %(default)s)"
    )
    parser.add_argument(
        "--elo-step",
        type=int,
        default=100,
        help="ELO increment per ladder rung (default: %(default)s)"
    )
    parser.add_argument(
        "--max-elo",
        type=int,
        default=2400,
        help="Maximum ELO rating (default: %(default)s)"
    )

    # Game settings
    parser.add_argument(
        "--think-time",
        type=float,
        default=0.3,
        help="Engine thinking time per move in seconds (default: %(default)s)"
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=300,
        help="Maximum half-moves per game (default: %(default)s)"
    )
    parser.add_argument(
        "--escalate-on",
        type=str,
        default="always",
        choices=["always", "on_win"],
        help="Advance ELO ladder: always or only on wins (default: %(default)s)"
    )

    # LLM settings
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=20.0,
        help="LLM response timeout in seconds (default: %(default)s)"
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature (default: %(default)s)"
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="Output directory for results (default: %(default)s)"
    )
    parser.add_argument(
        "--no-pgn",
        action="store_true",
        help="Skip saving PGN files"
    )

    # UI settings
    parser.add_argument(
        "--refresh-rate",
        type=int,
        default=6,
        help="Dashboard refresh rate in Hz (default: %(default)s)"
    )

    # Special modes
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run quick demo (random bots, ELO 600-800 by 100)"
    )
    parser.add_argument(
        "--robot-demo",
        action="store_true",
        help="Watch two robots play chess with beautiful board visualization"
    )
    parser.add_argument(
        "--quick-robot-demo",
        action="store_true",
        help="Quick robot demo with faster gameplay (no delays)"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run internal unit tests and exit"
    )

    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    return parser


def configure_logging(verbose: bool = False, debug: bool = False) -> None:
    """Configure logging based on command-line options."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    # Configure root logger
    logging.getLogger().setLevel(level)

    # Reduce noise from some libraries
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("chess.engine").setLevel(logging.WARNING)


def run_unit_tests() -> int:
    """Run built-in unit tests."""
    import unittest
    import sys
    from pathlib import Path

    # Get the project root directory
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"

    if not tests_dir.exists():
        print("No tests directory found", file=sys.stderr)
        return 1

    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_dir), pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


async def main_async(args: argparse.Namespace) -> int:
    """Main async entry point."""
    # Configure logging
    configure_logging(args.verbose, args.debug)

    # Handle information commands
    if args.list_models:
        print_available_models()
        return 0

    if args.list_presets:
        console.print("\n🎯 Available Presets\n")
        for preset_name, preset_info in PRESET_CONFIGS.items():
            console.print(f"[bold cyan]{preset_name}[/bold cyan]")
            console.print(f"  {preset_info['description']}")
            console.print(f"  Models: {len(preset_info['bots'])}")
            for bot in preset_info['bots']:
                console.print(f"    • {bot.name} ({bot.provider}:{bot.model})")
            console.print()
        return 0

    # Handle demo mode
    if args.demo:
        console.print("[bold yellow]Running demo mode[/bold yellow]")
        args.bots = "random::demo1,random::demo2"
        args.start_elo = 600
        args.elo_step = 100
        args.max_elo = 800

    # Handle robot demo mode
    if args.robot_demo:
        console.print("[bold cyan]🤖 Starting Robot Chess Battle! 🤖[/bold cyan]")
        return await robot_demo_mode(args, quick=False)

    # Handle quick robot demo mode
    if args.quick_robot_demo:
        console.print("[bold cyan]🚀 Quick Robot Chess Battle! 🚀[/bold cyan]")
        return await robot_demo_mode(args, quick=True)

    # Determine bot configuration
    if args.preset:
        if args.preset not in PRESET_CONFIGS:
            console.print(f"[red]Error: Unknown preset '{args.preset}'[/red]")
            return 1
        bot_specs = PRESET_CONFIGS[args.preset]["bots"]
        bots_string = format_bot_spec_string(bot_specs)
        console.print(f"[green]Using preset '{args.preset}': {PRESET_CONFIGS[args.preset]['description']}[/green]")
    elif args.bots:
        bots_string = args.bots
    else:
        # Default to premium preset
        console.print("[yellow]No bots specified, using premium preset[/yellow]")
        bot_specs = get_premium_bot_lineup()
        bots_string = format_bot_spec_string(bot_specs)

    # Create configuration
    config = Config(
        bots=bots_string,
        stockfish_path=args.stockfish,
        start_elo=args.start_elo,
        elo_step=args.elo_step,
        max_elo=args.max_elo,
        think_time=args.think_time,
        max_plies=args.max_plies,
        escalate_on=args.escalate_on,
        llm_timeout=args.llm_timeout,
        llm_temperature=args.llm_temperature,
        output_dir=args.output_dir,
        save_pgn=not args.no_pgn,
        refresh_rate=args.refresh_rate
    )

    # Create and run benchmark
    try:
        orchestrator = BenchmarkOrchestrator(config)
        result = await orchestrator.run_benchmark()

        console.print(f"\n[bold green]Benchmark completed successfully![/bold green]")
        console.print(f"Results saved to: [bold]{result.output_dir}[/bold]")

        return 0

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Benchmark interrupted by user[/bold yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[bold red]Benchmark failed: {e}[/bold red]")
        logger.exception("Benchmark failed with exception")
        return 1


async def robot_demo_mode(args: argparse.Namespace, quick: bool = False) -> int:
    """
    Run a beautiful robot vs robot demo with live chess board visualization.
    """

    console.print("\n[bold cyan]Setting up robot chess battle...[/bold cyan]")

    # Configure for robot demo
    if quick:
        config = Config(
            think_time=0.1,  # Fast gameplay
            max_plies=60,    # Shorter games
            llm_timeout=5.0,
            save_pgn=True
        )
    else:
        config = Config(
            think_time=1.5,  # Slower for dramatic effect
            max_plies=100,
            llm_timeout=10.0,
            save_pgn=True
        )

    # Find Stockfish
    stockfish_path = autodetect_stockfish()
    if not stockfish_path:
        console.print(f"[red]Error: {get_friendly_stockfish_hint()}[/red]")
        return 1

    # Create two robots
    white_bot = LLMClient(parse_bot_spec("random::WhiteBot")[0])
    black_bot = LLMClient(parse_bot_spec("random::BlackBot")[0])

    # Create engine
    engine = ChessEngine(stockfish_path, config)
    await engine.start()

    # Initialize game
    board = chess.Board()
    moves_played = []
    game_active = True

    if quick:
        console.print("\n[bold green]⚡ Quick Robot Chess Battle Starting![/bold green]")
        console.print("[yellow]Press Ctrl+C to stop the demo[/yellow]\n")
    else:
        console.print("\n[bold green]🎮 Robot Chess Battle Starting![/bold green]")
        console.print("[yellow]Press Ctrl+C to stop the demo[/yellow]\n")

    try:
        move_count = 0
        max_moves = 30 if quick else 50
        while game_active and not board.is_game_over() and move_count < max_moves:
            # Clear screen for animation effect
            console.clear()

            # Determine current player
            current_bot = white_bot if board.turn == chess.WHITE else black_bot
            bot_name = "WhiteBot" if board.turn == chess.WHITE else "BlackBot"
            opponent_name = "BlackBot" if board.turn == chess.WHITE else "WhiteBot"

            # Show current position
            last_move = moves_played[-1] if moves_played else None
            battle_panel = render_robot_battle(
                board=board,
                white_bot="🤖 WhiteBot",
                black_bot="🤖 BlackBot",
                last_move=last_move,
                engine_elo=None,
                moves=moves_played
            )

            console.print(battle_panel)

            if board.is_game_over():
                break

            if not quick:
                # Show thinking animation
                status = f"🤔 {bot_name} is thinking..."
                console.print(f"\n[bold yellow]{status}[/bold yellow]")

                # Add dramatic pause
                await asyncio.sleep(2.0)
            else:
                # Quick mode - minimal delay
                await asyncio.sleep(0.1)

            # Get move from current bot
            try:
                move = await current_bot.pick_move(board, temperature=0.3)
                # Get SAN notation BEFORE pushing the move
                move_san = board.san(move) if move in board.legal_moves else move.uci()
                moves_played.append(move)
                board.push(move)
                move_count += 1

                # Show the move
                if quick:
                    console.print(f"[bold green]⚡ {bot_name}: {move_san}[/bold green]")
                    await asyncio.sleep(0.2)
                else:
                    console.print(f"[bold green]⚡ {bot_name} plays: {move_san}[/bold green]")
                    await asyncio.sleep(1.0)

            except Exception as e:
                console.print(f"[red]Error: {bot_name} failed to make a move: {e}[/red]")
                break

        # Final position
        console.clear()
        final_panel = render_robot_battle(
            board=board,
            white_bot="🤖 WhiteBot",
            black_bot="🤖 BlackBot",
            last_move=moves_played[-1] if moves_played else None,
            moves=moves_played
        )
        console.print(final_panel)

        # Game result
        result = board.result()
        if result == "1-0":
            console.print("\n[bold green]🏆 WhiteBot wins![/bold green]")
        elif result == "0-1":
            console.print("\n[bold green]🏆 BlackBot wins![/bold green]")
        elif result == "1/2-1/2":
            console.print("\n[bold yellow]🤝 It's a draw![/bold yellow]")
        else:
            console.print("\n[bold blue]🎮 Demo ended[/bold blue]")

        console.print(f"\n[dim]Total moves: {len(moves_played)}[/dim]")

    except KeyboardInterrupt:
        console.print("\n[bold yellow]⏸️  Demo stopped by user[/bold yellow]")
    finally:
        await engine.stop()

    return 0


def main() -> int:

    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Handle self-test mode
    if args.self_test:
        return run_unit_tests()

    # Run main benchmark
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user[/bold yellow]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
