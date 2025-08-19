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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import chess
from rich.console import Console

from .core.models import Config, BotSpec, LiveState, LadderStats, BenchmarkResult
from .core.engine import ChessEngine, autodetect_stockfish, get_friendly_stockfish_hint, create_engine
from .core.human_engine import HumanLikeEngine, get_best_human_engine, get_human_engine_installation_hint
from .core.adaptive_engine import AdaptiveEngine
from .core.game import GameRunner, LadderRunner
from .llm.client import LLMClient, parse_bot_spec
from .llm.models import PRESET_CONFIGS, format_bot_spec_string, print_available_models, get_premium_bot_lineup
from .core.budget import start_budget_tracking, stop_budget_tracking
from .core.results import store_benchmark_results, show_leaderboard, show_provider_comparison, analyze_model
from .ui.dashboard import Dashboard
from .ui.board import render_robot_battle

# Set up logging
# Configure logging to not interfere with live dashboard
def setup_logging(level=logging.INFO):
    """Setup logging that doesn't interfere with Rich live display."""
    # Remove any existing handlers to avoid console output
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Use NullHandler during live display to suppress console output
    null_handler = logging.NullHandler()
    root_logger.addHandler(null_handler)
    root_logger.setLevel(level)

# Initialize with null handler to avoid console interference
setup_logging()
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
        self.budget_tracker = None

        # Runtime state
        self.bots: List[BotSpec] = []
        self.engines: Dict[str, Union[ChessEngine, HumanLikeEngine, AdaptiveEngine]] = {}
        self.clients: Dict[str, LLMClient] = {}
        self.states: Dict[str, LiveState] = {}
        self.stats: Dict[str, LadderStats] = {}

    async def run_benchmark(self) -> BenchmarkResult:
        """
        Run the complete benchmark with all configured bots.

        Returns:
            Complete benchmark results
        """
        # Start budget tracking if enabled
        budget_summary = None
        if self.config.budget_limit or self.config.show_costs:
            self.budget_tracker = start_budget_tracking(self.config.budget_limit)
            console.print(f"[green]💰 Budget tracking enabled[/green]")
            if self.config.budget_limit:
                console.print(f"[yellow]⚠️  Budget limit: ${self.config.budget_limit:.2f}[/yellow]")

        # Parse and validate bot specifications
        try:
            self.bots = parse_bot_spec(self.config.bots)
            if not self.bots:
                raise ValueError("No valid bots specified")
        except Exception as e:
            raise RuntimeError(f"Invalid bot specification: {e}")

        # Determine engine configuration
        if self.config.opponent_type:
            # Use explicitly specified opponent type
            opponent_type = self.config.opponent_type
            engine_path = None

            if opponent_type == "stockfish":
                engine_path = autodetect_stockfish(self.config.stockfish_path)
                if not engine_path:
                    raise RuntimeError(f"Stockfish not found.\n\n{get_friendly_stockfish_hint()}")
            elif opponent_type in ("maia", "lczero"):
                # Will be auto-detected by create_engine
                pass
            elif opponent_type in ("texel", "madchess"):
                # Will be auto-detected by create_engine
                pass

            logger.info(f"Using specified opponent: {opponent_type}")
            await self._initialize_components(engine_path, opponent_type, is_human_engine=(opponent_type in ("maia", "lczero")))
        elif self.config.use_human_engine:
            # Legacy human engine configuration
            if self.config.human_engine_path and self.config.human_engine_type:
                engine_path = self.config.human_engine_path
                engine_type = self.config.human_engine_type
                logger.info(f"Using human-like engine: {engine_type} at {engine_path}")
            else:
                # Auto-detect best human engine
                best_engine = get_best_human_engine()
                if best_engine:
                    engine_type, engine_path = best_engine
                    logger.info(f"Auto-detected human-like engine: {engine_type} at {engine_path}")
                else:
                    if self.config.human_engine_fallback:
                        # Fall back to Stockfish
                        logger.warning("No human-like engines found, falling back to Stockfish")
                        stockfish_path = autodetect_stockfish(self.config.stockfish_path)
                        if not stockfish_path:
                            raise RuntimeError(f"Neither human engines nor Stockfish found.\n\n{get_human_engine_installation_hint()}")
                        engine_path = stockfish_path
                        engine_type = "stockfish"
                    else:
                        raise RuntimeError(f"Human-like engines not found.\n\n{get_human_engine_installation_hint()}")

            await self._initialize_components(engine_path, engine_type, is_human_engine=True)
        else:
            # Use traditional Stockfish
            stockfish_path = autodetect_stockfish(self.config.stockfish_path)
            if not stockfish_path:
                raise RuntimeError(f"Stockfish not found.\n\n{get_friendly_stockfish_hint()}")

            logger.info(f"Using Stockfish: {stockfish_path}")
            await self._initialize_components(stockfish_path, "stockfish", is_human_engine=False)

        # Create output directory
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting benchmark with {len(self.bots)} bots")
        logger.info(f"Output directory: {output_dir}")

        try:
            # Run benchmark with live dashboard
            with self.dashboard.start_live_display():
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
                    await asyncio.sleep(0.1)  # Update every 100ms

                # Wait for all tasks to complete
                await asyncio.gather(*tasks, return_exceptions=True)

                # Final dashboard update
                self.dashboard.update_display(self.states, self.stats)

        finally:
            # Stop budget tracking and get summary
            if self.budget_tracker:
                budget_summary = stop_budget_tracking()

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

        # Store results in database for leaderboard and analysis
        try:
            store_benchmark_results(
                run_id=run_id,
                timestamp=result.timestamp,
                config=self.config.to_dict(),
                results=result,
                budget_summary=budget_summary
            )
            console.print(f"[green]📊 Results stored in database for analysis[/green]")
        except Exception as e:
            logger.warning(f"Failed to store results in database: {e}")

        # Display final cost summary
        if budget_summary and budget_summary.total_cost > 0:
            console.print(f"\n[bold green]💰 Total benchmark cost: ${budget_summary.total_cost:.4f}[/bold green]")
            if self.config.budget_limit:
                percentage = (budget_summary.total_cost / self.config.budget_limit) * 100
                console.print(f"[cyan]📊 Budget usage: {percentage:.1f}%[/cyan]")

        # Display final results
        self.dashboard.display_final_results(result)

        return result

    async def _initialize_components(self, engine_path: str, engine_type: str, is_human_engine: bool) -> None:
        """Initialize all components for the benchmark."""
        for bot in self.bots:
            # Initialize LLM client
            try:
                # Check if agent mode is enabled
                use_agent = getattr(self.config, 'use_agent', False)
                agent_strategy = getattr(self.config, 'agent_strategy', 'balanced')
                verbose_agent = getattr(self.config, 'verbose_agent', False)

                client = LLMClient(
                    bot,
                    use_agent=use_agent,
                    agent_strategy=agent_strategy,
                    verbose_agent=verbose_agent
                )

                if use_agent:
                    logger.info(f"Initialized agent-based LLM client: {bot} (strategy: {agent_strategy})")
                else:
                    logger.info(f"Initialized LLM client: {bot}")

                self.clients[bot.name] = client
            except Exception as e:
                logger.error(f"Failed to initialize client for {bot.name}: {e}")
                raise

            # Initialize chess engine (one per bot for isolation)
            try:
                # Create appropriate engine using factory
                engine = create_engine(self.config, engine_path, engine_type)
                await engine.start()
                self.engines[bot.name] = engine

                # Log the engine type that was started
                actual_engine_type = (
                    engine.current_engine_type if isinstance(engine, AdaptiveEngine)
                    else getattr(engine, 'engine_type', engine_type)
                )
                logger.debug(f"Started {actual_engine_type} engine for {bot.name}")
            except Exception as e:
                logger.error(f"Failed to start {engine_type} engine for {bot.name}: {e}")
                raise

            # Initialize state tracking
            self.states[bot.name] = LiveState(title=bot.name)
            self.stats[bot.name] = LadderStats()

    async def _run_bot_ladder(self, bot_name: str, output_dir: Path) -> None:
        """Run the complete ladder for a single bot."""
        try:
            logger.info(f"Starting ladder run for {bot_name}")
            client = self.clients[bot_name]
            engine = self.engines[bot_name]
            state = self.states[bot_name]
            bot_stats = self.stats[bot_name]

            # Create game and ladder runners
            game_runner = GameRunner(client, engine, self.config)
            ladder_runner = LadderRunner(game_runner, self.config)

            # Run the ladder with real-time stats updates
            max_elo, games = await ladder_runner.run_ladder(output_dir, state, bot_stats)

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

  # Custom ELO range with sub-1100 support
  %(prog)s --preset budget --start-elo 600 --max-elo 1600 --elo-step 100

  # Play against fixed opponent strengths
  %(prog)s --preset premium --fixed-opponent-elo 1200
  %(prog)s --preset budget --fixed-opponent-elo random
  %(prog)s --bots "openai:gpt-4o:GPT-4o" --fixed-opponent-elo 800

🧠 Human-like Engine Examples:
  # Use Maia (most human-like, auto-detected)
  %(prog)s --preset premium --use-human-engine

  # Specify Maia engine type explicitly
  %(prog)s --preset budget --use-human-engine --human-engine-type maia

  # Use Leela Chess Zero for human-like play
  %(prog)s --preset openai --use-human-engine --human-engine-type lczero

  # Human-like Stockfish (fallback option)
  %(prog)s --preset anthropic --use-human-engine --human-engine-type human_stockfish

  # Custom Maia path
  %(prog)s --preset premium --use-human-engine --human-engine-path /usr/local/bin/maia

🤖 Bot specification format: "provider:model:name"
  • provider: openai, anthropic, gemini, random
  • model: exact model ID (use --list-models to see available)
  • name: display name for the bot

📋 Available presets: premium, budget, recommended, openai, anthropic, gemini

🏆 Human-like Engines (More Realistic Opponents):
  • Maia: Neural network trained on human games (most human-like, ELO 600+)
  • LCZero: Neural network with human-like configuration
  • Human Stockfish: Traditional Stockfish with human-like settings

  Install Maia: https://github.com/CSSLab/maia-chess
  Install LCZero: brew install lc0 (macOS) or https://lczero.org/

🎯 Sub-1100 ELO Support:
  The system supports ELO ratings from 600-2400! For best results at sub-1100 levels:
  • Install specialized engines: Texel, MadChess, Fruit, or Crafty
  • Use --use-human-engine for more realistic beginner play
  • Maia-1100 model will be used for ELO < 1100 when available
  • WARNING: ELO values below 600 will be automatically corrected to 600

💰 Budget & Analysis Commands:
  # Track spending with $5 budget limit
  %(prog)s --preset premium --budget-limit 5.0 --show-costs

  # Show leaderboard of best performing models
  %(prog)s --leaderboard 10

  # Compare provider performance
  %(prog)s --provider-stats

  # Analyze specific model
  %(prog)s --analyze-model openai:gpt-4o
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

    # Ranking and analysis commands
    parser.add_argument(
        "--leaderboard",
        type=int,
        nargs="?",
        const=20,
        help="Show model leaderboard (default: top 20)"
    )
    parser.add_argument(
        "--provider-stats",
        action="store_true",
        help="Show provider performance comparison"
    )
    parser.add_argument(
        "--analyze-model",
        type=str,
        help="Analyze specific model performance (format: provider:model)"
    )

    # Budget tracking
    parser.add_argument(
        "--budget-limit",
        type=float,
        help="Set budget limit in USD (enables cost tracking and warnings)"
    )
    parser.add_argument(
        "--show-costs",
        action="store_true",
        help="Display detailed cost breakdown during and after benchmark"
    )

    # Engine configuration
    parser.add_argument(
        "--stockfish",
        type=str,
        help="Path to Stockfish executable (auto-detected if not specified)",
        dest="stockfish_path",
    )

    parser.add_argument(
        "--opponent",
        type=str,
        choices=["stockfish", "maia", "texel", "madchess"],
        help="Type of chess engine to use as opponent (default: stockfish)",
    )

    parser.add_argument(
        "--opponent-elo",
        type=int,
        help="Explicit ELO rating for the opponent engine (minimum: 600, overrides --start-elo)",
    )

    parser.add_argument(
        "--fixed-opponent-elo",
        type=str,
        choices=["random", "600", "800", "1000", "1200", "1400"],
        help="Play against a fixed opponent strength instead of climbing a ladder. Available options: random (random legal moves), 600, 800, 1000, 1200, 1400 ELO",
    )

    parser.add_argument(
        "--movetime-ms",
        type=int,
        default=300,
        help="Fixed time in milliseconds per move for both sides (default: %(default)s)",
    )

    parser.add_argument(
        "--adaptive-engines",
        dest="adaptive_elo_engines",
        action="store_true",
        help="Use different engines for different ELO ranges (default: enabled)",
    )

    parser.add_argument(
        "--no-adaptive-engines",
        dest="adaptive_elo_engines",
        action="store_false",
        help="Don't use different engines for different ELO ranges",
    )
    parser.set_defaults(adaptive_elo_engines=True)

    # Human-like engine configuration
    parser.add_argument(
        "--use-human-engine",
        action="store_true",
        help="Use human-like chess engines instead of Stockfish (more realistic opponents)"
    )
    parser.add_argument(
        "--human-engine-type",
        type=str,
        choices=["maia", "lczero", "human_stockfish"],
        default="maia",
        help="Type of human-like engine (default: %(default)s)"
    )
    parser.add_argument(
        "--human-engine-path",
        type=str,
        help="Path to human-like engine executable (auto-detected if not specified)"
    )
    parser.add_argument(
        "--no-human-engine-fallback",
        action="store_true",
        help="Don't fall back to Stockfish if human engines aren't available"
    )

    # ELO ladder settings
    parser.add_argument(
        "--start-elo",
        type=int,
        default=600,
        help="Starting ELO rating (minimum: 600, default: %(default)s)"
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
        help="Maximum ELO rating (minimum: 600, default: %(default)s)"
    )

    # Game settings
    parser.add_argument(
        "--think-time",
        type=float,
        default=0.3,
        help="Engine thinking time per move in seconds (default: %(default)s, overridden by --movetime-ms if specified)"
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

    # Agent mode settings
    parser.add_argument(
        "--use-agent",
        action="store_true",
        help="Use agent-based reasoning with tools instead of simple prompting"
    )
    parser.add_argument(
        "--agent-strategy",
        type=str,
        choices=["fast", "balanced", "deep", "adaptive"],
        default="balanced",
        help="Agent thinking strategy (default: %(default)s)"
    )
    parser.add_argument(
        "--verbose-agent",
        action="store_true",
        help="Show detailed agent reasoning process"
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


def validate_elo_arguments(args) -> None:
    """Validate ELO arguments and warn about values below minimum threshold."""
    MIN_ELO = 600

    def warn_low_elo(elo_value: int, arg_name: str) -> int:
        """Warn about low ELO and return corrected value."""
        if elo_value < MIN_ELO:
            logger.warning(f"⚠️  {arg_name} {elo_value} is below absolute minimum of {MIN_ELO} ELO")
            logger.warning(f"   Automatically correcting {arg_name} to {MIN_ELO}")
            return MIN_ELO
        return elo_value

    # Validate start_elo
    if hasattr(args, 'start_elo') and args.start_elo is not None:
        args.start_elo = warn_low_elo(args.start_elo, "--start-elo")

    # Validate opponent_elo
    if hasattr(args, 'opponent_elo') and args.opponent_elo is not None:
        args.opponent_elo = warn_low_elo(args.opponent_elo, "--opponent-elo")

    # Validate max_elo (less critical but still check)
    if hasattr(args, 'max_elo') and args.max_elo is not None and args.max_elo < MIN_ELO:
        logger.warning(f"⚠️  --max-elo {args.max_elo} is below absolute minimum of {MIN_ELO} ELO")
        logger.warning(f"   Setting max-elo to {MIN_ELO} (same as start-elo)")
        args.max_elo = MIN_ELO


def configure_logging(verbose: bool = False, debug: bool = False, suppress_console: bool = False) -> None:
    """Configure logging based on command-line options."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    # Configure root logger
    root_logger = logging.getLogger()

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if not suppress_console:
        # Add console handler for normal operation
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        root_logger.addHandler(console_handler)
    else:
        # Use null handler to suppress all console output during live display
        null_handler = logging.NullHandler()
        root_logger.addHandler(null_handler)

    root_logger.setLevel(level)

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
    # Configure logging (suppress console during live display)
    configure_logging(args.verbose, args.debug, suppress_console=True)

    # Validate ELO arguments and warn about values below minimum
    validate_elo_arguments(args)

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

    # Handle ranking and analysis commands
    if args.leaderboard is not None:
        show_leaderboard(args.leaderboard)
        return 0

    if args.provider_stats:
        show_provider_comparison()
        return 0

    if args.analyze_model:
        analyze_model(args.analyze_model)
        return 0

    # Handle demo mode
    if args.demo:
        console.print("[bold yellow]Running demo mode[/bold yellow]")
        args.bots = "random::demo1,random::demo2"
        # Only set demo ELO values if they weren't explicitly changed from defaults
        if args.start_elo == 600:  # Default value wasn't changed
            args.start_elo = 1400
        if args.elo_step == 100:  # Default value wasn't changed
            args.elo_step = 200
        if args.max_elo == 2400:  # Default value wasn't changed
            args.max_elo = 1600

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

    # Handle movetime-ms override for think_time if specified
    if hasattr(args, 'movetime_ms') and args.movetime_ms:
        think_time = args.movetime_ms / 1000.0
        logger.info(f"Using fixed move time of {args.movetime_ms}ms ({think_time}s)")
    else:
        think_time = args.think_time

    # Handle fixed opponent ELO
    fixed_opponent_elo = None
    if hasattr(args, 'fixed_opponent_elo') and args.fixed_opponent_elo:
        if args.fixed_opponent_elo == "random":
            fixed_opponent_elo = 0  # Special value for random moves
        else:
            fixed_opponent_elo = int(args.fixed_opponent_elo)
        console.print(f"[green]Using fixed opponent: {args.fixed_opponent_elo}[/green]")

    # Create configuration
    config = Config(
        bots=bots_string,
        stockfish_path=args.stockfish_path,
        use_human_engine=args.use_human_engine,
        human_engine_type=args.human_engine_type,
        human_engine_path=args.human_engine_path,
        human_engine_fallback=not args.no_human_engine_fallback,
        start_elo=args.opponent_elo if hasattr(args, 'opponent_elo') and args.opponent_elo else args.start_elo,
        elo_step=args.elo_step,
        max_elo=args.max_elo,
        fixed_opponent_elo=fixed_opponent_elo,
        think_time=think_time,
        max_plies=args.max_plies,
        escalate_on=args.escalate_on,
        llm_timeout=args.llm_timeout,
        llm_temperature=args.llm_temperature,
        output_dir=args.output_dir,
        save_pgn=not args.no_pgn,
        refresh_rate=args.refresh_rate,
        opponent_type=args.opponent if hasattr(args, 'opponent') else None,
        use_agent=args.use_agent,
        agent_strategy=args.agent_strategy,
        verbose_agent=args.verbose_agent
    )

    # Add budget tracking configuration
    config.budget_limit = args.budget_limit
    config.show_costs = args.show_costs

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

    # Configure engine settings
    config.adaptive_elo_engines = True
    config.use_human_engine = True

    # Create two robots
    white_bot = LLMClient(parse_bot_spec("random::WhiteBot")[0])
    black_bot = LLMClient(parse_bot_spec("random::BlackBot")[0])

    # Create adaptive engine
    try:
        engine = create_engine(config)
        await engine.start()
        logger.info(f"Using engine: {engine.current_engine_type if isinstance(engine, AdaptiveEngine) else 'stockfish'}")
    except Exception as e:
        console.print(f"[red]Error: Failed to initialize chess engine: {e}[/red]")
        return 1

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
