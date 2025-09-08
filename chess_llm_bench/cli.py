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
from .llm.models import PRESET_CONFIGS, format_bot_spec_string, print_available_models, get_latest_bot_lineup
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
  # Use latest models (default)
  %(prog)s
  
  # Use latest models with cost tracking
  %(prog)s --budget-limit 5.0 --show-costs

  # Use legacy models
  %(prog)s --preset legacy

  # Use agent-based reasoning instead of simple prompts
  %(prog)s --use-agent

  # Play against lowest ELO opponent instead of random moves
  %(prog)s --opponent lowest-elo

  # Custom bot lineup
  %(prog)s --bots "openai:gpt-4o:GPT-4o,anthropic:claude-3-5-sonnet:Claude-3.5-Sonnet"

💰 Budget & Analysis Commands:
  # Track spending with budget limit
  %(prog)s --budget-limit 5.0 --show-costs

  # Show leaderboard of best performing models
  %(prog)s --leaderboard 10

🤖 Bot specification format: "provider:model:name"
  • provider: openai, anthropic, gemini
  • model: exact model ID (use --list-models to see available)
  • name: display name for the bot

📋 Available presets: latest (default), legacy

🎮 Opponent Options:
  • random: Plays random legal moves (default)
  • lowest-elo: Plays at 600 ELO strength

🧠 Playing Modes:
  • Prompt-based: Simple LLM prompting (default)
  • Agent-based: Tool-based reasoning with analysis (--use-agent)
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
        default="latest",
        help="Use a predefined set of bots (default: latest)"
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

    # Opponent configuration
    parser.add_argument(
        "--opponent",
        type=str,
        choices=["random", "lowest-elo"],
        default="random",
        help="Choose opponent: random (random legal moves) or lowest-elo (600 ELO) (default: random)"
    )

    # Playing mode configuration
    parser.add_argument(
        "--use-agent",
        action="store_true",
        help="Use agent-based reasoning with tools instead of simple prompting (default: prompt-based)"
    )


    # Game settings
    parser.add_argument(
        "--max-games",
        type=int,
        default=10,
        help="Maximum number of games to play per model (default: %(default)s)"
    )





    return parser








async def main_async(args: argparse.Namespace) -> int:
    """Main async entry point."""
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

    # Determine bot configuration
    if args.bots:
        bots_string = args.bots
        console.print(f"[green]Using custom bots[/green]")
    else:
        # Use preset (latest is default)
        preset_name = args.preset if hasattr(args, 'preset') and args.preset else "latest"
        if preset_name not in PRESET_CONFIGS:
            console.print(f"[red]Error: Unknown preset '{preset_name}'[/red]")
            return 1
        bot_specs = PRESET_CONFIGS[preset_name]["bots"]
        bots_string = format_bot_spec_string(bot_specs)
        console.print(f"[green]Using '{preset_name}' models: {PRESET_CONFIGS[preset_name]['description']}[/green]")

    # Handle opponent selection
    opponent_type = args.opponent if hasattr(args, 'opponent') else "random"
    if opponent_type == "random":
        fixed_opponent_elo = 0  # Special value for random moves
        console.print(f"[green]Using opponent: Random moves[/green]")
    elif opponent_type == "lowest-elo":
        fixed_opponent_elo = 600  # Lowest ELO
        console.print(f"[green]Using opponent: 600 ELO engine[/green]")
    else:
        fixed_opponent_elo = 0  # Default to random
        console.print(f"[green]Using opponent: Random moves (default)[/green]")

    # Handle agent mode
    use_agent = args.use_agent if hasattr(args, 'use_agent') else False
    max_games = args.max_games if hasattr(args, 'max_games') else 10
    agent_mode_str = "agent-based reasoning" if use_agent else "prompt-based"
    console.print(f"[green]Using {agent_mode_str}[/green]")
    
    # Create configuration
    config = Config(
        bots=bots_string,
        fixed_opponent_elo=fixed_opponent_elo,
        use_agent=use_agent,
        # Set simple defaults for required fields
        start_elo=600,
        elo_step=100, 
        max_elo=2400,
        think_time=1.0,
        max_plies=200,
        llm_timeout=30.0,
        llm_temperature=0.0,
        output_dir="runs",
        save_pgn=True,
        escalate_on="always",
        agent_strategy="balanced",
        verbose_agent=False,
        refresh_rate=6
    )

    # Add budget tracking configuration  
    config.budget_limit = args.budget_limit if hasattr(args, 'budget_limit') else None
    config.show_costs = args.show_costs if hasattr(args, 'show_costs') else False

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




def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Run main benchmark
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user[/bold yellow]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
