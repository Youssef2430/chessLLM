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
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Union

from rich.console import Console

from .core.models import Config, BotSpec, LiveState, LadderStats, BenchmarkResult
from .core.engine import ChessEngine, autodetect_stockfish, get_friendly_stockfish_hint
from .core.human_engine import HumanLikeEngine
from .core.adaptive_engine import AdaptiveEngine
from .core.game import GameRunner, LadderRunner
from .llm.client import LLMClient, parse_bot_spec
from .llm.models import PRESET_CONFIGS, format_bot_spec_string, print_available_models
from .core.budget import start_budget_tracking, stop_budget_tracking
from .core.results import (
    ResultsDatabase,
    create_benchmark_record,
    show_leaderboard,
    show_provider_comparison,
    analyze_model,
)
from .core.estimate import estimate_run
from .core.budget import UnknownPricing
from .ui.dashboard import Dashboard


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
        self.engines: Dict[str, Union[ChessEngine, HumanLikeEngine, AdaptiveEngine]] = (
            {}
        )
        self.clients: Dict[str, LLMClient] = {}
        self.states: Dict[str, LiveState] = {}
        self.stats: Dict[str, LadderStats] = {}

    async def run_benchmark(self) -> BenchmarkResult:
        if self.config.use_human_engine or self.config.adaptive_elo_engines:
            raise ValueError(
                "Protocol v2 requires random or native UCI_Elo Stockfish; experimental engine adapters are not calibrated"
            )
        self.bots = parse_bot_spec(self.config.bots)
        if not self.bots:
            raise ValueError("No bots specified")
        self.budget_tracker = start_budget_tracking(self.config.budget_limit)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output_dir = Path(self.config.output_dir) / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        tasks = []
        run_error = None
        (output_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )
        result = BenchmarkResult(
            run_id, datetime.now(timezone.utc), self.config, self.stats, output_dir
        )
        handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
        logging.getLogger().addHandler(handler)
        try:
            # Validate every model before initializing clients or issuing requests.
            for bot in self.bots:
                if self.budget_tracker.get_pricing(bot.provider, bot.model) is None:
                    raise UnknownPricing(
                        f"No verified price for {bot.provider}:{bot.model}"
                    )
            engine_path = None
            if self.config.fixed_opponent_elo != 0:
                engine_path = autodetect_stockfish(self.config.stockfish_path)
                if not engine_path:
                    raise RuntimeError(get_friendly_stockfish_hint())
            await self._initialize_components(engine_path)
            (output_dir / "config.json").write_text(
                json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
            )
            from importlib.metadata import version, PackageNotFoundError

            dependencies = {}
            for package in (
                "python-chess",
                "openai",
                "anthropic",
                "google-genai",
                "rich",
            ):
                try:
                    dependencies[package] = version(package)
                except PackageNotFoundError:
                    pass
            (output_dir / "environment.json").write_text(
                json.dumps(
                    {
                        "python": platform.python_version(),
                        "platform": platform.platform(),
                        "dependencies": dependencies,
                        "engines": {
                            name: {
                                "name": engine._engine_name,
                                "path": engine.engine_path,
                                "elo_range": engine._supported_elo_range,
                            }
                            for name, engine in self.engines.items()
                            if engine is not None
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            with self.dashboard.start_live_display():
                tasks = [
                    asyncio.create_task(self._run_bot_ladder(bot.name, output_dir))
                    for bot in self.bots
                ]
                while any(not task.done() for task in tasks):
                    self.dashboard.update_display(self.states, self.stats)
                    await asyncio.sleep(0.1)
                await asyncio.gather(*tasks)
                self.dashboard.update_display(self.states, self.stats)
        except BaseException as exc:
            run_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Run interrupted or failed")
            raise
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await self._cleanup_components()
            budget_summary = stop_budget_tracking()
            self.budget_tracker.save_budget_report(output_dir / "budget.json")
            summary = {
                "run_id": run_id,
                "protocol_version": self.config.protocol_version,
                "config": self.config.to_dict(),
                "run_error": run_error,
                "bots": {
                    name: {
                        **asdict(stats),
                        "completed_games": stats.total_games,
                        "win_rate_95pct_interval": stats.win_rate_interval,
                        "score_rate": (
                            (stats.wins + 0.5 * stats.draws) / stats.total_games
                            if stats.total_games
                            else None
                        ),
                        "average_move_time": stats.average_move_time,
                        "error": self.states[name].error_message,
                    }
                    for name, stats in self.stats.items()
                },
            }
            (output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
            logging.getLogger().removeHandler(handler)
            handler.close()
        # Separate protocol-v2 database avoids mixing incompatible historical scores.
        result.timestamp = datetime.now(timezone.utc)
        db = ResultsDatabase(Path(self.config.results_db))
        db.store_benchmark(
            create_benchmark_record(
                run_id, result.timestamp, self.config.to_dict(), result, budget_summary
            )
        )
        db.store_games(result, self.bots)
        self.dashboard.display_final_results(result)
        console.print(f"Tracked list-price cost: ${budget_summary.total_cost:.4f}")
        return result

    async def _initialize_components(
        self, engine_path, engine_type="stockfish", is_human_engine=False
    ):
        for bot in self.bots:
            self.states[bot.name] = LiveState(title=bot.name)
            self.stats[bot.name] = LadderStats()
            client = LLMClient(
                bot,
                use_agent=self.config.use_agent,
                agent_strategy=self.config.agent_strategy,
                verbose_agent=self.config.verbose_agent,
                max_output_tokens=self.config.max_output_tokens,
                reasoning_effort=self.config.reasoning_effort,
            )
            self.clients[bot.name] = client
            engine = None
            if self.config.fixed_opponent_elo != 0:
                engine = ChessEngine(engine_path, self.config)
                self.engines[bot.name] = (
                    engine  # Register before start for failure cleanup.
                )
                await engine.start()
                if self.config.opponent_type == "lowest-elo":
                    if not engine._supported_elo_range:
                        raise RuntimeError("Engine does not advertise a UCI_Elo range")
                    self.config.fixed_opponent_elo = engine._supported_elo_range[0]
            self.engines[bot.name] = engine

    async def _run_bot_ladder(self, bot_name, output_dir):
        try:
            runner = GameRunner(
                self.clients[bot_name], self.engines[bot_name], self.config
            )
            await LadderRunner(runner, self.config).run_ladder(
                output_dir, self.states[bot_name], self.stats[bot_name]
            )
        except Exception as exc:
            logger.exception("Bot %s failed", bot_name)
            self.states[bot_name].set_error(str(exc))

    async def _cleanup_components(self):
        for component in [*self.clients.values(), *self.engines.values()]:
            if component is not None:
                try:
                    if isinstance(component, LLMClient):
                        await component.close()
                    else:
                        await component.stop()
                except Exception:
                    logger.exception("Component cleanup failed")


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
  • lowest-elo: Plays at lowest native UCI_Elo strength

🧠 Playing Modes:
  • Prompt-based: Simple LLM prompting (default)
  • Agent-based: Tool-based reasoning with analysis (--use-agent)
        """,
    )

    # Bot configuration (mutually exclusive with preset)
    bot_group = parser.add_mutually_exclusive_group()
    bot_group.add_argument(
        "--bots", type=str, help="Comma-separated bot specs: provider:model:name"
    )
    bot_group.add_argument(
        "--preset",
        type=str,
        choices=list(PRESET_CONFIGS.keys()),
        default="latest",
        help="Use a predefined set of bots (default: latest)",
    )

    # Information commands
    parser.add_argument(
        "--list-models", action="store_true", help="List all available models and exit"
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List all available presets and exit",
    )

    # Ranking and analysis commands
    parser.add_argument(
        "--leaderboard",
        type=int,
        nargs="?",
        const=20,
        help="Show model leaderboard (default: top 20)",
    )
    parser.add_argument(
        "--provider-stats",
        action="store_true",
        help="Show provider performance comparison",
    )
    parser.add_argument(
        "--analyze-model",
        type=str,
        help="Analyze specific model performance (format: provider:model)",
    )

    # Budget tracking
    parser.add_argument(
        "--budget-limit",
        type=float,
        help="Set budget limit in USD (enables cost tracking and warnings)",
    )
    parser.add_argument(
        "--show-costs",
        action="store_true",
        help="Display detailed cost breakdown during and after benchmark",
    )

    # Opponent configuration
    parser.add_argument(
        "--opponent",
        type=str,
        choices=["random", "lowest-elo"],
        default="random",
        help="Choose opponent: random (random legal moves) or lowest-elo (lowest native UCI_Elo) (default: random)",
    )

    # Playing mode configuration
    parser.add_argument(
        "--use-agent",
        action="store_true",
        help="Use agent-based reasoning with tools instead of simple prompting (default: prompt-based)",
    )

    # Game settings
    parser.add_argument(
        "--max-games",
        type=int,
        default=10,
        help="Maximum number of games to play per model (default: %(default)s)",
    )

    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Print offline USD scenarios and exit; no API calls",
    )
    parser.add_argument("--estimate-input-tokens", type=int, default=None)
    parser.add_argument(
        "--estimate-output-tokens",
        type=int,
        default=512,
        help="Include reasoning tokens (default: 512)",
    )
    parser.add_argument(
        "--estimate-moves",
        type=int,
        default=60,
        help="LLM decisions per game (default: 60)",
    )
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high"], default="low"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--results-db", default="data/results-v2.db")
    parser.add_argument("--stockfish-path")
    parser.add_argument(
        "--opponent-elo",
        type=int,
        help="Native Stockfish UCI_Elo; overrides --opponent",
    )
    return parser


async def main_async(args: argparse.Namespace) -> int:
    """Main async entry point."""
    # All analysis commands use the selected database, not a hidden global path.
    from .core import results as results_module

    results_module._results_db = None
    results_module._ranking_system = None
    if args.leaderboard is not None or args.provider_stats or args.analyze_model:
        results_module._results_db = ResultsDatabase(Path(args.results_db))
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
            for bot in preset_info["bots"]:
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

    announce = console.print if not args.estimate_cost else lambda *args, **kwargs: None
    # Determine bot configuration
    if args.bots:
        bots_string = args.bots
        announce(f"[green]Using custom bots[/green]")
    else:
        # Use preset (latest is default)
        preset_name = (
            args.preset if hasattr(args, "preset") and args.preset else "latest"
        )
        if preset_name not in PRESET_CONFIGS:
            announce(f"[red]Error: Unknown preset '{preset_name}'[/red]")
            return 1
        bot_specs = PRESET_CONFIGS[preset_name]["bots"]
        bots_string = format_bot_spec_string(bot_specs)
        announce(
            f"[green]Using '{preset_name}' models: {PRESET_CONFIGS[preset_name]['description']}[/green]"
        )

    # Handle opponent selection
    if args.opponent_elo is not None:
        announce(f"Opponent override: native UCI_Elo {args.opponent_elo}")
    opponent_type = args.opponent if hasattr(args, "opponent") else "random"
    if opponent_type == "random":
        fixed_opponent_elo = 0  # Special value for random moves
        announce(f"[green]Using opponent: Random moves[/green]")
    elif opponent_type == "lowest-elo":
        fixed_opponent_elo = 600  # Lowest ELO
        announce(f"[green]Using opponent: lowest native UCI_Elo engine[/green]")
    else:
        fixed_opponent_elo = 0  # Default to random
        announce(f"[green]Using opponent: Random moves (default)[/green]")

    # Handle agent mode
    use_agent = args.use_agent if hasattr(args, "use_agent") else False
    max_games = args.max_games if hasattr(args, "max_games") else 10
    agent_mode_str = "agent-based reasoning" if use_agent else "prompt-based"
    announce(f"[green]Using {agent_mode_str}[/green]")

    # Create configuration
    config = Config(
        bots=bots_string,
        fixed_opponent_elo=(
            args.opponent_elo if args.opponent_elo is not None else fixed_opponent_elo
        ),
        use_agent=use_agent,
        # Set simple defaults for required fields
        start_elo=600,
        elo_step=100,
        max_elo=2400,
        think_time=1.0,
        max_plies=args.max_plies,
        max_games=max_games,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        seed=args.seed,
        stockfish_path=args.stockfish_path,
        results_db=args.results_db,
        opponent_type=(
            "lowest-elo"
            if args.opponent == "lowest-elo" and args.opponent_elo is None
            else None
        ),
        budget_limit=args.budget_limit,
        show_costs=args.show_costs,
        llm_timeout=args.llm_timeout,
        llm_temperature=0.0,
        output_dir=args.output_dir,
        save_pgn=True,
        escalate_on="always",
        agent_strategy="balanced",
        verbose_agent=False,
        refresh_rate=6,
    )

    # Add budget tracking configuration
    config.budget_limit = args.budget_limit if hasattr(args, "budget_limit") else None
    config.show_costs = args.show_costs if hasattr(args, "show_costs") else False

    if args.estimate_cost:
        report = estimate_run(
            config,
            parse_bot_spec(config.bots),
            input_tokens=(
                args.estimate_input_tokens
                if args.estimate_input_tokens is not None
                else (1600 if use_agent else 800)
            ),
            output_tokens=args.estimate_output_tokens,
            moves_per_game=args.estimate_moves,
        )
        print(json.dumps(report, indent=2))
        return 0

    # Create and run benchmark
    try:
        orchestrator = BenchmarkOrchestrator(config)
        result = await orchestrator.run_benchmark()

        failed = any(state.error_message for state in orchestrator.states.values())
        console.print(
            "Benchmark finished with errors" if failed else "Benchmark finished"
        )
        for name, stats in result.bot_results.items():
            console.print(
                f"{name}: {stats.total_games} completed, {stats.aborted} incomplete; "
                f"{orchestrator.states[name].error_message or ''}"
            )
        console.print(f"Results saved to: [bold]{result.output_dir}[/bold]")

        return 1 if failed else 0

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Benchmark interrupted by user[/bold yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[bold red]Benchmark failed: {e}[/bold red]")
        logger.exception("Benchmark failed with exception")
        return 1


def main() -> int:
    """Main entry point."""
    from dotenv import load_dotenv

    load_dotenv()
    parser = create_argument_parser()
    args = parser.parse_args()

    # Run main benchmark
    try:
        return asyncio.run(main_async(args))
    except (ValueError, UnknownPricing) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user[/bold yellow]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
