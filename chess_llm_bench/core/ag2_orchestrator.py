"""
AG2-based Benchmark Orchestrator

This module provides an AG2-based orchestrator for running chess LLM benchmarks
using the new agent system. It coordinates agents, game runners, and UI components
while maintaining compatibility with the existing benchmark interface.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from rich.console import Console

from .models import Config, BotSpec, LiveState, LadderStats, BenchmarkResult
from .engine import ChessEngine, autodetect_stockfish, get_friendly_stockfish_hint, create_engine, autodetect_engine
from .human_engine import HumanLikeEngine, autodetect_human_engines, get_best_human_engine, get_human_engine_installation_hint, create_human_engine
from .adaptive_engine import AdaptiveEngine
from .ag2_game import AG2GameRunner, AG2LadderRunner
from .budget import start_budget_tracking, stop_budget_tracking, get_budget_tracker
from .results import store_benchmark_results, show_leaderboard, show_provider_comparison, analyze_model
from ..agents import (
    ChessAgent,
    GameAgent,
    create_agents,
    AgentCreationError,
    validate_bot_spec,
    get_factory_stats
)
from ..llm.client import parse_bot_spec
from ..llm.models import PRESET_CONFIGS, format_bot_spec_string, print_available_models, get_premium_bot_lineup
from ..ui.dashboard import Dashboard
from ..ui.board import render_robot_battle

logger = logging.getLogger(__name__)


class AG2BenchmarkOrchestrator:
    """
    AG2-based orchestrator for running chess LLM benchmarks.

    This orchestrator uses the new AG2 agent system instead of direct LLM clients,
    providing better modularity and extensibility while maintaining full compatibility
    with the existing benchmark interface and features.
    """

    def __init__(self, config: Config):
        """Initialize the AG2 benchmark orchestrator."""
        self.config = config
        self.dashboard = Dashboard(console, config)
        self.budget_tracker = None

        # Runtime state
        self.bots: List[BotSpec] = []
        self.engines: Dict[str, Union[ChessEngine, HumanLikeEngine, AdaptiveEngine]] = {}
        self.agents: Dict[str, ChessAgent] = {}  # Changed from clients to agents
        self.states: Dict[str, LiveState] = {}
        self.stats: Dict[str, LadderStats] = {}

        # AG2-specific components
        self.game_runners: Dict[str, AG2GameRunner] = {}
        self.ladder_runners: Dict[str, AG2LadderRunner] = {}

        logger.info("Initialized AG2BenchmarkOrchestrator")

    async def run_benchmark(self) -> BenchmarkResult:
        """
        Run the complete benchmark with all configured bots using AG2 agents.

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

        # Validate bot specs for AG2 compatibility
        await self._validate_bot_specs()

        # Initialize engines
        await self._initialize_engines()

        # Create agents (instead of LLM clients)
        await self._create_agents()

        # Initialize game components
        await self._initialize_game_components()

        # Initialize live states
        self._initialize_live_states()

        try:
            # Start live dashboard
            with self.dashboard:
                # Run ladder tests for each bot
                await self._run_all_ladders()

        finally:
            # Stop budget tracking and get summary
            if self.budget_tracker:
                budget_summary = stop_budget_tracking()

        # Compile final results
        return self._compile_results(budget_summary)

    async def _validate_bot_specs(self) -> None:
        """Validate bot specifications for AG2 compatibility."""
        invalid_bots = []

        for bot_spec in self.bots:
            is_valid, error_msg = validate_bot_spec(bot_spec)
            if not is_valid:
                invalid_bots.append((bot_spec, error_msg))
                logger.error(f"Invalid bot spec {bot_spec}: {error_msg}")

        if invalid_bots:
            error_details = "; ".join([f"{spec.name}: {msg}" for spec, msg in invalid_bots])
            raise RuntimeError(f"Invalid bot specifications: {error_details}")

        logger.info(f"Validated {len(self.bots)} bot specifications for AG2")

    async def _initialize_engines(self) -> None:
        """Initialize chess engines for each bot."""
        console.print("[blue]🔧 Initializing engines...[/blue]")

        for bot_spec in self.bots:
            try:
                # Create engine for this bot
                if self.config.use_human_engine:
                    # Use human-like engine
                    if self.config.human_engine_type == "auto":
                        result = get_best_human_engine()
                        if not result:
                            raise RuntimeError("No human engines found. " + get_human_engine_installation_hint())
                        engine_type, engine_path = result
                        engine = create_human_engine(engine_type, engine_path, self.config)
                    else:
                        engine_path = self.config.human_engine_path
                        if not engine_path:
                            raise RuntimeError(f"Human engine path not specified for type: {self.config.human_engine_type}")
                        engine = create_human_engine(self.config.human_engine_type, engine_path, self.config)

                    if self.config.adaptive_elo_engines:
                        engine = AdaptiveEngine([engine])

                    # Start the engine
                    await engine.start()
                    console.print(f"[green]✓[/green] Human engine ready for {bot_spec.name}")

                else:
                    # Use Stockfish
                    stockfish_path = autodetect_stockfish()
                    if not stockfish_path:
                        raise RuntimeError("Stockfish not found. " + get_friendly_stockfish_hint())

                    engine = create_engine(self.config, stockfish_path)

                    # Start the engine
                    await engine.start()
                    console.print(f"[green]✓[/green] Stockfish ready for {bot_spec.name}")

                self.engines[bot_spec.name] = engine

            except Exception as e:
                console.print(f"[red]✗[/red] Failed to initialize engine for {bot_spec.name}: {e}")
                raise RuntimeError(f"Engine initialization failed for {bot_spec.name}: {e}")

        console.print(f"[green]✓[/green] All {len(self.engines)} engines initialized")

    async def _create_agents(self) -> None:
        """Create chess agents from bot specifications."""
        console.print("[blue]🤖 Creating AG2 chess agents...[/blue]")

        try:
            # Create all agents at once using the factory
            agents = create_agents(
                bot_specs=self.bots,
                temperature=self.config.llm_temperature,
                timeout_seconds=self.config.llm_timeout
            )

            # Store agents by name
            for agent in agents:
                self.agents[agent.bot_spec.name] = agent
                console.print(f"[green]✓[/green] Created agent: {agent.name} ({agent.bot_spec.provider})")

            console.print(f"[green]✓[/green] All {len(agents)} AG2 agents created")

            # Log factory statistics
            factory_stats = get_factory_stats()
            logger.info(f"Agent factory stats: {factory_stats}")

        except AgentCreationError as e:
            console.print(f"[red]✗[/red] Failed to create agents: {e}")
            raise RuntimeError(f"Agent creation failed: {e}")

    async def _initialize_game_components(self) -> None:
        """Initialize AG2 game runners and ladder runners."""
        console.print("[blue]🎮 Initializing AG2 game components...[/blue]")

        for bot_spec in self.bots:
            try:
                engine = self.engines[bot_spec.name]

                # Create AG2 game runner
                game_runner = AG2GameRunner(self.config, engine)
                self.game_runners[bot_spec.name] = game_runner

                # Create AG2 ladder runner
                ladder_runner = AG2LadderRunner(self.config, engine)
                self.ladder_runners[bot_spec.name] = ladder_runner

                console.print(f"[green]✓[/green] Game components ready for {bot_spec.name}")

            except Exception as e:
                console.print(f"[red]✗[/red] Failed to initialize game components for {bot_spec.name}: {e}")
                raise RuntimeError(f"Game component initialization failed: {e}")

        console.print(f"[green]✓[/green] All {len(self.game_runners)} game components initialized")

    def _initialize_live_states(self) -> None:
        """Initialize live state objects for each bot."""
        for bot_spec in self.bots:
            state = LiveState(title=bot_spec.name)
            state.current_bot = bot_spec.name
            state.status = "Initializing..."
            self.states[bot_spec.name] = state

        logger.info(f"Initialized {len(self.states)} live states")

    async def _run_all_ladders(self) -> None:
        """Run ELO ladder tests for all bots."""
        console.print("[blue]🏆 Starting ELO ladder tests...[/blue]")

        # Create output directory
        output_dir = Path(f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, bot_spec in enumerate(self.bots):
            console.print(f"\n[yellow]▶️  Starting ladder for {bot_spec.name} ({i+1}/{len(self.bots)})[/yellow]")

            try:
                # Get components for this bot
                ladder_runner = self.ladder_runners[bot_spec.name]
                state = self.states[bot_spec.name]

                # Update dashboard to show current bot
                self.dashboard.update_display(self.states, self.stats)

                # Run the ladder
                ladder_stats = await ladder_runner.run_ladder(
                    bot_spec=bot_spec,
                    output_dir=output_dir / bot_spec.name,
                    state=state
                )

                # Store results
                self.stats[bot_spec.name] = ladder_stats

                # Show ladder completion
                console.print(
                    f"[green]✅ Ladder complete for {bot_spec.name}:[/green] "
                    f"{ladder_stats.wins}W-{ladder_stats.losses}L-{ladder_stats.draws}D, "
                    f"Final ELO: {ladder_stats.final_elo}, "
                    f"Effective ELO: {ladder_stats.effective_elo}"
                )

            except Exception as e:
                console.print(f"[red]❌ Ladder failed for {bot_spec.name}: {e}[/red]")
                logger.error(f"Ladder failed for {bot_spec.name}: {e}")

                # Create minimal stats for failed ladder
                self.stats[bot_spec.name] = LadderStats(
                    max_elo_reached=self.config.start_elo,
                    games=[],
                    wins=0,
                    losses=0,
                    draws=0,
                    total_move_time=0.0,
                    total_illegal_moves=0,
                    total_game_duration=0.0
                )

        console.print(f"\n[green]🏁 All ladder tests completed![/green]")

    def _compile_results(self, budget_summary: Optional[Dict] = None) -> BenchmarkResult:
        """Compile final benchmark results."""
        # Calculate aggregate statistics
        total_games = sum(stats.total_games for stats in self.stats.values())
        total_duration = sum(stats.total_game_duration for stats in self.stats.values())

        # Find best performing bot
        best_bot = None
        best_elo = 0
        for bot_name, stats in self.stats.items():
            if stats.max_elo_reached > best_elo:
                best_elo = stats.max_elo_reached
                best_bot = bot_name

        # Create benchmark result
        result = BenchmarkResult(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            config=self.config,
            bot_results=self.stats,
            output_dir=Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        # Store results
        try:
            store_benchmark_results(
                result.run_id,
                result.timestamp,
                result.config.to_dict(),
                result,
                budget_summary
            )
            console.print("[green]💾 Benchmark results saved[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Failed to save results: {e}[/yellow]")

        return result

    async def run_single_game(
        self,
        bot_name: str,
        elo: int,
        output_dir: Path,
        llm_plays_white: Optional[bool] = None,
        opening_moves: Optional[List[str]] = None
    ) -> Optional[any]:
        """
        Run a single game for testing purposes.

        Args:
            bot_name: Name of the bot to test
            elo: Engine ELO for the game
            output_dir: Directory to save game files
            llm_plays_white: Color assignment
            opening_moves: Opening moves to apply

        Returns:
            GameRecord if successful, None otherwise
        """
        if bot_name not in self.agents:
            logger.error(f"Bot {bot_name} not found in agents")
            return None

        if bot_name not in self.game_runners:
            logger.error(f"Game runner for {bot_name} not found")
            return None

        try:
            bot_spec = next(bot for bot in self.bots if bot.name == bot_name)
            game_runner = self.game_runners[bot_name]
            state = self.states[bot_name]

            game_record = await game_runner.play_game(
                bot_spec=bot_spec,
                elo=elo,
                output_dir=output_dir,
                state=state,
                llm_plays_white=llm_plays_white,
                opening_moves=opening_moves
            )

            return game_record

        except Exception as e:
            logger.error(f"Single game failed for {bot_name}: {e}")
            return None

    def get_orchestrator_stats(self) -> Dict[str, any]:
        """Get comprehensive orchestrator statistics."""
        agent_stats = {}
        game_runner_stats = {}

        for name, agent in self.agents.items():
            agent_stats[name] = agent.get_detailed_stats()

        for name, runner in self.game_runners.items():
            game_runner_stats[name] = runner.get_game_stats()

        return {
            "agents_created": len(self.agents),
            "engines_initialized": len(self.engines),
            "game_runners_active": len(self.game_runners),
            "ladders_completed": len(self.stats),
            "agent_stats": agent_stats,
            "game_runner_stats": game_runner_stats,
            "factory_stats": get_factory_stats(),
        }

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Close engines
            for engine in self.engines.values():
                if hasattr(engine, 'close'):
                    try:
                        await engine.close()
                    except:
                        pass

            # Clear caches
            for runner in self.game_runners.values():
                runner.clear_agent_cache()

            console.print("[green]🧹 Cleanup completed[/green]")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def __str__(self) -> str:
        """String representation of orchestrator."""
        return f"AG2BenchmarkOrchestrator(bots={len(self.bots)}, agents={len(self.agents)})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"AG2BenchmarkOrchestrator(bots={len(self.bots)}, "
            f"agents={len(self.agents)}, "
            f"engines={len(self.engines)}, "
            f"completed_ladders={len(self.stats)})"
        )


# Console instance for module-level operations
console = Console()
