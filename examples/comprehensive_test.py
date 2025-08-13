#!/usr/bin/env python3
"""
Comprehensive Chess LLM Benchmark Examples

This script demonstrates various ways to use the Chess LLM Benchmark tool,
showcasing different providers, configurations, and use cases.

Run this script to see examples of:
- All available model presets
- Custom bot configurations
- Different ELO ladder setups
- Advanced parameter tuning
- Comparative benchmarking strategies

Requirements:
- API keys set in .env file
- Stockfish installed and in PATH
- All dependencies from requirements.txt
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_llm_bench.core.models import Config
from chess_llm_bench.cli import BenchmarkOrchestrator
from chess_llm_bench.llm.models import (
    PRESET_CONFIGS,
    get_premium_bot_lineup,
    get_budget_bot_lineup,
    format_bot_spec_string,
    print_available_models
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


console = Console()


def print_section_header(title: str, description: str = ""):
    """Print a beautiful section header."""
    if description:
        content = f"[bold cyan]{title}[/bold cyan]\n{description}"
    else:
        content = f"[bold cyan]{title}[/bold cyan]"

    console.print(Panel(content, border_style="blue"))
    console.print()


async def demo_all_presets():
    """Demonstrate all available preset configurations."""
    print_section_header(
        "🎯 Preset Configurations Demo",
        "Running quick tests with all available presets"
    )

    for preset_name, preset_info in PRESET_CONFIGS.items():
        console.print(f"[green]Testing preset: {preset_name}[/green]")
        console.print(f"Description: {preset_info['description']}")
        console.print(f"Models: {len(preset_info['bots'])}")

        # Create a quick configuration for demo
        config = Config(
            bots=format_bot_spec_string(preset_info['bots']),
            start_elo=600,
            max_elo=700,  # Quick test - only one ELO level
            elo_step=100,
            think_time=0.1,  # Fast games
            max_plies=50,    # Short games
            output_dir=f"examples/demo_{preset_name}"
        )

        try:
            orchestrator = BenchmarkOrchestrator(config)
            console.print(f"✅ Successfully configured {preset_name} preset")

            # Don't actually run the benchmark in demo mode
            # await orchestrator.run_benchmark()

        except Exception as e:
            console.print(f"❌ Error with {preset_name}: {e}")

        console.print()


async def demo_provider_comparison():
    """Compare models from different providers on the same task."""
    print_section_header(
        "🤖 Provider Comparison",
        "Testing top models from each provider against each other"
    )

    # Select one top model from each provider
    comparison_bots = [
        "openai:gpt-4o:GPT-4o-Champion",
        "anthropic:claude-3-5-sonnet-20241022:Claude-3.5-Sonnet-Master",
        "gemini:gemini-1.5-pro:Gemini-1.5-Pro-Grandmaster"
    ]

    config = Config(
        bots=",".join(comparison_bots),
        start_elo=800,
        max_elo=1400,
        elo_step=200,
        think_time=0.5,
        max_plies=200,
        llm_temperature=0.0,  # Deterministic play
        output_dir="examples/provider_comparison"
    )

    console.print("Configuration:")
    console.print(f"  Bots: {len(comparison_bots)}")
    console.print(f"  ELO Range: {config.start_elo} → {config.max_elo}")
    console.print(f"  Steps: {config.elo_step}")
    console.print()

    # Would run: await BenchmarkOrchestrator(config).run_benchmark()
    console.print("✅ Provider comparison configured successfully")


async def demo_budget_vs_premium():
    """Compare budget models against premium models."""
    print_section_header(
        "💰 Budget vs Premium Showdown",
        "Testing cost-effective models against top-tier models"
    )

    # Create mixed lineup
    budget_bots = [
        "openai:gpt-4o-mini:Budget-GPT-4o-Mini",
        "anthropic:claude-3-5-haiku-20241022:Budget-Claude-Haiku",
        "gemini:gemini-1.5-flash:Budget-Gemini-Flash"
    ]

    premium_bots = [
        "openai:gpt-4o:Premium-GPT-4o",
        "anthropic:claude-3-5-sonnet-20241022:Premium-Claude-Sonnet",
        "gemini:gemini-1.5-pro:Premium-Gemini-Pro"
    ]

    all_bots = budget_bots + premium_bots

    config = Config(
        bots=",".join(all_bots),
        start_elo=600,
        max_elo=1600,
        elo_step=200,
        think_time=0.3,
        output_dir="examples/budget_vs_premium"
    )

    table = Table(title="Budget vs Premium Lineup")
    table.add_column("Category", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Provider", style="yellow")

    for bot in budget_bots:
        provider, model, name = bot.split(":")
        table.add_row("💰 Budget", name, provider.title())

    for bot in premium_bots:
        provider, model, name = bot.split(":")
        table.add_row("👑 Premium", name, provider.title())

    console.print(table)
    console.print("✅ Budget vs Premium comparison configured")


async def demo_creative_vs_deterministic():
    """Compare creative (high temperature) vs deterministic (low temperature) play."""
    print_section_header(
        "🎨 Creative vs Deterministic Play",
        "Testing the same models with different creativity settings"
    )

    base_model = "openai:gpt-4o:GPT-4o"

    # Test the same model with different temperatures
    creative_config = Config(
        bots=f"{base_model}-Creative",
        start_elo=600,
        max_elo=1000,
        elo_step=200,
        llm_temperature=0.8,  # High creativity
        think_time=0.5,
        output_dir="examples/creative_play"
    )

    deterministic_config = Config(
        bots=f"{base_model}-Deterministic",
        start_elo=600,
        max_elo=1000,
        elo_step=200,
        llm_temperature=0.0,  # No randomness
        think_time=0.5,
        output_dir="examples/deterministic_play"
    )

    console.print("Creative Configuration (Temperature 0.8):")
    console.print(f"  Expected: More varied, potentially risky moves")
    console.print(f"  Trade-off: Creativity vs consistency")
    console.print()

    console.print("Deterministic Configuration (Temperature 0.0):")
    console.print(f"  Expected: Consistent, logical moves")
    console.print(f"  Trade-off: Reliability vs exploration")
    console.print()

    console.print("✅ Creative vs Deterministic comparison configured")


async def demo_elo_ladder_strategies():
    """Demonstrate different ELO ladder progression strategies."""
    print_section_header(
        "🪜 ELO Ladder Strategies",
        "Testing different progression approaches"
    )

    test_bot = "anthropic:claude-3-5-sonnet-20241022:Claude-Test"

    strategies = {
        "Conservative": Config(
            bots=f"{test_bot}-Conservative",
            start_elo=400,
            max_elo=1200,
            elo_step=50,  # Small steps
            escalate_on="on_win",  # Only advance on wins
            output_dir="examples/conservative_ladder"
        ),
        "Aggressive": Config(
            bots=f"{test_bot}-Aggressive",
            start_elo=800,
            max_elo=2000,
            elo_step=200,  # Large steps
            escalate_on="always",  # Always advance
            output_dir="examples/aggressive_ladder"
        ),
        "Balanced": Config(
            bots=f"{test_bot}-Balanced",
            start_elo=600,
            max_elo=1600,
            elo_step=100,  # Medium steps
            escalate_on="always",
            output_dir="examples/balanced_ladder"
        )
    }

    table = Table(title="ELO Ladder Strategies")
    table.add_column("Strategy", style="cyan")
    table.add_column("Start ELO", style="green")
    table.add_column("Max ELO", style="green")
    table.add_column("Step Size", style="yellow")
    table.add_column("Advance Rule", style="red")

    for name, config in strategies.items():
        table.add_row(
            name,
            str(config.start_elo),
            str(config.max_elo),
            str(config.elo_step),
            config.escalate_on
        )

    console.print(table)
    console.print("✅ All ELO ladder strategies configured")


async def demo_speed_testing():
    """Demonstrate speed-optimized configurations for quick testing."""
    print_section_header(
        "⚡ Speed Testing Configuration",
        "Optimized settings for rapid evaluation and development"
    )

    speed_config = Config(
        bots="openai:gpt-4o-mini:Speed-Test,anthropic:claude-3-5-haiku-20241022:Speed-Test-2",
        start_elo=600,
        max_elo=800,  # Only 2 levels
        elo_step=200,
        think_time=0.1,  # Very fast engine
        max_plies=30,    # Short games
        llm_timeout=10.0,  # Quick timeouts
        refresh_rate=10,   # Fast UI updates
        output_dir="examples/speed_test"
    )

    console.print("Speed Optimization Settings:")
    console.print(f"  ⚡ Think Time: {speed_config.think_time}s (very fast)")
    console.print(f"  🎯 Max Plies: {speed_config.max_plies} (short games)")
    console.print(f"  ⏱️  LLM Timeout: {speed_config.llm_timeout}s")
    console.print(f"  📊 Refresh Rate: {speed_config.refresh_rate} Hz")
    console.print(f"  🪜 ELO Range: {speed_config.start_elo}-{speed_config.max_elo}")
    console.print()

    console.print("Use cases:")
    console.print("  • Quick model testing during development")
    console.print("  • Rapid iteration on prompts/parameters")
    console.print("  • CI/CD pipeline integration")
    console.print("  • Demo and presentation purposes")
    console.print()

    console.print("✅ Speed testing configuration ready")


async def demo_comprehensive_analysis():
    """Demonstrate a comprehensive analysis setup."""
    print_section_header(
        "📊 Comprehensive Analysis Setup",
        "Configuration for thorough model evaluation and research"
    )

    # Use all recommended models for comprehensive coverage
    analysis_config = Config(
        bots=format_bot_spec_string(get_premium_bot_lineup()),
        start_elo=400,    # Wide range
        max_elo=2000,
        elo_step=100,     # Detailed progression
        think_time=1.0,   # Stronger play
        max_plies=300,    # Full games
        llm_timeout=30.0, # Generous timeouts
        llm_temperature=0.0,  # Consistent results
        save_pgn=True,    # Full game records
        output_dir="examples/comprehensive_analysis"
    )

    console.print("Comprehensive Analysis Features:")
    console.print(f"  🤖 Models: {len(get_premium_bot_lineup())} premium models")
    console.print(f"  🪜 ELO Range: {analysis_config.start_elo}-{analysis_config.max_elo}")
    console.print(f"  📈 Data Points: {(analysis_config.max_elo - analysis_config.start_elo) // analysis_config.elo_step + 1} per model")
    console.print(f"  🎮 Game Quality: High (think time: {analysis_config.think_time}s)")
    console.print(f"  💾 Records: PGN files for all games")
    console.print(f"  🔬 Reproducibility: Temperature {analysis_config.llm_temperature}")
    console.print()

    estimated_games = len(get_premium_bot_lineup()) * ((analysis_config.max_elo - analysis_config.start_elo) // analysis_config.elo_step + 1)
    console.print(f"📊 Estimated total games: ~{estimated_games}")
    console.print(f"⏱️  Estimated duration: ~{estimated_games * 2} minutes")
    console.print()

    console.print("✅ Comprehensive analysis configured")


def show_environment_check():
    """Check and display environment setup."""
    print_section_header("🔧 Environment Check", "Verifying API keys and dependencies")

    # Check API keys
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY")
    }

    table = Table(title="API Key Status")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Available Models", style="yellow")

    for key_name, key_value in api_keys.items():
        provider = key_name.split("_")[0].title()
        status = "✅ Set" if key_value else "❌ Missing"

        if provider == "OPENAI":
            models = "GPT-4o, GPT-4o Mini, GPT-3.5 Turbo" if key_value else "None"
        elif provider == "ANTHROPIC":
            models = "Claude 3.5 Sonnet, Claude 3.5 Haiku" if key_value else "None"
        elif provider == "GEMINI":
            models = "Gemini 1.5 Pro, Gemini 1.5 Flash" if key_value else "None"
        else:
            models = "Unknown"

        table.add_row(provider, status, models)

    console.print(table)
    console.print()

    # Quick setup reminder
    missing_keys = [k for k, v in api_keys.items() if not v]
    if missing_keys:
        console.print("[yellow]Missing API keys. Add to .env file:[/yellow]")
        for key in missing_keys:
            console.print(f"  {key}=your-api-key-here")
        console.print()


async def main():
    """Run all demonstration examples."""
    console.print("[bold blue]🏆 Chess LLM Benchmark - Comprehensive Examples[/bold blue]")
    console.print()

    # Environment check
    show_environment_check()

    # Show available models
    console.print("[bold]Available Models:[/bold]")
    print_available_models()

    # Run all demonstrations
    demos = [
        ("All Presets", demo_all_presets),
        ("Provider Comparison", demo_provider_comparison),
        ("Budget vs Premium", demo_budget_vs_premium),
        ("Creative vs Deterministic", demo_creative_vs_deterministic),
        ("ELO Ladder Strategies", demo_elo_ladder_strategies),
        ("Speed Testing", demo_speed_testing),
        ("Comprehensive Analysis", demo_comprehensive_analysis)
    ]

    for demo_name, demo_func in demos:
        try:
            await demo_func()
        except Exception as e:
            console.print(f"[red]Error in {demo_name} demo: {e}[/red]")
        console.print("-" * 80)
        console.print()

    # Final summary
    print_section_header(
        "🎯 Next Steps",
        "Ready to run your own benchmarks!"
    )

    console.print("[green]Example commands to try:[/green]")
    console.print()
    console.print("# Quick start with premium models")
    console.print("python main.py --preset premium --start-elo 600 --max-elo 1000")
    console.print()
    console.print("# Budget comparison")
    console.print("python main.py --preset budget --start-elo 800 --max-elo 1200")
    console.print()
    console.print("# Custom model lineup")
    console.print("python main.py --bots \"openai:gpt-4o:Champion,anthropic:claude-3-5-sonnet-20241022:Master\"")
    console.print()
    console.print("# Speed test for development")
    console.print("python main.py --preset openai --start-elo 600 --max-elo 700 --think-time 0.1")
    console.print()

    console.print("[bold cyan]Happy benchmarking! 🤖♟️[/bold cyan]")


if __name__ == "__main__":
    # Run the comprehensive examples
    asyncio.run(main())
