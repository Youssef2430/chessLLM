#!/usr/bin/env python3
"""
Budget Tracking and Ranking System Demo

This comprehensive demo showcases the advanced budget tracking and ranking
features of the Chess LLM Benchmark tool.

Features demonstrated:
- Real-time cost tracking during benchmarks
- Budget limits with warnings and alerts
- Model performance ranking system
- Historical data analysis
- Provider cost comparisons
- Efficiency metrics and value scoring
- Database storage and retrieval

Run this script to see examples of:
- Budget-controlled benchmarks
- Cost-efficient model testing
- Performance leaderboards
- Model analysis and trends
- Provider efficiency comparisons
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_llm_bench.core.models import Config
from chess_llm_bench.core.budget import start_budget_tracking, get_budget_tracker, PROVIDER_PRICING
from chess_llm_bench.core.results import get_results_db, get_ranking_system, show_leaderboard, show_provider_comparison, analyze_model
from chess_llm_bench.cli import BenchmarkOrchestrator
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns

console = Console()


def print_demo_header(title: str, description: str = ""):
    """Print a beautiful demo section header."""
    content = f"[bold cyan]{title}[/bold cyan]"
    if description:
        content += f"\n[dim]{description}[/dim]"

    console.print(Panel(content, border_style="cyan", padding=(1, 2)))
    console.print()


def show_pricing_information():
    """Display current pricing information for all providers."""
    print_demo_header(
        "💰 Current LLM Pricing",
        "Understanding costs before running benchmarks"
    )

    pricing_table = Table(title="LLM API Pricing (USD per 1K tokens)")
    pricing_table.add_column("Provider", style="cyan")
    pricing_table.add_column("Model", style="green")
    pricing_table.add_column("Input", style="yellow", justify="right")
    pricing_table.add_column("Output", style="red", justify="right")
    pricing_table.add_column("Est. Cost/Game", style="blue", justify="right")

    for provider, models in PROVIDER_PRICING.items():
        if provider == "random":
            continue

        for model_id, pricing in models.items():
            # Estimate cost per game (assuming ~100 input + 10 output tokens per move, 40 moves)
            estimated_cost = pricing.calculate_cost(4000, 400)  # 40 moves * 100 input + 40 * 10 output

            pricing_table.add_row(
                provider.title(),
                model_id,
                f"${pricing.input_cost_per_1k_tokens:.4f}",
                f"${pricing.output_cost_per_1k_tokens:.4f}",
                f"${estimated_cost:.4f}"
            )

    console.print(pricing_table)
    console.print()

    console.print("[yellow]💡 Tips for Cost Management:[/yellow]")
    console.print("  • Use --budget-limit to set spending caps")
    console.print("  • Start with budget-friendly models like GPT-4o Mini or Claude Haiku")
    console.print("  • Use shorter ELO ranges for initial testing")
    console.print("  • Monitor costs in real-time with --show-costs")
    console.print()


def demo_budget_scenarios():
    """Demonstrate different budget scenarios."""
    print_demo_header(
        "🎯 Budget Scenario Planning",
        "Examples of different budget approaches"
    )

    scenarios = [
        {
            "name": "🔬 Research Budget",
            "description": "Comprehensive testing with generous budget",
            "budget": 50.0,
            "models": "Premium models across all providers",
            "elo_range": "600-2000 (detailed steps)",
            "use_case": "Academic research, thorough evaluation"
        },
        {
            "name": "💼 Business Budget",
            "description": "Practical evaluation for production use",
            "budget": 10.0,
            "models": "Top 2-3 models from each provider",
            "elo_range": "600-1400 (moderate testing)",
            "use_case": "Production model selection"
        },
        {
            "name": "🧪 Development Budget",
            "description": "Quick testing during development",
            "budget": 2.0,
            "models": "Budget models (GPT-4o Mini, Claude Haiku)",
            "elo_range": "600-800 (quick validation)",
            "use_case": "Development iteration, prompt testing"
        },
        {
            "name": "🎓 Learning Budget",
            "description": "Educational exploration",
            "budget": 0.50,
            "models": "Single model comparison",
            "elo_range": "600-700 (minimal testing)",
            "use_case": "Learning, experimentation"
        }
    ]

    for scenario in scenarios:
        console.print(f"[bold]{scenario['name']}[/bold]")
        console.print(f"Budget: [green]${scenario['budget']:.2f}[/green]")
        console.print(f"Models: {scenario['models']}")
        console.print(f"ELO Range: {scenario['elo_range']}")
        console.print(f"Use Case: [dim]{scenario['use_case']}[/dim]")
        console.print()

    # Show example commands
    console.print("[bold yellow]💡 Example Commands:[/bold yellow]")
    console.print()
    console.print("[cyan]# Research budget - comprehensive testing[/cyan]")
    console.print("python main.py --preset premium --budget-limit 50.0 --show-costs")
    console.print()
    console.print("[cyan]# Business budget - practical evaluation[/cyan]")
    console.print("python main.py --preset recommended --budget-limit 10.0 --start-elo 600 --max-elo 1400")
    console.print()
    console.print("[cyan]# Development budget - quick testing[/cyan]")
    console.print("python main.py --preset budget --budget-limit 2.0 --start-elo 600 --max-elo 800")
    console.print()
    console.print("[cyan]# Learning budget - minimal testing[/cyan]")
    console.print("python main.py --bots \"openai:gpt-4o-mini:Test\" --budget-limit 0.50 --start-elo 600 --max-elo 700")
    console.print()


async def run_budget_demo():
    """Run a live budget tracking demonstration."""
    print_demo_header(
        "🚀 Live Budget Tracking Demo",
        "Running a small benchmark with cost monitoring"
    )

    console.print("[yellow]Setting up budget-controlled benchmark...[/yellow]")

    # Create a minimal configuration for demo
    config = Config(
        bots="random::Budget-Demo-Bot",  # Using random bot to avoid API costs in demo
        start_elo=600,
        max_elo=700,
        elo_step=100,
        think_time=0.1,  # Fast demo
        max_plies=30,    # Short games
        budget_limit=0.01,  # Very small limit for demo
        show_costs=True,
        output_dir="examples/budget_demo"
    )

    try:
        # Initialize budget tracking
        tracker = start_budget_tracking(config.budget_limit)

        console.print(f"[green]✅ Budget tracker started with ${config.budget_limit:.3f} limit[/green]")

        # Simulate some API usage for demo
        for i in range(5):
            cost = tracker.record_usage(
                provider="demo",
                model="demo-model",
                bot_name="Budget-Demo-Bot",
                prompt=f"Demo prompt {i+1}",
                response=f"Demo response {i+1}",
                success=True
            )
            console.print(f"[dim]API call {i+1}: ${cost:.4f}[/dim]")

            # Show current status
            if i == 2:  # Show status partway through
                console.print(f"[blue]Current total: ${tracker.get_current_cost():.4f}[/blue]")

            time.sleep(0.5)  # Brief delay for demo effect

        # Show final budget status
        console.print()
        tracker.display_budget_status()

        # Show cost breakdown table
        console.print()
        console.print(tracker.create_cost_table())

    except Exception as e:
        console.print(f"[red]Demo error: {e}[/red]")

    console.print()


def demonstrate_ranking_features():
    """Demonstrate the ranking and analysis system."""
    print_demo_header(
        "🏆 Ranking System Features",
        "Advanced model performance analysis and comparison"
    )

    console.print("[yellow]The ranking system provides:[/yellow]")
    console.print()

    features = [
        ("🥇 Leaderboards", "Ranked lists of best performing models"),
        ("📊 Performance Metrics", "ELO ratings, win rates, consistency scores"),
        ("💰 Cost Analysis", "Efficiency scores, cost per ELO point"),
        ("📈 Trend Analysis", "Historical performance tracking"),
        ("🔍 Model Deep-Dive", "Detailed individual model analysis"),
        ("⚖️  Provider Comparison", "Cross-provider performance stats"),
        ("💎 Value Rankings", "Best performance per dollar spent"),
    ]

    for icon_name, description in features:
        console.print(f"  {icon_name}: [dim]{description}[/dim]")

    console.print()

    # Show example commands
    console.print("[bold yellow]📋 Available Analysis Commands:[/bold yellow]")
    console.print()

    commands = [
        ("--leaderboard 10", "Show top 10 performing models"),
        ("--provider-stats", "Compare performance across providers"),
        ("--analyze-model openai:gpt-4o", "Deep analysis of specific model"),
        ("--leaderboard", "Show top 20 models (default)"),
    ]

    for command, description in commands:
        console.print(f"[cyan]python main.py {command}[/cyan]")
        console.print(f"  {description}")
        console.print()

    # Show sample analysis
    console.print("[bold]📊 Sample Analysis Output:[/bold]")
    console.print()

    # Create a mock leaderboard table
    sample_table = Table(title="🏆 Sample Model Leaderboard")
    sample_table.add_column("Rank", style="cyan")
    sample_table.add_column("Model", style="green")
    sample_table.add_column("Max ELO", style="yellow", justify="right")
    sample_table.add_column("Win Rate", style="blue", justify="right")
    sample_table.add_column("Cost", style="red", justify="right")
    sample_table.add_column("Value", style="green", justify="right")
    sample_table.add_column("Trend", style="cyan")

    sample_data = [
        ("🥇 1", "GPT-4o\nopenai", "1847", "68.2%", "$0.234", "7891", "📈 Improving"),
        ("🥈 2", "Claude-3.5-Sonnet\nanthropic", "1823", "65.1%", "$0.456", "3998", "➡️ Stable"),
        ("🥉 3", "Gemini-1.5-Pro\ngemini", "1789", "61.3%", "$0.123", "14553", "📊 Stable"),
        ("4", "GPT-4o-Mini\nopenai", "1456", "52.7%", "$0.045", "32356", "📈 Strong Improvement"),
        ("5", "Claude-3.5-Haiku\nanthropic", "1398", "49.8%", "$0.089", "15708", "📊 Improving"),
    ]

    for rank, model, elo, win_rate, cost, value, trend in sample_data:
        sample_table.add_row(rank, model, elo, win_rate, cost, value, trend)

    console.print(sample_table)
    console.print()


def show_efficiency_insights():
    """Show insights about model efficiency and value."""
    print_demo_header(
        "💡 Efficiency Insights",
        "Understanding value and cost-effectiveness"
    )

    console.print("[yellow]Key Efficiency Metrics:[/yellow]")
    console.print()

    metrics = [
        ("💰 Cost per Game", "Total API cost divided by games played"),
        ("🎯 Cost per ELO", "Cost to achieve each ELO rating point"),
        ("⚡ Efficiency Score", "ELO points per dollar spent"),
        ("💎 Value Score", "Overall performance normalized by cost"),
        ("📊 Consistency", "Standard deviation of ELO performance"),
    ]

    for metric, description in metrics:
        console.print(f"  {metric}: [dim]{description}[/dim]")

    console.print()

    # Create efficiency comparison
    efficiency_table = Table(title="💎 Model Value Comparison (Example)")
    efficiency_table.add_column("Model", style="cyan")
    efficiency_table.add_column("Performance Tier", style="yellow")
    efficiency_table.add_column("Cost Tier", style="green")
    efficiency_table.add_column("Value Rating", style="blue")
    efficiency_table.add_column("Best For", style="magenta")

    value_data = [
        ("GPT-4o Mini", "High", "Low", "⭐⭐⭐⭐⭐", "Development, Testing"),
        ("Gemini-1.5-Flash", "High", "Very Low", "⭐⭐⭐⭐⭐", "Budget Conscious"),
        ("Claude-3.5-Haiku", "Medium-High", "Low", "⭐⭐⭐⭐", "Balanced Use"),
        ("GPT-4o", "Very High", "Medium", "⭐⭐⭐", "Maximum Performance"),
        ("Claude-3.5-Sonnet", "Very High", "High", "⭐⭐⭐", "Premium Applications"),
    ]

    for model, perf, cost, value, best_for in value_data:
        efficiency_table.add_row(model, perf, cost, value, best_for)

    console.print(efficiency_table)
    console.print()

    console.print("[bold yellow]💡 Strategic Recommendations:[/bold yellow]")
    console.print()
    console.print("🎯 [bold]For Development & Testing[/bold]")
    console.print("   • Use GPT-4o Mini or Gemini Flash for rapid iteration")
    console.print("   • Set low budget limits ($1-5) for safety")
    console.print("   • Focus on ELO ranges 600-1000 for quick validation")
    console.print()
    console.print("🏢 [bold]For Production Evaluation[/bold]")
    console.print("   • Test top models: GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Pro")
    console.print("   • Use comprehensive ELO ranges (600-1800+)")
    console.print("   • Budget $10-50 for thorough evaluation")
    console.print()
    console.print("🧪 [bold]For Research[/bold]")
    console.print("   • Compare all available models")
    console.print("   • Use maximum ELO ranges and multiple runs")
    console.print("   • Analyze trends and statistical significance")
    console.print()


def show_historical_analysis():
    """Demonstrate historical data analysis features."""
    print_demo_header(
        "📈 Historical Analysis & Trends",
        "Track model improvements and performance over time"
    )

    console.print("[yellow]Historical tracking provides:[/yellow]")
    console.print()

    # Create sample trend analysis
    trend_table = Table(title="📊 Model Performance Trends (Example)")
    trend_table.add_column("Model", style="cyan")
    trend_table.add_column("Benchmarks", style="blue", justify="right")
    trend_table.add_column("First ELO", style="green", justify="right")
    trend_table.add_column("Latest ELO", style="yellow", justify="right")
    trend_table.add_column("Improvement", style="red", justify="right")
    trend_table.add_column("Trend", style="magenta")

    trend_data = [
        ("GPT-4o", "12", "1623", "1847", "+224", "📈 Strong Growth"),
        ("Claude-3.5-Sonnet", "8", "1789", "1823", "+34", "📊 Stable+"),
        ("Gemini-1.5-Pro", "15", "1234", "1789", "+555", "🚀 Rapid Growth"),
        ("GPT-4o-Mini", "20", "1123", "1456", "+333", "📈 Steady Growth"),
        ("GPT-3.5-Turbo", "25", "1456", "1398", "-58", "📉 Declining"),
    ]

    for model, benchmarks, first, latest, improvement, trend in trend_data:
        trend_table.add_row(model, benchmarks, first, latest, improvement, trend)

    console.print(trend_table)
    console.print()

    console.print("[yellow]📋 Analysis Features:[/yellow]")
    console.print("  • Track ELO progression over multiple benchmarks")
    console.print("  • Identify improving vs. declining models")
    console.print("  • Statistical analysis (mean, median, std dev)")
    console.print("  • Cost trend analysis over time")
    console.print("  • Performance consistency scoring")
    console.print()

    console.print("[bold]🔍 Sample Model Deep-Dive:[/bold]")
    console.print()

    sample_analysis = """
    🤖 GPT-4o Mini Analysis
    📡 Provider: OpenAI
    🏆 Current Max ELO: 1,456
    📊 Win Rate: 52.7%
    💰 Total Cost: $2.34 (20 benchmarks)
    ⭐ Efficiency Score: 622 ELO/$ (Excellent)

    📈 Performance Trend: Strong Improvement
    📊 Statistics:
       • Mean ELO: 1,289
       • Best Performance: 1,456
       • Consistency Score: 12.4 (Very Good)
       • Improvement Rate: +16.7 ELO/benchmark

    💡 Insights:
       • Excellent value for money
       • Steady improvement over time
       • Consistent performance
       • Ideal for development use
    """

    console.print(Panel(sample_analysis.strip(), border_style="green"))
    console.print()


async def main():
    """Run the complete budget and ranking demo."""
    console.print("[bold blue]💰 🏆 Chess LLM Budget & Ranking System Demo[/bold blue]")
    console.print()

    # Check environment
    console.print("[dim]Checking environment...[/dim]")
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Gemini": os.getenv("GEMINI_API_KEY")
    }

    available_providers = [name for name, key in api_keys.items() if key]
    if available_providers:
        console.print(f"[green]✅ Available providers: {', '.join(available_providers)}[/green]")
    else:
        console.print("[yellow]⚠️ No API keys found - demo will use simulated data[/yellow]")
    console.print()

    # Run all demo sections
    demo_sections = [
        ("Pricing Information", show_pricing_information),
        ("Budget Scenarios", demo_budget_scenarios),
        ("Live Budget Demo", run_budget_demo),
        ("Ranking Features", demonstrate_ranking_features),
        ("Efficiency Insights", show_efficiency_insights),
        ("Historical Analysis", show_historical_analysis),
    ]

    for section_name, demo_func in demo_sections:
        try:
            if asyncio.iscoroutinefunction(demo_func):
                await demo_func()
            else:
                demo_func()
        except Exception as e:
            console.print(f"[red]Error in {section_name} demo: {e}[/red]")

        console.print("-" * 80)
        console.print()

    # Final summary
    print_demo_header(
        "🎯 Get Started with Budget & Ranking",
        "Ready to track costs and analyze performance!"
    )

    console.print("[green]Quick Start Commands:[/green]")
    console.print()
    console.print("[cyan]# Run with budget tracking[/cyan]")
    console.print("python main.py --preset budget --budget-limit 5.0 --show-costs")
    console.print()
    console.print("[cyan]# View current leaderboard[/cyan]")
    console.print("python main.py --leaderboard 10")
    console.print()
    console.print("[cyan]# Compare providers[/cyan]")
    console.print("python main.py --provider-stats")
    console.print()
    console.print("[cyan]# Analyze specific model[/cyan]")
    console.print("python main.py --analyze-model openai:gpt-4o-mini")
    console.print()

    console.print("[bold yellow]💡 Pro Tips:[/bold yellow]")
    console.print("  • Always set budget limits when testing new models")
    console.print("  • Use --show-costs to monitor spending in real-time")
    console.print("  • Check leaderboard regularly to see top performers")
    console.print("  • Analyze cost efficiency to optimize your budget")
    console.print("  • Track trends over time for better insights")
    console.print()

    console.print("[bold green]Happy benchmarking with cost control! 💰🤖♟️[/bold green]")


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(main())
