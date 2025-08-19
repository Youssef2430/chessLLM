"""
Budget tracking and cost estimation for Chess LLM Benchmark.

This module provides comprehensive budget tracking, cost estimation, and spending
monitoring for LLM API usage during chess benchmarks.

Features:
- Real-time cost tracking across all providers
- Token estimation and usage monitoring
- Budget limits with warnings and alerts
- Cost per game/move analysis
- Provider-specific pricing models
- CLI dashboard integration
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text


@dataclass
class ProviderPricing:
    """Pricing model for an LLM provider."""

    name: str
    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float
    base_cost_per_request: float = 0.0
    currency: str = "USD"

    def calculate_cost(self, input_tokens: int, output_tokens: int, requests: int = 1) -> float:
        """Calculate total cost for given usage."""
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k_tokens
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k_tokens
        request_cost = requests * self.base_cost_per_request
        return input_cost + output_cost + request_cost


# Current pricing as of 2024/2025 (prices may change)
PROVIDER_PRICING = {
    "openai": {
        "gpt-4o": ProviderPricing("GPT-4o", 0.0025, 0.01),
        "gpt-4o-mini": ProviderPricing("GPT-4o Mini", 0.00015, 0.0006),
        "gpt-4-turbo": ProviderPricing("GPT-4 Turbo", 0.01, 0.03),
        "gpt-4": ProviderPricing("GPT-4", 0.03, 0.06),  # Classic GPT-4
        "gpt-3.5-turbo": ProviderPricing("GPT-3.5 Turbo", 0.0005, 0.0015),
        "gpt-3.5": ProviderPricing("GPT-3.5", 0.0005, 0.0015),  # Alias
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022": ProviderPricing("Claude 3.5 Sonnet", 0.003, 0.015),
        "claude-3-5-sonnet": ProviderPricing("Claude 3.5 Sonnet", 0.003, 0.015),  # Alias
        "claude-3-5-haiku-20241022": ProviderPricing("Claude 3.5 Haiku", 0.0008, 0.004),
        "claude-3-5-haiku": ProviderPricing("Claude 3.5 Haiku", 0.0008, 0.004),  # Alias
        "claude-3-opus-20240229": ProviderPricing("Claude 3 Opus", 0.015, 0.075),
        "claude-3-opus": ProviderPricing("Claude 3 Opus", 0.015, 0.075),  # Alias
        "claude-3-haiku-20240307": ProviderPricing("Claude 3 Haiku", 0.00025, 0.00125),
        "claude-3-haiku": ProviderPricing("Claude 3 Haiku", 0.00025, 0.00125),  # Alias
        "claude-3": ProviderPricing("Claude 3", 0.003, 0.015),  # Generic Claude 3
    },
    "gemini": {
        "gemini-1.5-pro": ProviderPricing("Gemini 1.5 Pro", 0.00125, 0.005),
        "gemini-1.5-flash": ProviderPricing("Gemini 1.5 Flash", 0.000075, 0.0003),
        "gemini-1.0-pro": ProviderPricing("Gemini 1.0 Pro", 0.0005, 0.0015),
        "gemini-pro": ProviderPricing("Gemini Pro", 0.0005, 0.0015),  # Alias for gemini-1.0-pro
        "gemini": ProviderPricing("Gemini", 0.0005, 0.0015),  # Generic alias
    },
    "random": {
        "random": ProviderPricing("Random Bot", 0.0, 0.0),
        "": ProviderPricing("Random Bot", 0.0, 0.0),  # Empty model name for random
    }
}


@dataclass
class UsageRecord:
    """Record of API usage for a specific call."""

    timestamp: datetime
    provider: str
    model: str
    bot_name: str
    input_tokens: int
    output_tokens: int
    cost: float
    game_id: Optional[str] = None
    elo_level: Optional[int] = None
    move_number: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class BudgetSummary:
    """Summary of budget usage and costs."""

    total_cost: float = 0.0
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    costs_by_provider: Dict[str, float] = field(default_factory=dict)
    costs_by_model: Dict[str, float] = field(default_factory=dict)
    costs_by_bot: Dict[str, float] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration(self) -> Optional[timedelta]:
        """Total duration of the benchmark."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def cost_per_minute(self) -> float:
        """Cost per minute during the benchmark."""
        if self.duration and self.duration.total_seconds() > 0:
            return self.total_cost / (self.duration.total_seconds() / 60)
        return 0.0


class BudgetTracker:
    """
    Tracks and monitors budget usage during chess LLM benchmarks.

    Provides real-time cost tracking, budget limits, warnings, and
    comprehensive reporting of API usage and costs.
    """

    def __init__(self, budget_limit: Optional[float] = None, currency: str = "USD"):
        """
        Initialize budget tracker.

        Args:
            budget_limit: Maximum budget in currency units (None for unlimited)
            currency: Currency code (e.g., "USD")
        """
        self.budget_limit = budget_limit
        self.currency = currency
        self.console = Console()

        # Usage tracking
        self.usage_records: List[UsageRecord] = []
        self.summary = BudgetSummary()
        self.warnings_shown = set()

        # State tracking
        self.is_active = False
        self.start_time: Optional[datetime] = None

    def start_tracking(self) -> None:
        """Start budget tracking for a new benchmark."""
        self.is_active = True
        self.start_time = datetime.now()
        self.summary.start_time = self.start_time
        self.usage_records.clear()
        self.summary = BudgetSummary(start_time=self.start_time)
        self.warnings_shown.clear()

        if self.budget_limit:
            self.console.print(f"[green]💰 Budget tracking started (Limit: ${self.budget_limit:.4f})[/green]")
        else:
            self.console.print("[green]💰 Budget tracking started (No limit)[/green]")

    def stop_tracking(self) -> None:
        """Stop budget tracking and finalize summary."""
        if self.is_active:
            self.is_active = False
            self.summary.end_time = datetime.now()
            self.console.print(f"[blue]💰 Budget tracking stopped (Total: ${self.summary.total_cost:.4f})[/blue]")

    def estimate_tokens(self, prompt: str, response: str = "") -> tuple[int, int]:
        """
        Estimate token usage for a prompt and response.

        Uses a simple heuristic: ~4 characters per token for English text.
        This is approximate and may not match exact API token counts.
        """
        input_tokens = max(1, len(prompt) // 4)  # Minimum 1 token
        output_tokens = max(1, len(response) // 4) if response else 8  # Default estimate for chess moves
        return input_tokens, output_tokens

    def get_pricing(self, provider: str, model: str) -> Optional[ProviderPricing]:
        """Get pricing information for a provider/model combination."""
        provider_prices = PROVIDER_PRICING.get(provider.lower())
        if not provider_prices:
            return None

        # Try exact match first
        pricing = provider_prices.get(model)
        if pricing:
            return pricing

        # Try lowercase match
        pricing = provider_prices.get(model.lower())
        if pricing:
            return pricing

        # Try partial matches for common model name variations
        for model_key, pricing in provider_prices.items():
            if model.lower() in model_key.lower() or model_key.lower() in model.lower():
                return pricing

        return None

    def record_usage(
        self,
        provider: str,
        model: str,
        bot_name: str,
        prompt: str,
        response: str = "",
        actual_input_tokens: Optional[int] = None,
        actual_output_tokens: Optional[int] = None,
        game_id: Optional[str] = None,
        elo_level: Optional[int] = None,
        move_number: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> float:
        """
        Record API usage and calculate cost.

        Returns the cost of this API call.
        """
        if not self.is_active:
            return 0.0

        # Get token counts (actual or estimated)
        if actual_input_tokens is not None and actual_output_tokens is not None:
            input_tokens, output_tokens = actual_input_tokens, actual_output_tokens
        else:
            input_tokens, output_tokens = self.estimate_tokens(prompt, response)

        # Calculate cost
        pricing = self.get_pricing(provider, model)
        if pricing:
            cost = pricing.calculate_cost(input_tokens, output_tokens)
        else:
            cost = 0.0  # Unknown pricing, assume free

        # Create usage record
        record = UsageRecord(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            bot_name=bot_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            game_id=game_id,
            elo_level=elo_level,
            move_number=move_number,
            success=success,
            error_message=error_message
        )

        self.usage_records.append(record)
        self._update_summary(record)
        self._check_budget_warnings()

        return cost

    def _update_summary(self, record: UsageRecord) -> None:
        """Update the budget summary with a new usage record."""
        self.summary.total_cost += record.cost
        self.summary.total_requests += 1
        self.summary.total_input_tokens += record.input_tokens
        self.summary.total_output_tokens += record.output_tokens

        # Update by provider
        if record.provider not in self.summary.costs_by_provider:
            self.summary.costs_by_provider[record.provider] = 0.0
        self.summary.costs_by_provider[record.provider] += record.cost

        # Update by model
        model_key = f"{record.provider}:{record.model}"
        if model_key not in self.summary.costs_by_model:
            self.summary.costs_by_model[model_key] = 0.0
        self.summary.costs_by_model[model_key] += record.cost

        # Update by bot
        if record.bot_name not in self.summary.costs_by_bot:
            self.summary.costs_by_bot[record.bot_name] = 0.0
        self.summary.costs_by_bot[record.bot_name] += record.cost

    def _check_budget_warnings(self) -> None:
        """Check budget limits and show warnings if necessary."""
        if not self.budget_limit or self.summary.total_cost == 0:
            return

        percentage = (self.summary.total_cost / self.budget_limit) * 100

        # Warning thresholds
        warnings = [
            (50, "50% budget used"),
            (75, "75% budget used - Consider monitoring"),
            (90, "90% budget used - Approaching limit!"),
            (100, "Budget limit exceeded!"),
        ]

        for threshold, message in warnings:
            if percentage >= threshold and threshold not in self.warnings_shown:
                if threshold >= 100:
                    self.console.print(f"[bold red]🚨 {message} (${self.summary.total_cost:.4f}/${self.budget_limit:.4f})[/bold red]")
                elif threshold >= 90:
                    self.console.print(f"[bold yellow]⚠️  {message} (${self.summary.total_cost:.4f}/${self.budget_limit:.4f})[/bold yellow]")
                else:
                    self.console.print(f"[yellow]💰 {message} (${self.summary.total_cost:.4f}/${self.budget_limit:.4f})[/yellow]")

                self.warnings_shown.add(threshold)

    def get_current_cost(self) -> float:
        """Get current total cost."""
        return self.summary.total_cost

    def get_cost_by_bot(self, bot_name: str) -> float:
        """Get cost for a specific bot."""
        return self.summary.costs_by_bot.get(bot_name, 0.0)

    def get_summary(self) -> BudgetSummary:
        """Get current budget summary."""
        return self.summary

    def create_cost_table(self) -> Table:
        """Create a rich table showing cost breakdown."""
        table = Table(title="💰 Budget Summary")
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Item", style="green")
        table.add_column("Cost", style="yellow", justify="right")
        table.add_column("Usage", style="blue", justify="right")

        # Total cost
        table.add_row(
            "Total",
            "All Usage",
            f"${self.summary.total_cost:.4f}",
            f"{self.summary.total_requests} requests"
        )

        # By provider
        for provider, cost in sorted(self.summary.costs_by_provider.items()):
            if cost > 0:
                table.add_row("Provider", provider.title(), f"${cost:.4f}", "")

        # By model (top 5)
        top_models = sorted(self.summary.costs_by_model.items(), key=lambda x: x[1], reverse=True)[:5]
        for model_key, cost in top_models:
            if cost > 0:
                provider, model = model_key.split(":", 1)
                table.add_row("Model", f"{model} ({provider})", f"${cost:.4f}", "")

        # By bot (top 5)
        top_bots = sorted(self.summary.costs_by_bot.items(), key=lambda x: x[1], reverse=True)[:5]
        for bot_name, cost in top_bots:
            if cost > 0:
                table.add_row("Bot", bot_name, f"${cost:.4f}", "")

        return table

    def create_efficiency_table(self) -> Table:
        """Create a table showing cost efficiency metrics."""
        table = Table(title="📊 Cost Efficiency")
        table.add_column("Bot", style="cyan")
        table.add_column("Cost/Game", style="yellow", justify="right")
        table.add_column("Cost/Token", style="green", justify="right")
        table.add_column("Tokens/Request", style="blue", justify="right")

        # Calculate per-bot metrics
        bot_games = {}
        bot_tokens = {}
        bot_requests = {}

        for record in self.usage_records:
            bot = record.bot_name
            if bot not in bot_games:
                bot_games[bot] = set()
                bot_tokens[bot] = 0
                bot_requests[bot] = 0

            if record.game_id:
                bot_games[bot].add(record.game_id)
            bot_tokens[bot] += record.input_tokens + record.output_tokens
            bot_requests[bot] += 1

        for bot_name, cost in self.summary.costs_by_bot.items():
            if cost > 0:
                games = len(bot_games.get(bot_name, set()))
                tokens = bot_tokens.get(bot_name, 0)
                requests = bot_requests.get(bot_name, 0)

                cost_per_game = cost / games if games > 0 else 0
                cost_per_token = cost / tokens if tokens > 0 else 0
                tokens_per_request = tokens / requests if requests > 0 else 0

                table.add_row(
                    bot_name,
                    f"${cost_per_game:.4f}" if cost_per_game > 0 else "N/A",
                    f"${cost_per_token:.6f}" if cost_per_token > 0 else "N/A",
                    f"{tokens_per_request:.1f}" if tokens_per_request > 0 else "N/A"
                )

        return table

    def display_budget_status(self) -> None:
        """Display current budget status in the console."""
        if not self.is_active or self.summary.total_cost == 0:
            return

        # Create budget panel
        budget_text = f"💰 Current Cost: ${self.summary.total_cost:.4f}"

        if self.budget_limit:
            percentage = (self.summary.total_cost / self.budget_limit) * 100
            remaining = self.budget_limit - self.summary.total_cost
            budget_text += f"\n💳 Budget Limit: ${self.budget_limit:.4f}"
            budget_text += f"\n📊 Usage: {percentage:.1f}%"
            budget_text += f"\n💵 Remaining: ${remaining:.4f}"

            # Color based on usage
            if percentage >= 100:
                style = "bold red"
            elif percentage >= 90:
                style = "bold yellow"
            elif percentage >= 75:
                style = "yellow"
            else:
                style = "green"
        else:
            budget_text += "\n💳 Budget Limit: Unlimited"
            style = "green"

        if self.summary.duration:
            budget_text += f"\n⏱️  Duration: {str(self.summary.duration).split('.')[0]}"
            budget_text += f"\n🔥 Rate: ${self.summary.cost_per_minute:.4f}/min"

        panel = Panel(budget_text, title="Budget Status", border_style=style)
        self.console.print(panel)

    def get_budget_info(self) -> Dict[str, str]:
        """Get budget information for dashboard display."""
        if not self.is_active:
            return {}

        info = {
            "total_cost": f"${self.summary.total_cost:.4f}",
            "api_calls": str(self.summary.total_requests),
            "style": "green"
        }

        if self.summary.total_input_tokens > 0:
            info["input_tokens"] = f"{self.summary.total_input_tokens:,}"
        if self.summary.total_output_tokens > 0:
            info["output_tokens"] = f"{self.summary.total_output_tokens:,}"

        # Budget limit info
        if self.budget_limit:
            percentage = (self.summary.total_cost / self.budget_limit) * 100
            remaining = self.budget_limit - self.summary.total_cost

            info["budget_limit"] = f"${self.budget_limit:.4f}"
            info["usage_percentage"] = f"{percentage:.1f}%"
            info["remaining"] = f"${remaining:.4f}"

            # Color based on usage
            if percentage >= 100:
                info["style"] = "bold red"
            elif percentage >= 90:
                info["style"] = "bold yellow"
            elif percentage >= 75:
                info["style"] = "yellow"
            else:
                info["style"] = "green"
        else:
            info["budget_limit"] = "Unlimited"

        if self.summary.duration:
            info["duration"] = str(self.summary.duration).split('.')[0]
            info["rate_per_minute"] = f"${self.summary.cost_per_minute:.4f}/min"

        return info

    def save_budget_report(self, filepath: Path) -> None:
        """Save detailed budget report to file."""
        report = {
            "summary": {
                "total_cost": self.summary.total_cost,
                "total_requests": self.summary.total_requests,
                "total_input_tokens": self.summary.total_input_tokens,
                "total_output_tokens": self.summary.total_output_tokens,
                "start_time": self.summary.start_time.isoformat() if self.summary.start_time else None,
                "end_time": self.summary.end_time.isoformat() if self.summary.end_time else None,
                "duration_seconds": self.summary.duration.total_seconds() if self.summary.duration else None,
                "cost_per_minute": self.summary.cost_per_minute,
                "currency": self.currency,
                "budget_limit": self.budget_limit,
            },
            "costs_by_provider": self.summary.costs_by_provider,
            "costs_by_model": self.summary.costs_by_model,
            "costs_by_bot": self.summary.costs_by_bot,
            "usage_records": [
                {
                    "timestamp": record.timestamp.isoformat(),
                    "provider": record.provider,
                    "model": record.model,
                    "bot_name": record.bot_name,
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                    "cost": record.cost,
                    "game_id": record.game_id,
                    "elo_level": record.elo_level,
                    "move_number": record.move_number,
                    "success": record.success,
                    "error_message": record.error_message,
                }
                for record in self.usage_records
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

    def load_budget_report(self, filepath: Path) -> None:
        """Load budget report from file."""
        with open(filepath, 'r') as f:
            report = json.load(f)

        # Restore summary
        summary_data = report["summary"]
        self.summary = BudgetSummary(
            total_cost=summary_data["total_cost"],
            total_requests=summary_data["total_requests"],
            total_input_tokens=summary_data["total_input_tokens"],
            total_output_tokens=summary_data["total_output_tokens"],
            costs_by_provider=report["costs_by_provider"],
            costs_by_model=report["costs_by_model"],
            costs_by_bot=report["costs_by_bot"],
            start_time=datetime.fromisoformat(summary_data["start_time"]) if summary_data["start_time"] else None,
            end_time=datetime.fromisoformat(summary_data["end_time"]) if summary_data["end_time"] else None,
        )

        # Restore usage records
        self.usage_records = [
            UsageRecord(
                timestamp=datetime.fromisoformat(record["timestamp"]),
                provider=record["provider"],
                model=record["model"],
                bot_name=record["bot_name"],
                input_tokens=record["input_tokens"],
                output_tokens=record["output_tokens"],
                cost=record["cost"],
                game_id=record["game_id"],
                elo_level=record["elo_level"],
                move_number=record["move_number"],
                success=record["success"],
                error_message=record["error_message"],
            )
            for record in report["usage_records"]
        ]

        self.budget_limit = summary_data.get("budget_limit")
        self.currency = summary_data.get("currency", "USD")


# Global budget tracker instance
_budget_tracker: Optional[BudgetTracker] = None


def get_budget_tracker() -> BudgetTracker:
    """Get the global budget tracker instance."""
    global _budget_tracker
    if _budget_tracker is None:
        _budget_tracker = BudgetTracker()
    return _budget_tracker


def set_budget_limit(limit: float) -> None:
    """Set budget limit for the global tracker."""
    tracker = get_budget_tracker()
    tracker.budget_limit = limit


def start_budget_tracking(budget_limit: Optional[float] = None) -> BudgetTracker:
    """Start budget tracking with optional limit."""
    global _budget_tracker
    _budget_tracker = BudgetTracker(budget_limit)
    _budget_tracker.start_tracking()
    return _budget_tracker


def stop_budget_tracking() -> Optional[BudgetSummary]:
    """Stop budget tracking and return summary."""
    tracker = get_budget_tracker()
    if tracker.is_active:
        tracker.stop_tracking()
        return tracker.get_summary()
    return None


def record_llm_usage(
    provider: str,
    model: str,
    bot_name: str,
    prompt: str,
    response: str = "",
    **kwargs
) -> float:
    """Record LLM usage in the global tracker."""
    tracker = get_budget_tracker()
    return tracker.record_usage(provider, model, bot_name, prompt, response, **kwargs)
