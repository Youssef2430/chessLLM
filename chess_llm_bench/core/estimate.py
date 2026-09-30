"""Offline scenario estimates; no SDK clients, credentials or API calls."""

from .budget import BudgetTracker, UnknownPricing, PRICING_CHECKED, PRICING_SOURCES


def estimate_run(config, bots, input_tokens=800, output_tokens=512, moves_per_game=60):
    if input_tokens <= 0 or output_tokens <= 0 or moves_per_game <= 0:
        raise ValueError("Estimate token and move counts must be positive")
    if output_tokens > config.max_output_tokens:
        raise ValueError("Estimated output exceeds max_output_tokens")
    max_moves = (config.max_plies + 1) // 2
    moves = min(moves_per_game, max_moves)
    tracker = BudgetTracker()
    rows = []
    for bot in bots:
        pricing = tracker.get_pricing(bot.provider, bot.model)
        if pricing is None:
            raise UnknownPricing(f"No verified price for {bot.provider}:{bot.model}")
        # Scenario inputs are explicitly adjustable, not inferred from old broken bills.
        rows.append(
            {
                "bot": bot.name,
                "model": f"{bot.provider}:{bot.model}",
                "games": config.max_games,
                "scenario_usd": pricing.calculate_cost(input_tokens, output_tokens)
                * moves
                * config.max_games,
                "high_scenario_usd": pricing.calculate_cost(
                    input_tokens * 2, config.max_output_tokens
                )
                * max_moves
                * config.max_games,
                "input_usd_per_million": pricing.input_cost_per_1k_tokens * 1000,
                "output_usd_per_million": pricing.output_cost_per_1k_tokens * 1000,
            }
        )
    return {
        "currency": "USD",
        "pricing_checked": PRICING_CHECKED,
        "sources": PRICING_SOURCES,
        "assumptions": {
            "games_per_model": config.max_games,
            "llm_moves_per_game": moves,
            "input_tokens_per_request": input_tokens,
            "output_tokens_including_reasoning": output_tokens,
            "max_output_tokens": config.max_output_tokens,
            "max_plies": config.max_plies,
            "mode": "tool-assisted" if config.use_agent else "prompt",
            "requests_per_move": 1,
        },
        "models": rows,
        "scenario_usd": sum(row["scenario_usd"] for row in rows),
        "high_scenario_usd": sum(row["high_scenario_usd"] for row in rows),
        "note": "Scenarios, not quotes or guaranteed upper bounds. No cache discounts, tax, credits, retries or regional premiums. Flash promotional rates expire 2026-12-31.",
    }
