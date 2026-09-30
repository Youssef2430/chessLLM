"""Protocol-v2 examples. Importing this file never starts a benchmark."""

from chess_llm_bench.core.models import Config
from chess_llm_bench.llm.models import PRESET_CONFIGS, format_bot_spec_string

QUICK_CONFIG = Config(
    bots="random::baseline",
    fixed_opponent_elo=0,
    max_games=2,
    max_plies=12,
)
STANDARD_CONFIG = Config(
    bots=format_bot_spec_string(PRESET_CONFIGS["budget"]["bots"]),
    fixed_opponent_elo=0,
    max_games=10,
    max_plies=200,
    max_output_tokens=2048,
    reasoning_effort="low",
    llm_timeout=60,
    seed=42,
    budget_limit=10,
    show_costs=True,
)
RESEARCH_CONFIG = Config(
    bots=format_bot_spec_string(PRESET_CONFIGS["latest"]["bots"]),
    fixed_opponent_elo=1600,
    max_games=30,
    max_plies=300,
    max_output_tokens=4096,
    reasoning_effort="medium",
    llm_timeout=120,
    seed=42,
    budget_limit=150,
    show_costs=True,
)

if __name__ == "__main__":
    import json
    from chess_llm_bench.core.estimate import estimate_run
    from chess_llm_bench.llm.client import parse_bot_spec

    print(
        json.dumps(
            estimate_run(STANDARD_CONFIG, parse_bot_spec(STANDARD_CONFIG.bots)),
            indent=2,
        )
    )
