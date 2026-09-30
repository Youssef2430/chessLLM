"""Offline cost scenarios for the current benchmark presets."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chess_llm_bench.core.estimate import estimate_run
from chess_llm_bench.core.models import Config
from chess_llm_bench.llm.models import PRESET_CONFIGS

if __name__ == "__main__":
    for preset in ("budget", "latest"):
        print(preset)
        print(
            json.dumps(
                estimate_run(Config(max_games=10), PRESET_CONFIGS[preset]["bots"]),
                indent=2,
            )
        )
