"""Inspect local chess observations and estimate tool-assisted play without API calls."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess
from chess_llm_bench.llm.agents.chess_tools import ChessAnalysisTools
from chess_llm_bench.core.estimate import estimate_run
from chess_llm_bench.core.models import Config
from chess_llm_bench.llm.models import PRESET_CONFIGS

if __name__ == "__main__":
    tools = ChessAnalysisTools(chess.Board())
    print(
        json.dumps(
            {
                "material": tools.evaluate_material(),
                "position": tools.evaluate_position(),
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            estimate_run(
                Config(use_agent=True, max_games=2),
                PRESET_CONFIGS["budget"]["bots"],
                input_tokens=1600,
            ),
            indent=2,
        )
    )
