"""Free end-to-end paired schedule; results go to a temporary directory."""

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chess_llm_bench.cli import BenchmarkOrchestrator
from chess_llm_bench.core.models import Config


async def main():
    path = Path(tempfile.mkdtemp(prefix="chess-benchmark-demo-"))
    config = Config(
        bots="random::baseline-a,random::baseline-b",
        fixed_opponent_elo=0,
        max_games=2,
        max_plies=20,
        output_dir=str(path),
        results_db=str(path / "results.db"),
    )
    await BenchmarkOrchestrator(config).run_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
