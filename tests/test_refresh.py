"""Offline regression coverage for protocol v2; never needs provider credentials."""

import asyncio
import io
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

import chess
import chess.pgn
import pytest

from chess_llm_bench.cli import (
    BenchmarkOrchestrator,
    create_argument_parser,
    main_async,
)
from chess_llm_bench.core.budget import (
    BudgetTracker,
    BudgetExceeded,
    UnknownPricing,
    start_budget_tracking,
    stop_budget_tracking,
)
from chess_llm_bench.core.estimate import estimate_run
from chess_llm_bench.core.game import GameRunner, LadderRunner
from chess_llm_bench.core.models import (
    Config,
    BotSpec,
    LiveState,
    LadderStats,
    GameRecord,
)
from chess_llm_bench.core.openings import OpeningBook
from chess_llm_bench.core.results import ResultsDatabase
from chess_llm_bench.llm.client import (
    APIProvider,
    BaseLLMProvider,
    Completion,
    LLMClient,
    LLMProviderError,
    InvalidMoveError,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    parse_bot_spec,
)
from chess_llm_bench.llm.models import PRESET_CONFIGS


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    # SDK tests only exercise request building/response accounting with mocks.
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("Network is forbidden in protocol regression tests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    yield
    stop_budget_tracking()


@pytest.mark.parametrize("value", [0, -1, 3])
def test_invalid_game_count(value):
    with pytest.raises(ValueError):
        Config(max_games=value)


@pytest.mark.parametrize(
    "bots", ["random::same,random::same", "openai::bot", "random::../escape"]
)
def test_invalid_lineup(bots):
    with pytest.raises(ValueError):
        parse_bot_spec(bots)


def test_openings_are_reproducible_and_paired():
    one, two = OpeningBook(42), OpeningBook(42)
    assert [one.get_balanced_pair() for _ in range(10)] == [
        two.get_balanced_pair() for _ in range(10)
    ]
    a, b = one.get_balanced_pair()
    assert a == b


def test_exact_pricing_and_no_free_unknowns():
    tracker = BudgetTracker()
    assert tracker.get_pricing("openai", "gpt-4o-mini-unknown") is None
    assert (
        tracker.get_pricing("gemini", "gemini-2.5-pro").output_cost_per_1k_tokens
        == 0.01
    )
    with pytest.raises(UnknownPricing):
        tracker.reserve("openai", "unpriced", 1, 1)


def test_budget_reserves_concurrent_requests_and_zero_limit():
    tracker = BudgetTracker(0.015)
    tracker.start_tracking()
    amount = tracker.reserve("openai", "gpt-6-astra", 1000, 0)
    with pytest.raises(BudgetExceeded):
        tracker.reserve("openai", "gpt-6-astra", 1000, 0)
    tracker.release(amount)
    tracker.reserve("openai", "gpt-6-astra", 1000, 0)
    zero = BudgetTracker(0)
    zero.start_tracking()
    with pytest.raises(BudgetExceeded):
        zero.reserve("openai", "gpt-6-astra", 1, 1)


def test_estimate_scales_and_counts_reasoning():
    bots = PRESET_CONFIGS["latest"]["bots"]
    ten = estimate_run(Config(max_games=10), bots)
    twenty = estimate_run(Config(max_games=20), bots)
    assert ten["scenario_usd"] == pytest.approx(28.8336)
    assert twenty["scenario_usd"] == pytest.approx(2 * ten["scenario_usd"])
    assert ten["high_scenario_usd"] > ten["scenario_usd"]
    assert (
        estimate_run(Config(), [BotSpec("random", "", "baseline")])["scenario_usd"] == 0
    )


@pytest.mark.asyncio
async def test_estimate_does_not_initialize_clients_or_engines():
    args = create_argument_parser().parse_args(["--estimate-cost"])
    with patch(
        "chess_llm_bench.cli.BenchmarkOrchestrator",
        side_effect=AssertionError("paid path"),
    ):
        assert await main_async(args) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ["", "e2e5", "e2e4 e2e3", "Play e2e4", "e4"])
async def test_invalid_output_is_not_repaired_or_replaced(response):
    client = LLMClient(BotSpec("random", "", "test"))
    client.provider.generate_move = AsyncMock(return_value=response)
    with pytest.raises(InvalidMoveError):
        await client.pick_move(chess.Board())
    assert client.get_move_stats()[1] == 1


@pytest.mark.asyncio
async def test_history_includes_opening_and_provider_error_is_not_illegal():
    client = LLMClient(BotSpec("random", "", "test"))
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    client.provider.generate_move = AsyncMock(side_effect=LLMProviderError("offline"))
    with pytest.raises(LLMProviderError):
        await client.pick_move(board)
    assert client.provider.generate_move.call_args.args[-1] == ["e2e4", "e7e5"]
    assert client.get_move_stats()[1] == 0


def fake_provider():
    provider = APIProvider(BotSpec("openai", "gpt-6-luna", "test"))
    provider._request = AsyncMock(return_value=Completion("e2e4", 100, 512))
    return provider


@pytest.mark.asyncio
async def test_usage_records_reported_tokens_once_and_context(tmp_path):
    tracker = start_budget_tracking(1)
    provider = fake_provider()
    provider.usage_context = {"game_id": "2", "move_number": 8}
    assert await provider.complete("test", 0, 1) == "e2e4"
    assert tracker.summary.total_requests == 1
    assert tracker.summary.total_output_tokens == 512
    assert tracker.usage_records[0].usage_source == "reported"
    assert tracker.usage_records[0].game_id == "2"
    assert tracker.reserved_cost == 0
    tracker.save_budget_report(tmp_path / "budget.json")
    report = json.loads((tmp_path / "budget.json").read_text())
    assert report["usage_records"][0]["output_tokens"] == 512


@pytest.mark.asyncio
async def test_budget_stops_before_call():
    start_budget_tracking(0)
    provider = fake_provider()
    with pytest.raises(BudgetExceeded):
        await provider.complete("test", 0, 1)
    provider._request.assert_not_awaited()


@pytest.mark.asyncio
async def test_timeout_is_not_retried_or_reported_free():
    tracker = start_budget_tracking(1)
    provider = fake_provider()

    async def slow(*args):
        await asyncio.sleep(1)

    provider._request.side_effect = slow
    with pytest.raises(LLMProviderError):
        await provider.complete("test", 0, 0.001)
    provider._request.assert_awaited_once()
    assert tracker.summary.total_requests == 1
    assert tracker.summary.total_cost > 0
    assert tracker.usage_records[0].usage_source == "uncertain_upper_estimate"
    assert tracker.reserved_cost == 0


def sdk_provider(kind, model):
    provider = kind.__new__(kind)
    BaseLLMProvider.__init__(provider, BotSpec("test", model, "test"))
    return provider


@pytest.mark.asyncio
async def test_openai_reasoning_request_and_usage():
    provider = sdk_provider(OpenAIProvider, "gpt-6-astra")
    create = AsyncMock(
        return_value=NS(
            output_text="e2e4", usage=NS(input_tokens=80, output_tokens=600)
        )
    )
    provider.client = NS(responses=NS(create=create))
    result = await provider._request("prompt", 0, 5)
    kwargs = create.call_args.kwargs
    assert kwargs["reasoning"] == {"effort": "low"}
    assert kwargs["max_output_tokens"] == 2048
    assert "temperature" not in kwargs
    assert result.output_tokens == 600


@pytest.mark.asyncio
async def test_anthropic_reasoning_request_and_usage():
    provider = sdk_provider(AnthropicProvider, "claude-sonnet-5-5")
    create = AsyncMock(
        return_value=NS(
            content=[NS(type="thinking"), NS(type="text", text="e2e4")],
            usage=NS(input_tokens=80, output_tokens=600, cache_read_input_tokens=20),
        )
    )
    provider.client = NS(messages=NS(create=create))
    result = await provider._request("prompt", 0, 5)
    assert "temperature" not in create.call_args.kwargs
    assert create.call_args.kwargs["thinking"] == {"type": "adaptive"}
    assert result.input_tokens == 100
    assert result.output_tokens == 600


@pytest.mark.asyncio
async def test_gemini_counts_thought_tokens_without_using_thought_text():
    pytest.importorskip("google.genai", reason="requires the optional Gemini SDK")
    provider = sdk_provider(GeminiProvider, "gemini-3.8-flash")
    generate = AsyncMock(
        return_value=NS(
            usage_metadata=NS(
                prompt_token_count=100,
                candidates_token_count=8,
                thoughts_token_count=500,
            ),
            candidates=[
                NS(
                    content=NS(
                        parts=[
                            NS(text="thinking...", thought=True),
                            NS(text="e2e4", thought=False),
                        ]
                    )
                )
            ],
        )
    )
    provider.client = NS(aio=NS(models=NS(generate_content=generate)))
    result = await provider._request("prompt", 0, 5)
    assert result.text == "e2e4"
    assert result.output_tokens == 508
    assert (
        generate.call_args.kwargs["config"].thinking_config.thinking_level.value
        == "LOW"
    )


@pytest.mark.asyncio
async def test_random_schedule_without_engine_and_repeatable_pgn(tmp_path):
    async def run(path):
        config = Config(max_games=4, fixed_opponent_elo=0, max_plies=12)
        client = LLMClient(BotSpec("random", "", "test"))
        state, stats = LiveState("test"), LadderStats()
        await LadderRunner(GameRunner(client, None, config), config).run_ladder(
            path, state, stats
        )
        return stats

    one, two = await run(tmp_path / "a"), await run(tmp_path / "b")
    assert len(one.games) == 4
    assert [g.color_llm_white for g in one.games] == [True, False, True, False]
    assert one.games[0].opening == one.games[1].opening
    assert one.games[2].opening == one.games[3].opening
    for a, b in zip(one.games, two.games):
        pa = chess.pgn.read_game(io.StringIO(a.path.read_text()))
        pb = chess.pgn.read_game(io.StringIO(b.path.read_text()))
        assert list(pa.mainline_moves()) == list(pb.mainline_moves())
        assert "GameDuration" in pa.headers
        assert pa.headers["Result"] == "*"
    assert one.draws == 0 and one.aborted == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,termination,result",
    [
        (InvalidMoveError("bad move"), "invalid_move", "0-1"),
        (LLMProviderError("offline"), "provider_error", "*"),
        (BudgetExceeded("no budget"), "budget_exhausted", "*"),
    ],
)
async def test_forfeit_vs_infrastructure_abort(tmp_path, failure, termination, result):
    client = LLMClient(BotSpec("random", "", "test"))
    client.pick_move = AsyncMock(side_effect=failure)
    record = await GameRunner(client, None, Config(fixed_opponent_elo=0)).play_game(
        0, tmp_path, LiveState("test"), True
    )
    assert record.termination == termination
    assert record.result == result
    pgn = chess.pgn.read_game(io.StringIO(record.path.read_text()))
    assert pgn.headers["Result"] == result
    assert pgn.headers["Termination"] == termination


@pytest.mark.asyncio
async def test_draw_claim_ends_game_without_more_model_calls(tmp_path):
    client = LLMClient(BotSpec("random", "", "test"))
    client.pick_move = AsyncMock(side_effect=AssertionError("game already drawn"))
    opening = ["g1f3", "g8f6", "f3g1", "f6g8"] * 2
    record = await GameRunner(client, None, Config(fixed_opponent_elo=0)).play_game(
        0, tmp_path, LiveState("test"), True, opening
    )
    assert record.result == "1/2-1/2"
    assert record.termination == "threefold_repetition"
    client.pick_move.assert_not_awaited()


@pytest.mark.asyncio
async def test_orchestrator_artifacts_and_database(tmp_path):
    config = Config(
        bots="random::test",
        fixed_opponent_elo=0,
        max_games=2,
        max_plies=12,
        output_dir=str(tmp_path),
        results_db=str(tmp_path / "results.db"),
    )
    with patch(
        "chess_llm_bench.cli.autodetect_stockfish",
        side_effect=AssertionError("not needed"),
    ):
        result = await BenchmarkOrchestrator(config).run_benchmark()
    assert len(result.bot_results["test"].games) == 2
    for file in (
        "config.json",
        "summary.json",
        "budget.json",
        "games.jsonl",
        "environment.json",
        "run.log",
    ):
        assert (result.output_dir / file).exists()
    with sqlite3.connect(config.results_db) as conn:
        assert conn.execute("select count(*) from game_records").fetchone()[0] == 2
        assert conn.execute("select total_games from benchmarks").fetchone()[0] == 0
    assert (
        json.loads((result.output_dir / "summary.json").read_text())["bots"]["test"][
            "aborted"
        ]
        == 2
    )


def test_incomplete_games_do_not_count_as_draws_or_wins():
    stats = LadderStats()
    stats.add_game(GameRecord(0, True, "*", 12, Path("none"), llm_requests=5))
    stats.add_game(GameRecord(0, True, "1-0", 20, Path("none"), llm_requests=8))
    assert stats.total_games == 1 and stats.aborted == 1 and stats.draws == 0
    stats.add_timing_stats(26, 0)
    assert stats.average_move_time == 2
    assert stats.win_rate_interval[0] < 1


def test_leaderboard_rejects_sql_identifiers(tmp_path):
    with pytest.raises(ValueError):
        ResultsDatabase(tmp_path / "db").get_leaderboard(
            metric="win_rate; DROP TABLE benchmarks"
        )


@pytest.mark.asyncio
async def test_partial_initialization_closes_prior_client_and_saves_diagnostics(
    tmp_path,
):
    config = Config(
        bots="random::first,random::second",
        fixed_opponent_elo=0,
        output_dir=str(tmp_path),
        results_db=str(tmp_path / "db"),
    )
    client = LLMClient(BotSpec("random", "", "first"))
    client.close = AsyncMock()
    with patch(
        "chess_llm_bench.cli.LLMClient",
        side_effect=[client, LLMProviderError("missing key")],
    ):
        # Patch cleanup independently: isinstance cannot use a mocked constructor.
        runner = BenchmarkOrchestrator(config)

        async def cleanup():
            for created in runner.clients.values():
                await created.close()

        runner._cleanup_components = cleanup
        with pytest.raises(LLMProviderError):
            await runner.run_benchmark()
    client.close.assert_awaited_once()
    assert len(list(tmp_path.glob("*/budget.json"))) == 1
    assert len(list(tmp_path.glob("*/summary.json"))) == 1


@pytest.mark.asyncio
async def test_request_cancellation_releases_reservation_and_keeps_uncertain_usage():
    tracker = start_budget_tracking(1)
    provider = fake_provider()
    started = asyncio.Event()

    async def pending(*args):
        started.set()
        await asyncio.Event().wait()

    provider._request.side_effect = pending
    task = asyncio.create_task(provider.complete("test", 0, 10))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert tracker.reserved_cost == 0
    assert tracker.summary.total_requests == 1
    assert tracker.usage_records[0].usage_source == "uncertain_upper_estimate"


@pytest.mark.asyncio
async def test_provider_failure_produces_cli_error_and_incomplete_record(tmp_path):
    args = create_argument_parser().parse_args(
        [
            "--bots",
            "random::failed",
            "--max-games",
            "2",
            "--output-dir",
            str(tmp_path),
            "--results-db",
            str(tmp_path / "db"),
        ]
    )
    with patch(
        "chess_llm_bench.llm.client.RandomProvider.generate_move",
        new=AsyncMock(side_effect=LLMProviderError("offline")),
    ):
        assert await main_async(args) == 1
    summary = json.loads(next(tmp_path.glob("*/summary.json")).read_text())
    assert summary["bots"]["failed"]["aborted"] == 1
    assert summary["bots"]["failed"]["wins"] == 0
    assert summary["bots"]["failed"]["draws"] == 0


@pytest.mark.asyncio
async def test_checkmate_at_move_limit_is_still_a_win(tmp_path):
    client = LLMClient(BotSpec("random", "", "test"))
    client.pick_move = AsyncMock(side_effect=AssertionError("already mate"))
    opening = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]
    record = await GameRunner(
        client, None, Config(fixed_opponent_elo=0, max_plies=8)
    ).play_game(0, tmp_path, LiveState("test"), True, opening)
    assert record.result == "1-0" and record.termination == "checkmate"


def test_invalid_move_cannot_be_pushed():
    runner = GameRunner(
        LLMClient(BotSpec("random", "", "test")), None, Config(fixed_opponent_elo=0)
    )
    board = chess.Board()
    with pytest.raises(ValueError):
        runner._execute_move(
            board, chess.Move.from_uci("e2e5"), None, LiveState("test"), "test"
        )
    assert board == chess.Board()
