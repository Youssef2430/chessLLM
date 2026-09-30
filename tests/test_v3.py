"""Protocol 3 invariants. All provider responses are simulated, never billed."""

import asyncio
import json
import os
import shutil
from datetime import datetime, timezone

import chess
import pytest

from chess_llm_bench.core.journal import RunLease, append_row, atomic_json, read_rows
from chess_llm_bench.core.models import BotSpec
from chess_llm_bench.core.moves import move_schema, resolve_move
from chess_llm_bench.llm.client import InvalidMoveError, LLMProviderError
from chess_llm_bench.llm.decision import DecisionClient
from chess_llm_bench.llm.subscription import SubscriptionProvider, parse_events
from chess_llm_bench.observatory import read_run, resolve_run, run_health
from chess_llm_bench.runner import RunConfig, recover_plies, run, schedule
from chess_llm_bench.subscription_run import export_data


@pytest.mark.parametrize(
    "answer,method",
    [
        ("e2e4", "strict"),
        ('{"move":"e2e4"}', "json"),
        ("I considered d2d4.\ne2e4", "final_line"),
        ("Checks considered.\nFinal move: `e2e4`", "marked_final"),
        ("My choice:\n```uci\ne2e4\n```", "code_block"),
    ],
)
def test_recovers_only_explicit_final_choice(answer, method):
    result = resolve_move(answer, chess.Board())
    assert result.move.uci() == "e2e4"
    assert result.method == method
    assert result.strict_valid == (method == "strict")


@pytest.mark.parametrize(
    "answer",
    [
        "e2e4 or d2d4",
        "Do not play e2e4",
        "e2e4 is bad. d2d4 is better.",
        "```\ne2e4\nd2d4\n```",
        '{"moves":["e2e4","d2d4"]}',
        "a1a8",
    ],
)
def test_never_guesses_or_substitutes_a_legal_move(answer):
    assert resolve_move(answer, chess.Board()).move is None


def test_promotions_and_schema():
    board = chess.Board("8/P7/8/8/8/8/6k1/4K3 w - - 0 1")
    assert resolve_move("move: a7a8n", board).move.uci() == "a7a8n"
    assert resolve_move("a7a8", board).error == "illegal_move"
    assert set(move_schema(board)["properties"]["move"]["enum"]) == {
        m.uci() for m in board.legal_moves
    }


def fake_client(tmp_path, answers, **kwargs):
    provider = SubscriptionProvider(BotSpec("claude", "claude-opus-5-5", "opus"))
    calls = []

    async def transport(prompt, timeout):
        calls.append(prompt)
        item = answers[len(calls) - 1]
        if isinstance(item, Exception):
            raise item
        return 0, json.dumps(dict(type="result", result=item, usage={})), ""

    provider.transport = transport
    return DecisionClient(provider, tmp_path, asyncio.Semaphore(1), **kwargs), calls


@pytest.mark.asyncio
async def test_correction_then_restart_reuses_saved_answer(tmp_path):
    client, calls = fake_client(tmp_path, ["a1a8", "I choose:\ne2e4"])
    assert (await client.choose(chess.Board(), "1")).uci() == "e2e4"
    assert len(calls) == 2
    assert "illegal_move" in calls[1]
    rows = read_rows(tmp_path / "requests.jsonl")
    assert [r["legal"] for r in rows] == [False, True]
    assert rows[1]["format_recovered"]
    resumed, no_calls = fake_client(tmp_path, [])
    assert (await resumed.choose(chess.Board(), "1")).uci() == "e2e4"
    assert not no_calls


@pytest.mark.asyncio
async def test_invalid_forfeit_only_after_three_attempts(tmp_path):
    client, calls = fake_client(tmp_path, ["a1a8"] * 3)
    with pytest.raises(InvalidMoveError):
        await client.choose(chess.Board(), "1")
    assert len(calls) == 3
    assert len(read_rows(tmp_path / "requests.jsonl")) == 3
    assert not list((tmp_path / "pending").glob("*.json"))


@pytest.mark.asyncio
async def test_quota_stops_without_burning_correction_attempts(tmp_path):
    client, calls = fake_client(tmp_path, [LLMProviderError("rate limit 429")])
    with pytest.raises(LLMProviderError):
        await client.choose(chess.Board(), "1")
    assert len(calls) == 1
    row = read_rows(tmp_path / "requests.jsonl")[0]
    assert row["legal"] is None and row["error_category"] == "quota"


@pytest.mark.asyncio
async def test_timeout_is_service_failure_and_preserves_partial(tmp_path):
    error = TimeoutError("timed out")
    error.stdout = "partial response"
    client, _ = fake_client(tmp_path, [error], max_attempts=1)
    with pytest.raises(LLMProviderError):
        await client.choose(chess.Board(), "1")
    row = read_rows(tmp_path / "requests.jsonl")[0]
    assert row["error_category"] == "timeout" and row["legal"] is None
    assert (
        json.loads((tmp_path / row["partial_trace_path"]).read_text())["stdout"]
        == "partial response"
    )


def test_structured_serialization_tool_is_allowed_but_bash_is_not():
    events = [
        dict(
            type="assistant",
            message=dict(
                model="claude-opus-5-5",
                content=[dict(type="tool_use", name="StructuredOutput")],
            ),
        ),
        dict(type="result", structured_output={"move": "e2e4"}, usage={}),
    ]
    raw = "\n".join(map(json.dumps, events))
    result = parse_events("claude", raw, "", "claude-opus-5-5", structured=True)
    assert result["error"] is None and json.loads(result["response"])["move"] == "e2e4"
    events[0]["message"]["content"][0]["name"] = "Bash"
    assert parse_events(
        "claude",
        "\n".join(map(json.dumps, events)),
        "",
        "claude-opus-5-5",
        structured=True,
    )["error"]


def test_high_antigravity_variant_checked_exactly():
    raw = json.dumps(dict(type="acp_trace", native_model="gemini-pro-agent"))
    assert (
        parse_events(
            "antigravity", raw, "", "gemini-3.1-pro", expected_native="gemini-pro-agent"
        )["error"]
        is None
    )
    assert parse_events(
        "antigravity", raw, "", "gemini-3.1-pro", expected_native="gemini-3.1-pro-low"
    )["error"]


def test_incomplete_tail_preserved_and_complete_corruption_rejected(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"id":1}\n{"id":')
    assert read_rows(path) == [{"id": 1}]
    append_row(path, {"id": 2})
    assert read_rows(path) == [{"id": 1}, {"id": 2}]
    assert path.with_suffix(".jsonl.partial").read_bytes() == b'{"id":'
    path.write_bytes(b"{broken}\n")
    with pytest.raises(json.JSONDecodeError):
        read_rows(path)


def test_exclusive_lease_releases(tmp_path):
    with RunLease(tmp_path):
        with pytest.raises(RuntimeError):
            with RunLease(tmp_path):
                pass
    with RunLease(tmp_path):
        pass


def test_checkpoint_replays_already_journaled_ply(tmp_path):
    board = chess.Board()
    before = board.fen()
    board.push_uci("e2e4")
    append_row(
        tmp_path / "plies.jsonl",
        dict(
            bot="opus",
            game_id="1",
            ply=1,
            fen=before,
            fen_after=board.fen(),
            move_uci="e2e4",
            wall_seconds=2,
        ),
    )
    state = dict(game_id="1", history=[], fen=before)
    assert recover_plies(tmp_path, "opus", state).fen() == board.fen()
    assert state["history"] == ["e2e4"] and state["active_seconds"] == 2
    recover_plies(tmp_path, "opus", state)
    assert state["active_seconds"] == 2


def test_health_and_path_safety(tmp_path):
    atomic_json(tmp_path / "manifest.json", {})
    assert run_health(tmp_path, {})["status"] == "interrupted"
    atomic_json(
        tmp_path / "run_status.json",
        dict(
            status="running",
            pid=os.getpid(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            models={},
        ),
    )
    assert run_health(tmp_path, {})["status"] == "running"
    for name in ("../outside", "/etc", ""):
        with pytest.raises(ValueError):
            resolve_run(tmp_path, name)


def test_summary_replaces_aborted_game_and_preserves_unknown_usage(tmp_path):
    atomic_json(
        tmp_path / "manifest.json",
        dict(protocol="3-subscription", requested_models=["opus"]),
    )
    game = dict(
        bot="opus",
        game_id="1",
        color_llm_white=True,
        result="*",
        termination="provider_error",
    )
    append_row(tmp_path / "games.jsonl", game)
    append_row(
        tmp_path / "games.jsonl", dict(game, result="1-0", termination="checkmate")
    )
    row = dict(
        request_id="a",
        bot="opus",
        game_id="1",
        ply=1,
        side_to_move="white",
        provider="claude",
        usage={},
        wall_seconds=None,
        legal=None,
        error="interrupted",
    )
    append_row(tmp_path / "requests.jsonl", row)
    result = read_run(tmp_path)["models"][0]
    assert result["games"] == result["wins"] == 1
    assert result["incomplete"] == 0 and result["input_tokens"] is None
    assert export_data(tmp_path)["opus"]["median_request_seconds"] is None


def test_schedule_pairs_and_immutable_configuration():
    fixtures = schedule(RunConfig(games=10))
    assert len(fixtures) == 10
    for i in range(0, 10, 2):
        assert fixtures[i]["opening_moves"] == fixtures[i + 1]["opening_moves"]
        assert fixtures[i]["white"] and not fixtures[i + 1]["white"]
    with pytest.raises(ValueError):
        RunConfig(games=3)


@pytest.mark.asyncio
@pytest.mark.skipif(not shutil.which("stockfish"), reason="Local Stockfish required")
async def test_offline_run_resume_never_duplicates_games_or_decisions(tmp_path):
    directory = tmp_path / "cohort"
    config = RunConfig(offline=True, games=2, max_plies=8, think_time=0.01)
    assert await run(directory, config) == "finished"
    before = (directory / "requests.jsonl").read_bytes()
    assert await run(directory, config, resume=True) == "finished"
    assert (directory / "requests.jsonl").read_bytes() == before
    assert len(read_rows(directory / "games.jsonl")) == 2
    manifest = json.loads((directory / "manifest.json").read_text())
    assert len(manifest["executions"]) == 2


@pytest.mark.parametrize("indent", [None, 2])
def test_claude_verbose_json_array_retains_output_model_and_usage(indent):
    events = [
        dict(type="system", subtype="init", model="claude-opus-5-5"),
        dict(
            type="assistant",
            message=dict(content=[dict(type="tool_use", name="StructuredOutput")]),
        ),
        dict(
            type="result",
            result="",
            structured_output={"move": "e2e4"},
            usage=dict(input_tokens=12, output_tokens=3),
        ),
    ]
    result = parse_events(
        "claude",
        json.dumps(events, indent=indent),
        "",
        "claude-opus-5-5",
        structured=True,
    )
    assert result["error"] is None
    assert result["reported_model"] == "claude-opus-5-5"
    assert result["usage"]["output_tokens"] == 3
    assert json.loads(result["response"])["move"] == "e2e4"


def test_missing_client_events_are_transport_failure_not_move_forfeit():
    result = parse_events(
        "claude", "client failure without JSON", "", "claude-opus-5-5"
    )
    assert result["error"] and result["response"] == ""


@pytest.mark.asyncio
async def test_resume_preserves_tool_violation_block(tmp_path):
    from chess_llm_bench.runner import run_model

    config = RunConfig(games=2)
    atomic_json(
        tmp_path / "state/gpt-6-astra.json",
        dict(
            phase="blocked",
            game_id="1",
            error="Tool use invalidates unaided chess benchmark",
        ),
    )
    statuses = {}
    await run_model(
        tmp_path,
        "gpt-6-astra",
        config,
        schedule(config),
        asyncio.Semaphore(1),
        "/must-not-start-engine",
        {"codex": {"available": True}},
        {},
        statuses,
    )
    assert statuses["gpt-6-astra"]["status"] == "blocked"
    assert not (tmp_path / "requests.jsonl").exists()


def test_cancelled_requests_are_not_service_errors_or_chess_losses(tmp_path):
    atomic_json(
        tmp_path / "manifest.json",
        dict(protocol="3-subscription", requested_models=["opus"]),
    )
    base = dict(
        bot="opus",
        game_id="1",
        ply=1,
        side_to_move="white",
        provider="claude",
        usage={},
        wall_seconds=2,
        legal=None,
    )
    append_row(
        tmp_path / "requests.jsonl",
        dict(
            base,
            request_id="cancelled",
            error="CancelledError",
            error_category="cancelled",
        ),
    )
    append_row(
        tmp_path / "requests.jsonl",
        dict(base, request_id="timeout", error="timed out", error_category="timeout"),
    )
    data = read_run(tmp_path)
    summary = data["models"][0]
    assert summary["interruptions"] == 1 and summary["service_errors"] == 1
    assert summary["losses"] == 0 and summary["forfeits"] == 0
    assert data["requests"][0]["interrupted"] is True
    exported = export_data(tmp_path)["opus"]
    assert exported["provider_errors"] == 1 and exported["interrupted_requests"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("slots", [1, 3])
async def test_codex_requests_overlap_up_to_configured_limit(slots):
    from chess_llm_bench.runner import request_gates

    gates = request_gates(RunConfig(codex_concurrency=slots), ["codex", "claude"])
    entered = []
    release = asyncio.Event()

    async def request(index):
        async with gates["codex"]:
            entered.append(index)
            await release.wait()

    tasks = [asyncio.create_task(request(i)) for i in range(3)]
    try:
        await asyncio.sleep(0)
        assert len(entered) == slots
        assert gates["claude"].request_limit == 1
    finally:
        release.set()
        await asyncio.gather(*tasks)
    assert len(entered) == 3


def test_pilot_defaults_and_concurrency_validation():
    assert RunConfig().games == 2
    assert RunConfig().codex_concurrency == 3
    for value in (0, 4, 1.5, True):
        with pytest.raises(ValueError):
            RunConfig(codex_concurrency=value)
