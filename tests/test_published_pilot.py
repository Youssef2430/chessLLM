"""Audit the committed pilot from chess records, without client/network calls."""

import json
from pathlib import Path

import chess
import chess.pgn

from chess_llm_bench.core.journal import read_rows
from chess_llm_bench.observatory import read_run

SNAPSHOT = Path(__file__).resolve().parents[1] / "benchmarks/2026-09-30-pilot"


def test_published_totals_and_complete_analysis_coverage():
    data = read_run(SNAPSHOT)
    assert len(data["requests"]) == 475
    accepted = {r["request_id"] for r in data["requests"] if r.get("legal")}
    assert len(accepted) == 466
    assert {a["request_id"] for a in data["analysis"]} == accepted
    assert len(data["analysis"]) == 466
    assert len(data["games"]) == 13
    assert sum(m["games"] for m in data["models"]) == 11
    assert sum(m["forfeits"] for m in data["models"]) == 0
    assert sum(m["interruptions"] for m in data["models"]) == 3
    assert sum(m["service_errors"] for m in data["models"]) == 3
    assert sum(m["invalid_attempts"] for m in data["models"]) == 3


def test_published_pgns_replay_and_match_recorded_outcomes():
    for row in read_rows(SNAPSHOT / "games.jsonl"):
        path = (SNAPSHOT / row["path"]).resolve()
        assert path.is_relative_to(SNAPSHOT)
        with path.open() as stream:
            game = chess.pgn.read_game(stream)
        assert not game.errors
        board = game.end().board()
        assert board.ply() == row["ply_count"]
        assert game.headers["Result"] == row["result"]
        if row["termination"] == "checkmate":
            assert board.is_checkmate()
            assert board.result() == row["result"]
        else:
            assert row["result"] == "*"
            assert row["termination"] in {"max_plies", "provider_error"}


def test_published_positions_and_moves_are_consistent():
    requests = {r["request_id"]: r for r in read_rows(SNAPSHOT / "requests.jsonl")}
    for row in read_rows(SNAPSHOT / "plies.jsonl"):
        board = chess.Board(row["fen"])
        move = board.parse_uci(row["move_uci"])
        assert board.san(move) == row["move_san"]
        board.push(move)
        assert board.fen() == row["fen_after"]
        if row.get("request_id"):
            request = requests[row["request_id"]]
            assert request["legal"] and request["move_uci"] == row["move_uci"]
            assert request["fen"] == row["fen"]
    for row in requests.values():
        board = chess.Board()
        for move in row["history_uci"]:
            board.push_uci(move)
        assert board.fen() == row["fen"]


def test_public_requests_omit_client_transcripts_and_machine_paths():
    forbidden = {
        "prompt",
        "response",
        "trace_path",
        "partial_trace_path",
        "rate_limits",
        "client_timing",
    }
    for row in read_rows(SNAPSHOT / "requests.jsonl"):
        assert not forbidden.intersection(row)
    for path in SNAPSHOT.rglob("*"):
        if path.suffix in {".json", ".jsonl", ".pgn"}:
            text = path.read_text()
            assert "/Users/" not in text and "/private/var/" not in text
            assert "oauth_token" not in text.lower()
    manifest = json.loads((SNAPSHOT / "manifest.json").read_text())
    assert manifest["settings"]["games"] == 2
    assert manifest["settings"]["codex_concurrency"] == 3
    assert manifest["api_key_fallback"] is False
