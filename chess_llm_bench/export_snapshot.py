"""Export a portable, allowlisted public snapshot without raw client traces."""

import argparse
import hashlib
import json
from pathlib import Path

from .core.journal import atomic_json, latest, read_rows
from .observatory import read_run

REQUEST_FIELDS = """schema_version request_id decision_id attempt bot game_id provider
requested_model reported_model model_verification reasoning_effort structured_requested
prompt_style started_at finished_at ply side_to_move fen history_uci legal_moves
legal_move_count in_check pieces halfmove_clock prompt_sha256 prompt_bytes queue_seconds
provider_concurrency billing_route incremental_charge_usd output_token_cap legal
error_category exit_code strict_uci_valid parse_method validation_error format_recovered
move_uci move_san fen_after capture castling gives_check promotion wall_seconds
api_seconds""".split()
USAGE_FIELDS = (
    """input_tokens output_tokens cached_input_tokens cache_write_input_tokens
cache_read_input_tokens cache_creation_input_tokens reasoning_output_tokens""".split()
)
GAME_FIELDS = """bot game_id pair_id elo color_llm_white result termination ply_count
timestamp opening seed game_duration llm_requests llm_moves format_recoveries retry_requests""".split()
PLY_FIELDS = (
    """bot game_id ply player fen move_uci move_san capture castling gives_check
promotion legal_move_count wall_seconds request_id fen_after""".split()
)
ANALYSIS_FIELDS = """schema_version request_id bot game_id ply side_to_move fen move_uci
move_san node_budget engine engine_sha256 best_move_match centipawn_loss raw_centipawn_loss
expected_score_loss missed_forced_mate cp_loss_bucket material_white material_black
material_advantage nonpawn_material phase pawn_count attacked_own_pieces legal_captures
legal_checks""".split()
SCORE_FIELDS = "cp mate expected_score wdl depth seldepth nodes time pv".split()


def pick(record, fields):
    return {key: record[key] for key in fields if key in record}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def export_snapshot(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    data = read_run(source)
    if data["health"]["status"] == "running":
        raise ValueError("Stop the run before publishing a consistent snapshot")
    if data["manifest"].get("invalidated_reason"):
        raise ValueError("Invalidated runs cannot be published as benchmark results")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = data["manifest"]
    public = pick(
        manifest,
        [
            "protocol",
            "schema_version",
            "started_at",
            "last_stopped_at",
            "settings",
            "requested_models",
            "python",
            "billing_route",
            "api_key_fallback",
            "output_token_cap",
            "temperature",
            "analysis_assistance",
            "config",
            "schedule",
            "source_hashes",
            "analysis_plan",
        ],
    )
    public["source_run"] = source.name
    public["export_policy"] = (
        "Allowlisted chess data; no raw prompts/responses, client traces, authentication details, local paths or process IDs."
    )
    public["engine"] = pick(manifest["engine"], ["name", "native_elo_range", "sha256"])
    public["clients"] = {
        name: pick(value, ["available", "version"])
        for name, value in manifest.get("clients", {}).items()
    }
    public["executions"] = [
        pick(x, ["started_at", "resume", "source_hashes"])
        for x in manifest.get("executions", [])
    ]
    public["amendments"] = [
        pick(
            x,
            [
                "timestamp",
                "reason",
                "original_games",
                "new_games",
                "original_schedule",
                "previous_codex_concurrency",
                "new_codex_concurrency",
            ],
        )
        for x in manifest.get("amendments", [])
    ]
    public["source_data_sha256"] = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in (
            "manifest.json",
            "games.jsonl",
            "requests.jsonl",
            "plies.jsonl",
            "analysis.jsonl",
        )
    }
    atomic_json(destination / "manifest.json", public)
    health = pick(data["health"], ["status", "started_at", "updated_at"])
    health["models"] = {
        name: pick(value, ["status", "games", "game_id"])
        for name, value in data["health"].get("models", {}).items()
    }
    for name, value in health["models"].items():
        if value["status"] == "blocked":
            value["reason"] = (
                "Tool use invalidated the unaided benchmark"
                if name == "gemini-3.1-pro"
                else "See request error categories"
            )
    atomic_json(destination / "run_status.json", health)
    requests = []
    for row in data["requests"]:
        record = pick(row, REQUEST_FIELDS)
        record["usage"] = pick(row.get("usage", {}), USAGE_FIELDS)
        thinking = (
            row.get("usage", {}).get("output_tokens_details", {}).get("thinking_tokens")
        )
        if thinking is not None:
            record["usage"]["output_tokens_details"] = {"thinking_tokens": thinking}
        record["error"] = (
            row.get("error_category", "provider_error") if row.get("error") else None
        )
        requests.append(record)
    write_rows(destination / "requests.jsonl", requests)
    games = []
    for row in latest(read_rows(source / "games.jsonl"), ("bot", "game_id")):
        record = pick(row, GAME_FIELDS)
        path = Path(row["path"]).resolve()
        if not path.is_relative_to(source):
            raise ValueError("PGN outside source run")
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        record["path"] = relative.as_posix()
        games.append(record)
    write_rows(destination / "games.jsonl", games)
    write_rows(
        destination / "plies.jsonl", [pick(r, PLY_FIELDS) for r in data["plies"]]
    )
    quality = []
    for row in data["analysis"]:
        record = pick(row, ANALYSIS_FIELDS)
        record.update(
            best=pick(row["best"], SCORE_FIELDS),
            chosen=pick(row["chosen"], SCORE_FIELDS),
        )
        quality.append(record)
    write_rows(destination / "analysis.jsonl", quality)
    summary = read_run(destination)
    atomic_json(
        destination / "summary.json",
        dict(
            models=summary["models"],
            requests=len(requests),
            analyzed_moves=len(quality),
        ),
    )
    expected = {r["request_id"] for r in requests if r.get("legal")}
    if expected != {r["request_id"] for r in quality}:
        raise ValueError("Analysis coverage is incomplete")
    print(
        f"Exported {len(games)} game records, {len(requests)} requests and {len(quality)} analyzed moves to {destination.name}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export_snapshot(args.source, args.destination)


if __name__ == "__main__":
    main()
