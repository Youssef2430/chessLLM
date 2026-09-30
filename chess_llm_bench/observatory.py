"""Read-only local results observatory. No endpoints can start model requests."""

import argparse
import json
import math
import mimetypes
import os
import statistics
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from .core.journal import latest, read_rows
from .llm.subscription import normalized_tokens, request_interrupted


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def run_health(directory, manifest):
    if manifest.get("invalidated_reason"):
        return dict(status="invalid", models={}, reason=manifest["invalidated_reason"])
    path = directory / "run_status.json"
    if path.exists():
        status = json.loads(path.read_text())
        if status["status"] == "running":
            try:
                os.kill(status["pid"], 0)
            except (ProcessLookupError, PermissionError):
                status["status"] = "interrupted"
            age = (
                datetime.now(timezone.utc)
                - datetime.fromisoformat(status["updated_at"])
            ).total_seconds()
            if age > 90:
                status["status"] = "interrupted"
        return status
    return {
        "status": "finished" if manifest.get("finished_at") else "interrupted",
        "models": {},
    }


def read_run(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    health = run_health(directory, manifest)
    requests = read_rows(directory / "requests.jsonl")
    for r in requests:
        r.update(normalized_tokens(r["provider"], r.get("usage", {})))
        r["interrupted"] = request_interrupted(r)
        r.setdefault("attempt", 1)
        r.setdefault("parse_method", "strict" if r.get("legal") else "invalid")
    raw_games = read_rows(directory / "games.jsonl")
    counter = Counter()
    for g in raw_games:
        counter[g["bot"]] += 1
        g.setdefault("game_id", str(counter[g["bot"]]))
    games = latest(raw_games, ("bot", "game_id"))
    plies = latest(read_rows(directory / "plies.jsonl"), ("bot", "game_id", "ply"))
    analysis = latest(read_rows(directory / "analysis.jsonl"), ("request_id",))
    statuses = {r["model"]: r for r in read_rows(directory / "status.jsonl")}
    statuses.update(health.get("models", {}))
    settings = manifest.get("settings", manifest.get("config", {}))
    models = (
        manifest.get("requested_models")
        or settings.get("models")
        or sorted({r["bot"] for r in requests})
    )
    # Include active or interrupted games for replay without inventing outcomes.
    known = {(g["bot"], g["game_id"]) for g in games}
    for r in requests:
        key = (r["bot"], r["game_id"])
        if key not in known:
            fixture = next(
                (
                    f
                    for f in manifest.get("schedule", [])
                    if f["game_id"] == r["game_id"]
                ),
                {},
            )
            games.append(
                dict(
                    bot=key[0],
                    game_id=key[1],
                    result="*",
                    termination=(
                        "in_progress"
                        if health["status"] == "running"
                        else "interrupted"
                    ),
                    color_llm_white=r["side_to_move"] == "white",
                    opening=fixture.get("opening", "Unfinished game"),
                    ply_count=max(
                        (p["ply"] for p in plies if (p["bot"], p["game_id"]) == key),
                        default=r["ply"] - 1,
                    ),
                    elo=settings.get(
                        "opponent_elo", settings.get("fixed_opponent_elo")
                    ),
                )
            )
            known.add(key)
    summaries = []
    for model in models:
        rr = [r for r in requests if r["bot"] == model]
        gg = [g for g in games if g["bot"] == model]
        qq = [q for q in analysis if q["bot"] == model]
        completed = [g for g in gg if g["result"] in {"1-0", "0-1", "1/2-1/2"}]
        wins = sum(
            (g["result"] == "1-0" and g["color_llm_white"])
            or (g["result"] == "0-1" and not g["color_llm_white"])
            for g in completed
        )
        draws = sum(g["result"] == "1/2-1/2" for g in completed)
        cp = [q["centipawn_loss"] for q in qq if q["centipawn_loss"] is not None]
        timing = [r["wall_seconds"] for r in rr if r.get("wall_seconds") is not None]
        token_rows = [r for r in rr if r["tokens_reported"]]
        state = statuses.get(model, {}).get("status", "queued")
        if health["status"] != "running" and state in {"running", "queued"}:
            state = "interrupted"
        summaries.append(
            dict(
                model=model,
                provider=next((r["provider"] for r in rr), ""),
                status=state,
                status_detail=statuses.get(model, {}).get("error")
                or statuses.get(model, {}).get("reason"),
                games=len(completed),
                wins=wins,
                draws=draws,
                losses=len(completed) - wins - draws,
                incomplete=sum(g["result"] == "*" for g in gg),
                score=(wins + 0.5 * draws) / len(completed) if completed else None,
                requests=len(rr),
                decisions=len(
                    {r.get("decision_id") or (r["game_id"], r["ply"]) for r in rr}
                ),
                legal_moves=sum(r.get("legal") is True for r in rr),
                invalid_attempts=sum(r.get("legal") is False for r in rr),
                recoveries=sum(bool(r.get("format_recovered")) for r in rr),
                retries=sum(r["attempt"] > 1 for r in rr),
                service_errors=sum(
                    bool(r.get("error")) and not r["interrupted"] for r in rr
                ),
                interruptions=sum(r["interrupted"] for r in rr),
                forfeits=sum(g["termination"] == "invalid_move" for g in gg),
                median_seconds=statistics.median(timing) if timing else None,
                p95_seconds=percentile(timing, 0.95),
                acpl=statistics.mean(cp) if cp else None,
                cp_samples=len(cp),
                analyzed=len(qq),
                best_move_rate=(
                    sum(q["best_move_match"] for q in qq) / len(qq) if qq else None
                ),
                input_tokens=(
                    sum(r["input_tokens_total"] for r in token_rows)
                    if token_rows
                    else None
                ),
                output_tokens=(
                    sum(r["output_tokens_total"] for r in token_rows)
                    if token_rows
                    else None
                ),
                token_coverage=len(token_rows) / len(rr) if rr else None,
            )
        )
    return dict(
        id=directory.name,
        manifest=manifest,
        health=health,
        models=summaries,
        games=games,
        requests=requests,
        plies=plies,
        analysis=analysis,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


def resolve_run(root, name):
    if not name or Path(name).name != name:
        raise ValueError("Invalid run ID")
    path = (root / name).resolve()
    if path.parent != root.resolve() or not (path / "manifest.json").is_file():
        raise ValueError("Unknown run")
    return path


def make_handler(root):
    assets = Path(__file__).parent / "web"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            params = parse_qs(url.query)
            try:
                if url.path == "/api/runs":
                    runs = []
                    for path in sorted(root.iterdir(), reverse=True):
                        if not (path / "manifest.json").is_file():
                            continue
                        try:
                            m = json.loads((path / "manifest.json").read_text())
                            runs.append(
                                dict(
                                    id=path.name,
                                    protocol=m.get("protocol", "legacy"),
                                    started_at=m.get("started_at"),
                                    status=run_health(path, m)["status"],
                                    effort=m.get("settings", m.get("config", {})).get(
                                        "effort",
                                        m.get("config", {}).get("reasoning_effort"),
                                    ),
                                )
                            )
                        except (ValueError, OSError):
                            continue
                    self.send_json(runs)
                    return
                if url.path == "/api/run":
                    self.send_json(
                        read_run(resolve_run(root, params.get("id", [""])[0]))
                    )
                    return
                if url.path == "/api/pgn":
                    directory = resolve_run(root, params.get("id", [""])[0])
                    name = params.get("model", [""])[0]
                    gid = params.get("game", [""])[0]
                    run = read_run(directory)
                    g = next(
                        (
                            g
                            for g in run["games"]
                            if g["bot"] == name and g["game_id"] == gid
                        ),
                        None,
                    )
                    path = Path(g["path"]) if g and g.get("path") else None
                    if path is not None:
                        if not path.is_absolute() and (directory / path).is_file():
                            path = directory / path
                        path = path.resolve()
                    if (
                        not path
                        or not path.is_relative_to(directory.resolve())
                        or not path.is_file()
                    ):
                        raise ValueError("PGN unavailable for unfinished game")
                    self.send_bytes(
                        path.read_bytes(),
                        "application/x-chess-pgn",
                        attachment="game.pgn",
                    )
                    return
                if url.path == "/plotly.min.js":
                    from plotly.offline import get_plotlyjs

                    self.send_bytes(get_plotlyjs().encode(), "text/javascript")
                    return
                files = {
                    "/": "index.html",
                    "/app.js": "app.js",
                    "/style.css": "style.css",
                }
                if url.path not in files:
                    raise ValueError("Not found")
                path = assets / files[url.path]
                self.send_bytes(
                    path.read_bytes(),
                    mimetypes.guess_type(path.name)[0] or "text/plain",
                )
            except (ValueError, OSError, KeyError) as exc:
                self.send_json({"error": str(exc)}, 404)

        def send_json(self, data, status=200):
            self.send_bytes(
                json.dumps(data, allow_nan=False, default=str).encode(),
                "application/json",
                status,
            )

        def send_bytes(self, data, mime, status=200, attachment=None):
            self.send_response(status)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
            )
            if attachment:
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{attachment}"'
                )
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    root = Path(args.runs_dir).resolve()
    root.mkdir(exist_ok=True, parents=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(root))
    print(f"Chess Observatory: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
