"""Wait for final benchmark artifacts, then issue a local macOS notification."""

import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path

from .core.journal import atomic_json, latest, read_rows
from .observatory import run_health


def completion_report(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    health = run_health(directory, manifest)
    if health["status"] == "running":
        return None
    figures = directory / "figures"
    progress = json.loads((figures / "progress.json").read_text())
    if not progress.get("finished"):
        return None
    snapshot = json.loads((figures / "figure-data.json").read_text())
    requests = read_rows(directory / "requests.jsonl")
    if {r["request_id"] for r in requests} != {
        r["request_id"] for r in snapshot["requests"]
    }:
        return None
    accepted = {r["request_id"] for r in requests if r.get("legal")}
    if not accepted.issubset({r["request_id"] for r in snapshot["analysis"]}):
        return None
    games = latest(read_rows(directory / "games.jsonl"), ("bot", "game_id"))
    signature = lambda rows: {
        (g["bot"], g["game_id"], g["result"], g["termination"]) for g in rows
    }
    if signature(games) != signature(snapshot["games"]):
        return None
    if not all(
        (figures / f).is_file()
        for f in ("benchmark-figures.html", "game-evaluations.html")
    ):
        return None
    settings = manifest["settings"]
    stopped_early = [
        name
        for name, status in health.get("models", {}).items()
        if status["status"] != "finished"
    ]
    projections = []
    for model in settings["models"]:
        completed = [g for g in games if g["bot"] == model and g["result"] != "*"]
        wins = sum(
            (g["result"] == "1-0" and g["color_llm_white"])
            or (g["result"] == "0-1" and not g["color_llm_white"])
            for g in completed
        )
        draws = sum(g["result"] == "1/2-1/2" for g in completed)
        observed = dict(
            games=len(completed),
            wins=wins,
            draws=draws,
            losses=len(completed) - wins - draws,
        )
        target = manifest.get("analysis_plan", {}).get("project_to_games", 10)
        projections.append(
            dict(
                model=model,
                observed=observed,
                projected_games=target,
                projected_outcomes=(
                    {
                        k: target * observed[k] / len(completed)
                        for k in ("wins", "draws", "losses")
                    }
                    if len(completed) >= 2
                    else None
                ),
            )
        )
    return dict(
        status=health["status"],
        scheduled_games=settings["games"] * len(settings["models"]),
        completed_games=sum(g["result"] != "*" for g in games),
        incomplete_games=sum(g["result"] == "*" for g in games),
        requests=len(requests),
        analyzed_decisions=len(accepted),
        stopped_early=stopped_early,
        models=health.get("models", {}),
        figures=str(figures.resolve()),
        dashboard="http://127.0.0.1:8770",
        pilot_projections=projections,
        projection_caution="Projections are scaled observed rates, not played games or reliable predictions. One opening pair is too small to establish a model ranking; future outcomes may differ substantially.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    with (directory / ".completion-notifier.lock").open("a+") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination = directory / "completion-notification.json"
        if destination.exists():
            return
        print("Monitoring benchmark and final chart coverage.", flush=True)
        while True:
            try:
                report = completion_report(directory)
            except (OSError, ValueError, KeyError) as exc:
                print("Waiting for complete artifacts: " + str(exc), flush=True)
                report = None
            if report is not None:
                message = f"{report['completed_games']}/{report['scheduled_games']} games completed. Final charts are ready."
                if report["stopped_early"]:
                    message += f" {len(report['stopped_early'])} model(s) stopped early; see the dashboard."
                try:
                    subprocess.run(
                        [
                            "osascript",
                            "-e",
                            "on run argv\ndisplay notification (item 2 of argv) with title (item 1 of argv)\nend run",
                            "Chess benchmark finished running",
                            message,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    report["notification_submitted"] = True
                except (OSError, subprocess.SubprocessError) as exc:
                    report["notification_submitted"] = False
                    report["notification_error"] = str(exc)
                atomic_json(destination, report)
                print(json.dumps(report), flush=True)
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
