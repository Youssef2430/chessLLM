"""Refresh offline analysis and figures until the supplied benchmark runs finish."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .analyze_run import analyze
from .figures import build
from .subscription_run import read_jsonl
from .observatory import run_health


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--interval", type=float, default=120)
    args = parser.parse_args()
    if args.interval < 10:
        parser.error("--interval must be at least 10 seconds")
    if any(not (Path(d) / "manifest.json").is_file() for d in args.directories):
        parser.error("Start each benchmark before starting its analysis watcher")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            for directory in args.directories:
                analyze(directory)
            build(args.directories, output)
            snapshots = []
            finished = True
            for directory in args.directories:
                path = Path(directory)
                manifest = json.loads((path / "manifest.json").read_text())
                health = run_health(path, manifest)
                finished = finished and health["status"] != "running"
                snapshots.append(
                    dict(
                        directory=str(path.resolve()),
                        finished_at=manifest.get("finished_at"),
                        status=health["status"],
                        requests=len(read_jsonl(path / "requests.jsonl")),
                        games=len(read_jsonl(path / "games.jsonl")),
                    )
                )
            (output / "progress.json").write_text(
                json.dumps(
                    dict(
                        updated_at=datetime.now(timezone.utc).isoformat(),
                        runs=snapshots,
                        finished=finished,
                    ),
                    indent=2,
                )
            )
            if finished:
                # A model may have finished while the first analysis pass was
                # working on an earlier snapshot. Drain the final decisions.
                for directory in args.directories:
                    analyze(directory)
                build(args.directories, output)
                print(
                    "All recorded runs stopped; final analysis and figures exported.",
                    flush=True,
                )
                return
        except Exception as exc:
            # No model calls happen here. Keep existing successful artifacts and
            # make refresh failures visible, rather than silently losing the loop.
            (output / "refresh-error.json").write_text(
                json.dumps(
                    dict(
                        timestamp=datetime.now(timezone.utc).isoformat(), error=str(exc)
                    ),
                    indent=2,
                )
            )
            raise
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
