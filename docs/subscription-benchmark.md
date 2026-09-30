# Subscription benchmark and telemetry

> This page documents the preserved **protocol 2 strict baseline**. Use the
> [protocol 3 guide](protocol-v3.md) for the improved runner, retries, resume and UI.

This is a **subscription-product comparison**, using the installed client harnesses.
It is a separate cohort from direct API protocol-v2 runs. Client system context,
reasoning implementations and output limits differ. No Elo rating for an LLM is
inferred from the opponent's UCI_Elo setting.

## Current roster

| Model | Transport | Requested effort / native ID |
|---|---|---|
| GPT-6 Astra | Codex / ChatGPT login | low / `gpt-6-astra` |
| GPT-6 Sol | Codex / ChatGPT login | low / `gpt-6-sol` |
| GPT-6 Luna | Codex / ChatGPT login | low / `gpt-6-luna` |
| Claude Opus 5.5 | Claude Code / claude.ai login | low / `claude-opus-5-5` |
| Claude Sonnet 5.5 | Claude Code / claude.ai login | low / `claude-sonnet-5-5` |
| Gemini 3.1 Pro | Antigravity ACP / personal Google login | `gemini-3.1-pro-low` |
| Gemini 3.8 Flash | Antigravity ACP / personal Google login | `gemini-3.8-flash-low` |
| GPT-OSS-120B | Antigravity | Not in the authenticated ACP catalog on 2026-09-29 |

Haiku and historical models are excluded. Models are never silently substituted.
The ACP server's full model catalog is saved in the manifest. Codex exec currently
does not independently echo its model in JSON output; those rows accurately say
`model_verification=requested_only`. Claude's reported model and Antigravity's
confirmed selection are preserved, including the native effort suffix.

## Running and inspecting

```sh
python -m chess_llm_bench.subscription_run \
  --output-dir runs/subscriptions-NEW-RUN

# Offline analysis; repeat to analyze newly recorded decisions only.
python -m chess_llm_bench.analyze_run runs/subscriptions-NEW-RUN

# Standalone interactive figures with all JS embedded; no remote chart assets.
pip install -e '.[charts]'
python -m chess_llm_bench.figures runs/subscriptions-NEW-RUN \
  --output-dir runs/subscriptions-NEW-RUN/figures

# Refresh analysis/figures every two minutes until the run finishes.
python -m chess_llm_bench.watch_run runs/subscriptions-NEW-RUN \
  --output-dir runs/subscriptions-NEW-RUN/figures
```

The run refuses to reuse an existing output directory. It defaults to 10 games
per available model, 5 paired openings with reversed colors, seed 42, 200 plies,
120 seconds per CLI turn, and Stockfish 17.1 at native UCI_Elo 1320 with 0.3 seconds
per move. One active request per subscription provider limits contention; different
providers run concurrently. Queue time is separate from request wall time. Client
startup is included in request latency. Antigravity keeps its ACP server alive but
starts a fresh session for every decision; Codex and Claude start a fresh process.

The raw API `.env` file is never loaded. Child environments exclude API keys,
cloud-provider switches and custom API endpoints. Codex is forced to ChatGPT auth;
Claude authentication is checked before play. Antigravity reuses the existing T3
personal OAuth profile, without reading or copying token contents. Built-in tools,
plugins and context discovery are disabled where exposed; ACP client operations
and permission requests are denied. Any observed tool invocation invalidates the
unaided comparison and stops that model as a provider error.

There are no application-level retries. CLI-internal retry behavior may differ.
Timeouts terminate the CLI process group. Service errors stop the affected model;
they are aborted games, not chess losses. Non-UCI output or an illegal move is a
forfeit. A max-ply cutoff is incomplete, never a draw. Earlier completed games and
individual decision traces survive later failures. An interrupted game is not
automatically resumed or silently regenerated.

## Saved data

| Artifact | Grain and contents |
|---|---|
| `manifest.json` | Requested roster, settings, CLI versions, auth availability, platform and timestamps |
| `status.jsonl` | Model start/finish/failure/unavailability, engine identity and native range |
| `requests.jsonl` | One model decision: timestamp, FEN, full UCI history, legal moves, prompt and SHA256, response, timing, usage, legality, error, and trace link |
| `traces/*.json` | Original CLI stdout/stderr or ACP notifications, including exposed client timings and usage; no credential files or environment dumps |
| `plies.jsonl` | Every played model/opponent move, before/after FEN, SAN/UCI, capture/check/castling/promotion flags |
| `games.jsonl` | Result, termination, color, opening, seed, duration, plies, request counts and PGN path |
| `<model>/*.pgn` | Replayable games with protocol, model, effort and opponent metadata |
| `requests.csv`, `plies.csv`, `games.csv` | Flat chart exports; request export enriches metrics from retained raw traces |
| `analysis.jsonl`, `analysis.csv` | Local Stockfish analysis per legal model decision |
| `summary.json` | Observed per-model totals; missing tokens and charges remain null |
| `benchmark-figures.html` | Standalone interactive overview; SVG export in the plot toolbar |
| `game-evaluations.html` | Select a model/game to inspect its evaluation timeline |
| `positions/*.svg` | Board diagrams: blue = engine best move, gold = played move |
| `figure-data.json` | Exact source rows and paths used to build a figure snapshot |

`bot + game_id + ply` joins decisions, played moves and analysis. `request_id` joins
raw traces to analysis. `game_id` alone is **not** globally unique. Opening moves
are recorded in the initial decision's history and PGN; they are not model decisions.

## Metric definitions

- **Request latency:** elapsed CLI/ACP request time, including local startup and
  server latency, excluding the application's provider queue. It is not pure inference time.
- **TTFT:** client-reported time to first token, only when exposed; retained in
  `client_timing`. It is different from total latency.
- **Input tokens:** Codex's reported total includes cached input. Claude's total
  adds uncached input, cache creation and cache reads. Do not add caches twice.
- **Reasoning tokens:** reported detail only. A missing detail is not evidence of
  zero reasoning. Antigravity ACP currently reports neither token counts nor TTFT.
- **Legality:** true = a legal UCI response; false = an invalid response;
  null = an infrastructure failure. Formatting failures remain separate from
  illegal chess moves in the raw response, although both forfeit the game.
- **Centipawn loss:** max(0, unrestricted best-root score minus played-root score),
  both from the mover's perspective. Search uses one Stockfish thread, cleared
  hash and 50,000 nodes independently for each root search. Raw differences are
  retained because finite search can produce negative differences. Mate lines have
  null centipawn loss and explicit signed mate distance, never an artificial large CP.
- **Expected score loss:** nonnegative difference between Stockfish WDL
  expectations for best and played root. A heuristic engine estimate, not a
  calibrated probability that this LLM wins against the chosen opponent.
- **Best move agreement:** exact UCI match with the unrestricted engine PV's first
  move at the declared search budget. Tied alternatives need not match.
- **Missed forced mate:** the best search finds a positive mate, but the played-root
  search does not. All mate claims are limited by the analysis search budget.
- **Phase:** opening through ply 20; thereafter endgame at total non-pawn material
  <=26 (N/B=3, R=5, Q=9), otherwise middlegame. This is a declared heuristic.
- **CP buckets:** under 50; 50–99; 100–299; 300+; mate-line separately. These are
  transparent thresholds, not claims of universally accepted mistake categories.
- **Billing:** `incremental_charge_usd=null` means the client does not expose an
  actual charge. Claude's `total_cost_usd` is retained as `api_equivalent_usd`, not
  counted as money spent through the subscription. Existing subscription fees,
  remaining quota and any account-level extra-usage charges are not inferred.

Small samples, unequal game completion and paired-opening dependence limit model
rankings. Show per-model denominators, colors, openings, failure types and incomplete
games. Do not treat thousands of moves from a few games as independent games.

## Client references

- [Codex configuration](https://learn.chatgpt.com/docs/config-file/config-reference)
- [Claude programmatic mode](https://code.claude.com/docs/en/headless)
- [Antigravity model documentation](https://antigravity.google/docs/models)
- [ACP session protocol](https://agentclientprotocol.com/protocol/session-setup)

Installed CLI help and the live authenticated catalog take precedence over a
generic advertised model list. Claude `--bare` is intentionally avoided because
it bypasses subscription login and requires API authentication.
