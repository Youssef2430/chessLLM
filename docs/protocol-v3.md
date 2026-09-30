# Improved subscription benchmark · protocol 3

Protocol 3 measures chess with explicit move recovery, stronger prompting and
high reasoning effort. Protocol 2 remains the strict baseline. Keep their results
in separate experiment directories: changing the prompt, output contract, effort
and retry budget changes what is being measured.

## Run and inspect

```sh
# Uses installed clients and their existing subscription sign-ins. No API keys.
python -m pip install -e '.[charts]'
python -m chess_llm_bench.runner --output-dir runs/improved-NEW --detach

# Local results interface: overview, replay, diagnostics and methodology.
python -m chess_llm_bench.observatory --port 8770
# Open http://127.0.0.1:8770

# Resume an interrupted protocol 3 experiment with its saved settings.
python -m chess_llm_bench.runner --output-dir runs/improved-NEW --resume --detach

# Analyze new decisions and export standalone charts while the run progresses.
python -m chess_llm_bench.watch_run runs/improved-NEW \
  --output-dir runs/improved-NEW/figures --interval 120

# Local integration smoke test; no model requests or account quota.
python -m chess_llm_bench.runner --output-dir /tmp/chess-smoke-NEW \
  --offline --games 2 --max-plies 8
```

Detached runner PIDs and log paths are printed. SIGTERM/SIGINT checkpoint and stop
active work. A machine restart requires an explicit `--resume`; no startup daemon
is installed. Only one writer can own a run. Resume requires the same Stockfish
binary and experiment settings; each execution records its source hashes.

The default roster is GPT-6 Astra, GPT-6 Sol, GPT-6 Luna, Claude Opus 5.5, Claude
Sonnet 5.5, Gemini 3.1 Pro and Gemini 3.8 Flash. Haiku and historical models are
excluded. GPT-OSS-120B was absent from this machine's authenticated Antigravity
catalog; unavailable selections are recorded without substitutions. Antigravity
high effort maps to `gemini-pro-agent` for Pro and `gemini-3.8-flash-high` for Flash.
A requested effort level is not a standardized compute budget across providers.

Defaults: 2 games/model, one opening pair with reversed colors, seed 42,
200 total plies (including the opening), high effort, 180-second request timeout,
three maximum application attempts per decision, and Stockfish at its advertised
minimum native UCI_Elo with 0.3 seconds per move. The initial live improved cohort
uses a 360-second timeout. GPT models run concurrently by default, with up to three active Codex requests
(`--codex-concurrency 3`) and one request per model. Claude and Antigravity each
retain one shared slot; different providers also run concurrently. Subscription
limits still apply. Provider queue time is saved
separately from request latency. No output-token cap or temperature is asserted.

## Decision handling

1. Supply FEN, ASCII board, piece locations, full history, legal UCI/SAN choices,
   and a checklist covering forcing replies, threats, material and king safety.
2. Request a JSON move object constrained to the legal UCI choices from Codex
   and Claude. Antigravity receives the same position with a final-line UCI contract.
3. Accept a legal exact UCI, JSON `move`, final single-move code block, explicitly
   marked final move, or bare UCI final line. Never select a token arbitrarily
   from analysis, choose between alternatives, or substitute a random move.
4. If invalid, send specific validation feedback and retry within the cap.
   Transient transport errors/timeouts also consume attempts, with bounded backoff.
   Quota, authentication, model mismatch and forbidden tool use stop that model.
5. Exhausted invalid answers forfeit the game; service failures leave it incomplete.
   Every attempt, including rejected and interrupted requests, stays in the ledger.

The attempt cap applies across restarts. Resume can use remaining attempts or a
previously recorded legal answer; it does not reset exhausted budgets. If all
attempts for a service failure were used, start a separately labeled cohort to
try again. Client-internal retries may differ and are not controlled by this cap.

Model tools and external assistance remain disabled. Claude's `StructuredOutput`
serialization event is permitted only for schema output; chess-engine/file/shell
operations are not. Stockfish post-game analysis is never passed into model prompts.

## Durability and provenance

`requests.jsonl` is the request ledger. `decision_id` groups attempts by model,
game and ply; `request_id` identifies one call. Accepted decisions also appear in
`decisions.jsonl`. `plies.jsonl` records every actual move by both players.

Each move is journaled and the full position/history checkpoint is replaced
atomically in `state/`. Resume replays any journaled move newer than its checkpoint,
validates FEN/history, and reuses saved accepted responses. In-flight requests whose
responses were lost are recorded with unknown usage and elapsed time. Partial
traces are retained when the transport exposes them. Incomplete JSONL tails can be
recovered; complete malformed rows are errors. Raw attempts are never overwritten.

`games.jsonl` may contain an incomplete snapshot followed by the completed record
for the same game. Readers take the latest `(bot, game_id)` record. Stable ply keys
are `(bot, game_id, ply)`. A fresh heartbeat plus a live PID distinguishes running
experiments from interrupted ones. `run_status.json` exposes model progress and
stops; `manifest.json` records settings, schedule, engine hash and execution hashes.

## Read the charts correctly

The Observatory supports model/run filters, W/D/L, forfeits, recovery/retry counts,
request latency distributions, centipawn-loss distributions, token coverage,
interactive board replay, per-move evaluation, raw response/prompt inspection,
CSV model-summary export and completed-game PGN downloads. It serves local Plotly
assets, binds to loopback, and has no endpoints that start model calls.

Use the offline watcher for analysis, detailed CSV exports, standalone Plotly HTML
and annotated SVG positions. It exits when the run stops, after a final analysis
pass. Restart it when resuming a run. The watcher makes no model requests.

- Score is `(wins + 0.5 × draws) / completed games`. It includes invalid-response
  forfeits; service failures and max-ply cutoffs are excluded, not counted as draws.
- CP loss compares the selected move with Stockfish's best move at a fixed node
  budget. Mate scores remain separate, and sample coverage is shown.
- Token counts and client timings are included only when reported. Missing values
  are unknown, not zero. API-equivalent costs are not subscription charges.
- Subscription runs consume account allowance. The runner disables API-key fallback;
  it cannot infer remaining plan quota or verify an additional charge from token logs.
- Ten games per model provide exploratory evidence, not a reliable rating/ranking.
  A higher score in v3 can reflect recovered formatting as well as better chess.
  Inspect move quality and forfeits separately before attributing gains to reasoning.

Concurrency changes are recorded as run amendments. Request rows record the active
provider concurrency limit, so timing before and after the change can be separated.
Older manifests without that setting retain their original single-slot behavior
unless explicitly amended.
