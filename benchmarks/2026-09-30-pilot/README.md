# Subscription pilot · September 30, 2026

Source run: `improved-v3-20260929-02`. Started September 29 at 23:57 UTC;
runner stopped September 30 at 04:39 UTC, with final analysis completed at 04:40 UTC.
All data here is from the improved cohort. The strict baseline and the invalidated
Claude parser validation run are excluded.

## Observed results

W/D/L counts completed chess outcomes. Unfinished games are excluded from the
score denominator. Mean CP loss excludes mate lines; parentheses show the number
of analyzed decisions with finite CP scores. Median request time includes client
startup and recorded failed/interrupted attempts, excludes provider queue time,
and mixes concurrency regimes; it is not a pure model-speed comparison.

| Requested model | W/D/L | Unfinished | Requests | Mean CP loss (n) | Median request seconds |
|---|---:|---:|---:|---:|---:|
| `gpt-6-astra` | 2/0/0 | 0 | 51 | 23.9 (43) | 71.3 |
| `gpt-6-sol` | 0/0/1 | 1 | 126 | 82.5 (110) | 101.1 |
| `gpt-6-luna` | 0/0/2 | 0 | 62 | 104.7 (53) | 48.9 |
| `claude-opus-5-5` | 1/0/1 | 0 | 58 | 43.8 (36) | 12.4 |
| `claude-sonnet-5-5` | 1/0/1 | 0 | 72 | 53.1 (62) | 9.9 |
| `gemini-3.1-pro` | 0/0/0 | 1 | 7 | 15.3 (6) | 40.2 |
| `gemini-3.8-flash` | 1/0/1 | 0 | 99 | 45.5 (79) | 13.5 |

![Pilot outcomes and move quality](overview.svg)

- **14 games scheduled; 13 started; 11 finished by checkmate.** Sol's second
  game hit 200 plies. Gemini Pro's first game stopped for prohibited tool use;
  its second game was not started. Neither unfinished game is a draw.
- **475 requests, 466 accepted model moves, all 466 analyzed.** There were three
  invalid Gemini Flash responses corrected on retry, two GPT Sol timeouts,
  one Gemini Pro tool-use violation and three operator cancellations.
- **No invalid-response forfeits.** Cancellations happened while shortening the
  schedule and are classified as interruptions, not provider failures.
- Astra won both colors. Opus, Sonnet and Flash each won as White and lost as
  Black. This color/opening pattern is another reason not to infer a stable ranking.

## Conditions and amendments

Stockfish 17.1, native `UCI_Elo=1320`, 0.3 seconds per opponent move; one King's
Indian Defense opening pair, seed 42; 200-ply cap including book moves; high
reasoning effort, 360-second request timeout and at most three application attempts.
Codex and Claude used legal-move JSON schemas; Antigravity used a final-line UCI
contract. Models received no engine analysis. Offline analysis used full-strength
Stockfish with 50,000 nodes per best/chosen search, one thread and cleared hash.

The run initially scheduled ten games/model, then was reduced to two while the
first pair was in progress. Existing games were retained. GPT calls initially
shared one slot; on September 30 at approximately 01:08 UTC they switched to
three concurrent requests. The active response was saved before that restart.
The manifest retains amendments and execution source hashes. Request-level
`provider_concurrency` marks the later regime; earlier missing values mean one slot.

Claude and Gemini identify the selected model through client metadata. Codex's
JSON did not independently report the serving model, so GPT identity is recorded
as `requested_only`. The high-effort Gemini Pro native ID was `gemini-pro-agent`;
Flash used `gemini-3.8-flash-high`. Same-named effort levels are not equivalent
compute budgets across products. Antigravity did not expose token usage.

Subscription allowance was used with API-key fallback disabled. Actual incremental
charges are unreported. Reported token counts are not an invoice.

## What can be extrapolated

Astra's observed score was 100%; Opus, Sonnet and Flash each scored 50%; Luna
scored 0%. These are two-game observations from a single opening pair. Sol has
only one finished outcome and Gemini Pro has none, so neither gets a two-game
projection.

[projections.json](projections.json) scales observed proportions to ten games
(Astra 10 wins, the 1–1 models 5 wins/5 losses, Luna 10 losses). This is explicitly
**arithmetic illustration, not a reliable prediction or measured result**.
Move-quality averages cover more decisions but those decisions are correlated
within games and do not turn the pilot into hundreds of independent trials.

## Files and reproduction

| File | Contents |
|---|---|
| [manifest.json](manifest.json) | Settings, paired schedule, engine hash, client versions, amendments and source hashes |
| [summary.json](summary.json) | Observed model totals, analysis coverage, latency and reported tokens |
| [games.jsonl](games.jsonl) | Latest record for each of 13 started games |
| [requests.jsonl](requests.jsonl) | All 475 attempt summaries with board/history, usage and error categories |
| [plies.jsonl](plies.jsonl) | Both players' moves with before/after positions |
| [analysis.jsonl](analysis.jsonl) | All 466 offline evaluations and principal variations |
| Model directories | Two PGNs per available model; one incomplete PGN for Gemini Pro |

Raw client transcripts, prompts/responses, authentication details, local paths
and process IDs are omitted. Prompt hashes, chess state and usage remain. Original
local source-ledger hashes are in the manifest. This snapshot supports replay and
metric verification without exposing client logs. Original engine evaluations are
retained; reanalysis with another Stockfish version can differ.

From the repository root:

```sh
python -m pip install -e '.[charts,reports]'
python -m chess_llm_bench.observatory --runs-dir benchmarks --port 8770
python -m chess_llm_bench.plot_snapshot benchmarks/2026-09-30-pilot
python -m chess_llm_bench.figures benchmarks/2026-09-30-pilot \
  --output-dir output/pilot-figures
```

The Observatory also supports PGN downloads from this portable snapshot. To export
another completed local run, use `python -m chess_llm_bench.export_snapshot SOURCE
DESTINATION`, then inspect the resulting archive before publishing it.
