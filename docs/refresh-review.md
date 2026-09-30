# Benchmark refresh review — 2026-09-29

> Historical protocol 2 review. Current behavior and pilot validation are documented
> in [architecture](architecture.md), [protocol 3](protocol-v3.md), and
> [the published pilot](../benchmarks/2026-09-30-pilot/README.md).

Reviewed the CLI, orchestrator, game schedule/rules, UCI engine adapters, provider
clients, agent/tool path, pricing and results storage, terminal reporting,
packaging, examples and tests. No paid provider calls were made. Historical data
was read only; `data/results.db` and archived runs were not modified.

## Findings and changes

| Area | Previous behavior | Protocol v2 |
| --- | --- | --- |
| Random opponent | Invalid `random_opponent` argument; unnecessarily required Stockfish | Engine-free random games work end to end |
| Schedule | Ignored `--max-games`; inconsistent random openings per color/model | Bounded even game count, identical opening pairs, seeded schedules shared across bots |
| Invalid output | Random/first-legal fallbacks; string/move type error in fallback | Strict legal UCI; invalid response forfeits; API failures abort separately |
| Draws | Inconsistent claim checks; move limit called a draw | Consistent claims; cap marked incomplete |
| History | Omitted forced openings and reconstructed turns incorrectly | Authoritative board move stack in prompt |
| Providers | Blocking SDK work in threads; 16-token caps; retired Google SDK | Async calls, current SDKs, reasoning-aware requests and configurable output cap |
| Agents | Circular import could disable agents; usage double counted; heuristic fallback | Lazy imports; one tool-assisted decision request; no move substitution |
| Costs | Character guesses; Gemini 2.5 matched wrong prices; unknown prices free | Reported token counts including thoughts; exact price matches; unknown rates fail |
| Budgets | Warnings only | Reserve before each request; concurrent reservations; explicit uncertain billing |
| Statistics | Time divided by both players' plies; aborted outcomes treated as draws | Time per model request; completed and incomplete games separated; descriptive intervals |
| Elo | Invented low-strength mapping; opponent setting advertised as model rating | Native UCI_Elo only; no derived model Elo or grandmaster labels |
| Storage | No game rows; metadata missing; matched model names by substring | Exact bot mapping, game rows, config/summary/usage/checkpoint files; separate v2 DB |
| Ranking | Mixed latest results across different runs; arbitrary Elo/cost bonuses | Latest single-run table, observed outcomes and costs |
| Lifecycle | Initialization before cleanup scope; swallowed failures appeared successful | Cleanup covers partial initialization; bot errors shown and nonzero CLI exit |
| Tests | Four baseline failures; async unittest methods weren't awaited | Async harness repaired, stale fixtures corrected, offline end-to-end regressions |
| Documentation | Old models/prices, unsupported commands and unimplemented claims | Current commands, explicit protocol and estimate assumptions |

The archived database contained 60 benchmark rows, 159 model-performance rows,
and **zero individual game rows**. Archived run directories held 533 PGN files.
Historical costs are not a reliable calibration dataset because v1 pricing,
usage accounting, failures and fallback behavior were inconsistent.

## Cost planning

See the [API protocol guide](api-protocol-v2.md) model table or run `python main.py --estimate-cost`. The six-model
10-game scenario costs $28.8336; the high-use scenario costs $169.584. GPT-6 Astra
contributes $20.16 / $118.40 respectively. The budget preset costs $3.1296 /
$18.624. These are USD list-price scenarios, not measured future consumption.

A useful first step is a two-game budget pilot with a $5 local guard. Use its
`budget.json` reported input/output tokens and `summary.json` completion and
forfeit rates to tune time/output limits before a larger run. For all six models,
a $40 local guard covers the central ten-game scenario with some margin but can
stop early if reasoning usage approaches the high scenario. No cap guarantees
that the full schedule will finish.

## Validation and remaining limits

Validation: **122 tests passed** on Python 3.13, with no warnings. Coverage includes provider request/usage mocks,
engine-free paired random games, cancellation/cleanup, CLI errors, installed
console commands, examples, and a real local Stockfish smoke run. A CI workflow
runs the offline suite on Python 3.10 and 3.13; remote CI has not been run here. The local
engine reported minimum UCI_Elo 1320 and completed a two-game capped schedule.
Paid provider availability, authentication, and live response quality remain
unverified. Prices were checked against official documentation, not account bills.

The current CLI agent is a bounded tool-assisted condition, not a general tool
calling agent. Existing direct-agent, human-engine and adaptive-engine classes
are retained as experimental APIs, not certified benchmark conditions. The cost, agent and comprehensive examples now run offline or with random bots.
The human-engine demo remains an experimental standalone tool; its strength
mappings are not supported benchmark conditions. The README, config example,
CLI and Makefile document the v2 entry points.

Other future work requiring a deliberate study design: held-out position suites,
centipawn-loss analysis, calibrated opponent pools, model snapshot pinning,
paired statistical comparisons across repeated seeds, and automatic run resume.
The current 21-opening book is small and public. Ten games is a pilot-sized sample.
A model's chess-game score with legal moves supplied does not measure general
reasoning ability or independently establish an Elo rating.
