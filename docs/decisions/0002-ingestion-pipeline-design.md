# ADR 0002: Ingestion pipeline design

## Context

ADR 0001 settled the problem definition: study window ~January 2024 → present, hourly
resolution, national scope, forecast cutoff determined empirically. This ADR covers how
the ingestion pipeline that feeds that definition is built: storage, tooling, data flow,
and the merge logic across the three sources (RTE, weather, calendar).

Constraints already fixed by the project's engineering standards, not re-decided
here: Python 3.12, `uv`, `src/` layout, `ruff`, `pytest`, conventional commits. No
orchestrator in v1 — explicitly deferred to v2 in `PROJECT_NOTES.md`.

## Decision

### Storage: local Parquet files, raw/processed layers

- `data/raw/<source>/...`: data as received from each API, untouched.
- `data/processed/dataset.parquet`: one row per hour, joined and validated, ready for
  modeling.
- Both directories are gitignored; raw data is never committed and must always be
  rebuildable from source APIs (existing project rule).
- Rejected: DuckDB (unnecessary extra tool for this data volume, ~20-30k hourly rows),
  Postgres (duplicates the data-engineering story already planned for the
  `french-business-registry` project), a cloud warehouse / BigQuery (no volume, latency,
  or sharing need justifies it at this stage; revisit for v2, where an always-on daily
  pipeline and a served dashboard make a warehouse's always-available property actually
  pay for itself).

### Data manipulation: pandas

Chosen over polars: the ML ecosystem this project uses downstream (LightGBM,
scikit-learn) expects pandas/numpy natively, the data volume is far too small for
polars's performance advantage to matter, and it avoids an extra library to learn on a
project whose stated weak area to strengthen is ML rigor, not tooling breadth.

### HTTP: `requests`, synchronous

This is a batch job (backfill once, refresh once a day) with no concurrency
requirement — no justification for an async client.

### Validation: `pandera`, applied at the output of `build-dataset`

Schema checks (types) plus plausibility ranges (e.g., temperature within -25°C to 45°C,
national consumption within 20,000-100,000 MW) catch a silent upstream API change or a
parsing bug early, rather than three weeks later inside a misbehaving model.

### Weather: multiple cities, population-weighted national temperature

A single point (e.g., Paris) is not representative of national demand — a cold snap
localized in one region would not show up in the signal while still affecting national
consumption. ~6-8 major cities spread across the territory (e.g., Paris, Lyon,
Marseille, Lille, Nantes, Strasbourg, Toulouse, Bordeaux), combined into one national
temperature per hour via a population-weighted average. This mirrors how grid operators
actually reason about weather-driven demand. Rejected: a systematic geographic grid —
materially more API calls and complexity for a marginal accuracy gain over a
well-chosen set of major population centers.

### RTE: two datasets chained, because of RTE's own data lifecycle

RTE data moves through three stages — temps réel (near-instant estimate) →
consolidated (~1 month lag, meter-verified) → definitive (up to ~18 months lag, fully
confirmed). ODRÉ exposes this as two separate datasets:

- `eco2mix-national-cons-def`: full history, but always trailing today by a few months
  (consolidation lag).
- `eco2mix-national-tr`: only a rolling ~3-month window, but including the most recent
  data.

Verified: their boundary records match exactly, no gap between them. Consequence for
the pipeline: **backfill** must query both — `cons-def` for the older part of the study
window, `tr` for whatever is more recent than `cons-def`'s current end. **Refresh**
(daily) only needs to query `tr`, since that is the only one of the two that changes
day to day.

Known limitation, not resolved now: a value ingested today from `tr` is a real-time
estimate that may be slightly revised once it later appears in `cons-def`. Re-validating
old rows against `cons-def` once they migrate is a possible future improvement, not a
v1 blocker — documented here so it isn't forgotten.

**Update (2026-09-25, first real backfill)**: the ODRÉ Explore v2.1 API hard-caps
`offset + limit <= 10000` per query (`InvalidRESTParameterError` beyond that) — a real
constraint invisible to unit tests, which only ever mocked 1-2 pages. `cons-def` alone
has 80k+ records over the study window, so naive offset-based pagination was guaranteed
to fail on a real multi-year backfill. Fixed by switching `fetch_records` to
cursor-based pagination on `date_heure` (the same column already used as the dedup key
elsewhere), with an inclusive `>=` comparator and de-duplication on `date_heure` as rows
accumulate, to correctly handle the case where two records could share an exact
timestamp at a page boundary (not observed in the real data so far, but not assumable
away either) without either looping forever or silently dropping rows.

**Update (2026-09-25)**: also confirmed empirically that `consommation` is published at
30-minute resolution throughout the 2024-2026 study window (null at :15/:45), while
`prevision_j1`/`prevision_j` are genuine 15-minute with no nulls. This resolves the
resolution-consistency question ADR 0001 left open — see that ADR's update below.
Handled transparently by `aggregate_rte_hourly`'s mean-based hourly aggregation either
way, no code change needed.

### Hourly aggregation

RTE publishes at 15-min or 30-min resolution depending on the period; weather (Previous
Runs API) is natively hourly; calendar data is daily. To reach the target "one row per
hour":

- RTE: average the sub-hourly readings within each hour (a mean of a power/flow
  quantity, not a last-value pick, so the whole hour's behavior is represented, not
  just its first instant). Same treatment for `prevision_j1` and `prevision_j`.
- Weather: already hourly per city; combine the cities into one national value per hour
  via the population weighting described above.
- Calendar: one value per day, broadcast to all 24 hours of that day at merge time.

### Merge

RTE's hourly table (it carries the target, `consommation`) is the backbone. The
national weighted temperature is left-joined on `(date, heure)`. Calendar flags
(`est_ferie`, school-holiday zone) are left-joined on `date` alone, each day's value
repeated across its 24 hourly rows.

## Structure

```
src/
  ingestion/
    rte.py        # fetch + parse éCO2mix (cons-def + tr)
    weather.py      # fetch + parse Open-Meteo Previous Runs (8 cities)
    calendar.py       # fetch + parse public/school holidays
    storage.py          # Parquet read/write, partitioning helpers
  processing/
    build_dataset.py    # hourly aggregation, national temperature, join, output
    validation.py          # pandera schemas
  cli.py                    # backfill / refresh / build-dataset commands
```

Each connector (`rte.py`, `weather.py`, `calendar.py`) exposes a `backfill()` and a
`refresh()` function, both writing to `data/raw/`, both required to be idempotent
(re-running must not duplicate rows — deduplicate on the `(date, heure)` key after
write).

## Alternatives considered

Covered inline per decision above (DuckDB/Postgres/warehouse for storage, polars for
data manipulation, async HTTP, single-point weather, a geographic grid for weather).

## Consequences

- Three independent, individually testable connectors, each with a clear
  backfill/refresh split matching how its underlying source actually behaves (RTE's
  consolidation lag, the daily-growing clean weather window).
- No infrastructure to operate for v1 — just Python scripts run manually or via a basic
  cron, consistent with the orchestrator decision already deferred to v2.
- Testing focuses on parsing, aggregation, and merge logic, using recorded real API
  responses as fixtures — not on the act of making an HTTP call itself.
- The `tr` → `cons-def` revision nuance is a known, documented limitation, not a
  blocker; worth a small note in the eventual data-quality write-up.
