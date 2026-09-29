# ADR 0010: Read-only FastAPI serving layer

## Context

`PROJECT_NOTES.md`'s v2 scope names "serving via FastAPI" as the next slice after
Airflow scheduling (ADR 0008) and MLflow tracking (ADR 0009). Today, the pipeline's
results (`forecast_latest.parquet`, `quantile_backtest_results.parquet`) are only
reachable by reading Parquet files directly or through a notebook.

## Decision

### Read-only, not a prediction-triggering API

The API only serves results Airflow's daily pipeline already computed
(`forecast_latest.parquet`, `quantile_backtest_results.parquet` joined with
`dataset.parquet`) — it never calls `predict_next_day()` or any other modeling
function itself. Letting the API trigger predictions on demand would duplicate what
Airflow already does daily, and raises real questions (concurrent requests each
retraining 3 LightGBM models? on what schedule?) for no benefit this project actually
needs. "Airflow computes, the API serves" keeps each piece doing one thing.

### The historical join happens in the API, not in the pipeline

`quantile_backtest_results.parquet` only has `date_heure`, `q10_pred`, `q90_pred` (no
actual consumption, no RTE forecast, no median) — confirmed by reading
`quantile_backtest()`'s own code and its test
(`test_quantile_backtest_returns_one_row_per_backtest_hour_with_quantile_columns`)
before writing this ADR, not assumed. Producing a "model vs. RTE vs. actual"
comparison means joining it with `dataset.parquet` (`consommation`, `prevision_j1`) on
`date_heure`. This join happens inside `GET /forecast/history` at request time, rather
than adding a new processing step that writes a precomputed
`history_comparison.parquet` — the API can assemble the same result from files that
already exist, at a data volume (a few thousand rows) where a per-request join costs
nothing measurable. Revisit if this ever becomes a real bottleneck.

### No caching, no new storage layer

The API re-reads Parquet files on every request via the existing
`felec.ingestion.storage.load_processed()` helper — the same one `cli.py` already
uses. No in-memory cache, no database. Simpler, and correctness (always reflecting the
latest `ingest predict`/`ingest quantile-backtest` run) is free this way — a cache
would need an invalidation story for no measured performance problem today.

### Deployment/hosting: deferred

Where this API actually runs (and how a deployed instance would reach data currently
only produced locally by Airflow) is a separate, real decision — considered during
brainstorming and explicitly deferred rather than solved here. The API is built and
tested to run locally first (`uv run fastapi dev`); hosting is a follow-up ADR once
this slice itself is working.

## Alternatives considered

- **API triggers predictions live**: rejected — see above.
- **Precompute and store the history join** (a new `history_comparison.parquet`
  written by a modified `quantile_backtest()` or a new processing step): rejected for
  now — no measured need, and it would touch modeling code for a serving-layer
  convenience. Revisit if the join becomes a real bottleneck or if other consumers
  need the same joined data.
- **Caching read results in memory**: rejected — premature at this data volume and
  request rate, and adds an invalidation question with no current answer.

## Consequences

- `src/felec/api/app.py` is a new subpackage alongside `ingestion/`, `modeling/`, and
  `processing/` — a fourth clearly-scoped area of the codebase, matching the existing
  package-per-responsibility structure.
- `fastapi[standard]` is now a direct (non-dev) dependency.
- `GET /forecast/history` depends on both `quantile_backtest_results.parquet` and
  `dataset.parquet` existing; if either is missing, it 404s with a message naming the
  CLI command that produces it.
- Deployment remains unsolved by this ADR, on purpose — a public, hosted version of
  this API is a real near-term goal (not yet a settled plan), and will need its own
  decision about where the API runs and how it reaches data produced by the still-local
  Airflow pipeline (the same class of problem ADR 0008 already named for scheduling).
