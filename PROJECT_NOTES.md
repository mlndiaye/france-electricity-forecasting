# Project: france-electricity-forecasting

Day-ahead probabilistic forecasting of French electricity demand, benchmarked
daily against RTE's official forecast. These notes track the project's goal, the
data sources behind it, and the decisions made along the way — see
`docs/decisions/` for the full reasoning behind each one.

## Goal

Every day, produce a probabilistic forecast of French national electricity consumption
for the next day (hourly or half-hourly), compare it automatically to RTE's official
day-ahead forecast and to actual consumption, and monitor performance over time.

The point is not to "beat RTE" (unlikely at national level) but to build a rigorous,
honest forecasting system and to understand where and why errors happen.

## Data sources (verified — see `docs/decisions/0001-problem-definition-scope-and-weather-window.md`)

- **RTE éCO2mix** via ODRÉ: two datasets chained with no gap —
  `eco2mix-national-cons-def` (2012 → ~2026, consolidated/definitive) and
  `eco2mix-national-tr` (rolling ~3-month window, 15-min refresh). Confirmed fields
  include `consommation`, `prevision_j1`, `prevision_j`, full production-by-source
  breakdown, cross-border exchanges, CO2 intensity. `prevision_j1` is frozen at
  publication time — safe to use as a benchmark, no leakage. Confirmed via the real
  2024-02-01→2026-09-25 backfill: `consommation` is published at 30-minute resolution
  even in 2024-2026 (null at :15/:45), while `prevision_j1`/`prevision_j` are genuinely
  15-minute with no nulls — the earlier "resolution uniform?" open point is resolved:
  it's 30-min for realized consumption throughout the study window, handled correctly
  by `aggregate_rte_hourly`'s mean-based hourly aggregation regardless. The ODRÉ
  Explore v2.1 API also hard-caps `offset + limit <= 10000` per query — the RTE
  connector paginates by `date_heure` cursor, not offset, to avoid this.
- **Weather**: genuine day-ahead *forecasts* (not observations) are only archived from
  **~January 2024** onward (empirically verified against real API responses, not just
  docs), via Open-Meteo's Previous Runs API (AROME 2.5 km, fixed lead times). This is
  the binding constraint on the study window — see ADR 0001. Météo-France's public API
  only keeps 14 rolling days, unusable for backtesting. ERA5 is observed weather, not
  forecast — reference only, never a model input.
- **Calendar**: public holidays via `calendrier.api.gouv.fr`, school holidays via the
  Ministry of Education's school calendar API (zones A/B/C). Both confirmed, no issues.

## Key technical concern: data leakage

A forecast issued on day D-1 must only use information available at that time.
In particular, using *observed* weather for day D to predict day D is leakage:
in production only *forecast* weather is available. This is handled explicitly and
documented (see ADR 0001).

## Planned scope

**v1 (target: mid-November 2026)**
- Reproducible ingestion pipeline (RTE + weather + calendar)
- Exploratory analysis
- Baselines: seasonal naive, RTE day-ahead forecast as benchmark
- Gradient boosting model (LightGBM), then quantile regression or conformal
  prediction for prediction intervals
- Rolling-origin backtesting over at least one full year
- Error analysis by type of day (weekday/weekend, holidays, cold spells, seasons)

**v2**
- Daily automated run, model registry (MLflow), performance and drift monitoring
- Serving via FastAPI, simple dashboard (forecast vs RTE vs actual)
- Extension: probability of RTE "Tempo" red days, derived from the probabilistic forecast

## Metrics

- Point forecast: MAE, MAPE, RMSE
- Probabilistic forecast: pinball loss, interval coverage and width
- Always reported relative to baselines and to RTE's forecast

## Settled decisions (2026-09-24, see ADR 0001)

- **Study window**: ~January 2024 → present (growing daily) — bounded by real weather
  forecast availability, not by RTE's longer history. Training, backtesting, and the
  RTE comparison all stay inside this window.
- **Target resolution**: hourly.
- **Forecast issue time**: not fixed arbitrarily — determined empirically during
  ingestion from when `prevision_j1` actually appears and when D-1 weather forecasts
  become available, then fixed to the latest compatible cutoff.
- **Scope**: national only for v1 (already implied by the data: éCO2mix's "France"
  perimeter excludes Corsica, which RTE doesn't operate).

## Open decisions (deferred, not blocking v1 start)

- Dashboard technology — to be justified when that slice of v2 starts
- A live-monitoring check to confirm or correct the assumed 12:00 forecast cutoff hour
  (see `docs/decisions/0004-feature-engineering-and-baselines.md`)

## Known data gaps (found during the first real backfill, 2026-09-25)

The processed dataset has 4 missing hours out of 23,232 in the 2024-02-01→2026-09-25
window (0.017%), all traced to real upstream RTE characteristics, not pipeline bugs:
2024-10-27 and 2025-10-26 (DST transitions — RTE's own feed has no data for that hour),
and 2026-06-30 22:00-23:00 (a transient publish-lag gap right at the `cons-def`/`tr`
boundary, where neither dataset yet covers that window). Small enough to treat as a
documented limitation, not a blocker.

## Current phase

Ingestion pipeline, exploratory analysis, feature engineering, baselines, a daily
walk-forward backtest, error analysis by day type, and a first probabilistic (multi-
quantile) forecast are all built, tested, and verified against real data end-to-end.

- **Point forecast** (`docs/decisions/0004-feature-engineering-and-baselines.md`): real
  backtest over 2025-10-16→2026-09-26 (347 days), model MAE 1286 MW vs. seasonal-naive MAE
  3654 MW vs. RTE's own day-ahead forecast MAE 1298 MW — the model clears the naive
  baseline decisively and essentially matches RTE.
- **Error analysis by day type** (`docs/decisions/0005-error-analysis-by-day-type.md`):
  the model matches or beats RTE almost everywhere (notably on public holidays and in
  Spring), but is clearly worse than RTE on cold days (2760 MW vs. 1764 MW MAE) — a real,
  documented limitation attributed to few cold-day examples in the backtest history.
- **Probabilistic forecast** (`docs/decisions/0006-probabilistic-forecasting.md`): an 80%
  prediction interval (LightGBM quantile regression at 0.1/0.9, alongside the existing 0.5
  median). The real backtest empirical coverage is **55.1%**, well below the 80% target —
  the interval is meaningfully overconfident (roughly symmetric: actual demand exceeds the
  upper bound 25.0% of the time, falls below the lower bound 19.8% of the time). A real,
  honestly-reported limitation, not swept under the rug: the natural next step is
  conformal prediction (rejected earlier for complexity, now motivated by this result).

v2 has started: **daily forecast pipeline**
(`docs/decisions/0007-daily-forecast-pipeline.md`) — `ingest daily-forecast` chains a real
data refresh, dataset rebuild, retraining, and prediction into one command, verified
end-to-end against the live RTE and Open-Meteo APIs (both connectors previously stopped
fetching at "today" even though tomorrow's data is already published live — a real gap
found and fixed as part of this work).

**Scheduling** (`docs/decisions/0008-airflow-daily-scheduling.md`): the pipeline now runs
automatically once a day via a local Airflow instance (Docker, `LocalExecutor`, 3 separate
tasks — `refresh` → `build_dataset` → `predict` — so a transient failure in one step
doesn't force re-running the whole pipeline), scheduled at 16:00 Europe/Paris with 1 retry
per task, `catchup=False`. Verified end-to-end for real: built the Docker image, brought up
the full 5-service stack, triggered the DAG, and confirmed all 3 tasks reached `success`
with a real `forecast_latest.parquet` written to the host filesystem. One real operational
gotcha found during that verification: a manually-triggered run on a *paused* DAG (Airflow's
default) is created but never executes — it sits in `queued` forever, since pause blocks
task execution for manual triggers too, not just the scheduler's automatic firing. The
README's setup instructions include the required `airflow dags unpause` step.

**Model tracking** (`docs/decisions/0009-mlflow-model-tracking.md`): every `ingest predict`
run now logs to MLflow — hyperparameters, cutoff hour, training set size, and the three
trained per-quantile LightGBM models — as a local, file-based audit trail (run-level
tracking, not a full Model Registry; see the ADR for why). Verified end-to-end for real:
ran `ingest predict` against live-refreshed data, confirmed real (non-null) params/metrics
in the resulting run via `mlflow.search_runs`, confirmed all 3 model artifacts
(`model_q10`/`model_q50`/`model_q90`) were actually persisted, and confirmed a second
invocation creates an independent run rather than overwriting the first. One real
implementation-time finding: MLflow 3.x puts its raw filesystem tracking backend
(`file:./mlruns`) in maintenance mode and refuses to use it without an explicit opt-out —
switched to a local SQLite file (`mlruns.db`) for tracking metadata instead (model
artifacts still land in `mlruns/`); found by actually running the command, not assumed.

**Serving API** (`docs/decisions/0010-serving-api.md`): a read-only FastAPI service
(`src/felec/api/app.py`) exposes `GET /health`, `GET /forecast/latest`, and
`GET /forecast/history?start=&end=`. It never triggers a prediction itself — Airflow
already computes everything daily; the API only serves already-written Parquet
results via the existing `load_processed()` helper. `quantile_backtest_results.parquet`
turned out to only hold `date_heure`/`q10_pred`/`q90_pred` (no actual consumption, no
RTE forecast — confirmed by reading `quantile_backtest()`'s own code and tests before
designing this), so `/forecast/history` joins it with `dataset.parquet` at request
time rather than requiring a new precomputed file. Verified end-to-end for real:
started the server, hit all 3 endpoints against live-refreshed data and a real (30-day)
quantile backtest, confirmed the join and date filtering both work against real
numbers, confirmed 404s when a file is genuinely missing. Deployment (a public link)
was raised as a near-term goal during brainstorming but deliberately deferred to its
own future decision — the same class of "local produces the data, a public instance
would need to reach it" problem ADR 0008 already named for scheduling.

A dashboard consuming this API, deploying it publicly, and the Tempo extension are
still ahead.

**Known limitation**: the current 80% prediction interval is not well-calibrated (55.1%
real coverage). Point-forecast numbers above are unaffected by this — it's specific to the
interval, not the median prediction.

Next: v1's planned scope is otherwise complete; remaining work is either portfolio polish
(README, results summary) or a follow-up phase on conformal prediction to fix the
interval's calibration.
