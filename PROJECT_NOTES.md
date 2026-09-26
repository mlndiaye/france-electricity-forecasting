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

- Orchestrator for v2 (Airflow vs lighter options) — to be justified when v2 starts
- Dashboard technology — same
- Exact forecast cutoff hour — still to be pinned down empirically during feature
  engineering (RTE resolution consistency is now resolved, see Data sources above)

## Known data gaps (found during the first real backfill, 2026-09-25)

The processed dataset has 4 missing hours out of 23,232 in the 2024-02-01→2026-09-25
window (0.017%), all traced to real upstream RTE characteristics, not pipeline bugs:
2024-10-27 and 2025-10-26 (DST transitions — RTE's own feed has no data for that hour),
and 2026-06-30 22:00-23:00 (a transient publish-lag gap right at the `cons-def`/`tr`
boundary, where neither dataset yet covers that window). Small enough to treat as a
documented limitation, not a blocker.

## Current phase

Ingestion pipeline (v1) and exploratory analysis notebook are built, tested, and
verified against real data end-to-end. Next: baselines and modeling.
