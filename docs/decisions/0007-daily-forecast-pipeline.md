# ADR 0007: Daily forecast pipeline

## Context

`PROJECT_NOTES.md`'s v2 scope starts with "daily automated run" — the piece that turns
the validated v1 model into something that actually produces a real forecast for
tomorrow, not just backtested historical evaluations. This ADR covers the first, smallest
slice of that: a single function/command that, run today, produces tomorrow's forecast.
Scheduling it to run automatically (an external orchestrator — Airflow or a lighter
alternative) is a separate, later decision, not covered here.

Before designing this, two live checks were run against the real APIs (not assumed):

- **Weather**: the Previous Runs API (already used by `weather.py`, endpoint
  `previous-runs-api.open-meteo.com`, parameter `temperature_2m_previous_day1`) was
  queried live for tomorrow's date. Result: all 24 hours returned non-null temperatures.
  The archive is populated close to real time — no new API integration needed.
- **RTE**: the `eco2mix-national-tr` dataset was queried live for tomorrow's date.
  Result: `prevision_j1` (RTE's own day-ahead forecast) is already populated for
  tomorrow's rows, `consommation` is null (correctly — it hasn't happened yet).

**A real gap found**: neither `weather.refresh()` nor `rte.refresh()` actually fetches
tomorrow — both currently compute a window ending at `today` (`backfill(today - N,
today)`), even though the underlying data for tomorrow is already available live. This
is a genuine bug relative to this feature's needs, not a design question — fixed as part
of this work (see Decision).

## Decision

### Fix `refresh()`'s date window to include tomorrow

`weather.refresh()` and `rte.refresh()` both change their computed end date from `today`
to `today + 1 day`. One-line change in each, no new API integration, no change to
`backfill()` itself (which already accepts an arbitrary end date). `calendar.refresh()`
already covers the current and next year, so tomorrow is never a problem there, even
across a year boundary.

### A new module, `src/felec/modeling/forecast.py`, not an addition to `backtest.py`

`backtest.py` already holds two public functions (`walk_forward_backtest`,
`quantile_backtest`) plus their shared helper — evaluating model performance against
*known* historical outcomes. Predicting tomorrow is conceptually different: there is no
ground truth to compare against, no loop over a range of days, just one training pass and
one prediction. Keeping it in a separate module keeps each file's responsibility
narrow — `backtest.py` stays about backtesting, `forecast.py` is about producing a real,
actionable forecast.

`predict_next_day(df, target_date, cutoff_hour=12, quantiles=(0.1, 0.5, 0.9),
lgbm_params=None) -> pd.DataFrame`:
- Builds features the same way as `backtest.py` (`build_features`), trains one LightGBM
  quantile model per requested quantile on every row strictly before `target_date`'s D-1
  cutoff (same leakage-safety guarantee as the backtest — no new logic invented here).
- Predicts `target_date`'s 24 hours for each quantile.
- Sorts each row's predictions across quantile levels before assigning them to columns,
  reusing the exact fix already built and tested for `quantile_backtest`'s "quantile
  crossing" bug (ADR 0006) — extracted into a small shared helper,
  `_sorted_quantile_columns`, in `backtest.py`, imported by `forecast.py`, so the fix
  lives in one place rather than being copied.
- Returns `date_heure`, `q10_pred`, `q50_pred`, `q90_pred`, `naive_pred` (via the existing
  `seasonal_naive`), `rte_pred` (from the dataset's own `prevision_j1`, already available
  for tomorrow per the live check above). No `consommation` column — it doesn't exist yet
  for a future day, and including an always-null column would be misleading rather than
  useful.

Rejected: reusing `_iter_backtest_days` for this. That generator is built to iterate over
a *range* of days with known outcomes; forcing a single, ground-truth-less future day
through it would mean either faking a one-day range (awkward) or generalizing the helper
to handle a case it was never meant for. A small, direct function is clearer.

### New CLI subcommand: `ingest daily-forecast`

Mirrors the existing subcommands' shape: `refresh()` → `build_dataset()` →
`predict_next_day(df, target_date=tomorrow)` → save to
`data/processed/forecast_latest.parquet` (overwritten each run, same convention as
`backtest_results.parquet`) → print a one-line summary.

No partial-failure handling: if any step raises, the whole command fails loudly. This
matches every existing CLI command in this project and is the right behavior for
something eventually run by a scheduler — a cron job's failure should be visible (a
non-zero exit code, an unmistakable log line), not silently half-completed. Retry logic
or graceful degradation (e.g., falling back to stale weather data if the API is down) is
explicitly out of scope for this first version — it would risk producing a forecast that
*looks* normal but is quietly built on stale inputs, which is worse than a visible
failure.

### Output includes the naive and RTE baselines, not just the model

Same three-way comparison already used everywhere else in this project
(`backtest_results.parquet`, the notebooks). Costs nothing extra (the naive baseline is
already computed via `seasonal_naive`, RTE's forecast is already in the ingested data)
and means tomorrow's forecast is immediately readable in context, without a separate
lookup.

## Alternatives considered

Covered inline above (reusing `_iter_backtest_days`, adding to `backtest.py` instead of a
new module, partial-failure handling, model-only output). Also:

- **Building the external scheduler (Airflow/cron) in this same pass**: rejected — this
  ADR is deliberately scoped to "can a single command produce tomorrow's forecast
  correctly," which is a precondition for scheduling anything. Automating *when* it runs
  is a separate decision with its own tradeoffs (Airflow vs. lighter options,
  `PROJECT_NOTES.md`'s own open question), not to be bundled in.

## Consequences

- `data/processed/forecast_latest.parquet` is a new, small artifact (24 rows, one day),
  gitignored like every other file in `data/processed/`, overwritten daily once this is
  scheduled.
- This command hits real external APIs (RTE, Open-Meteo) every time it runs. Running it
  manually is fine; running it unattended on a schedule is a separate decision requiring
  its own explicit setup, not something this ADR turns on by itself.
- `forecast.py` and `backtest.py` now both depend on `_sorted_quantile_columns` — a
  change to that helper (e.g., a different crossing-resolution strategy) affects both
  real backtesting and real daily forecasts identically, which is the intended behavior,
  not a coincidence to be careful about.
- The daily forecast, once run, has no baseline to be measured against until the real day
  passes — unlike the backtest, "did this forecast turn out to be good" for a specific
  day can't be known immediately. Comparing predicted vs. actual over time is a natural
  candidate for a later monitoring feature, not built here.
