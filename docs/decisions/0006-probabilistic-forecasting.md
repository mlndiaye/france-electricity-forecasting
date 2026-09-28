# ADR 0006: Probabilistic forecasting (multi-quantile prediction intervals)

## Context

`PROJECT_NOTES.md`'s planned v1 scope calls for a probabilistic forecast, not just a point
forecast: "Gradient boosting model (LightGBM), then quantile regression or conformal
prediction for prediction intervals." ADR 0004 already built the point-forecast half of
this — `src/felec/modeling/backtest.py`'s `walk_forward_backtest` trains a LightGBM
quantile regressor at `alpha=0.5` (the median) each backtest day, tuned once via grid
search (`DEFAULT_LGBM_PARAMS`) and held fixed — but deferred the rest: turning that single
median prediction into an actual interval with a stated confidence level.

This ADR covers that remaining piece: producing a prediction interval alongside the
existing point forecast, and checking empirically whether that interval is trustworthy.

## Decision

### Approach: multi-quantile LightGBM, not conformal prediction

LightGBM already supports training a model to target any quantile via
`objective="quantile", alpha=<level>` — exactly the mechanism `walk_forward_backtest`
already uses for the median. Extending to two more quantiles (0.1 and 0.9, giving an 80%
central interval) is a direct, low-risk extension of infrastructure already built, tested,
and tuned.

Conformal prediction (e.g. conformalized quantile regression) gives a stronger
theoretical guarantee (finite-sample marginal coverage) but requires real care to adapt
correctly to time-series/walk-forward data (the standard conformal recipe assumes
exchangeable data, which daily electricity demand is not) — meaningfully more complexity
for a project already this close to its stated v1 deadline (mid-November 2026), without a
correspondingly large practical benefit at this stage. Rejected for now; a natural
candidate to revisit if empirical coverage from the simpler approach turns out
significantly miscalibrated.

### Quantile levels: 0.1 / 0.5 / 0.9

An 80% central interval — wide enough to be a meaningful test of calibration, narrow
enough to keep compute cost bounded. The median (0.5) is already computed and stored in
`data/processed/backtest_results.parquet` (`model_pred`, from the existing
`walk_forward_backtest`); only 0.1 and 0.9 are new work, saving roughly a third of the
compute a naive re-run of all three levels would cost.

Rejected: a finer 5-level grid (0.1/0.25/0.5/0.75/0.9) — would roughly double the new
compute again for a pinball-loss curve this project doesn't currently need; the coverage/
width evaluation below only needs the two tail levels.

### Hyperparameters: reused from the median model, not re-tuned per quantile

`DEFAULT_LGBM_PARAMS` (tuned once against the median's pinball loss, ADR 0004) is reused
unchanged for the 0.1 and 0.9 models. This is standard practice for multi-quantile
gradient boosting (the same tree structure typically transfers reasonably well across
quantile levels) and avoids a second expensive tuning cycle for an uncertain gain,
consistent with ADR 0004's existing "tune once, hold fixed" position on hyperparameters.

Rejected: tuning each tail quantile separately — more rigorous, but a full extra
notebook-and-grid-search cycle per quantile, disproportionate against this project's
remaining timeline.

### Code structure: a new function sharing the existing walk-forward loop, not a rewrite

`walk_forward_backtest` (ADR 0004, already reviewed, merged, and depended on by
`notebooks/error_analysis.ipynb` via its `model_pred` column) is not modified in behavior,
signature, or output schema. The shared day-by-day walk-forward mechanics (expanding
training window, D-1 cutoff, per-day train/predict split) are extracted into a private
helper that both `walk_forward_backtest` and a new `quantile_backtest` function consume,
avoiding duplicating that logic while leaving the existing, already-validated artifact
untouched.

`quantile_backtest(df, backtest_start, quantiles=(0.1, 0.9), cutoff_hour=12,
lgbm_params=None)` trains one model per (day, quantile) and returns `date_heure`,
`q10_pred`, `q90_pred` — deliberately not `consommation`/`naive_pred`/`rte_pred` again,
since those are already in `backtest_results.parquet` and would just be redundant.

A new CLI subcommand, `ingest quantile-backtest`, mirrors the existing `backtest`
subcommand's pattern and saves to `data/processed/quantile_backtest_results.parquet`.

Rejected: generalizing `walk_forward_backtest` itself to accept a list of quantiles and
rename `model_pred` to something like `q50_pred`. Would touch an already-reviewed,
merged, and depended-upon function and its output schema for no functional benefit here —
additive is safer than modify-in-place for work that's already shipped.

### A real finding: quantile crossing, fixed by sorting per row

The first real run of `quantile_backtest` against the full dataset found 41 of 8,303 rows
(0.49%) where the q0.1 model's prediction exceeded the q0.9 model's — an inverted
interval. This is a known property of independently-trained quantile regressors: nothing
in training a `LGBMRegressor(objective="quantile", alpha=0.1)` and a separate
`LGBMRegressor(objective="quantile", alpha=0.9)` constrains their outputs to agree on
which is larger for any given row, since each model minimizes its own pinball loss with
no awareness of the other.

The fix: sort each row's predictions across quantile levels before assigning them back to
columns (the standard "rearrangement" approach, Chernozhukov, Fernández-Val, and Galichon,
2010). This guarantees `q10_pred <= q90_pred` for every row by construction, without
retraining — it only reorders each row's already-computed predictions. Verified with a
dedicated test that deliberately forces a crossing (via a mocked `predict`) and confirms
the function corrects it.

Rejected: leaving the 0.49% of crossed rows as-is and excluding them from the coverage
evaluation. Would avoid touching the model code, but leaves a real, avoidable defect in
the function's contract (an interval whose "lower" bound exceeds its "upper" bound is not
a valid interval at all) for a one-line, well-established fix — not proportionate to skip
here.

### Evaluation: coverage and interval width, not a per-category breakdown

A new notebook, `notebooks/probabilistic_evaluation.ipynb`, joins the two results files on
`date_heure` and reports:

- **Empirical coverage**: the fraction of backtest hours where `consommation` falls inside
  `[q10_pred, q90_pred]`. For a well-calibrated 80% interval, this should sit close to 80%.
- **Mean interval width**: `mean(q90_pred - q10_pred)`, in MW — how sharp (narrow) the
  interval is; a technically well-calibrated but extremely wide interval would still be a
  weak result.
- One illustrative plot: the `[q10, q90]` band against actual consumption over a sample
  week, so the interval's real shape is visible, not just its summary statistics.

Not repeating `error_analysis.ipynb`'s full weekday/holiday/season/cold-day breakdown for
the interval this pass — that notebook already identified where the *point* forecast
struggles (notably cold days); a full re-run of that same breakdown for interval coverage
is a reasonable follow-up once the basic global calibration result is in hand, not
something to build blind before knowing whether it's needed.

### No comparison to RTE for the interval itself

RTE's public forecast (`prevision_j1`) is a point forecast only — RTE does not publish a
comparable interval in this dataset. The interval evaluation is therefore a self-check of
this project's own calibration, not a benchmark comparison; `rte_pred` remains relevant
only for the (already-covered) point-forecast comparison in `backtest_results.parquet`.

## Alternatives considered

Covered inline per decision above (conformal prediction, a finer quantile grid,
per-quantile retuning, modifying `walk_forward_backtest` in place, a per-category interval
breakdown). Also:

- **Skipping this phase and calling `prevision_j1`/naive/model MAE numbers "good enough"
  for v1**: rejected — the project's own stated goal explicitly asks for a probabilistic
  forecast, not just a point forecast; this is the one clearly-scoped piece of v1 not yet
  built.

## Consequences

- `data/processed/quantile_backtest_results.parquet` is a second results file,
  intentionally not merged into `backtest_results.parquet` — anything consuming both needs
  to join them on `date_heure` (already how `error_analysis.ipynb` joins
  `backtest_results.parquet` with `dataset.parquet`, so this is a familiar pattern, not a
  new one).
- Running `quantile_backtest` over the real ~347-day backtest window costs roughly twice
  the already-known ~1h40 per-quantile runtime (two new quantiles, not three, since the
  median is reused) — a real, bounded, one-time compute cost, run in the background as the
  existing `backtest` command already was.
- If empirical coverage comes out far from 80% (e.g. well under, meaning the interval is
  overconfident), that is a real, reportable finding — consistent with this project's
  standard of reporting honestly rather than only when results are flattering — and would
  be a natural motivation to revisit conformal prediction later, not something to hide.
  **This is exactly what happened**: the real backtest's empirical coverage is 55.1%
  (roughly symmetric miss — 25.0% of hours above `q90_pred`, 19.8% below `q10_pred`), well
  under the 80% target. See `notebooks/probabilistic_evaluation.ipynb` for the full
  breakdown. Conformal prediction, rejected above for complexity reasons, is now the
  concretely motivated next step rather than a hypothetical one.
- Hyperparameters shared across quantiles are a documented simplification, not a proven
  optimum for the tail quantiles specifically.
