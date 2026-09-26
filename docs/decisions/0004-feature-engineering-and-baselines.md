# ADR 0004: Feature engineering, baselines, and backtesting protocol

## Context

ADR 0001 deferred the exact forecast cutoff hour to feature engineering time. ADR 0003's
notebook produced two findings that directly shape this phase: consumption 24h and 168h
ago are both strongly correlated with the target (0.934 and 0.891), and RTE's own
`prevision_j1` has a systematic, time-varying bias — the actual benchmark to compare
against, not just an abstract target.

Nothing in `src/felec` yet builds features or trains a model; this ADR covers that next
phase, before writing the implementation plan.

## Decision

### Forecast cutoff: 12:00 (noon) on D-1, as a documented working assumption

Our own ingestion pipeline only keeps the latest known value per `date_heure` key
(`storage.save_raw`'s dedup-by-key design, ADR 0002) — it was never built to record
*when* a value first appeared, so the actual historical publication time of RTE's
`prevision_j1` cannot be reconstructed retroactively from data already collected.
Determining it precisely would require monitoring `refresh()` live, going forward, for
several days.

Rather than block this phase on that observation, 12:00 is adopted now as a working
assumption — it matches the French day-ahead electricity market's own gate closure
time, a real external convention, not an arbitrary guess. A lightweight live-monitoring
check (comparing this project's real day-to-day `refresh()` runs) can confirm or correct
it later without invalidating anything built on this assumption in the meantime, since
the cutoff is a parameter to the feature-building code, not baked into its logic.

Rejected: waiting several days for a live-measured cutoff before starting this phase —
too high a cost for a parameter unlikely to move far from the market-convention value.

### Lag features: 168h (1 week) only, not 24h and not 48h

A cutoff at noon on D-1 makes a naive "24h-ago" feature unsafe for roughly half of a
day's target hours: for a target hour before noon, D-1 at that hour has already
happened by the cutoff; for a target hour after noon, it hasn't — the feature would
silently use information from the future relative to the model's own prediction
instant. This is the same leakage class already handled for weather in ADR 0001, just
resurfacing inside the project's own historical data, not an external source.

A 48h-ago feature is not affected by this specific problem (2 full days before D-1 is
always in the past, whatever the cutoff), but was checked and rejected on its own
merits: its autocorrelation with the target (0.863) is measurably weaker than 168h's
(0.891) — verified directly against the real dataset, alongside 72h/96h/120h (0.836 /
0.821 / 0.816), all lower still. The dip between 2 and 6 days and the recovery at 7
days is explained by day-of-week misalignment: a Monday target's 48h-ago point falls on
a Saturday, a different consumption regime (already characterized in the ADR
0003 notebook); 168h-ago always falls on the same weekday as the target, restoring that
alignment. 168h is therefore both safe and, among the safe options, the strongest
single lag — 48h would add a weaker, redundant feature for no identified benefit.

### Recent-trend feature: an as-of-cutoff aggregate, not a variable-safety lag

To recover the "recent conditions" signal a naive 24h-lag would have given, without its
per-target-hour safety problem: an aggregate anchored to the cutoff instant itself
(e.g., mean consumption over the 24h window ending at noon on D-1), computed once per
forecast issuance and joined to all 24 target-hour rows for day D. Safe by
construction — it never references anything later than the cutoff, regardless of which
hour of D it's attached to.

### Full feature set

- `lag_168h`: consumption 168h before the target hour.
- `recent_trend`: mean consumption over the 24h window ending at the cutoff.
- `temperature_nationale`: already forecast-based, not observed (ADR 0001/0002) — no
  additional leakage handling needed.
- `est_ferie`, `vacances_zone_a/b/c`: known months in advance — no leakage risk.
- Calendar-arithmetic features (hour of day, day of week, month): known in advance by
  construction.
- **Explicitly excluded**: `prevision_j1` (RTE's own forecast) as a model input —
  already decided earlier in this project: it is the benchmark to compare against, not
  an ingredient, or the comparison would be meaningless.

### Baselines

- **Seasonal naive**: the `lag_168h` value itself, used directly as the prediction, no
  model at all. The floor any real model must clear.
- **RTE's `prevision_j1`**: already in the dataset, the real benchmark.

Both require no additional leakage handling: the naive baseline reuses the already-safe
`lag_168h` feature, and RTE's forecast is already a frozen, published value (ADR 0001).

### No scaling or categorical encoding — a consequence of the model choice, not an omission

Gradient-boosted trees split on thresholds, so they are invariant to monotonic
transforms of numeric features (no benefit from scaling/standardizing) and handle
integer/categorical features and missing values natively (no one-hot encoding, no
manual imputation needed). This is a real advantage of the model choice below, not
something skipped by oversight. The one real cleaning step: rows with a null
`consommation` (the documented tail-of-dataset gap) are excluded from training, since
there is no label to learn from — they are not otherwise imputed or altered.

### Model choice, sanity check, and hyperparameter tuning

**LightGBM is the target model**, not picked arbitrarily now — it was already this
project's stated plan (`PROJECT_NOTES.md`), and it fits this problem concretely: it
supports quantile regression natively (directly serving the probabilistic-forecast
requirement), and gradient-boosted trees are the standard choice for tabular data with
mixed numeric/categorical features and non-linear relationships (the temperature/
consumption J-shape, the holiday/vacation confounding — both found in the ADR 0003
notebook). Re-deriving the model family from scratch via a full multi-algorithm
comparison was considered and rejected as disproportionate effort for this project's
timeline (`PROJECT_NOTES.md`: v1 target mid-November 2026) given the concrete reasons
above — but committing to it *without any check* was also rejected. Two lighter-weight
verification steps instead:

1. **A sanity check, not a bake-off**: carve the last ~2-3 months off the initial
   training window as a held-out validation slice (see backtesting protocol below for
   the exact windows). Train both a default-ish LightGBM and a plain linear regression
   on the same feature set on the remaining training data, evaluate both on the
   validation slice. This confirms — on this project's actual data, not by assumption —
   that gradient boosting earns its added complexity over the simplest reasonable
   alternative, before investing further effort in it.
2. **Hyperparameter tuning, once, on the same validation slice** — not skipped, and not
   re-run inside the daily backtest loop. Search over LightGBM's key hyperparameters
   (`num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`) to minimize
   pinball loss on the validation slice, using the same train/validation split as the
   sanity check. The resulting hyperparameters are then held fixed for every retrain in
   the walk-forward backtest — only the training *data* grows day to day, not the
   hyperparameters. Retuning at every one of the ~365 backtest days would be
   disproportionate and is not standard practice; a single tuning pass, validated on
   data the backtest window never touches, avoids leaking backtest information into the
   tuning decision.

### Backtesting protocol: daily walk-forward, expanding window

- **Initial training window**: ~2024-02-01 → ~2025-10-15 (~20 months). Split further for
  the sanity check and tuning above: ~2024-02-01 → ~2025-08-15 (~18 months) to fit on,
  ~2025-08-16 → ~2025-10-15 (~2 months) as the held-out validation slice. Neither piece
  overlaps the backtest window below, so tuning never sees backtest data.
- **Backtest window**: ~2025-10-16 → present (~11-12 months) — meets the "at least one
  full year" requirement already stated in `PROJECT_NOTES.md`.
- **Mechanics**: for each day D in the backtest window, train on every row strictly
  before the D-1 noon cutoff, predict D's 24 hours, record the error, advance one day,
  repeat. The training window expands (never shrinks or slides) — more history should
  only help, and history is already scarce.
- This mirrors the v2 production design (`PROJECT_NOTES.md`: "daily automated run")
  exactly, rather than approximating it with a single train/test split. LightGBM trains
  in seconds at this data volume, so ~365 retrains costs a few minutes total, not a
  meaningful compute burden.
- Produces ~365 independent daily error measurements (per baseline and per model),
  enabling the error-by-day-type analysis already planned in `PROJECT_NOTES.md`
  (weekday/weekend, holidays, cold spells, seasons), rather than a single aggregate
  number that hides how performance varies.

Rejected: a single train/test split — cheaper to compute, but doesn't mirror how the
system will actually run in production, and collapses a year of varying conditions into
one number instead of ~365 comparable data points.

### Code structure

New `src/felec/modeling/` package, parallel to `ingestion/` and `processing/` (a third
pipeline stage: ingest → assemble → model):

```
src/felec/modeling/
  features.py    # lag_168h, recent_trend, calendar-arithmetic features
  baselines.py   # seasonal naive
  backtest.py    # walk-forward loop, metrics vs. both baselines
```

## Alternatives considered

Covered inline per decision above (waiting for a measured cutoff, 24h/48h lags, a
single train/test split). Also:

- **Full multi-algorithm comparison** (linear/ridge, random forest, LightGBM, XGBoost,
  each with real tuning) before committing to a model family: more rigorous, but a
  disproportionate time cost against this project's v1 deadline given LightGBM already
  has concrete, specific reasons to expect it to fit well here (native quantile
  support, standard for this data shape). The lighter sanity-check-plus-tuning approach
  above was chosen instead as the middle ground — not skipping verification entirely,
  but not re-deriving the model family from zero either.

## Consequences

- The cutoff hour (12:00) is a documented assumption, not a measured fact — flagged
  here so it isn't mistaken for one; a future live-monitoring check can confirm or
  correct it without requiring a redesign, since it's a parameter to the feature code.
- Only ~11-12 months of backtest history is available given the data volume — fewer
  rare-event examples (severe cold snaps, heatwaves) than a longer history would give,
  consistent with the limitation already flagged in ADR 0001.
- `prevision_j` (RTE's same-day updated forecast) remains in the dataset but unused by
  either the model or the baselines, for the same reason `prevision_j1` is excluded from
  model inputs — it is not available at the D-1 cutoff either.
- If the linear-regression sanity check were ever to *beat* LightGBM on the validation
  slice, that would be a real, actionable finding — worth stopping to re-examine before
  proceeding to tuning and the full backtest, not something to override by continuing
  with LightGBM regardless.
