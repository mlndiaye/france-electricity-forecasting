# ADR 0005: Error analysis by day type

## Context

The walk-forward backtest (ADR 0004) produced a single aggregate result: model MAE 1286
MW vs. seasonal-naive 3654 MW vs. RTE's own forecast 1298 MW, over 347 backtest days. An
aggregate number hides how performance varies with the type of day, which is exactly what
`PROJECT_NOTES.md`'s planned scope calls for ("error analysis by type of day: weekday/
weekend, holidays, cold spells, seasons") and what the daily walk-forward design (rather
than a single train/test split) was chosen to enable.

`data/processed/backtest_results.parquet` has `date_heure`, `consommation`, `model_pred`,
`naive_pred`, `rte_pred` — no calendar or weather context. `data/processed/dataset.parquet`
has `est_ferie`, `vacances_zone_a/b/c`, `temperature_nationale` alongside the same
`date_heure` key, so the two must be joined to build day-type categories.

## Decision

### Deliverable: a notebook, not a CLI command

`notebooks/error_analysis.ipynb`, following the same style/conventions as
`exploration.ipynb` and `model_selection.ipynb` — a one-off analysis, not something that
needs to run automatically yet. Automating this (e.g. as part of a v2 daily monitoring
run) is a natural later step, not needed now.

### Day-type categories

- **`is_weekend`**: Paris-local day of week ≥ 5 (Saturday/Sunday).
- **`is_holiday`**: `est_ferie`, already provided.
- **`is_school_holiday`**: `vacances_zone_a | vacances_zone_b | vacances_zone_c` — any
  zone on vacation. Not broken out per zone: at national aggregate demand level, the
  zones' effects mix together anyway, and splitting three ways would fragment an already
  limited ~347-day backtest sample for no identified benefit.
- **`is_cold`**: national daily mean temperature ≤ 5°C. Chosen empirically, not
  arbitrarily: over the full dataset (2024-02-01 to 2026-09-26, 970 days), 5°C is close
  to the 5th percentile of daily mean `temperature_nationale` (5th percentile ≈ 4.75°C,
  10th ≈ 6.89°C). Within the backtest window specifically, this threshold selects 23 of
  347 days — a small but workable bucket, in the same spirit as the ~347-day backtest
  itself already being treated as a limited sample (ADR 0004).
- **`season`**: meteorological seasons (Winter = Dec/Jan/Feb, Spring = Mar/Apr/May,
  Summer = Jun/Jul/Aug, Autumn = Sep/Oct/Nov), the standard convention — no need to
  justify further.

Categories are not mutually exclusive (e.g. a cold weekend holiday is all three at once);
each is analyzed independently (category vs. its complement), not as a combinatorial
cross-product, which would fragment the sample far too much to be meaningful.

### Metrics

For each category (and its complement), MAE and MAPE for `model_pred`, `naive_pred`, and
`rte_pred` against `consommation`. MAE keeps the same unit (MW) as the already-documented
aggregate result; MAPE (relative error) is added because national consumption itself
varies a lot by season (higher in winter), so an absolute MAE comparison across seasons
would partly just reflect that baseline level rather than relative forecast quality.

### Presentation

Tables plus bar charts per category, with a short interpretation of where the model
over- or under-performs relative to the naive baseline and to RTE — consistent with this
project's goal of understanding *where and why* errors happen, not just reporting a
single number.

## Alternatives considered

- **Continuous temperature correlation instead of a cold/not-cold split**: avoids picking
  an arbitrary threshold, but the project's stated planned scope explicitly calls out
  "cold spells" as a category, and a simple, empirically-grounded threshold is easier to
  read alongside the other binary categories (weekend, holiday) than a separate
  continuous analysis. Rejected for this pass; a scatter/binned view could be added later
  if the threshold-based result looks too coarse.
- **True multi-day cold-spell detection** (consecutive days below a threshold, the
  stricter meteorological definition): more rigorous, but adds real complexity
  (sequence detection, choosing a minimum run length) for a dataset with only ~347
  backtest days — too little history to expect many genuine multi-day spells anyway.
  Rejected as disproportionate for this phase.
- **Per-zone school holiday breakdown**: rejected, see Decision above.
- **A CLI command instead of a notebook**: rejected for now since nothing yet needs this
  to run automatically; matches the project's existing precedent of doing analysis in
  notebooks before promoting anything to `src/`.

## Consequences

- The 5°C cold-day threshold is a documented, adjustable parameter — not a universal
  constant. If a future season shifts the temperature distribution meaningfully, this
  threshold may need revisiting.
- Category sample sizes are small (23 cold days, and likely similarly small counts for
  some other categories once split) — per-category MAE/MAPE numbers should be read as
  indicative, not statistically definitive, consistent with the backtest's own
  already-documented "~347 days is a limited sample" caveat (ADR 0004).
- This analysis reads two existing files (`backtest_results.parquet`,
  `dataset.parquet`) and does not change the model, baselines, or backtest logic in any
  way — a pure read-only analysis layer on top of already-validated results.
