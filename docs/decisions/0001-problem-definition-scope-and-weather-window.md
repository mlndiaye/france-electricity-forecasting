# ADR 0001: Problem definition — study window, resolution, forecast issuance

## Context

Before writing any ingestion or modeling code, we verified the actual availability and
shape of the data sources listed in `PROJECT_NOTES.md`.

**RTE éCO2mix / ODRÉ**, confirmed by querying the API directly:

- Two datasets chain together with no gap: `eco2mix-national-cons-def` (Jan 2012 →
  ~2026, consolidated/definitive, daily refresh) and `eco2mix-national-tr` (rolling
  ~3-month window, 15-min refresh). Their boundary records match exactly.
- Confirmed fields include `consommation`, `prevision_j1`, `prevision_j`, and a full
  production-by-source breakdown, cross-border exchanges, and CO2 intensity.
- `prevision_j1` is frozen at publication time (verified: a record for a future day
  already carries a non-null `prevision_j1` while `consommation` is still null) — it is
  RTE's actual published day-ahead forecast, safe to use as a benchmark with no leakage.
- Open question, not yet resolved: whether realized consumption is at 15-min resolution
  throughout, or 30-min pre-~2021 and 15-min after. Conflicting signals from available
  documentation; must be checked empirically against raw records during ingestion.
  **Resolved (2026-09-25, first real backfill)**: `consommation` is published at
  30-minute resolution throughout the 2024-2026 study window (null at :15/:45), while
  `prevision_j1`/`prevision_j` are genuine 15-minute with no nulls. Not a pre-2021 vs.
  post-2021 split as the documentation ambiguity suggested — just a permanent
  per-field resolution difference. `aggregate_rte_hourly`'s mean-based hourly
  aggregation handles this transparently either way, so no code change was needed.
- Access: paginated REST API + bulk CSV/JSON export, 50k calls/month quota — ample for a
  daily batch pipeline.

**Weather**, the critical finding:

- No source provides genuine archived day-ahead weather *forecasts* for the full
  2012–2026 span.
- Open-Meteo's Historical Forecast API stitches together the first hours of successive
  model runs — this approximates fresh observations, not a forecast at a fixed lead time.
- Open-Meteo's **Previous Runs API** gives each variable at a fixed lead time (1–7 days)
  — the correct tool to reconstruct "what the D-1 forecast for day D actually said." The
  high-resolution model relevant to France (AROME 2.5 km) was empirically tested by
  querying real dates until we found the boundary: null before ~2024-01-15, real values
  from 2024-02-01 — so the archive effectively starts **~mid/late January 2024**, not the
  November 2022 figure first assumed from a different, unrelated Open-Meteo product
  (the Historical Forecast API, which is not what we use). A coarser global model (GFS,
  ~25–50 km) archives further back, to ~2021 — a documented alternative, not chosen
  (see Alternatives).
- Météo-France's public API only retains 14 rolling days of forecast archives — usable
  for v2 production, not for historical backtesting.
- ERA5 reanalysis (available since 1940) is *observed*, reconstructed weather, not a
  forecast — usable only as a controlled reference, never as an input feature for a
  model that must emit at D-1.
- Free tier, no API key, 10,000 calls/day non-commercial — sufficient.

**Calendar**: two official free APIs confirmed — `calendrier.api.gouv.fr` (public
holidays) and the Ministry of Education's school calendar API (zones A/B/C, historical
coverage). No issues here.

The consequence: genuine day-ahead weather forecasts (AROME, high-resolution) only exist
from ~January 2024, while RTE's own data goes back to 2012. This mismatch has to be
resolved deliberately — it is exactly the data-leakage concern already flagged in
`PROJECT_NOTES.md`.

## Decision

1. **Study window = the clean weather window.** The project's official study period is
   ~January 2024 → present (and growing daily), not the full RTE history. Training,
   backtesting, and the comparison against RTE's forecast all happen inside this window.
2. **Target resolution: hourly.** Chosen over 30-min or 15-min because it sidesteps the
   unresolved RTE resolution ambiguity across years, matches the native resolution of the
   weather models, and is standard in load-forecasting literature.
3. **Forecast issuance (cutoff) time: determined empirically, not assumed.** During
   ingestion, inspect exactly when `prevision_j1` first appears for a given target day in
   the RTE data, and when D-1 weather forecasts become available via the Previous Runs
   API. Pick the latest cutoff compatible with both, so the model is as informed as RTE
   is and the comparison stays fair.

## Alternatives considered

- **Use ERA5 as a weather proxy before Nov 2022, real forecasts after**: rejected as the
  default — it reintroduces the exact leakage this project is meant to avoid, non-uniformly
  across time, which would be hard to defend and would quietly inflate pre-2022 performance.
- **Train on the full 2012–2026 history with ERA5 everywhere, evaluate only on the clean
  window**: kept as a possible v1.x extension, not the v1 baseline. Could be useful to
  quantify how much the leakage inflates apparent skill (a legitimate, documented
  experiment) — but only as a clearly-labeled secondary analysis, never as the headline
  result compared to RTE.
- **Fixed arbitrary cutoff (e.g., 12:00 local, day-ahead market closure convention)**:
  rejected as the primary approach because it risks being incompatible with actual data
  availability (either RTE's forecast or the weather forecast might not exist yet at that
  hour), which would silently reintroduce leakage or waste information. Empirical
  determination first; a market-convention cutoff can be revisited once we know the real
  constraints.

## Consequences

- v1 has roughly 2.5 years of backtestable history (growing daily) rather than 12+.
  Rare events (severe cold snaps, heatwaves) are more sparsely represented — to be called
  out explicitly in error analysis, not hidden.
- The pipeline must ingest and reconcile two RTE datasets (`cons-def` + `tr`) and two
  Open-Meteo endpoints (Previous Runs for the modeling window, Historical Forecast/Archive
  only for exploratory or secondary analysis).
- The RTE resolution question is now resolved (see the update above). The exact
  forecast cutoff hour is still open — that determination belongs to feature
  engineering (using the now-real `dataset.parquet`), not to ingestion itself; tracked
  in `PROJECT_NOTES.md`'s "Open decisions."
