# ADR 0003: Exploratory analysis notebook design

## Context

The ingestion pipeline (ADR 0002) is built, tested, and verified against live APIs —
`data/processed/dataset.parquet` now exists for real (23,252 hourly rows,
2024-02-01 → present). Per `PROJECT_NOTES.md`'s v1 scope, "Exploratory analysis" is the next
step, before baselines and modeling. This ADR covers what that notebook is for, what
tool it uses, and what it actually looks at.

## Decision

### Role: personal exploration first, not a portfolio showpiece

The notebook's primary job is to help understand the data before modeling —
distributions, seasonality, data quality — not to be a polished, narrated artifact from
the first draft. It gets cleaned up and documented afterward if it earns a place in the
portfolio, but that's not the constraint driving how it's written now. This avoids
over-polishing exploratory work that may well change once real patterns are seen.

### Tool: Jupyter (`.ipynb`)

Chosen over marimo: Jupyter is the universal standard, and GitHub renders `.ipynb`
files with their output directly — useful for later portfolio visibility. marimo's
reactive, pure-`.py` design is more modern and would be a legitimate differentiator to
mention in an interview, but the "personal exploration first" framing above means the
diff-cleanliness/reproducibility argument for marimo matters less right now than
sticking to the tool everyone already expects.

### Scope: the processed dataset only

Explores `date_heure`, `consommation`, `prevision_j1`, `prevision_j`,
`temperature_nationale`, `est_ferie`, `vacances_zone_a/b/c` — the columns that actually
feed the eventual model. Explicitly excludes the raw production-mix columns ingested
alongside RTE's consumption data (nuclear, wind, solar, CO2 intensity, cross-border
exchanges) — those describe the electricity system in general, not the demand-forecasting
problem this project is scoped to. Revisit only if a specific modeling need arises.

### Content plan

1. **Overview and data quality**: date range, null rates per column, duplicate check —
   formalizes what was already checked ad hoc during initial data review, with real plots.
2. **Consumption distribution and seasonality**: by hour of day, day of week,
   month/season — the target variable's own behavior before bringing in predictors.
3. **Temperature vs. consumption**: scatter plot of the main driver; check whether the
   relationship differs on weekdays vs. weekends.
4. **Public holiday and school vacation effect**: compare average consumption
   (same-hour) on holiday/vacation vs. normal days — verify these costly-to-fetch
   calendar features actually carry explanatory power.
5. **RTE's own forecast error**: MAE/MAPE of `prevision_j1` vs. `consommation` — the
   benchmark this project is ultimately compared against, useful to know now rather
   than only at evaluation time.
6. **Autocorrelation**: does consumption 24h/48h/1 week ago help predict now? Informs
   which lag features to build during feature engineering.

## Alternatives considered

- **Portfolio-artifact-first framing**: rejected for now — narrating and polishing
  every cell before knowing what's actually interesting in the data is premature effort
  that may need redoing once real patterns are found.
- **marimo**: legitimate, not chosen — see Decision above. Worth reconsidering for a
  future, more production-facing notebook (e.g., a v2 monitoring dashboard notebook),
  where reactivity and clean diffs matter more.
- **Including the raw production-mix columns**: rejected — out of scope for a demand-
  forecasting project; would broaden this into general power-system analysis.

## Consequences

- The notebook lives at `notebooks/exploration.ipynb` (new top-level directory).
- Unlike `data/`, the notebook itself (code + output) is meant to be committed — it's
  part of the project's technical narrative, not raw/rebuildable data.
- Six content sections, each independently useful for either understanding the data or
  directly informing the next phase (feature engineering) — no section is decorative.
