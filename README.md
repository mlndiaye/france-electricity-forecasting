# France Electricity Forecasting

Day-ahead probabilistic forecasting of French national electricity consumption,
benchmarked daily against RTE's own official forecast and against actual consumption.

The goal is not to "beat RTE" — unlikely at national level, and not the point — but to
build a rigorous, honest forecasting system and understand *where and why* it errs. Every
result below, including the disappointing one, is reported as measured.

See [`PROJECT_NOTES.md`](PROJECT_NOTES.md) for the full running log of decisions and
results, and [`docs/decisions/`](docs/decisions/) for the reasoning behind each one (ADR
format: context, decision, alternatives considered, consequences).

## Results

Walk-forward backtest, 2025-10-16 → 2026-09-26 (347 days, retrained daily on an expanding
window, evaluated strictly on data unavailable at prediction time):

| | MAE (MW) |
|---|---|
| Seasonal naive (same hour, 1 week ago) | 3,654 |
| **This model** (LightGBM, gradient boosting) | **1,286** |
| RTE's own day-ahead forecast | 1,298 |

The model clears the naive baseline decisively and essentially matches RTE's own
forecast — without access to anything RTE has beyond public weather forecasts and
calendar data.

**Where it doesn't match RTE**: on cold days (national mean temperature ≤ 5°C), the model
is clearly worse (2,760 vs. 1,764 MW MAE) — a real limitation, most likely from the
backtest window containing only 23 genuinely cold days to learn from. See
[`notebooks/error_analysis.ipynb`](notebooks/error_analysis.ipynb) for the full
breakdown by weekday/weekend, public holidays, school holidays, cold days, and season.

**Prediction intervals**: an 80% interval (LightGBM quantile regression at the 0.1/0.9
levels) achieves only **55.1% real coverage** — meaningfully overconfident, and roughly
symmetric (misses high 25.0% of the time, low 19.8%). This is reported as a genuine,
unresolved limitation, not smoothed over — see
[`notebooks/probabilistic_evaluation.ipynb`](notebooks/probabilistic_evaluation.ipynb)
and [ADR 0006](docs/decisions/0006-probabilistic-forecasting.md). Conformal prediction,
initially rejected for added complexity, is the concretely motivated next step.

## How it works

```
ingest → assemble dataset → engineer features → backtest → evaluate
```

- **Ingestion** (`src/felec/ingestion/`): RTE éCO2mix consumption + forecasts (via ODRÉ),
  AROME day-ahead weather forecasts (via Open-Meteo's Previous Runs API — genuine
  forecasts, not observations), and French public holiday / school holiday calendars.
  Idempotent, deduplicated, resumable.
- **Processing** (`src/felec/processing/`): joins the three sources into a single hourly
  dataset, validated with `pandera` schemas.
- **Modeling** (`src/felec/modeling/`): feature engineering (safe lag features, a
  cutoff-anchored recent-trend feature, calendar features — all built to never leak
  information unavailable at the D-1 forecast cutoff), a seasonal-naive baseline, and a
  daily walk-forward backtest that retrains a fresh LightGBM model every day, exactly
  mirroring how the system would actually run in production.
- **Notebooks** (`notebooks/`): exploration, model selection (hyperparameter tuning),
  error analysis by day type, probabilistic evaluation — each a self-contained, executed
  record of a real analysis, not illustrative code.

### The central technical concern: data leakage

A forecast issued the day before must only use information genuinely available at that
moment. This shows up in several places, each documented in an ADR: observed vs. forecast
weather, a 168h (not 24h) lag feature chosen specifically because a naive 24h lag would
leak future information relative to a midday cutoff, and a DST-safe cutoff-time
computation (naive timestamp arithmetic silently mishandles the one ambiguous hour per
year). See [ADR 0001](docs/decisions/0001-problem-definition-scope-and-weather-window.md)
and [ADR 0004](docs/decisions/0004-feature-engineering-and-baselines.md).

### A real bug, found and fixed in the open

The first real run of the probabilistic backtest found that 0.49% of rows had an inverted
interval (`q10 > q90`) — "quantile crossing," a known failure mode of independently
trained quantile models. Fixed by sorting each row's predictions across quantile levels
(the standard technique), verified with a dedicated test, documented in
[ADR 0006](docs/decisions/0006-probabilistic-forecasting.md), and the real backtest was
re-run to confirm the fix. Kept in the project history rather than squashed away, because
finding and handling this kind of issue honestly is the actual point of doing this
rigorously.

## Getting started

```bash
uv sync --extra dev

# Backfill raw data (RTE, weather, calendar) from 2024-02-01 to today
uv run ingest backfill

# Assemble the processed hourly dataset
uv run ingest build-dataset

# Run the daily walk-forward point-forecast backtest (~1h40 for ~1 year)
uv run ingest backtest

# Run the probabilistic (quantile) backtest (~2x longer: 2 extra models/day)
uv run ingest quantile-backtest

# Day-to-day incremental refresh (for a running system)
uv run ingest refresh

# Produce tomorrow's real forecast: refresh + rebuild + retrain + predict,
# in one command. Hits the live RTE and Open-Meteo APIs (~20s).
uv run ingest daily-forecast
```

```bash
uv run pytest        # 69 tests
uv run ruff check .  # lint
uv run ruff format .
```

## Scheduling (Airflow, local)

The daily forecast pipeline (`refresh` → `build-dataset` → `predict`) can run
automatically once a day via a local Airflow instance — see
[ADR 0008](docs/decisions/0008-airflow-daily-scheduling.md) for why Airflow (not a
lighter local scheduler) and why local (not cloud) for now, including the honest
limitation this implies: if this machine is asleep or off at 16:00 Paris time, that
day's forecast is simply not produced.

```bash
cd airflow
cp .env.example .env
# Generate a real FERNET_KEY and paste it into .env:
python3 -c "import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"

docker compose up -d
# Wait for airflow-apiserver to report healthy, then:
docker compose exec airflow-scheduler airflow dags unpause daily_forecast
```

The Airflow UI is then at http://localhost:8080 (default login: `airflow` / `airflow` —
this is a local-only, non-production setup; do not reuse these credentials anywhere
real). The DAG runs daily at 16:00 Europe/Paris with 3 tasks (`refresh`, `build_dataset`,
`predict`) and 1 retry per task. Note: a DAG stays paused by default — a manually
triggered run on a paused DAG is created but never executes (it sits in `queued`
forever), so the `unpause` step above is required, not optional.

## Serving API

A read-only FastAPI service exposes the pipeline's results over HTTP — see
[ADR 0010](docs/decisions/0010-serving-api.md) for why it's read-only (it never
triggers a prediction itself; Airflow already does that daily) and where the
historical comparison data actually comes from (joined at request time, not
precomputed).

```bash
uv run fastapi dev src/felec/api/app.py
```

Interactive docs at http://localhost:8000/docs. Endpoints:
- `GET /health`
- `GET /forecast/latest` — tomorrow's forecast (404 if `ingest predict` hasn't run yet)
- `GET /forecast/history?start=&end=` — model vs. RTE vs. actual over the backtest
  window, optionally filtered by date (404 if `ingest quantile-backtest` hasn't run yet)

Local only for now — deployment (a public link) is a deliberately separate,
not-yet-made decision (see ADR 0010's Consequences).

## Dashboard

A Streamlit dashboard visualizes the serving API's results — see
[ADR 0011](docs/decisions/0011-dashboard.md) for why Streamlit, and why it shows
interval coverage and RTE's MAE but not a "model MAE" (the underlying data has no
median forecast to compute one honestly from).

Requires the serving API running separately (see the Serving API section above):

```bash
uv run streamlit run src/felec/dashboard/app.py
```

Opens at http://localhost:8501. Shows tomorrow's forecast (with its 80% interval) and
a historical model-vs-RTE-vs-actual comparison over a selectable date range, with
live-computed coverage and RTE MAE metrics.

## Tech stack

Python 3.12 · `uv` · pandas · LightGBM · scikit-learn · pandera (data validation) ·
pytest · ruff · Jupyter / matplotlib · Airflow (local scheduling) · MLflow (model
tracking) · FastAPI (serving) · Streamlit + Plotly (dashboard)

## Project status

**v1 (shipped)**: ingestion pipeline, exploratory analysis, feature engineering, seasonal
naive + RTE baselines, LightGBM point forecast, daily walk-forward backtest over a full
year, error analysis by day type, and a first probabilistic forecast (multi-quantile
LightGBM) — all built, tested, and verified against real data end-to-end.

**v2 (in progress)**: `ingest daily-forecast` chains real data refresh, dataset rebuild,
model retraining, and prediction into a single command, verified end-to-end against the
live RTE and Open-Meteo APIs (see
[ADR 0007](docs/decisions/0007-daily-forecast-pipeline.md)). This pipeline now also runs
automatically once a day via a local Airflow instance (see
[ADR 0008](docs/decisions/0008-airflow-daily-scheduling.md) and the Scheduling section
above). Every `ingest predict` run is also now logged to MLflow (hyperparameters,
cutoff hour, training set size, and the trained models themselves) as a local,
file-based audit trail (see
[ADR 0009](docs/decisions/0009-mlflow-model-tracking.md)). A read-only serving API
(see [ADR 0010](docs/decisions/0010-serving-api.md) and the Serving API section above)
exposes these results over HTTP, locally for now. A Streamlit dashboard (see
[ADR 0011](docs/decisions/0011-dashboard.md) and the Dashboard section above)
visualizes them: tomorrow's forecast and a historical model-vs-RTE-vs-actual
comparison with live coverage/MAE metrics. Still ahead: deploying the API and
dashboard publicly, and an extension estimating the probability of an RTE "Tempo" red
day from the probabilistic forecast.

## Engineering standards

`src/` layout, `uv`-managed environment, `ruff` for lint + format, `pytest` for the logic
that matters (data transformations, feature engineering, evaluation — not boilerplate),
conventional commits, no secrets in the repo, raw data never committed (pipelines rebuild
it), and every non-trivial technical decision recorded as an ADR before being built.
