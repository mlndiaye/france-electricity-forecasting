# ADR 0009: MLflow model tracking for the daily forecast

## Context

ADR 0007/0008 built `ingest predict`, run automatically once a day by Airflow. It
trains three fresh LightGBM quantile models every run, with no record of what
hyperparameters were used, how much training data went in, or what the trained models
actually were. There is no way today to answer "which model produced this specific
forecast?" after the fact. `PROJECT_NOTES.md`'s v2 scope names a "model registry
(MLflow)" as the next slice.

## Decision

### Run-level tracking, not a full Model Registry

MLflow offers two levels: run tracking (params/metrics/artifacts per execution) and
the Model Registry proper (named, versioned models with stages like
"staging"/"production"). The actual need here is an audit trail — being able to trace
a specific forecast back to the model that produced it — not promoting a fixed model
to serve instead of retraining daily. Run-level tracking answers that need directly;
the Registry's naming/versioning/staging machinery would be unused complexity. Revisit
if a future serving API needs to load one specific "production" model rather than
always retraining (see PROJECT_NOTES.md's still-ahead FastAPI/dashboard slice).

### Local file-based tracking, not a dockerized MLflow server

Consistent with ADR 0008's "run locally" decision: a single writer (the daily
`predict` task) and a single reader (this developer, via `mlflow ui` on demand) don't
need an always-on tracking server with its own backend store and artifact store.
`mlflow.set_tracking_uri("file:...")` pointing at a local `mlruns/` directory gives
the same audit trail without a fifth Docker service to maintain.

### Logged from `cli.py`, not from `forecast.py`

`predict_next_day()` stays free of any MLflow dependency — it is pure modeling logic,
unit-tested without side effects, exactly as before. `cli.py`'s `predict()` (the only
production caller) does the actual logging, wrapping the call in
`with mlflow.start_run():` and reading everything it needs (hyperparameters, cutoff
hour, training set size, the trained models) off a new `ForecastResult` dataclass that
`predict_next_day()` now returns instead of a bare `pd.DataFrame`.

`ForecastResult` carries the *actual* `cutoff_hour` and `params` used (including
`forecast.py`'s own internal defaults when the caller didn't override them), not
values `cli.py` would otherwise have to hardcode or duplicate — the same reasoning
that already applied to `params` extends naturally to `cutoff_hour`.

### Scope: only the daily production run, not backtests

Only `predict_next_day()` (via `ingest predict`) logs to MLflow. `backtest` and
`quantile-backtest` are development/evaluation tools whose results already get saved
to `data/processed/*.parquet` and analyzed in notebooks (ADR 0004-0006) — adding
MLflow tracking there is a real but separate need (comparing backtest metrics across
runs over time), deferred until there's a concrete reason to do it.

## Alternatives considered

- **Full MLflow Model Registry** (named/versioned models, stages): rejected as more
  than the stated audit-trail goal requires — see above.
- **Dockerized MLflow tracking server**: rejected as unnecessary infrastructure for a
  single-writer, single-reader local setup — see above.
- **MLflow logging inside `forecast.py`**: rejected — would couple the pure modeling
  module to an infrastructure concern, and every existing unit test calling
  `predict_next_day()` would silently create a real MLflow run unless each test
  explicitly disabled it.
- **Also logging backtest runs**: deferred, not rejected — a natural follow-up once
  there's a concrete need to compare backtest metrics across runs over time.

## Consequences

- `predict_next_day()`'s return type changed from a bare `pd.DataFrame` to a
  `ForecastResult` dataclass — a breaking change to its only production caller
  (`cli.py`'s `predict()`, updated as part of this change) and to the 7 existing unit
  tests exercising it directly (updated to read `result.predictions` instead of
  `result`).
- `mlflow` is now a direct (non-dev) dependency, since `ingest predict` needs it at
  runtime — including inside the Airflow container, where it installs via the same
  `uv run` mechanism as every other dependency (see ADR 0008's `uv`-inside-Docker
  note).
- `mlruns/` lives only on the machine that runs `ingest predict` — in practice, on the
  host filesystem under this project's Airflow volume mount. There is no shared or
  centralized tracking server; if this machine's disk is lost, the tracking history is
  lost with it. This is the same class of limitation already documented for local
  scheduling in ADR 0008, applied consistently rather than solved differently here.
