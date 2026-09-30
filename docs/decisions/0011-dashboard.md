# ADR 0011: Streamlit dashboard consuming the serving API

## Context

ADR 0010 built a read-only FastAPI service exposing the pipeline's results. There was
still no visual way to see any of it beyond raw JSON or the API's auto-generated
`/docs`. `PROJECT_NOTES.md`'s v2 scope names "simple dashboard (forecast vs RTE vs
actual)" as the piece that consumes this API.

## Decision

### Streamlit, not a custom HTML/JS frontend or server-rendered templates

100% Python, fast to build, a well-recognized choice for data/ML portfolio
dashboards, and integrates natively with Plotly for interactive charts (zoom,
hover). A custom JS frontend was rejected as outside the Data/AI Engineering skill
set this portfolio is meant to demonstrate; server-rendered Jinja2 templates were
rejected as adding a templating layer for less interactivity than Streamlit gives for
free.

### Consumes the API over HTTP, not Parquet files directly

The dashboard calls `GET /forecast/latest` and `GET /forecast/history` exactly like
any other API client, via `requests` (already a project dependency, same pattern
already used by the ingestion connectors). Reading Parquet directly would bypass
ADR 0010's API entirely and duplicate the `quantile_backtest_results.parquet` +
`dataset.parquet` join it already implements.

### No "model MAE" metric -- only interval coverage and RTE MAE

`GET /forecast/history` has no `q50_pred` — `quantile_backtest()` is called with its
default `quantiles=(0.1, 0.9)` (see ADR 0010) — so there is no honest point-forecast
to compute a model MAE from. Rather than approximate one from the `(q10+q90)/2`
midpoint (implying a precision the data doesn't have), the dashboard shows only what
is genuinely computable: the 80% interval's actual coverage over the selected range,
and RTE's MAE (`rte_pred` vs. `consommation`, both directly available) as a reference
point.

### `data.py`/`charts.py`/`app.py` split

HTTP calls and pure computations (`data.py`) and chart construction (`charts.py`) are
unit-tested plain functions with no Streamlit dependency. `app.py` itself — the actual
Streamlit script — is not unit-tested; it only wires those functions to `st.*` calls
and is verified manually in a real browser instead (the same reasoning already applied
to notebooks in this project: rendering/presentation code isn't where the logic that
matters lives).

## Alternatives considered

- **Custom HTML/JS frontend**: rejected — outside this portfolio's target skill set.
- **FastAPI + Jinja2 server-rendered templates**: rejected — a templating layer for
  less interactivity than Streamlit + Plotly gives natively.
- **Reading Parquet directly instead of the API**: rejected — bypasses ADR 0010 and
  duplicates its join logic.
- **A `(q10+q90)/2`-based "model MAE"**: rejected — would misrepresent the model's
  actual precision.

## Consequences

- `src/felec/dashboard/` is a new subpackage; `streamlit` and `plotly` are now direct
  (non-dev) dependencies.
- Running the dashboard requires the serving API running separately
  (`uv run fastapi dev src/felec/api/app.py`) — two local processes, not one. The
  dashboard's own error handling (`ApiUnavailableError`, `ForecastNotReadyError`)
  exists specifically because this is a real, expected failure mode, not an edge case.
- Same deployment status as the API itself (ADR 0010): local only for now, hosting
  deliberately deferred to a separate future decision.
