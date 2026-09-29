# ADR 0008: Scheduling the daily forecast with Airflow

## Context

ADR 0007 built `ingest daily-forecast`, a single command that produces a real forecast
for tomorrow, verified against the live RTE and Open-Meteo APIs. It still has to be run
by hand. `PROJECT_NOTES.md`'s v2 scope names "daily automated run" as the first item, and
explicitly left the orchestrator choice open ("Airflow vs lighter options — to be
justified when v2 starts").

### Where this runs matters more than what schedules it

Before choosing an orchestrator, the more consequential question is *where* the job
runs. This project's own experience already surfaced the risk directly: several backtest
runs in this project stretched from an expected ~1h40-3h to as long as 11h because the
machine went to sleep mid-run. A job scheduled to fire on a laptop that may be asleep or
off at the scheduled time will simply not run that day, regardless of which tool
schedules it.

Two real options exist:
- **Run locally** (on this machine): the historical raw data already lives in
  `data/raw/`, so `refresh()` is cheap (a few days' incremental fetch, not a full
  ~20-month backfill). Whatever schedules the job — `launchd`, `cron`, or Airflow's own
  scheduler — the job simply won't fire if the machine is asleep or off at the scheduled
  time. This limitation is identical across every local scheduling option; it is a
  property of running locally, not of the orchestrator chosen.
- **Run in the cloud** (e.g. a scheduled GitHub Actions workflow): solves the
  reliability problem for real, but a fresh cloud runner starts with none of the ~20
  months of historical raw data this project's model needs to train on. That would
  require either a full re-backfill every run (slow, and hits the public APIs harder
  than necessary) or persisting data across runs (a cache or external storage) — a real
  additional piece of work, not a configuration toggle.

**Decision: run locally, on this machine, for now.** The reliability gap is real and is
being documented as a known limitation (see Consequences), not hidden. Solving it
properly (moving to the cloud with real data persistence) is a legitimate future
iteration, not something to bolt on as an afterthought to this phase.

### Airflow, not a lighter local scheduler

Given "run locally," the simplest tool would be `launchd` (macOS's native scheduler,
the direct equivalent of cron). It was considered and rejected: it has no transferable,
professional value (`launchd` is specific to this laptop, not something used in a
production data engineering context), whereas Airflow is both what this project's own
`PROJECT_NOTES.md` already named as the intended v2 orchestrator and directly relevant to
the Data Engineering roles this portfolio targets. Since Airflow run locally has the
*same* reliability characteristics as `launchd` run locally (both are just as unable to
fire while the machine sleeps — the limitation comes from running locally at all, not
from the choice between them), there is no reliability cost to choosing Airflow over
`launchd` — only a real setup cost (Docker, a multi-container stack, a DAG to write) in
exchange for a genuinely more representative artifact.

Rejected: `launchd`/cron (no professional signal, despite being simpler to set up);
Airflow in the cloud, e.g. via a managed service (solves reliability but is a much larger
infrastructure and cost commitment than justified at this stage — a candidate for a much
later iteration, not this one).

## Decision

### Executor: `LocalExecutor`, not `CeleryExecutor`

A single DAG running on a single machine has no need for `CeleryExecutor`'s distributed
worker pool (which would add Redis/RabbitMQ and multiple worker containers). Apache
Airflow's own quick-start Docker Compose template, using `LocalExecutor`, is the
proportionate choice: one scheduler, one webserver, one Postgres metadata database.

### Execution: `BashOperator` calling the existing CLI, not `DockerOperator`

Each Airflow task shells out to `uv run ingest <subcommand>` via `BashOperator`, inside
a custom image that extends the official Airflow image with `uv` installed, with this
project's directory mounted as a volume so `data/` persists across runs and stays
visible on the host filesystem.

Rejected: `DockerOperator` (spinning up a fresh, isolated container per task from a
dedicated project image) — architecturally cleaner separation between Airflow's own
runtime and this project's runtime, and closer to how a real production setup would
isolate task execution environments, but requires mounting the Docker socket into the
Airflow container (a real security/complexity cost) for no benefit at this project's
scale (one DAG, one project environment, no competing dependency versions between
tasks). Revisit if this pipeline ever needs to run alongside DAGs from unrelated
projects with conflicting dependencies.

### Three tasks, not one, with a small CLI refactor to support it

`ingest daily-forecast` currently does everything in one function: refresh, rebuild the
dataset, predict. Wrapping that single command as one opaque Airflow task would work, but
misses the actual point of using Airflow: per-step visibility (which stage failed) and
per-step retries (a transient RTE API hiccup during `refresh` shouldn't force retrying
the whole pipeline, including a full dataset rebuild and model retrain).

`src/felec/cli.py` gains a new `ingest predict` subcommand — the predict-and-save half of
`daily_forecast()`, extracted into its own function. `daily_forecast()` itself is
refactored to call `refresh()`, `build_dataset()`, and this new `predict()` in sequence,
preserving its existing behavior and CLI contract exactly (still one command, still does
the same thing) — it remains available for quick manual runs outside Airflow. The DAG
then defines three tasks — `refresh` → `build-dataset` → `predict`, each calling the
corresponding CLI subcommand directly — with `>>` dependencies matching that order.

### Schedule: daily at 16:00 Europe/Paris, 1 retry

16:00 gives a comfortable margin past both RTE's `prevision_j1` publication and
tomorrow's weather forecast becoming available (both already confirmed live during ADR
0007's design). The exact wall-clock run time doesn't affect leakage safety — every
model still trains strictly on data before the D-1 noon cutoff regardless of what time
the command actually executes (see ADR 0004) — so 16:00 is chosen purely for API-freshness
margin, not correctness.

One retry (`retries=1`, a short fixed delay) per task, not `catchup=True` or a longer
retry backoff schedule: catching up on a day the machine was asleep for would require the
model to retroactively construct a forecast for a target date that's already partially or
fully in the past — a real change in semantics (which `predict_next_day` doesn't support
and shouldn't be forced to), not just an operational nicety. A missed day is a missed
day, logged and visible in the Airflow UI, not silently skipped and not retroactively
invented.

## Alternatives considered

Covered inline per decision above (running in the cloud, `launchd`, `CeleryExecutor`,
`DockerOperator`, catch-up scheduling).

## Consequences

- **The reliability gap is real and known**: if this machine is asleep or off at 16:00
  Paris time, that day's forecast simply doesn't get produced — Airflow's scheduler
  itself must be running (inside a Docker container on this machine) for the DAG to fire
  at all. This is not a shortcoming of Airflow specifically; it is the direct consequence
  of the "run locally" decision above, made consciously rather than discovered later.
- Running the Airflow stack (scheduler, webserver, Postgres) means keeping Docker running
  continuously on this machine for the schedule to have any chance of firing — a real,
  ongoing resource cost, not a one-time setup cost.
- The three-task DAG design (and the `ingest predict` CLI split it required) is reusable
  beyond Airflow — `daily_forecast()`'s own behavior is unchanged, so anything already
  depending on it (manual runs, a future different orchestrator) is unaffected.
- Moving this to a reliably-available environment (cloud-hosted Airflow, or a scheduled
  GitHub Actions workflow with real data persistence) remains a legitimate future
  iteration, deliberately deferred rather than solved partially here.
