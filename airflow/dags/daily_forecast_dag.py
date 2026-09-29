"""Daily forecast pipeline: refresh real RTE/weather data, rebuild the
processed dataset, and predict tomorrow's consumption. Three separate tasks
(rather than one command wrapping felec's own daily_forecast()) so a
transient failure in one step doesn't force retrying the whole pipeline --
see ADR 0007 and ADR 0008.
"""

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

PROJECT_DIR = "/opt/project"

with DAG(
    dag_id="daily_forecast",
    description=(
        "Refresh RTE/weather/calendar data, rebuild the dataset, and predict "
        "tomorrow's consumption -- see docs/decisions/0007 and 0008."
    ),
    schedule="0 16 * * *",  # 16:00 Europe/Paris -- see ADR 0008 for why
    start_date=pendulum.datetime(2026, 9, 29, tz="Europe/Paris"),
    catchup=False,  # a missed day stays missed, not retroactively invented -- ADR 0008
    default_args={"retries": 1},
    tags=["felec"],
) as dag:
    refresh = BashOperator(
        task_id="refresh",
        bash_command=f"cd {PROJECT_DIR} && uv run ingest refresh",
    )
    build_dataset = BashOperator(
        task_id="build_dataset",
        bash_command=f"cd {PROJECT_DIR} && uv run ingest build-dataset",
    )
    predict = BashOperator(
        task_id="predict",
        bash_command=f"cd {PROJECT_DIR} && uv run ingest predict",
    )

    refresh >> build_dataset >> predict
