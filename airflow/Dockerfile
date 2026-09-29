FROM apache/airflow:3.3.2

USER root
# libgomp1: LightGBM's compiled extension needs libgomp.so.1 (OpenMP) at import
# time -- not present in the base image, found by actually running `uv run
# ingest` in a built container, not assumed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/airflow/.local/bin:${PATH}"
