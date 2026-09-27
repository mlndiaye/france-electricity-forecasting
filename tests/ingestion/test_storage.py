import pandas as pd

from felec.ingestion.storage import (
    PROCESSED_DIR,
    load_processed,
    load_raw,
    save_processed,
    save_raw,
)


def test_save_raw_creates_new_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame({"date": ["2024-01-01"], "value": [1]})

    save_raw(df, "source/data.parquet", key_cols=["date"])

    result = load_raw("source/data.parquet")
    assert result.to_dict("records") == [{"date": "2024-01-01", "value": 1}]


def test_save_raw_deduplicates_on_key_cols(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_raw(pd.DataFrame({"date": ["2024-01-01"], "value": [1]}), "s.parquet", ["date"])

    save_raw(pd.DataFrame({"date": ["2024-01-01"], "value": [2]}), "s.parquet", ["date"])

    result = load_raw("s.parquet")
    assert len(result) == 1
    assert result.iloc[0]["value"] == 2


def test_save_raw_appends_new_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_raw(pd.DataFrame({"date": ["2024-01-01"], "value": [1]}), "s.parquet", ["date"])

    save_raw(pd.DataFrame({"date": ["2024-01-02"], "value": [2]}), "s.parquet", ["date"])

    result = load_raw("s.parquet")
    assert len(result) == 2


def test_load_raw_returns_empty_dataframe_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = load_raw("does/not/exist.parquet")
    assert result.empty


def test_save_processed_overwrites_existing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_processed(pd.DataFrame({"date": ["2024-01-01"], "value": [1]}), "out.parquet")

    save_processed(pd.DataFrame({"date": ["2024-02-01"], "value": [2]}), "out.parquet")

    result = pd.read_parquet(PROCESSED_DIR / "out.parquet")
    assert result.to_dict("records") == [{"date": "2024-02-01", "value": 2}]


def test_load_processed_reads_back_a_saved_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_processed(pd.DataFrame({"date": ["2024-01-01"], "value": [1]}), "out.parquet")

    result = load_processed("out.parquet")

    assert result.to_dict("records") == [{"date": "2024-01-01", "value": 1}]
