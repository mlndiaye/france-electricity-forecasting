"""Read/write helpers for the local Parquet data lake."""

import os
import tempfile
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write df to path via a temp file + atomic replace, so a crash mid-write
    never leaves a truncated/corrupt file at path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def save_raw(df: pd.DataFrame, relative_path: str, key_cols: list[str]) -> None:
    """Write df to data/raw/<relative_path>, merging with any existing file.

    Deduplicates on key_cols, keeping the newest value for each key (rows in df
    win over any existing row with the same key). Idempotent: calling this
    twice with the same df does not create duplicate rows.
    """
    path = RAW_DIR / relative_path

    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    combined = combined.sort_values(key_cols).reset_index(drop=True)
    _atomic_write_parquet(combined, path)


def load_raw(relative_path: str) -> pd.DataFrame:
    """Read data/raw/<relative_path>. Returns an empty DataFrame if it doesn't exist yet."""
    path = RAW_DIR / relative_path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def save_processed(df: pd.DataFrame, filename: str) -> None:
    """Write df to data/processed/<filename>, overwriting any existing file."""
    path = PROCESSED_DIR / filename
    _atomic_write_parquet(df, path)


def load_processed(filename: str) -> pd.DataFrame:
    """Read data/processed/<filename>."""
    return pd.read_parquet(PROCESSED_DIR / filename)
