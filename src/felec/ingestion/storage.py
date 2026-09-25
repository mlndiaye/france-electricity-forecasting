"""Read/write helpers for the local Parquet data lake."""
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def save_raw(df: pd.DataFrame, relative_path: str, key_cols: list[str]) -> None:
    """Write df to data/raw/<relative_path>, merging with any existing file.

    Deduplicates on key_cols, keeping the newest value for each key (rows in df
    win over any existing row with the same key). Idempotent: calling this
    twice with the same df does not create duplicate rows.
    """
    path = RAW_DIR / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    combined = combined.sort_values(key_cols).reset_index(drop=True)
    combined.to_parquet(path, index=False)


def load_raw(relative_path: str) -> pd.DataFrame:
    """Read data/raw/<relative_path>. Returns an empty DataFrame if it doesn't exist yet."""
    path = RAW_DIR / relative_path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def save_processed(df: pd.DataFrame, filename: str) -> None:
    """Write df to data/processed/<filename>, overwriting any existing file."""
    path = PROCESSED_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
