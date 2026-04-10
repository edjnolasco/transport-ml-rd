from __future__ import annotations

import pandas as pd


def load_dataset(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [col.strip().lower() for col in cleaned.columns]
    return cleaned.drop_duplicates()


def save_processed_dataset(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index=False)
