from __future__ import annotations

import pandas as pd


def split_features_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no existe en el dataset.")

    x = df.drop(columns=[target_column])
    y = df[target_column]
    return x, y


def detect_feature_types(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x.select_dtypes(exclude=["number"]).columns.tolist()
    return numeric_features, categorical_features
