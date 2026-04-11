from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd
import pytest


def normalize_province(text: str) -> str:
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = " ".join(text.split())
    return text


def find_dataset_path() -> Path | None:
    candidates = [
        Path("data/processed/transport_dataset.csv"),
        Path("data/processed/dataset_clean.csv"),
        Path("data/processed/df_clean.csv"),
        Path("data/raw/transport_dataset.csv"),
        Path("data/raw/dataset.csv"),
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def test_provincia_has_32_territorial_units() -> None:
    dataset_path = find_dataset_path()

    if dataset_path is None:
        pytest.skip(
            "No se encontró un dataset CSV en el repositorio. "
            "Se omite la validación del dataset real en CI."
        )

    df = pd.read_csv(dataset_path)

    assert "provincia" in df.columns, (
        "La columna 'provincia' no existe en el dataset. "
        "Verifica el nombre real de la columna territorial."
    )

    provincia_clean = df["provincia"].dropna().astype(str).map(normalize_province)
    n_units = provincia_clean.nunique()

    assert n_units == 32, (
        f"Se esperaban 32 unidades territoriales "
        f"(31 provincias + Distrito Nacional), pero se detectaron {n_units}."
    )
