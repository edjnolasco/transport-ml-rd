from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd


def normalize_province(text: str) -> str:
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = " ".join(text.split())
    return text


def find_dataset_path() -> Path:
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

    raise FileNotFoundError(
        "No se encontró un dataset CSV en rutas esperadas. "
        "Ajusta find_dataset_path() al archivo real de tu proyecto."
    )


def test_provincia_has_32_territorial_units() -> None:
    dataset_path = find_dataset_path()
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
