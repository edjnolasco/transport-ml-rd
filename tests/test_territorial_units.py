from __future__ import annotations

import unicodedata

import pandas as pd


def normalize_province(text: str) -> str:
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = " ".join(text.split())
    return text


def test_provincia_normalization_collapses_to_32_units() -> None:
    raw_values = [
        "AZUA", "Azua",
        "BAHORUCO", "Bahoruco",
        "BARAHONA", "Barahona",
        "DAJABON", "Dajabón",
        "DISTRITO NACIONAL", "Distrito Nacional",
        "DUARTE", "Duarte",
        "EL SEIBO", "El Seibo",
        "ELIAS PIÑA", "Elías Piña",
        "ESPAILLAT", "Espaillat",
        "HATO MAYOR", "Hato Mayor",
        "HERMANAS MIRABAL", "Hermanas Mirabal",
        "INDEPENDENCIA", "Independencia",
        "LA ALTAGRACIA", "La Altagracia",
        "LA ROMANA", "La Romana",
        "LA VEGA", "La Vega",
        "MARIA TRINIDAD SANCHEZ", "María Trinidad Sánchez",
        "MONSEÑOR NOUEL",
        "MONTE PLATA",
        "MONTECRISTI",
        "PEDERNALES",
        "PERAVIA",
        "PUERTO PLATA",
        "SAMANA", "Samaná",
        "SAN CRISTOBAL", "San Cristóbal",
        "SAN JOSE DE OCOA", "San José de Ocoa",
        "SAN JUAN",
        "SAN PEDRO DE MACORIS", "San Pedro de Macorís",
        "SANCHEZ RAMIREZ", "Sánchez Ramírez",
        "SANTIAGO",
        "SANTIAGO RODRIGUEZ", "Santiago Rodríguez",
        "SANTO DOMINGO",
        "VALVERDE",
    ]

    df = pd.DataFrame({"provincia": raw_values})
    provincia_clean = df["provincia"].map(normalize_province)

    assert provincia_clean.nunique() == 32
