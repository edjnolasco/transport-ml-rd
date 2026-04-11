"""
Funciones de visualización para análisis de modelos y Green AI.

Incluye generación de figuras tipo paper para analizar el trade-off
entre desempeño predictivo y costo computacional, con soporte para:

- color por modelo
- marcador por estrategia
- frontera de Pareto
- detección del mejor modelo global
- exportación opcional a PNG, PDF y SVG
- metadata JSON
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def _get_pareto_front(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    """
    Calcula la frontera de Pareto para un escenario donde:
    - x se minimiza
    - y se maximiza

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame con los puntos a evaluar.
    x_col : str
        Columna a minimizar.
    y_col : str
        Columna a maximizar.

    Returns
    -------
    pd.DataFrame
        Subconjunto del DataFrame correspondiente a la frontera de Pareto.
    """
    ordered = data.sort_values(by=[x_col, y_col], ascending=[True, False]).copy()

    pareto_rows: list[pd.Series] = []
    best_y = float("-inf")

    for _, row in ordered.iterrows():
        current_y = row[y_col]
        if current_y > best_y:
            pareto_rows.append(row)
            best_y = current_y

    if not pareto_rows:
        return pd.DataFrame(columns=data.columns)

    return pd.DataFrame(pareto_rows)


def _get_best_global(
    data: pd.DataFrame,
    score_col: str,
    time_col: str,
) -> pd.Series:
    """
    Selecciona el mejor modelo global bajo el criterio:
    - mayor score
    - en caso de empate, menor tiempo

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame con resultados.
    score_col : str
        Columna principal de desempeño.
    time_col : str
        Columna de costo computacional.

    Returns
    -------
    pd.Series
        Fila correspondiente al mejor modelo global.
    """
    ordered = data.sort_values(
        by=[score_col, time_col],
        ascending=[False, True],
    ).copy()

    if ordered.empty:
        raise ValueError("No hay datos disponibles para seleccionar el mejor modelo.")

    return ordered.iloc[0]


def save_scatter_plot_paper_v2(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    model_col: str,
    strategy_col: str,
    filename_base: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_dir: str | Path = "reports/figures",
    show_pareto: bool = True,
    pareto_label_col: str | None = None,
    use_adjust_text: bool = True,
    highlight_best: bool = True,
    best_score_col: str | None = None,
    best_time_col: str | None = None,
    export_png: bool = True,
    export_pdf: bool = False,
    export_svg: bool = False,
    export_metadata: bool = False,
) -> dict[str, str | dict[str, Any] | None]:
    """
    Genera una figura tipo paper para visualizar trade-offs entre
    desempeño y tiempo de entrenamiento.

    La figura incluye:
    - puntos coloreados por modelo
    - marcadores por estrategia
    - frontera de Pareto
    - resaltado del mejor modelo global
    - exportación opcional a múltiples formatos

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con resultados de experimentos.
    x_col : str
        Columna para el eje X.
    y_col : str
        Columna para el eje Y.
    model_col : str
        Columna con el nombre del modelo.
    strategy_col : str
        Columna con el nombre de la estrategia.
    filename_base : str
        Nombre base de los archivos a exportar, sin extensión.
    title : str
        Título de la figura.
    xlabel : str
        Etiqueta del eje X.
    ylabel : str
        Etiqueta del eje Y.
    output_dir : str | Path, optional
        Directorio de salida.
    show_pareto : bool, optional
        Si True, dibuja la frontera de Pareto.
    pareto_label_col : str | None, optional
        Columna de etiquetas para los puntos de Pareto.
    use_adjust_text : bool, optional
        Si True, intenta ajustar automáticamente las etiquetas.
    highlight_best : bool, optional
        Si True, resalta el mejor modelo global.
    best_score_col : str | None, optional
        Columna para seleccionar el mejor modelo. Si es None, usa y_col.
    best_time_col : str | None, optional
        Columna de tiempo para desempate. Si es None, usa x_col.
    export_png : bool, optional
        Si True, exporta PNG.
    export_pdf : bool, optional
        Si True, exporta PDF.
    export_svg : bool, optional
        Si True, exporta SVG.
    export_metadata : bool, optional
        Si True, exporta metadata JSON.

    Returns
    -------
    dict[str, str | dict[str, Any] | None]
        Diccionario con rutas de exportación y datos del mejor modelo.
    """
    # Limpieza global para evitar overlays y artefactos visuales en notebooks.
    plt.close("all")

    required_cols = [x_col, y_col, model_col, strategy_col]
    if pareto_label_col is not None:
        required_cols.append(pareto_label_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Faltan columnas requeridas en el DataFrame: {missing_cols}"
        )

    plot_df = df.dropna(subset=[x_col, y_col, model_col, strategy_col]).copy()

    if plot_df.empty:
        raise ValueError("No hay datos válidos para construir la figura.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_path = output_path / filename_base

    png_path = str(base_path.with_suffix(".png")) if export_png else None
    pdf_path = str(base_path.with_suffix(".pdf")) if export_pdf else None
    svg_path = str(base_path.with_suffix(".svg")) if export_svg else None
    metadata_path = (
        str(output_path / f"{filename_base}_metadata.json")
        if export_metadata
        else None
    )

    pareto_df = (
        _get_pareto_front(plot_df, x_col=x_col, y_col=y_col)
        if show_pareto
        else pd.DataFrame(columns=plot_df.columns)
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    models = sorted(plot_df[model_col].dropna().unique())
    strategies = sorted(plot_df[strategy_col].dropna().unique())

    cmap = plt.get_cmap("tab10")
    color_map = {model: cmap(i % 10) for i, model in enumerate(models)}

    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
    marker_map = {
        strategy: markers[i % len(markers)]
        for i, strategy in enumerate(strategies)
    }

    # Scatter principal.
    for _, row in plot_df.iterrows():
        ax.scatter(
            row[x_col],
            row[y_col],
            color=color_map[row[model_col]],
            marker=marker_map[row[strategy_col]],
            edgecolors="black",
            linewidths=0.4,
            s=80,
            alpha=0.90,
            zorder=2,
        )

    # Frontera de Pareto.
    if show_pareto and not pareto_df.empty:
        ax.plot(
            pareto_df[x_col],
            pareto_df[y_col],
            "--",
            color="black",
            linewidth=1.5,
            zorder=3,
        )

        for _, row in pareto_df.iterrows():
            ax.scatter(
                row[x_col],
                row[y_col],
                facecolors="none",
                edgecolors="black",
                marker=marker_map[row[strategy_col]],
                linewidths=1.2,
                s=130,
                zorder=4,
            )

    # Etiquetas.
    texts: list[Any] = []

    if pareto_label_col is not None and not pareto_df.empty:
        for _, row in pareto_df.iterrows():
            label_artist = ax.text(
                row[x_col],
                row[y_col],
                str(row[pareto_label_col]),
                fontsize=8,
                zorder=5,
            )
            texts.append(label_artist)

    # Mejor modelo global.
    best_row: pd.Series | None = None

    if highlight_best:
        score_col = best_score_col or y_col
        time_col = best_time_col or x_col
        best_row = _get_best_global(plot_df, score_col=score_col, time_col=time_col)

        best_label = f"{best_row[model_col]} | {best_row[strategy_col]}"

        ax.scatter(
            best_row[x_col],
            best_row[y_col],
            s=220,
            marker="*",
            edgecolors="black",
            facecolors="none",
            linewidths=2,
            zorder=6,
        )

        best_artist = ax.text(
            best_row[x_col],
            best_row[y_col],
            f"Best: {best_label}",
            fontsize=9,
            fontweight="bold",
            zorder=7,
        )
        texts.append(best_artist)

    # Ajuste automático de etiquetas.
    if use_adjust_text and texts:
        try:
            from adjustText import adjust_text

            adjust_text(
                texts,
                ax=ax,
                arrowprops={"arrowstyle": "-", "lw": 0.5},
            )
        except Exception:
            pass

    # Estilo general.
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    # Leyendas.
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color_map[model],
            linestyle="",
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=8,
            label=model,
        )
        for model in models
    ]

    strategy_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map[strategy],
            color="gray",
            linestyle="",
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=8,
            label=strategy,
        )
        for strategy in strategies
    ]

    legend_models = ax.legend(
        handles=model_handles,
        title="Modelo",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
    )
    ax.add_artist(legend_models)

    legend_strategies = ax.legend(
        handles=strategy_handles,
        title="Estrategia",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.50),
    )
    ax.add_artist(legend_strategies)

    if highlight_best:
        best_handle = Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            linestyle="",
            markersize=10,
            label="Mejor modelo",
        )
        legend_best = ax.legend(
            handles=[best_handle],
            loc="upper left",
            bbox_to_anchor=(1.02, 0.20),
        )
        ax.add_artist(legend_best)

    plt.tight_layout()

    # Exportación.
    if export_png and png_path is not None:
        plt.savefig(png_path, dpi=300, bbox_inches="tight")

    if export_pdf and pdf_path is not None:
        plt.savefig(pdf_path, bbox_inches="tight")

    if export_svg and svg_path is not None:
        plt.savefig(svg_path, bbox_inches="tight")

    plt.close()

    # Metadata.
    if export_metadata and metadata_path is not None:
        metadata = {
            "filename": filename_base,
            "generated_at": datetime.now(UTC).isoformat(),
            "x_col": x_col,
            "y_col": y_col,
            "model_col": model_col,
            "strategy_col": strategy_col,
            "show_pareto": show_pareto,
            "highlight_best": highlight_best,
            "best_model": best_row.to_dict() if best_row is not None else None,
        }

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

    return {
        "png": png_path,
        "pdf": pdf_path,
        "svg": svg_path,
        "metadata": metadata_path,
        "best_model": best_row.to_dict() if best_row is not None else None,
    }
