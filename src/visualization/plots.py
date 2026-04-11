# ============================================================
# FIGURA TIPO PAPER – VERSION FINAL ESTABLE
# ============================================================

from pathlib import Path
from datetime import datetime, UTC
import json
import os
import matplotlib.pyplot as plt
import pandas as pd


def save_scatter_plot_paper_v2(
    df,
    x_col,
    y_col,
    model_col,
    strategy_col,
    filename_base,
    title,
    xlabel,
    ylabel,
    output_dir="reports/figures",
    show_pareto=True,
    pareto_label_col=None,
    use_adjust_text=True,
    highlight_best=True,
    best_score_col=None,
    best_time_col=None,
    export_png=True,
    export_pdf=False,
    export_svg=False,
    export_metadata=False,
):
    # ============================================================
    # 🔴 LIMPIEZA GLOBAL DE FIGURAS (FIX CRÍTICO)
    # ============================================================
    plt.close("all")

    from matplotlib.lines import Line2D

    # ============================================================
    # FUNCIONES AUXILIARES
    # ============================================================
    def get_pareto_front(data):
        ordered = data.sort_values(by=[x_col, y_col], ascending=[True, False]).copy()
        pareto_rows = []
        best_y = float("-inf")

        for _, row in ordered.iterrows():
            if row[y_col] > best_y:
                pareto_rows.append(row)
                best_y = row[y_col]

        return pd.DataFrame(pareto_rows)

    def get_best_global(data):
        score_col = best_score_col or y_col
        time_col = best_time_col or x_col

        ordered = data.sort_values(
            by=[score_col, time_col],
            ascending=[False, True]
        ).copy()

        return ordered.iloc[0]

    # ============================================================
    # PREPARACIÓN DE DATOS
    # ============================================================
    df = df.dropna(subset=[x_col, y_col, model_col, strategy_col]).copy()

    if df.empty:
        raise ValueError("No hay datos válidos para construir la figura.")

    pareto_df = get_pareto_front(df) if show_pareto else pd.DataFrame()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_path = output_dir / filename_base

    png_path = str(base_path.with_suffix(".png")) if export_png else None
    pdf_path = str(base_path.with_suffix(".pdf")) if export_pdf else None
    svg_path = str(base_path.with_suffix(".svg")) if export_svg else None
    metadata_path = str(output_dir / f"{filename_base}_metadata.json") if export_metadata else None

    # ============================================================
    # CREACIÓN DE FIGURA
    # ============================================================
    fig, ax = plt.subplots(figsize=(11, 7))

    models = sorted(df[model_col].unique())
    strategies = sorted(df[strategy_col].unique())

    cmap = plt.get_cmap("tab10")
    color_map = {m: cmap(i % 10) for i, m in enumerate(models)}

    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
    marker_map = {s: markers[i % len(markers)] for i, s in enumerate(strategies)}

    # ============================================================
    # SCATTER PRINCIPAL
    # ============================================================
    for _, row in df.iterrows():
        ax.scatter(
            row[x_col],
            row[y_col],
            color=color_map[row[model_col]],
            marker=marker_map[row[strategy_col]],
            edgecolors="black",
            linewidths=0.4,
            s=80,
            alpha=0.9,
            zorder=2,
        )

    # ============================================================
    # PARETO FRONT
    # ============================================================
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

    # ============================================================
    # ETIQUETAS
    # ============================================================
    texts = []

    if pareto_label_col and not pareto_df.empty:
        for _, row in pareto_df.iterrows():
            t = ax.text(
                row[x_col],
                row[y_col],
                str(row[pareto_label_col]),
                fontsize=8,
                zorder=5,
            )
            texts.append(t)

    # ============================================================
    # MEJOR MODELO GLOBAL
    # ============================================================
    best_row = None

    if highlight_best:
        best_row = get_best_global(df)

        label = f"{best_row[model_col]} | {best_row[strategy_col]}"

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

        t = ax.text(
            best_row[x_col],
            best_row[y_col],
            f"Best: {label}",
            fontsize=9,
            fontweight="bold",
            zorder=7,
        )
        texts.append(t)

    # ============================================================
    # AJUSTE AUTOMÁTICO DE TEXTO
    # ============================================================
    if use_adjust_text and texts:
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.5))
        except Exception:
            pass

    # ============================================================
    # ESTILO
    # ============================================================
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    # ============================================================
    # LEYENDAS
    # ============================================================
    model_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color=color_map[m],
            linestyle="",
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=8,
            label=m,
        )
        for m in models
    ]

    strat_handles = [
        Line2D(
            [0], [0],
            marker=marker_map[s],
            color="gray",
            linestyle="",
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=8,
            label=s,
        )
        for s in strategies
    ]

    l1 = ax.legend(handles=model_handles, title="Modelo", loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.add_artist(l1)

    l2 = ax.legend(handles=strat_handles, title="Estrategia", loc="upper left", bbox_to_anchor=(1.02, 0.5))
    ax.add_artist(l2)

    if highlight_best:
        best_handle = Line2D([0], [0], marker="*", color="black", linestyle="", markersize=10, label="Mejor modelo")
        l3 = ax.legend(handles=[best_handle], loc="upper left", bbox_to_anchor=(1.02, 0.2))
        ax.add_artist(l3)

    plt.tight_layout()

    # ============================================================
    # EXPORTACIÓN
    # ============================================================
    if export_png:
        plt.savefig(png_path, dpi=300, bbox_inches="tight")

    if export_pdf:
        plt.savefig(pdf_path, bbox_inches="tight")

    if export_svg:
        plt.savefig(svg_path, bbox_inches="tight")

    plt.close()  # 🔥 CRÍTICO PARA NOTEBOOKS

    # ============================================================
    # METADATA
    # ============================================================
    metadata = None

    if export_metadata:
        metadata = {
            "filename": filename_base,
            "generated_at": datetime.now(UTC).isoformat(),
            "best_model": best_row.to_dict() if best_row is not None else None,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return {
        "png": png_path,
        "pdf": pdf_path,
        "svg": svg_path,
        "metadata": metadata_path,
        "best_model": best_row.to_dict() if best_row is not None else None,
    }
