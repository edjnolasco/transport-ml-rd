from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.visualization.plots import save_scatter_plot_paper_v2


def test_save_scatter_plot_paper_v2_exports_png(tmp_path: Path) -> None:
    """
    Verifica que la función genera correctamente un archivo PNG
    y devuelve la ruta esperada.
    """
    df = pd.DataFrame(
        {
            "model": [
                "LogisticRegression",
                "DecisionTreeClassifier",
                "RandomForestClassifier",
            ],
            "strategy": [
                "baseline",
                "tree_pruning",
                "select_k_best",
            ],
            "train_time_sec": [0.08, 0.03, 0.12],
            "f1_weighted": [0.91, 0.87, 0.92],
            "plot_label": [
                "LR | baseline",
                "DT | prune",
                "RF | kbest",
            ],
        }
    )

    result = save_scatter_plot_paper_v2(
        df=df,
        x_col="train_time_sec",
        y_col="f1_weighted",
        model_col="model",
        strategy_col="strategy",
        filename_base="test_tradeoff_figure",
        title="Trade-off entre F1-score y tiempo",
        xlabel="Tiempo (s)",
        ylabel="F1-score",
        output_dir=tmp_path,
        show_pareto=True,
        pareto_label_col="plot_label",
        use_adjust_text=False,
        highlight_best=True,
        export_png=True,
        export_pdf=False,
        export_svg=False,
        export_metadata=False,
    )

    png_path = result["png"]

    assert png_path is not None
    assert Path(png_path).exists()
    assert Path(png_path).suffix == ".png"


def test_save_scatter_plot_paper_v2_exports_multiple_formats(
    tmp_path: Path,
) -> None:
    """
    Verifica que la función puede exportar múltiples formatos
    y metadata JSON.
    """
    df = pd.DataFrame(
        {
            "model": [
                "LogisticRegression",
                "DecisionTreeClassifier",
                "RandomForestClassifier",
                "SVC",
            ],
            "strategy": [
                "baseline",
                "tree_pruning",
                "select_k_best",
                "regularization",
            ],
            "train_time_sec": [0.08, 0.03, 0.12, 0.10],
            "f1_weighted": [0.91, 0.87, 0.92, 0.90],
            "plot_label": [
                "LR | baseline",
                "DT | prune",
                "RF | kbest",
                "SVC | reg",
            ],
        }
    )

    result = save_scatter_plot_paper_v2(
        df=df,
        x_col="train_time_sec",
        y_col="f1_weighted",
        model_col="model",
        strategy_col="strategy",
        filename_base="test_tradeoff_multi",
        title="Trade-off entre F1-score y tiempo",
        xlabel="Tiempo (s)",
        ylabel="F1-score",
        output_dir=tmp_path,
        show_pareto=True,
        pareto_label_col="plot_label",
        use_adjust_text=False,
        highlight_best=True,
        export_png=True,
        export_pdf=True,
        export_svg=True,
        export_metadata=True,
    )

    assert result["png"] is not None
    assert result["pdf"] is not None
    assert result["svg"] is not None
    assert result["metadata"] is not None

    assert Path(result["png"]).exists()
    assert Path(result["pdf"]).exists()
    assert Path(result["svg"]).exists()
    assert Path(result["metadata"]).exists()


def test_save_scatter_plot_paper_v2_raises_on_empty_dataframe(
    tmp_path: Path,
) -> None:
    """
    Verifica que la función falle de manera explícita cuando
    no hay datos válidos para graficar.
    """
    df = pd.DataFrame(
        columns=[
            "model",
            "strategy",
            "train_time_sec",
            "f1_weighted",
            "plot_label",
        ]
    )

    try:
        save_scatter_plot_paper_v2(
            df=df,
            x_col="train_time_sec",
            y_col="f1_weighted",
            model_col="model",
            strategy_col="strategy",
            filename_base="empty_case",
            title="Figura vacía",
            xlabel="Tiempo (s)",
            ylabel="F1-score",
            output_dir=tmp_path,
            pareto_label_col="plot_label",
            export_png=True,
        )
        assert False, "Se esperaba ValueError para un DataFrame vacío."
    except ValueError as exc:
        assert "No hay datos válidos" in str(exc)
