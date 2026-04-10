from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import build_config
from src.data_processing import basic_cleaning, load_dataset, save_processed_dataset
from src.evaluation import (
    calculate_metrics,
    save_classification_report,
    save_confusion_matrix,
    save_metrics_table,
    save_roc_curve,
    save_runtime_summary,
)
from src.features import detect_feature_types, split_features_target
from src.modeling import build_model
from src.utils import Timer, ensure_project_dirs


def main() -> int:
    project_root = Path(__file__).resolve().parent
    config = build_config(project_root)

    ensure_project_dirs(
        [
            config.data_raw_dir,
            config.data_processed_dir,
            config.reports_figures_dir,
            config.reports_tables_dir,
            config.models_dir,
        ]
    )

    if not config.input_file.exists():
        print(
            f"[ERROR] No se encontró el archivo de entrada: {config.input_file}\n"
            "Coloca tu dataset en data/raw/transport_data.csv"
        )
        return 1

    df = load_dataset(str(config.input_file))
    df = basic_cleaning(df)
    save_processed_dataset(df, str(config.processed_file))

    x, y = split_features_target(df, config.target_column)
    numeric_features, categorical_features = detect_feature_types(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    model = build_model(
        x=x,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        kernel=config.kernel,
        c_value=config.c_value,
        gamma=config.gamma,
    )

    with Timer() as train_timer:
        model.fit(x_train, y_train)

    with Timer() as inference_timer:
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        y_proba_train = model.predict_proba(x_train)[:, 1]
        y_proba_test = model.predict_proba(x_test)[:, 1]

    train_metrics = calculate_metrics(y_train, y_pred_train, y_proba_train)
    test_metrics = calculate_metrics(y_test, y_pred_test, y_proba_test)

    metrics_comparison = pd.DataFrame(
        [
            {"split": "train", **train_metrics},
            {"split": "test", **test_metrics},
        ]
    )
    metrics_comparison.to_csv(config.reports_tables_dir / "metrics_comparison.csv", index=False)

    save_metrics_table(train_metrics, config.reports_tables_dir / "metrics_train.csv")
    save_metrics_table(test_metrics, config.reports_tables_dir / "metrics_test.csv")
    save_classification_report(
        y_test,
        y_pred_test,
        config.reports_tables_dir / "classification_report_test.csv",
    )
    save_confusion_matrix(
        y_test,
        y_pred_test,
        config.reports_figures_dir / "confusion_matrix.png",
    )
    save_roc_curve(
        y_test,
        y_proba_test,
        config.reports_figures_dir / "roc_curve.png",
    )

    runtime_summary = {
        "training_time_seconds": round(train_timer.elapsed, 6),
        "inference_time_seconds": round(inference_timer.elapsed, 6),
        "n_rows": int(len(df)),
        "n_features": int(x.shape[1]),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
    }
    save_runtime_summary(runtime_summary, config.reports_tables_dir / "runtime_summary.json")

    joblib.dump(model, config.model_file)

    print("Proceso completado correctamente.")
    print(f"Dataset procesado: {config.processed_file}")
    print(f"Modelo guardado: {config.model_file}")
    print(f"Métricas: {config.reports_tables_dir / 'metrics_comparison.csv'}")
    print(f"Matriz de confusión: {config.reports_figures_dir / 'confusion_matrix.png'}")
    print(f"Curva ROC: {config.reports_figures_dir / 'roc_curve.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
