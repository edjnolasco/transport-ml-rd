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
from src.features import detect_feature_types
from src.modeling import build_model
from src.utils import Timer, ensure_project_dirs


def main() -> int:
    # ==============================================================
    # 1. CONFIGURACIÓN GENERAL DEL PROYECTO
    # ==============================================================

    # Ruta raíz del proyecto
    project_root = Path(__file__).resolve().parent

    # Construcción de configuración central (rutas + parámetros)
    config = build_config(project_root)

    # Crear carpetas necesarias si no existen
    ensure_project_dirs(
        [
            config.data_raw_dir,
            config.data_processed_dir,
            config.reports_figures_dir,
            config.reports_tables_dir,
            config.models_dir,
        ]
    )

    # ==============================================================
    # 2. VALIDACIÓN DE INPUT
    # ==============================================================

    # Verificar que exista el dataset en data/raw
    if not config.input_file.exists():
        print(
            f"[ERROR] No se encontró el archivo de entrada: {config.input_file}\n"
            "Coloca tu dataset en data/raw/transport_data.csv"
        )
        return 1

    # ==============================================================
    # 3. CARGA Y LIMPIEZA DEL DATASET
    # ==============================================================

    # Cargar dataset
    df = load_dataset(str(config.input_file))

    # Limpieza básica (columnas, duplicados, etc.)
    df = basic_cleaning(df)

    # Validar columnas mínimas necesarias
    required_columns = {"provincia", "year", "fallecidos"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        print(
            "[ERROR] El dataset no contiene las columnas requeridas: "
            f"{', '.join(sorted(missing_columns))}"
        )
        return 1

    # ==============================================================
    # 4. PREPARACIÓN DE VARIABLE OBJETIVO (TARGET)
    # ==============================================================

    # Convertir a numérico (evita errores si hay strings o valores inválidos)
    df["fallecidos"] = pd.to_numeric(df["fallecidos"], errors="coerce")

    # Eliminar filas sin valor en fallecidos
    df = df.dropna(subset=["fallecidos"]).copy()

    # Validar que el dataset no quedó vacío
    if df.empty:
        print("[ERROR] El dataset quedó vacío después de limpiar 'fallecidos'.")
        return 1

    # Crear variable objetivo (clasificación binaria)
    # Se define como alto riesgo (1) si supera el percentil 75
    threshold = df["fallecidos"].quantile(0.75)

    df["target"] = (df["fallecidos"] >= threshold).astype(int)

    # Guardar dataset procesado (para trazabilidad y análisis posterior)
    save_processed_dataset(df, str(config.processed_file))

    # ==============================================================
    # 5. SEPARACIÓN DE VARIABLES (X, y)
    # ==============================================================

    # ⚠️ IMPORTANTE:
    # Se elimina "fallecidos" para evitar fuga de información (data leakage),
    # ya que el target se construyó a partir de esta variable.
    x = df.drop(columns=["target", "fallecidos"])

    # Variable objetivo
    y = df["target"].astype(int)

    # Diagnóstico de clases
    print("\nDistribución de clases:")
    print(y.value_counts(dropna=False))

    # Validar que existan al menos dos clases
    if y.nunique() < 2:
        print("[ERROR] La variable objetivo no tiene al menos dos clases.")
        return 1

    # Detectar tipos de variables automáticamente
    numeric_features, categorical_features = detect_feature_types(x)

    # ==============================================================
    # 6. DIVISIÓN TRAIN / TEST
    # ==============================================================

    # Se utiliza estratificación para mantener proporción de clases
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    # ==============================================================
    # 7. CONSTRUCCIÓN DEL MODELO SVM
    # ==============================================================

    # Se construye un pipeline completo:
    # preprocesamiento + modelo SVM
    model = build_model(
        x=x,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        kernel=config.kernel,
        c_value=config.c_value,
        gamma=config.gamma,
    )

    # ==============================================================
    # 8. ENTRENAMIENTO DEL MODELO
    # ==============================================================

    # Medición de tiempo de entrenamiento
    with Timer() as train_timer:
        model.fit(x_train, y_train)

    # ==============================================================
    # 9. PREDICCIÓN E INFERENCIA
    # ==============================================================

    with Timer() as inference_timer:
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

        # Probabilidades necesarias para curva ROC
        y_proba_train = model.predict_proba(x_train)[:, 1]
        y_proba_test = model.predict_proba(x_test)[:, 1]

    # ==============================================================
    # 10. EVALUACIÓN DEL MODELO
    # ==============================================================

    train_metrics = calculate_metrics(y_train, y_pred_train, y_proba_train)
    test_metrics = calculate_metrics(y_test, y_pred_test, y_proba_test)

    # Comparación train vs test (clave para detectar sobreajuste)
    metrics_comparison = pd.DataFrame(
        [
            {"split": "train", **train_metrics},
            {"split": "test", **test_metrics},
        ]
    )

    metrics_comparison.to_csv(
        config.reports_tables_dir / "metrics_comparison.csv",
        index=False,
    )

    # Guardar métricas individuales
    save_metrics_table(train_metrics, config.reports_tables_dir / "metrics_train.csv")
    save_metrics_table(test_metrics, config.reports_tables_dir / "metrics_test.csv")

    # Reporte detallado de clasificación
    save_classification_report(
        y_test,
        y_pred_test,
        config.reports_tables_dir / "classification_report_test.csv",
    )

    # Matriz de confusión
    save_confusion_matrix(
        y_test,
        y_pred_test,
        config.reports_figures_dir / "confusion_matrix.png",
    )

    # Curva ROC
    save_roc_curve(
        y_test,
        y_proba_test,
        config.reports_figures_dir / "roc_curve.png",
    )

    # ==============================================================
    # 11. COSTE COMPUTACIONAL
    # ==============================================================

    runtime_summary = {
        "training_time_seconds": round(train_timer.elapsed, 6),
        "inference_time_seconds": round(inference_timer.elapsed, 6),
        "n_rows": int(len(df)),
        "n_features": int(x.shape[1]),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "threshold_fallecidos_p75": float(threshold),
    }

    save_runtime_summary(
        runtime_summary,
        config.reports_tables_dir / "runtime_summary.json",
    )

    # ==============================================================
    # 12. GUARDAR MODELO ENTRENADO
    # ==============================================================

    joblib.dump(model, config.model_file)

    # ==============================================================
    # 13. SALIDA FINAL
    # ==============================================================

    print("\nProceso completado correctamente.")
    print(f"Dataset procesado: {config.processed_file}")
    print(f"Modelo guardado: {config.model_file}")
    print(f"Métricas: {config.reports_tables_dir / 'metrics_comparison.csv'}")
    print(f"Matriz de confusión: {config.reports_figures_dir / 'confusion_matrix.png'}")
    print(f"Curva ROC: {config.reports_figures_dir / 'roc_curve.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
