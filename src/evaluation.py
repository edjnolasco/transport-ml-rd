from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(y_true, y_pred, y_proba) -> dict[str, float | None]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": None,
    }

    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        metrics["auc"] = None

    return metrics


def save_metrics_table(metrics: dict, filepath: Path) -> None:
    pd.DataFrame([metrics]).to_csv(filepath, index=False)


def save_classification_report(y_true, y_pred, filepath: Path) -> None:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(filepath, index=True)


def save_confusion_matrix(y_true, y_pred, filepath: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()


def save_roc_curve(y_true, y_proba, filepath: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()


def save_runtime_summary(summary: dict, filepath: Path) -> None:
    filepath.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
