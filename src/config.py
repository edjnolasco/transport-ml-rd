from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    project_root: Path
    data_raw_dir: Path
    data_processed_dir: Path
    reports_figures_dir: Path
    reports_tables_dir: Path
    models_dir: Path
    input_file: Path
    processed_file: Path
    model_file: Path
    target_column: str = "target"
    test_size: float = 0.2
    random_state: int = 42
    kernel: str = "rbf"
    c_value: float = 3.0
    gamma: str = "scale"


def build_config(project_root: Path) -> Config:
    data_raw_dir = project_root / "data" / "raw"
    data_processed_dir = project_root / "data" / "processed"
    reports_figures_dir = project_root / "reports" / "figures"
    reports_tables_dir = project_root / "reports" / "tables"
    models_dir = project_root / "models"

    return Config(
        project_root=project_root,
        data_raw_dir=data_raw_dir,
        data_processed_dir=data_processed_dir,
        reports_figures_dir=reports_figures_dir,
        reports_tables_dir=reports_tables_dir,
        models_dir=models_dir,
        input_file=data_raw_dir / "transport_data.csv",
        processed_file=data_processed_dir / "transport_clean.csv",
        model_file=models_dir / "svm_transport_model.joblib",
    )
