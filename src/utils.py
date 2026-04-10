from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_project_dirs(paths: list[Path]) -> None:
    for path in paths:
        ensure_dir(path)


def save_json(data: dict[str, Any], filepath: Path) -> None:
    filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class Timer:
    def __enter__(self) -> "Timer":
        self.start = perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.elapsed = perf_counter() - self.start
