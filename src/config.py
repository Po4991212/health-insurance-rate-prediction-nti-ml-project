"""Project configuration (single source of truth).

Inspired by the "parameters.py" approach used in the reference repo, this file
keeps the most important knobs in one place and makes CLI scripts cleaner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrainConfig:
    data_path: Path = ROOT / "data" / "insurance.csv"
    artifact_dir: Path = ROOT / "artifacts"
    test_size: float = 0.2
    random_state: int = 42
