from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sharpe_ratio(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    std = arr.std(ddof=1) if arr.size > 1 else 0.0
    if std <= 1e-12:
        return 0.0
    return float(arr.mean() / std * math.sqrt(arr.size))


def chunked_range(start: int, stop: int, step: int) -> list[int]:
    values = []
    cursor = start
    while cursor < stop:
        values.append(cursor)
        cursor += step
    return values
