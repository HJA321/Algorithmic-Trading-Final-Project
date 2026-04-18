from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from attn_lob_pytorch.data import normalize_lob_window

ORDERBOOK_COLUMNS = [
    item
    for level in range(1, 11)
    for item in (
        f"ask_price_{level}",
        f"ask_size_{level}",
        f"bid_price_{level}",
        f"bid_size_{level}",
    )
]
MARKET_STATE_COLUMNS = [f"market_state_{index}" for index in range(24)]
SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}


@dataclass
class PreparedMarketData:
    ticker: str
    frame: pd.DataFrame
    market_state_mean: np.ndarray
    market_state_std: np.ndarray


def default_processed_root() -> Path:
    return Path(__file__).resolve().parents[2] / "lobster_attnlob_us_repro" / "data" / "lobster_processed"


def load_prepared_market_data(ticker: str, processed_root: str | Path | None = None) -> PreparedMarketData:
    root = Path(processed_root) if processed_root is not None else default_processed_root()
    frame_path = root / ticker / "events.pkl.gz"
    if not frame_path.exists():
        raise FileNotFoundError(f"Processed frame not found: {frame_path}")

    frame = pd.read_pickle(frame_path, compression="gzip")
    train_mask = frame["split_id"] == SPLIT_TO_ID["train"]
    market_values = frame.loc[train_mask, MARKET_STATE_COLUMNS]
    market_state_mean = market_values.mean(axis=0).to_numpy(dtype=np.float32)
    market_state_std = market_values.std(axis=0).replace(0.0, 1.0).fillna(1.0).to_numpy(dtype=np.float32)
    return PreparedMarketData(
        ticker=ticker,
        frame=frame,
        market_state_mean=market_state_mean,
        market_state_std=market_state_std,
    )


def split_index_bounds(frame: pd.DataFrame, split: str) -> tuple[int, int]:
    split_id = SPLIT_TO_ID[split]
    split_indices = frame.index[frame["split_id"] == split_id].to_numpy(dtype=np.int64)
    if split_indices.size == 0:
        raise ValueError(f"No rows found for split={split}")
    return int(split_indices[0]), int(split_indices[-1]) + 1


def normalized_market_state(frame: pd.DataFrame, row_index: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    market_state = frame.loc[row_index, MARKET_STATE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    return ((market_state - mean) / std).astype(np.float32)


def normalized_lob_observation(frame: pd.DataFrame, row_index: int, time_window: int) -> np.ndarray:
    start = row_index - time_window
    if start < 0:
        raise ValueError("Not enough history to construct lob observation")
    window = frame.iloc[start:row_index][ORDERBOOK_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    return normalize_lob_window(window)
