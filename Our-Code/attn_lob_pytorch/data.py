from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DEFAULT_TICKERS = ["AAPL", "MSFT", "AMZN"]
DEFAULT_TRADING_DATE = "2012-06-21"
MESSAGE_COLUMNS = ["time", "event_type", "order_id", "size", "price", "direction"]
SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}


class LobWindowDataset(Dataset):
    def __init__(
        self,
        frames: Dict[str, pd.DataFrame],
        samples: list[tuple[str, int]],
        feature_columns: Sequence[str],
        time_window: int,
    ):
        self.frames = frames
        self.samples = samples
        self.feature_columns = list(feature_columns)
        self.time_window = time_window
        self.feature_arrays = {
            ticker: frame.loc[:, self.feature_columns].to_numpy(dtype=np.float32, copy=True)
            for ticker, frame in frames.items()
        }
        self.label_arrays = {
            ticker: frame["label"].to_numpy(dtype=np.int64, copy=True)
            for ticker, frame in frames.items()
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ticker, row_index = self.samples[index]
        start = row_index - self.time_window
        window = self.feature_arrays[ticker][start:row_index]
        feature = torch.from_numpy(window.copy()).unsqueeze(0)
        label = torch.tensor(self.label_arrays[ticker][row_index], dtype=torch.long)
        return feature, label


@dataclass
class DatasetBundle:
    train: LobWindowDataset
    val: LobWindowDataset
    test: LobWindowDataset
    feature_columns: list[str]
    metadata: dict
    num_classes: int = 3


def _orderbook_columns(levels: int) -> List[str]:
    columns: List[str] = []
    for level in range(1, levels + 1):
        columns.extend(
            [
                f"ask_price_{level}",
                f"ask_size_{level}",
                f"bid_price_{level}",
                f"bid_size_{level}",
            ]
        )
    return columns


def _discover_raw_files(data_root: Path, ticker: str, trading_date: str) -> tuple[Path, Path]:
    ticker_dir = data_root / ticker / trading_date
    message_files = sorted(ticker_dir.glob("*_message_*.csv"))
    orderbook_files = sorted(ticker_dir.glob("*_orderbook_*.csv"))
    if not message_files or not orderbook_files:
        raise FileNotFoundError(f"Missing LOBSTER raw files for {ticker} in {ticker_dir}")
    return message_files[0], orderbook_files[0]


def _sanitize_book(frame: pd.DataFrame, levels: int, tick_size: float) -> pd.DataFrame:
    ask_price_cols = [f"ask_price_{level}" for level in range(1, levels + 1)]
    bid_price_cols = [f"bid_price_{level}" for level in range(1, levels + 1)]
    ask_size_cols = [f"ask_size_{level}" for level in range(1, levels + 1)]
    bid_size_cols = [f"bid_size_{level}" for level in range(1, levels + 1)]

    mid = (frame["ask_price_1"] + frame["bid_price_1"]) / 2.0
    for level in range(levels):
        ask_col = ask_price_cols[level]
        bid_col = bid_price_cols[level]
        ask_mask = frame[ask_col] > 100_000
        bid_mask = frame[bid_col] < 0

        if level == 0:
            frame.loc[ask_mask, ask_col] = mid[ask_mask] + tick_size
            frame.loc[bid_mask, bid_col] = mid[bid_mask] - tick_size
        else:
            frame.loc[ask_mask, ask_col] = frame.loc[ask_mask, ask_price_cols[level - 1]] + tick_size
            frame.loc[bid_mask, bid_col] = frame.loc[bid_mask, bid_price_cols[level - 1]] - tick_size

        frame.loc[ask_mask, ask_size_cols[level]] = 0
        frame.loc[bid_mask, bid_size_cols[level]] = 0
    return frame


def _compute_labels(mid_price: np.ndarray, horizon_k: int, threshold: float = 1e-5) -> np.ndarray:
    """Reproduce the original paper code's mid-price direction labels.

    The original implementation computes:
    - a trailing rolling mean over the previous ``horizon_k`` mid-prices;
    - a future mean by shifting that rolling mean forward by ``horizon_k`` steps;
    - the relative change between future and past means.

    We keep this window definition but map labels to the internal class order used in
    this repo: 0 = down, 1 = stationary, 2 = up.
    """
    mid_series = pd.Series(mid_price, dtype=np.float64)
    price_past = mid_series.rolling(window=horizon_k).mean()
    price_future = price_past.shift(-horizon_k)
    pct_change = (price_future - price_past) / price_past

    labels = np.full(len(mid_series), -99, dtype=np.int64)
    valid_mask = pct_change.notna().to_numpy() & np.isfinite(pct_change.to_numpy()) & (price_past.to_numpy() > 0)
    pct_values = pct_change.to_numpy(dtype=np.float64)

    up_mask = valid_mask & (pct_values >= threshold)
    mid_mask = valid_mask & (pct_values < threshold) & (pct_values > -threshold)
    down_mask = valid_mask & (pct_values <= -threshold)

    labels[down_mask] = 0
    labels[mid_mask] = 1
    labels[up_mask] = 2
    return labels


def load_lobster_frame(
    ticker: str,
    trading_date: str,
    data_root: str | Path,
    levels: int = 10,
    price_scale: float = 10_000.0,
    tick_size: float = 0.01,
    horizon: int = 10,
    label_threshold: float = 1e-5,
    split_train: float = 0.6,
    split_val: float = 0.2,
) -> pd.DataFrame:
    """Load one LOBSTER trading day as an event-aligned frame.

    Assumptions matched to the raw LOBSTER sample format:
    - both CSV files are headerless;
    - message rows follow: time, event_type, order_id, size, price, direction;
    - orderbook rows follow: ask1 price, ask1 size, bid1 price, bid1 size, ask2 ...;
    - message row k and orderbook row k describe the same event-state snapshot;
    - message time is measured in seconds since midnight, so regular trading starts
      around 34200 seconds (09:30:00).
    """
    data_root = Path(data_root)
    message_path, orderbook_path = _discover_raw_files(data_root, ticker, trading_date)
    message = pd.read_csv(
        message_path,
        header=None,
        names=MESSAGE_COLUMNS,
        dtype={
            "time": np.float64,
            "event_type": np.int64,
            "order_id": np.int64,
            "size": np.float64,
            "price": np.float64,
            "direction": np.int64,
        },
    )
    orderbook_columns = _orderbook_columns(levels)
    orderbook = pd.read_csv(orderbook_path, header=None, names=orderbook_columns, dtype=np.float64)
    if len(message) != len(orderbook):
        raise ValueError(f"LOBSTER message/orderbook row mismatch for {ticker}")
    if not message["time"].is_monotonic_increasing:
        raise ValueError(f"LOBSTER message timestamps are not monotonic for {ticker}")

    # LOBSTER aligns message row k with orderbook row k, so we concatenate by row index
    # rather than merging on time values.
    frame = pd.concat([message, orderbook], axis=1)
    price_columns = [column for column in frame.columns if "price" in column]
    size_columns = [column for column in frame.columns if column.endswith("size")]

    for column in price_columns:
        frame[column] = frame[column].astype(np.float64) / price_scale
    for column in size_columns:
        frame[column] = frame[column].astype(np.float64)

    frame = _sanitize_book(frame, levels=levels, tick_size=tick_size)
    frame["mid_price"] = (frame["ask_price_1"] + frame["bid_price_1"]) / 2.0
    frame["spread"] = frame["ask_price_1"] - frame["bid_price_1"]

    val_end = int(len(frame) * (split_train + split_val))
    frame["label"] = _compute_labels(
        frame["mid_price"].to_numpy(dtype=np.float64),
        horizon,
        threshold=label_threshold,
    )
    frame["split_id"] = np.select(
        [frame.index < int(len(frame) * split_train), frame.index < val_end],
        [SPLIT_TO_ID["train"], SPLIT_TO_ID["val"]],
        default=SPLIT_TO_ID["test"],
    ).astype(np.int8)
    frame["time_hours"] = frame["time"] / 3600.0
    frame["ticker"] = ticker
    frame["trading_date"] = trading_date
    return frame


def normalize_lob_window(window: np.ndarray) -> np.ndarray:
    normalized = np.asarray(window, dtype=np.float32).copy()
    ask1 = normalized[:, 0]
    bid1 = normalized[:, 2]
    mid = ((ask1 + bid1) / 2.0).reshape(-1, 1)

    price_indices = [index for index in range(normalized.shape[1]) if index % 4 in (0, 2)]
    size_indices = [index for index in range(normalized.shape[1]) if index % 4 in (1, 3)]

    normalized[:, price_indices] = (normalized[:, price_indices] - mid) / np.clip(mid, 1e-6, None)
    price_mean = normalized[:, price_indices].mean(axis=0, keepdims=True)
    price_std = normalized[:, price_indices].std(axis=0, keepdims=True)
    normalized[:, price_indices] = (normalized[:, price_indices] - price_mean) / np.clip(price_std, 1e-6, None)

    max_volume = np.max(np.abs(normalized[:, size_indices]), axis=0, keepdims=True)
    normalized[:, size_indices] = normalized[:, size_indices] / np.clip(max_volume, 1.0, None)
    return normalized.astype(np.float32)


def normalize_lob_batch(batch: torch.Tensor) -> torch.Tensor:
    normalized = batch.to(dtype=torch.float32).clone()
    feature_count = normalized.shape[-1]
    price_indices = [index for index in range(feature_count) if index % 4 in (0, 2)]
    size_indices = [index for index in range(feature_count) if index % 4 in (1, 3)]

    mid = ((normalized[..., 0:1] + normalized[..., 2:3]) / 2.0).clamp_min(1e-6)

    price_features = normalized[..., price_indices]
    price_features = (price_features - mid) / mid
    price_mean = price_features.mean(dim=-2, keepdim=True)
    price_std = price_features.std(dim=-2, unbiased=False, keepdim=True)
    normalized[..., price_indices] = (price_features - price_mean) / price_std.clamp_min(1e-6)

    size_features = normalized[..., size_indices]
    max_volume = size_features.abs().amax(dim=-2, keepdim=True)
    normalized[..., size_indices] = size_features / max_volume.clamp_min(1.0)
    return normalized


def _collect_split_samples(frame: pd.DataFrame, ticker: str, split: str, time_window: int) -> list[tuple[str, int]]:
    split_id = SPLIT_TO_ID[split]
    valid_indices = frame.index[
        (frame["split_id"] == split_id)
        & (frame["label"] >= 0)
        & (frame.index >= time_window)
        & (frame["event_type"] != 7)
    ]
    return [(ticker, int(index)) for index in valid_indices]


def load_and_process_day(
    ticker: str,
    trading_date: str,
    data_root: str | Path,
    levels: int = 10,
    horizon: int = 10,
    time_window: int = 50,
    price_scale: float = 10_000.0,
    tick_size: float = 0.01,
    label_threshold: float = 1e-5,
    split_train: float = 0.6,
    split_val: float = 0.2,
) -> tuple[pd.DataFrame, list[str], dict]:
    frame = load_lobster_frame(
        ticker=ticker,
        trading_date=trading_date,
        data_root=data_root,
        levels=levels,
        price_scale=price_scale,
        tick_size=tick_size,
        horizon=horizon,
        label_threshold=label_threshold,
        split_train=split_train,
        split_val=split_val,
    )
    feature_columns = _orderbook_columns(levels)
    metadata = {
        "ticker": ticker,
        "trading_date": trading_date,
        "rows": len(frame),
        "levels": levels,
        "horizon": horizon,
        "label_threshold": label_threshold,
        "time_window": time_window,
        "split_train": split_train,
        "split_val": split_val,
        "split_test": 1.0 - split_train - split_val,
    }
    return frame, feature_columns, metadata


def create_datasets(
    tickers: Sequence[str],
    trading_date: str,
    data_root: str | Path,
    levels: int = 10,
    horizon: int = 10,
    time_window: int = 50,
    price_scale: float = 10_000.0,
    tick_size: float = 0.01,
    label_threshold: float = 1e-5,
    split_train: float = 0.6,
    split_val: float = 0.2,
) -> DatasetBundle:
    frames: Dict[str, pd.DataFrame] = {}
    train_samples: list[tuple[str, int]] = []
    val_samples: list[tuple[str, int]] = []
    test_samples: list[tuple[str, int]] = []
    all_metadata: Dict[str, dict] = {}
    feature_columns = _orderbook_columns(levels)

    for ticker in tickers:
        frame, _, metadata = load_and_process_day(
            ticker=ticker,
            trading_date=trading_date,
            data_root=data_root,
            levels=levels,
            horizon=horizon,
            time_window=time_window,
            price_scale=price_scale,
            tick_size=tick_size,
            label_threshold=label_threshold,
            split_train=split_train,
            split_val=split_val,
        )
        frames[ticker] = frame
        all_metadata[ticker] = metadata
        train_samples.extend(_collect_split_samples(frame, ticker, "train", time_window))
        val_samples.extend(_collect_split_samples(frame, ticker, "val", time_window))
        test_samples.extend(_collect_split_samples(frame, ticker, "test", time_window))

    metadata = {
        "tickers": list(tickers),
        "trading_date": trading_date,
        "levels": levels,
        "horizon": horizon,
        "label_threshold": label_threshold,
        "time_window": time_window,
        "split_train": split_train,
        "split_val": split_val,
        "split_test": 1.0 - split_train - split_val,
        "per_ticker": all_metadata,
    }
    return DatasetBundle(
        train=LobWindowDataset(frames, train_samples, feature_columns, time_window),
        val=LobWindowDataset(frames, val_samples, feature_columns, time_window),
        test=LobWindowDataset(frames, test_samples, feature_columns, time_window),
        feature_columns=feature_columns,
        metadata=metadata,
    )
