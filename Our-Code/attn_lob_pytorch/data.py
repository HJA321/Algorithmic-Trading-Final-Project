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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ticker, row_index = self.samples[index]
        frame = self.frames[ticker]
        start = row_index - self.time_window
        window = frame.iloc[start:row_index][self.feature_columns].to_numpy(dtype=np.float32)
        feature = torch.from_numpy(normalize_lob_window(window)).unsqueeze(0).float()
        label = torch.tensor(int(frame.iloc[row_index]["label"]), dtype=torch.long)
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


def _compute_alpha(mid_price: np.ndarray, train_end: int, tick_size: float) -> float:
    train_mid = mid_price[:train_end]
    valid = train_mid[np.isfinite(train_mid) & (train_mid > 0)]
    if valid.size == 0:
        return tick_size
    return float(np.median(tick_size / valid))


def _compute_labels(mid_price: np.ndarray, horizon_k: int, alpha: float) -> np.ndarray:
    labels = np.full(len(mid_price), -99, dtype=np.int64)
    for index in range(horizon_k, len(mid_price) - horizon_k):
        price_past = mid_price[index - horizon_k : index + 1].mean()
        price_future = mid_price[index : index + horizon_k + 1].mean()
        if price_past <= 0:
            continue
        ratio = (price_future - price_past) / price_past
        if ratio > alpha:
            labels[index] = 2
        elif ratio < -alpha:
            labels[index] = 0
        else:
            labels[index] = 1
    return labels


def load_lobster_frame(
    ticker: str,
    trading_date: str,
    data_root: str | Path,
    levels: int = 10,
    price_scale: float = 10_000.0,
    tick_size: float = 0.01,
    horizon: int = 10,
    split_train: float = 0.6,
    split_val: float = 0.2,
) -> pd.DataFrame:
    data_root = Path(data_root)
    message_path, orderbook_path = _discover_raw_files(data_root, ticker, trading_date)
    message = pd.read_csv(message_path, header=None, names=MESSAGE_COLUMNS)
    orderbook = pd.read_csv(orderbook_path, header=None, names=_orderbook_columns(levels))
    if len(message) != len(orderbook):
        raise ValueError(f"LOBSTER message/orderbook row mismatch for {ticker}")

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

    train_end = int(len(frame) * split_train)
    val_end = int(len(frame) * (split_train + split_val))
    alpha = _compute_alpha(frame["mid_price"].to_numpy(dtype=np.float64), train_end, tick_size)
    frame["label"] = _compute_labels(frame["mid_price"].to_numpy(dtype=np.float64), horizon, alpha)
    frame["split_id"] = np.select(
        [frame.index < train_end, frame.index < val_end],
        [SPLIT_TO_ID["train"], SPLIT_TO_ID["val"]],
        default=SPLIT_TO_ID["test"],
    ).astype(np.int8)
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
