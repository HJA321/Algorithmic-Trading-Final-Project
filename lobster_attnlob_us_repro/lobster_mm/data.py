from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from urllib.request import urlopen

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .config import ExperimentConfig
from .utils import dump_json, ensure_dir, load_json


MESSAGE_COLUMNS = ["time", "event_type", "order_id", "size", "price", "direction"]
SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}
ID_TO_SPLIT = {value: key for key, value in SPLIT_TO_ID.items()}


@dataclass
class PreparedTicker:
    ticker: str
    frame: pd.DataFrame
    metadata: dict


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


def download_public_samples(config: ExperimentConfig, tickers: Optional[Sequence[str]] = None, force: bool = False) -> None:
    selected = list(tickers or config.tickers)
    for ticker in selected:
        if ticker not in config.sample_urls:
            raise KeyError(f"No sample URL configured for ticker {ticker}")
        target_dir = ensure_dir(config.ticker_raw_dir(ticker))
        url = config.sample_urls[ticker]
        zip_path = target_dir / Path(url).name
        if force or not zip_path.exists():
            with urlopen(url, timeout=60) as response:
                zip_path.write_bytes(response.read())
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                destination = target_dir / Path(member).name
                if force or not destination.exists():
                    destination.write_bytes(archive.read(member))


def _discover_raw_files(config: ExperimentConfig, ticker: str) -> tuple[Path, Path]:
    raw_dir = config.ticker_raw_dir(ticker)
    message_files = sorted(raw_dir.glob("*_message_*.csv"))
    orderbook_files = sorted(raw_dir.glob("*_orderbook_*.csv"))
    if not message_files or not orderbook_files:
        raise FileNotFoundError(f"Missing raw LOBSTER files for {ticker} in {raw_dir}")
    return message_files[0], orderbook_files[0]


def _sanitize_book(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    ask_price_cols = [f"ask_price_{i}" for i in range(1, config.levels + 1)]
    bid_price_cols = [f"bid_price_{i}" for i in range(1, config.levels + 1)]
    ask_size_cols = [f"ask_size_{i}" for i in range(1, config.levels + 1)]
    bid_size_cols = [f"bid_size_{i}" for i in range(1, config.levels + 1)]
    mid = (df["ask_price_1"] + df["bid_price_1"]) / 2.0

    for level in range(config.levels):
        ask_col = ask_price_cols[level]
        bid_col = bid_price_cols[level]
        ask_mask = df[ask_col] > 100_000
        bid_mask = df[bid_col] < 0
        if level == 0:
            df.loc[ask_mask, ask_col] = mid[ask_mask] + config.tick_size
            df.loc[bid_mask, bid_col] = mid[bid_mask] - config.tick_size
        else:
            prev_ask = ask_price_cols[level - 1]
            prev_bid = bid_price_cols[level - 1]
            df.loc[ask_mask, ask_col] = df.loc[ask_mask, prev_ask] + config.tick_size
            df.loc[bid_mask, bid_col] = df.loc[bid_mask, prev_bid] - config.tick_size
        df.loc[ask_mask, ask_size_cols[level]] = 0
        df.loc[bid_mask, bid_size_cols[level]] = 0
    return df


def _compute_alpha(mid_price: np.ndarray, train_end: int, tick_size: float) -> float:
    train_mid = mid_price[:train_end]
    valid = train_mid[np.isfinite(train_mid) & (train_mid > 0)]
    if valid.size == 0:
        return tick_size
    return float(np.median(tick_size / valid))


def _compute_labels(mid_price: np.ndarray, horizon_k: int, alpha: float) -> np.ndarray:
    n = len(mid_price)
    labels = np.full(n, -99, dtype=np.int64)
    for idx in range(horizon_k, n - horizon_k):
        past = mid_price[idx - horizon_k : idx + 1].mean()
        future = mid_price[idx : idx + horizon_k + 1].mean()
        if past <= 0:
            continue
        ratio = (future - past) / past
        if ratio > alpha:
            labels[idx] = 2
        elif ratio < -alpha:
            labels[idx] = 0
        else:
            labels[idx] = 1
    return labels


def _window_search_indices(times: np.ndarray, window_seconds: float) -> np.ndarray:
    thresholds = times - window_seconds
    return np.searchsorted(times, thresholds, side="left")


def _imbalance_features(
    times: np.ndarray,
    buy_volume: np.ndarray,
    sell_volume: np.ndarray,
    buy_count: np.ndarray,
    sell_count: np.ndarray,
    windows: Sequence[float],
) -> list[np.ndarray]:
    def cumulative(arr: np.ndarray) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(arr, dtype=np.float64)))

    buy_vol_c = cumulative(buy_volume)
    sell_vol_c = cumulative(sell_volume)
    buy_cnt_c = cumulative(buy_count)
    sell_cnt_c = cumulative(sell_count)
    features: list[np.ndarray] = []
    for window in windows:
        start = _window_search_indices(times, window)
        idx = np.arange(len(times)) + 1
        bv = buy_vol_c[idx] - buy_vol_c[start]
        sv = sell_vol_c[idx] - sell_vol_c[start]
        bc = buy_cnt_c[idx] - buy_cnt_c[start]
        sc = sell_cnt_c[idx] - sell_cnt_c[start]
        features.append((bv - sv) / (bv + sv + 1e-8))
        features.append((bc - sc) / (bc + sc + 1e-8))
    return features


def _compute_dynamic_features(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    times = frame["time"].to_numpy(dtype=np.float64)
    size = frame["size"].to_numpy(dtype=np.float64)
    direction = frame["direction"].to_numpy(dtype=np.int64)
    event_type = frame["event_type"].to_numpy(dtype=np.int64)

    limit_mask = event_type == 1
    cancel_mask = np.isin(event_type, [2, 3])
    exec_mask = np.isin(event_type, [4, 5])

    limit_buy = size * limit_mask * (direction == 1)
    limit_sell = size * limit_mask * (direction == -1)
    cancel_buy = size * cancel_mask * (direction == 1)
    cancel_sell = size * cancel_mask * (direction == -1)
    market_buy = size * exec_mask * (direction == -1)
    market_sell = size * exec_mask * (direction == 1)

    feature_arrays = []
    for buy_v, sell_v, buy_c, sell_c in [
        (market_buy, market_sell, exec_mask * (direction == -1), exec_mask * (direction == 1)),
        (limit_buy, limit_sell, limit_mask * (direction == 1), limit_mask * (direction == -1)),
        (cancel_buy, cancel_sell, cancel_mask * (direction == 1), cancel_mask * (direction == -1)),
    ]:
        feature_arrays.extend(
            _imbalance_features(times, buy_v, sell_v, buy_c.astype(np.float64), sell_c.astype(np.float64), [10.0, 60.0, 300.0])
        )

    log_mid = np.log(frame["mid_price"].to_numpy(dtype=np.float64).clip(min=1e-8))
    log_ret_sq = np.zeros(len(frame), dtype=np.float64)
    log_ret_sq[1:] = np.diff(log_mid) ** 2
    diff_mid = np.zeros(len(frame), dtype=np.float64)
    diff_mid[1:] = np.diff(frame["mid_price"].to_numpy(dtype=np.float64))
    gain = np.maximum(diff_mid, 0.0)
    loss = np.maximum(-diff_mid, 0.0)
    c_ret = np.concatenate(([0.0], np.cumsum(log_ret_sq)))
    c_gain = np.concatenate(([0.0], np.cumsum(gain)))
    c_loss = np.concatenate(([0.0], np.cumsum(loss)))

    for window in [300.0, 600.0, 1800.0]:
        start = _window_search_indices(times, window)
        idx = np.arange(len(frame)) + 1
        rv_sum = c_ret[idx] - c_ret[start + 1]
        rv = np.sqrt(np.maximum(rv_sum, 0.0))
        gains = c_gain[idx] - c_gain[start + 1]
        losses = c_loss[idx] - c_loss[start + 1]
        rsi = gains / (gains + losses + 1e-8)
        feature_arrays.append(rv)
        feature_arrays.append(rsi)

    columns = [f"market_state_{i}" for i in range(24)]
    values = np.column_stack(feature_arrays)
    return pd.DataFrame(values, columns=columns, index=frame.index)


def prepare_ticker_data(config: ExperimentConfig, ticker: str) -> Path:
    message_path, orderbook_path = _discover_raw_files(config, ticker)
    orderbook = pd.read_csv(orderbook_path, header=None, names=_orderbook_columns(config.levels))
    message = pd.read_csv(message_path, header=None, names=MESSAGE_COLUMNS)
    if len(orderbook) != len(message):
        raise ValueError(f"LOBSTER message/orderbook row mismatch for {ticker}")

    frame = pd.concat([message, orderbook], axis=1)
    price_cols = [col for col in frame.columns if "price" in col]
    size_cols = [col for col in frame.columns if col.endswith("size")]
    for col in price_cols:
        frame[col] = frame[col].astype(np.float64) / config.price_scale
    for col in size_cols:
        frame[col] = frame[col].astype(np.float64)

    frame = _sanitize_book(frame, config)
    frame["mid_price"] = (frame["ask_price_1"] + frame["bid_price_1"]) / 2.0
    frame["spread"] = frame["ask_price_1"] - frame["bid_price_1"]
    frame["is_execution"] = frame["event_type"].isin([4, 5]).astype(np.int8)
    frame["trade_price"] = np.where(frame["is_execution"] == 1, frame["price"], np.nan)
    frame["trade_size"] = np.where(frame["is_execution"] == 1, frame["size"], 0.0)

    n_rows = len(frame)
    train_end = int(n_rows * config.split_train)
    val_end = int(n_rows * (config.split_train + config.split_val))
    alpha = _compute_alpha(frame["mid_price"].to_numpy(dtype=np.float64), train_end, config.tick_size)
    frame["label"] = _compute_labels(frame["mid_price"].to_numpy(dtype=np.float64), config.horizon_k, alpha)
    frame["split_id"] = np.select(
        [frame.index < train_end, frame.index < val_end],
        [SPLIT_TO_ID["train"], SPLIT_TO_ID["val"]],
        default=SPLIT_TO_ID["test"],
    ).astype(np.int8)
    frame = pd.concat([frame, _compute_dynamic_features(frame, config)], axis=1)

    output_dir = ensure_dir(config.ticker_processed_dir(ticker))
    frame_path = output_dir / "events.pkl.gz"
    frame.to_pickle(frame_path, compression="gzip")
    metadata = {
        "ticker": ticker,
        "trading_date": config.trading_date,
        "rows": n_rows,
        "levels": config.levels,
        "time_window": config.time_window,
        "horizon_k": config.horizon_k,
        "alpha": alpha,
        "train_end": train_end,
        "val_end": val_end,
        "test_end": n_rows,
        "message_path": str(message_path),
        "orderbook_path": str(orderbook_path),
        "frame_path": str(frame_path),
    }
    dump_json(output_dir / "metadata.json", metadata)
    return frame_path


def load_prepared_ticker(config: ExperimentConfig, ticker: str) -> PreparedTicker:
    output_dir = config.ticker_processed_dir(ticker)
    frame = pd.read_pickle(output_dir / "events.pkl.gz", compression="gzip")
    metadata = load_json(output_dir / "metadata.json")
    return PreparedTicker(ticker=ticker, frame=frame, metadata=metadata)


def prepare_all_tickers(config: ExperimentConfig, tickers: Optional[Sequence[str]] = None) -> List[Path]:
    selected = list(tickers or config.tickers)
    return [prepare_ticker_data(config, ticker) for ticker in selected]


def normalize_lob_window(book_window: np.ndarray) -> np.ndarray:
    window = np.asarray(book_window, dtype=np.float32).copy()
    ask1 = window[:, 0]
    bid1 = window[:, 2]
    mid = ((ask1 + bid1) / 2.0).reshape(-1, 1)
    price_idx = [idx for idx in range(window.shape[1]) if idx % 4 in (0, 2)]
    size_idx = [idx for idx in range(window.shape[1]) if idx % 4 in (1, 3)]

    window[:, price_idx] = (window[:, price_idx] - mid) / np.clip(mid, 1e-6, None)
    price_mean = window[:, price_idx].mean(axis=0, keepdims=True)
    price_std = window[:, price_idx].std(axis=0, keepdims=True)
    window[:, price_idx] = (window[:, price_idx] - price_mean) / np.clip(price_std, 1e-6, None)
    max_volume = np.max(np.abs(window[:, size_idx]), axis=0, keepdims=True)
    window[:, size_idx] = window[:, size_idx] / np.clip(max_volume, 1.0, None)
    return window.astype(np.float32)


class LobPretrainDataset(Dataset):
    def __init__(self, config: ExperimentConfig, tickers: Sequence[str], split: str):
        self.config = config
        self.frames: Dict[str, pd.DataFrame] = {}
        self.samples: list[tuple[str, int]] = []
        self.labels: list[int] = []

        for ticker in tickers:
            prepared = load_prepared_ticker(config, ticker)
            frame = prepared.frame
            self.frames[ticker] = frame
            split_id = SPLIT_TO_ID[split]
            valid = frame.index[
                (frame["split_id"] == split_id)
                & (frame["label"] >= 0)
                & (frame.index >= config.time_window)
            ]
            for idx in valid:
                idx = int(idx)
                self.samples.append((ticker, idx))
                self.labels.append(int(frame.iloc[idx]["label"]))

        self.book_columns = _orderbook_columns(config.levels)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray]:
        ticker, idx = self.samples[item]
        frame = self.frames[ticker]
        start = idx - self.config.time_window
        lob_window = frame.iloc[start:idx][self.book_columns].to_numpy(dtype=np.float32)
        lob_window = normalize_lob_window(lob_window)
        label = int(frame.iloc[idx]["label"])
        return lob_window, np.asarray(label, dtype=np.int64)
