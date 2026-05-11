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
        aux_feature_columns: Sequence[str],
        lob_size_max: Dict[str, np.ndarray],
        time_window: int,
    ):
        self.frames = frames
        self.samples = samples
        self.feature_columns = list(feature_columns)
        self.aux_feature_columns = list(aux_feature_columns)
        self.lob_size_max = lob_size_max
        self.time_window = time_window
        self.feature_arrays = {
            ticker: frame.loc[:, self.feature_columns].to_numpy(dtype=np.float32, copy=True)
            for ticker, frame in frames.items()
        }
        self.aux_feature_arrays = {
            ticker: frame.loc[:, self.aux_feature_columns].to_numpy(dtype=np.float32, copy=True)
            for ticker, frame in frames.items()
        }
        self.label_arrays = {
            ticker: frame["label"].to_numpy(dtype=np.int64, copy=True)
            for ticker, frame in frames.items()
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ticker, row_index = self.samples[index]
        end = row_index + 1
        start = end - self.time_window
        window = self.feature_arrays[ticker][start:end]
        window = normalize_lob_window(window, size_max=self.lob_size_max[ticker])
        feature = torch.from_numpy(window).unsqueeze(0)
        aux_feature = torch.from_numpy(self.aux_feature_arrays[ticker][row_index].copy())
        label = torch.tensor(self.label_arrays[ticker][row_index], dtype=torch.long)
        return feature, aux_feature, label


@dataclass
class DatasetBundle:
    train: LobWindowDataset
    val: LobWindowDataset
    test: LobWindowDataset
    feature_columns: list[str]
    aux_feature_columns: list[str]
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


def _safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray, eps: float = 1e-6) -> pd.Series:
    numerator_series = pd.Series(numerator, copy=False, dtype=np.float64)
    denominator_series = pd.Series(denominator, copy=False, dtype=np.float64)
    return numerator_series / denominator_series.clip(lower=eps)


def _compute_auxiliary_features(frame: pd.DataFrame, levels: int, time_window: int) -> list[str]:
    ask_size_cols = [f"ask_size_{level}" for level in range(1, levels + 1)]
    bid_size_cols = [f"bid_size_{level}" for level in range(1, levels + 1)]
    ask_price_cols = [f"ask_price_{level}" for level in range(1, levels + 1)]
    bid_price_cols = [f"bid_price_{level}" for level in range(1, levels + 1)]

    eps = 1e-6
    rolling_window = max(int(time_window), 2)
    mid = frame["mid_price"].clip(lower=eps)
    spread = frame["spread"]

    ask_depth_5 = frame[ask_size_cols[: min(5, levels)]].sum(axis=1)
    bid_depth_5 = frame[bid_size_cols[: min(5, levels)]].sum(axis=1)
    ask_depth_all = frame[ask_size_cols].sum(axis=1)
    bid_depth_all = frame[bid_size_cols].sum(axis=1)

    mid_log_return = np.log(mid).diff().fillna(0.0)
    gains = mid_log_return.clip(lower=0.0)
    losses = (-mid_log_return).clip(lower=0.0)

    signed_size = frame["size"] * frame["direction"]
    is_limit = frame["event_type"] == 1
    is_cancel = frame["event_type"].isin([2, 3])
    is_market = frame["event_type"].isin([4, 5])

    limit_signed = signed_size.where(is_limit, 0.0)
    cancel_signed = signed_size.where(is_cancel, 0.0)
    market_signed = signed_size.where(is_market, 0.0)

    micro_price = (
        frame["ask_price_1"] * frame["bid_size_1"] + frame["bid_price_1"] * frame["ask_size_1"]
    ) / (frame["ask_size_1"] + frame["bid_size_1"] + eps)
    weighted_ask = np.average(frame[ask_price_cols].to_numpy(), axis=1, weights=frame[ask_size_cols].to_numpy() + eps)
    weighted_bid = np.average(frame[bid_price_cols].to_numpy(), axis=1, weights=frame[bid_size_cols].to_numpy() + eps)

    feature_map = {
        "rel_spread": _safe_divide(spread, mid, eps),
        "top_depth_imbalance": _safe_divide(frame["bid_size_1"] - frame["ask_size_1"], frame["bid_size_1"] + frame["ask_size_1"], eps),
        "depth_imbalance_5": _safe_divide(bid_depth_5 - ask_depth_5, bid_depth_5 + ask_depth_5, eps),
        "depth_imbalance_10": _safe_divide(bid_depth_all - ask_depth_all, bid_depth_all + ask_depth_all, eps),
        "microprice_bias": _safe_divide(micro_price - mid, mid, eps),
        "depth_slope_bias": _safe_divide((weighted_ask - mid) - (mid - weighted_bid), mid, eps),
        "realized_volatility": mid_log_return.pow(2).rolling(rolling_window, min_periods=2).sum().pow(0.5).fillna(0.0),
        "rsi_like": _safe_divide(
            gains.rolling(rolling_window, min_periods=2).mean(),
            gains.rolling(rolling_window, min_periods=2).mean() + losses.rolling(rolling_window, min_periods=2).mean(),
            eps,
        ).fillna(0.5),
        "signed_flow_imbalance": _safe_divide(
            signed_size.rolling(rolling_window, min_periods=2).sum(),
            signed_size.abs().rolling(rolling_window, min_periods=2).sum(),
            eps,
        ).fillna(0.0),
        "limit_flow_imbalance": _safe_divide(
            limit_signed.rolling(rolling_window, min_periods=2).sum(),
            limit_signed.abs().rolling(rolling_window, min_periods=2).sum(),
            eps,
        ).fillna(0.0),
        "cancel_flow_imbalance": _safe_divide(
            cancel_signed.rolling(rolling_window, min_periods=2).sum(),
            cancel_signed.abs().rolling(rolling_window, min_periods=2).sum(),
            eps,
        ).fillna(0.0),
        "market_flow_imbalance": _safe_divide(
            market_signed.rolling(rolling_window, min_periods=2).sum(),
            market_signed.abs().rolling(rolling_window, min_periods=2).sum(),
            eps,
        ).fillna(0.0),
    }

    for column, values in feature_map.items():
        frame[column] = pd.Series(values, index=frame.index, dtype=np.float64).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return list(feature_map.keys())


def load_lobster_frame(
    ticker: str,
    trading_date: str,
    data_root: str | Path,
    levels: int = 10,
    price_scale: float = 10_000.0,
    tick_size: float = 0.01,
    horizon: int = 20,
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


def lob_size_columns(levels: int = 10) -> list[str]:
    columns: list[str] = []
    for level in range(1, levels + 1):
        columns.extend([f"ask_size_{level}", f"bid_size_{level}"])
    return columns


def compute_lob_size_max(frame: pd.DataFrame, levels: int = 10, mask: pd.Series | np.ndarray | None = None) -> np.ndarray:
    """Return max-volume denominators in the same order as normalize_lob_window expects."""
    size_columns = lob_size_columns(levels)
    source = frame.loc[mask, size_columns] if mask is not None else frame.loc[:, size_columns]
    size_max = source.max(axis=0).replace(0.0, 1.0).fillna(1.0).to_numpy(dtype=np.float32)
    return np.maximum(size_max, 1.0)


def normalize_lob_window(window: np.ndarray, size_max: np.ndarray | None = None) -> np.ndarray:
    """Normalize one raw LOB window with the paper author's price/volume convention.

    Input column order is ask_price, ask_size, bid_price, bid_size for levels 1..N.
    Prices become relative to the row mid-price, and sizes use max-normalization.
    Passing train-set ``size_max`` keeps pretraining and RL observations on the
    same scale while avoiding validation/test leakage.
    """
    window = np.asarray(window, dtype=np.float32).copy()
    if window.ndim != 2 or window.shape[1] % 4 != 0:
        raise ValueError("LOB window must have shape (time, 4 * levels)")

    price_idx = [idx for idx in range(window.shape[1]) if idx % 4 in (0, 2)]
    size_idx = [idx for idx in range(window.shape[1]) if idx % 4 in (1, 3)]
    mid = ((window[:, 0] + window[:, 2]) / 2.0).reshape(-1, 1)

    window[:, price_idx] = window[:, price_idx] / np.clip(mid, 1e-7, None) - 1.0
    if size_max is None:
        denominator = np.max(np.abs(window[:, size_idx]), axis=0, keepdims=True)
    else:
        denominator = np.asarray(size_max, dtype=np.float32).reshape(1, -1)
        if denominator.shape[1] != len(size_idx):
            raise ValueError("size_max length must match the number of LOB size columns")
    window[:, size_idx] = window[:, size_idx] / np.clip(denominator, 1.0, None)
    return np.nan_to_num(window, copy=False).astype(np.float32)


def normalize_lob_batch(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(dtype=torch.float32)


def _collect_split_samples(frame: pd.DataFrame, ticker: str, split: str, time_window: int) -> list[tuple[str, int]]:
    split_id = SPLIT_TO_ID[split]
    valid_indices = frame.index[
        (frame["split_id"] == split_id)
        & (frame["label"] >= 0)
        & (frame.index >= time_window - 1)
        & (frame["event_type"] != 7)
    ]
    return [(ticker, int(index)) for index in valid_indices]


def load_and_process_day(
    ticker: str,
    trading_date: str,
    data_root: str | Path,
    levels: int = 10,
    horizon: int = 20,
    time_window: int = 50,
    price_scale: float = 10_000.0,
    tick_size: float = 0.01,
    label_threshold: float = 1e-5,
    split_train: float = 0.6,
    split_val: float = 0.2,
    include_aux_features: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str], dict]:
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
    aux_feature_columns = (
        _compute_auxiliary_features(frame, levels=levels, time_window=time_window)
        if include_aux_features
        else []
    )
    metadata = {
        "ticker": ticker,
        "trading_date": trading_date,
        "rows": len(frame),
        "levels": levels,
        "horizon": horizon,
        "label_threshold": label_threshold,
        "time_window": time_window,
        "aux_feature_columns": aux_feature_columns,
        "include_aux_features": include_aux_features,
        "split_train": split_train,
        "split_val": split_val,
        "split_test": 1.0 - split_train - split_val,
    }
    return frame, feature_columns, aux_feature_columns, metadata


def create_datasets(
    tickers: Sequence[str],
    trading_date: str,
    data_root: str | Path,
    levels: int = 10,
    horizon: int = 20,
    time_window: int = 50,
    price_scale: float = 10_000.0,
    tick_size: float = 0.01,
    label_threshold: float = 1e-5,
    split_train: float = 0.6,
    split_val: float = 0.2,
    include_aux_features: bool = False,
) -> DatasetBundle:
    frames: Dict[str, pd.DataFrame] = {}
    train_samples: list[tuple[str, int]] = []
    val_samples: list[tuple[str, int]] = []
    test_samples: list[tuple[str, int]] = []
    all_metadata: Dict[str, dict] = {}
    feature_columns = _orderbook_columns(levels)
    aux_feature_columns: list[str] | None = None

    for ticker in tickers:
        frame, _, current_aux_feature_columns, metadata = load_and_process_day(
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
            include_aux_features=include_aux_features,
        )
        if aux_feature_columns is None:
            aux_feature_columns = current_aux_feature_columns
        elif aux_feature_columns != current_aux_feature_columns:
            raise ValueError("Auxiliary feature columns must match across tickers")
        frames[ticker] = frame
        all_metadata[ticker] = metadata
        train_samples.extend(_collect_split_samples(frame, ticker, "train", time_window))
        val_samples.extend(_collect_split_samples(frame, ticker, "val", time_window))
        test_samples.extend(_collect_split_samples(frame, ticker, "test", time_window))

    if aux_feature_columns is None:
        aux_feature_columns = []

    lob_size_max = {
        ticker: compute_lob_size_max(frame, levels=levels, mask=frame["split_id"] == SPLIT_TO_ID["train"])
        for ticker, frame in frames.items()
    }

    if aux_feature_columns:
        train_aux_frames = [frame.loc[frame["split_id"] == SPLIT_TO_ID["train"], aux_feature_columns] for frame in frames.values()]
        train_aux = pd.concat(train_aux_frames, axis=0, ignore_index=True)
        aux_mean = train_aux.mean(axis=0)
        aux_std = train_aux.std(axis=0).replace(0.0, 1.0).fillna(1.0)

        for frame in frames.values():
            frame.loc[:, aux_feature_columns] = ((frame.loc[:, aux_feature_columns] - aux_mean) / aux_std).astype(np.float32)
        aux_feature_stats = {
            "mean": {column: float(aux_mean[column]) for column in aux_feature_columns},
            "std": {column: float(aux_std[column]) for column in aux_feature_columns},
        }
    else:
        aux_feature_stats = {"mean": {}, "std": {}}

    metadata = {
        "tickers": list(tickers),
        "trading_date": trading_date,
        "levels": levels,
        "horizon": horizon,
        "label_threshold": label_threshold,
        "time_window": time_window,
        "aux_feature_columns": aux_feature_columns,
        "include_aux_features": include_aux_features,
        "aux_feature_stats": aux_feature_stats,
        "lob_normalization": {
            "price": "price / row_mid_price - 1",
            "size": "size / train_split_max_size_per_column",
            "size_max": {ticker: values.tolist() for ticker, values in lob_size_max.items()},
        },
        "split_train": split_train,
        "split_val": split_val,
        "split_test": 1.0 - split_train - split_val,
        "per_ticker": all_metadata,
    }
    return DatasetBundle(
        train=LobWindowDataset(frames, train_samples, feature_columns, aux_feature_columns, lob_size_max, time_window),
        val=LobWindowDataset(frames, val_samples, feature_columns, aux_feature_columns, lob_size_max, time_window),
        test=LobWindowDataset(frames, test_samples, feature_columns, aux_feature_columns, lob_size_max, time_window),
        feature_columns=feature_columns,
        aux_feature_columns=aux_feature_columns,
        metadata=metadata,
    )


def load_processed_frames(
    tickers: Sequence[str],
    processed_root: str | Path,
) -> Dict[str, pd.DataFrame]:
    root = Path(processed_root)
    frames: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame_path = root / ticker / "events.pkl.gz"
        if not frame_path.exists():
            raise FileNotFoundError(f"Processed frame not found for {ticker}: {frame_path}")
        frame = pd.read_pickle(frame_path, compression="gzip").reset_index(drop=True)
        required_columns = set(_orderbook_columns(10)) | {"label", "split_id"}
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            raise ValueError(f"Processed frame {frame_path} is missing required columns: {missing}")
        frames[ticker] = frame
    return frames


def _normalize_processed_aux_features(
    frames: Dict[str, pd.DataFrame],
    aux_feature_columns: list[str],
) -> dict:
    if not aux_feature_columns:
        return {"mean": {}, "std": {}}

    train_aux_frames = [
        frame.loc[frame["split_id"] == SPLIT_TO_ID["train"], aux_feature_columns]
        for frame in frames.values()
    ]
    train_aux = pd.concat(train_aux_frames, axis=0, ignore_index=True)
    aux_mean = train_aux.mean(axis=0)
    aux_std = train_aux.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    for frame in frames.values():
        frame.loc[:, aux_feature_columns] = ((frame.loc[:, aux_feature_columns] - aux_mean) / aux_std).astype(np.float32)
    return {
        "mean": {column: float(aux_mean[column]) for column in aux_feature_columns},
        "std": {column: float(aux_std[column]) for column in aux_feature_columns},
    }


def create_datasets_from_processed_root(
    tickers: Sequence[str],
    processed_root: str | Path,
    levels: int = 10,
    time_window: int = 50,
    include_aux_features: bool = False,
) -> DatasetBundle:
    frames = load_processed_frames(tickers, processed_root)
    feature_columns = _orderbook_columns(levels)
    missing_by_ticker = {
        ticker: sorted(set(feature_columns) - set(frame.columns))
        for ticker, frame in frames.items()
        if sorted(set(feature_columns) - set(frame.columns))
    }
    if missing_by_ticker:
        raise ValueError(f"Processed frames are missing orderbook columns: {missing_by_ticker}")

    aux_feature_columns = (
        [column for column in frames[next(iter(frames))].columns if column.startswith("market_state_")]
        if include_aux_features and frames
        else []
    )
    aux_feature_stats = _normalize_processed_aux_features(frames, aux_feature_columns)

    train_samples: list[tuple[str, int]] = []
    val_samples: list[tuple[str, int]] = []
    test_samples: list[tuple[str, int]] = []
    for ticker, frame in frames.items():
        train_samples.extend(_collect_split_samples(frame, ticker, "train", time_window))
        val_samples.extend(_collect_split_samples(frame, ticker, "val", time_window))
        test_samples.extend(_collect_split_samples(frame, ticker, "test", time_window))

    lob_size_max = {
        ticker: compute_lob_size_max(frame, levels=levels, mask=frame["split_id"] == SPLIT_TO_ID["train"])
        for ticker, frame in frames.items()
    }
    metadata = {
        "source": "processed_root",
        "processed_root": str(processed_root),
        "tickers": list(tickers),
        "levels": levels,
        "time_window": time_window,
        "aux_feature_columns": aux_feature_columns,
        "include_aux_features": include_aux_features,
        "aux_feature_stats": aux_feature_stats,
        "lob_normalization": {
            "price": "price / row_mid_price - 1",
            "size": "size / train_split_max_size_per_column",
            "size_max": {ticker: values.tolist() for ticker, values in lob_size_max.items()},
        },
        "split_samples": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
    }
    return DatasetBundle(
        train=LobWindowDataset(frames, train_samples, feature_columns, aux_feature_columns, lob_size_max, time_window),
        val=LobWindowDataset(frames, val_samples, feature_columns, aux_feature_columns, lob_size_max, time_window),
        test=LobWindowDataset(frames, test_samples, feature_columns, aux_feature_columns, lob_size_max, time_window),
        feature_columns=feature_columns,
        aux_feature_columns=aux_feature_columns,
        metadata=metadata,
    )
