from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from .reconstruct_sz_orderbook import (
        L2DateFiles,
        OrderBookReplay,
        complete_symbols,
        default_data_root,
        discover_date_files,
        hdf_key,
        read_symbol_frame,
        reconstruct_orderbook,
    )
except ImportError:
    from reconstruct_sz_orderbook import (
        L2DateFiles,
        OrderBookReplay,
        complete_symbols,
        default_data_root,
        discover_date_files,
        hdf_key,
        read_symbol_frame,
        reconstruct_orderbook,
    )


SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}
MARKET_STATE_COLUMNS = [f"market_state_{idx}" for idx in range(24)]
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


@dataclass(frozen=True)
class PrepareConfig:
    data_root: Path
    output_root: Path
    symbols: str | None = None
    num_symbols: int = 10
    candidate_limit: int = 60
    validation_stride: int = 5
    start_time: int = 93_000_000
    end_time: int = 145_700_000
    price_scale: float = 10_000.0
    depth: int = 10
    horizon: int = 10
    label_threshold: float = 1e-5
    min_match_rate: float = 0.95


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ordered_date_files(data_root: Path) -> list[L2DateFiles]:
    files = discover_date_files(data_root)
    return sorted(files, key=lambda item: item.date)


def split_dates(date_files: Sequence[L2DateFiles]) -> dict[str, list[str]]:
    if len(date_files) < 5:
        raise ValueError("The requested 3:1:1 split needs at least five trading dates.")
    train_count = int(len(date_files) * 3 / 5)
    val_count = int(len(date_files) * 1 / 5)
    train_count = max(train_count, 1)
    val_count = max(val_count, 1)
    if train_count + val_count >= len(date_files):
        train_count = max(len(date_files) - 2, 1)
        val_count = 1
    train_dates = [item.date for item in date_files[:train_count]]
    val_dates = [item.date for item in date_files[train_count : train_count + val_count]]
    test_dates = [item.date for item in date_files[train_count + val_count :]]
    return {"train": train_dates, "val": val_dates, "test": test_dates}


def common_sz_symbols(date_files: Sequence[L2DateFiles]) -> list[str]:
    common: set[str] | None = None
    for files in date_files:
        symbols = {symbol for symbol in complete_symbols(files) if symbol.endswith(".SZ")}
        common = symbols if common is None else common & symbols
    return sorted(common or set())


def storer_nrows(path: Path, symbol: str) -> int:
    with pd.HDFStore(path, mode="r") as store:
        return int(store.get_storer(hdf_key(symbol)).nrows)


def rank_by_completeness(date_files: Sequence[L2DateFiles], symbols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        market_rows: list[int] = []
        order_rows: list[int] = []
        trade_rows: list[int] = []
        for files in date_files:
            market_rows.append(storer_nrows(files.market_path, symbol))
            order_rows.append(storer_nrows(files.order_path, symbol))
            trade_rows.append(storer_nrows(files.trade_path, symbol))
        rows.append(
            {
                "symbol": symbol,
                "dates": len(date_files),
                "market_rows_total": int(np.sum(market_rows)),
                "order_rows_total": int(np.sum(order_rows)),
                "trade_rows_total": int(np.sum(trade_rows)),
                "min_market_rows": int(np.min(market_rows)),
                "min_order_rows": int(np.min(order_rows)),
                "min_trade_rows": int(np.min(trade_rows)),
                "completeness_score": int(np.sum(market_rows) + np.sum(order_rows) + np.sum(trade_rows)),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["dates", "min_market_rows", "min_trade_rows", "completeness_score", "symbol"],
            ascending=[False, False, False, False, True],
        )
        .reset_index(drop=True)
    )


def side_code(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.strip().str.upper().map({"B": 1, "S": -1}).fillna(0).astype(np.int8).to_numpy()


def trade_code_is_cancel(values: pd.Series, price_raw: np.ndarray) -> np.ndarray:
    code_cancel = values.astype(str).str.strip().str.upper().eq("C").to_numpy()
    return code_cancel | (price_raw <= 0)


def fast_event_tuples(order: pd.DataFrame, trade: pd.DataFrame) -> list[tuple]:
    order_time = pd.to_numeric(order["时间"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    order_event_id = pd.to_numeric(order["委托编号"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    order_id = pd.to_numeric(order["交易所委托号"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    order_side = side_code(order["委托代码"])
    order_price = pd.to_numeric(order["委托价格"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    order_quantity = pd.to_numeric(order["委托数量"], errors="coerce").fillna(0.0).astype(np.float64).to_numpy()

    trade_time = pd.to_numeric(trade["时间"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    trade_event_id = pd.to_numeric(trade["成交编号"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    trade_price = pd.to_numeric(trade["成交价格"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    trade_quantity = pd.to_numeric(trade["成交数量"], errors="coerce").fillna(0.0).astype(np.float64).to_numpy()
    is_cancel = trade_code_is_cancel(trade["成交代码"], trade_price)
    sell_order_id = pd.to_numeric(trade["叫卖序号"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    buy_order_id = pd.to_numeric(trade["叫买序号"], errors="coerce").fillna(0).astype(np.int64).to_numpy()

    events: list[tuple] = []
    events.extend(
        zip(
            order_time,
            np.zeros(len(order), dtype=np.int8),
            order_event_id,
            np.zeros(len(order), dtype=np.int8),
            order_id,
            order_side,
            order_price,
            order_quantity,
            np.zeros(len(order), dtype=bool),
            np.zeros(len(order), dtype=np.int8),
            np.zeros(len(order), dtype=np.int64),
            np.zeros(len(order), dtype=np.int64),
            strict=False,
        )
    )
    events.extend(
        zip(
            trade_time,
            np.ones(len(trade), dtype=np.int8),
            trade_event_id,
            np.ones(len(trade), dtype=np.int8),
            np.zeros(len(trade), dtype=np.int64),
            np.zeros(len(trade), dtype=np.int8),
            trade_price,
            trade_quantity,
            is_cancel,
            np.zeros(len(trade), dtype=np.int8),
            sell_order_id,
            buy_order_id,
            strict=False,
        )
    )
    events.sort(key=lambda item: (int(item[0]), int(item[1]), int(item[2])))
    return events


def fast_validate_symbol_day(
    order: pd.DataFrame,
    trade: pd.DataFrame,
    market: pd.DataFrame,
    start_time: int,
    end_time: int,
    snapshot_stride: int,
) -> dict[str, float]:
    events = fast_event_tuples(order, trade)
    book = OrderBookReplay()
    market_slice = market.loc[(market["时间"] >= start_time) & (market["时间"] <= end_time)].sort_values("时间")
    if snapshot_stride > 1:
        market_slice = market_slice.iloc[::snapshot_stride]
    market_times = pd.to_numeric(market_slice["时间"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    market_ask_1 = pd.to_numeric(market_slice["申卖价1"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    market_bid_1 = pd.to_numeric(market_slice["申买价1"], errors="coerce").fillna(0).astype(np.int64).to_numpy()

    position = 0
    ask_matches = 0
    bid_matches = 0
    both_matches = 0
    for snapshot_time, expected_ask, expected_bid in zip(market_times, market_ask_1, market_bid_1, strict=False):
        while position < len(events) and int(events[position][0]) <= int(snapshot_time):
            (
                _raw_time,
                _event_sort,
                _event_id,
                source,
                order_id,
                side,
                price_raw,
                quantity,
                is_cancel,
                _bs_flag,
                sell_order_id,
                buy_order_id,
            ) = events[position]
            position += 1
            quantity = float(quantity)
            if int(source) == 0:
                side_value = "B" if int(side) == 1 else "S" if int(side) == -1 else ""
                book.add_order(int(order_id), side_value, int(price_raw), quantity)
            elif bool(is_cancel):
                found = False
                for candidate_order_id in (int(sell_order_id), int(buy_order_id)):
                    found, *_ = book.reduce_order(candidate_order_id, quantity)
                    if found:
                        break
            else:
                book.reduce_order(int(sell_order_id), quantity)
                book.reduce_order(int(buy_order_id), quantity)

        top = book.top_levels_raw(depth=1)[0]
        replay_ask = top[0]
        replay_bid = top[2]
        ask_match = replay_ask == int(expected_ask)
        bid_match = replay_bid == int(expected_bid)
        ask_matches += int(ask_match)
        bid_matches += int(bid_match)
        both_matches += int(ask_match and bid_match)

    snapshots = len(market_times)
    return {
        "snapshots": float(snapshots),
        "events_consumed": float(position),
        "missing_references": float(book.missing_references),
        "top1_ask_price_match_rate": ask_matches / max(snapshots, 1),
        "top1_bid_price_match_rate": bid_matches / max(snapshots, 1),
        "top1_both_price_match_rate": both_matches / max(snapshots, 1),
    }


def score_candidates(
    config: PrepareConfig,
    date_files: Sequence[L2DateFiles],
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rank, record in enumerate(candidates.to_dict("records"), start=1):
        symbol = str(record["symbol"])
        daily_metrics: list[dict[str, float]] = []
        for files in date_files:
            order = read_symbol_frame(files.order_path, symbol)
            trade = read_symbol_frame(files.trade_path, symbol)
            market = read_symbol_frame(files.market_path, symbol)
            daily_metrics.append(
                fast_validate_symbol_day(
                    order=order,
                    trade=trade,
                    market=market,
                    start_time=config.start_time,
                    end_time=config.end_time,
                    snapshot_stride=config.validation_stride,
                )
            )
        score_row = dict(record)
        score_row.update(
            {
                "candidate_rank": rank,
                "mean_top1_ask_match": float(np.mean([item["top1_ask_price_match_rate"] for item in daily_metrics])),
                "mean_top1_bid_match": float(np.mean([item["top1_bid_price_match_rate"] for item in daily_metrics])),
                "mean_top1_both_match": float(np.mean([item["top1_both_price_match_rate"] for item in daily_metrics])),
                "min_top1_both_match": float(np.min([item["top1_both_price_match_rate"] for item in daily_metrics])),
                "total_validation_snapshots": int(np.sum([item["snapshots"] for item in daily_metrics])),
                "total_events_consumed": int(np.sum([item["events_consumed"] for item in daily_metrics])),
                "total_missing_references": int(np.sum([item["missing_references"] for item in daily_metrics])),
            }
        )
        rows.append(score_row)
        print(
            f"scored {rank}/{len(candidates)} {symbol}: "
            f"mean_both={score_row['mean_top1_both_match']:.2%} "
            f"min_both={score_row['min_top1_both_match']:.2%} "
            f"rows={score_row['completeness_score']}",
            flush=True,
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "mean_top1_both_match",
                "min_top1_both_match",
                "min_market_rows",
                "min_trade_rows",
                "completeness_score",
                "symbol",
            ],
            ascending=[False, False, False, False, False, True],
        )
        .reset_index(drop=True)
    )


def compute_labels(mid_price: np.ndarray, horizon: int, threshold: float) -> np.ndarray:
    mid_series = pd.Series(mid_price, dtype=np.float64)
    price_past = mid_series.rolling(window=horizon).mean()
    price_future = price_past.shift(-horizon)
    pct_change = (price_future - price_past) / price_past
    labels = np.full(len(mid_series), -99, dtype=np.int64)
    valid_mask = pct_change.notna().to_numpy() & np.isfinite(pct_change.to_numpy()) & (price_past.to_numpy() > 0)
    pct_values = pct_change.to_numpy(dtype=np.float64)
    labels[valid_mask & (pct_values >= threshold)] = 2
    labels[valid_mask & (pct_values < threshold) & (pct_values > -threshold)] = 1
    labels[valid_mask & (pct_values <= -threshold)] = 0
    return labels


def window_search_indices(times: np.ndarray, window_seconds: float) -> np.ndarray:
    return np.searchsorted(times, times - window_seconds, side="left")


def imbalance_features(
    times: np.ndarray,
    buy_volume: np.ndarray,
    sell_volume: np.ndarray,
    buy_count: np.ndarray,
    sell_count: np.ndarray,
    windows: Sequence[float],
) -> list[np.ndarray]:
    def cumulative(values: np.ndarray) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))

    buy_vol_c = cumulative(buy_volume)
    sell_vol_c = cumulative(sell_volume)
    buy_cnt_c = cumulative(buy_count)
    sell_cnt_c = cumulative(sell_count)
    idx = np.arange(len(times)) + 1
    features: list[np.ndarray] = []
    for window in windows:
        start = window_search_indices(times, window)
        bv = buy_vol_c[idx] - buy_vol_c[start]
        sv = sell_vol_c[idx] - sell_vol_c[start]
        bc = buy_cnt_c[idx] - buy_cnt_c[start]
        sc = sell_cnt_c[idx] - sell_cnt_c[start]
        features.append((bv - sv) / (bv + sv + 1e-8))
        features.append((bc - sc) / (bc + sc + 1e-8))
    return features


def compute_dynamic_features(frame: pd.DataFrame) -> pd.DataFrame:
    times = frame["time"].to_numpy(dtype=np.float64)
    execution_buy = frame["execution_buy_volume"].to_numpy(dtype=np.float64)
    execution_sell = frame["execution_sell_volume"].to_numpy(dtype=np.float64)
    order_buy = frame["order_buy_volume"].to_numpy(dtype=np.float64)
    order_sell = frame["order_sell_volume"].to_numpy(dtype=np.float64)
    cancel_buy = frame["cancel_buy_volume"].to_numpy(dtype=np.float64)
    cancel_sell = frame["cancel_sell_volume"].to_numpy(dtype=np.float64)

    feature_arrays: list[np.ndarray] = []
    for buy_volume, sell_volume in [
        (execution_buy, execution_sell),
        (order_buy, order_sell),
        (cancel_buy, cancel_sell),
    ]:
        feature_arrays.extend(
            imbalance_features(
                times=times,
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                buy_count=(buy_volume > 0).astype(np.float64),
                sell_count=(sell_volume > 0).astype(np.float64),
                windows=[10.0, 60.0, 300.0],
            )
        )

    mid = frame["mid_price"].to_numpy(dtype=np.float64)
    log_mid = np.log(np.clip(mid, 1e-8, None))
    log_ret_sq = np.zeros(len(frame), dtype=np.float64)
    log_ret_sq[1:] = np.diff(log_mid) ** 2
    diff_mid = np.zeros(len(frame), dtype=np.float64)
    diff_mid[1:] = np.diff(mid)
    gains = np.maximum(diff_mid, 0.0)
    losses = np.maximum(-diff_mid, 0.0)
    ret_c = np.concatenate(([0.0], np.cumsum(log_ret_sq)))
    gain_c = np.concatenate(([0.0], np.cumsum(gains)))
    loss_c = np.concatenate(([0.0], np.cumsum(losses)))
    idx = np.arange(len(frame)) + 1
    for window in [300.0, 600.0, 1800.0]:
        start = window_search_indices(times, window)
        rv = np.sqrt(np.maximum(ret_c[idx] - ret_c[start], 0.0))
        gain_sum = gain_c[idx] - gain_c[start]
        loss_sum = loss_c[idx] - loss_c[start]
        rsi = gain_sum / (gain_sum + loss_sum + 1e-8)
        feature_arrays.append(rv)
        feature_arrays.append(rsi)

    values = np.column_stack(feature_arrays).astype(np.float32)
    return pd.DataFrame(values, columns=MARKET_STATE_COLUMNS, index=frame.index)


def finalize_day_frame(frame: pd.DataFrame, symbol: str, date: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["ticker"] = symbol
    frame["trading_date"] = date
    frame["is_execution"] = (frame["execution_events_in_group"] > 0).astype(np.int8)
    frame["trade_price"] = np.where(frame["is_execution"] == 1, frame["last_trade_price"], np.nan)
    frame["trade_size"] = np.where(frame["is_execution"] == 1, frame["execution_volume"], 0.0)
    valid = (
        np.isfinite(frame["mid_price"])
        & np.isfinite(frame["spread"])
        & (frame["ask_price_1"] > 0)
        & (frame["bid_price_1"] > 0)
        & (frame["spread"] >= 0)
    )
    frame = frame.loc[valid].reset_index(drop=True)
    dynamic = compute_dynamic_features(frame)
    frame = pd.concat([frame, dynamic], axis=1)
    return frame


def prepare_symbol(
    config: PrepareConfig,
    date_files: Sequence[L2DateFiles],
    symbol: str,
    splits: dict[str, list[str]],
) -> dict[str, object]:
    day_frames: list[pd.DataFrame] = []
    replay_stats: list[dict[str, object]] = []
    for files in date_files:
        order = read_symbol_frame(files.order_path, symbol)
        trade = read_symbol_frame(files.trade_path, symbol)
        result = reconstruct_orderbook(
            order=order,
            trade=trade,
            price_scale=config.price_scale,
            depth=config.depth,
            start_time=config.start_time,
            end_time=config.end_time,
            emit="time",
        )
        day_frame = finalize_day_frame(result.frame, symbol=symbol, date=files.date)
        day_frames.append(day_frame)
        replay_stats.append(
            {
                "date": files.date,
                "events_seen": result.events_seen,
                "rows": len(day_frame),
                "missing_references": result.missing_references,
                "duplicate_orders": result.duplicate_orders,
            }
        )

    frame = pd.concat(day_frames, axis=0, ignore_index=True)
    date_to_split = {
        date: SPLIT_TO_ID[split_name]
        for split_name, dates in splits.items()
        for date in dates
    }
    frame["split_id"] = frame["trading_date"].map(date_to_split).astype(np.int8)
    frame["label"] = compute_labels(
        frame["mid_price"].to_numpy(dtype=np.float64),
        horizon=config.horizon,
        threshold=config.label_threshold,
    )

    output_dir = config.output_root / symbol
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / "events.pkl.gz"
    frame.to_pickle(frame_path, compression="gzip")
    metadata = {
        "symbol": symbol,
        "rows": len(frame),
        "frame_path": str(frame_path),
        "dates": [files.date for files in date_files],
        "splits": splits,
        "split_counts": {
            split_name: int((frame["split_id"] == split_id).sum())
            for split_name, split_id in SPLIT_TO_ID.items()
        },
        "valid_label_counts": {
            split_name: int(((frame["split_id"] == split_id) & (frame["label"] >= 0)).sum())
            for split_name, split_id in SPLIT_TO_ID.items()
        },
        "horizon": config.horizon,
        "label_threshold": config.label_threshold,
        "price_scale": config.price_scale,
        "start_time": config.start_time,
        "end_time": config.end_time,
        "replay_stats": replay_stats,
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def prepare_dataset(config: PrepareConfig) -> dict[str, object]:
    date_files = ordered_date_files(config.data_root)
    splits = split_dates(date_files)
    common_symbols = common_sz_symbols(date_files)
    completeness = rank_by_completeness(date_files, common_symbols)

    config.output_root.mkdir(parents=True, exist_ok=True)
    completeness.to_csv(config.output_root / "sz_completeness.csv", index=False)

    if config.symbols:
        selected_symbols = [symbol.strip().upper() for symbol in config.symbols.split(",") if symbol.strip()]
        missing = sorted(set(selected_symbols) - set(common_symbols))
        if missing:
            raise ValueError(f"Explicit symbols are not complete across all dates: {missing}")
        scores = score_candidates(config, date_files, completeness.loc[completeness["symbol"].isin(selected_symbols)])
        scores.to_csv(config.output_root / "selection_scores.csv", index=False)
        selected_scores = (
            scores.set_index("symbol")
            .loc[selected_symbols]
            .reset_index()
        )
    else:
        candidate_limit = len(completeness) if config.candidate_limit <= 0 else min(config.candidate_limit, len(completeness))
        candidates = completeness.head(candidate_limit)
        scores = score_candidates(config, date_files, candidates)
        scores.to_csv(config.output_root / "selection_scores.csv", index=False)
        eligible_scores = scores.loc[scores["mean_top1_both_match"] >= config.min_match_rate]
        if len(eligible_scores) < config.num_symbols:
            print(
                f"Only {len(eligible_scores)} candidates reached min_match_rate={config.min_match_rate:.2%}; "
                "filling from the best remaining scores."
            )
            selected_scores = scores.head(config.num_symbols)
        else:
            selected_scores = eligible_scores.head(config.num_symbols)
        selected_symbols = selected_scores["symbol"].tolist()

    metadata_by_symbol = []
    for position, symbol in enumerate(selected_symbols, start=1):
        print(f"preparing {position}/{len(selected_symbols)} {symbol}", flush=True)
        metadata_by_symbol.append(prepare_symbol(config, date_files, symbol, splits))

    manifest = {
        "config": {
            **asdict(config),
            "data_root": str(config.data_root),
            "output_root": str(config.output_root),
        },
        "dates": [files.date for files in date_files],
        "splits": splits,
        "common_sz_symbols": len(common_symbols),
        "candidate_limit": None if config.symbols else candidate_limit,
        "selected_symbols": selected_symbols,
        "selection_scores_path": str(config.output_root / "selection_scores.csv"),
        "completeness_path": str(config.output_root / "sz_completeness.csv"),
        "symbols": metadata_by_symbol,
    }
    write_json(config.output_root / "manifest.json", manifest)
    selected_scores.to_csv(config.output_root / "selected_symbols.csv", index=False)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select and prepare reconstructed SZ L2 data for Attn-LOB/RL.")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--output-root", type=Path, default=Path("Our-Code/data/sz_l2_processed"))
    parser.add_argument(
        "--symbols",
        default=None,
        help="Explicit comma-separated SZ symbol list. When provided, selection uses this exact order.",
    )
    parser.add_argument("--num-symbols", type=int, default=10)
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=60,
        help="Validate this many most-complete common SZ symbols before selecting. Use 0 to validate all common SZ symbols.",
    )
    parser.add_argument("--validation-stride", type=int, default=5)
    parser.add_argument("--start-time", type=int, default=93_000_000)
    parser.add_argument("--end-time", type=int, default=145_700_000)
    parser.add_argument("--price-scale", type=float, default=10_000.0)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--label-threshold", type=float, default=1e-5)
    parser.add_argument("--min-match-rate", type=float, default=0.95)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = PrepareConfig(**vars(args))
    manifest = prepare_dataset(config)
    print()
    print("prepared symbols:")
    for symbol in manifest["selected_symbols"]:
        print(f"- {symbol}")
    print(f"manifest: {config.output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
