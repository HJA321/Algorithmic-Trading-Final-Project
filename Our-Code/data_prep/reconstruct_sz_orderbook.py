from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


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

SENTINEL_ASK_RAW = 10_000_000_000
SENTINEL_BID_RAW = -10_000_000_000


@dataclass(frozen=True)
class L2DateFiles:
    date: str
    directory: Path
    market_path: Path
    order_path: Path
    trade_path: Path


@dataclass
class ReplayResult:
    frame: pd.DataFrame
    events_seen: int
    events_written: int
    missing_references: int
    duplicate_orders: int


class OrderBookReplay:
    def __init__(self) -> None:
        self.orders: dict[int, tuple[str, int, float]] = {}
        self.levels: dict[str, defaultdict[int, float]] = {
            "B": defaultdict(float),
            "S": defaultdict(float),
        }
        self.missing_references = 0
        self.duplicate_orders = 0

    def add_order(self, order_id: int, side: str, price_raw: int, quantity: float) -> bool:
        if order_id <= 0 or side not in {"B", "S"} or price_raw <= 0 or quantity <= 0:
            return False
        if order_id in self.orders:
            self.duplicate_orders += 1
            self._remove_existing_order(order_id)
        self.orders[order_id] = (side, price_raw, float(quantity))
        self.levels[side][price_raw] += float(quantity)
        return True

    def reduce_order(self, order_id: int, quantity: float) -> tuple[bool, str, int, float, bool]:
        if order_id <= 0 or quantity <= 0:
            self.missing_references += 1
            return False, "", 0, 0.0, False
        if order_id not in self.orders:
            self.missing_references += 1
            return False, "", 0, 0.0, False

        side, price_raw, old_quantity = self.orders[order_id]
        reduction = min(float(quantity), old_quantity)
        new_quantity = old_quantity - reduction
        self.levels[side][price_raw] -= reduction
        if self.levels[side][price_raw] <= 1e-9:
            del self.levels[side][price_raw]

        removed = new_quantity <= 1e-9
        if removed:
            del self.orders[order_id]
        else:
            self.orders[order_id] = (side, price_raw, new_quantity)
        return True, side, price_raw, reduction, removed

    def top_levels_raw(self, depth: int = 10) -> list[tuple[int, float, int, float]]:
        asks = sorted(self.levels["S"].items())[:depth]
        bids = sorted(self.levels["B"].items(), reverse=True)[:depth]
        rows: list[tuple[int, float, int, float]] = []
        for idx in range(depth):
            ask_price, ask_size = asks[idx] if idx < len(asks) else (SENTINEL_ASK_RAW, 0.0)
            bid_price, bid_size = bids[idx] if idx < len(bids) else (SENTINEL_BID_RAW, 0.0)
            rows.append((int(ask_price), float(ask_size), int(bid_price), float(bid_size)))
        return rows

    def snapshot_columns(self, price_scale: float, depth: int = 10) -> dict[str, float]:
        snapshot: dict[str, float] = {}
        for level, (ask_price, ask_size, bid_price, bid_size) in enumerate(self.top_levels_raw(depth), start=1):
            snapshot[f"ask_price_{level}"] = ask_price / price_scale
            snapshot[f"ask_size_{level}"] = ask_size
            snapshot[f"bid_price_{level}"] = bid_price / price_scale
            snapshot[f"bid_size_{level}"] = bid_size
        ask_1 = snapshot["ask_price_1"]
        bid_1 = snapshot["bid_price_1"]
        if ask_1 > 0 and bid_1 > 0 and ask_1 < SENTINEL_ASK_RAW / price_scale and bid_1 > SENTINEL_BID_RAW / price_scale:
            snapshot["mid_price"] = (ask_1 + bid_1) / 2.0
            snapshot["spread"] = ask_1 - bid_1
        else:
            snapshot["mid_price"] = np.nan
            snapshot["spread"] = np.nan
        return snapshot

    def _remove_existing_order(self, order_id: int) -> None:
        side, price_raw, quantity = self.orders.pop(order_id)
        self.levels[side][price_raw] -= quantity
        if self.levels[side][price_raw] <= 1e-9:
            del self.levels[side][price_raw]


def default_data_root() -> Path:
    return Path(__file__).resolve().parents[2] / "l2_zip"


def discover_date_files(data_root: Path) -> list[L2DateFiles]:
    date_files: list[L2DateFiles] = []
    for directory in sorted(data_root.glob("*_hdf")):
        date = directory.name.split("_", 1)[0]
        market_path = directory / f"market_{date}.h5"
        order_path = directory / f"order_{date}.h5"
        trade_path = directory / f"trade_{date}.h5"
        if market_path.exists() and order_path.exists() and trade_path.exists():
            date_files.append(
                L2DateFiles(
                    date=date,
                    directory=directory,
                    market_path=market_path,
                    order_path=order_path,
                    trade_path=trade_path,
                )
            )
    if not date_files:
        raise FileNotFoundError(f"No complete *_hdf date directories found under {data_root}")
    return date_files


def hdf_symbols(path: Path) -> set[str]:
    with pd.HDFStore(path, mode="r") as store:
        return {key.strip("/") for key in store.keys()}


def hdf_key(symbol: str) -> str:
    symbol = symbol.strip().strip("/")
    return f"/{symbol}"


def read_symbol_frame(path: Path, symbol: str) -> pd.DataFrame:
    return pd.read_hdf(path, key=hdf_key(symbol))


def complete_symbols(files: L2DateFiles) -> set[str]:
    return hdf_symbols(files.market_path) & hdf_symbols(files.order_path) & hdf_symbols(files.trade_path)


def summarize_universe(data_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    common_all: set[str] | None = None
    common_sz: set[str] | None = None
    common_sh: set[str] | None = None
    union_all: set[str] = set()
    union_sz: set[str] = set()
    union_sh: set[str] = set()

    for files in discover_date_files(data_root):
        symbols = complete_symbols(files)
        sz_symbols = {symbol for symbol in symbols if symbol.endswith(".SZ")}
        sh_symbols = {symbol for symbol in symbols if symbol.endswith(".SH")}
        rows.append(
            {
                "date": files.date,
                "complete_symbols": len(symbols),
                "sz_symbols": len(sz_symbols),
                "sh_symbols": len(sh_symbols),
                "sample_sz": ",".join(sorted(sz_symbols)[:5]),
            }
        )

        union_all |= symbols
        union_sz |= sz_symbols
        union_sh |= sh_symbols
        common_all = symbols if common_all is None else common_all & symbols
        common_sz = sz_symbols if common_sz is None else common_sz & sz_symbols
        common_sh = sh_symbols if common_sh is None else common_sh & sh_symbols

    summary = pd.DataFrame(rows)
    summary.attrs["num_dates"] = len(rows)
    summary.attrs["union_all"] = len(union_all)
    summary.attrs["union_sz"] = len(union_sz)
    summary.attrs["union_sh"] = len(union_sh)
    summary.attrs["common_all"] = len(common_all or set())
    summary.attrs["common_sz"] = len(common_sz or set())
    summary.attrs["common_sh"] = len(common_sh or set())
    summary.attrs["common_sz_sample"] = sorted(common_sz or set())[:10]
    return summary


def parse_l2_time_to_seconds(time_value: int | float) -> float:
    value = int(time_value)
    hour = value // 10_000_000
    minute = (value // 100_000) % 100
    second = (value // 1_000) % 100
    millisecond = value % 1_000
    return hour * 3600.0 + minute * 60.0 + second + millisecond / 1_000.0


def normalize_side(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def raw_int(value: object, default: int = 0) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    return int(numeric)


def raw_float(value: object, default: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    return float(numeric)


def side_to_direction(side: str) -> int:
    if side == "B":
        return 1
    if side == "S":
        return -1
    return 0


def execution_direction_from_bs_flag(bs_flag: str) -> int:
    if bs_flag == "B":
        return -1
    if bs_flag == "S":
        return 1
    return 0


def build_event_stream(order: pd.DataFrame, trade: pd.DataFrame) -> pd.DataFrame:
    order_events = pd.DataFrame(
        {
            "raw_time": pd.to_numeric(order["时间"], errors="coerce").fillna(0).astype("int64"),
            "event_sort": 0,
            "event_id": pd.to_numeric(order["委托编号"], errors="coerce").fillna(0).astype("int64"),
            "source": "order",
            "order_id": pd.to_numeric(order["交易所委托号"], errors="coerce").fillna(0).astype("int64"),
            "side": order["委托代码"].map(normalize_side),
            "price_raw": pd.to_numeric(order["委托价格"], errors="coerce").fillna(0).astype("int64"),
            "quantity": pd.to_numeric(order["委托数量"], errors="coerce").fillna(0.0),
        }
    )
    trade_events = pd.DataFrame(
        {
            "raw_time": pd.to_numeric(trade["时间"], errors="coerce").fillna(0).astype("int64"),
            "event_sort": 1,
            "event_id": pd.to_numeric(trade["成交编号"], errors="coerce").fillna(0).astype("int64"),
            "source": "trade",
            "trade_code": trade["成交代码"].map(normalize_side),
            "bs_flag": trade["BS标志"].map(normalize_side),
            "price_raw": pd.to_numeric(trade["成交价格"], errors="coerce").fillna(0).astype("int64"),
            "quantity": pd.to_numeric(trade["成交数量"], errors="coerce").fillna(0.0),
            "sell_order_id": pd.to_numeric(trade["叫卖序号"], errors="coerce").fillna(0).astype("int64"),
            "buy_order_id": pd.to_numeric(trade["叫买序号"], errors="coerce").fillna(0).astype("int64"),
        }
    )
    events = pd.concat([order_events, trade_events], axis=0, ignore_index=True, sort=False)
    return events.sort_values(["raw_time", "event_sort", "event_id"], kind="mergesort").reset_index(drop=True)


def handle_event(book: OrderBookReplay, event: pd.Series, price_scale: float) -> dict[str, object]:
    raw_time = int(event["raw_time"])
    source = str(event["source"])
    event_row: dict[str, object] = {
        "time": parse_l2_time_to_seconds(raw_time),
        "raw_time": raw_time,
        "event_type": 0,
        "order_id": 0,
        "size": 0.0,
        "price": 0.0,
        "direction": 0,
        "source": source,
        "source_event_id": int(event["event_id"]),
        "bs_flag": "",
        "buy_order_id": 0,
        "sell_order_id": 0,
        "match_status": "",
    }

    if source == "order":
        order_id = raw_int(event.get("order_id"))
        side = normalize_side(event.get("side"))
        price_raw = raw_int(event.get("price_raw"))
        quantity = raw_float(event.get("quantity"))
        added = book.add_order(order_id, side, price_raw, quantity)
        event_row.update(
            {
                "event_type": 1 if added else 0,
                "order_id": order_id,
                "size": quantity,
                "price": price_raw / price_scale,
                "direction": side_to_direction(side),
                "match_status": "added" if added else "ignored_order",
            }
        )
        return event_row

    trade_code = normalize_side(event.get("trade_code"))
    bs_flag = normalize_side(event.get("bs_flag"))
    price_raw = raw_int(event.get("price_raw"))
    quantity = raw_float(event.get("quantity"))
    sell_order_id = raw_int(event.get("sell_order_id"))
    buy_order_id = raw_int(event.get("buy_order_id"))
    is_cancel = trade_code == "C" or price_raw <= 0

    event_row.update(
        {
            "size": quantity,
            "price": price_raw / price_scale,
            "bs_flag": bs_flag,
            "buy_order_id": buy_order_id,
            "sell_order_id": sell_order_id,
        }
    )

    if is_cancel:
        found = False
        for order_id in (sell_order_id, buy_order_id):
            found, side, cancelled_price_raw, reduced, removed = book.reduce_order(order_id, quantity)
            if found:
                event_row.update(
                    {
                        "event_type": 3 if removed else 2,
                        "order_id": order_id,
                        "size": reduced,
                        "price": cancelled_price_raw / price_scale,
                        "direction": side_to_direction(side),
                        "match_status": "cancelled",
                    }
                )
                break
        if not found:
            event_row.update({"event_type": 2, "match_status": "missing_cancel_reference"})
        return event_row

    sell_found, sell_side, sell_price_raw, sell_reduced, _ = book.reduce_order(sell_order_id, quantity)
    buy_found, buy_side, buy_price_raw, buy_reduced, _ = book.reduce_order(buy_order_id, quantity)
    if bs_flag == "B":
        resting_order_id = sell_order_id
        resting_price_raw = sell_price_raw if sell_found else price_raw
        resting_size = sell_reduced if sell_found else quantity
    elif bs_flag == "S":
        resting_order_id = buy_order_id
        resting_price_raw = buy_price_raw if buy_found else price_raw
        resting_size = buy_reduced if buy_found else quantity
    else:
        resting_order_id = sell_order_id if sell_found else buy_order_id
        resting_price_raw = sell_price_raw if sell_found else buy_price_raw if buy_found else price_raw
        resting_size = sell_reduced if sell_found else buy_reduced if buy_found else quantity

    status_parts = []
    status_parts.append("sell_ref_ok" if sell_found else "sell_ref_missing")
    status_parts.append("buy_ref_ok" if buy_found else "buy_ref_missing")
    event_row.update(
        {
            "event_type": 4,
            "order_id": resting_order_id,
            "size": resting_size,
            "price": resting_price_raw / price_scale,
            "direction": execution_direction_from_bs_flag(bs_flag),
            "match_status": ";".join(status_parts),
        }
    )
    return event_row


def reconstruct_orderbook(
    order: pd.DataFrame,
    trade: pd.DataFrame,
    price_scale: float = 10_000.0,
    depth: int = 10,
    start_time: int | None = None,
    end_time: int | None = None,
    max_events: int | None = None,
    emit: str = "time",
) -> ReplayResult:
    events = build_event_stream(order, trade)
    book = OrderBookReplay()
    rows: list[dict[str, object]] = []
    events_seen = 0

    if emit not in {"time", "event"}:
        raise ValueError("emit must be either 'time' or 'event'")

    if emit == "time":
        for raw_time_value, group in events.groupby("raw_time", sort=True):
            raw_time_value = int(raw_time_value)
            if end_time is not None and raw_time_value > end_time:
                break

            group_stats = {
                "events_in_group": int(len(group)),
                "order_events_in_group": 0,
                "trade_events_in_group": 0,
                "execution_events_in_group": 0,
                "cancel_events_in_group": 0,
                "order_volume": 0.0,
                "order_buy_volume": 0.0,
                "order_sell_volume": 0.0,
                "cancel_volume": 0.0,
                "cancel_buy_volume": 0.0,
                "cancel_sell_volume": 0.0,
                "execution_volume": 0.0,
                "execution_buy_volume": 0.0,
                "execution_sell_volume": 0.0,
                "last_trade_price": 0.0,
            }
            last_event_row: dict[str, object] | None = None
            for _, event in group.iterrows():
                last_event_row = handle_event(book, event, price_scale=price_scale)
                events_seen += 1
                if str(event["source"]) == "order":
                    group_stats["order_events_in_group"] += 1
                    side = normalize_side(event.get("side"))
                    quantity = raw_float(event.get("quantity"))
                    group_stats["order_volume"] += quantity
                    if side == "B":
                        group_stats["order_buy_volume"] += quantity
                    elif side == "S":
                        group_stats["order_sell_volume"] += quantity
                else:
                    group_stats["trade_events_in_group"] += 1
                    trade_code = normalize_side(event.get("trade_code"))
                    price_raw = raw_int(event.get("price_raw"))
                    quantity = raw_float(event.get("quantity"))
                    if trade_code == "C" or price_raw <= 0:
                        group_stats["cancel_events_in_group"] += 1
                        group_stats["cancel_volume"] += quantity
                        if last_event_row["direction"] == 1:
                            group_stats["cancel_buy_volume"] += quantity
                        elif last_event_row["direction"] == -1:
                            group_stats["cancel_sell_volume"] += quantity
                    else:
                        group_stats["execution_events_in_group"] += 1
                        group_stats["execution_volume"] += quantity
                        group_stats["last_trade_price"] = price_raw / price_scale
                        bs_flag = normalize_side(event.get("bs_flag"))
                        if bs_flag == "B":
                            group_stats["execution_buy_volume"] += quantity
                        elif bs_flag == "S":
                            group_stats["execution_sell_volume"] += quantity

            if start_time is not None and raw_time_value < start_time:
                continue

            if group_stats["execution_volume"] > 0:
                event_type = 4
                size = group_stats["execution_volume"]
                price = group_stats["last_trade_price"]
                direction = -1 if group_stats["execution_buy_volume"] >= group_stats["execution_sell_volume"] else 1
            elif group_stats["cancel_volume"] > 0:
                event_type = 2
                size = group_stats["cancel_volume"]
                price = 0.0
                direction = 1 if group_stats["cancel_buy_volume"] >= group_stats["cancel_sell_volume"] else -1
            elif group_stats["order_volume"] > 0:
                event_type = 1
                size = group_stats["order_volume"]
                price = 0.0
                direction = 1 if group_stats["order_buy_volume"] >= group_stats["order_sell_volume"] else -1
            else:
                event_type = 0
                size = 0.0
                price = 0.0
                direction = 0

            row = {
                "time": parse_l2_time_to_seconds(raw_time_value),
                "raw_time": raw_time_value,
                "event_type": event_type,
                "order_id": 0,
                "size": size,
                "price": price,
                "direction": direction,
                "source": "time_group",
                "source_event_id": 0 if last_event_row is None else last_event_row["source_event_id"],
                "bs_flag": "" if last_event_row is None else last_event_row["bs_flag"],
                "buy_order_id": 0,
                "sell_order_id": 0,
                "match_status": "" if last_event_row is None else last_event_row["match_status"],
            }
            row.update(group_stats)
            row.update(book.snapshot_columns(price_scale=price_scale, depth=depth))
            rows.append(row)
            if max_events is not None and len(rows) >= max_events:
                break

        return ReplayResult(
            frame=pd.DataFrame(rows),
            events_seen=events_seen,
            events_written=len(rows),
            missing_references=book.missing_references,
            duplicate_orders=book.duplicate_orders,
        )

    for _, event in events.iterrows():
        raw_time_value = int(event["raw_time"])
        if end_time is not None and raw_time_value > end_time:
            break

        event_row = handle_event(book, event, price_scale=price_scale)
        events_seen += 1
        if start_time is not None and raw_time_value < start_time:
            continue

        event_row.update(book.snapshot_columns(price_scale=price_scale, depth=depth))
        rows.append(event_row)
        if max_events is not None and len(rows) >= max_events:
            break

    return ReplayResult(
        frame=pd.DataFrame(rows),
        events_seen=events_seen,
        events_written=len(rows),
        missing_references=book.missing_references,
        duplicate_orders=book.duplicate_orders,
    )


def market_level_raw(row: pd.Series, side: str, level: int) -> tuple[int, float]:
    if side == "ask":
        return raw_int(row[f"申卖价{level}"]), raw_float(row[f"申卖量{level}"])
    if side == "bid":
        return raw_int(row[f"申买价{level}"]), raw_float(row[f"申买量{level}"])
    raise ValueError(f"Unknown side: {side}")


def validate_replay_against_market(
    order: pd.DataFrame,
    trade: pd.DataFrame,
    market: pd.DataFrame,
    price_scale: float = 10_000.0,
    depth: int = 10,
    start_time: int = 93_000_000,
    end_time: int = 145_700_000,
) -> tuple[pd.DataFrame, dict[str, float]]:
    events = build_event_stream(order, trade)
    market_slice = (
        market.loc[(market["时间"] >= start_time) & (market["时间"] <= end_time)]
        .sort_values("时间", kind="mergesort")
        .reset_index(drop=True)
    )
    book = OrderBookReplay()
    event_position = 0
    event_records = events.to_dict("records")
    validation_rows: list[dict[str, object]] = []

    for _, market_row in market_slice.iterrows():
        snapshot_time = int(market_row["时间"])
        while event_position < len(event_records) and int(event_records[event_position]["raw_time"]) <= snapshot_time:
            handle_event(book, pd.Series(event_records[event_position]), price_scale=price_scale)
            event_position += 1

        replay_levels = book.top_levels_raw(depth)
        row: dict[str, object] = {
            "raw_time": snapshot_time,
            "time": parse_l2_time_to_seconds(snapshot_time),
            "events_consumed": event_position,
        }
        for level, (ask_price, ask_size, bid_price, bid_size) in enumerate(replay_levels, start=1):
            market_ask_price, market_ask_size = market_level_raw(market_row, "ask", level)
            market_bid_price, market_bid_size = market_level_raw(market_row, "bid", level)
            row[f"ask_price_{level}_match"] = ask_price == market_ask_price
            row[f"bid_price_{level}_match"] = bid_price == market_bid_price
            row[f"market_ask_price_{level}"] = market_ask_price / price_scale
            row[f"replay_ask_price_{level}"] = ask_price / price_scale
            row[f"market_bid_price_{level}"] = market_bid_price / price_scale
            row[f"replay_bid_price_{level}"] = bid_price / price_scale
            row[f"market_ask_size_{level}"] = market_ask_size
            row[f"replay_ask_size_{level}"] = ask_size
            row[f"market_bid_size_{level}"] = market_bid_size
            row[f"replay_bid_size_{level}"] = bid_size
            row[f"ask_size_{level}_abs_error"] = abs(ask_size - market_ask_size)
            row[f"bid_size_{level}_abs_error"] = abs(bid_size - market_bid_size)
        validation_rows.append(row)

    validation = pd.DataFrame(validation_rows)
    metrics: dict[str, float] = {
        "snapshots": float(len(validation)),
        "events_consumed": float(event_position),
        "missing_references": float(book.missing_references),
        "duplicate_orders": float(book.duplicate_orders),
    }
    if not validation.empty:
        metrics["top1_ask_price_match_rate"] = float(validation["ask_price_1_match"].mean())
        metrics["top1_bid_price_match_rate"] = float(validation["bid_price_1_match"].mean())
        metrics["top1_both_price_match_rate"] = float(
            (validation["ask_price_1_match"] & validation["bid_price_1_match"]).mean()
        )
        ask_match_cols = [f"ask_price_{level}_match" for level in range(1, depth + 1)]
        bid_match_cols = [f"bid_price_{level}_match" for level in range(1, depth + 1)]
        metrics["levelwise_ask_price_match_rate"] = float(validation[ask_match_cols].to_numpy(dtype=float).mean())
        metrics["levelwise_bid_price_match_rate"] = float(validation[bid_match_cols].to_numpy(dtype=float).mean())
        metrics["top1_ask_size_mae"] = float(validation["ask_size_1_abs_error"].mean())
        metrics["top1_bid_size_mae"] = float(validation["bid_size_1_abs_error"].mean())
    return validation, metrics


def find_date_files(data_root: Path, date: str) -> L2DateFiles:
    for files in discover_date_files(data_root):
        if files.date == date:
            return files
    raise FileNotFoundError(f"Date {date} not found under {data_root}")


def require_symbol(files: L2DateFiles, symbol: str, allow_non_sz: bool = False) -> str:
    normalized = symbol.strip().strip("/").upper()
    if not allow_non_sz and not normalized.endswith(".SZ"):
        raise ValueError(f"This script is intended for Shenzhen symbols. Got {normalized}.")
    symbols = complete_symbols(files)
    if normalized not in symbols:
        raise KeyError(f"Symbol {normalized} is not present in all market/order/trade files for {files.date}")
    return normalized


def write_frame(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(output_path.suffixes)
    if suffixes.endswith(".pkl.gz") or output_path.suffix == ".pkl":
        frame.to_pickle(output_path, compression="gzip" if suffixes.endswith(".gz") else None)
    elif output_path.suffix == ".csv":
        frame.to_csv(output_path, index=False)
    else:
        raise ValueError("Output path must end with .pkl, .pkl.gz, or .csv")


def print_summary(summary: pd.DataFrame) -> None:
    print(summary.to_string(index=False))
    print()
    print(f"dates: {summary.attrs['num_dates']}")
    print(f"unique complete symbols: {summary.attrs['union_all']}")
    print(f"unique SZ complete symbols: {summary.attrs['union_sz']}")
    print(f"unique SH complete symbols: {summary.attrs['union_sh']}")
    print(f"common complete symbols across all dates: {summary.attrs['common_all']}")
    print(f"common SZ complete symbols across all dates: {summary.attrs['common_sz']}")
    print(f"common SH complete symbols across all dates: {summary.attrs['common_sh']}")
    print(f"common SZ sample: {', '.join(summary.attrs['common_sz_sample'])}")


def print_metrics(metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        if key.endswith("_rate"):
            print(f"{key}: {value:.2%}")
        elif float(value).is_integer():
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.4f}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=default_data_root(), help="Root directory containing *_hdf folders.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct Shenzhen event-level order books from the uploaded L2 HDF5 files.")
    add_common_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("summary", help="Print available dates and complete symbol counts.")

    validate_parser = subparsers.add_parser("validate", help="Replay one SZ symbol and compare against market snapshots.")
    validate_parser.add_argument("--date", required=True, help="Trading date, e.g. 20250121.")
    validate_parser.add_argument("--symbol", required=True, help="SZ symbol, e.g. 000608.SZ.")
    validate_parser.add_argument("--start-time", type=int, default=93_000_000)
    validate_parser.add_argument("--end-time", type=int, default=145_700_000)
    validate_parser.add_argument("--depth", type=int, default=10)
    validate_parser.add_argument("--price-scale", type=float, default=10_000.0)
    validate_parser.add_argument("--output", type=Path, default=None, help="Optional validation detail output (.csv/.pkl/.pkl.gz).")
    validate_parser.add_argument("--allow-non-sz", action="store_true", help="Allow testing non-SZ symbols for diagnostics.")

    convert_parser = subparsers.add_parser("convert", help="Write one reconstructed event-level orderbook table.")
    convert_parser.add_argument("--date", required=True, help="Trading date, e.g. 20250121.")
    convert_parser.add_argument("--symbol", required=True, help="SZ symbol, e.g. 000608.SZ.")
    convert_parser.add_argument("--start-time", type=int, default=93_000_000)
    convert_parser.add_argument("--end-time", type=int, default=145_700_000)
    convert_parser.add_argument("--depth", type=int, default=10)
    convert_parser.add_argument("--price-scale", type=float, default=10_000.0)
    convert_parser.add_argument("--max-events", type=int, default=None)
    convert_parser.add_argument(
        "--emit",
        choices=["time", "event"],
        default="time",
        help="time groups all rows with the same raw timestamp before emitting a book state; event emits after each raw row.",
    )
    convert_parser.add_argument("--output", type=Path, required=True, help="Output path ending with .csv, .pkl, or .pkl.gz.")
    convert_parser.add_argument("--allow-non-sz", action="store_true", help="Allow testing non-SZ symbols for diagnostics.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "summary":
        print_summary(summarize_universe(args.data_root))
        return

    files = find_date_files(args.data_root, args.date)
    symbol = require_symbol(files, args.symbol, allow_non_sz=args.allow_non_sz)
    order = read_symbol_frame(files.order_path, symbol)
    trade = read_symbol_frame(files.trade_path, symbol)

    if args.command == "validate":
        market = read_symbol_frame(files.market_path, symbol)
        validation, metrics = validate_replay_against_market(
            order=order,
            trade=trade,
            market=market,
            price_scale=args.price_scale,
            depth=args.depth,
            start_time=args.start_time,
            end_time=args.end_time,
        )
        print(f"date: {args.date}")
        print(f"symbol: {symbol}")
        print_metrics(metrics)
        if args.output is not None:
            write_frame(validation, args.output)
            print(f"wrote: {args.output}")
        return

    if args.command == "convert":
        result = reconstruct_orderbook(
            order=order,
            trade=trade,
            price_scale=args.price_scale,
            depth=args.depth,
            start_time=args.start_time,
            end_time=args.end_time,
            max_events=args.max_events,
            emit=args.emit,
        )
        write_frame(result.frame, args.output)
        print(f"date: {args.date}")
        print(f"symbol: {symbol}")
        print(f"events_seen: {result.events_seen}")
        print(f"events_written: {result.events_written}")
        print(f"missing_references: {result.missing_references}")
        print(f"duplicate_orders: {result.duplicate_orders}")
        print(f"wrote: {args.output}")
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
