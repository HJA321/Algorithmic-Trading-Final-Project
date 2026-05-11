from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from .reconstruct_sz_orderbook import (
        complete_symbols,
        default_data_root,
        discover_date_files,
        hdf_key,
    )
except ImportError:
    from reconstruct_sz_orderbook import (
        complete_symbols,
        default_data_root,
        discover_date_files,
        hdf_key,
    )


@dataclass(frozen=True)
class AnalyzeConfig:
    data_root: Path
    output_dir: Path
    start_time: int = 93_000_000
    end_time: int = 145_700_000
    price_scale: float = 10_000.0
    tick_raw: int = 100
    horizon: int = 10
    label_threshold: float = 1e-5
    top_k: int = 30
    min_median_price: float = 5.0
    max_mean_spread_bps: float = 35.0
    max_stationary_fraction: float = 0.80
    min_total_trade_count: int = 20_000


def common_sz_symbols(data_root: Path) -> list[str]:
    common: set[str] | None = None
    for files in discover_date_files(data_root):
        symbols = {symbol for symbol in complete_symbols(files) if symbol.endswith(".SZ")}
        common = symbols if common is None else common & symbols
    return sorted(common or set())


def storer_nrows(path: Path, symbol: str) -> int:
    with pd.HDFStore(path, mode="r") as store:
        return int(store.get_storer(hdf_key(symbol)).nrows)


def compute_labels(mid: np.ndarray, horizon: int, threshold: float) -> np.ndarray:
    mid_series = pd.Series(mid, dtype=np.float64)
    past = mid_series.rolling(window=horizon).mean()
    future = past.shift(-horizon)
    pct_change = (future - past) / past
    labels = np.full(len(mid), -99, dtype=np.int8)
    values = pct_change.to_numpy(dtype=np.float64)
    valid = pct_change.notna().to_numpy() & np.isfinite(values) & (past.to_numpy(dtype=np.float64) > 0)
    labels[valid & (values >= threshold)] = 2
    labels[valid & (values < threshold) & (values > -threshold)] = 1
    labels[valid & (values <= -threshold)] = 0
    return labels


def entropy_from_counts(counts: Sequence[int]) -> float:
    values = np.asarray(counts, dtype=np.float64)
    total = values.sum()
    if total <= 0:
        return 0.0
    probabilities = values / total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(values)))


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def symbol_day_metrics(config: AnalyzeConfig, files, symbol: str) -> dict[str, float]:
    market = pd.read_hdf(files.market_path, hdf_key(symbol))
    mask = (
        (numeric_series(market, "时间") >= config.start_time)
        & (numeric_series(market, "时间") <= config.end_time)
    )
    market = market.loc[mask].copy()

    ask_1 = numeric_series(market, "申卖价1").to_numpy(dtype=np.float64)
    bid_1 = numeric_series(market, "申买价1").to_numpy(dtype=np.float64)
    ask_size_1 = numeric_series(market, "申卖量1").to_numpy(dtype=np.float64)
    bid_size_1 = numeric_series(market, "申买量1").to_numpy(dtype=np.float64)
    valid = (ask_1 > 0) & (bid_1 > 0) & np.isfinite(ask_1) & np.isfinite(bid_1)
    spread_raw = ask_1 - bid_1
    valid &= spread_raw >= 0
    mid = (ask_1 + bid_1) / 2.0 / config.price_scale
    valid_mid = mid[valid]
    valid_spread_raw = spread_raw[valid]

    labels = compute_labels(valid_mid, horizon=config.horizon, threshold=config.label_threshold)
    label_counts = np.bincount(labels[labels >= 0], minlength=3)

    turnover = float(numeric_series(market, "当日成交额").max())
    cumulative_volume = float(numeric_series(market, "当日累计成交量").max())
    open_mid = valid_mid[0] if len(valid_mid) else np.nan
    close_mid = valid_mid[-1] if len(valid_mid) else np.nan
    day_return = float(close_mid / open_mid - 1.0) if np.isfinite(open_mid) and open_mid > 0 else np.nan
    intraday_range = float(valid_mid.max() / valid_mid.min() - 1.0) if len(valid_mid) and valid_mid.min() > 0 else np.nan

    trade = pd.read_hdf(files.trade_path, hdf_key(symbol))
    trade_time = numeric_series(trade, "时间")
    trade_price = numeric_series(trade, "成交价格").fillna(0).to_numpy(dtype=np.float64)
    trade_quantity = numeric_series(trade, "成交数量").fillna(0).to_numpy(dtype=np.float64)
    trade_mask = (
        (trade_time >= config.start_time)
        & (trade_time <= config.end_time)
        & (trade_price > 0)
        & (trade_quantity > 0)
    ).to_numpy()
    trade_count = int(trade_mask.sum())
    trade_amount = float((trade_price[trade_mask] * trade_quantity[trade_mask] / config.price_scale).sum())

    return {
        "date": files.date,
        "market_rows": float(len(market)),
        "valid_book_rows": float(valid.sum()),
        "valid_book_fraction": float(valid.mean()) if len(valid) else 0.0,
        "median_price": float(np.nanmedian(valid_mid)) if len(valid_mid) else np.nan,
        "mean_price": float(np.nanmean(valid_mid)) if len(valid_mid) else np.nan,
        "mean_spread_ticks": float(np.nanmean(valid_spread_raw / config.tick_raw)) if len(valid_spread_raw) else np.nan,
        "median_spread_ticks": float(np.nanmedian(valid_spread_raw / config.tick_raw)) if len(valid_spread_raw) else np.nan,
        "mean_spread_bps": float(np.nanmean((valid_spread_raw / config.price_scale) / valid_mid * 10_000))
        if len(valid_mid)
        else np.nan,
        "one_tick_spread_fraction": float((valid_spread_raw <= config.tick_raw).mean()) if len(valid_spread_raw) else 0.0,
        "locked_fraction": float((valid_spread_raw <= 0).mean()) if len(valid_spread_raw) else 0.0,
        "top1_depth_mean": float(np.nanmean(ask_size_1[valid] + bid_size_1[valid])) if valid.sum() else np.nan,
        "turnover": turnover,
        "cumulative_volume": cumulative_volume,
        "trade_count": float(trade_count),
        "trade_amount": trade_amount,
        "day_return_abs": float(abs(day_return)) if np.isfinite(day_return) else np.nan,
        "intraday_range": intraday_range,
        "label_down": int(label_counts[0]),
        "label_stationary": int(label_counts[1]),
        "label_up": int(label_counts[2]),
        "label_total": int(label_counts.sum()),
    }


def aggregate_symbol_metrics(config: AnalyzeConfig, symbol: str, files_by_date: Sequence) -> dict[str, float | str]:
    daily = [symbol_day_metrics(config, files, symbol) for files in files_by_date]
    frame = pd.DataFrame(daily)
    label_counts = frame[["label_down", "label_stationary", "label_up"]].sum(axis=0).to_numpy(dtype=np.int64)
    label_total = int(label_counts.sum())
    stationary_fraction = float(label_counts[1] / max(label_total, 1))

    return {
        "symbol": symbol,
        "dates": len(files_by_date),
        "market_rows_total": int(frame["market_rows"].sum()),
        "valid_book_rows_total": int(frame["valid_book_rows"].sum()),
        "valid_book_fraction_mean": float(frame["valid_book_fraction"].mean()),
        "median_price": float(frame["median_price"].median()),
        "mean_price": float(frame["mean_price"].mean()),
        "mean_spread_bps": float(frame["mean_spread_bps"].mean()),
        "mean_spread_ticks": float(frame["mean_spread_ticks"].mean()),
        "median_spread_ticks": float(frame["median_spread_ticks"].median()),
        "one_tick_spread_fraction_mean": float(frame["one_tick_spread_fraction"].mean()),
        "locked_fraction_mean": float(frame["locked_fraction"].mean()),
        "top1_depth_mean": float(frame["top1_depth_mean"].mean()),
        "turnover_total": float(frame["turnover"].sum()),
        "cumulative_volume_total": float(frame["cumulative_volume"].sum()),
        "trade_count_total": int(frame["trade_count"].sum()),
        "trade_amount_total": float(frame["trade_amount"].sum()),
        "mean_abs_day_return": float(frame["day_return_abs"].mean()),
        "mean_intraday_range": float(frame["intraday_range"].mean()),
        "label_down": int(label_counts[0]),
        "label_stationary": int(label_counts[1]),
        "label_up": int(label_counts[2]),
        "label_total": label_total,
        "stationary_fraction": stationary_fraction,
        "label_entropy": entropy_from_counts(label_counts),
    }


def minmax_score(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = values.astype(float).replace([np.inf, -np.inf], np.nan)
    lo = values.quantile(0.05)
    hi = values.quantile(0.95)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        score = pd.Series(0.5, index=values.index)
    else:
        score = ((values.clip(lo, hi) - lo) / (hi - lo)).fillna(0.0)
    return score if higher_is_better else 1.0 - score


def score_candidates(config: AnalyzeConfig, metrics: pd.DataFrame) -> pd.DataFrame:
    scored = metrics.copy()
    scored["price_ok"] = scored["median_price"] >= config.min_median_price
    scored["spread_ok"] = scored["mean_spread_bps"] <= config.max_mean_spread_bps
    scored["label_ok"] = scored["stationary_fraction"] <= config.max_stationary_fraction
    scored["activity_ok"] = scored["trade_count_total"] >= config.min_total_trade_count
    scored["book_ok"] = scored["valid_book_fraction_mean"] >= 0.995
    scored["passes_default_filters"] = (
        scored["price_ok"]
        & scored["spread_ok"]
        & scored["label_ok"]
        & scored["activity_ok"]
        & scored["book_ok"]
    )

    scored["s_liquidity"] = minmax_score(np.log1p(scored["trade_amount_total"]), True)
    scored["s_trade_count"] = minmax_score(np.log1p(scored["trade_count_total"]), True)
    scored["s_spread"] = minmax_score(scored["mean_spread_bps"], False)
    scored["s_price"] = minmax_score(scored["median_price"], True)
    scored["s_label_entropy"] = minmax_score(scored["label_entropy"], True)
    scored["s_stationary"] = minmax_score(scored["stationary_fraction"], False)
    scored["s_book_valid"] = minmax_score(scored["valid_book_fraction_mean"], True)
    scored["training_score"] = (
        0.22 * scored["s_liquidity"]
        + 0.18 * scored["s_trade_count"]
        + 0.18 * scored["s_spread"]
        + 0.12 * scored["s_price"]
        + 0.16 * scored["s_label_entropy"]
        + 0.10 * scored["s_stationary"]
        + 0.04 * scored["s_book_valid"]
    )
    scored.loc[~scored["passes_default_filters"], "training_score"] *= 0.65
    return scored.sort_values(["training_score", "trade_amount_total", "symbol"], ascending=[False, False, True])


def analyze(config: AnalyzeConfig) -> pd.DataFrame:
    date_files = sorted(discover_date_files(config.data_root), key=lambda item: item.date)
    symbols = common_sz_symbols(config.data_root)
    rows: list[dict[str, float | str]] = []
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for idx, symbol in enumerate(symbols, start=1):
        row = aggregate_symbol_metrics(config, symbol, date_files)
        rows.append(row)
        print(
            f"analyzed {idx}/{len(symbols)} {symbol}: "
            f"price={row['median_price']:.2f} "
            f"spread={row['mean_spread_bps']:.1f}bps "
            f"trades={row['trade_count_total']} "
            f"stationary={row['stationary_fraction']:.1%}",
            flush=True,
        )

    metrics = pd.DataFrame(rows)
    metrics = score_candidates(config, metrics).reset_index(drop=True)
    metrics.to_csv(config.output_dir / "sz_candidate_training_scores.csv", index=False)
    metrics.head(config.top_k).to_csv(config.output_dir / "sz_candidate_recommendations.csv", index=False)

    selected_path = config.output_dir.parent / "sz_l2_processed" / "selected_symbols.csv"
    if selected_path.exists():
        selected = pd.read_csv(selected_path)
        selected_metrics = metrics.loc[metrics["symbol"].isin(selected["symbol"])]
        selected_metrics.to_csv(config.output_dir / "current_selected_data_profile.csv", index=False)

    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze SZ L2 candidates using data-only training suitability metrics.")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--output-dir", type=Path, default=Path("Our-Code/data/sz_l2_analysis"))
    parser.add_argument("--start-time", type=int, default=93_000_000)
    parser.add_argument("--end-time", type=int, default=145_700_000)
    parser.add_argument("--price-scale", type=float, default=10_000.0)
    parser.add_argument("--tick-raw", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--label-threshold", type=float, default=1e-5)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-median-price", type=float, default=5.0)
    parser.add_argument("--max-mean-spread-bps", type=float, default=35.0)
    parser.add_argument("--max-stationary-fraction", type=float, default=0.80)
    parser.add_argument("--min-total-trade-count", type=int, default=20_000)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    config = AnalyzeConfig(**vars(parser.parse_args(argv)))
    metrics = analyze(config)
    print()
    print("top recommendations:")
    columns = [
        "symbol",
        "training_score",
        "median_price",
        "mean_spread_bps",
        "trade_count_total",
        "trade_amount_total",
        "stationary_fraction",
        "label_entropy",
        "passes_default_filters",
    ]
    print(metrics.head(config.top_k)[columns].to_string(index=False))
    print(f"wrote: {config.output_dir / 'sz_candidate_training_scores.csv'}")
    print(f"wrote: {config.output_dir / 'sz_candidate_recommendations.csv'}")


if __name__ == "__main__":
    main()
