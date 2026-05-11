from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl_trading.env import LOBContinuousMarketMakingEnv, LOBEnvConfig


REGIME_COLUMNS = ["realized_volatility", "avg_market_spread", "trade_intensity_per_second"]
REGIME_LABELS = {
    "realized_volatility": "volatility_regime",
    "avg_market_spread": "spread_regime",
    "trade_intensity_per_second": "trade_intensity_regime",
}


@dataclass
class RegimeConfig:
    experiment_root: str = "experiment_results/sz10_full_experiment"
    processed_root: str = "data/sz_l2_processed"
    output_dir: str | None = None
    split: str = "test"
    episode_length: int = 2000
    time_window: int = 50
    latency: int = 1


def parse_args() -> RegimeConfig:
    parser = argparse.ArgumentParser(description="Analyze method performance by market regime.")
    parser.add_argument("--experiment-root", default="experiment_results/sz10_full_experiment")
    parser.add_argument("--processed-root", default="data/sz_l2_processed")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--episode-length", type=int, default=2000)
    parser.add_argument("--time-window", type=int, default=50)
    parser.add_argument("--latency", type=int, default=1)
    return RegimeConfig(**vars(parser.parse_args()))


def read_tickers(experiment_root: Path) -> list[str]:
    rl_summary = pd.read_csv(experiment_root / "reports" / "rl_summary.csv")
    return sorted(rl_summary["ticker"].dropna().unique().tolist())


def local_artifact_path(experiment_root: Path, value: str) -> Path:
    artifact_path = Path(value)
    if artifact_path.exists():
        return artifact_path
    marker = f"checkpoints/{experiment_root.name}/"
    if marker in value:
        return experiment_root / value.split(marker, 1)[1]
    marker = f"experiment_results/{experiment_root.name}/"
    if marker in value:
        return experiment_root / value.split(marker, 1)[1]
    return artifact_path


def episode_market_regimes(config: RegimeConfig, ticker: str) -> pd.DataFrame:
    env = LOBContinuousMarketMakingEnv(
        LOBEnvConfig(
            ticker=ticker,
            split=config.split,
            processed_root=config.processed_root,
            time_window=config.time_window,
            latency=config.latency,
            episode_length=config.episode_length,
        )
    )
    rows: list[dict[str, float | str]] = []
    for episode_index in range(env.episode_count()):
        start = env._episode_start_for_index(episode_index)  # internal deterministic episode schedule
        end = min(start + config.episode_length, env.split_end - 1)
        frame = env.frame.iloc[start:end]
        mid = frame["mid_price"].to_numpy(dtype=np.float64)
        log_mid = np.log(np.clip(mid, 1e-8, None))
        returns = np.diff(log_mid)
        elapsed_seconds = max(float(frame["time"].iloc[-1] - frame["time"].iloc[0]), 1e-7) if len(frame) > 1 else 1e-7
        rows.append(
            {
                "ticker": ticker,
                "episode_index": float(episode_index),
                "realized_volatility": float(np.std(returns, ddof=0)) if returns.size else 0.0,
                "avg_market_spread": float(frame["spread"].mean()),
                "trade_intensity_per_second": float(frame["is_execution"].sum() / elapsed_seconds),
                "episode_seconds": elapsed_seconds,
            }
        )
    return pd.DataFrame(rows)


def add_regime_labels(regimes: pd.DataFrame) -> pd.DataFrame:
    regimes = regimes.copy()
    for value_column, label_column in REGIME_LABELS.items():
        ranks = regimes[value_column].rank(method="first")
        regimes[label_column] = pd.qcut(ranks, q=3, labels=["low", "mid", "high"])
        regimes[label_column] = regimes[label_column].astype(str)
    return regimes


def load_method_episodes(experiment_root: Path) -> pd.DataFrame:
    rl_summary = pd.read_csv(experiment_root / "reports" / "rl_summary.csv")
    frames: list[pd.DataFrame] = []
    for _, row in rl_summary.iterrows():
        episode = pd.read_csv(local_artifact_path(experiment_root, str(row["test_episode_results"])))
        episode["method"] = row["method"]
        episode["scope"] = row["scope"]
        episode["agent"] = row["agent"]
        episode["ticker"] = row["ticker"]
        frames.append(episode)

    baseline_summary = pd.read_csv(experiment_root / "reports" / "baseline_summary.csv")
    for _, row in baseline_summary.iterrows():
        ticker = row["ticker"]
        baseline = row["baseline"]
        path = experiment_root / "baselines" / ticker / "test" / f"{baseline}_episode_results.csv"
        episode = pd.read_csv(path)
        episode["method"] = row["method"]
        episode["scope"] = "baseline"
        episode["agent"] = baseline
        episode["ticker"] = ticker
        frames.append(episode)
    return pd.concat(frames, ignore_index=True, sort=False)


def summarize_by_regime(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    metric_columns = [
        column
        for column in ["pnl", "nd_pnl", "pnl_map", "profit_ratio", "avg_abs_position", "avg_spread", "volume", "episode_reward"]
        if column in episodes
    ]
    for regime_column in REGIME_LABELS.values():
        grouped = episodes.groupby(["method", regime_column], dropna=False)
        summary = grouped[metric_columns].agg(["mean", "std", "count"]).reset_index()
        summary.columns = [
            "_".join(str(part) for part in column if part)
            if isinstance(column, tuple)
            else str(column)
            for column in summary.columns
        ]
        summary = summary.rename(columns={regime_column: "regime"})
        summary.insert(1, "regime_type", regime_column)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True, sort=False)


def plot_regime_results(summary: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    for regime_type in REGIME_LABELS.values():
        frame = summary[summary["regime_type"] == regime_type]
        if frame.empty or "pnl_mean" not in frame:
            continue
        pivot = frame.pivot(index="regime", columns="method", values="pnl_mean").reindex(["low", "mid", "high"])
        ax = pivot.plot(kind="bar", figsize=(14, 7), width=0.85)
        ax.set_title(f"Mean Test PnL by {regime_type.replace('_', ' ').title()}")
        ax.set_xlabel("Regime")
        ax.set_ylabel("Mean PnL")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{regime_type}_pnl.png", dpi=180)
        plt.close()


def main() -> None:
    config = parse_args()
    experiment_root = Path(config.experiment_root)
    output_root = Path(config.output_dir) if config.output_dir else experiment_root / "regime_sensitivity"
    reports_dir = output_root / "reports"
    plots_dir = output_root / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)

    regimes = pd.concat([episode_market_regimes(config, ticker) for ticker in read_tickers(experiment_root)], ignore_index=True)
    regimes = add_regime_labels(regimes)
    episodes = load_method_episodes(experiment_root)
    merged = episodes.merge(regimes, on=["ticker", "episode_index"], how="left")
    summary = summarize_by_regime(merged)

    regimes.to_csv(reports_dir / "episode_regimes.csv", index=False)
    merged.to_csv(reports_dir / "regime_episode_results.csv", index=False)
    summary.to_csv(reports_dir / "regime_sensitivity_summary.csv", index=False)
    plot_regime_results(summary, plots_dir)

    print(f"episode_regimes_written: {reports_dir / 'episode_regimes.csv'}")
    print(f"merged_results_written: {reports_dir / 'regime_episode_results.csv'}")
    print(f"summary_written: {reports_dir / 'regime_sensitivity_summary.csv'}")
    print(f"plots_written: {plots_dir}")


if __name__ == "__main__":
    main()
