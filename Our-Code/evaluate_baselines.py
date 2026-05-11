from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from rl_trading.baselines import AvellanedaStoikovPolicy, FixedLevelPolicy, RandomLevelPolicy, make_baseline_policy
from rl_trading.env import LOBContinuousMarketMakingEnv, LOBEnvConfig
from rl_trading.evaluation import evaluate_quote_policy


@dataclass
class BaselineEvalConfig:
    ticker: str = "AAPL"
    split: str = "test"
    processed_root: str = "../lobster_attnlob_us_repro/data/lobster_processed"
    output_dir: str = "experiment_results/baselines"
    baselines: str = "fixed_1,fixed_2,fixed_3,random,as"
    eval_episodes: int = 10
    time_window: int = 50
    latency: int = 1
    order_latency_events: int = 0
    order_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
    order_loss_probability: float = 0.0
    episode_length: int = 2000
    trade_unit: int = 100
    max_inventory_units: int = 10
    tick_size: float = 0.01
    max_bias: float = 0.05
    max_spread: float = 0.10
    reward_eta: float = 0.5
    reward_theta: float = 0.01
    transaction_cost_bps: float = 0.0
    fixed_fee_per_trade: float = 0.0
    random_max_level: int = 5
    as_gamma: float = 0.1
    as_kappa: float = 20.0
    as_sigma_window: int = 200
    seed: int = 7


def parse_args() -> BaselineEvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate non-learning market-making baselines on processed LOBSTER event data.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--processed-root", default="../lobster_attnlob_us_repro/data/lobster_processed")
    parser.add_argument("--output-dir", default="experiment_results/baselines")
    parser.add_argument("--baselines", default="fixed_1,fixed_2,fixed_3,random,as")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--time-window", type=int, default=50)
    parser.add_argument("--latency", type=int, default=1)
    parser.add_argument("--order-latency-events", type=int, default=0)
    parser.add_argument("--order-latency-ms", type=float, default=0.0)
    parser.add_argument("--inference-latency-ms", type=float, default=0.0)
    parser.add_argument("--order-loss-probability", type=float, default=0.0)
    parser.add_argument("--episode-length", type=int, default=2000)
    parser.add_argument("--trade-unit", type=int, default=100)
    parser.add_argument("--max-inventory-units", type=int, default=10)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--max-bias", type=float, default=0.05)
    parser.add_argument("--max-spread", type=float, default=0.10)
    parser.add_argument("--reward-eta", type=float, default=0.5)
    parser.add_argument("--reward-theta", type=float, default=0.01)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    parser.add_argument("--fixed-fee-per-trade", type=float, default=0.0)
    parser.add_argument("--random-max-level", type=int, default=5)
    parser.add_argument("--as-gamma", type=float, default=0.1)
    parser.add_argument("--as-kappa", type=float, default=20.0)
    parser.add_argument("--as-sigma-window", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    return BaselineEvalConfig(**vars(parser.parse_args()))


def build_env(config: BaselineEvalConfig) -> LOBContinuousMarketMakingEnv:
    env_config = LOBEnvConfig(
        ticker=config.ticker,
        split=config.split,
        processed_root=config.processed_root,
        time_window=config.time_window,
        latency=config.latency,
        order_latency_events=config.order_latency_events,
        order_latency_ms=config.order_latency_ms,
        inference_latency_ms=config.inference_latency_ms,
        order_loss_probability=config.order_loss_probability,
        episode_length=config.episode_length,
        trade_unit=config.trade_unit,
        max_inventory_units=config.max_inventory_units,
        tick_size=config.tick_size,
        max_bias=config.max_bias,
        max_spread=config.max_spread,
        reward_eta=config.reward_eta,
        reward_theta=config.reward_theta,
        transaction_cost_bps=config.transaction_cost_bps,
        fixed_fee_per_trade=config.fixed_fee_per_trade,
        seed=config.seed,
    )
    return LOBContinuousMarketMakingEnv(env_config)


def build_policies(config: BaselineEvalConfig):
    names = [item.strip() for item in config.baselines.split(",") if item.strip()]
    policies = []
    for name in names:
        policy = make_baseline_policy(name, seed=config.seed, random_max_level=config.random_max_level)
        if isinstance(policy, AvellanedaStoikovPolicy):
            policy.gamma = config.as_gamma
            policy.kappa = config.as_kappa
            policy.sigma_window = config.as_sigma_window
        policies.append(policy)
    return policies


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir) / config.ticker / config.split
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | str]] = []
    episode_paths: dict[str, str] = {}
    for policy in build_policies(config):
        env = build_env(config)
        episode_results, summary = evaluate_quote_policy(policy, env, config.eval_episodes)
        episode_path = output_dir / f"{policy.name}_episode_results.csv"
        episode_results.to_csv(episode_path, index=False)
        episode_paths[policy.name] = str(episode_path)
        summary_rows.append({"baseline": policy.name, **summary})
        print(f"baseline={policy.name} episodes={int(summary.get('episodes', 0))} pnl_mean={summary.get('pnl_mean', 0.0):.4f}")

    summary_frame = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    metadata = {
        "config": asdict(config),
        "summary_path": str(summary_path),
        "episode_results": episode_paths,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"summary_written path={summary_path}")
    print(f"metadata_written path={metadata_path}")


if __name__ == "__main__":
    main()
