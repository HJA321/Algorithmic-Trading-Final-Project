from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

from rl_trading.baselines import AvellanedaStoikovPolicy, make_baseline_policy
from rl_trading.env import LOBContinuousMarketMakingEnv, LOBDiscreteMarketMakingEnv, LOBEnvConfig
from rl_trading.evaluation import evaluate_agent, evaluate_quote_policy
from rl_trading.model import DuelingDQNPolicy, MultiInputLOBExtractor  # noqa: F401


DEFAULT_SCENARIOS = {
    "paper_base": {},
    "obs_latency_5": {"latency": 5},
    "obs_latency_10": {"latency": 10},
    "order_latency_50ms": {"order_latency_ms": 50.0},
    "loss_5pct": {"order_loss_probability": 0.05},
    "obs10_loss5pct": {"latency": 10, "order_loss_probability": 0.05},
    "infer_50ms": {"inference_latency_ms": 50.0},
    "infer50_loss5pct": {"inference_latency_ms": 50.0, "order_loss_probability": 0.05},
}

DEFAULT_METHODS = ",".join(
    [
        "single_ppo",
        "joint_ppo",
        "single_dueling_dqn",
        "joint_dueling_dqn",
        "baseline_fixed_1",
        "baseline_fixed_2",
        "baseline_fixed_3",
        "baseline_random_5level",
        "baseline_as",
    ]
)


@dataclass
class LatencyEvalConfig:
    experiment_root: str = "experiment_results/sz10_full_experiment"
    processed_root: str = "data/sz_l2_processed"
    output_dir: str | None = None
    split: str = "test"
    tickers: str = ""
    methods: str = DEFAULT_METHODS
    scenarios: str = ",".join(DEFAULT_SCENARIOS)
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


@dataclass
class ScenarioConfig:
    name: str
    latency: int
    order_latency_events: int
    order_latency_ms: float
    inference_latency_ms: float
    order_loss_probability: float


def parse_args() -> LatencyEvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate trained RL agents and baselines under latency/loss scenarios.")
    parser.add_argument("--experiment-root", default="experiment_results/sz10_full_experiment")
    parser.add_argument("--processed-root", default="data/sz_l2_processed")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--tickers", default="")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
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
    return LatencyEvalConfig(**vars(parser.parse_args()))


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_scenario_names(value: str) -> list[str]:
    separator = ";" if ";" in value else ","
    return [item.strip() for item in value.split(separator) if item.strip()]


def scenario_from_name(name: str, config: LatencyEvalConfig) -> ScenarioConfig:
    values = {
        "latency": config.latency,
        "order_latency_events": config.order_latency_events,
        "order_latency_ms": config.order_latency_ms,
        "inference_latency_ms": config.inference_latency_ms,
        "order_loss_probability": config.order_loss_probability,
    }
    if name in DEFAULT_SCENARIOS:
        values.update(DEFAULT_SCENARIOS[name])
        return ScenarioConfig(name=name, **values)

    scenario_name, _, assignments = name.partition(":")
    if not assignments:
        raise ValueError(f"Unknown scenario: {name}")
    for item in assignments.split(","):
        key, _, raw_value = item.partition("=")
        if key not in values or not raw_value:
            raise ValueError(f"Invalid scenario item '{item}' in '{name}'")
        values[key] = int(raw_value) if key in {"latency", "order_latency_events"} else float(raw_value)
    return ScenarioConfig(name=scenario_name, **values)


def build_env(config: LatencyEvalConfig, scenario: ScenarioConfig, ticker: str, agent: str | None):
    env_config = LOBEnvConfig(
        ticker=ticker,
        split=config.split,
        processed_root=config.processed_root,
        time_window=config.time_window,
        latency=scenario.latency,
        order_latency_events=scenario.order_latency_events,
        order_latency_ms=scenario.order_latency_ms,
        inference_latency_ms=scenario.inference_latency_ms,
        order_loss_probability=scenario.order_loss_probability,
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
    if agent == "ppo" or agent is None:
        return LOBContinuousMarketMakingEnv(env_config)
    return LOBDiscreteMarketMakingEnv(env_config)


def available_tickers(registry: pd.DataFrame, config: LatencyEvalConfig) -> list[str]:
    if config.tickers:
        return parse_list(config.tickers)
    rl_rows = registry[registry["kind"] == "rl"]
    return sorted(rl_rows["ticker"].dropna().unique().tolist())


def rl_model_row(registry: pd.DataFrame, scope: str, agent: str, ticker: str) -> pd.Series:
    matches = registry[
        (registry["kind"] == "rl")
        & (registry["scope"] == scope)
        & (registry["agent"] == agent)
        & (registry["ticker"] == ticker)
    ]
    if matches.empty:
        raise FileNotFoundError(f"No registered model for {scope}_{agent} {ticker}")
    return matches.iloc[0]


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


def evaluate_rl_method(
    config: LatencyEvalConfig,
    registry: pd.DataFrame,
    scenario: ScenarioConfig,
    method: str,
    ticker: str,
    episode_dir: Path,
) -> dict:
    scope, agent = method.split("_", 1)
    row = rl_model_row(registry, scope, agent, ticker)
    env = build_env(config, scenario, ticker, agent)
    wrapped_env = Monitor(env)
    model_cls = PPO if agent == "ppo" else DQN
    experiment_root = Path(config.experiment_root)
    model = model_cls.load(local_artifact_path(experiment_root, str(row["best_model"])), env=wrapped_env)
    episodes, summary = evaluate_agent(model, wrapped_env, config.eval_episodes)
    episode_path = episode_dir / scenario.name / method / f"{ticker}.csv"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(episode_path, index=False)
    return {
        "scenario": scenario.name,
        "method": method,
        "scope": scope,
        "agent": agent,
        "ticker": ticker,
        **asdict(scenario),
        **summary,
        "episode_results": str(episode_path),
    }


def baseline_name(method: str) -> str:
    name = method.removeprefix("baseline_")
    if name == "random_5level":
        return "random"
    return name


def evaluate_baseline_method(
    config: LatencyEvalConfig,
    scenario: ScenarioConfig,
    method: str,
    ticker: str,
    episode_dir: Path,
) -> dict:
    policy = make_baseline_policy(baseline_name(method), seed=config.seed, random_max_level=config.random_max_level)
    if isinstance(policy, AvellanedaStoikovPolicy):
        policy.gamma = config.as_gamma
        policy.kappa = config.as_kappa
        policy.sigma_window = config.as_sigma_window
    env = build_env(config, scenario, ticker, agent=None)
    episodes, summary = evaluate_quote_policy(policy, env, config.eval_episodes)
    episode_path = episode_dir / scenario.name / method / f"{ticker}.csv"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(episode_path, index=False)
    return {
        "scenario": scenario.name,
        "method": method,
        "scope": "baseline",
        "agent": baseline_name(method),
        "ticker": ticker,
        **asdict(scenario),
        **summary,
        "episode_results": str(episode_path),
    }


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    numeric_columns = summary.select_dtypes(include="number").columns.tolist()
    aggregate = summary.groupby(["scenario", "method"], as_index=False)[numeric_columns].mean()
    ticker_counts = summary.groupby(["scenario", "method"])["ticker"].nunique()
    aggregate.insert(
        2,
        "ticker_count",
        [int(ticker_counts.loc[(row.scenario, row.method)]) for row in aggregate.itertuples(index=False)],
    )
    return aggregate


def plot_outputs(aggregate: pd.DataFrame, output_dir: Path) -> None:
    if aggregate.empty or "pnl_mean" not in aggregate:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    pivot = aggregate.pivot(index="scenario", columns="method", values="pnl_mean")
    ax = pivot.plot(kind="bar", figsize=(14, 7), width=0.85)
    ax.set_title("Mean Test PnL by Latency Scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Mean PnL")
    ax.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "latency_pnl_mean_by_scenario.png", dpi=180)
    plt.close()

    if "paper_base" in pivot.index:
        delta = pivot.subtract(pivot.loc["paper_base"], axis=1)
        ax = delta.plot(kind="line", marker="o", figsize=(14, 7))
        ax.set_title("PnL Change from Paper-Style Base Scenario")
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Mean PnL Delta")
        ax.axhline(0.0, color="black", linewidth=0.8)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "latency_pnl_delta_from_base.png", dpi=180)
        plt.close()

    if "decision_time_ms_mean_mean" in aggregate:
        fig, ax = plt.subplots(figsize=(10, 7))
        for method, rows in aggregate.groupby("method"):
            ax.scatter(rows["decision_time_ms_mean_mean"], rows["pnl_mean"], label=method, s=40)
        ax.set_title("Measured Decision Time vs Mean PnL")
        ax.set_xlabel("Measured decision time mean (ms)")
        ax.set_ylabel("Mean PnL")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(output_dir / "latency_decision_time_vs_pnl.png", dpi=180)
        plt.close()

    if "order_drop_rate_mean" in aggregate:
        fig, ax = plt.subplots(figsize=(10, 7))
        for method, rows in aggregate.groupby("method"):
            ax.scatter(rows["order_drop_rate_mean"], rows["pnl_mean"], label=method, s=40)
        ax.set_title("Order Drop Rate vs Mean PnL")
        ax.set_xlabel("Realized order drop rate")
        ax.set_ylabel("Mean PnL")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(output_dir / "latency_drop_rate_vs_pnl.png", dpi=180)
        plt.close()


def main() -> None:
    config = parse_args()
    experiment_root = Path(config.experiment_root)
    output_dir = Path(config.output_dir) if config.output_dir else experiment_root / "latency_scenarios"
    reports_dir = output_dir / "reports"
    plots_dir = output_dir / "plots"
    episode_dir = output_dir / "episodes"
    reports_dir.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(experiment_root / "reports" / "model_registry.csv")
    tickers = available_tickers(registry, config)
    methods = parse_list(config.methods)
    scenarios = [scenario_from_name(name, config) for name in parse_scenario_names(config.scenarios)]

    rows: list[dict] = []
    for scenario in scenarios:
        for method in methods:
            for ticker in tickers:
                print(f"scenario={scenario.name} method={method} ticker={ticker}")
                if method.startswith("baseline_"):
                    rows.append(evaluate_baseline_method(config, scenario, method, ticker, episode_dir))
                else:
                    rows.append(evaluate_rl_method(config, registry, scenario, method, ticker, episode_dir))

    summary = pd.DataFrame(rows)
    aggregate = aggregate_summary(summary)
    summary_path = reports_dir / "latency_summary.csv"
    aggregate_path = reports_dir / "latency_aggregate.csv"
    summary.to_csv(summary_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    (reports_dir / "metadata.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "scenarios": [asdict(scenario) for scenario in scenarios],
                "tickers": tickers,
                "methods": methods,
                "summary": str(summary_path),
                "aggregate": str(aggregate_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_outputs(aggregate, plots_dir)
    print(f"summary_written: {summary_path}")
    print(f"aggregate_written: {aggregate_path}")
    print(f"plots_written: {plots_dir}")


if __name__ == "__main__":
    main()
