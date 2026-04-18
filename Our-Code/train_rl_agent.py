from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rl_trading.env import (
    LOBContinuousMarketMakingEnv,
    LOBDiscreteMarketMakingEnv,
    LOBEnvConfig,
)
from rl_trading.model import MultiInputLOBExtractor

try:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:  # pragma: no cover - dependency availability is environment-specific
    raise ImportError(
        "train_rl_agent.py requires stable-baselines3 and gymnasium. "
        "Please install the RL dependencies from requirements.txt."
    ) from exc


@dataclass
class RLTrainConfig:
    agent: str = "ppo"
    ticker: str = "AAPL"
    split: str = "train"
    eval_split: str = "val"
    processed_root: str = "../lobster_attnlob_us_repro/data/lobster_processed"
    output_dir: str = "checkpoints/rl"
    encoder_path: str = "checkpoints/attn_lob/AAPL_2012-06-21/best_encoder.pt"
    time_window: int = 50
    episode_length: int = 2000
    trade_unit: int = 100
    max_inventory_units: int = 10
    tick_size: float = 0.01
    max_bias: float = 0.05
    max_spread: float = 0.10
    reward_eta: float = 0.5
    reward_theta: float = 0.01
    freeze_encoder: bool = True
    extractor_dim: int = 128
    total_timesteps: int = 20_000
    learning_rate: float = 3e-4
    batch_size: int = 256
    n_steps: int = 1024
    gamma: float = 0.99
    buffer_size: int = 100_000
    learning_starts: int = 1_000
    train_freq: int = 4
    target_update_interval: int = 1_000
    seed: int = 7


def parse_args() -> RLTrainConfig:
    parser = argparse.ArgumentParser(description="Train PPO or DQN market making agent in PyTorch.")
    parser.add_argument("--agent", choices=("ppo", "dqn"), default="ppo")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--eval-split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--processed-root", default="../lobster_attnlob_us_repro/data/lobster_processed")
    parser.add_argument("--output-dir", default="checkpoints/rl")
    parser.add_argument("--encoder-path", default="checkpoints/attn_lob/AAPL_2012-06-21/best_encoder.pt")
    parser.add_argument("--time-window", type=int, default=50)
    parser.add_argument("--episode-length", type=int, default=2000)
    parser.add_argument("--trade-unit", type=int, default=100)
    parser.add_argument("--max-inventory-units", type=int, default=10)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--max-bias", type=float, default=0.05)
    parser.add_argument("--max-spread", type=float, default=0.10)
    parser.add_argument("--reward-eta", type=float, default=0.5)
    parser.add_argument("--reward-theta", type=float, default=0.01)
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extractor-dim", type=int, default=128)
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    return RLTrainConfig(**vars(parser.parse_args()))


def build_env(config: RLTrainConfig, split: str):
    env_config = LOBEnvConfig(
        ticker=config.ticker,
        split=split,
        processed_root=config.processed_root,
        time_window=config.time_window,
        episode_length=config.episode_length,
        trade_unit=config.trade_unit,
        max_inventory_units=config.max_inventory_units,
        tick_size=config.tick_size,
        max_bias=config.max_bias,
        max_spread=config.max_spread,
        reward_eta=config.reward_eta,
        reward_theta=config.reward_theta,
        seed=config.seed,
    )
    if config.agent == "ppo":
        env = LOBContinuousMarketMakingEnv(env_config)
    else:
        env = LOBDiscreteMarketMakingEnv(env_config)
    return Monitor(env)


def build_agent(config: RLTrainConfig, env):
    policy_kwargs = dict(
        features_extractor_class=MultiInputLOBExtractor,
        features_extractor_kwargs=dict(
            time_window=config.time_window,
            encoder_path=config.encoder_path,
            freeze_encoder=config.freeze_encoder,
            output_dim=config.extractor_dim,
        ),
        net_arch=[256, 256],
    )
    if config.agent == "ppo":
        return PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            gamma=config.gamma,
            seed=config.seed,
            policy_kwargs=policy_kwargs,
            verbose=1,
        )
    return DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        gamma=config.gamma,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        train_freq=config.train_freq,
        target_update_interval=config.target_update_interval,
        seed=config.seed,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )


def main() -> None:
    config = parse_args()
    train_env = build_env(config, config.split)
    eval_env = build_env(config, config.eval_split)
    agent = build_agent(config, train_env)

    output_dir = Path(config.output_dir) / config.agent / config.ticker
    output_dir.mkdir(parents=True, exist_ok=True)
    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best"),
        log_path=str(output_dir / "eval"),
        eval_freq=max(config.episode_length, 1_000),
        deterministic=True,
        render=False,
    )
    agent.learn(total_timesteps=config.total_timesteps, callback=callback)
    agent.save(output_dir / "final_model")

    metadata = {
        "config": asdict(config),
        "output_dir": str(output_dir),
        "best_model_dir": str(output_dir / "best"),
        "final_model_path": str(output_dir / "final_model"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
