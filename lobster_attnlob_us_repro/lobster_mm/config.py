from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class ExperimentConfig:
    root_dir: Path = ROOT_DIR
    tickers: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "AMZN"])
    trading_date: str = "2012-06-21"
    sample_urls: Dict[str, str] = field(
        default_factory=lambda: {
            "AAPL": "https://data.lobsterdata.com/info/sample/LOBSTER_SampleFile_AAPL_2012-06-21_10.zip",
            "MSFT": "https://data.lobsterdata.com/info/sample/LOBSTER_SampleFile_MSFT_2012-06-21_10.zip",
            "AMZN": "https://data.lobsterdata.com/info/sample/LOBSTER_SampleFile_AMZN_2012-06-21_10.zip",
        }
    )
    price_scale: float = 10_000.0
    tick_size: float = 0.01
    levels: int = 10
    time_window: int = 50
    horizon_k: int = 10
    split_train: float = 0.60
    split_val: float = 0.20
    split_test: float = 0.20
    trade_unit: int = 100
    max_inventory_units: int = 10
    max_bias: float = 0.05
    max_spread: float = 0.10
    latency_events: int = 1
    episode_length: int = 2000
    reward_eta: float = 0.5
    reward_theta: float = 0.01
    pretrain_batch_size: int = 128
    pretrain_epochs: int = 5
    pretrain_lr: float = 1e-3
    ppo_total_timesteps: int = 20_000
    ppo_learning_rate: float = 3e-4
    ppo_batch_size: int = 256
    ppo_n_steps: int = 1024
    ppo_gamma: float = 0.99
    ppo_seed: int = 7
    as_gamma: float = 0.1
    as_kappa: float = 20.0
    as_sigma_window: int = 200
    extractor_dim: int = 128

    @property
    def raw_dir(self) -> Path:
        return self.root_dir / "data" / "lobster_raw"

    @property
    def processed_dir(self) -> Path:
        return self.root_dir / "data" / "lobster_processed"

    @property
    def outputs_dir(self) -> Path:
        return self.root_dir / "outputs"

    @property
    def pretrain_dir(self) -> Path:
        return self.outputs_dir / "pretrain"

    @property
    def rl_dir(self) -> Path:
        return self.outputs_dir / "rl"

    @property
    def evaluation_dir(self) -> Path:
        return self.outputs_dir / "evaluation"

    def ticker_raw_dir(self, ticker: str) -> Path:
        return self.raw_dir / ticker / self.trading_date

    def ticker_processed_dir(self, ticker: str) -> Path:
        return self.processed_dir / ticker

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["root_dir"] = str(self.root_dir)
        payload["raw_dir"] = str(self.raw_dir)
        payload["processed_dir"] = str(self.processed_dir)
        payload["outputs_dir"] = str(self.outputs_dir)
        payload["pretrain_dir"] = str(self.pretrain_dir)
        payload["rl_dir"] = str(self.rl_dir)
        payload["evaluation_dir"] = str(self.evaluation_dir)
        return payload
