from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_TICKERS = [
    "000004.SZ",
    "002193.SZ",
    "002205.SZ",
    "002231.SZ",
    "002858.SZ",
    "002780.SZ",
    "003003.SZ",
    "301228.SZ",
    "002199.SZ",
    "300106.SZ",
]

DEFAULT_TRADING_DATE = "20250121-20250127"
DEFAULT_SINGLE_TAG = "SZ10_single_best_modified_20250121-20250127"
DEFAULT_JOINT_TAG = "SZ10_joint_best_modified_20250121-20250127"

BEST_MODIFIED_ATTN_LOB = {
    "optimizer": "adamw",
    "loss": "weighted_cross_entropy",
    "embedding_dim": "96",
    "conv_channels": "32",
    "inception_channels": "96",
}


@dataclass
class TradingSystemConfig:
    code_root: Path = Path(__file__).resolve().parent
    processed_root: Path = Path("data/sz_l2_processed")
    output_root: Path = Path("experiment_results/sz10_full_experiment")
    device: str = "cuda"
    python_executable: str = sys.executable

    def resolve(self) -> "TradingSystemConfig":
        code_root = self.code_root.resolve()
        return TradingSystemConfig(
            code_root=code_root,
            processed_root=(code_root / self.processed_root).resolve()
            if not self.processed_root.is_absolute()
            else self.processed_root,
            output_root=(code_root / self.output_root).resolve()
            if not self.output_root.is_absolute()
            else self.output_root,
            device=self.device,
            python_executable=self.python_executable,
        )

    @property
    def logs_dir(self) -> Path:
        return self.output_root / "logs"


@dataclass
class CommandResult:
    command: list[str]
    log_path: Path
    skipped: bool = False


def parse_ticker_list(tickers: str | Sequence[str], config: TradingSystemConfig) -> list[str]:
    if isinstance(tickers, str):
        normalized = tickers.strip()
        if normalized.lower() in {"all", "selected", ""}:
            return selected_tickers(config)
        return [item.strip() for item in normalized.split(",") if item.strip()]
    return list(tickers)


def selected_tickers(config: TradingSystemConfig) -> list[str]:
    config = config.resolve()
    selected_path = config.processed_root / "selected_symbols.csv"
    if not selected_path.exists():
        return DEFAULT_TICKERS
    with selected_path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        tickers = [row["symbol"] for row in rows if row.get("symbol")]
    return tickers or DEFAULT_TICKERS


def run_command(command: list[str], log_path: Path, *, cwd: Path, force: bool = False, done_path: Path | None = None) -> CommandResult:
    if done_path is not None and done_path.exists() and not force:
        return CommandResult(command=command, log_path=log_path, skipped=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(command, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}; see {log_path}")
    return CommandResult(command=command, log_path=log_path, skipped=False)


def attn_experiment_name(tickers: Sequence[str], trading_date: str) -> str:
    return f"{'-'.join(tickers)}_{trading_date}"


def attn_output_dir(config: TradingSystemConfig, mode: str, tickers: Sequence[str], trading_date: str) -> Path:
    return config.output_root / f"attn_lob_{mode}" / attn_experiment_name(tickers, trading_date)


def encoder_paths(config: TradingSystemConfig, mode: str, ticker: str, encoder_tickers: Sequence[str]) -> tuple[Path, Path]:
    if mode == "single":
        base = attn_output_dir(config, "single", [ticker], DEFAULT_SINGLE_TAG)
    elif mode == "joint":
        base = attn_output_dir(config, "joint", encoder_tickers, DEFAULT_JOINT_TAG)
    else:
        raise ValueError(f"encoder mode must be single or joint, got {mode}")
    return base / "best_encoder.pt", base / "metadata.json"


def pretrain(
    config: TradingSystemConfig,
    *,
    tickers: str | Sequence[str] = "all",
    mode: str = "joint",
    epochs: int = 50,
    batch_size: int = 512,
    trading_date: str | None = None,
    force: bool = False,
) -> list[CommandResult]:
    config = config.resolve()
    selected = parse_ticker_list(tickers, config)
    if mode not in {"single", "joint"}:
        raise ValueError("pretrain mode must be single or joint")
    jobs = [[ticker] for ticker in selected] if mode == "single" else [selected]
    results: list[CommandResult] = []
    output_base = config.output_root / f"attn_lob_{mode}"
    for job_tickers in jobs:
        tag = trading_date or (DEFAULT_SINGLE_TAG if mode == "single" else DEFAULT_JOINT_TAG)
        command = [
            config.python_executable,
            "train_attn_lob.py",
            "--dataset-format",
            "processed",
            "--tickers",
            ",".join(job_tickers),
            "--trading-date",
            tag,
            "--data-root",
            str(config.processed_root),
            "--output-dir",
            str(output_base),
            "--device",
            config.device,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--optimizer",
            BEST_MODIFIED_ATTN_LOB["optimizer"],
            "--loss",
            BEST_MODIFIED_ATTN_LOB["loss"],
            "--embedding-dim",
            BEST_MODIFIED_ATTN_LOB["embedding_dim"],
            "--conv-channels",
            BEST_MODIFIED_ATTN_LOB["conv_channels"],
            "--inception-channels",
            BEST_MODIFIED_ATTN_LOB["inception_channels"],
        ]
        done_path = attn_output_dir(config, mode, job_tickers, tag) / "results_summary_latest.json"
        log_name = f"api_pretrain_{mode}_{'-'.join(job_tickers)}.log"
        results.append(run_command(command, config.logs_dir / log_name, cwd=config.code_root, force=force, done_path=done_path))
    return results


def train_rl(
    config: TradingSystemConfig,
    *,
    tickers: str | Sequence[str] = "all",
    agent: str = "ppo",
    encoder_mode: str = "joint",
    timesteps: int = 20_000,
    eval_episodes: int = 10,
    force: bool = False,
) -> list[CommandResult]:
    config = config.resolve()
    selected = parse_ticker_list(tickers, config)
    if agent not in {"ppo", "dqn", "dueling_dqn"}:
        raise ValueError("agent must be ppo, dqn, or dueling_dqn")
    if encoder_mode not in {"single", "joint"}:
        raise ValueError("encoder_mode must be single or joint")
    output_base = config.output_root / f"rl_{encoder_mode}"
    results: list[CommandResult] = []
    for ticker in selected:
        encoder_path, encoder_metadata_path = encoder_paths(config, encoder_mode, ticker, selected_tickers(config))
        command = [
            config.python_executable,
            "train_rl_agent.py",
            "--agent",
            agent,
            "--ticker",
            ticker,
            "--processed-root",
            str(config.processed_root),
            "--output-dir",
            str(output_base),
            "--encoder-path",
            str(encoder_path),
            "--encoder-metadata-path",
            str(encoder_metadata_path),
            "--total-timesteps",
            str(timesteps),
            "--eval-episodes",
            str(eval_episodes),
            "--eval-model",
            "best",
        ]
        done_path = output_base / agent / ticker / "metadata.json"
        log_name = f"api_rl_{encoder_mode}_{agent}_{ticker}.log"
        results.append(run_command(command, config.logs_dir / log_name, cwd=config.code_root, force=force, done_path=done_path))
    return results


def evaluate_baselines(
    config: TradingSystemConfig,
    *,
    tickers: str | Sequence[str] = "all",
    baselines: str = "fixed_1,fixed_2,fixed_3,random,as",
    eval_episodes: int = 10,
    force: bool = False,
) -> list[CommandResult]:
    config = config.resolve()
    results: list[CommandResult] = []
    output_base = config.output_root / "baselines"
    for ticker in parse_ticker_list(tickers, config):
        command = [
            config.python_executable,
            "evaluate_baselines.py",
            "--ticker",
            ticker,
            "--split",
            "test",
            "--processed-root",
            str(config.processed_root),
            "--output-dir",
            str(output_base),
            "--baselines",
            baselines,
            "--eval-episodes",
            str(eval_episodes),
        ]
        done_path = output_base / ticker / "test" / "summary.csv"
        log_name = f"api_baselines_{ticker}.log"
        results.append(run_command(command, config.logs_dir / log_name, cwd=config.code_root, force=force, done_path=done_path))
    return results


def evaluate_latency(
    config: TradingSystemConfig,
    *,
    tickers: str = "",
    methods: str | None = None,
    scenarios: str | None = None,
    eval_episodes: int = 10,
    output_dir: str | Path | None = None,
) -> CommandResult:
    config = config.resolve()
    command = [
        config.python_executable,
        "evaluate_latency_scenarios.py",
        "--experiment-root",
        str(config.output_root),
        "--processed-root",
        str(config.processed_root),
        "--eval-episodes",
        str(eval_episodes),
    ]
    if tickers:
        command += ["--tickers", tickers]
    if methods:
        command += ["--methods", methods]
    if scenarios:
        command += ["--scenarios", scenarios]
    if output_dir:
        command += ["--output-dir", str(output_dir)]
    return run_command(command, config.logs_dir / "api_latency.log", cwd=config.code_root, force=True)


def analyze_regimes(config: TradingSystemConfig, *, output_dir: str | Path | None = None) -> CommandResult:
    config = config.resolve()
    command = [
        config.python_executable,
        "analyze_regime_sensitivity.py",
        "--experiment-root",
        str(config.output_root),
        "--processed-root",
        str(config.processed_root),
    ]
    if output_dir:
        command += ["--output-dir", str(output_dir)]
    return run_command(command, config.logs_dir / "api_regime_sensitivity.log", cwd=config.code_root, force=True)


def summarize(config: TradingSystemConfig) -> list[CommandResult]:
    config = config.resolve()
    summarize_command = [
        config.python_executable,
        "run_sz10_full_experiment.py",
        "--stage",
        "summarize",
        "--output-root",
        str(config.output_root),
        "--processed-root",
        str(config.processed_root),
    ]
    plot_command = [
        config.python_executable,
        "run_sz10_full_experiment.py",
        "--stage",
        "plots",
        "--output-root",
        str(config.output_root),
        "--processed-root",
        str(config.processed_root),
        "--force-plots",
    ]
    return [
        run_command(summarize_command, config.logs_dir / "api_summarize.log", cwd=config.code_root, force=True),
        run_command(plot_command, config.logs_dir / "api_final_plots.log", cwd=config.code_root, force=True),
    ]


def run_full_pipeline(
    config: TradingSystemConfig,
    *,
    stage: str = "all",
    epochs: int = 50,
    rl_timesteps: int = 20_000,
    eval_episodes: int = 10,
    rl_workers: int = 4,
    baseline_workers: int = 4,
) -> CommandResult:
    config = config.resolve()
    command = [
        config.python_executable,
        "run_sz10_full_experiment.py",
        "--stage",
        stage,
        "--output-root",
        str(config.output_root),
        "--processed-root",
        str(config.processed_root),
        "--device",
        config.device,
        "--epochs",
        str(epochs),
        "--rl-timesteps",
        str(rl_timesteps),
        "--eval-episodes",
        str(eval_episodes),
        "--rl-workers",
        str(rl_workers),
        "--baseline-workers",
        str(baseline_workers),
    ]
    return run_command(command, config.logs_dir / f"api_full_{stage}.log", cwd=config.code_root, force=True)
