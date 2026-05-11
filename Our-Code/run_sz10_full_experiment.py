from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import pandas as pd


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

TRADING_DATE_SINGLE = "SZ10_single_best_modified_20250121-20250127"
TRADING_DATE_JOINT = "SZ10_joint_best_modified_20250121-20250127"
ATNN_SETTING = {
    "optimizer": "adamw",
    "loss": "weighted_cross_entropy",
    "embedding_dim": "96",
    "conv_channels": "32",
    "inception_channels": "96",
}
RL_AGENTS = ["dueling_dqn", "ppo"]
BASELINES = ["fixed_1", "fixed_2", "fixed_3", "random", "as"]
PAPER_METRICS = [
    "pnl",
    "nd_pnl",
    "pnl_map",
    "profit_ratio",
    "avg_position",
    "avg_abs_position",
    "avg_spread",
    "volume",
    "episode_reward",
    "reward_dampened_pnl",
    "reward_trading_pnl",
    "reward_inventory_penalty",
    "transaction_cost",
    "sharpe",
]


@dataclass
class ExperimentPaths:
    code_root: Path
    output_root: Path
    processed_root: Path

    @property
    def logs(self) -> Path:
        return self.output_root / "logs"

    @property
    def reports(self) -> Path:
        return self.output_root / "reports"


def experiment_name(tickers: Iterable[str], trading_date: str) -> str:
    return f"{'-'.join(tickers)}_{trading_date}"


def read_tickers(paths: ExperimentPaths) -> list[str]:
    selected = paths.processed_root / "selected_symbols.csv"
    if not selected.exists():
        return DEFAULT_TICKERS
    with selected.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        tickers = [row["symbol"] for row in reader if row.get("symbol")]
    return tickers[:10] or DEFAULT_TICKERS


def run_logged(
    command: list[str],
    log_path: Path,
    *,
    cwd: Path,
    done_path: Path | None = None,
    force: bool = False,
) -> None:
    if done_path is not None and done_path.exists() and not force:
        print(f"skip existing: {done_path}")
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("run:", " ".join(command))
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"command failed with exit code {process.returncode}; see {log_path}")


def attn_output_dir(paths: ExperimentPaths, scope: str, tickers: list[str], trading_date: str) -> Path:
    return paths.output_root / f"attn_lob_{scope}" / experiment_name(tickers, trading_date)


def pretrain_command(paths: ExperimentPaths, tickers: list[str], trading_date: str, output_dir: Path, args) -> list[str]:
    return [
        sys.executable,
        "train_attn_lob.py",
        "--dataset-format",
        "processed",
        "--tickers",
        ",".join(tickers),
        "--trading-date",
        trading_date,
        "--data-root",
        str(paths.processed_root),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--optimizer",
        ATNN_SETTING["optimizer"],
        "--loss",
        ATNN_SETTING["loss"],
        "--embedding-dim",
        ATNN_SETTING["embedding_dim"],
        "--conv-channels",
        ATNN_SETTING["conv_channels"],
        "--inception-channels",
        ATNN_SETTING["inception_channels"],
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--precision",
        args.precision,
        "--channels-last",
    ]


def run_pretrain(paths: ExperimentPaths, tickers: list[str], args) -> None:
    single_root = paths.output_root / "attn_lob_single"
    joint_root = paths.output_root / "attn_lob_joint"
    for ticker in tickers:
        output = attn_output_dir(paths, "single", [ticker], TRADING_DATE_SINGLE)
        command = pretrain_command(paths, [ticker], TRADING_DATE_SINGLE, single_root, args)
        run_logged(
            command,
            paths.logs / f"pretrain_single_{ticker}.log",
            cwd=paths.code_root,
            done_path=output / "results_summary_latest.json",
            force=args.force_pretrain,
        )

    joint_output = attn_output_dir(paths, "joint", tickers, TRADING_DATE_JOINT)
    joint_command = pretrain_command(paths, tickers, TRADING_DATE_JOINT, joint_root, args)
    run_logged(
        joint_command,
        paths.logs / "pretrain_joint10.log",
        cwd=paths.code_root,
        done_path=joint_output / "results_summary_latest.json",
        force=args.force_pretrain,
    )


def encoder_paths(paths: ExperimentPaths, scope: str, ticker: str, tickers: list[str]) -> tuple[Path, Path]:
    if scope == "single":
        base = attn_output_dir(paths, "single", [ticker], TRADING_DATE_SINGLE)
    elif scope == "joint":
        base = attn_output_dir(paths, "joint", tickers, TRADING_DATE_JOINT)
    else:
        raise ValueError(f"unknown scope: {scope}")
    return base / "best_encoder.pt", base / "metadata.json"


def rl_output_root(paths: ExperimentPaths, scope: str) -> Path:
    return paths.output_root / f"rl_{scope}"


def run_one_rl(paths: ExperimentPaths, scope: str, ticker: str, agent: str, tickers: list[str], args) -> None:
    encoder_path, metadata_path = encoder_paths(paths, scope, ticker, tickers)
    if not encoder_path.exists():
        raise FileNotFoundError(f"missing encoder for {scope}/{ticker}: {encoder_path}")
    output_root = rl_output_root(paths, scope)
    done_path = output_root / agent / ticker / "metadata.json"
    command = [
        sys.executable,
        "train_rl_agent.py",
        "--agent",
        agent,
        "--ticker",
        ticker,
        "--processed-root",
        str(paths.processed_root),
        "--output-dir",
        str(output_root),
        "--encoder-path",
        str(encoder_path),
        "--encoder-metadata-path",
        str(metadata_path),
        "--total-timesteps",
        str(args.rl_timesteps),
        "--eval-episodes",
        str(args.eval_episodes),
        "--eval-model",
        "best",
    ]
    run_logged(
        command,
        paths.logs / f"rl_{scope}_{agent}_{ticker}.log",
        cwd=paths.code_root,
        done_path=done_path,
        force=args.force_rl,
    )


def run_rl(paths: ExperimentPaths, tickers: list[str], args) -> None:
    jobs = [(scope, ticker, agent) for scope in ["single", "joint"] for ticker in tickers for agent in RL_AGENTS]
    with ThreadPoolExecutor(max_workers=args.rl_workers) as executor:
        futures = [
            executor.submit(run_one_rl, paths, scope, ticker, agent, tickers, args)
            for scope, ticker, agent in jobs
        ]
        for future in as_completed(futures):
            future.result()


def run_one_baseline(paths: ExperimentPaths, ticker: str, args) -> None:
    output_root = paths.output_root / "baselines"
    done_path = output_root / ticker / "test" / "summary.csv"
    command = [
        sys.executable,
        "evaluate_baselines.py",
        "--ticker",
        ticker,
        "--split",
        "test",
        "--processed-root",
        str(paths.processed_root),
        "--output-dir",
        str(output_root),
        "--baselines",
        ",".join(BASELINES),
        "--eval-episodes",
        str(args.eval_episodes),
    ]
    run_logged(
        command,
        paths.logs / f"baselines_{ticker}.log",
        cwd=paths.code_root,
        done_path=done_path,
        force=args.force_baselines,
    )


def run_baselines(paths: ExperimentPaths, tickers: list[str], args) -> None:
    with ThreadPoolExecutor(max_workers=args.baseline_workers) as executor:
        futures = [executor.submit(run_one_baseline, paths, ticker, args) for ticker in tickers]
        for future in as_completed(futures):
            future.result()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def local_artifact_path(paths: ExperimentPaths, value: str) -> str:
    """Map artifact paths from an old experiment root to the current one."""
    artifact_path = Path(value)
    if artifact_path.exists():
        return str(artifact_path)
    marker = f"checkpoints/{paths.output_root.name}/"
    if marker in value:
        return str(paths.output_root / value.split(marker, 1)[1])
    marker = f"experiment_results/{paths.output_root.name}/"
    if marker in value:
        return str(paths.output_root / value.split(marker, 1)[1])
    return value


def flatten_summary(prefix: str, summary: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def collect_pretrain(paths: ExperimentPaths, tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        summary_path = attn_output_dir(paths, "single", [ticker], TRADING_DATE_SINGLE) / "results_summary_latest.json"
        payload = read_json(summary_path)
        best = payload["best_validation"]
        test = payload["test_metrics"]
        rows.append(
            {
                "scope": "single",
                "ticker": ticker,
                "encoder_tickers": ticker,
                "best_epoch": best["epoch"],
                "val_accuracy": best["accuracy"],
                "val_precision": best["precision"],
                "val_recall": best["recall"],
                "val_f1": best["f1"],
                "test_accuracy": test["accuracy"],
                "test_precision": test["precision"],
                "test_recall": test["recall"],
                "test_f1": test["f1"],
                "test_loss": test["loss"],
                "best_model": local_artifact_path(paths, payload["checkpoints"]["best_model"]),
                "best_encoder": local_artifact_path(paths, payload["checkpoints"]["best_encoder"]),
                "metadata": local_artifact_path(paths, payload["checkpoints"]["metadata"]),
            }
        )

    joint_path = attn_output_dir(paths, "joint", tickers, TRADING_DATE_JOINT) / "results_summary_latest.json"
    payload = read_json(joint_path)
    best = payload["best_validation"]
    test = payload["test_metrics"]
    rows.append(
        {
            "scope": "joint10",
            "ticker": "ALL_10",
            "encoder_tickers": ",".join(tickers),
            "best_epoch": best["epoch"],
            "val_accuracy": best["accuracy"],
            "val_precision": best["precision"],
            "val_recall": best["recall"],
            "val_f1": best["f1"],
            "test_accuracy": test["accuracy"],
            "test_precision": test["precision"],
            "test_recall": test["recall"],
            "test_f1": test["f1"],
            "test_loss": test["loss"],
            "best_model": local_artifact_path(paths, payload["checkpoints"]["best_model"]),
            "best_encoder": local_artifact_path(paths, payload["checkpoints"]["best_encoder"]),
            "metadata": local_artifact_path(paths, payload["checkpoints"]["metadata"]),
        }
    )
    return pd.DataFrame(rows)


def collect_rl(paths: ExperimentPaths, tickers: list[str]) -> pd.DataFrame:
    rows = []
    for scope in ["single", "joint"]:
        for agent in RL_AGENTS:
            for ticker in tickers:
                metadata_path = rl_output_root(paths, scope) / agent / ticker / "metadata.json"
                payload = read_json(metadata_path)
                row = {
                    "scope": scope,
                    "ticker": ticker,
                    "agent": agent,
                    "method": f"{scope}_{agent}",
                    "evaluation_model": payload["evaluation_model"],
                    "evaluation_model_path": local_artifact_path(paths, payload["evaluation_model_path"]),
                    "test_episode_results": local_artifact_path(paths, payload["test_episode_results"]),
                    "val_episode_results": local_artifact_path(paths, payload["val_episode_results"]),
                }
                row.update(flatten_summary("val", payload["val_summary"]))
                row.update(flatten_summary("test", payload["test_summary"]))
                rows.append(row)
    return pd.DataFrame(rows)


def collect_baselines(paths: ExperimentPaths, tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        summary_path = paths.output_root / "baselines" / ticker / "test" / "summary.csv"
        frame = pd.read_csv(summary_path)
        for _, baseline_row in frame.iterrows():
            row = {"ticker": ticker, "baseline": baseline_row["baseline"], "method": f"baseline_{baseline_row['baseline']}"}
            for key, value in baseline_row.items():
                if key == "baseline":
                    continue
                row[f"test_{key}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = comparison.select_dtypes(include="number").columns.tolist()
    aggregate = comparison.groupby("method", as_index=False)[numeric_columns].mean()
    if "ticker_count" not in aggregate:
        aggregate.insert(1, "ticker_count", comparison.groupby("method")["ticker"].nunique().reindex(aggregate["method"]).to_numpy())
    return aggregate


def save_model_registry(paths: ExperimentPaths, pretrain: pd.DataFrame, rl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in pretrain.iterrows():
        rows.append(
            {
                "kind": "attn_lob",
                "scope": row["scope"],
                "ticker": row["ticker"],
                "agent": "",
                "best_model": row["best_model"],
                "best_encoder": row["best_encoder"],
                "metadata": row["metadata"],
            }
        )
    for _, row in rl.iterrows():
        rows.append(
            {
                "kind": "rl",
                "scope": row["scope"],
                "ticker": row["ticker"],
                "agent": row["agent"],
                "best_model": row["evaluation_model_path"],
                "best_encoder": "",
                "metadata": str(rl_output_root(paths, row["scope"]) / row["agent"] / row["ticker"] / "metadata.json"),
            }
        )
    return pd.DataFrame(rows)


def summarize(paths: ExperimentPaths, tickers: list[str]) -> None:
    paths.reports.mkdir(parents=True, exist_ok=True)
    pretrain = collect_pretrain(paths, tickers)
    rl = collect_rl(paths, tickers)
    baselines = collect_baselines(paths, tickers)

    rl_comparison = rl[["method", "scope", "ticker", "agent"] + [f"test_{metric}_mean" for metric in PAPER_METRICS if f"test_{metric}_mean" in rl.columns]].copy()
    baseline_comparison = baselines[["method", "ticker"] + [f"test_{metric}_mean" for metric in PAPER_METRICS if f"test_{metric}_mean" in baselines.columns]].copy()
    baseline_comparison.insert(1, "scope", "baseline")
    baseline_comparison.insert(3, "agent", baseline_comparison["method"].str.removeprefix("baseline_"))
    comparison = pd.concat([rl_comparison, baseline_comparison], ignore_index=True, sort=False)
    aggregate = aggregate_comparison(comparison)
    model_registry = save_model_registry(paths, pretrain, rl)

    pretrain.to_csv(paths.reports / "pretrain_summary.csv", index=False)
    rl.to_csv(paths.reports / "rl_summary.csv", index=False)
    baselines.to_csv(paths.reports / "baseline_summary.csv", index=False)
    comparison.to_csv(paths.reports / "comparison_test_summary.csv", index=False)
    aggregate.to_csv(paths.reports / "comparison_test_aggregate.csv", index=False)
    model_registry.to_csv(paths.reports / "model_registry.csv", index=False)

    print(f"reports_written: {paths.reports}")
    print(aggregate.sort_values("test_pnl_mean", ascending=False).to_string(index=False))


def run_final_plots(paths: ExperimentPaths, args) -> None:
    done_path = paths.output_root / "final_plots" / "plot_index.csv"
    command = [
        sys.executable,
        "plot_final_results.py",
        "--experiment-root",
        str(paths.output_root),
        "--experiment-label",
        "10 Stocks",
    ]
    run_logged(command, paths.logs / "final_plots.log", cwd=paths.code_root, done_path=done_path, force=args.force_plots)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full SZ10 Attn-LOB + RL + baseline experiment.")
    parser.add_argument("--stage", choices=("all", "pretrain", "rl", "baselines", "summarize", "plots"), default="all")
    parser.add_argument("--output-root", default="experiment_results/sz10_full_experiment")
    parser.add_argument("--processed-root", default="data/sz_l2_processed")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--rl-timesteps", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--rl-workers", type=int, default=4)
    parser.add_argument("--baseline-workers", type=int, default=4)
    parser.add_argument("--force-pretrain", action="store_true")
    parser.add_argument("--force-rl", action="store_true")
    parser.add_argument("--force-baselines", action="store_true")
    parser.add_argument("--force-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code_root = Path(__file__).resolve().parent
    paths = ExperimentPaths(
        code_root=code_root,
        output_root=(code_root / args.output_root).resolve(),
        processed_root=(code_root / args.processed_root).resolve(),
    )
    paths.logs.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)

    tickers = read_tickers(paths)
    print("tickers:", ",".join(tickers))
    print("output_root:", paths.output_root)

    if args.stage in {"all", "pretrain"}:
        run_pretrain(paths, tickers, args)
    if args.stage in {"all", "rl"}:
        run_rl(paths, tickers, args)
    if args.stage in {"all", "baselines"}:
        run_baselines(paths, tickers, args)
    if args.stage in {"all", "summarize"}:
        summarize(paths, tickers)
    if args.stage in {"all", "plots"}:
        run_final_plots(paths, args)


if __name__ == "__main__":
    main()
