from __future__ import annotations

import argparse
import sys
from pathlib import Path

from system_api import (
    TradingSystemConfig,
    analyze_regimes,
    evaluate_baselines,
    evaluate_latency,
    parse_ticker_list,
    pretrain,
    run_full_pipeline,
    selected_tickers,
    summarize,
    train_rl,
)


def base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified API/CLI for the Attn-LOB market-making system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--processed-root", default="data/sz_l2_processed")
    parser.add_argument("--output-root", default="experiment_results/sz10_full_experiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-tickers", help="Print selected symbols from processed data.")

    pretrain_parser = subparsers.add_parser("pretrain", help="Pre-train Attn-LOB encoders.")
    pretrain_parser.add_argument("--mode", choices=("single", "joint"), default="joint")
    pretrain_parser.add_argument("--tickers", default="all", help="Comma-separated symbols or 'all'.")
    pretrain_parser.add_argument("--epochs", type=int, default=50)
    pretrain_parser.add_argument("--batch-size", type=int, default=512)
    pretrain_parser.add_argument("--force", action="store_true")

    rl_parser = subparsers.add_parser("rl", help="Train RL market-making agents.")
    rl_parser.add_argument("--tickers", default="all", help="Comma-separated symbols or 'all'.")
    rl_parser.add_argument("--agent", choices=("ppo", "dqn", "dueling_dqn"), default="ppo")
    rl_parser.add_argument("--encoder-mode", choices=("single", "joint"), default="joint")
    rl_parser.add_argument("--timesteps", type=int, default=20_000)
    rl_parser.add_argument("--eval-episodes", type=int, default=10)
    rl_parser.add_argument("--force", action="store_true")

    baseline_parser = subparsers.add_parser("baseline", help="Backtest non-learning quote baselines.")
    baseline_parser.add_argument("--tickers", default="all", help="Comma-separated symbols or 'all'.")
    baseline_parser.add_argument("--baselines", default="fixed_1,fixed_2,fixed_3,random,as")
    baseline_parser.add_argument("--eval-episodes", type=int, default=10)
    baseline_parser.add_argument("--force", action="store_true")

    latency_parser = subparsers.add_parser("latency", help="Run latency/order-loss/inference-delay stress tests.")
    latency_parser.add_argument("--tickers", default="", help="Optional comma-separated symbols. Empty means all registered models.")
    latency_parser.add_argument("--methods", default=None, help="Optional comma-separated methods.")
    latency_parser.add_argument("--scenarios", default=None, help="Optional scenario names, or custom scenarios separated by semicolons.")
    latency_parser.add_argument("--eval-episodes", type=int, default=10)
    latency_parser.add_argument("--output-dir", default=None)

    regime_parser = subparsers.add_parser("regime", help="Analyze saved backtests by volatility/spread/trade-intensity regime.")
    regime_parser.add_argument("--output-dir", default=None)

    subparsers.add_parser("summarize", help="Rebuild result tables and plots from saved runs.")

    full_parser = subparsers.add_parser("full", help="Run or resume the full SZ10 pipeline.")
    full_parser.add_argument("--stage", choices=("all", "pretrain", "rl", "baselines", "summarize", "plots"), default="all")
    full_parser.add_argument("--epochs", type=int, default=50)
    full_parser.add_argument("--rl-timesteps", type=int, default=20_000)
    full_parser.add_argument("--eval-episodes", type=int, default=10)
    full_parser.add_argument("--rl-workers", type=int, default=4)
    full_parser.add_argument("--baseline-workers", type=int, default=4)
    return parser


def build_config(args) -> TradingSystemConfig:
    return TradingSystemConfig(
        processed_root=Path(args.processed_root),
        output_root=Path(args.output_root),
        device=args.device,
        python_executable=args.python,
    ).resolve()


def print_results(results) -> None:
    if not isinstance(results, list):
        results = [results]
    for result in results:
        status = "skipped" if result.skipped else "done"
        print(f"{status}: log={result.log_path}")


def main() -> None:
    parser = base_parser()
    args = parser.parse_args()
    config = build_config(args)

    if args.command == "list-tickers":
        print("\n".join(selected_tickers(config)))
        return

    if args.command == "pretrain":
        print(f"pretrain mode={args.mode} tickers={parse_ticker_list(args.tickers, config)}")
        print_results(
            pretrain(
                config,
                mode=args.mode,
                tickers=args.tickers,
                epochs=args.epochs,
                batch_size=args.batch_size,
                force=args.force,
            )
        )
        return

    if args.command == "rl":
        print(f"rl agent={args.agent} encoder_mode={args.encoder_mode} tickers={parse_ticker_list(args.tickers, config)}")
        print_results(
            train_rl(
                config,
                tickers=args.tickers,
                agent=args.agent,
                encoder_mode=args.encoder_mode,
                timesteps=args.timesteps,
                eval_episodes=args.eval_episodes,
                force=args.force,
            )
        )
        return

    if args.command == "baseline":
        print(f"baseline tickers={parse_ticker_list(args.tickers, config)} baselines={args.baselines}")
        print_results(
            evaluate_baselines(
                config,
                tickers=args.tickers,
                baselines=args.baselines,
                eval_episodes=args.eval_episodes,
                force=args.force,
            )
        )
        return

    if args.command == "latency":
        print("latency stress test")
        print_results(
            evaluate_latency(
                config,
                tickers=args.tickers,
                methods=args.methods,
                scenarios=args.scenarios,
                eval_episodes=args.eval_episodes,
                output_dir=args.output_dir,
            )
        )
        return

    if args.command == "regime":
        print("regime sensitivity")
        print_results(analyze_regimes(config, output_dir=args.output_dir))
        return

    if args.command == "summarize":
        print_results(summarize(config))
        return

    if args.command == "full":
        print_results(
            run_full_pipeline(
                config,
                stage=args.stage,
                epochs=args.epochs,
                rl_timesteps=args.rl_timesteps,
                eval_episodes=args.eval_episodes,
                rl_workers=args.rl_workers,
                baseline_workers=args.baseline_workers,
            )
        )
        return

    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
