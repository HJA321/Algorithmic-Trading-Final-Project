# Project Runbook

This document maps the current implementation to the course Option A requirements and lists the commands needed to reproduce the main results.

## Option A Checklist

| Requirement | Current status | Evidence / entry point |
| --- | --- | --- |
| Summary of original paper | Covered in presentation/report material, not code | `Market Making with Deep Reinforcement Learning from Limit Order Books.pdf` |
| Reproduction results | Covered | `experiment_results/sz10_full_experiment/reports/pretrain_summary.csv`, `rl_summary.csv` |
| Differences vs original | Covered | PyTorch Attn-LOB, Stable-Baselines3 PPO/Dueling-DQN, Shenzhen 2025 L2 data, single-stock vs joint encoder experiments |
| Robustness checks | Covered | `evaluate_latency_scenarios.py`, `experiment_results/sz10_full_experiment/latency_scenarios/` |
| Regime sensitivity | Covered | `analyze_regime_sensitivity.py`, `experiment_results/sz10_full_experiment/regime_sensitivity/` |
| Risk diagnostics | Mostly covered | PnL, ND-PnL, PnLMAP, Profit Ratio, Sharpe, inventory, spread, reward components |
| Clean data pipeline | Covered | `data_prep/reconstruct_sz_orderbook.py`, `data_prep/prepare_sz_l2_dataset.py` |
| Backtest engine | Covered | `rl_trading/env.py` |
| Transaction cost modeling | Interface covered; run cost scenarios before final presentation | `--transaction-cost-bps`, `--fixed-fee-per-trade` in RL/baseline/latency evaluators |
| Performance dashboard | Covered as static reports/plots; interactive UI optional | `reports/*.csv`, `final_plots/*.png`, robustness `plots/*.png` |
| Ability to run new simulations | Covered | CLI scripts below |
| No lookahead bias | Mostly covered | Train/val/test split is date based; RL normalization uses train split only; observations use past windows and `t-latency` state |
| Clear training vs testing separation | Covered | train `20250121-20250123`, val `20250124`, test `20250127` |
| Sensitivity analysis | Covered | latency/loss/inference stress tests, regime sensitivity, optional cost scenarios |

Remaining recommendation before the final deck:

- Run at least one explicit transaction-cost scenario, e.g. 1 bp and 5 bp, and report how rankings change.
- Include one slide explaining that the paper's latency is information latency, while our extension separates information latency, order-arrival latency, order loss, and inference delay.
- Include one slide or appendix table for regime sensitivity: low/mid/high volatility, spread, and trade-intensity regimes.

## Data Interface

Processed data lives under:

```text
data/sz_l2_processed/
  manifest.json
  selected_symbols.csv
  <ticker>/events.pkl.gz
  <ticker>/metadata.json
```

Each processed `events.pkl.gz` contains:

- 10-level order book columns: `ask_price_1 ... ask_size_10`, `bid_price_1 ... bid_size_10`
- event features: `is_execution`, `trade_price`, `trade_size`, `direction`, order/cancel/execution volumes
- split column: `split_id` where `0=train`, `1=val`, `2=test`
- pretraining label: `label` in `{-1, 0, 1}`

Prepare the selected Shenzhen dataset:

```bash
cd ..
./.venv/bin/python3 Our-Code/data_prep/prepare_sz_l2_dataset.py \
  --candidate-limit 20 \
  --validation-stride 10 \
  --num-symbols 10 \
  --output-root Our-Code/data/sz_l2_processed
```

## Main Experiment Interface

Run or resume the full SZ10 pipeline:

```bash
cd Our-Code
../.venv/bin/python3 run_sz10_full_experiment.py --stage all
```

Stages can be run independently:

```bash
../.venv/bin/python3 run_sz10_full_experiment.py --stage pretrain
../.venv/bin/python3 run_sz10_full_experiment.py --stage rl --rl-workers 4
../.venv/bin/python3 run_sz10_full_experiment.py --stage baselines --baseline-workers 4
../.venv/bin/python3 run_sz10_full_experiment.py --stage summarize
../.venv/bin/python3 run_sz10_full_experiment.py --stage plots --force-plots
```

Primary outputs:

```text
experiment_results/sz10_full_experiment/
  attn_lob_single/
  attn_lob_joint/
  rl_single/
  rl_joint/
  baselines/
  reports/
  final_plots/
  logs/
```

Most useful reports:

- `reports/pretrain_summary.csv`
- `reports/rl_summary.csv`
- `reports/baseline_summary.csv`
- `reports/comparison_test_summary.csv`
- `reports/comparison_test_aggregate.csv`
- `reports/model_registry.csv`

## Backtest / Environment Interface

The RL environment is configured through `LOBEnvConfig` in `rl_trading/env.py`. The important knobs are:

```text
latency                    paper-style information delay in events
order_latency_events       event-count order-arrival delay
order_latency_ms           wall-clock order-arrival delay
inference_latency_ms       simulated model calculation delay
order_loss_probability     independent bid/ask quote drop probability
transaction_cost_bps       proportional transaction cost
fixed_fee_per_trade        fixed cost per executed trade
```

All new execution-friction parameters default to zero, so old experiments remain reproducible.

## Baseline Interface

Run paper-style baselines:

```bash
cd Our-Code
../.venv/bin/python3 evaluate_baselines.py \
  --ticker 000004.SZ \
  --processed-root data/sz_l2_processed \
  --baselines fixed_1,fixed_2,fixed_3,random,as \
  --eval-episodes 10
```

Run the same baseline with explicit transaction costs:

```bash
../.venv/bin/python3 evaluate_baselines.py \
  --ticker 000004.SZ \
  --processed-root data/sz_l2_processed \
  --transaction-cost-bps 1.0 \
  --eval-episodes 10
```

## Latency And Loss Stress Tests

Run the default robustness grid:

```bash
cd Our-Code
../.venv/bin/python3 evaluate_latency_scenarios.py \
  --experiment-root experiment_results/sz10_full_experiment \
  --processed-root data/sz_l2_processed \
  --eval-episodes 10
```

Default scenarios:

```text
paper_base
obs_latency_5
obs_latency_10
order_latency_50ms
loss_5pct
obs10_loss5pct
infer_50ms
infer50_loss5pct
```

Cost-sensitive latency run:

```bash
../.venv/bin/python3 evaluate_latency_scenarios.py \
  --experiment-root experiment_results/sz10_full_experiment \
  --processed-root data/sz_l2_processed \
  --transaction-cost-bps 1.0 \
  --eval-episodes 10 \
  --output-dir experiment_results/sz10_full_experiment/latency_scenarios_cost_1bp
```

Outputs:

```text
latency_scenarios/
  reports/latency_summary.csv
  reports/latency_aggregate.csv
  episodes/
  plots/
```

## Regime Sensitivity

Run regime analysis from saved episode results:

```bash
cd Our-Code
../.venv/bin/python3 analyze_regime_sensitivity.py \
  --experiment-root experiment_results/sz10_full_experiment \
  --processed-root data/sz_l2_processed
```

The script reconstructs episode-level market regimes from the processed test split:

- realized volatility
- average market spread
- trade intensity per second

Outputs:

```text
regime_sensitivity/
  reports/episode_regimes.csv
  reports/regime_episode_results.csv
  reports/regime_sensitivity_summary.csv
  plots/volatility_regime_pnl.png
  plots/spread_regime_pnl.png
  plots/trade_intensity_regime_pnl.png
```

## Suggested Final Presentation Tables

Use these CSVs directly in the final deck:

- Attn-LOB classification: `pretrain_summary.csv`
- Overall trading comparison: `comparison_test_aggregate.csv`
- Per-stock trading comparison: `comparison_test_summary.csv`
- Latency robustness: `latency_scenarios/reports/latency_aggregate.csv`
- Regime sensitivity: `regime_sensitivity/reports/regime_sensitivity_summary.csv`
- Model inventory: `model_registry.csv`
