# Attn-LOB Market-Making System

This is the clean entry point for the project implementation. The system supports:

- Shenzhen L2 data processing
- Attn-LOB pretraining with one stock or multiple stocks
- PPO / Dueling-DQN market-making agents
- fixed/random/Avellaneda-Stoikov baselines
- latency, order-loss, and inference-delay stress tests
- regime-sensitivity analysis

The recommended interface is:

```bash
cd Our-Code
../.venv/bin/python3 run_system.py <command> [options]
```

The Python API is in:

```text
system_api.py
```

## Compared Methods

The main reports and final plots compare three groups:

- Joint-encoder RL: `joint_ppo`, `joint_dueling_dqn`. These use one Attn-LOB encoder pre-trained on all 10 selected stocks, then train one RL policy per stock.
- Single-encoder RL: `single_ppo`, `single_dueling_dqn`. These use one Attn-LOB encoder pre-trained separately for each stock, then train one RL policy per stock.
- Non-learning baselines: fixed level 1/2/3, random top-5-level quoting, and Avellaneda-Stoikov.

The baseline names in CSV files are:

```text
baseline_fixed_1
baseline_fixed_2
baseline_fixed_3
baseline_random_5level
baseline_as
```

## Data

Processed data should be stored as:

```text
data/sz_l2_processed/
  selected_symbols.csv
  manifest.json
  <ticker>/events.pkl.gz
  <ticker>/metadata.json
```

List selected symbols:

```bash
../.venv/bin/python3 run_system.py list-tickers
```

## 1. Pretrain Attn-LOB

Train one joint encoder using all selected stocks:

```bash
../.venv/bin/python3 run_system.py pretrain \
  --mode joint \
  --tickers all \
  --epochs 50
```

Train one encoder per stock:

```bash
../.venv/bin/python3 run_system.py pretrain \
  --mode single \
  --tickers all \
  --epochs 50
```

Train only one stock:

```bash
../.venv/bin/python3 run_system.py pretrain \
  --mode single \
  --tickers 000004.SZ \
  --epochs 50
```

Pretraining outputs:

```text
experiment_results/sz10_full_experiment/attn_lob_single/
experiment_results/sz10_full_experiment/attn_lob_joint/
```

Each model directory contains:

```text
best_model.pt
best_encoder.pt
metadata.json
results_summary_latest.json
```

## 2. Train RL Agents

Train PPO with the joint encoder for all stocks:

```bash
../.venv/bin/python3 run_system.py rl \
  --agent ppo \
  --encoder-mode joint \
  --tickers all \
  --timesteps 20000
```

Train Dueling-DQN with single-stock encoders:

```bash
../.venv/bin/python3 run_system.py rl \
  --agent dueling_dqn \
  --encoder-mode single \
  --tickers all \
  --timesteps 20000
```

Train only one stock:

```bash
../.venv/bin/python3 run_system.py rl \
  --agent ppo \
  --encoder-mode joint \
  --tickers 000004.SZ
```

RL outputs:

```text
experiment_results/sz10_full_experiment/rl_single/
experiment_results/sz10_full_experiment/rl_joint/
```

Each RL run contains:

```text
best/best_model.zip
final_model.zip
val_episode_results.csv
test_episode_results.csv
metadata.json
```

## 3. Backtest Baselines

Run paper-style baselines:

```bash
../.venv/bin/python3 run_system.py baseline \
  --tickers all \
  --baselines fixed_1,fixed_2,fixed_3,random,as \
  --eval-episodes 10
```

Baseline outputs:

```text
experiment_results/sz10_full_experiment/baselines/
```

## 4. Summarize Results

Rebuild all summary tables and plots from saved model runs:

```bash
../.venv/bin/python3 run_system.py summarize
```

Main reports:

```text
experiment_results/sz10_full_experiment/reports/
  pretrain_summary.csv
  rl_summary.csv
  baseline_summary.csv
  comparison_test_summary.csv
  comparison_test_aggregate.csv
  model_registry.csv
```

Main presentation plots:

```text
experiment_results/sz10_full_experiment/final_plots/
```

## 5. Latency / Order-Loss Backtests

Run the default robustness grid:

```bash
../.venv/bin/python3 run_system.py latency \
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

Run selected methods and selected stocks:

```bash
../.venv/bin/python3 run_system.py latency \
  --tickers 000004.SZ,002193.SZ \
  --methods joint_ppo,baseline_fixed_2,baseline_as \
  --scenarios paper_base,obs_latency_10,loss_5pct
```

Latency outputs:

```text
experiment_results/sz10_full_experiment/latency_scenarios/
  reports/latency_summary.csv
  reports/latency_aggregate.csv
  episodes/
  plots/
```

## 6. Regime Sensitivity

Analyze saved episode results by market regime:

```bash
../.venv/bin/python3 run_system.py regime
```

Regimes:

- realized volatility
- average market spread
- trade intensity per second

Outputs:

```text
experiment_results/sz10_full_experiment/regime_sensitivity/
  reports/episode_regimes.csv
  reports/regime_episode_results.csv
  reports/regime_sensitivity_summary.csv
  plots/
```

## 7. Presentation-Quality Plots

Rebuild cleaner final figures from saved CSV results:

```bash
../.venv/bin/python3 plot_final_results.py
```

Outputs:

```text
experiment_results/sz10_full_experiment/final_plots/
  01_overall_test_pnl.png
  02_risk_return_inventory.png
  03_pretrain_test_f1.png
  04_per_stock_pnl_heatmap.png
  05_latency_pnl_delta.png
  06_latency_pnl_heatmap.png
  07_regime_sensitivity_pnl.png
  08_cumulative_test_pnl.png
```

Each figure is also saved as `.pdf` for slides or reports.

## 8. Full Pipeline

Run or resume the complete SZ10 experiment:

```bash
../.venv/bin/python3 run_system.py full \
  --stage all \
  --epochs 50 \
  --rl-timesteps 20000 \
  --eval-episodes 10
```

Run one stage only:

```bash
../.venv/bin/python3 run_system.py full --stage pretrain
../.venv/bin/python3 run_system.py full --stage rl
../.venv/bin/python3 run_system.py full --stage baselines
../.venv/bin/python3 run_system.py full --stage summarize
../.venv/bin/python3 run_system.py full --stage plots
```

## Python API Example

```python
from system_api import TradingSystemConfig, pretrain, train_rl, evaluate_latency

config = TradingSystemConfig(
    processed_root="data/sz_l2_processed",
    output_root="experiment_results/sz10_full_experiment",
    device="cuda",
)

pretrain(config, mode="joint", tickers="all", epochs=50)
train_rl(config, agent="ppo", encoder_mode="joint", tickers="000004.SZ")
evaluate_latency(config, tickers="000004.SZ", methods="joint_ppo,baseline_as")
```

## Assumptions

- Train/validation/test split is date based.
- LOB and market-state normalization use the training split only.
- The paper assumes zero transaction cost and negligible market impact. We keep that as the default because the experiment studies small-order high-frequency market making.
- Latency tests are execution robustness checks; they do not retrain the RL agents unless explicitly run through `train_rl_agent.py` with latency parameters.
