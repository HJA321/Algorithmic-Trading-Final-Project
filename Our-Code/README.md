# Our-Code

This folder contains a PyTorch reimplementation of the paper's Attn-LOB pretraining stage adapted to LOBSTER sample data.

For the clean API/CLI usage guide, see [`SYSTEM_README.md`](SYSTEM_README.md).
For the project-level checklist and final-presentation pointers, see [`PROJECT_RUNBOOK.md`](PROJECT_RUNBOOK.md).

## Scope

The current implementation covers:

- reading `ask.csv`, `bid.csv`, and `price.csv`
- reading raw LOBSTER `message` and `orderbook` CSV files
- aligning message and orderbook rows into one event stream
- reproducing the original mid-price direction task on top of LOBSTER data
- generating sliding windows directly from the 10-level order book
- replacing the TensorFlow/Keras Attn-LOB model with a PyTorch implementation
- training and evaluating the classifier, while saving the encoder checkpoint for later RL use
- training PPO/DQN market-making agents on processed LOBSTER event streams
- reusing the pre-trained Attn-LOB encoder inside the RL policy network
- evaluating RL episodes with PnL, ND-PnL, PnL-MAP, inventory, spread, volume, and reward-component summaries
- evaluating execution frictions including transaction costs, observation latency, order-arrival latency, order loss, and inference delay
- analyzing regime sensitivity by volatility, spread, and trade intensity

## Compared Methods

The saved SZ10 reports and final plots compare:

- `single_ppo`: PPO trained per stock using that stock's own Attn-LOB encoder
- `joint_ppo`: PPO trained per stock using the joint encoder pre-trained on all 10 selected stocks
- `single_dueling_dqn`: Dueling-DQN trained per stock using that stock's own Attn-LOB encoder
- `joint_dueling_dqn`: Dueling-DQN trained per stock using the joint encoder pre-trained on all 10 selected stocks
- `baseline_fixed_1`, `baseline_fixed_2`, `baseline_fixed_3`: fixed-depth market-making quotes at L1/L2/L3
- `baseline_random_5level`: random bid/ask quote levels from the top 5 levels
- `baseline_as`: Avellaneda-Stoikov inventory-aware quoting baseline

Baselines do not use Attn-LOB pretraining. In plots, methods with `joint_` use the 10-stock pre-trained encoder, while methods with `single_` use one encoder per stock.

## Expected data layout

The scripts expect the downloaded LOBSTER sample layout:

```text
lobster_attnlob_us_repro/data/lobster_raw/
  AAPL/
    2012-06-21/
      AAPL_2012-06-21_..._message_10.csv
      AAPL_2012-06-21_..._orderbook_10.csv
```

## Main files

- `attn_lob_pytorch/data.py`: LOBSTER loading, label generation, window normalization, and dataset creation
- `attn_lob_pytorch/model.py`: PyTorch Attn-LOB encoder and classifier
- `train_attn_lob.py`: training entry point
- `data_prep/reconstruct_sz_orderbook.py`: Shenzhen L2 order/order-trade replay and market snapshot validation
- `data_prep/prepare_sz_l2_dataset.py`: selects high-quality SZ symbols and writes processed `events.pkl.gz` files
- `data_prep/analyze_sz_l2_candidates.py`: data-only SZ candidate scoring and diagnostics
- `rl_trading/data.py`: processed market-data loading and normalization for RL
- `rl_trading/env.py`: Gymnasium market-making environments aligned with the paper's simulator, action spaces, and hybrid reward
- `rl_trading/baselines.py`: fixed-level, random-level, and Avellaneda-Stoikov quote policies
- `rl_trading/evaluation.py`: shared multi-episode evaluation helpers for agents and quote baselines
- `rl_trading/model.py`: Stable-Baselines3 feature extractor that loads the pre-trained Attn-LOB encoder
- `train_rl_agent.py`: PPO/DQN training and RL evaluation entry point
- `evaluate_baselines.py`: baseline evaluation entry point
- `evaluate_latency_scenarios.py`: latency, order-loss, and inference-delay stress-test evaluation for saved RL models and baselines
- `analyze_regime_sensitivity.py`: post-hoc regime sensitivity analysis from saved episode results
- `system_api.py`: Python API for the project pipeline
- `run_system.py`: unified CLI wrapper around the API
- `plot_final_results.py`: presentation-quality plots from saved reports

## Example

From the project root:

```bash
cd Our-Code
../.venv/bin/python train_attn_lob.py --tickers AAPL --trading-date 2012-06-21 --data-root ../lobster_attnlob_us_repro/data/lobster_raw --device cpu
```

Checkpoints are saved under:

```text
Our-Code/experiment_results/attn_lob/<ticker>_<date>/
```

The encoder checkpoint for later RL integration is saved as `best_encoder.pt`.

After preparing processed data, train a continuous PPO market-making agent with:

```bash
cd Our-Code
../.venv/bin/python train_rl_agent.py --agent ppo --ticker AAPL --processed-root ../lobster_attnlob_us_repro/data/lobster_processed --encoder-path experiment_results/attn_lob/AAPL_2012-06-21/best_encoder.pt
```

The RL script evaluates multiple validation/test episodes. By default, the final report loads the best checkpoint saved by the callback; use `--eval-model final` to report the final checkpoint instead.

Evaluate non-learning baselines with:

```bash
cd Our-Code
../.venv/bin/python evaluate_baselines.py --ticker AAPL --processed-root ../lobster_attnlob_us_repro/data/lobster_processed --baselines fixed_1,fixed_2,fixed_3,random,as
```

RL checkpoints and episode metrics are saved under:

```text
Our-Code/experiment_results/rl/<agent>/<ticker>/
```

Baseline summaries are saved under:

```text
Our-Code/experiment_results/baselines/<ticker>/<split>/
```

## Shenzhen L2 Data Flow

The uploaded `l2_zip` HDF5 data can be converted into the same processed layout used by the RL code:

```text
Our-Code/data/sz_l2_processed/
  manifest.json
  selected_symbols.csv
  <symbol>/
    events.pkl.gz
    metadata.json
```

Prepare the 10 selected SZ symbols with a date split of 3:1:1:

```bash
# from the project root
./.venv/bin/python3 Our-Code/data_prep/prepare_sz_l2_dataset.py \
  --candidate-limit 20 \
  --validation-stride 10 \
  --num-symbols 10 \
  --output-root Our-Code/data/sz_l2_processed
```

The manifest records:

- train: `20250121,20250122,20250123`
- val: `20250124`
- test: `20250127`

Train Attn-LOB from the processed SZ data:

```bash
cd Our-Code
../.venv/bin/python3 train_attn_lob.py \
  --dataset-format processed \
  --tickers 000004.SZ,002193.SZ,002205.SZ,002231.SZ,002858.SZ,002780.SZ,003003.SZ,301228.SZ,002199.SZ,300106.SZ \
  --trading-date 20250121-20250127 \
  --data-root data/sz_l2_processed \
  --device cpu
```

Then train/evaluate RL for one selected ticker using the encoder produced above:

```bash
cd Our-Code
../.venv/bin/python3 train_rl_agent.py \
  --agent ppo \
  --ticker 000004.SZ \
  --processed-root data/sz_l2_processed \
  --encoder-path experiment_results/attn_lob/000004.SZ-002193.SZ-002205.SZ-002231.SZ-002858.SZ-002780.SZ-003003.SZ-301228.SZ-002199.SZ-300106.SZ_20250121-20250127/best_encoder.pt \
  --encoder-metadata-path experiment_results/attn_lob/000004.SZ-002193.SZ-002205.SZ-002231.SZ-002858.SZ-002780.SZ-003003.SZ-301228.SZ-002199.SZ-300106.SZ_20250121-20250127/metadata.json
```

Run baselines on the same processed data:

```bash
cd Our-Code
../.venv/bin/python3 evaluate_baselines.py \
  --ticker 000004.SZ \
  --processed-root data/sz_l2_processed \
  --baselines fixed_1,fixed_2,fixed_3,random,as
```

## Latency And Order-Loss Evaluation

The main reported experiments set transaction costs and market impact to zero because the strategy is modeled as small-order high-frequency market making, where individual quotes are assumed to be small relative to displayed depth. This keeps the comparison focused on policy behavior under a shared execution model; cost parameters remain available for sensitivity tests.

The paper-style `--latency` parameter is an event delay on the information used by the policy: the agent observes and quotes from `t - latency`. For stricter backtests, the RL and baseline environments also support:

- `--order-latency-events`: raw event-count delay before the submitted quote can reach the simulated exchange
- `--order-latency-ms`: wall-clock delay in milliseconds, converted with the processed `time` column
- `--inference-latency-ms`: simulated model calculation delay added to order arrival
- `--order-loss-probability`: independent probability that each submitted bid/ask quote is not accepted
- `--transaction-cost-bps` and `--fixed-fee-per-trade`: optional cost controls for robustness checks

Evaluate saved SZ10 models and baselines under a default stress-test grid:

```bash
cd Our-Code
../.venv/bin/python3 evaluate_latency_scenarios.py \
  --experiment-root experiment_results/sz10_full_experiment \
  --processed-root data/sz_l2_processed \
  --eval-episodes 10
```

Outputs are saved under:

```text
Our-Code/experiment_results/sz10_full_experiment/latency_scenarios/
  reports/latency_summary.csv
  reports/latency_aggregate.csv
  plots/
  episodes/
```
