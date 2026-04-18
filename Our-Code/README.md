# Our-Code

This folder contains a PyTorch reimplementation of the paper's Attn-LOB pretraining stage adapted to LOBSTER sample data.

## Scope

The current implementation covers:

- reading `ask.csv`, `bid.csv`, and `price.csv`
- reading raw LOBSTER `message` and `orderbook` CSV files
- aligning message and orderbook rows into one event stream
- reproducing the original mid-price direction task on top of LOBSTER data
- generating sliding windows directly from the 10-level order book
- replacing the TensorFlow/Keras Attn-LOB model with a PyTorch implementation
- training and evaluating the classifier, while saving the encoder checkpoint for later RL use

It does not yet cover the reinforcement learning stage.

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

## Example

From the project root:

```bash
cd Our-Code
../.venv/bin/python train_attn_lob.py --tickers AAPL --trading-date 2012-06-21 --data-root ../lobster_attnlob_us_repro/data/lobster_raw --device cpu
```

Checkpoints are saved under:

```text
Our-Code/checkpoints/attn_lob/<ticker>_<date>/
```

The encoder checkpoint for later RL integration is saved as `best_encoder.pt`.
