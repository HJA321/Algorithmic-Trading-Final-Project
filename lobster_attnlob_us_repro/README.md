## LOBSTER Data + Attn-LOB

This folder contains the code extracted from the local reproduction project for:

- downloading public 10-level LOBSTER sample data
- preparing LOBSTER message/orderbook files into processed event-level datasets
- the PyTorch Attn-LOB encoder used for LOB representation learning

### Files

- `lobster_mm/config.py`
  - experiment configuration, sample URLs, and directory layout
- `lobster_mm/utils.py`
  - small helpers for JSON IO, directory creation, and reproducibility
- `lobster_mm/data.py`
  - public LOBSTER sample download
  - raw message/orderbook alignment
  - feature preparation and dataset utilities
- `lobster_mm/modeling.py`
  - Attn-LOB encoder
  - mid-price classification head
  - multi-input feature extractor for downstream RL

### Notes

- This upload intentionally excludes raw data, processed data, checkpoints, and evaluation outputs.
- The code targets the public 10-level LOBSTER samples for `AAPL`, `MSFT`, and `AMZN`.
- The Attn-LOB implementation has been aligned more closely to the paper author's Keras structure, including same-padding behavior in the convolution stack.

### Minimal dependencies

- `numpy`
- `pandas`
- `torch`
- `gymnasium`
- `stable-baselines3`

### Quick start

From this folder, you can download the public LOBSTER samples with:

```bash
python run_lobster_data.py download-data
```

To prepare the downloaded raw files into processed event-level datasets:

```bash
python run_lobster_data.py prepare-data
```

You can also restrict to selected tickers, for example:

```bash
python run_lobster_data.py download-data --tickers AAPL MSFT
```
