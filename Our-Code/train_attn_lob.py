from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from attn_lob_pytorch.data import DEFAULT_TICKERS, DEFAULT_TRADING_DATE, create_datasets
from attn_lob_pytorch.model import AttnLOBClassifier


@dataclass
class TrainConfig:
    tickers: str = ",".join(DEFAULT_TICKERS)
    trading_date: str = DEFAULT_TRADING_DATE
    data_root: str = "../lobster_attnlob_us_repro/data/lobster_raw"
    output_dir: str = "checkpoints/attn_lob"
    levels: int = 10
    horizon: int = 10
    time_window: int = 50
    split_train: float = 0.6
    split_val: float = 0.2
    price_scale: float = 10000.0
    tick_size: float = 0.01
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0
    seed: int = 42
    device: str = "cpu"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Pre-train the Attn-LOB encoder in PyTorch.")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--trading-date", default=DEFAULT_TRADING_DATE)
    parser.add_argument("--data-root", default="../lobster_attnlob_us_repro/data/lobster_raw")
    parser.add_argument("--output-dir", default="checkpoints/attn_lob")
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--time-window", type=int, default=50)
    parser.add_argument("--split-train", type=float, default=0.6)
    parser.add_argument("--split-val", type=float, default=0.2)
    parser.add_argument("--price-scale", type=float, default=10000.0)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def parse_day_list(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def experiment_name(tickers: list[str], trading_date: str) -> str:
    return f"{'-'.join(tickers)}_{trading_date}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    predictions = logits.argmax(dim=1)
    num_classes = logits.shape[1]
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for target, prediction in zip(labels.view(-1), predictions.view(-1), strict=False):
        confusion[target.long(), prediction.long()] += 1

    support = confusion.sum(dim=1).float()
    true_positive = confusion.diag().float()
    predicted_positive = confusion.sum(dim=0).float()

    precision = true_positive / predicted_positive.clamp_min(1.0)
    recall = true_positive / support.clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

    accuracy = true_positive.sum() / confusion.sum().clamp_min(1)
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    return {
        "accuracy": accuracy.item(),
        "precision": macro_precision.item(),
        "recall": macro_recall.item(),
        "f1": macro_f1.item(),
    }


def run_epoch(
    model: AttnLOBClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    progress = tqdm(loader, leave=False)
    for features, labels in progress:
        features = features.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(features)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    metrics = compute_metrics(torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0))
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


def save_checkpoint(
    output_dir: Path,
    model: AttnLOBClassifier,
    config: TrainConfig,
    dataset_metadata: dict,
    feature_columns: list[str],
    metrics: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "best_model.pt")
    torch.save(model.encoder.state_dict(), output_dir / "best_encoder.pt")

    metadata = {
        "config": asdict(config),
        "feature_columns": feature_columns,
        "dataset_metadata": dataset_metadata,
        "metrics": metrics,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    tickers = parse_day_list(config.tickers)
    datasets = create_datasets(
        tickers=tickers,
        trading_date=config.trading_date,
        data_root=config.data_root,
        levels=config.levels,
        horizon=config.horizon,
        time_window=config.time_window,
        price_scale=config.price_scale,
        tick_size=config.tick_size,
        split_train=config.split_train,
        split_val=config.split_val,
    )

    device = torch.device(config.device)
    model = AttnLOBClassifier(time_window=config.time_window, num_classes=datasets.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_loader = DataLoader(
        datasets.train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        datasets.val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        datasets.test,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    best_val_f1 = float("-inf")
    best_metrics: dict[str, float] = {}
    output_dir = Path(config.output_dir) / experiment_name(tickers, config.trading_date)

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_metrics = {"epoch": epoch, **val_metrics}
            save_checkpoint(output_dir, model, config, datasets.metadata, datasets.feature_columns, best_metrics)

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    test_metrics = run_epoch(model, test_loader, criterion, device, optimizer=None)
    print(
        "test_metrics "
        f"loss={test_metrics['loss']:.4f} "
        f"accuracy={test_metrics['accuracy']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} "
        f"f1={test_metrics['f1']:.4f}"
    )

    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["test_metrics"] = test_metrics
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
