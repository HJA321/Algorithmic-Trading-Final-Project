from __future__ import annotations

import argparse
import os
import json
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from attn_lob_pytorch.data import DEFAULT_TICKERS, DEFAULT_TRADING_DATE, create_datasets, normalize_lob_batch
from attn_lob_pytorch.model import AttnLOBClassifier


CLASS_NAMES = ("down", "stationary", "up")


@dataclass
class TrainConfig:
    tickers: str = ",".join(DEFAULT_TICKERS)
    trading_date: str = DEFAULT_TRADING_DATE
    data_root: str = "../lobster_attnlob_us_repro/data/lobster_raw"
    output_dir: str = "checkpoints/attn_lob"
    levels: int = 10
    horizon: int = 20
    label_threshold: float = 1e-5
    time_window: int = 50
    split_train: float = 0.6
    split_val: float = 0.2
    price_scale: float = 10000.0
    tick_size: float = 0.01
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = -1
    prefetch_factor: int = 4
    seed: int = 42
    device: str = "cpu"
    precision: str = "auto"
    channels_last: bool = True


def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Pre-train the Attn-LOB encoder in PyTorch.")
    parser.add_argument("--tickers", default=defaults.tickers)
    parser.add_argument("--trading-date", default=defaults.trading_date)
    parser.add_argument("--data-root", default=defaults.data_root)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--levels", type=int, default=defaults.levels)
    parser.add_argument("--horizon", type=int, default=defaults.horizon)
    parser.add_argument("--label-threshold", type=float, default=defaults.label_threshold)
    parser.add_argument("--time-window", type=int, default=defaults.time_window)
    parser.add_argument("--split-train", type=float, default=defaults.split_train)
    parser.add_argument("--split-val", type=float, default=defaults.split_val)
    parser.add_argument("--price-scale", type=float, default=defaults.price_scale)
    parser.add_argument("--tick-size", type=float, default=defaults.tick_size)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--prefetch-factor", type=int, default=defaults.prefetch_factor)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default=defaults.precision)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=defaults.channels_last)
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

    metrics = {
        "accuracy": accuracy.item(),
        "precision": macro_precision.item(),
        "recall": macro_recall.item(),
        "f1": macro_f1.item(),
    }
    for class_id, class_name in enumerate(CLASS_NAMES[:num_classes]):
        metrics[f"precision_{class_name}"] = precision[class_id].item()
        metrics[f"recall_{class_name}"] = recall[class_id].item()
        metrics[f"f1_{class_name}"] = f1[class_id].item()
    return metrics


def summarize_dataset_split(dataset, num_classes: int) -> dict[str, object]:
    counts = np.zeros(num_classes, dtype=np.int64)
    for ticker, row_index in dataset.samples:
        label = int(dataset.label_arrays[ticker][row_index])
        if 0 <= label < num_classes:
            counts[label] += 1
    total = int(counts.sum())
    percentages = counts / max(total, 1)
    return {
        "samples": total,
        "counts": counts.tolist(),
        "percentages": percentages.tolist(),
    }


def format_split_summary(name: str, summary: dict[str, object]) -> str:
    counts = summary["counts"]
    percentages = summary["percentages"]
    class_parts = [
        f"class_{class_id}={count} ({percentage:.1%})"
        for class_id, (count, percentage) in enumerate(zip(counts, percentages, strict=False))
    ]
    return f"{name}: samples={summary['samples']} " + " ".join(class_parts)


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return (
        f"{prefix}_loss={metrics['loss']:.4f} "
        f"{prefix}_accuracy={metrics['accuracy']:.4f} "
        f"{prefix}_precision={metrics['precision']:.4f} "
        f"{prefix}_recall={metrics['recall']:.4f} "
        f"{prefix}_f1={metrics['f1']:.4f}"
    )


def format_class_metrics(prefix: str, metrics: dict[str, float]) -> str:
    class_parts = []
    for class_name in CLASS_NAMES:
        precision_key = f"precision_{class_name}"
        recall_key = f"recall_{class_name}"
        f1_key = f"f1_{class_name}"
        if precision_key not in metrics:
            continue
        class_parts.append(
            f"{class_name}(p={metrics[precision_key]:.4f},r={metrics[recall_key]:.4f},f1={metrics[f1_key]:.4f})"
        )
    return f"{prefix}_per_class " + " ".join(class_parts)


def configure_torch_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def resolve_num_workers(config: TrainConfig) -> int:
    if config.num_workers >= 0:
        return config.num_workers
    cpu_count = os.cpu_count() or 1
    return min(8, max(2, cpu_count // 2))


def resolve_autocast_dtype(config: TrainConfig, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if config.precision == "fp32":
        return None
    if config.precision == "bf16":
        return torch.bfloat16
    if config.precision == "fp16":
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def run_epoch(
    model: AttnLOBClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    autocast_dtype: torch.dtype | None,
    channels_last: bool,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    progress = tqdm(loader, leave=False)
    autocast_context = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )

    for step, (features, labels) in enumerate(progress, start=1):
        features = features.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        features = normalize_lob_batch(features)
        if channels_last and device.type == "cuda":
            features = features.contiguous(memory_format=torch.channels_last)

        with torch.set_grad_enabled(is_train):
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with autocast_context:
                logits = model(features)
                loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        if step == 1 or step % 50 == 0:
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

    device = torch.device(config.device)
    configure_torch_runtime(device)

    tickers = parse_day_list(config.tickers)
    datasets = create_datasets(
        tickers=tickers,
        trading_date=config.trading_date,
        data_root=config.data_root,
        levels=config.levels,
        horizon=config.horizon,
        label_threshold=config.label_threshold,
        time_window=config.time_window,
        price_scale=config.price_scale,
        tick_size=config.tick_size,
        split_train=config.split_train,
        split_val=config.split_val,
    )

    autocast_dtype = resolve_autocast_dtype(config, device)
    num_workers = resolve_num_workers(config)

    model = AttnLOBClassifier(time_window=config.time_window, num_classes=datasets.num_classes).to(device)
    if config.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    train_loader = DataLoader(
        datasets.train,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        datasets.val,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        datasets.test,
        shuffle=False,
        **loader_kwargs,
    )

    output_dir = Path(config.output_dir) / experiment_name(tickers, config.trading_date)

    print(
        f"device={device} precision={autocast_dtype or torch.float32} "
        f"num_workers={num_workers} channels_last={config.channels_last} "
        f"batch_size={config.batch_size} learning_rate={config.learning_rate} horizon={config.horizon}"
    )
    print(f"output_dir={output_dir}")
    print(
        format_split_summary("train", summarize_dataset_split(datasets.train, datasets.num_classes))
    )
    print(
        format_split_summary("val", summarize_dataset_split(datasets.val, datasets.num_classes))
    )
    print(
        format_split_summary("test", summarize_dataset_split(datasets.test, datasets.num_classes))
    )
    print(
        f"dataloader: train_batches={len(train_loader)} val_batches={len(val_loader)} test_batches={len(test_loader)}"
    )

    best_val_f1 = float("-inf")
    best_metrics: dict[str, float] = {}

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, autocast_dtype, config.channels_last)
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None, autocast_dtype=autocast_dtype, channels_last=config.channels_last)
        epoch_seconds = time.perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        improved = val_metrics["f1"] > best_val_f1

        print(
            f"epoch={epoch}/{config.epochs} lr={current_lr:.6g} time={epoch_seconds:.1f}s "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"best_val_f1={max(best_val_f1, val_metrics['f1']):.4f} improved={improved}"
        )

        if improved:
            best_val_f1 = val_metrics["f1"]
            best_metrics = {"epoch": epoch, **val_metrics}
            save_checkpoint(output_dir, model, config, datasets.metadata, datasets.feature_columns, best_metrics)
            print(
                f"saved_best_checkpoint epoch={epoch} val_f1={val_metrics['f1']:.4f} "
                f"model={output_dir / 'best_model.pt'} encoder={output_dir / 'best_encoder.pt'}"
            )

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    test_metrics = run_epoch(model, test_loader, criterion, device, optimizer=None, autocast_dtype=autocast_dtype, channels_last=config.channels_last)
    if best_metrics:
        print(
            f"best_validation epoch={best_metrics['epoch']} "
            f"{format_metrics('best_val', best_metrics)}"
        )
        print(format_class_metrics("best_val", best_metrics))
    print(
        f"test_summary {format_metrics('test', test_metrics)} checkpoint={output_dir / 'best_model.pt'}"
    )
    print(format_class_metrics("test", test_metrics))

    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["test_metrics"] = test_metrics
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"metadata_written path={metadata_path}")


if __name__ == "__main__":
    main()
