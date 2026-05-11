from __future__ import annotations

import argparse
import os
import json
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from attn_lob_pytorch.data import (
    DEFAULT_TICKERS,
    DEFAULT_TRADING_DATE,
    create_datasets,
    create_datasets_from_processed_root,
    normalize_lob_batch,
)
from attn_lob_pytorch.model import AttnLOBClassifier


CLASS_NAMES = ("down", "stationary", "up")


@dataclass
class TrainConfig:
    dataset_format: str = "lobster_raw"
    tickers: str = ",".join(DEFAULT_TICKERS)
    trading_date: str = DEFAULT_TRADING_DATE
    data_root: str = "../lobster_attnlob_us_repro/data/lobster_raw"
    output_dir: str = "experiment_results/attn_lob"
    levels: int = 10
    horizon: int = 10
    label_threshold: float = 1e-5
    time_window: int = 50
    split_train: float = 0.6
    split_val: float = 0.2
    price_scale: float = 10000.0
    tick_size: float = 0.01
    batch_size: int = 64
    epochs: int = 50
    optimizer: str = "adamw"
    loss: str = "weighted_cross_entropy"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    embedding_dim: int = 64
    conv_channels: int = 32
    inception_channels: int = 64
    num_heads: int = 10
    head_dim: int = 16
    use_aux_features: bool = False
    num_workers: int = -1
    prefetch_factor: int = 4
    seed: int = 42
    device: str = "cpu"
    precision: str = "auto"
    channels_last: bool = True


def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Pre-train the Attn-LOB encoder in PyTorch.")
    parser.add_argument(
        "--dataset-format",
        choices=("lobster_raw", "processed"),
        default=defaults.dataset_format,
        help="Use lobster_raw for raw LOBSTER CSVs or processed for prepared events.pkl.gz files.",
    )
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
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default=defaults.optimizer)
    parser.add_argument("--loss", choices=("cross_entropy", "weighted_cross_entropy"), default=defaults.loss)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--embedding-dim", type=int, default=defaults.embedding_dim)
    parser.add_argument("--conv-channels", type=int, default=defaults.conv_channels)
    parser.add_argument("--inception-channels", type=int, default=defaults.inception_channels)
    parser.add_argument("--num-heads", type=int, default=defaults.num_heads)
    parser.add_argument("--head-dim", type=int, default=defaults.head_dim)
    parser.add_argument(
        "--use-aux-features",
        action=argparse.BooleanOptionalAction,
        default=defaults.use_aux_features,
        help="Optional experimental branch. Default is pure LOB, matching the paper pretraining task.",
    )
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


def compute_balanced_class_weights(split_summary: dict[str, object], device: torch.device) -> torch.Tensor:
    counts = torch.tensor(split_summary["counts"], dtype=torch.float32, device=device)
    weights = torch.zeros_like(counts)
    present = counts > 0
    if present.any():
        weights[present] = counts[present].sum() / (present.sum().float() * counts[present])
    return weights


def build_criterion(
    config: TrainConfig,
    train_split_summary: dict[str, object],
    device: torch.device,
) -> tuple[nn.Module, list[float] | None]:
    if config.loss == "cross_entropy":
        return nn.CrossEntropyLoss(), None
    if config.loss == "weighted_cross_entropy":
        class_weights = compute_balanced_class_weights(train_split_summary, device)
        return nn.CrossEntropyLoss(weight=class_weights), class_weights.detach().cpu().tolist()
    raise ValueError(f"Unsupported loss: {config.loss}")


def build_optimizer(config: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def count_parameters(module: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}


def summarize_model(
    model: AttnLOBClassifier,
    config: TrainConfig,
    feature_columns: list[str],
    aux_feature_columns: list[str],
) -> dict[str, object]:
    input_channels = 1
    return {
        "model_class": model.__class__.__name__,
        "encoder_class": model.encoder.__class__.__name__,
        "architecture": str(model),
        "parameters": count_parameters(model),
        "hyperparameters": {
            "embedding_dim": config.embedding_dim,
            "conv_channels": config.conv_channels,
            "inception_channels": config.inception_channels,
            "num_heads": config.num_heads,
            "head_dim": config.head_dim,
            "attention_dim": config.num_heads * config.head_dim,
        },
        "input": {
            "lob_state_shape": [input_channels, config.time_window, len(feature_columns)],
            "time_window": config.time_window,
            "levels": config.levels,
            "num_lob_features": len(feature_columns),
            "feature_columns": feature_columns,
            "uses_aux_features": config.use_aux_features,
            "num_aux_features": len(aux_feature_columns),
        },
    }


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

    for step, (features, aux_features, labels) in enumerate(progress, start=1):
        features = features.to(device, non_blocking=device.type == "cuda")
        aux_features = aux_features.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        features = normalize_lob_batch(features)
        if channels_last and device.type == "cuda":
            features = features.contiguous(memory_format=torch.channels_last)

        with torch.set_grad_enabled(is_train):
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with autocast_context:
                logits = model(features, aux_features=aux_features)
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
    model_summary: dict[str, object],
    metrics: dict[str, float],
    class_weights: list[float] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "best_model.pt")
    torch.save(model.encoder.state_dict(), output_dir / "best_encoder.pt")

    metadata = {
        "model": model_summary,
        "training": {
            "dataset_format": config.dataset_format,
            "tickers": config.tickers,
            "trading_date": config.trading_date,
            "data_root": config.data_root,
            "horizon": config.horizon,
            "label_threshold": config.label_threshold,
            "optimizer": config.optimizer,
            "loss": config.loss,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "class_weights": class_weights,
            "best_validation": metrics,
        },
        "checkpoints": {
            "best_model": str(output_dir / "best_model.pt"),
            "best_encoder": str(output_dir / "best_encoder.pt"),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    device = torch.device(config.device)
    configure_torch_runtime(device)

    tickers = parse_day_list(config.tickers)
    if config.dataset_format == "processed":
        datasets = create_datasets_from_processed_root(
            tickers=tickers,
            processed_root=config.data_root,
            levels=config.levels,
            time_window=config.time_window,
            include_aux_features=config.use_aux_features,
        )
    else:
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
            include_aux_features=config.use_aux_features,
        )

    autocast_dtype = resolve_autocast_dtype(config, device)
    num_workers = resolve_num_workers(config)
    train_split_summary = summarize_dataset_split(datasets.train, datasets.num_classes)
    val_split_summary = summarize_dataset_split(datasets.val, datasets.num_classes)
    test_split_summary = summarize_dataset_split(datasets.test, datasets.num_classes)

    model = AttnLOBClassifier(
        time_window=config.time_window,
        embedding_dim=config.embedding_dim,
        conv_channels=config.conv_channels,
        inception_channels=config.inception_channels,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        num_classes=datasets.num_classes,
        # Pure LOB pretraining, matching the paper/original Keras code.
        # The auxiliary feature branch remains available behind --use-aux-features.
        aux_feature_dim=len(datasets.aux_feature_columns) if config.use_aux_features else 0,
    ).to(device)
    if config.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    criterion, class_weights = build_criterion(config, train_split_summary, device)
    optimizer = build_optimizer(config, model)

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
    output_dir.mkdir(parents=True, exist_ok=True)

    run_started_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + f"_pid{os.getpid()}"
    summary_path = output_dir / f"results_summary_{run_id}.json"
    latest_summary_path = output_dir / "results_summary_latest.json"
    model_summary = summarize_model(
        model=model,
        config=config,
        feature_columns=datasets.feature_columns,
        aux_feature_columns=datasets.aux_feature_columns,
    )
    results_summary: dict[str, object] = {
        "run_started_at": run_started_at,
        "run_id": run_id,
        "model": model_summary,
        "training": {
            "dataset_format": config.dataset_format,
            "tickers": tickers,
            "trading_date": config.trading_date,
            "data_root": config.data_root,
            "horizon": config.horizon,
            "label_threshold": config.label_threshold,
            "optimizer": config.optimizer,
            "loss": config.loss,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "class_weights": class_weights,
        },
        "best_validation": {},
        "test_metrics": {},
        "checkpoints": {},
    }
    write_json(summary_path, results_summary)
    write_json(latest_summary_path, results_summary)

    print(
        f"device={device} precision={autocast_dtype or torch.float32} "
        f"num_workers={num_workers} channels_last={config.channels_last} "
        f"batch_size={config.batch_size} optimizer={config.optimizer} loss={config.loss} "
        f"learning_rate={config.learning_rate} weight_decay={config.weight_decay} "
        f"dataset_format={config.dataset_format} data_root={config.data_root} "
        f"horizon={config.horizon} use_aux_features={config.use_aux_features} "
        f"embedding_dim={config.embedding_dim} conv_channels={config.conv_channels} "
        f"inception_channels={config.inception_channels} num_heads={config.num_heads} head_dim={config.head_dim}"
    )
    print(f"output_dir={output_dir}")
    if class_weights is not None:
        print(f"class_weights={class_weights}")
    print(format_split_summary("train", train_split_summary))
    print(format_split_summary("val", val_split_summary))
    print(format_split_summary("test", test_split_summary))
    print(f"dataloader: train_batches={len(train_loader)} val_batches={len(val_loader)} test_batches={len(test_loader)}")

    best_val_f1 = float("-inf")
    best_metrics: dict[str, float] = {}

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, autocast_dtype, config.channels_last)
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None, autocast_dtype=autocast_dtype, channels_last=config.channels_last)
        epoch_seconds = time.perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        improved = val_metrics["f1"] > best_val_f1

        best_val_f1_for_log = max(best_val_f1, val_metrics["f1"])
        print(
            f"epoch={epoch}/{config.epochs} lr={current_lr:.6g} time={epoch_seconds:.1f}s "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"best_val_f1={best_val_f1_for_log:.4f} improved={improved}"
        )

        if improved:
            best_val_f1 = val_metrics["f1"]
            best_metrics = {"epoch": epoch, **val_metrics}
            save_checkpoint(
                output_dir,
                model,
                config,
                model_summary,
                best_metrics,
                class_weights,
            )
            results_summary["best_validation"] = best_metrics
            results_summary["checkpoints"] = {
                "best_model": str(output_dir / "best_model.pt"),
                "best_encoder": str(output_dir / "best_encoder.pt"),
                "metadata": str(output_dir / "metadata.json"),
            }
            write_json(summary_path, results_summary)
            write_json(latest_summary_path, results_summary)
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
    print(f"test_summary {format_metrics('test', test_metrics)} checkpoint={output_dir / 'best_model.pt'}")
    print(format_class_metrics("test", test_metrics))

    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["test_metrics"] = test_metrics
    metadata["results_summary_path"] = str(summary_path)
    metadata["latest_results_summary_path"] = str(latest_summary_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    results_summary["best_validation"] = best_metrics
    results_summary["test_metrics"] = test_metrics
    results_summary["run_finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    results_summary["checkpoints"] = {
        "best_model": str(output_dir / "best_model.pt"),
        "best_encoder": str(output_dir / "best_encoder.pt"),
        "metadata": str(metadata_path),
    }
    write_json(summary_path, results_summary)
    write_json(latest_summary_path, results_summary)
    print(f"metadata_written path={metadata_path}")
    print(f"results_summary_written path={summary_path}")
    print(f"latest_results_summary_written path={latest_summary_path}")


if __name__ == "__main__":
    main()
