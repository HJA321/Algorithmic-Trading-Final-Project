from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "single_ppo": "Single PPO",
    "joint_ppo": "Joint PPO",
    "single_dueling_dqn": "Single D-DQN",
    "joint_dueling_dqn": "Joint D-DQN",
    "baseline_fixed_1": "Fixed L1",
    "baseline_fixed_2": "Fixed L2",
    "baseline_fixed_3": "Fixed L3",
    "baseline_random_5level": "Random",
    "baseline_as": "AS",
}

METHOD_ORDER = [
    "baseline_fixed_2",
    "baseline_fixed_3",
    "joint_ppo",
    "baseline_as",
    "baseline_random_5level",
    "single_ppo",
    "joint_dueling_dqn",
    "single_dueling_dqn",
    "baseline_fixed_1",
]

SCENARIO_ORDER = [
    "paper_base",
    "obs_latency_5",
    "obs_latency_10",
    "order_latency_50ms",
    "loss_5pct",
    "obs10_loss5pct",
    "infer_50ms",
    "infer50_loss5pct",
]

SCENARIO_LABELS = {
    "paper_base": "Base",
    "obs_latency_5": "Obs 5",
    "obs_latency_10": "Obs 10",
    "order_latency_50ms": "Order 50ms",
    "loss_5pct": "Loss 5%",
    "obs10_loss5pct": "Obs10 + Loss",
    "infer_50ms": "Infer 50ms",
    "infer50_loss5pct": "Infer + Loss",
}

REGIME_LABELS = {
    "volatility_regime": "Volatility",
    "spread_regime": "Spread",
    "trade_intensity_regime": "Trade Intensity",
}

FAMILY_COLORS = {
    "ppo": "#2F7D6D",
    "dqn": "#4E79A7",
    "baseline": "#8A8F98",
    "fixed_best": "#C76E2C",
    "as": "#7B5EA7",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Create presentation-quality plots from saved experiment results.")
    parser.add_argument("--experiment-root", default="experiment_results/sz10_full_experiment")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-methods", default="baseline_fixed_2,baseline_fixed_3,joint_ppo,baseline_as,single_ppo")
    parser.add_argument("--experiment-label", default="10 Stocks")
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E6E8EB",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def display_method(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def method_color(method: str) -> str:
    if method == "baseline_fixed_2":
        return FAMILY_COLORS["fixed_best"]
    if method == "baseline_as":
        return FAMILY_COLORS["as"]
    if "ppo" in method:
        return FAMILY_COLORS["ppo"]
    if "dqn" in method:
        return FAMILY_COLORS["dqn"]
    return FAMILY_COLORS["baseline"]


def ordered_methods(methods: list[str]) -> list[str]:
    known = [method for method in METHOD_ORDER if method in methods]
    rest = sorted(method for method in methods if method not in known)
    return known + rest


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_overall_pnl(comparison: pd.DataFrame, output_dir: Path, experiment_label: str) -> None:
    rows = []
    for method, group in comparison.groupby("method"):
        pnl = group["test_pnl_mean"].astype(float)
        rows.append(
            {
                "method": method,
                "pnl_mean": pnl.mean(),
                "pnl_sem": pnl.std(ddof=0) / np.sqrt(max(len(pnl), 1)),
            }
        )
    frame = pd.DataFrame(rows).sort_values("pnl_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [method_color(method) for method in frame["method"]]
    ax.barh(
        frame["method"].map(display_method),
        frame["pnl_mean"],
        xerr=frame["pnl_sem"],
        color=colors,
        alpha=0.92,
        error_kw={"ecolor": "#343A40", "elinewidth": 1.0, "capsize": 3},
    )
    ax.axvline(0.0, color="#222222", linewidth=1.0)
    ax.set_title(f"Mean Test PnL Across {experiment_label}")
    ax.set_xlabel("PnL per episode")
    ax.set_ylabel("")
    for y, value in enumerate(frame["pnl_mean"]):
        ha = "left" if value >= 0 else "right"
        offset = 0.7 if value >= 0 else -0.7
        ax.text(value + offset, y, f"{value:.1f}", va="center", ha=ha, fontsize=9, color="#252A31")
    save_figure(fig, output_dir, "01_overall_test_pnl")


def plot_risk_return(aggregate: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.2))
    for _, row in aggregate.iterrows():
        method = row["method"]
        size = np.clip(float(row["test_volume_mean"]) / 300.0, 35, 240)
        ax.scatter(
            row["test_avg_abs_position_mean"],
            row["test_pnl_mean"],
            s=size,
            color=method_color(method),
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.annotate(display_method(method), (row["test_avg_abs_position_mean"], row["test_pnl_mean"]), xytext=(5, 4), textcoords="offset points", fontsize=9)
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_title("Risk-Return View")
    ax.set_xlabel("Average absolute inventory")
    ax.set_ylabel("Mean test PnL")
    ax.text(0.99, 0.02, "Bubble size = trading volume", transform=ax.transAxes, ha="right", va="bottom", color="#5B616E", fontsize=9)
    save_figure(fig, output_dir, "02_risk_return_inventory")


def plot_pretrain_f1(pretrain: pd.DataFrame, output_dir: Path, experiment_label: str) -> None:
    singles = pretrain[pretrain["scope"] == "single"].copy().sort_values("test_f1", ascending=False)
    joint = pretrain[pretrain["scope"].astype(str).str.startswith("joint")]
    joint_f1 = float(joint["test_f1"].iloc[0]) if not joint.empty else np.nan

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.bar(singles["ticker"], singles["test_f1"], color="#4E79A7", alpha=0.9)
    if np.isfinite(joint_f1):
        ax.axhline(joint_f1, color="#C76E2C", linewidth=2.2, linestyle="--", label=f"Joint encoder ({experiment_label}, {joint_f1:.3f})")
    ax.set_title("Attn-LOB Pretraining: Test Macro F1")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, max(0.78, singles["test_f1"].max() + 0.08))
    ax.tick_params(axis="x", rotation=35)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.012, f"{height:.2f}", ha="center", va="bottom", fontsize=8)
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "03_pretrain_test_f1")


def plot_method_heatmap(comparison: pd.DataFrame, output_dir: Path) -> None:
    methods = ["single_ppo", "joint_ppo", "single_dueling_dqn", "joint_dueling_dqn", "baseline_fixed_2", "baseline_as"]
    frame = comparison[comparison["method"].isin(methods)].copy()
    pivot = frame.pivot(index="ticker", columns="method", values="test_pnl_mean")
    pivot = pivot[[method for method in methods if method in pivot.columns]]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    vmax = np.nanmax(np.abs(pivot.to_numpy(dtype=float)))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_title("Per-Stock Test PnL")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([display_method(method) for method in pivot.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.grid(False)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=8, color="#171A1F")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.02)
    colorbar.set_label("PnL")
    save_figure(fig, output_dir, "04_per_stock_pnl_heatmap")


def plot_latency_delta(latency: pd.DataFrame, output_dir: Path, top_methods: list[str]) -> None:
    frame = latency[latency["method"].isin(top_methods)].copy()
    pivot = frame.pivot(index="scenario", columns="method", values="pnl_mean").reindex(SCENARIO_ORDER)
    if "paper_base" in pivot.index:
        pivot = pivot.subtract(pivot.loc["paper_base"], axis=1)

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    for method in pivot.columns:
        ax.plot(
            [SCENARIO_LABELS.get(scenario, scenario) for scenario in pivot.index],
            pivot[method],
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            color=method_color(method),
            label=display_method(method),
        )
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_title("Latency and Order-Loss Robustness")
    ax.set_xlabel("Stress scenario")
    ax.set_ylabel("PnL change vs base")
    ax.tick_params(axis="x", rotation=28)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, output_dir, "05_latency_pnl_delta")


def plot_latency_heatmap(latency: pd.DataFrame, output_dir: Path) -> None:
    frame = latency.copy()
    pivot = frame.pivot(index="method", columns="scenario", values="pnl_mean")
    pivot = pivot.reindex(ordered_methods(list(pivot.index)))
    pivot = pivot[[scenario for scenario in SCENARIO_ORDER if scenario in pivot.columns]]

    fig, ax = plt.subplots(figsize=(12, 6.6))
    vmax = np.nanmax(np.abs(pivot.to_numpy(dtype=float)))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_title("Mean PnL Under Execution Stress")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([SCENARIO_LABELS.get(scenario, scenario) for scenario in pivot.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([display_method(method) for method in pivot.index])
    ax.grid(False)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8, color="#171A1F")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.02)
    colorbar.set_label("Mean PnL")
    save_figure(fig, output_dir, "06_latency_pnl_heatmap")


def plot_regime_facets(regime: pd.DataFrame, output_dir: Path, top_methods: list[str]) -> None:
    frame = regime[regime["method"].isin(top_methods)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), sharey=True)
    for ax, regime_type in zip(axes, REGIME_LABELS, strict=False):
        sub = frame[frame["regime_type"] == regime_type]
        pivot = sub.pivot(index="regime", columns="method", values="pnl_mean").reindex(["low", "mid", "high"])
        x = np.arange(len(pivot.index))
        for method in pivot.columns:
            ax.plot(x, pivot[method], marker="o", linewidth=2.0, color=method_color(method), label=display_method(method))
        ax.axhline(0.0, color="#222222", linewidth=0.9)
        ax.set_title(REGIME_LABELS[regime_type])
        ax.set_xticks(x)
        ax.set_xticklabels(["Low", "Mid", "High"])
        ax.set_xlabel("Regime")
    axes[0].set_ylabel("Mean test PnL")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Regime Sensitivity", y=1.03, fontsize=16)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_figure(fig, output_dir, "07_regime_sensitivity_pnl")


def load_episode_curves(experiment_root: Path, comparison: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows = []
    for _, row in comparison[comparison["method"].isin(methods)].iterrows():
        method = row["method"]
        ticker = row["ticker"]
        if method.startswith("baseline_"):
            baseline = method.removeprefix("baseline_")
            path = experiment_root / "baselines" / ticker / "test" / f"{baseline}_episode_results.csv"
        else:
            path = experiment_root / f"rl_{row['scope']}" / row["agent"] / ticker / "test_episode_results.csv"
        if not path.exists():
            continue
        episode = pd.read_csv(path)
        episode["method"] = method
        episode["ticker"] = ticker
        rows.append(episode[["method", "ticker", "episode_index", "pnl"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_cumulative_pnl(experiment_root: Path, comparison: pd.DataFrame, output_dir: Path, top_methods: list[str]) -> None:
    episodes = load_episode_curves(experiment_root, comparison, top_methods)
    if episodes.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    for method, group in episodes.groupby("method"):
        ordered = group.sort_values(["ticker", "episode_index"]).reset_index(drop=True)
        ax.plot(
            np.arange(1, len(ordered) + 1),
            ordered["pnl"].cumsum(),
            linewidth=2.2,
            color=method_color(method),
            label=display_method(method),
        )
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_title("Cumulative Test PnL")
    ax.set_xlabel("Test episodes concatenated across stocks")
    ax.set_ylabel("Cumulative PnL")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, output_dir, "08_cumulative_test_pnl")


def main() -> None:
    args = parse_args()
    set_style()
    experiment_root = Path(args.experiment_root)
    output_dir = Path(args.output_dir) if args.output_dir else experiment_root / "final_plots"
    reports = experiment_root / "reports"

    pretrain = pd.read_csv(reports / "pretrain_summary.csv")
    aggregate = pd.read_csv(reports / "comparison_test_aggregate.csv")
    comparison = pd.read_csv(reports / "comparison_test_summary.csv")
    latency = pd.read_csv(experiment_root / "latency_scenarios" / "reports" / "latency_aggregate.csv")
    regime = pd.read_csv(experiment_root / "regime_sensitivity" / "reports" / "regime_sensitivity_summary.csv")
    top_methods = [item.strip() for item in args.top_methods.split(",") if item.strip()]

    plot_overall_pnl(comparison, output_dir, args.experiment_label)
    plot_risk_return(aggregate, output_dir)
    plot_pretrain_f1(pretrain, output_dir, args.experiment_label)
    plot_method_heatmap(comparison, output_dir)
    plot_latency_delta(latency, output_dir, top_methods)
    plot_latency_heatmap(latency, output_dir)
    plot_regime_facets(regime, output_dir, top_methods)
    plot_cumulative_pnl(experiment_root, comparison, output_dir, top_methods)

    files = sorted(output_dir.glob("*.png"))
    index = pd.DataFrame({"plot": [path.name for path in files], "path": [str(path) for path in files]})
    index.to_csv(output_dir / "plot_index.csv", index=False)
    print(f"plots_written: {output_dir}")
    print(index.to_string(index=False))


if __name__ == "__main__":
    main()
