from __future__ import annotations

import time
from typing import Protocol

import numpy as np
import pandas as pd


class PredictiveAgent(Protocol):
    def predict(self, observation, deterministic: bool = True):
        ...


class QuotePolicy(Protocol):
    name: str

    def reset(self, episode_index: int) -> None:
        ...

    def quote(self, env) -> tuple[float, float]:
        ...


def unwrap_env(env):
    current = env
    while hasattr(current, "env"):
        current = current.env
    return current


def summarize_episode_frame(frame: pd.DataFrame) -> dict[str, float]:
    summary: dict[str, float] = {"episodes": float(len(frame))}
    if frame.empty:
        return summary
    for column in frame.columns:
        if column == "episode_index":
            continue
        summary[f"{column}_mean"] = float(frame[column].mean())
        summary[f"{column}_sum"] = float(frame[column].sum())
        summary[f"{column}_std"] = float(frame[column].std(ddof=0))
    if "pnl" in frame:
        pnl_std = float(frame["pnl"].std(ddof=0))
        summary["sharpe"] = float(frame["pnl"].mean() / pnl_std) if pnl_std > 1e-7 else 0.0
    return summary


def evaluate_agent(agent: PredictiveAgent, env, max_episodes: int, deterministic: bool = True) -> tuple[pd.DataFrame, dict[str, float]]:
    base_env = unwrap_env(env)
    episode_count = min(max_episodes, base_env.episode_count())
    rows: list[dict[str, float]] = []
    for episode_index in range(episode_count):
        observation, _ = env.reset(options={"episode_index": episode_index})
        done = False
        decision_times_ms: list[float] = []
        while not done:
            started = time.perf_counter()
            action, _ = agent.predict(observation, deterministic=deterministic)
            decision_times_ms.append((time.perf_counter() - started) * 1000.0)
            observation, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
        result = base_env.get_final_result()
        result.update(_decision_time_summary(decision_times_ms))
        result["episode_index"] = float(episode_index)
        rows.append(result)

    frame = pd.DataFrame(rows)
    return frame, summarize_episode_frame(frame)


def evaluate_quote_policy(policy: QuotePolicy, env, max_episodes: int) -> tuple[pd.DataFrame, dict[str, float]]:
    episode_count = min(max_episodes, env.episode_count())
    rows: list[dict[str, float]] = []
    for episode_index in range(episode_count):
        env.reset(options={"episode_index": episode_index})
        policy.reset(episode_index)
        done = False
        decision_times_ms: list[float] = []
        while not done:
            started = time.perf_counter()
            ask_price, bid_price = policy.quote(env)
            decision_times_ms.append((time.perf_counter() - started) * 1000.0)
            _, _, terminated, truncated, _ = env.step_quotes(ask_price, bid_price)
            done = bool(terminated or truncated)
        result = env.get_final_result()
        result.update(_decision_time_summary(decision_times_ms))
        result["episode_index"] = float(episode_index)
        rows.append(result)

    frame = pd.DataFrame(rows)
    return frame, summarize_episode_frame(frame)


def _decision_time_summary(decision_times_ms: list[float]) -> dict[str, float]:
    if not decision_times_ms:
        return {
            "decision_count": 0.0,
            "decision_time_ms_mean": 0.0,
            "decision_time_ms_p95": 0.0,
            "decision_time_ms_max": 0.0,
        }
    values = np.asarray(decision_times_ms, dtype=np.float64)
    return {
        "decision_count": float(values.size),
        "decision_time_ms_mean": float(values.mean()),
        "decision_time_ms_p95": float(np.percentile(values, 95)),
        "decision_time_ms_max": float(values.max()),
    }
