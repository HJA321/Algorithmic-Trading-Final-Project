from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _level(row, side: str, level: int) -> float:
    return float(row[f"{side}_price_{int(level)}"])


@dataclass
class FixedLevelPolicy:
    level: int = 1

    @property
    def name(self) -> str:
        return f"fixed_{self.level}"

    def reset(self, episode_index: int) -> None:
        return None

    def quote(self, env) -> tuple[float, float]:
        level = int(np.clip(self.level, 1, 10))
        row = env.current_decision_row()
        return _level(row, "ask", level), _level(row, "bid", level)


@dataclass
class RandomLevelPolicy:
    max_level: int = 5
    seed: int = 7

    @property
    def name(self) -> str:
        return f"random_{self.max_level}level"

    def reset(self, episode_index: int) -> None:
        self.random_state = np.random.default_rng(self.seed + int(episode_index))

    def quote(self, env) -> tuple[float, float]:
        if not hasattr(self, "random_state"):
            self.reset(0)
        max_level = int(np.clip(self.max_level, 1, 10))
        ask_level = int(self.random_state.integers(1, max_level + 1))
        bid_level = int(self.random_state.integers(1, max_level + 1))
        row = env.current_decision_row()
        return _level(row, "ask", ask_level), _level(row, "bid", bid_level)


@dataclass
class AvellanedaStoikovPolicy:
    gamma: float = 0.1
    kappa: float = 20.0
    sigma_window: int = 200
    min_spread_ticks: int = 1

    @property
    def name(self) -> str:
        return "as"

    def reset(self, episode_index: int) -> None:
        return None

    def quote(self, env) -> tuple[float, float]:
        row = env.current_decision_row()
        mid_price = float(row["mid_price"])
        remaining = max(0.0, 1.0 - env.episode_progress())
        inventory_units = float(env.inventory) / max(float(env.config.trade_unit), 1.0)
        sigma_sq = self._realized_variance(env)

        gamma = max(float(self.gamma), 1e-8)
        kappa = max(float(self.kappa), 1e-8)
        reservation = mid_price - inventory_units * gamma * sigma_sq * remaining
        spread = gamma * sigma_sq * remaining + (2.0 / gamma) * np.log1p(gamma / kappa)
        spread = max(float(spread), env.config.tick_size * max(1, int(self.min_spread_ticks)))

        return reservation + spread / 2.0, reservation - spread / 2.0

    def _realized_variance(self, env) -> float:
        end = env.current_decision_index() + 1
        start = max(0, end - max(2, int(self.sigma_window)))
        mid = env.frame.iloc[start:end]["mid_price"].to_numpy(dtype=np.float64)
        if mid.size < 3:
            return 0.0
        returns = np.diff(np.log(np.clip(mid, 1e-8, None)))
        if returns.size == 0:
            return 0.0
        return float(np.mean(returns * returns))


def make_baseline_policy(name: str, *, seed: int = 7, random_max_level: int = 5) -> FixedLevelPolicy | RandomLevelPolicy | AvellanedaStoikovPolicy:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"random", "random_level"}:
        return RandomLevelPolicy(max_level=random_max_level, seed=seed)
    if normalized in {"as", "avellaneda_stoikov"}:
        return AvellanedaStoikovPolicy()
    if normalized.startswith("fixed_"):
        return FixedLevelPolicy(level=int(normalized.split("_", 1)[1]))
    if normalized.startswith("fixed"):
        suffix = normalized.removeprefix("fixed")
        return FixedLevelPolicy(level=int(suffix or 1))
    raise ValueError(f"Unknown baseline policy: {name}")

