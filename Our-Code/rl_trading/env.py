from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import (
    MARKET_STATE_COLUMNS,
    ORDERBOOK_COLUMNS,
    PreparedMarketData,
    load_prepared_market_data,
    normalized_lob_observation,
    normalized_market_state,
    split_index_bounds,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - dependency availability is environment-specific
    class _FallbackEnv:
        pass

    class _FallbackSpaces:
        def __getattr__(self, name: str):
            raise ImportError("gymnasium is required for RL environments")

    class _FallbackGym:
        Env = _FallbackEnv

    gym = _FallbackGym()
    spaces = _FallbackSpaces()


@dataclass
class LOBEnvConfig:
    ticker: str = "AAPL"
    split: str = "train"
    processed_root: str | None = None
    time_window: int = 50
    episode_length: int = 2000
    trade_unit: int = 100
    max_inventory_units: int = 10
    tick_size: float = 0.01
    max_bias: float = 0.05
    max_spread: float = 0.10
    reward_eta: float = 0.5
    reward_theta: float = 0.01
    seed: int = 7


class BaseLOBMarketMakingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: LOBEnvConfig):
        super().__init__()
        self.config = config
        self.market_data: PreparedMarketData = load_prepared_market_data(config.ticker, config.processed_root)
        self.frame = self.market_data.frame.reset_index(drop=True)
        self.random_state = np.random.default_rng(config.seed)
        self.split_start, self.split_end = split_index_bounds(self.frame, config.split)
        self.max_inventory = config.max_inventory_units * config.trade_unit

        self.observation_space = spaces.Dict(
            {
                "lob_state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(config.time_window, len(ORDERBOOK_COLUMNS)),
                    dtype=np.float32,
                ),
                "market_state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(MARKET_STATE_COLUMNS),),
                    dtype=np.float32,
                ),
                "agent_state": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(24,),
                    dtype=np.float32,
                ),
            }
        )

        self.episode_start = 0
        self.episode_end = 0
        self.current_index = 0
        self.inventory = 0
        self.cash = 0.0
        self.value = 0.0
        self.prev_value = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.random_state = np.random.default_rng(seed)

        min_start = max(self.split_start, self.config.time_window)
        max_start = max(min_start, self.split_end - self.config.episode_length - 1)
        if max_start <= min_start or self.config.split != "train":
            self.episode_start = min_start
        else:
            self.episode_start = int(self.random_state.integers(min_start, max_start + 1))
        self.episode_end = min(self.episode_start + self.config.episode_length, self.split_end - 1)

        self.current_index = self.episode_start
        self.inventory = 0
        self.cash = 0.0
        self.value = 0.0
        self.prev_value = 0.0
        observation = self._observation_at(self.current_index)
        return observation, self._info(trade_price=0.0, trade_volume=0)

    def step(self, action):
        row = self.frame.iloc[self.current_index]
        next_index = min(self.current_index + 1, self.split_end - 1)
        next_row = self.frame.iloc[next_index]
        ask_price, bid_price = self._action_to_quotes(row, action)

        trade_price, trade_volume = self._match_orders(row, next_row, ask_price, bid_price)
        self._apply_trade(next_row["mid_price"], trade_price, trade_volume)

        pnl = self.value - self.prev_value
        dampened_pnl = pnl - max(0.0, self.config.reward_eta * pnl)
        inventory_penalty = self.config.reward_theta * (self.inventory / max(self.max_inventory, 1)) ** 2
        reward = float(dampened_pnl - inventory_penalty)
        self.prev_value = self.value

        self.current_index = next_index
        terminated = self.current_index >= self.episode_end
        truncated = False

        if terminated and self.inventory != 0:
            close_price = next_row["bid_price_1"] if self.inventory > 0 else next_row["ask_price_1"]
            self._apply_trade(next_row["mid_price"], close_price, -self.inventory)
            reward += float(self.value - self.prev_value)
            self.prev_value = self.value

        observation = self._observation_at(self.current_index)
        return observation, reward, terminated, truncated, self._info(trade_price=trade_price, trade_volume=trade_volume)

    def _observation_at(self, row_index: int) -> dict[str, np.ndarray]:
        agent_state = self._agent_state(row_index)
        market_state = normalized_market_state(
            self.frame,
            row_index,
            self.market_data.market_state_mean,
            self.market_data.market_state_std,
        )
        lob_state = normalized_lob_observation(self.frame, row_index, self.config.time_window)
        return {
            "lob_state": lob_state.astype(np.float32),
            "market_state": market_state.astype(np.float32),
            "agent_state": agent_state.astype(np.float32),
        }

    def _agent_state(self, row_index: int) -> np.ndarray:
        inventory_norm = np.clip(self.inventory / max(self.max_inventory, 1), -1.0, 1.0)
        progress = np.clip((row_index - self.episode_start) / max(self.episode_end - self.episode_start, 1), 0.0, 1.0)
        return np.asarray([inventory_norm] * 12 + [progress] * 12, dtype=np.float32)

    def _apply_trade(self, mid_price: float, trade_price: float, trade_volume: int) -> None:
        self.inventory += int(trade_volume)
        self.cash -= float(trade_volume) * float(trade_price)
        self.value = self.cash + self.inventory * float(mid_price)

    def _match_orders(self, row, next_row, ask_price: float, bid_price: float) -> tuple[float, int]:
        trade_price = 0.0
        trade_volume = 0

        if ask_price > 0.0 and self.inventory > -self.max_inventory:
            if ask_price <= float(row["bid_price_1"]):
                trade_price = float(row["bid_price_1"])
                trade_volume -= self.config.trade_unit
            elif (
                int(next_row["is_execution"]) == 1
                and int(next_row["direction"]) == -1
                and float(next_row["trade_price"]) >= ask_price
            ):
                depth = max(float(row["ask_size_1"]), 1.0)
                fill_prob = min(1.0, float(next_row["trade_size"]) / (float(next_row["trade_size"]) + depth))
                if self.random_state.random() < fill_prob:
                    trade_price = ask_price
                    trade_volume -= self.config.trade_unit

        if bid_price > 0.0 and self.inventory < self.max_inventory:
            if bid_price >= float(row["ask_price_1"]):
                trade_price = float(row["ask_price_1"])
                trade_volume += self.config.trade_unit
            elif (
                int(next_row["is_execution"]) == 1
                and int(next_row["direction"]) == 1
                and float(next_row["trade_price"]) <= bid_price
            ):
                depth = max(float(row["bid_size_1"]), 1.0)
                fill_prob = min(1.0, float(next_row["trade_size"]) / (float(next_row["trade_size"]) + depth))
                if self.random_state.random() < fill_prob:
                    trade_price = bid_price
                    trade_volume += self.config.trade_unit

        return trade_price, trade_volume

    def _legal_prices(self, ask_price: float, bid_price: float) -> tuple[float, float]:
        ask_price = np.ceil(ask_price / self.config.tick_size) * self.config.tick_size
        bid_price = np.floor(bid_price / self.config.tick_size) * self.config.tick_size
        ask_price = round(float(ask_price), 2)
        bid_price = round(float(bid_price), 2)
        if ask_price <= bid_price:
            ask_price = round(bid_price + self.config.tick_size, 2)
        return ask_price, bid_price

    def _info(self, trade_price: float, trade_volume: int) -> dict[str, float]:
        return {
            "inventory": float(self.inventory),
            "cash": float(self.cash),
            "value": float(self.value),
            "trade_price": float(trade_price),
            "trade_volume": float(trade_volume),
        }

    def _action_to_quotes(self, row, action) -> tuple[float, float]:
        raise NotImplementedError


class LOBDiscreteMarketMakingEnv(BaseLOBMarketMakingEnv):
    def __init__(self, config: LOBEnvConfig):
        super().__init__(config)
        self.action_space = spaces.Discrete(5)

    def _action_to_quotes(self, row, action) -> tuple[float, float]:
        ask_1 = float(row["ask_price_1"])
        bid_1 = float(row["bid_price_1"])
        tick = self.config.tick_size
        action_id = int(action)

        if action_id == 0:
            ask_price, bid_price = ask_1, bid_1
        elif action_id == 1:
            ask_price, bid_price = ask_1 + tick, bid_1
        elif action_id == 2:
            ask_price, bid_price = ask_1, bid_1 - tick
        elif action_id == 3:
            ask_price, bid_price = ask_1 + tick, bid_1 - tick
        else:
            if self.inventory > 0:
                ask_price, bid_price = bid_1, 0.0
            elif self.inventory < 0:
                ask_price, bid_price = 0.0, ask_1
            else:
                ask_price, bid_price = ask_1, bid_1

        if ask_price > 0.0 and bid_price > 0.0:
            ask_price, bid_price = self._legal_prices(ask_price, bid_price)
        return ask_price, bid_price


class LOBContinuousMarketMakingEnv(BaseLOBMarketMakingEnv):
    def __init__(self, config: LOBEnvConfig):
        super().__init__(config)
        self.action_space = spaces.Box(
            low=np.asarray([-1.0, 0.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _action_to_quotes(self, row, action) -> tuple[float, float]:
        action = np.asarray(action, dtype=np.float32)
        mid_price = float(row["mid_price"])
        inventory_norm = np.clip(self.inventory / max(self.max_inventory, 1), -1.0, 1.0)
        bias = float(action[0]) * self.config.max_bias
        spread = self.config.tick_size + float(action[1]) * (self.config.max_spread - self.config.tick_size)
        reservation = mid_price - bias - 0.5 * inventory_norm * self.config.max_bias
        ask_price = reservation + spread / 2.0
        bid_price = reservation - spread / 2.0
        return self._legal_prices(ask_price, bid_price)
