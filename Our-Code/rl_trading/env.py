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
    latency: int = 1
    order_latency_events: int = 0
    order_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
    order_loss_probability: float = 0.0
    episode_length: int = 2000
    trade_unit: int = 100
    max_inventory_units: int = 10
    tick_size: float = 0.01
    max_bias: float = 0.05
    max_spread: float = 0.10
    reward_eta: float = 0.5
    reward_theta: float = 0.01
    transaction_cost_bps: float = 0.0
    fixed_fee_per_trade: float = 0.0
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
        self.time_values = self.frame["time"].to_numpy(dtype=np.float64) if "time" in self.frame else None
        self.date_values = self.frame["trading_date"].astype(str).to_numpy() if "trading_date" in self.frame else None
        self.trade_indices = self.frame.index[
            (self.frame.index >= self.split_start)
            & (self.frame.index < self.split_end)
            & (self.frame["is_execution"] == 1)
        ].to_numpy(dtype=np.int64)

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
        self.volume = 0.0
        self.episode_reward = 0.0
        self.reward_dampened_pnl = 0.0
        self.reward_trading_pnl = 0.0
        self.reward_inventory_penalty = 0.0
        self.transaction_cost_total = 0.0
        self.position_abs_sum = 0.0
        self.position_sum = 0.0
        self.position_samples = 0
        self.spread_sum = 0.0
        self.spread_samples = 0
        self.order_attempts = 0
        self.order_acceptances = 0
        self.order_drops = 0
        self.order_latency_samples = 0
        self.order_latency_events_sum = 0.0
        self.order_latency_seconds_sum = 0.0
        self._sequential_episode_index = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.random_state = np.random.default_rng(seed)
            self._sequential_episode_index = 0

        min_start = max(self.split_start, self.config.time_window + self.config.latency)
        max_start = max(min_start, self.split_end - self.config.episode_length - 1)
        episode_index = None if options is None else options.get("episode_index")
        if episode_index is not None:
            self.episode_start = self._episode_start_for_index(int(episode_index), min_start)
        elif max_start <= min_start or self.config.split != "train":
            self.episode_start = self._episode_start_for_index(self._sequential_episode_index, min_start)
            self._sequential_episode_index = (self._sequential_episode_index + 1) % self.episode_count()
        else:
            self.episode_start = int(self.random_state.integers(min_start, max_start + 1))
        self.episode_end = min(self.episode_start + self.config.episode_length, self.split_end - 1)

        self.current_index = self._next_trade_index(self.episode_start)
        self.inventory = 0
        self.cash = 0.0
        self.value = 0.0
        self.prev_value = 0.0
        self.volume = 0.0
        self.episode_reward = 0.0
        self.reward_dampened_pnl = 0.0
        self.reward_trading_pnl = 0.0
        self.reward_inventory_penalty = 0.0
        self.transaction_cost_total = 0.0
        self.position_abs_sum = 0.0
        self.position_sum = 0.0
        self.position_samples = 0
        self.spread_sum = 0.0
        self.spread_samples = 0
        self.order_attempts = 0
        self.order_acceptances = 0
        self.order_drops = 0
        self.order_latency_samples = 0
        self.order_latency_events_sum = 0.0
        self.order_latency_seconds_sum = 0.0
        observation = self._observation_at(self._decision_index(self.current_index))
        return observation, self._info(trade_price=0.0, trade_volume=0)

    def episode_count(self) -> int:
        min_start = max(self.split_start, self.config.time_window + self.config.latency)
        available = max(self.split_end - min_start, 0)
        return max(1, available // max(self.config.episode_length, 1))

    def _episode_start_for_index(self, episode_index: int, min_start: int | None = None) -> int:
        if min_start is None:
            min_start = max(self.split_start, self.config.time_window + self.config.latency)
        episode_count = self.episode_count()
        bounded_index = max(0, min(int(episode_index), episode_count - 1))
        return min(min_start + bounded_index * self.config.episode_length, self.split_end - 1)

    def step(self, action):
        quote_row = self.current_decision_row()
        ask_price, bid_price = self._action_to_quotes(quote_row, action)
        return self.step_quotes(ask_price, bid_price)

    def step_quotes(self, ask_price: float, bid_price: float):
        decision_index = self.current_index
        if ask_price > 0.0 and bid_price > 0.0 and np.isfinite(ask_price) and np.isfinite(bid_price):
            ask_price, bid_price = self._legal_prices(ask_price, bid_price)
        ask_price, bid_price = self._apply_order_acceptance(ask_price, bid_price)
        self._record_quote(ask_price, bid_price)

        arrival_index = self._arrival_index(decision_index)
        execution_index = self._next_trade_index(arrival_index)
        self._record_effective_latency(decision_index, execution_index, ask_price, bid_price)

        trade_price = 0.0
        trade_volume = 0
        reward = 0.0
        has_execution_event = execution_index < self.episode_end and execution_index < self.split_end
        if has_execution_event:
            event_row = self.frame.iloc[execution_index]
            previous_row = self.frame.iloc[max(execution_index - 1, 0)]
            trade_price, trade_volume = self._match_orders(event_row, previous_row, ask_price, bid_price)
            reward = self._apply_trade_and_reward(event_row["mid_price"], trade_price, trade_volume)
            self.current_index = execution_index
        else:
            close_reference_index = max(min(self.episode_end, self.split_end, len(self.frame) - 1) - 1, 0)
            previous_row = self.frame.iloc[close_reference_index]
            self.current_index = close_reference_index

        next_trade_index = self._next_trade_index(self.current_index + 1)
        terminated = next_trade_index >= self.episode_end or next_trade_index >= self.split_end
        truncated = False

        if terminated and self.inventory != 0:
            close_price = previous_row["bid_price_1"] if self.inventory > 0 else previous_row["ask_price_1"]
            reward += self._apply_trade_and_reward(previous_row["mid_price"], close_price, -self.inventory)
        else:
            self.current_index = next_trade_index

        self.episode_reward += float(reward)
        self.position_sum += float(self.inventory)
        self.position_abs_sum += abs(float(self.inventory))
        self.position_samples += 1

        observation = self._observation_at(self._decision_index(self.current_index))
        return observation, reward, terminated, truncated, self._info(trade_price=trade_price, trade_volume=trade_volume)

    def _next_trade_index(self, row_index: int) -> int:
        minimum_index = max(row_index, self.split_start + self.config.time_window + self.config.latency)
        position = int(np.searchsorted(self.trade_indices, minimum_index, side="left"))
        if position >= len(self.trade_indices):
            return self.split_end
        return int(self.trade_indices[position])

    def _decision_index(self, row_index: int) -> int:
        return max(self.config.time_window, int(row_index) - self.config.latency)

    def _arrival_index(self, row_index: int) -> int:
        event_delay = max(0, int(self.config.order_latency_events))
        arrival_index = min(int(row_index) + event_delay, self.split_end)
        delay_ms = max(0.0, float(self.config.order_latency_ms) + float(self.config.inference_latency_ms))
        if delay_ms > 0.0:
            arrival_index = max(arrival_index, self._row_index_after_delay_ms(row_index, delay_ms))
        return min(arrival_index, self.split_end)

    def _row_index_after_delay_ms(self, row_index: int, delay_ms: float) -> int:
        if self.time_values is None or delay_ms <= 0.0:
            return int(row_index)
        start_index = int(row_index)
        target_time = self.time_values[start_index] + delay_ms / 1000.0
        start_date = self.date_values[start_index] if self.date_values is not None else None
        index = start_index
        while index < self.split_end:
            if start_date is not None and self.date_values[index] != start_date:
                break
            if self.time_values[index] >= target_time:
                return int(index)
            index += 1
        return int(min(index, self.split_end))

    def _apply_order_acceptance(self, ask_price: float, bid_price: float) -> tuple[float, float]:
        loss_probability = float(np.clip(self.config.order_loss_probability, 0.0, 1.0))
        ask_attempted = ask_price > 0.0
        bid_attempted = bid_price > 0.0
        ask_accepted = ask_attempted
        bid_accepted = bid_attempted
        if loss_probability > 0.0:
            ask_accepted = ask_attempted and self.random_state.random() >= loss_probability
            bid_accepted = bid_attempted and self.random_state.random() >= loss_probability

        self.order_attempts += int(ask_attempted) + int(bid_attempted)
        self.order_acceptances += int(ask_accepted) + int(bid_accepted)
        self.order_drops += int(ask_attempted and not ask_accepted) + int(bid_attempted and not bid_accepted)

        return ask_price if ask_accepted else 0.0, bid_price if bid_accepted else 0.0

    def _record_effective_latency(self, decision_index: int, execution_index: int, ask_price: float, bid_price: float) -> None:
        if ask_price <= 0.0 and bid_price <= 0.0:
            return
        bounded_execution_index = min(int(execution_index), self.split_end - 1, len(self.frame) - 1)
        bounded_decision_index = min(int(decision_index), bounded_execution_index)
        self.order_latency_samples += 1
        self.order_latency_events_sum += max(0, bounded_execution_index - bounded_decision_index)
        if self.time_values is not None:
            same_day = True
            if self.date_values is not None:
                same_day = self.date_values[bounded_execution_index] == self.date_values[bounded_decision_index]
            if same_day:
                self.order_latency_seconds_sum += max(
                    0.0,
                    float(self.time_values[bounded_execution_index] - self.time_values[bounded_decision_index]),
                )

    def current_decision_index(self) -> int:
        return self._decision_index(self.current_index)

    def current_decision_row(self):
        return self.frame.iloc[self.current_decision_index()]

    def episode_progress(self) -> float:
        return float(
            np.clip(
                (self.current_index - self.episode_start) / max(self.episode_end - self.episode_start, 1),
                0.0,
                1.0,
            )
        )

    def _observation_at(self, row_index: int) -> dict[str, np.ndarray]:
        agent_state = self._agent_state(row_index)
        market_state = normalized_market_state(
            self.frame,
            row_index,
            self.market_data.market_state_mean,
            self.market_data.market_state_std,
        )
        lob_state = normalized_lob_observation(
            self.frame,
            row_index,
            self.config.time_window,
            self.market_data.lob_size_max,
        )
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
        notional = abs(float(trade_volume) * float(trade_price))
        transaction_cost = self._transaction_cost(notional, trade_volume)
        self.cash -= float(trade_volume) * float(trade_price)
        self.cash -= transaction_cost
        self.transaction_cost_total += transaction_cost
        self.value = self.cash + self.inventory * float(mid_price)
        if trade_volume > 0:
            self.volume += float(trade_volume) * float(trade_price)

    def _transaction_cost(self, notional: float, trade_volume: int) -> float:
        if trade_volume == 0:
            return 0.0
        variable_cost = notional * max(0.0, float(self.config.transaction_cost_bps)) / 10_000.0
        fixed_cost = max(0.0, float(self.config.fixed_fee_per_trade))
        return float(variable_cost + fixed_cost)

    def _apply_trade_and_reward(self, mid_price: float, trade_price: float, trade_volume: int) -> float:
        self._apply_trade(mid_price, trade_price, trade_volume)
        pnl = float(self.value - self.prev_value)
        dampened_pnl = pnl - max(0.0, self.config.reward_eta * pnl)
        trading_pnl = float(trade_volume) * (float(mid_price) - float(trade_price)) if trade_volume else 0.0
        inventory_penalty = self.config.reward_theta * (self.inventory / max(self.config.trade_unit, 1)) ** 2
        reward = dampened_pnl + trading_pnl - inventory_penalty

        self.reward_dampened_pnl += dampened_pnl
        self.reward_trading_pnl += trading_pnl
        self.reward_inventory_penalty += inventory_penalty
        self.prev_value = self.value
        return float(reward)

    def _match_orders(self, event_row, previous_row, ask_price: float, bid_price: float) -> tuple[float, int]:
        trade_price = 0.0
        trade_volume = 0

        if ask_price > 0.0 and self.inventory > -self.max_inventory:
            ask_order_volume = -self.config.trade_unit
            if bid_price == 0.0 and ask_price <= self.config.tick_size and self.inventory > 0:
                ask_order_volume = -int(self.inventory)
            if ask_price <= float(previous_row["bid_price_1"]):
                trade_price = float(previous_row["bid_price_1"])
                trade_volume = ask_order_volume
            elif (
                int(event_row["is_execution"]) == 1
                and int(event_row["direction"]) == -1
                and float(event_row["trade_price"]) >= ask_price
            ):
                fill_prob = 1.0
                if float(event_row["trade_price"]) == ask_price:
                    depth = max(float(previous_row["ask_size_1"]), 1.0)
                    fill_prob = min(1.0, float(event_row["trade_size"]) / (float(event_row["trade_size"]) + depth))
                if self.random_state.random() < fill_prob:
                    trade_price = ask_price
                    trade_volume = ask_order_volume

        if bid_price > 0.0 and self.inventory < self.max_inventory:
            bid_order_volume = self.config.trade_unit
            if not np.isfinite(bid_price) and self.inventory < 0:
                bid_order_volume = -int(self.inventory)
            if bid_price >= float(previous_row["ask_price_1"]):
                trade_price = float(previous_row["ask_price_1"])
                trade_volume = bid_order_volume
            elif (
                int(event_row["is_execution"]) == 1
                and int(event_row["direction"]) == 1
                and float(event_row["trade_price"]) <= bid_price
            ):
                fill_prob = 1.0
                if float(event_row["trade_price"]) == bid_price:
                    depth = max(float(previous_row["bid_size_1"]), 1.0)
                    fill_prob = min(1.0, float(event_row["trade_size"]) / (float(event_row["trade_size"]) + depth))
                if self.random_state.random() < fill_prob:
                    trade_price = bid_price
                    trade_volume = bid_order_volume

        return trade_price, trade_volume

    def _legal_prices(self, ask_price: float, bid_price: float) -> tuple[float, float]:
        if not np.isfinite(ask_price) or not np.isfinite(bid_price):
            return ask_price, bid_price
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
            "episode_reward": float(self.episode_reward),
        }

    def _record_quote(self, ask_price: float, bid_price: float) -> None:
        if ask_price > 0.0 and bid_price > 0.0 and np.isfinite(ask_price) and np.isfinite(bid_price):
            self.spread_sum += float(ask_price - bid_price)
            self.spread_samples += 1

    def get_final_result(self) -> dict[str, float]:
        avg_abs_position = self.position_abs_sum / max(self.position_samples, 1)
        avg_position = self.position_sum / max(self.position_samples, 1)
        avg_spread = self.spread_sum / max(self.spread_samples, 1)
        pnl = float(self.value)
        order_drop_rate = self.order_drops / max(self.order_attempts, 1)
        avg_order_latency_events = self.order_latency_events_sum / max(self.order_latency_samples, 1)
        avg_order_latency_ms = 1000.0 * self.order_latency_seconds_sum / max(self.order_latency_samples, 1)
        return {
            "pnl": pnl,
            "nd_pnl": pnl / max(avg_spread, 1e-7),
            "pnl_map": pnl / max(avg_abs_position, 1e-7),
            "profit_ratio": pnl / max(self.volume, 1e-7),
            "avg_position": avg_position,
            "avg_abs_position": avg_abs_position,
            "avg_spread": avg_spread,
            "volume": float(self.volume),
            "episode_reward": float(self.episode_reward),
            "reward_dampened_pnl": float(self.reward_dampened_pnl),
            "reward_trading_pnl": float(self.reward_trading_pnl),
            "reward_inventory_penalty": float(self.reward_inventory_penalty),
            "transaction_cost": float(self.transaction_cost_total),
            "order_attempts": float(self.order_attempts),
            "order_acceptances": float(self.order_acceptances),
            "order_drops": float(self.order_drops),
            "order_drop_rate": float(order_drop_rate),
            "avg_order_latency_events": float(avg_order_latency_events),
            "avg_order_latency_ms": float(avg_order_latency_ms),
            "configured_observation_latency_events": float(self.config.latency),
            "configured_order_latency_events": float(self.config.order_latency_events),
            "configured_order_latency_ms": float(self.config.order_latency_ms),
            "configured_inference_latency_ms": float(self.config.inference_latency_ms),
            "configured_order_loss_probability": float(self.config.order_loss_probability),
        }

    def _action_to_quotes(self, row, action) -> tuple[float, float]:
        raise NotImplementedError


class LOBDiscreteMarketMakingEnv(BaseLOBMarketMakingEnv):
    def __init__(self, config: LOBEnvConfig):
        super().__init__(config)
        self.action_space = spaces.Discrete(8)

    def _action_to_quotes(self, row, action) -> tuple[float, float]:
        ask_1 = float(row["ask_price_1"])
        bid_1 = float(row["bid_price_1"])
        tick = self.config.tick_size
        action_id = int(action)

        if action_id == 0:
            ask_price, bid_price = ask_1, bid_1
        elif action_id == 1:
            ask_price, bid_price = ask_1, bid_1 - tick
        elif action_id == 2:
            ask_price, bid_price = ask_1 + tick, bid_1
        elif action_id == 3:
            ask_price, bid_price = ask_1 + tick, bid_1 - tick
        elif action_id == 4:
            ask_price, bid_price = ask_1, bid_1 - 2 * tick
        elif action_id == 5:
            ask_price, bid_price = ask_1 + 2 * tick, bid_1
        elif action_id == 6:
            ask_price, bid_price = ask_1 + 2 * tick, bid_1 - 2 * tick
        else:
            if self.inventory > 0:
                ask_price, bid_price = tick, 0.0
            elif self.inventory < 0:
                ask_price, bid_price = 0.0, np.inf
            else:
                ask_price, bid_price = 0.0, 0.0

        if ask_price > 0.0 and bid_price > 0.0 and np.isfinite(ask_price) and np.isfinite(bid_price):
            ask_price, bid_price = self._legal_prices(ask_price, bid_price)
        return ask_price, bid_price


class LOBContinuousMarketMakingEnv(BaseLOBMarketMakingEnv):
    def __init__(self, config: LOBEnvConfig):
        super().__init__(config)
        self.action_space = spaces.Box(
            low=np.asarray([0.0, 0.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _action_to_quotes(self, row, action) -> tuple[float, float]:
        action = np.asarray(action, dtype=np.float32)
        mid_price = float(row["mid_price"])
        clipped = np.clip(action, 0.0, 1.0)
        delta_price = float(clipped[0]) * self.config.max_bias
        spread = float(clipped[1]) * self.config.max_spread
        reservation = mid_price - float(np.sign(self.inventory)) * delta_price
        ask_price = reservation + spread / 2.0
        bid_price = reservation - spread / 2.0
        return self._legal_prices(ask_price, bid_price)
