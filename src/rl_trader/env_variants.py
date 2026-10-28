"""Variante del entorno con comisión configurable (extensión; el entorno base no se toca)."""
import numpy as np

from rl_trader.env import SimpleTradingEnv

BASE_COST_RATE = 0.001


class CostTradingEnv(SimpleTradingEnv):
    """Igual que SimpleTradingEnv, con la comisión (en efectivo y en recompensa) como parámetro."""

    def __init__(self, prices: np.ndarray, cost_rate: float = BASE_COST_RATE):
        super().__init__(prices)
        self.cost_rate = cost_rate

    def step(self, action):
        prev_price = self.prices[self.current_step]
        self.current_step += 1
        curr_price = self.prices[self.current_step]
        reward = 0.0
        cost_rate = self.cost_rate

        if action == 2 and self.position == 0:
            commission = self.cash * cost_rate
            self.shares = (self.cash - commission) / prev_price
            self.cash = 0.0
            self.position = 1
            reward -= cost_rate
        elif action == 0 and self.position == 1:
            sale_value = self.shares * prev_price
            self.cash = sale_value * (1 - cost_rate)
            self.shares = 0.0
            self.position = 0
            reward -= cost_rate

        if self.position == 1:
            reward += (curr_price - prev_price) / prev_price
            self.balance = self.cash + self.shares * curr_price
        else:
            self.balance = self.cash

        terminated = self.current_step >= self.max_steps
        return self._get_obs(), reward, terminated, False, {"balance": self.balance}


def make_env(prices: np.ndarray, cost_rate: float = BASE_COST_RATE) -> SimpleTradingEnv:
    """Entorno base si la comisión es la del enunciado; si no, la variante."""
    if cost_rate == BASE_COST_RATE:
        return SimpleTradingEnv(prices)
    return CostTradingEnv(prices, cost_rate)
