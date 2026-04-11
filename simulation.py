"""Core stochastic simulation functions for exchange-rate analysis.

This module implements the random walk model required by the course project:

    S(t+1) = S(t) + mu + sigma * Z(t)

Where:
    S(t)   = current exchange rate
    mu     = average daily change
    sigma  = volatility
    Z(t)   = random draw from the standard normal distribution N(0, 1)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def random_walk_path(
    S0: float,
    mu: float,
    sigma: float,
    days: int,
    Z_values: Iterable[float] | None = None,
) -> np.ndarray:
    """Generate one exchange-rate path using the required random walk equation."""
    if days < 1:
        raise ValueError("days must be at least 1")

    if Z_values is None:
        Z_values = np.random.normal(0, 1, days)

    shocks = np.asarray(list(Z_values), dtype=float)
    if shocks.size != days:
        raise ValueError("Z_values must contain exactly 'days' normal random values")

    path = np.empty(days + 1, dtype=float)
    path[0] = float(S0)

    for day in range(days):
        path[day + 1] = path[day] + mu + sigma * shocks[day]
        path[day + 1] = max(path[day + 1], 0.01)

    return path


def monte_carlo_simulation(
    S0: float,
    mu: float,
    sigma: float,
    days: int,
    n_simulations: int,
) -> np.ndarray:
    """Generate many possible future paths with Monte Carlo simulation."""
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1")

    random_draws = np.random.normal(0, 1, size=(n_simulations, days))
    paths = [
        random_walk_path(S0=S0, mu=mu, sigma=sigma, days=days, Z_values=draws)
        for draws in random_draws
    ]
    return np.vstack(paths)


def strategy_buy_hold(prices: np.ndarray, initial_capital: float) -> float:
    """Buy foreign currency on the first day and sell it on the last day."""
    prices = np.asarray(prices, dtype=float)
    if prices.size < 2:
        raise ValueError("prices must contain at least two observations")

    units_bought = initial_capital / prices[0]
    final_capital = units_bought * prices[-1]
    return float(final_capital - initial_capital)


def strategy_threshold(
    prices: np.ndarray,
    initial_capital: float,
    buy_threshold: float,
    sell_threshold: float,
) -> float:
    """Trade when the rate drops enough to buy, then rises enough to sell.

    A buy signal happens when today's rate falls by the buy_threshold
    percentage relative to yesterday's rate.
    A sell signal happens when the held rate increases by sell_threshold
    relative to the buy price.
    """

    prices = np.asarray(prices, dtype=float)
    if prices.size < 2:
        raise ValueError("prices must contain at least two observations")

    cash = float(initial_capital)
    holdings = 0.0
    buy_price = None

    for day in range(1, prices.size):
        current_price = prices[day]
        previous_price = prices[day - 1]
        daily_change = (current_price - previous_price) / previous_price

        if holdings == 0.0 and daily_change <= buy_threshold:
            holdings = cash / current_price
            cash = 0.0
            buy_price = current_price
        elif holdings > 0.0 and buy_price is not None:
            gain_from_entry = (current_price - buy_price) / buy_price
            if gain_from_entry >= sell_threshold:
                cash = holdings * current_price
                holdings = 0.0
                buy_price = None

    if holdings > 0.0:
        cash = holdings * prices[-1]

    return float(cash - initial_capital)


def strategy_trend_following(
    prices: np.ndarray,
    initial_capital: float,
    lookback_days: int,
) -> float:
    """Buy after an upward trend and sell after a downward trend."""

    prices = np.asarray(prices, dtype=float)
    if prices.size < lookback_days + 1:
        raise ValueError("prices must be longer than lookback_days")
    if lookback_days < 1:
        raise ValueError("lookback_days must be at least 1")

    cash = float(initial_capital)
    holdings = 0.0

    for day in range(lookback_days, prices.size):
        recent_window = prices[day - lookback_days : day + 1]
        recent_diffs = np.diff(recent_window)

        if holdings == 0.0 and np.all(recent_diffs > 0):
            holdings = cash / prices[day]
            cash = 0.0
        elif holdings > 0.0 and np.all(recent_diffs < 0):
            cash = holdings * prices[day]
            holdings = 0.0

    if holdings > 0.0:
        cash = holdings * prices[-1]

    return float(cash - initial_capital)


def calculate_metrics(profits: Iterable[float]) -> tuple[float, float, float]:
    """Return average return, risk, and total profit for a profit series."""
    profits_array = np.asarray(list(profits), dtype=float)
    if profits_array.size == 0:
        raise ValueError("profits must not be empty")

    mean_return = float(np.mean(profits_array))
    risk_std = float(np.std(profits_array, ddof=1)) if profits_array.size > 1 else 0.0
    total_profit = float(np.sum(profits_array))
    return mean_return, risk_std, total_profit
