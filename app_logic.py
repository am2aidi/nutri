"""Shared data-processing and strategy helpers used by the Flask app."""

from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd

from simulation import (
    calculate_metrics,
    strategy_buy_hold,
    strategy_threshold,
    strategy_trend_following,
)


DEFAULTS = {
    "input_mode": "manual",
    "starting_rate": 1200.0,
    "mu": 5.0,
    "sigma": 10.0,
    "days": 30,
    "n_simulations": 1000,
    "initial_capital": 1_000_000.0,
    "buy_threshold": -0.01,
    "sell_threshold": 0.015,
    "trend_lookback": 3,
}


STRATEGY_LABELS = {
    "Strategy A": "Buy & Hold",
    "Strategy B": "Threshold",
    "Strategy C": "Trend-following",
}


def validate_csv(dataframe: pd.DataFrame) -> tuple[bool, str]:
    if "Date" not in dataframe.columns:
        return False, "CSV must contain a Date column."

    numeric_columns = [
        column
        for column in dataframe.columns
        if column != "Date" and pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    if numeric_columns:
        return True, ""

    candidate_columns = [column for column in ["Close", "High", "Low", "Open"] if column in dataframe.columns]
    if candidate_columns:
        return True, ""

    return False, "CSV must contain a Date column and at least one numeric exchange-rate column."


def get_rate_columns(dataframe: pd.DataFrame) -> list[str]:
    preferred_columns = [column for column in ["Close", "High", "Low", "Open"] if column in dataframe.columns]
    other_numeric_columns = [
        column
        for column in dataframe.columns
        if column not in {"Date", "currency_pair"} | set(preferred_columns)
        and pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    return preferred_columns + other_numeric_columns


def get_currency_pair_options(dataframe: pd.DataFrame) -> list[str]:
    if "currency_pair" not in dataframe.columns:
        return []
    return sorted(dataframe["currency_pair"].dropna().astype(str).unique().tolist())


def choose_default_pair(dataframe: pd.DataFrame) -> str | None:
    pair_options = get_currency_pair_options(dataframe)
    return pair_options[0] if pair_options else None


def choose_default_rate_column(dataframe: pd.DataFrame) -> str:
    rate_columns = get_rate_columns(dataframe)
    if not rate_columns:
        raise ValueError("No numeric rate column was found in the uploaded file.")
    return rate_columns[0]


def extract_history_frame(
    dataframe: pd.DataFrame,
    rate_column: str,
    selected_pair: str | None = None,
) -> pd.DataFrame:
    working_df = dataframe.copy()
    working_df["Date"] = pd.to_datetime(working_df["Date"], errors="coerce")
    working_df = working_df.dropna(subset=["Date"])

    if selected_pair and "currency_pair" in working_df.columns:
        working_df = working_df[working_df["currency_pair"].astype(str) == selected_pair]

    if working_df.empty:
        raise ValueError("No valid rows were found for the selected currency pair.")

    working_df = working_df.sort_values("Date")
    history_df = working_df[["Date", rate_column]].copy()
    history_df["Rate"] = pd.to_numeric(history_df[rate_column], errors="coerce")
    history_df = history_df.dropna(subset=["Rate"])
    return history_df[["Date", "Rate"]]


def derive_parameters_from_history(rate_series: pd.Series) -> tuple[float, float, float]:
    clean_series = pd.to_numeric(rate_series, errors="coerce").dropna()
    if clean_series.size < 2:
        raise ValueError("Uploaded rate column must have at least two valid rows.")

    daily_changes = clean_series.diff().dropna()
    mu = float(daily_changes.mean())
    sigma = float(daily_changes.std(ddof=1)) if daily_changes.size > 1 else 0.0
    starting_rate = float(clean_series.iloc[-1])
    return starting_rate, mu, sigma


def build_weekday_market_summary(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame()

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    summary_df = history_df.copy()
    summary_df["Weekday"] = summary_df["Date"].dt.day_name()
    summary_df["Daily Change %"] = summary_df["Rate"].pct_change() * 100

    grouped = (
        summary_df.groupby("Weekday", observed=False)
        .agg(
            Average_Rate=("Rate", "mean"),
            Average_Change=("Daily Change %", "mean"),
            Observations=("Rate", "count"),
        )
        .reset_index()
    )

    grouped["Weekday"] = pd.Categorical(grouped["Weekday"], categories=weekday_order, ordered=True)
    grouped = grouped.sort_values("Weekday").dropna(subset=["Average_Rate"])
    grouped["Market Mood"] = np.where(
        grouped["Average_Change"].fillna(0) >= 0,
        "Usually stronger",
        "Usually weaker",
    )
    return grouped


def build_reference_levels(
    best_strategy_code: str,
    paths: np.ndarray,
    starting_rate: float,
    buy_threshold: float,
    sell_threshold: float,
) -> tuple[float, float, str]:
    future_values = paths[:, 1:].reshape(-1)
    mean_terminal_rate = float(np.mean(paths[:, -1]))

    if best_strategy_code == "Strategy A":
        return starting_rate, mean_terminal_rate, "Buy at the current level and hold until the planned exit period."
    if best_strategy_code == "Strategy B":
        buy_level = starting_rate * (1 + buy_threshold)
        sell_level = buy_level * (1 + sell_threshold)
        return buy_level, sell_level, "Wait for a dip before entering, then lock profit after a rebound."

    buy_level = float(np.percentile(future_values, 45))
    sell_level = float(np.percentile(future_values, 65))
    return buy_level, sell_level, "Enter after upward momentum appears and exit once the upward phase begins to fade."


def build_strategy_rules_table(simulated_days: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy": "A - Buy & Hold",
                "when_to_buy": "Buy immediately at the starting exchange rate.",
                "when_to_sell": f"Sell after {simulated_days} simulated days.",
                "meaning": "Best when you expect a steady rise over the whole period.",
            },
            {
                "strategy": "B - Threshold",
                "when_to_buy": "Buy only after the rate drops by 1% or more.",
                "when_to_sell": "Sell after the bought rate rises by 1.5% or more.",
                "meaning": "Best when prices move up and down and you want disciplined entry and exit points.",
            },
            {
                "strategy": "C - Trend-following",
                "when_to_buy": "Buy after 3 straight days of upward movement.",
                "when_to_sell": "Sell after 3 straight days of downward movement.",
                "meaning": "Best when the market shows short-term momentum trends.",
            },
        ]
    )


def run_all_strategies(
    paths: np.ndarray,
    initial_capital: float,
    buy_threshold: float,
    sell_threshold: float,
    trend_lookback: int,
) -> pd.DataFrame:
    results = []

    strategy_functions = {
        "Strategy A": lambda prices: strategy_buy_hold(prices, initial_capital),
        "Strategy B": lambda prices: strategy_threshold(
            prices, initial_capital, buy_threshold, sell_threshold
        ),
        "Strategy C": lambda prices: strategy_trend_following(
            prices, initial_capital, trend_lookback
        ),
    }

    for code, strategy_function in strategy_functions.items():
        profits = np.array([strategy_function(path) for path in paths], dtype=float)
        average_return, risk_std, total_profit = calculate_metrics(profits)
        risk_adjusted_score = average_return / (risk_std + 1e-9)

        results.append(
            {
                "strategy_code": code,
                "name": STRATEGY_LABELS[code],
                "average_return": average_return,
                "risk_std_dev": risk_std,
                "total_profit": total_profit,
                "risk_adjusted_score": risk_adjusted_score,
            }
        )

    return pd.DataFrame(results)


def format_rwf(value: float) -> str:
    return f"{value:,.2f} RWF"


def build_downloadable_results(summary_df: pd.DataFrame) -> str:
    output = StringIO()
    summary_df.to_csv(output, index=False)
    return output.getvalue()


def get_best_strategy(summary_df: pd.DataFrame) -> pd.Series:
    ranked = summary_df.sort_values(
        by=["risk_adjusted_score", "average_return"],
        ascending=[False, False],
    )
    return ranked.iloc[0]


def explain_best_strategy(best_strategy: pd.Series) -> str:
    if best_strategy["strategy_code"] == "Strategy A":
        return "The model expects a mostly steady increase, so buying once and holding gives the best result."
    if best_strategy["strategy_code"] == "Strategy B":
        return "The model favors waiting for a cheaper entry and taking profit after a clear rebound."
    return "The model favors following short upward and downward trends instead of trading immediately."
