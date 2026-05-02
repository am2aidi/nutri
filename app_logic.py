"""Shared data-processing and strategy helpers used by the Flask app."""

from __future__ import annotations

from io import BytesIO, StringIO

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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
                "meaning": "Good when you expect the price to keep moving up in a steady way.",
            },
            {
                "strategy": "B - Threshold",
                "when_to_buy": "Buy only after the rate drops by 1% or more.",
                "when_to_sell": "Sell after the bought rate rises by 1.5% or more.",
                "meaning": "Good when price goes down first and then comes back up.",
            },
            {
                "strategy": "C - Trend-following",
                "when_to_buy": "Buy after 3 straight days of upward movement.",
                "when_to_sell": "Sell after 3 straight days of downward movement.",
                "meaning": "Good when the market shows a short clear trend.",
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
        return "We picked Buy and Hold because the model expects the rate to move up in a more steady way."
    if best_strategy["strategy_code"] == "Strategy B":
        return "We picked Threshold because the model likes waiting for a dip and then selling after a clear bounce."
    return "We picked Trend-following because the model sees short trends that can be followed step by step."


def parse_extra_currency_rates(
    raw_text: str,
    local_currency: str,
    target_currency: str,
    target_rate: float,
) -> dict[str, float]:
    rates = {
        local_currency: 1.0,
        target_currency: float(target_rate),
    }

    for line in raw_text.splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue

        if "=" in clean_line:
            code_part, value_part = clean_line.split("=", 1)
        elif ":" in clean_line:
            code_part, value_part = clean_line.split(":", 1)
        else:
            parts = clean_line.split()
            if len(parts) != 2:
                raise ValueError(
                    "Write extra rates like EUR=1320. Example: 1 EUR = 1320 in your local money."
                )
            code_part, value_part = parts

        currency_code = code_part.strip().upper()
        if not currency_code:
            raise ValueError("Each extra rate must have a currency code, like EUR or GBP.")

        try:
            rate_value = float(value_part.strip())
        except ValueError as exc:
            raise ValueError(
                f"Rate for {currency_code} must be a number. Example: EUR=1320"
            ) from exc

        if rate_value <= 0:
            raise ValueError(f"Rate for {currency_code} must be greater than zero.")

        rates[currency_code[:12]] = rate_value

    return rates


def convert_currency_amount(
    amount: float,
    from_currency: str,
    to_currency: str,
    rate_map: dict[str, float],
) -> float:
    if from_currency not in rate_map:
        raise ValueError(f"'{from_currency}' is missing from the calculator rates.")
    if to_currency not in rate_map:
        raise ValueError(f"'{to_currency}' is missing from the calculator rates.")

    local_amount = float(amount) * float(rate_map[from_currency])
    return local_amount / float(rate_map[to_currency])


def build_pdf_report(report: dict) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]
    body_style.fontName = "Helvetica"
    body_style.leading = 14
    small_style = ParagraphStyle(
        "SmallBody",
        parent=body_style,
        fontSize=9,
        leading=12,
    )

    story = [
        Paragraph("Forex Simulation Report", title_style),
        Spacer(1, 8),
        Paragraph(
            f"Best strategy: {report['best_strategy_code']} - {report['best_strategy_name']}",
            heading_style,
        ),
        Paragraph(report["best_explanation"], body_style),
        Spacer(1, 8),
        Paragraph(report["best_reason_short"], body_style),
        Spacer(1, 12),
        Paragraph("Simple Summary", heading_style),
        Paragraph(report["simulation_story"], body_style),
        Spacer(1, 12),
        Paragraph("Main Inputs", heading_style),
    ]

    input_rows = [
        ["Input", "Value"],
        ["Source", report["source_label"]],
        ["Opening rate", report["starting_rate"]],
        ["Local currency", report["local_currency"]],
        ["Target currency", report["target_currency"]],
        ["mu", report["mu"]],
        ["sigma", report["sigma"]],
        ["Days", str(report["days"])],
        ["Monte Carlo runs", str(report["n_simulations"])],
        ["Buy level", report["suggested_buy"]],
        ["Sell level", report["suggested_sell"]],
    ]

    input_table = Table(input_rows, colWidths=[55 * mm, 110 * mm], repeatRows=1)
    input_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#172033")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8d1e1")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.extend([input_table, Spacer(1, 12), Paragraph("Money View", heading_style)])

    for card in report["scenario_cards"]:
        story.append(Paragraph(f"<b>{card['label']}:</b> {card['value']}", body_style))
        story.append(Paragraph(card["detail"], small_style))
        story.append(Spacer(1, 4))

    story.extend(
        [
            Spacer(1, 10),
            Paragraph("Calculator Result", heading_style),
            Paragraph(
                f"{report['calculator_summary']['from_amount']} becomes "
                f"{report['calculator_summary']['to_amount']}.",
                body_style,
            ),
            Paragraph(report["calculator_summary"]["rate_note"], small_style),
            Spacer(1, 12),
            Paragraph("Strategy Comparison", heading_style),
        ]
    )

    strategy_rows = [["Strategy", "Average return", "Risk", "Total profit", "Score"]]
    for row in report["strategy_rows"]:
        strategy_rows.append(
            [
                row["name"],
                row["average_return"],
                row["risk_std_dev"],
                row["total_profit"],
                row["risk_adjusted_score"],
            ]
        )

    strategy_table = Table(
        strategy_rows,
        colWidths=[34 * mm, 34 * mm, 34 * mm, 34 * mm, 22 * mm],
        repeatRows=1,
    )
    strategy_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fde7d9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d9d3c5")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.extend([strategy_table, Spacer(1, 12), Paragraph("End Rate Summary", heading_style)])

    for key, label in [
        ("minimum", "Minimum"),
        ("median", "Median"),
        ("maximum", "Maximum"),
        ("above_start_probability", "Finished above start"),
    ]:
        story.append(Paragraph(f"<b>{label}:</b> {report['terminal_summary'][key]}", body_style))

    if report["terminal_histogram_rows"]:
        story.extend([Spacer(1, 10), Paragraph("Histogram Details", heading_style)])
        histogram_rows = [["Range", "Paths"]]
        for row in report["terminal_histogram_rows"]:
            histogram_rows.append([row["label"], str(row["count"])])

        histogram_table = Table(histogram_rows, colWidths=[110 * mm, 40 * mm], repeatRows=1)
        histogram_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3f3f5")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c7dede")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        story.append(histogram_table)

    if report["weekday_rows"]:
        story.extend([Spacer(1, 12), Paragraph("Weekday Summary", heading_style)])
        weekday_rows = [["Day", "Average rate", "Average change", "Count", "Mood"]]
        for row in report["weekday_rows"]:
            weekday_rows.append(
                [
                    row["weekday"],
                    row["average_rate"],
                    row["average_change"],
                    str(row["observations"]),
                    row["market_mood"],
                ]
            )

        weekday_table = Table(
            weekday_rows,
            colWidths=[30 * mm, 40 * mm, 35 * mm, 18 * mm, 45 * mm],
            repeatRows=1,
        )
        weekday_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cdd5ee")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        story.append(weekday_table)

    story.extend([Spacer(1, 12), Paragraph("Trading Rules", heading_style)])
    rule_rows = [["Strategy", "Buy", "Sell", "Meaning"]]
    for row in report["strategy_rules_rows"]:
        rule_rows.append(
            [
                row["strategy"],
                row["when_to_buy"],
                row["when_to_sell"],
                row["meaning"],
            ]
        )

    rule_table = Table(
        rule_rows,
        colWidths=[30 * mm, 46 * mm, 46 * mm, 46 * mm],
        repeatRows=1,
    )
    rule_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ecfdf5")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbe8d7")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(rule_table)

    document.build(story)
    return buffer.getvalue()
