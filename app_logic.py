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
    "input_mode": "upload",
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
    "Strategy A": "Buy and wait",
    "Strategy B": "Buy after a drop",
    "Strategy C": "Follow the trend",
}


def _find_column_name(columns, candidates: list[str]) -> str | None:
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        match = normalized.get(candidate.lower())
        if match:
            return match
    return None


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_uploaded_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    working_df = dataframe.copy()
    working_df.columns = [str(column).strip() for column in working_df.columns]

    date_column = _find_column_name(
        working_df.columns,
        ["Date", "Day", "Timestamp", "Time", "Recorded At"],
    )
    if date_column and date_column != "Date":
        working_df = working_df.rename(columns={date_column: "Date"})

    pair_column = _find_column_name(
        working_df.columns,
        ["currency_pair", "Currency Pair", "Pair", "CurrencyPair"],
    )
    if pair_column and pair_column != "currency_pair":
        working_df = working_df.rename(columns={pair_column: "currency_pair"})

    for column in working_df.columns:
        if column in {"Date", "currency_pair"}:
            continue
        working_df[column] = _coerce_numeric_series(working_df[column])

    if "currency_pair" in working_df.columns:
        working_df["currency_pair"] = working_df["currency_pair"].astype(str).str.strip()

    return working_df


def _is_usable_numeric_column(dataframe: pd.DataFrame, column: str) -> bool:
    return pd.api.types.is_numeric_dtype(dataframe[column]) and int(dataframe[column].notna().sum()) >= 2


def validate_csv(dataframe: pd.DataFrame) -> tuple[bool, str]:
    cleaned_df = clean_uploaded_dataframe(dataframe)

    if "Date" not in cleaned_df.columns:
        return False, "Your CSV needs a Date column."

    numeric_columns = [
        column
        for column in cleaned_df.columns
        if column != "Date" and column != "currency_pair" and _is_usable_numeric_column(cleaned_df, column)
    ]
    if numeric_columns:
        return True, ""

    candidate_columns = [
        column
        for column in ["Close", "High", "Low", "Open"]
        if column in cleaned_df.columns and _is_usable_numeric_column(cleaned_df, column)
    ]
    if candidate_columns:
        return True, ""

    return False, "Your CSV needs a Date column and one number column with exchange rates."


def get_rate_columns(dataframe: pd.DataFrame) -> list[str]:
    dataframe = clean_uploaded_dataframe(dataframe)
    preferred_columns = [
        column
        for column in ["Close", "High", "Low", "Open"]
        if column in dataframe.columns and _is_usable_numeric_column(dataframe, column)
    ]
    other_numeric_columns = [
        column
        for column in dataframe.columns
        if column not in {"Date", "currency_pair"} | set(preferred_columns)
        and _is_usable_numeric_column(dataframe, column)
    ]
    return preferred_columns + other_numeric_columns


def get_currency_pair_options(dataframe: pd.DataFrame) -> list[str]:
    dataframe = clean_uploaded_dataframe(dataframe)
    if "currency_pair" not in dataframe.columns:
        return []
    values = dataframe["currency_pair"].replace({"nan": np.nan}).dropna().astype(str)
    return sorted(value for value in values.unique().tolist() if value)


def choose_default_pair(dataframe: pd.DataFrame) -> str | None:
    pair_options = get_currency_pair_options(dataframe)
    return pair_options[0] if pair_options else None


def choose_default_rate_column(dataframe: pd.DataFrame) -> str:
    rate_columns = get_rate_columns(dataframe)
    if not rate_columns:
        raise ValueError("We could not find a number column with exchange rates in the uploaded file.")
    return rate_columns[0]


def extract_history_frame(
    dataframe: pd.DataFrame,
    rate_column: str,
    selected_pair: str | None = None,
) -> pd.DataFrame:
    working_df = clean_uploaded_dataframe(dataframe)
    working_df["Date"] = pd.to_datetime(working_df["Date"], errors="coerce")
    working_df = working_df.dropna(subset=["Date"])

    if selected_pair and "currency_pair" in working_df.columns:
        working_df = working_df[working_df["currency_pair"].astype(str) == selected_pair]

    if working_df.empty:
        raise ValueError("We could not find valid rows for the currency pair you selected.")

    working_df = working_df.sort_values("Date")
    history_df = working_df[["Date", rate_column]].copy()
    history_df["Rate"] = pd.to_numeric(history_df[rate_column], errors="coerce")
    history_df = history_df.dropna(subset=["Rate"])
    history_df = history_df.drop_duplicates(subset=["Date"], keep="last")
    return history_df[["Date", "Rate"]]


def prepare_uploaded_history(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cleaned_df = clean_uploaded_dataframe(dataframe)
    is_valid, message = validate_csv(cleaned_df)
    if not is_valid:
        raise ValueError(message)

    selected_pair = choose_default_pair(cleaned_df)
    rate_column = choose_default_rate_column(cleaned_df)

    dated_df = cleaned_df.copy()
    dated_df["Date"] = pd.to_datetime(dated_df["Date"], errors="coerce")
    valid_date_df = dated_df.dropna(subset=["Date"])

    pair_filtered_df = valid_date_df
    if selected_pair and "currency_pair" in pair_filtered_df.columns:
        pair_filtered_df = pair_filtered_df[pair_filtered_df["currency_pair"].astype(str) == selected_pair]

    if pair_filtered_df.empty:
        raise ValueError("We could not find valid rows for the currency pair you selected.")

    rate_df = pair_filtered_df[["Date", rate_column]].copy()
    rate_df["Rate"] = pd.to_numeric(rate_df[rate_column], errors="coerce")
    valid_rate_df = rate_df.dropna(subset=["Rate"])

    if valid_rate_df.empty:
        raise ValueError("We could not find valid rate values in the uploaded file.")

    duplicate_date_rows = int(valid_rate_df.duplicated(subset=["Date"], keep="last").sum())
    history_df = (
        valid_rate_df.drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)[["Date", "Rate"]]
    )

    metadata = {
        "selected_pair": selected_pair,
        "rate_column": rate_column,
        "source_name": selected_pair or rate_column,
        "original_rows": int(len(cleaned_df)),
        "valid_date_rows": int(len(valid_date_df)),
        "pair_rows": int(len(pair_filtered_df)),
        "valid_rate_rows": int(len(valid_rate_df)),
        "kept_rows": int(len(history_df)),
        "invalid_date_rows": int(len(cleaned_df) - len(valid_date_df)),
        "invalid_rate_rows": int(len(pair_filtered_df) - len(valid_rate_df)),
        "duplicate_date_rows": duplicate_date_rows,
    }
    return history_df, metadata


def derive_parameters_from_history(rate_series: pd.Series) -> tuple[float, float, float]:
    clean_series = pd.to_numeric(rate_series, errors="coerce").dropna()
    if clean_series.size < 2:
        raise ValueError("Please give at least two valid rate values.")

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
        "Often up",
        "Often down",
    )
    return grouped


def build_history_analysis(history_df: pd.DataFrame) -> dict:
    if history_df.empty:
        return {}

    rates = pd.to_numeric(history_df["Rate"], errors="coerce").dropna()
    if rates.empty:
        return {}

    daily_changes = rates.diff().dropna()
    first_rate = float(rates.iloc[0])
    latest_rate = float(rates.iloc[-1])
    change_value = latest_rate - first_rate
    change_percent = (change_value / first_rate * 100) if first_rate else 0.0

    if change_value > 0:
        trend_label = "Overall up"
    elif change_value < 0:
        trend_label = "Overall down"
    else:
        trend_label = "Mostly flat"

    return {
        "first_rate": first_rate,
        "latest_rate": latest_rate,
        "lowest_rate": float(rates.min()),
        "highest_rate": float(rates.max()),
        "average_rate": float(rates.mean()),
        "overall_change_value": float(change_value),
        "overall_change_percent": float(change_percent),
        "average_daily_change": float(daily_changes.mean()) if not daily_changes.empty else 0.0,
        "market_swing": float(daily_changes.std(ddof=1)) if len(daily_changes) > 1 else 0.0,
        "trend_label": trend_label,
        "date_start": history_df["Date"].min(),
        "date_end": history_df["Date"].max(),
    }


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
        return starting_rate, mean_terminal_rate, "Buy now and wait until the end of the test."
    if best_strategy_code == "Strategy B":
        buy_level = starting_rate * (1 + buy_threshold)
        sell_level = buy_level * (1 + sell_threshold)
        return buy_level, sell_level, "Wait for the price to drop, then sell after it goes back up."

    buy_level = float(np.percentile(future_values, 45))
    sell_level = float(np.percentile(future_values, 65))
    return buy_level, sell_level, "Buy when the price keeps rising and sell when that rise starts to fade."


def build_strategy_rules_table(simulated_days: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy": "A - Buy and wait",
                "when_to_buy": "Buy now at the starting rate.",
                "when_to_sell": f"Sell after {simulated_days} test days.",
                "meaning": "Good when you think the price will slowly keep going up.",
            },
            {
                "strategy": "B - Buy after a drop",
                "when_to_buy": "Buy only after the price drops by 1% or more.",
                "when_to_sell": "Sell after the bought price rises by 1.5% or more.",
                "meaning": "Good when price falls first and then comes back up.",
            },
            {
                "strategy": "C - Follow the trend",
                "when_to_buy": "Buy after 3 days in a row of price increases.",
                "when_to_sell": "Sell after 3 days in a row of price drops.",
                "meaning": "Good when the market shows a short clear direction.",
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
        return "We picked buy and wait because the price looks more likely to rise in a steady way."
    if best_strategy["strategy_code"] == "Strategy B":
        return "We picked buy after a drop because waiting for a lower price looks safer here."
    return "We picked follow the trend because the price seems to move in short clear directions."


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
                    "Write extra rates like EUR=1320. Example: 1 EUR = 1320 in your local currency."
                )
            code_part, value_part = parts

        currency_code = code_part.strip().upper()
        if not currency_code:
            raise ValueError("Each extra rate needs a currency code, like EUR or GBP.")

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
        raise ValueError(f"'{from_currency}' is not in your rate list.")
    if to_currency not in rate_map:
        raise ValueError(f"'{to_currency}' is not in your rate list.")

    local_amount = float(amount) * float(rate_map[from_currency])
    return local_amount / float(rate_map[to_currency])


def _rtf_escape(text: str) -> str:
    escaped = []
    for char in str(text):
        code = ord(char)
        if char in {"\\", "{", "}"}:
            escaped.append(f"\\{char}")
        elif char == "\n":
            escaped.append("\\line ")
        elif 32 <= code <= 126:
            escaped.append(char)
        else:
            escaped.append(f"\\u{code}?")
    return "".join(escaped)


def build_word_report(report: dict) -> bytes:
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0 Calibri;}}",
        r"\fs32\b Rate Test Report\b0\fs22\par",
        r"\par",
        rf"\b Best choice:\b0 {_rtf_escape(report.get('best_strategy_label', report['best_strategy_code']))} - {_rtf_escape(report['best_strategy_name'])}\par",
        rf"{_rtf_escape(report['best_explanation'])}\par",
        rf"{_rtf_escape(report['best_reason_short'])}\par",
        r"\par",
        r"\b Quick summary\b0\par",
        rf"{_rtf_escape(report['simulation_story'])}\par",
        rf"{_rtf_escape(report['learning_summary'])}\par",
        r"\par",
        r"\b Your inputs\b0\par",
        rf"Source: {_rtf_escape(report['source_label'])}\par",
        rf"Start rate: {_rtf_escape(report['starting_rate'])}\par",
        rf"Local currency: {_rtf_escape(report['local_currency'])}\par",
        rf"Target currency: {_rtf_escape(report['target_currency'])}\par",
        rf"Average daily change: {_rtf_escape(report['mu'])}\par",
        rf"Market swing: {_rtf_escape(report['sigma'])}\par",
        rf"Days: {_rtf_escape(report['days'])}\par",
        rf"Number of test runs: {_rtf_escape(report['n_simulations'])}\par",
        rf"Buy rate idea: {_rtf_escape(report['suggested_buy'])}\par",
        rf"Sell rate idea: {_rtf_escape(report['suggested_sell'])}\par",
        r"\par",
        r"\b Money view\b0\par",
    ]

    for card in report["scenario_cards"]:
        lines.append(rf"{_rtf_escape(card['label'])}: {_rtf_escape(card['value'])}\par")
        lines.append(rf"{_rtf_escape(card['detail'])}\par")

    lines.extend(
        [
            r"\par",
            r"\b Calculator result\b0\par",
            rf"{_rtf_escape(report['calculator_summary']['from_amount'])} becomes {_rtf_escape(report['calculator_summary']['to_amount'])}.\par",
            rf"{_rtf_escape(report['calculator_summary']['rate_note'])}\par",
            r"\par",
            r"\b All options\b0\par",
        ]
    )

    for row in report["strategy_rows"]:
        lines.append(
            rf"{_rtf_escape(row['strategy_code'])} - {_rtf_escape(row['name'])}: "
            rf"average gain {_rtf_escape(row['average_return'])}, "
            rf"risk {_rtf_escape(row['risk_std_dev'])}, "
            rf"total gain {_rtf_escape(row['total_profit'])}, "
            rf"balance score {_rtf_escape(row['risk_adjusted_score'])}.\par"
        )

    lines.extend(
        [
            r"\par",
            r"\b Ending rate summary\b0\par",
            rf"Lowest: {_rtf_escape(report['terminal_summary']['minimum'])}\par",
            rf"Middle: {_rtf_escape(report['terminal_summary']['median'])}\par",
            rf"Highest: {_rtf_escape(report['terminal_summary']['maximum'])}\par",
            rf"Chance it ends above today: {_rtf_escape(report['terminal_summary']['above_start_probability'])}\par",
        ]
    )

    if report["weekday_rows"]:
        lines.extend([r"\par", r"\b Daily pattern\b0\par"])
        for row in report["weekday_rows"]:
            lines.append(
                rf"{_rtf_escape(row['weekday'])}: average rate {_rtf_escape(row['average_rate'])}, "
                rf"average change {_rtf_escape(row['average_change'])}, "
                rf"count {_rtf_escape(row['observations'])}, "
                rf"pattern {_rtf_escape(row['market_mood'])}.\par"
            )

    lines.extend([r"\par", r"\b Buy and sell rules\b0\par"])
    for row in report["strategy_rules_rows"]:
        lines.append(
            rf"{_rtf_escape(row['strategy'])}: buy {_rtf_escape(row['when_to_buy'])} "
            rf"sell {_rtf_escape(row['when_to_sell'])} "
            rf"meaning {_rtf_escape(row['meaning'])}.\par"
        )

    lines.append("}")
    return "".join(lines).encode("utf-8")


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
        Paragraph("Rate Test Report", title_style),
        Spacer(1, 8),
        Paragraph(
            f"Best choice: {report.get('best_strategy_label', report['best_strategy_code'])} - {report['best_strategy_name']}",
            heading_style,
        ),
        Paragraph(report["best_explanation"], body_style),
        Spacer(1, 8),
        Paragraph(report["best_reason_short"], body_style),
        Spacer(1, 12),
        Paragraph("Quick summary", heading_style),
        Paragraph(report["simulation_story"], body_style),
        Paragraph(report["learning_summary"], body_style),
        Spacer(1, 12),
        Paragraph("Your inputs", heading_style),
    ]

    input_rows = [
        ["Input", "Value"],
        ["Source", report["source_label"]],
        ["Start rate", report["starting_rate"]],
        ["Local currency", report["local_currency"]],
        ["Target currency", report["target_currency"]],
        ["Average daily change", report["mu"]],
        ["Market swing", report["sigma"]],
        ["Days", str(report["days"])],
        ["Number of test runs", str(report["n_simulations"])],
        ["Buy rate idea", report["suggested_buy"]],
        ["Sell rate idea", report["suggested_sell"]],
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
    story.extend([input_table, Spacer(1, 12), Paragraph("Money view", heading_style)])

    for card in report["scenario_cards"]:
        story.append(Paragraph(f"<b>{card['label']}:</b> {card['value']}", body_style))
        story.append(Paragraph(card["detail"], small_style))
        story.append(Spacer(1, 4))

    story.extend(
        [
            Spacer(1, 10),
            Paragraph("Calculator result", heading_style),
            Paragraph(
                f"{report['calculator_summary']['from_amount']} becomes "
                f"{report['calculator_summary']['to_amount']}.",
                body_style,
            ),
            Paragraph(report["calculator_summary"]["rate_note"], small_style),
            Spacer(1, 12),
            Paragraph("All options", heading_style),
        ]
    )

    strategy_rows = [["Option", "Average gain", "Risk", "Total gain", "Balance score"]]
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
    story.extend([strategy_table, Spacer(1, 12), Paragraph("Ending rate summary", heading_style)])

    for key, label in [
        ("minimum", "Lowest"),
        ("median", "Middle"),
        ("maximum", "Highest"),
        ("above_start_probability", "Chance it ends above today"),
    ]:
        story.append(Paragraph(f"<b>{label}:</b> {report['terminal_summary'][key]}", body_style))

    if report["terminal_histogram_rows"]:
        story.extend([Spacer(1, 10), Paragraph("End result table", heading_style)])
        histogram_rows = [["Range", "Runs"]]
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
        story.extend([Spacer(1, 12), Paragraph("Daily pattern", heading_style)])
        weekday_rows = [["Day", "Average rate", "Average change", "Count", "Pattern"]]
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

    story.extend([Spacer(1, 12), Paragraph("Buy and sell rules", heading_style)])
    rule_rows = [["Option", "Buy", "Sell", "Meaning"]]
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
