"""Flask entrypoint for the Vercel deployment and browser UI."""

from __future__ import annotations

from base64 import b64encode

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from app_logic import (
    DEFAULTS,
    build_history_analysis,
    build_pdf_report,
    build_reference_levels,
    build_strategy_rules_table,
    build_weekday_market_summary,
    build_word_report,
    convert_currency_amount,
    derive_parameters_from_history,
    explain_best_strategy,
    get_best_strategy,
    parse_extra_currency_rates,
    prepare_uploaded_history,
    run_all_strategies,
)
from simulation import monte_carlo_simulation


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024


FIELD_LABELS = {
    "starting_rate": "Today's market rate",
    "mu": "Average daily change",
    "sigma": "Market swing",
    "days": "Days to test",
    "n_simulations": "Number of test runs",
    "initial_capital": "Money you start with",
    "buy_threshold_percent": "Buy drop percent",
    "sell_threshold_percent": "Sell gain percent",
    "trend_lookback": "Trend check days",
    "calculator_amount": "Amount",
}


def _friendly_field_name(name: str) -> str:
    return FIELD_LABELS.get(name, name.replace("_", " "))


def _parse_float(name: str, default: float) -> float:
    raw_value = request.form.get(name, str(default)).strip()
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{_friendly_field_name(name)} must be a number.") from exc


def _parse_required_float(name: str) -> float:
    raw_value = request.form.get(name, "").strip()
    if not raw_value:
        raise ValueError(f"Enter {_friendly_field_name(name).lower()}.")
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{_friendly_field_name(name)} must be a number.") from exc


def _parse_int(name: str, default: int) -> int:
    raw_value = request.form.get(name, str(default)).strip()
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{_friendly_field_name(name)} must be a whole number.") from exc


def _data_url(content: str | bytes, mime_type: str) -> str:
    payload = content.encode("utf-8") if isinstance(content, str) else content
    encoded = b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _format_money(value: float, currency_label: str) -> str:
    return f"{float(value):,.2f} {currency_label}"


def _format_rate(value: float, local_currency: str, target_currency: str) -> str:
    return f"{float(value):,.4f} {local_currency}/{target_currency}"


def _clean_currency_label(raw_value: str, fallback: str) -> str:
    label = raw_value.strip().upper()
    return label[:12] if label else fallback


def _display_strategy_code(code: str) -> str:
    return code.replace("Strategy", "Option")


def _build_chart_rows(summary_df: pd.DataFrame, metric_key: str, currency_label: str) -> list[dict]:
    ceiling = float(summary_df[metric_key].abs().max())
    if ceiling == 0:
        ceiling = 1.0

    rows = []
    for _, row in summary_df.iterrows():
        width = max(12, int(abs(float(row[metric_key])) / ceiling * 100))
        rows.append(
            {
                "name": row["name"],
                "value": _format_money(float(row[metric_key]), currency_label),
                "width": width,
            }
        )
    return rows


def _format_percent(value: float) -> str:
    return f"{float(value) * 100:,.1f}%"


def _format_signed_number(value: float, digits: int = 4) -> str:
    sign = "+" if float(value) > 0 else ""
    return f"{sign}{float(value):,.{digits}f}"


def _format_signed_percent(value: float) -> str:
    sign = "+" if float(value) > 0 else ""
    return f"{sign}{float(value):,.2f}%"


def _build_sample_path_rows(paths) -> list[dict]:
    sample_path_count = min(6, len(paths))
    sample_days = min(15, paths.shape[1])
    sample_df = pd.DataFrame(paths[:sample_path_count, :sample_days].T)
    sample_df.columns = [f"Path {index + 1}" for index in range(sample_path_count)]
    sample_df.insert(0, "Day", range(sample_days))

    rows = []
    for _, row in sample_df.iterrows():
        rows.append(
            {
                key: (int(value) if key == "Day" else f"{float(value):,.2f}")
                for key, value in row.items()
            }
        )
    return rows


def _build_histogram_rows(values, bins: int = 5) -> list[dict]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return []

    min_value = float(array.min())
    max_value = float(array.max())
    if min_value == max_value:
        max_value = min_value + 1.0

    edges = np.linspace(min_value, max_value, bins + 1)
    counts, _ = np.histogram(array, bins=edges)
    peak_count = int(counts.max()) if counts.size else 1
    peak_count = peak_count or 1
    total_count = int(array.size) or 1
    band_names = ["Very low", "Low", "Middle", "High", "Very high"]

    rows = []
    for index, count in enumerate(counts):
        count_value = int(count)
        rows.append(
            {
                "label": f"{edges[index]:,.1f} - {edges[index + 1]:,.1f}",
                "count": count_value,
                "height": max(16, int((count_value / peak_count) * 100)) if count_value > 0 else 10,
                "width": max(8, int((count_value / total_count) * 100)) if count_value > 0 else 4,
                "percent": f"{(count_value / total_count) * 100:.1f}%",
                "band_name": band_names[index] if index < len(band_names) else f"Band {index + 1}",
            }
        )
    return rows


def _build_line_points(
    values,
    width: int = 420,
    height: int = 200,
    pad_left: int = 52,
    pad_right: int = 16,
    pad_top: int = 18,
    pad_bottom: int = 28,
) -> str:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return ""
    if array.size == 1:
        return f"{width / 2:.1f},{height / 2:.1f}"

    x_values = np.linspace(pad_left, width - pad_right, array.size)
    min_value = float(array.min())
    max_value = float(array.max())
    if min_value == max_value:
        max_value = min_value + 1.0

    chart_height = height - pad_top - pad_bottom
    y_values = height - pad_bottom - ((array - min_value) / (max_value - min_value)) * chart_height
    return " ".join(f"{x_value:.1f},{y_value:.1f}" for x_value, y_value in zip(x_values, y_values))


def _build_chart_panel(
    values,
    local_currency: str,
    target_currency: str,
    start_label: str,
    middle_label: str,
    end_label: str,
    width: int = 420,
    height: int = 200,
) -> dict:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {}

    pad_left = 52
    pad_right = 16
    pad_top = 18
    pad_bottom = 28
    min_value = float(array.min())
    max_value = float(array.max())
    if min_value == max_value:
        max_value = min_value + 1.0

    chart_height = height - pad_top - pad_bottom
    tick_values = np.linspace(max_value, min_value, 4)
    y_ticks = []
    for tick_value in tick_values:
        y_position = height - pad_bottom - ((tick_value - min_value) / (max_value - min_value)) * chart_height
        y_ticks.append(
            {
                "y": f"{y_position:.1f}",
                "label": f"{tick_value:,.1f}",
            }
        )

    x_ticks = [
        {"x": f"{pad_left:.1f}", "label": start_label},
        {"x": f"{((pad_left + (width - pad_right)) / 2):.1f}", "label": middle_label},
        {"x": f"{(width - pad_right):.1f}", "label": end_label},
    ]

    latest_value = float(array[-1])
    return {
        "points": _build_line_points(
            array,
            width=width,
            height=height,
            pad_left=pad_left,
            pad_right=pad_right,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
        ),
        "y_ticks": y_ticks,
        "x_ticks": x_ticks,
        "last_value": _format_rate(latest_value, local_currency, target_currency),
        "low_value": _format_rate(float(array.min()), local_currency, target_currency),
        "high_value": _format_rate(float(array.max()), local_currency, target_currency),
    }


def _build_terminal_summary(terminal_values, starting_rate: float, local_currency: str, target_currency: str) -> dict:
    values = np.asarray(terminal_values, dtype=float)
    if values.size == 0:
        return {}

    return {
        "minimum": _format_rate(float(values.min()), local_currency, target_currency),
        "median": _format_rate(float(np.median(values)), local_currency, target_currency),
        "maximum": _format_rate(float(values.max()), local_currency, target_currency),
        "minimum_value": f"{float(values.min()):,.4f}",
        "median_value": f"{float(np.median(values)):,.4f}",
        "maximum_value": f"{float(values.max()):,.4f}",
        "pair_label": f"{local_currency}/{target_currency}",
        "above_start_probability": f"{float((values >= starting_rate).mean() * 100):.1f}%",
    }


def _build_forecast_checkpoint_rows(
    values,
    local_currency: str,
    target_currency: str,
) -> list[dict]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return []

    last_index = array.size - 1
    checkpoints = [0, last_index // 4, last_index // 2, (last_index * 3) // 4, last_index]
    unique_points = []
    for point in checkpoints:
        if point not in unique_points:
            unique_points.append(point)

    rows = []
    for point in unique_points:
        label = "Today" if point == 0 else f"Day {point}"
        rows.append(
            {
                "label": label,
                "value": _format_rate(float(array[point]), local_currency, target_currency),
            }
        )
    return rows


def _build_most_likely_range(histogram_rows: list[dict]) -> dict:
    if not histogram_rows:
        return {}
    return max(histogram_rows, key=lambda row: row["count"])


def _parse_manual_series(raw_text: str) -> pd.Series:
    cleaned = raw_text.replace(",", "\n")
    values = []
    for part in cleaned.splitlines():
        candidate = part.strip()
        if not candidate:
            continue
        try:
            values.append(float(candidate))
        except ValueError as exc:
            raise ValueError(
                "The rate list must contain numbers only, separated by commas or one number per line."
            ) from exc

    if values and len(values) < 2:
        raise ValueError("Enter at least two rate values so the app can estimate the trend and market swing.")

    return pd.Series(values, dtype=float)


def _serialize_strategy_rows(summary_df: pd.DataFrame, currency_label: str) -> list[dict]:
    rows = []
    for _, row in summary_df.iterrows():
        rows.append(
            {
                "strategy_code": _display_strategy_code(row["strategy_code"]),
                "name": row["name"],
                "average_return": _format_money(float(row["average_return"]), currency_label),
                "risk_std_dev": _format_money(float(row["risk_std_dev"]), currency_label),
                "total_profit": _format_money(float(row["total_profit"]), currency_label),
                "risk_adjusted_score": f"{float(row['risk_adjusted_score']):.2f}",
            }
        )
    return rows


def _serialize_weekday_rows(weekday_df: pd.DataFrame, local_currency: str, target_currency: str) -> list[dict]:
    rows = []
    for _, row in weekday_df.iterrows():
        rows.append(
            {
                "weekday": row["Weekday"],
                "average_rate": _format_rate(float(row["Average_Rate"]), local_currency, target_currency),
                "average_change": f"{float(row['Average_Change']):,.2f}%",
                "observations": int(row["Observations"]),
                "market_mood": row["Market Mood"],
            }
        )
    return rows


def _build_history_preview_rows(history_df: pd.DataFrame, local_currency: str, target_currency: str) -> list[dict]:
    if history_df.empty:
        return []

    preview = history_df.tail(6).copy()
    rows = []
    for _, row in preview.iterrows():
        rows.append(
            {
                "date": row["Date"].strftime("%Y-%m-%d"),
                "rate": _format_rate(float(row["Rate"]), local_currency, target_currency),
            }
        )
    return rows


def _collect_form_values() -> dict:
    return {
        "input_mode": request.form.get("input_mode", DEFAULTS["input_mode"]),
        "history_file_name": request.files.get("history_file").filename if request.files.get("history_file") else "",
        "starting_rate": request.form.get("starting_rate", ""),
        "mu": request.form.get("mu", str(DEFAULTS["mu"])),
        "sigma": request.form.get("sigma", str(DEFAULTS["sigma"])),
        "days": request.form.get("days", str(DEFAULTS["days"])),
        "n_simulations": request.form.get("n_simulations", str(DEFAULTS["n_simulations"])),
        "initial_capital": request.form.get("initial_capital", str(DEFAULTS["initial_capital"])),
        "buy_threshold_percent": request.form.get("buy_threshold_percent", str(DEFAULTS["buy_threshold"] * 100)),
        "sell_threshold_percent": request.form.get("sell_threshold_percent", str(DEFAULTS["sell_threshold"] * 100)),
        "trend_lookback": request.form.get("trend_lookback", str(DEFAULTS["trend_lookback"])),
        "manual_series": request.form.get("manual_series", ""),
        "local_currency": request.form.get("local_currency", "RWF"),
        "target_currency": request.form.get("target_currency", "USD"),
        "extra_currency_rates": request.form.get("extra_currency_rates", "EUR=1300\nGBP=1500"),
        "calculator_amount": request.form.get("calculator_amount", request.form.get("initial_capital", str(DEFAULTS["initial_capital"]))),
        "calculator_from_currency": request.form.get("calculator_from_currency", request.form.get("local_currency", "RWF")),
        "calculator_to_currency": request.form.get("calculator_to_currency", request.form.get("target_currency", "USD")),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    errors: list[str] = []
    results = None
    form_values = {
        "input_mode": DEFAULTS["input_mode"],
        "history_file_name": "",
        "starting_rate": "",
        "mu": f"{DEFAULTS['mu']}",
        "sigma": f"{DEFAULTS['sigma']}",
        "days": f"{DEFAULTS['days']}",
        "n_simulations": f"{DEFAULTS['n_simulations']}",
        "initial_capital": f"{DEFAULTS['initial_capital']}",
        "buy_threshold_percent": f"{DEFAULTS['buy_threshold'] * 100}",
        "sell_threshold_percent": f"{DEFAULTS['sell_threshold'] * 100}",
        "trend_lookback": f"{DEFAULTS['trend_lookback']}",
        "manual_series": "",
        "local_currency": "RWF",
        "target_currency": "USD",
        "extra_currency_rates": "EUR=1300\nGBP=1500",
        "calculator_amount": f"{DEFAULTS['initial_capital']}",
        "calculator_from_currency": "RWF",
        "calculator_to_currency": "USD",
    }

    if request.method == "POST":
        form_values = _collect_form_values()

        try:
            input_mode = form_values["input_mode"]
            starting_rate = _parse_required_float("starting_rate")
            days = _parse_int("days", DEFAULTS["days"])
            n_simulations = _parse_int("n_simulations", DEFAULTS["n_simulations"])
            initial_capital = _parse_float("initial_capital", DEFAULTS["initial_capital"])
            buy_threshold = _parse_float("buy_threshold_percent", DEFAULTS["buy_threshold"] * 100) / 100
            sell_threshold = _parse_float("sell_threshold_percent", DEFAULTS["sell_threshold"] * 100) / 100
            trend_lookback = _parse_int("trend_lookback", DEFAULTS["trend_lookback"])
            local_currency = _clean_currency_label(form_values["local_currency"], "RWF")
            target_currency = _clean_currency_label(form_values["target_currency"], "USD")
            calculator_amount = _parse_float("calculator_amount", initial_capital)
            form_values["local_currency"] = local_currency
            form_values["target_currency"] = target_currency

            history_df = pd.DataFrame()
            history_metadata = None
            source_label = "Typed by hand"

            if input_mode == "upload":
                uploaded_file = request.files.get("history_file")
                if uploaded_file is None or not uploaded_file.filename:
                    raise ValueError("Upload your past CSV file so the app can learn from earlier exchange rates.")

                uploaded_df = pd.read_csv(uploaded_file)
                history_df, history_metadata = prepare_uploaded_history(uploaded_df)
                _, mu, sigma = derive_parameters_from_history(history_df["Rate"])
                source_name = history_metadata["source_name"]
                source_label = f"{source_name} ({len(history_df)} past rows)"

                form_values["mu"] = f"{mu:.4f}"
                form_values["sigma"] = f"{sigma:.4f}"
            else:
                manual_series = _parse_manual_series(form_values["manual_series"])
                if not manual_series.empty:
                    _, mu, sigma = derive_parameters_from_history(manual_series)
                    history_df = pd.DataFrame(
                        {
                            "Date": pd.date_range(
                                end=pd.Timestamp.today().normalize(),
                                periods=len(manual_series),
                                freq="D",
                            ),
                            "Rate": manual_series.to_numpy(dtype=float),
                        }
                    )
                    history_metadata = {
                        "source_name": "Rate list you typed",
                        "original_rows": int(len(manual_series)),
                        "kept_rows": int(len(manual_series)),
                        "invalid_date_rows": 0,
                        "invalid_rate_rows": 0,
                        "duplicate_date_rows": 0,
                    }
                    source_label = f"Rate list you typed ({len(history_df)} past rows)"
                    form_values["mu"] = f"{mu:.4f}"
                    form_values["sigma"] = f"{sigma:.4f}"
                else:
                    mu = _parse_float("mu", DEFAULTS["mu"])
                    sigma = _parse_float("sigma", DEFAULTS["sigma"])

            if starting_rate <= 0:
                raise ValueError("Starting rate must be greater than zero.")
            if initial_capital <= 0:
                raise ValueError("Money you start with must be greater than zero.")
            if days < 2:
                raise ValueError("Days to test must be at least 2.")
            if n_simulations < 1:
                raise ValueError("Number of test runs must be at least 1.")
            if sigma < 0:
                raise ValueError("Market swing cannot be negative.")
            if sell_threshold <= 0:
                raise ValueError("Sell gain percent must be greater than zero.")
            if trend_lookback > days:
                raise ValueError("Trend check days must not be greater than days to test.")

            calculator_rates = parse_extra_currency_rates(
                form_values["extra_currency_rates"],
                local_currency=local_currency,
                target_currency=target_currency,
                target_rate=float(starting_rate),
            )
            available_currencies = list(calculator_rates.keys())
            calculator_from_currency = _clean_currency_label(
                form_values["calculator_from_currency"],
                local_currency,
            )
            calculator_to_currency = _clean_currency_label(
                form_values["calculator_to_currency"],
                target_currency,
            )
            if calculator_from_currency not in calculator_rates:
                calculator_from_currency = local_currency
            if calculator_to_currency not in calculator_rates:
                calculator_to_currency = target_currency
            form_values["calculator_from_currency"] = calculator_from_currency
            form_values["calculator_to_currency"] = calculator_to_currency
            calculator_result = convert_currency_amount(
                amount=calculator_amount,
                from_currency=calculator_from_currency,
                to_currency=calculator_to_currency,
                rate_map=calculator_rates,
            )

            paths = monte_carlo_simulation(
                S0=starting_rate,
                mu=mu,
                sigma=sigma,
                days=days,
                n_simulations=n_simulations,
            )
            summary_df = run_all_strategies(
                paths=paths,
                initial_capital=initial_capital,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                trend_lookback=trend_lookback,
            )
            best_strategy = get_best_strategy(summary_df)
            terminal_values = paths[:, -1]
            suggested_buy, suggested_sell, level_explanation = build_reference_levels(
                best_strategy["strategy_code"],
                paths,
                float(starting_rate),
                buy_threshold,
                sell_threshold,
            )
            weekday_df = build_weekday_market_summary(history_df) if not history_df.empty else pd.DataFrame()
            history_analysis = build_history_analysis(history_df) if not history_df.empty else {}
            median_terminal_rate = float(np.median(terminal_values))
            terminal_histogram_rows = _build_histogram_rows(terminal_values)
            history_chart = (
                _build_chart_panel(
                    history_df["Rate"].tail(24),
                    local_currency,
                    target_currency,
                    "Oldest",
                    "Middle",
                    "Latest",
                )
                if not history_df.empty
                else {}
            )
            forecast_chart = _build_chart_panel(
                paths.mean(axis=0),
                local_currency,
                target_currency,
                "Today",
                f"Day {max(1, days // 2)}",
                f"Day {days}",
            )
            starting_foreign_units = initial_capital / float(starting_rate)
            calculator_foreign_amount = calculator_amount / float(starting_rate)
            scenario_cards = [
                {
                    "label": "Start with",
                    "value": _format_money(initial_capital, local_currency),
                    "detail": "This is the money you start with.",
                },
                {
                    "label": "Can buy now",
                    "value": f"{starting_foreign_units:,.2f} {target_currency}",
                    "detail": f"Using the start rate of {_format_rate(starting_rate, local_currency, target_currency)}.",
                },
                {
                    "label": "Likely end value",
                    "value": _format_money(starting_foreign_units * median_terminal_rate, local_currency),
                    "detail": f"If the ending rate is close to {_format_rate(median_terminal_rate, local_currency, target_currency)}.",
                },
                {
                    "label": "Possible range",
                    "value": (
                        f"{_format_money(starting_foreign_units * float(np.min(terminal_values)), local_currency)}"
                        f" to {_format_money(starting_foreign_units * float(np.max(terminal_values)), local_currency)}"
                    ),
                    "detail": "A simple low-to-high range from the test results.",
                },
            ]
            best_reason_short = (
                f"It gave the best mix of gain and risk in this test. "
                f"Average gain was {_format_money(float(best_strategy['average_return']), local_currency)} "
                f"with risk of {_format_money(float(best_strategy['risk_std_dev']), local_currency)}."
            )
            if history_metadata and input_mode == "upload":
                learning_summary = (
                    f"The app cleaned your CSV, kept {history_metadata['kept_rows']} good rows, "
                    f"then learned the average daily change and market swing. "
                    f"The current market rate came from what you typed."
                )
            elif not history_df.empty:
                learning_summary = (
                    "The app used the values you typed by hand to learn the trend and market swing. "
                    "The current market rate came from what you typed."
                )
            else:
                learning_summary = "The app used the values you typed by hand, including the current market rate."

            data_quality_cards = []
            history_cards = []
            cleaning_note = ""
            history_note = ""
            history_preview_rows = []
            entered_rate_cards = [
                {
                    "label": "Rate you entered today",
                    "value": _format_rate(starting_rate, local_currency, target_currency),
                }
            ]

            if history_metadata:
                original_rows = int(history_metadata.get("original_rows", len(history_df)))
                kept_rows = int(history_metadata.get("kept_rows", len(history_df)))
                removed_rows = max(0, original_rows - kept_rows)
                date_span = "Not available"
                if history_analysis:
                    date_span = (
                        f"{history_analysis['date_start'].strftime('%Y-%m-%d')} to "
                        f"{history_analysis['date_end'].strftime('%Y-%m-%d')}"
                    )

                data_quality_cards = [
                    {"label": "Rows found", "value": f"{original_rows:,}"},
                    {"label": "Rows used", "value": f"{kept_rows:,}"},
                    {"label": "Rows removed", "value": f"{removed_rows:,}"},
                    {"label": "Date span", "value": date_span},
                ]
                cleaning_note = (
                    f"Source used: {source_label}. "
                    f"Removed rows usually had a missing date, missing rate, or a repeated date."
                    if input_mode == "upload"
                    else "No CSV cleaning was needed because you typed the rate list by hand."
                )

            if history_analysis:
                history_cards = [
                    {
                        "label": "First rate",
                        "value": _format_rate(history_analysis["first_rate"], local_currency, target_currency),
                    },
                    {
                        "label": "Latest rate",
                        "value": _format_rate(history_analysis["latest_rate"], local_currency, target_currency),
                    },
                    {
                        "label": "Lowest rate",
                        "value": _format_rate(history_analysis["lowest_rate"], local_currency, target_currency),
                    },
                    {
                        "label": "Highest rate",
                        "value": _format_rate(history_analysis["highest_rate"], local_currency, target_currency),
                    },
                ]
                history_note = (
                    f"{history_analysis['trend_label']}. "
                    f"Average rate was {_format_rate(history_analysis['average_rate'], local_currency, target_currency)}. "
                    f"Overall change was {_format_signed_percent(history_analysis['overall_change_percent'])} "
                    f"({_format_signed_number(history_analysis['overall_change_value'])} in rate units). "
                    f"Average daily change was {_format_signed_number(history_analysis['average_daily_change'])} "
                    f"and market swing was {float(history_analysis['market_swing']):,.4f}."
                )
                history_preview_rows = _build_history_preview_rows(history_df, local_currency, target_currency)
                entered_rate_cards.append(
                    {
                        "label": "Last rate in your history",
                        "value": _format_rate(history_analysis["latest_rate"], local_currency, target_currency),
                    }
                )
                entered_rate_cards.append(
                    {
                        "label": "Difference",
                        "value": _format_signed_number(starting_rate - history_analysis["latest_rate"]),
                    }
                )

            model_cards = [
                {
                    "label": "Model type",
                    "value": "Stochastic, dynamic, discrete-time",
                    "detail": "This matches the class idea of a random model that changes over time.",
                },
                {
                    "label": "Method",
                    "value": "Random walk and Monte Carlo",
                    "detail": "The app uses many random test runs to estimate possible futures.",
                },
                {
                    "label": "Data use",
                    "value": "User rate plus past history",
                    "detail": "You enter the real current rate. Past rates are only used to estimate trend and swing.",
                },
                {
                    "label": "Why results change",
                    "value": "Random numbers are involved",
                    "detail": "Each run may be a little different because this is not a deterministic model.",
                },
            ]
            report_data = {
                "best_strategy_code": best_strategy["strategy_code"],
                "best_strategy_label": _display_strategy_code(best_strategy["strategy_code"]),
                "best_strategy_name": best_strategy["name"],
                "best_explanation": explain_best_strategy(best_strategy),
                "best_reason_short": best_reason_short,
                "simulation_story": (
                    f"If you start with {_format_money(initial_capital, local_currency)}, you can buy about "
                    f"{starting_foreign_units:,.2f} {target_currency} at the market rate you entered. "
                    f"The test then shows simple possible up and down moves over {days} days."
                ),
                "learning_summary": learning_summary,
                "source_label": source_label,
                "starting_rate": _format_rate(float(starting_rate), local_currency, target_currency),
                "local_currency": local_currency,
                "target_currency": target_currency,
                "mu": f"{float(mu):,.4f}",
                "sigma": f"{float(sigma):,.4f}",
                "days": days,
                "n_simulations": n_simulations,
                "suggested_buy": _format_rate(suggested_buy, local_currency, target_currency),
                "suggested_sell": _format_rate(suggested_sell, local_currency, target_currency),
                "scenario_cards": scenario_cards,
                "calculator_summary": {
                    "from_amount": f"{calculator_amount:,.2f} {calculator_from_currency}",
                    "to_amount": f"{calculator_result:,.2f} {calculator_to_currency}",
                    "rate_note": (
                        f"Rate used: 1 {target_currency} = {float(starting_rate):,.4f} {local_currency}. "
                        f"This is the market rate you entered."
                    ),
                },
                "strategy_rows": _serialize_strategy_rows(summary_df, local_currency),
                "terminal_summary": _build_terminal_summary(
                    terminal_values,
                    float(starting_rate),
                    local_currency,
                    target_currency,
                ),
                "terminal_histogram_rows": terminal_histogram_rows,
                "forecast_checkpoint_rows": _build_forecast_checkpoint_rows(
                    paths.mean(axis=0),
                    local_currency,
                    target_currency,
                ),
                "weekday_rows": _serialize_weekday_rows(weekday_df, local_currency, target_currency),
                "strategy_rules_rows": build_strategy_rules_table(days).to_dict(orient="records"),
            }
            pdf_report = build_pdf_report(report_data)
            word_report = build_word_report(report_data)

            results = {
                "input_mode_label": "CSV file" if input_mode == "upload" else "Typed by hand",
                "source_label": source_label,
                "learning_summary": learning_summary,
                "data_quality_cards": data_quality_cards,
                "cleaning_note": cleaning_note,
                "history_cards": history_cards,
                "history_note": history_note,
                "history_preview_rows": history_preview_rows,
                "entered_rate_cards": entered_rate_cards,
                "model_cards": model_cards,
                "local_currency": local_currency,
                "target_currency": target_currency,
                "starting_rate": _format_rate(float(starting_rate), local_currency, target_currency),
                "mu": f"{float(mu):,.4f}",
                "sigma": f"{float(sigma):,.4f}",
                "days": days,
                "n_simulations": n_simulations,
                "best_strategy_code": _display_strategy_code(best_strategy["strategy_code"]),
                "best_strategy_name": best_strategy["name"],
                "best_average_return": _format_money(float(best_strategy["average_return"]), local_currency),
                "best_risk": _format_money(float(best_strategy["risk_std_dev"]), local_currency),
                "best_total_profit": _format_money(float(best_strategy["total_profit"]), local_currency),
                "best_explanation": explain_best_strategy(best_strategy),
                "best_reason_short": best_reason_short,
                "suggested_buy": _format_rate(suggested_buy, local_currency, target_currency),
                "suggested_sell": _format_rate(suggested_sell, local_currency, target_currency),
                "level_explanation": level_explanation,
                "buy_threshold_display": _format_percent(buy_threshold),
                "sell_threshold_display": _format_percent(sell_threshold),
                "history_chart": history_chart,
                "forecast_chart": forecast_chart,
                "terminal_histogram_rows": terminal_histogram_rows,
                "most_likely_range": _build_most_likely_range(terminal_histogram_rows),
                "terminal_summary": _build_terminal_summary(
                    terminal_values,
                    float(starting_rate),
                    local_currency,
                    target_currency,
                ),
                "strategy_rows": _serialize_strategy_rows(summary_df, local_currency),
                "weekday_rows": _serialize_weekday_rows(weekday_df, local_currency, target_currency),
                "strategy_rules_rows": build_strategy_rules_table(days).to_dict(orient="records"),
                "profit_chart_rows": _build_chart_rows(summary_df, "average_return", local_currency),
                "risk_chart_rows": _build_chart_rows(summary_df, "risk_std_dev", local_currency),
                "sample_path_rows": _build_sample_path_rows(paths),
                "summary_download_url": _data_url(pdf_report, "application/pdf"),
                "word_download_url": _data_url(word_report, "application/rtf"),
                "exchange_card": {
                    "amount_local": _format_money(calculator_amount, local_currency),
                    "amount_foreign": f"{calculator_foreign_amount:,.2f} {target_currency}",
                    "rate_line": f"1 {target_currency} = {float(starting_rate):,.4f} {local_currency}",
                    "round_trip": _format_money(calculator_foreign_amount * median_terminal_rate, local_currency),
                },
                "calculator_summary": {
                    "from_currency": calculator_from_currency,
                    "to_currency": calculator_to_currency,
                    "from_amount": f"{calculator_amount:,.2f} {calculator_from_currency}",
                    "to_amount": f"{calculator_result:,.2f} {calculator_to_currency}",
                    "rate_note": (
                        f"We convert using the market rate you typed. "
                        f"1 {target_currency} = {float(starting_rate):,.4f} {local_currency}"
                    ),
                },
                "available_currencies": available_currencies,
                "scenario_cards": scenario_cards,
                "forecast_checkpoint_rows": _build_forecast_checkpoint_rows(
                    paths.mean(axis=0),
                    local_currency,
                    target_currency,
                ),
                "simulation_story": (
                    f"If you start with {_format_money(initial_capital, local_currency)}, you can buy about "
                    f"{starting_foreign_units:,.2f} {target_currency} at the market rate you entered. "
                    f"The test then shows simple possible up and down moves over {days} days."
                ),
            }
        except Exception as exc:
            errors.append(str(exc))

    return render_template("index.html", form=form_values, errors=errors, results=results)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
