"""Flask entrypoint for the Vercel deployment and browser UI."""

from __future__ import annotations

from base64 import b64encode

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from app_logic import (
    DEFAULTS,
    build_pdf_report,
    build_reference_levels,
    build_strategy_rules_table,
    build_weekday_market_summary,
    choose_default_pair,
    choose_default_rate_column,
    convert_currency_amount,
    derive_parameters_from_history,
    explain_best_strategy,
    extract_history_frame,
    get_best_strategy,
    parse_extra_currency_rates,
    run_all_strategies,
    validate_csv,
)
from simulation import monte_carlo_simulation


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024


def _parse_float(name: str, default: float) -> float:
    raw_value = request.form.get(name, str(default)).strip()
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"'{name}' must be a valid number.") from exc


def _parse_int(name: str, default: int) -> int:
    raw_value = request.form.get(name, str(default)).strip()
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"'{name}' must be a whole number.") from exc


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


def _build_histogram_rows(values, bins: int = 9) -> list[dict]:
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

    rows = []
    for index, count in enumerate(counts):
        height = max(16, int((int(count) / peak_count) * 100)) if int(count) > 0 else 10
        rows.append(
            {
                "label": f"{edges[index]:,.1f} - {edges[index + 1]:,.1f}",
                "count": int(count),
                "height": height,
            }
        )
    return rows


def _build_line_points(values, width: int = 420, height: int = 160, pad: int = 16) -> str:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return ""
    if array.size == 1:
        return f"{width / 2:.1f},{height / 2:.1f}"

    x_values = np.linspace(pad, width - pad, array.size)
    min_value = float(array.min())
    max_value = float(array.max())
    if min_value == max_value:
        max_value = min_value + 1.0

    y_values = height - pad - ((array - min_value) / (max_value - min_value)) * (height - (pad * 2))
    return " ".join(f"{x_value:.1f},{y_value:.1f}" for x_value, y_value in zip(x_values, y_values))


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
                "Manual rates must be numbers only, separated by commas or one value per line."
            ) from exc

    if values and len(values) < 2:
        raise ValueError("Enter at least two manual rate values so the app can estimate mu and sigma.")

    return pd.Series(values, dtype=float)


def _serialize_strategy_rows(summary_df: pd.DataFrame, currency_label: str) -> list[dict]:
    rows = []
    for _, row in summary_df.iterrows():
        rows.append(
            {
                "strategy_code": row["strategy_code"],
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


def _collect_form_values() -> dict:
    return {
        "input_mode": request.form.get("input_mode", DEFAULTS["input_mode"]),
        "history_file_name": request.files.get("history_file").filename if request.files.get("history_file") else "",
        "starting_rate": request.form.get("starting_rate", str(DEFAULTS["starting_rate"])),
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
        "starting_rate": f"{DEFAULTS['starting_rate']}",
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
            source_label = "Manual entry"
            selected_pair = None
            rate_column = None

            if input_mode == "upload":
                uploaded_file = request.files.get("history_file")
                if uploaded_file is None or not uploaded_file.filename:
                    raise ValueError("Upload a CSV file to estimate the model from historical exchange rates.")

                uploaded_df = pd.read_csv(uploaded_file)
                is_valid, message = validate_csv(uploaded_df)
                if not is_valid:
                    raise ValueError(message)

                selected_pair = choose_default_pair(uploaded_df)
                rate_column = choose_default_rate_column(uploaded_df)
                history_df = extract_history_frame(uploaded_df, rate_column, selected_pair)
                starting_rate, mu, sigma = derive_parameters_from_history(history_df["Rate"])
                source_label = selected_pair or rate_column

                form_values["starting_rate"] = f"{starting_rate:.4f}"
                form_values["mu"] = f"{mu:.4f}"
                form_values["sigma"] = f"{sigma:.4f}"
            else:
                manual_series = _parse_manual_series(form_values["manual_series"])
                if not manual_series.empty:
                    starting_rate, mu, sigma = derive_parameters_from_history(manual_series)
                    source_label = "Manual rate list"
                    form_values["starting_rate"] = f"{starting_rate:.4f}"
                    form_values["mu"] = f"{mu:.4f}"
                    form_values["sigma"] = f"{sigma:.4f}"
                else:
                    starting_rate = _parse_float("starting_rate", DEFAULTS["starting_rate"])
                    mu = _parse_float("mu", DEFAULTS["mu"])
                    sigma = _parse_float("sigma", DEFAULTS["sigma"])

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
            median_terminal_rate = float(np.median(terminal_values))
            starting_foreign_units = initial_capital / float(starting_rate)
            calculator_foreign_amount = calculator_amount / float(starting_rate)
            scenario_cards = [
                {
                    "label": "Start with",
                    "value": _format_money(initial_capital, local_currency),
                    "detail": f"Capital entered before the model starts.",
                },
                {
                    "label": f"Can buy now",
                    "value": f"{starting_foreign_units:,.2f} {target_currency}",
                    "detail": f"At {_format_rate(starting_rate, local_currency, target_currency)}.",
                },
                {
                    "label": "Median end value",
                    "value": _format_money(starting_foreign_units * median_terminal_rate, local_currency),
                    "detail": f"If the end rate lands near {_format_rate(median_terminal_rate, local_currency, target_currency)}.",
                },
                {
                    "label": "Possible range",
                    "value": (
                        f"{_format_money(starting_foreign_units * float(np.min(terminal_values)), local_currency)}"
                        f" to {_format_money(starting_foreign_units * float(np.max(terminal_values)), local_currency)}"
                    ),
                    "detail": "Worst and best end values from the simulated terminal rates.",
                },
            ]
            best_reason_short = (
                f"It had the best balance of return and risk score, with an average return of "
                f"{_format_money(float(best_strategy['average_return']), local_currency)} "
                f"and risk of {_format_money(float(best_strategy['risk_std_dev']), local_currency)}."
            )
            pdf_report = build_pdf_report(
                {
                    "best_strategy_code": best_strategy["strategy_code"],
                    "best_strategy_name": best_strategy["name"],
                    "best_explanation": explain_best_strategy(best_strategy),
                    "best_reason_short": best_reason_short,
                    "simulation_story": (
                        f"If you start with {_format_money(initial_capital, local_currency)}, the model treats that as "
                        f"about {starting_foreign_units:,.2f} {target_currency} at the opening rate. "
                        f"From there, the Monte Carlo paths show how that value could rise or fall over {days} days."
                    ),
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
                            f"Rate map used: 1 {target_currency} = {float(starting_rate):,.4f} {local_currency}. "
                            f"You can also add more rates in the calculator box."
                        ),
                    },
                    "strategy_rows": _serialize_strategy_rows(summary_df, local_currency),
                    "terminal_summary": _build_terminal_summary(
                        terminal_values,
                        float(starting_rate),
                        local_currency,
                        target_currency,
                    ),
                    "terminal_histogram_rows": _build_histogram_rows(terminal_values),
                    "weekday_rows": _serialize_weekday_rows(weekday_df, local_currency, target_currency),
                    "strategy_rules_rows": build_strategy_rules_table(days).to_dict(orient="records"),
                }
            )

            results = {
                "input_mode_label": "CSV upload" if input_mode == "upload" else "Manual entry",
                "source_label": source_label,
                "local_currency": local_currency,
                "target_currency": target_currency,
                "starting_rate": _format_rate(float(starting_rate), local_currency, target_currency),
                "mu": f"{float(mu):,.4f}",
                "sigma": f"{float(sigma):,.4f}",
                "days": days,
                "n_simulations": n_simulations,
                "best_strategy_code": best_strategy["strategy_code"],
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
                "forecast_line_points": _build_line_points(paths.mean(axis=0)),
                "history_line_points": _build_line_points(history_df["Rate"].tail(24)) if not history_df.empty else "",
                "terminal_histogram_rows": _build_histogram_rows(terminal_values),
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
                        f"We convert through {local_currency}. "
                        f"1 {target_currency} = {float(starting_rate):,.4f} {local_currency}"
                    ),
                },
                "available_currencies": available_currencies,
                "scenario_cards": scenario_cards,
                "simulation_story": (
                    f"If you start with {_format_money(initial_capital, local_currency)}, the model treats that as "
                    f"about {starting_foreign_units:,.2f} {target_currency} at the opening rate. "
                    f"From there, the Monte Carlo paths show how that value could rise or fall over {days} days."
                ),
            }
        except Exception as exc:
            errors.append(str(exc))

    return render_template("index.html", form=form_values, errors=errors, results=results)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
