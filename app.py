"""Flask entrypoint for the Vercel deployment and browser UI."""

from __future__ import annotations

from base64 import b64encode

import pandas as pd
from flask import Flask, render_template, request

from app_logic import (
    DEFAULTS,
    build_downloadable_results,
    build_reference_levels,
    build_strategy_rules_table,
    build_weekday_market_summary,
    choose_default_pair,
    choose_default_rate_column,
    derive_parameters_from_history,
    explain_best_strategy,
    extract_history_frame,
    format_rwf,
    get_best_strategy,
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


def _csv_data_url(content: str) -> str:
    encoded = b64encode(content.encode("utf-8")).decode("ascii")
    return f"data:text/csv;base64,{encoded}"


def _build_chart_rows(summary_df: pd.DataFrame, metric_key: str) -> list[dict]:
    ceiling = float(summary_df[metric_key].abs().max())
    if ceiling == 0:
        ceiling = 1.0

    rows = []
    for _, row in summary_df.iterrows():
        width = max(12, int(abs(float(row[metric_key])) / ceiling * 100))
        rows.append(
            {
                "name": row["name"],
                "value": format_rwf(float(row[metric_key])),
                "width": width,
            }
        )
    return rows


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


def _serialize_strategy_rows(summary_df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in summary_df.iterrows():
        rows.append(
            {
                "strategy_code": row["strategy_code"],
                "name": row["name"],
                "average_return": format_rwf(float(row["average_return"])),
                "risk_std_dev": format_rwf(float(row["risk_std_dev"])),
                "total_profit": format_rwf(float(row["total_profit"])),
                "risk_adjusted_score": f"{float(row['risk_adjusted_score']):.2f}",
            }
        )
    return rows


def _serialize_weekday_rows(weekday_df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in weekday_df.iterrows():
        rows.append(
            {
                "weekday": row["Weekday"],
                "average_rate": f"{float(row['Average_Rate']):,.3f}",
                "average_change": f"{float(row['Average_Change']):,.2f}%",
                "observations": int(row["Observations"]),
                "market_mood": row["Market Mood"],
            }
        )
    return rows


def _collect_form_values() -> dict:
    return {
        "input_mode": request.form.get("input_mode", DEFAULTS["input_mode"]),
        "starting_rate": request.form.get("starting_rate", str(DEFAULTS["starting_rate"])),
        "mu": request.form.get("mu", str(DEFAULTS["mu"])),
        "sigma": request.form.get("sigma", str(DEFAULTS["sigma"])),
        "days": request.form.get("days", str(DEFAULTS["days"])),
        "n_simulations": request.form.get("n_simulations", str(DEFAULTS["n_simulations"])),
        "initial_capital": request.form.get("initial_capital", str(DEFAULTS["initial_capital"])),
        "buy_threshold_percent": request.form.get("buy_threshold_percent", str(DEFAULTS["buy_threshold"] * 100)),
        "sell_threshold_percent": request.form.get("sell_threshold_percent", str(DEFAULTS["sell_threshold"] * 100)),
        "trend_lookback": request.form.get("trend_lookback", str(DEFAULTS["trend_lookback"])),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    errors: list[str] = []
    results = None
    form_values = {
        "input_mode": DEFAULTS["input_mode"],
        "starting_rate": f"{DEFAULTS['starting_rate']}",
        "mu": f"{DEFAULTS['mu']}",
        "sigma": f"{DEFAULTS['sigma']}",
        "days": f"{DEFAULTS['days']}",
        "n_simulations": f"{DEFAULTS['n_simulations']}",
        "initial_capital": f"{DEFAULTS['initial_capital']}",
        "buy_threshold_percent": f"{DEFAULTS['buy_threshold'] * 100}",
        "sell_threshold_percent": f"{DEFAULTS['sell_threshold'] * 100}",
        "trend_lookback": f"{DEFAULTS['trend_lookback']}",
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
                starting_rate = _parse_float("starting_rate", DEFAULTS["starting_rate"])
                mu = _parse_float("mu", DEFAULTS["mu"])
                sigma = _parse_float("sigma", DEFAULTS["sigma"])

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
            suggested_buy, suggested_sell, level_explanation = build_reference_levels(
                best_strategy["strategy_code"],
                paths,
                float(starting_rate),
                buy_threshold,
                sell_threshold,
            )
            weekday_df = build_weekday_market_summary(history_df) if not history_df.empty else pd.DataFrame()

            results = {
                "input_mode_label": "CSV upload" if input_mode == "upload" else "Manual entry",
                "source_label": source_label,
                "starting_rate": format_rwf(float(starting_rate)),
                "mu": f"{float(mu):,.4f}",
                "sigma": f"{float(sigma):,.4f}",
                "days": days,
                "n_simulations": n_simulations,
                "best_strategy_code": best_strategy["strategy_code"],
                "best_strategy_name": best_strategy["name"],
                "best_average_return": format_rwf(float(best_strategy["average_return"])),
                "best_risk": format_rwf(float(best_strategy["risk_std_dev"])),
                "best_total_profit": format_rwf(float(best_strategy["total_profit"])),
                "best_explanation": explain_best_strategy(best_strategy),
                "suggested_buy": format_rwf(suggested_buy),
                "suggested_sell": format_rwf(suggested_sell),
                "level_explanation": level_explanation,
                "strategy_rows": _serialize_strategy_rows(summary_df),
                "weekday_rows": _serialize_weekday_rows(weekday_df),
                "strategy_rules_rows": build_strategy_rules_table(days).to_dict(orient="records"),
                "profit_chart_rows": _build_chart_rows(summary_df, "average_return"),
                "risk_chart_rows": _build_chart_rows(summary_df, "risk_std_dev"),
                "sample_path_rows": _build_sample_path_rows(paths),
                "summary_download_url": _csv_data_url(build_downloadable_results(summary_df)),
            }
        except Exception as exc:
            errors.append(str(exc))

    return render_template("index.html", form=form_values, errors=errors, results=results)


if __name__ == "__main__":
    app.run(debug=True)
