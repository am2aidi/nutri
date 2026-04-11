from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st

from simulation import (
    calculate_metrics,
    monte_carlo_simulation,
    strategy_buy_hold,
    strategy_threshold,
    strategy_trend_following,
)


DEFAULTS = {
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


st.set_page_config(
    page_title="Currency Exchange Rate Simulation",
    layout="wide",
)


st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.12), transparent 30%),
                linear-gradient(180deg, #f7fafc 0%, #eef3f8 100%);
            color: #15314b;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f2740 0%, #173b5e 100%);
        }
        [data-testid="stSidebar"] * {
            color: #f7fbff;
        }
        .hero-card {
            background: linear-gradient(135deg, #0f2740 0%, #1f4f79 100%);
            border: 1px solid rgba(20, 61, 99, 0.16);
            border-radius: 20px;
            padding: 1.5rem 1.75rem;
            color: #f7fbff;
            box-shadow: 0 12px 32px rgba(15, 39, 64, 0.12);
            margin-bottom: 1rem;
        }
        .hero-card h1 {
            margin: 0 0 0.4rem 0;
            font-size: 2rem;
            font-weight: 700;
        }
        .hero-card p {
            margin: 0;
            line-height: 1.6;
            font-size: 1rem;
        }
        .recommendation-card {
            background: linear-gradient(135deg, #ffffff 0%, #eef6ff 100%);
            border-left: 6px solid #2563eb;
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 10px 25px rgba(37, 99, 235, 0.08);
            margin-bottom: 1rem;
        }
        .metric-caption {
            color: #51667d;
            font-size: 0.94rem;
            margin-top: 0.35rem;
        }
        .formula-box {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(20, 61, 99, 0.12);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin-top: 1rem;
        }
        .stButton button, .stDownloadButton button {
            border-radius: 12px;
            border: 0;
        }
        .stButton button {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: #ffffff;
            font-weight: 700;
            width: 100%;
            min-height: 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def validate_csv(dataframe: pd.DataFrame) -> tuple[bool, str]:
    required_columns = {"Date", "USD/RWF", "EUR/RWF"}
    if not required_columns.issubset(set(dataframe.columns)):
        return False, "CSV must contain columns: Date, USD/RWF, EUR/RWF."
    return True, ""


def derive_parameters_from_history(rate_series: pd.Series) -> tuple[float, float, float]:
    clean_series = pd.to_numeric(rate_series, errors="coerce").dropna()
    if clean_series.size < 2:
        raise ValueError("Uploaded rate column must have at least two valid rows.")

    daily_changes = clean_series.diff().dropna()
    mu = float(daily_changes.mean())
    sigma = float(daily_changes.std(ddof=1)) if daily_changes.size > 1 else 0.0
    starting_rate = float(clean_series.iloc[-1])
    return starting_rate, mu, sigma


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
                "Strategy": code,
                "Name": STRATEGY_LABELS[code],
                "Average Return (RWF)": average_return,
                "Risk - Std Dev (RWF)": risk_std,
                "Total Profit (RWF)": total_profit,
                "Risk-Adjusted Score": risk_adjusted_score,
            }
        )

    return pd.DataFrame(results)


def format_rwf(value: float) -> str:
    return f"{value:,.2f} RWF"


def build_downloadable_results(summary_df: pd.DataFrame) -> str:
    output = StringIO()
    summary_df.to_csv(output, index=False)
    return output.getvalue()


def store_results(payload: dict) -> None:
    st.session_state["simulation_results"] = payload


def get_best_strategy(summary_df: pd.DataFrame) -> pd.Series:
    ranked = summary_df.sort_values(
        by=["Risk-Adjusted Score", "Average Return (RWF)"],
        ascending=[False, False],
    )
    return ranked.iloc[0]


with st.sidebar:
    st.markdown("## Simulation Inputs")
    st.caption("Choose either a historical CSV file or manual model parameters.")

    with st.form("simulation_form"):
        input_mode = st.radio(
            "Input method",
            options=["Upload CSV", "Manual entry"],
            help="Use a CSV to estimate mu and sigma from historical data, or enter them manually.",
        )

        uploaded_df = None
        currency_column = "USD/RWF"
        starting_rate = DEFAULTS["starting_rate"]
        mu = DEFAULTS["mu"]
        sigma = DEFAULTS["sigma"]

        if input_mode == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload exchange-rate CSV",
                type=["csv"],
                help="Expected columns: Date, USD/RWF, EUR/RWF",
            )
            if uploaded_file is not None:
                uploaded_df = pd.read_csv(uploaded_file)
                is_valid, message = validate_csv(uploaded_df)
                if not is_valid:
                    st.error(message)
                else:
                    currency_column = st.selectbox(
                        "Choose currency column",
                        options=["USD/RWF", "EUR/RWF"],
                    )
                    try:
                        starting_rate, mu, sigma = derive_parameters_from_history(
                            uploaded_df[currency_column]
                        )
                        st.success("Historical parameters calculated successfully.")
                        st.write(f"Starting rate: {starting_rate:,.2f}")
                        st.write(f"mu: {mu:,.2f}")
                        st.write(f"sigma: {sigma:,.2f}")
                    except ValueError as exc:
                        st.error(str(exc))
        else:
            starting_rate = st.number_input(
                "Starting rate (RWF per currency unit)",
                min_value=0.01,
                value=DEFAULTS["starting_rate"],
                step=1.0,
                help="Initial exchange rate S(0) used to start every simulated path.",
            )
            mu = st.number_input(
                "mu - average daily change",
                value=DEFAULTS["mu"],
                step=0.5,
                help="Mean daily change added in the random walk model.",
            )
            sigma = st.number_input(
                "sigma - volatility",
                min_value=0.0,
                value=DEFAULTS["sigma"],
                step=0.5,
                help="Standard deviation of daily changes in the random walk model.",
            )

        days = st.number_input(
            "Days to simulate",
            min_value=7,
            max_value=365,
            value=DEFAULTS["days"],
            step=1,
        )
        n_simulations = st.number_input(
            "Monte Carlo simulations",
            min_value=1000,
            max_value=10000,
            value=DEFAULTS["n_simulations"],
            step=100,
        )
        initial_capital = st.number_input(
            "Initial capital (RWF)",
            min_value=1000.0,
            value=DEFAULTS["initial_capital"],
            step=1000.0,
        )

        st.markdown("### Strategy Controls")
        buy_threshold_percent = st.number_input(
            "Buy threshold (%)",
            value=DEFAULTS["buy_threshold"] * 100,
            step=0.1,
            help="Strategy B buys when the daily rate change is at or below this percentage.",
        )
        sell_threshold_percent = st.number_input(
            "Sell threshold (%)",
            value=DEFAULTS["sell_threshold"] * 100,
            step=0.1,
            help="Strategy B sells when the held position reaches this gain.",
        )
        trend_lookback = st.number_input(
            "Trend lookback days",
            min_value=2,
            max_value=10,
            value=DEFAULTS["trend_lookback"],
            step=1,
            help="Strategy C checks whether prices move in the same direction for this many days.",
        )

        run_clicked = st.form_submit_button("Run Simulation")


st.markdown(
    """
    <div class="hero-card">
        <h1>Stochastic Simulation of Currency Exchange Rates for Profit Optimization</h1>
        <p>
            A System Modeling and Simulation project for analyzing USD/RWF and EUR/RWF rate uncertainty
            with a stochastic random walk model and Monte Carlo simulation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_intro, col_formula = st.columns([1.4, 1])

with col_intro:
    st.write(
        """
        This web application simulates future exchange-rate movements, tests three trading strategies,
        and compares expected profit against risk so students can identify the best decision rule.
        """
    )

with col_formula:
    st.markdown(
        """
        <div class="formula-box">
            <strong>Random walk model</strong><br>
            S(t+1) = S(t) + mu + sigma x Z(t)<br><br>
            <strong>Monte Carlo simulation</strong><br>
            1000 or more simulated future paths are generated for comparison.
        </div>
        """,
        unsafe_allow_html=True,
    )


if run_clicked:
    if input_mode == "Upload CSV" and uploaded_df is None:
        st.error("Please upload a CSV file before running the simulation.")
    elif input_mode == "Upload CSV" and uploaded_df is not None:
        is_valid, message = validate_csv(uploaded_df)
        if not is_valid:
            st.error(message)
        else:
            with st.spinner("Running Monte Carlo simulation and evaluating strategies..."):
                paths = monte_carlo_simulation(
                    S0=starting_rate,
                    mu=mu,
                    sigma=sigma,
                    days=int(days),
                    n_simulations=int(n_simulations),
                )
                summary_df = run_all_strategies(
                    paths=paths,
                    initial_capital=initial_capital,
                    buy_threshold=buy_threshold_percent / 100,
                    sell_threshold=sell_threshold_percent / 100,
                    trend_lookback=int(trend_lookback),
                )
                store_results(
                    {
                        "summary_df": summary_df,
                        "paths": paths,
                        "currency_column": currency_column,
                        "input_mode": input_mode,
                        "starting_rate": starting_rate,
                        "mu": mu,
                        "sigma": sigma,
                        "days": days,
                    }
                )
    else:
        with st.spinner("Running Monte Carlo simulation and evaluating strategies..."):
            paths = monte_carlo_simulation(
                S0=starting_rate,
                mu=mu,
                sigma=sigma,
                days=int(days),
                n_simulations=int(n_simulations),
            )
            summary_df = run_all_strategies(
                paths=paths,
                initial_capital=initial_capital,
                buy_threshold=buy_threshold_percent / 100,
                sell_threshold=sell_threshold_percent / 100,
                trend_lookback=int(trend_lookback),
            )
            store_results(
                {
                    "summary_df": summary_df,
                    "paths": paths,
                    "currency_column": currency_column,
                    "input_mode": input_mode,
                    "starting_rate": starting_rate,
                    "mu": mu,
                    "sigma": sigma,
                    "days": days,
                }
            )


if "simulation_results" in st.session_state:
    summary_df = st.session_state["simulation_results"]["summary_df"].copy()
    best_strategy = get_best_strategy(summary_df)

    st.markdown(
        f"""
        <div class="recommendation-card">
            <h3 style="margin: 0 0 0.4rem 0; color: #143d63;">Best Strategy Recommendation</h3>
            <div style="font-size: 1.2rem; font-weight: 700; color: #1d4ed8;">
                {best_strategy["Strategy"]} - {best_strategy["Name"]}
            </div>
            <div class="metric-caption">
                Expected profit: {format_rwf(best_strategy["Average Return (RWF)"])} |
                Risk (Std Dev): {format_rwf(best_strategy["Risk - Std Dev (RWF)"])} |
                Total profit across simulations: {format_rwf(best_strategy["Total Profit (RWF)"])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Input Mode", st.session_state["simulation_results"]["input_mode"])
    metric_col2.metric("Starting Rate", format_rwf(st.session_state["simulation_results"]["starting_rate"]))
    metric_col3.metric("Simulated Days", int(st.session_state["simulation_results"]["days"]))

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Profit Comparison")
        profit_chart = (
            summary_df.set_index("Name")[["Average Return (RWF)"]]
            .rename(columns={"Average Return (RWF)": "Average Profit"})
        )
        st.bar_chart(profit_chart)

    with chart_col2:
        st.subheader("Risk Comparison")
        risk_chart = (
            summary_df.set_index("Name")[["Risk - Std Dev (RWF)"]]
            .rename(columns={"Risk - Std Dev (RWF)": "Risk"})
        )
        st.bar_chart(risk_chart)

    st.subheader("Summary Table")
    display_df = summary_df.copy()
    for column in [
        "Average Return (RWF)",
        "Risk - Std Dev (RWF)",
        "Total Profit (RWF)",
        "Risk-Adjusted Score",
    ]:
        display_df[column] = display_df[column].map(lambda value: round(float(value), 2))
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    sample_paths = st.session_state["simulation_results"]["paths"][:15]
    sample_path_df = pd.DataFrame(sample_paths.T)
    sample_path_df.index.name = "Day"

    st.subheader("Sample Exchange Rate Paths")
    st.line_chart(sample_path_df)

    st.download_button(
        "Download Summary CSV",
        data=build_downloadable_results(summary_df),
        file_name="simulation_results_summary.csv",
        mime="text/csv",
    )
else:
    st.info("Run the simulation from the sidebar to generate strategy comparisons and charts.")
