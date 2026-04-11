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


DEMO_CREDENTIALS = {
    "username": "student",
    "password": "demo123",
    "display_name": "Student Demo",
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
                radial-gradient(circle at top right, rgba(249, 115, 22, 0.18), transparent 24%),
                radial-gradient(circle at bottom left, rgba(37, 99, 235, 0.16), transparent 28%),
                linear-gradient(180deg, #05070d 0%, #0b1020 52%, #101728 100%);
            color: #eef2ff;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #070b14 0%, #0f1728 100%);
        }
        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }
        .block-container {
            padding-top: 1.3rem;
        }
        h1, h2, h3, h4, p, li, label, span {
            color: #eef2ff;
        }
        .hero-card {
            background:
                radial-gradient(circle at top right, rgba(249, 115, 22, 0.24), transparent 24%),
                linear-gradient(135deg, rgba(9, 14, 28, 0.96) 0%, rgba(18, 24, 41, 0.96) 68%, rgba(28, 20, 38, 0.96) 100%);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 24px;
            padding: 1.6rem 1.8rem;
            color: #f8fafc;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35);
            margin-bottom: 1rem;
        }
        .hero-card h1 {
            margin: 0 0 0.4rem 0;
            font-size: 2.15rem;
            font-weight: 700;
        }
        .hero-card p {
            margin: 0;
            line-height: 1.6;
            font-size: 1rem;
            color: #cbd5e1;
        }
        .recommendation-card {
            background: linear-gradient(145deg, rgba(15, 23, 42, 0.96) 0%, rgba(18, 37, 63, 0.96) 100%);
            border: 1px solid rgba(59, 130, 246, 0.25);
            border-radius: 22px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 14px 32px rgba(2, 6, 23, 0.28);
            margin-bottom: 1rem;
        }
        .metric-caption {
            color: #cbd5e1;
            font-size: 0.94rem;
            margin-top: 0.35rem;
        }
        .formula-box {
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(59, 130, 246, 0.16);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            margin-top: 1rem;
            color: #e2e8f0;
        }
        .guide-card {
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            min-height: 180px;
            box-shadow: 0 10px 24px rgba(2, 6, 23, 0.22);
        }
        .guide-card h4 {
            margin: 0 0 0.5rem 0;
            color: #f8fafc;
        }
        .guide-card p {
            margin: 0;
            color: #cbd5e1;
            line-height: 1.6;
        }
        .auth-shell {
            background: linear-gradient(145deg, rgba(8, 12, 22, 0.94), rgba(19, 28, 48, 0.92));
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 22px;
            padding: 1.4rem 1.5rem;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
        }
        .auth-note {
            background: rgba(37, 99, 235, 0.12);
            border-left: 4px solid #f97316;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            color: #e2e8f0;
            margin-bottom: 1rem;
        }
        .accent-panel {
            background: linear-gradient(135deg, rgba(249, 115, 22, 0.16), rgba(59, 130, 246, 0.12));
            border: 1px solid rgba(249, 115, 22, 0.16);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 22px rgba(0, 0, 0, 0.18);
        }
        .market-chip {
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
        }
        .market-chip h5 {
            margin: 0 0 0.3rem 0;
            color: #f8fafc;
            font-size: 1rem;
        }
        .market-chip p {
            margin: 0;
            color: #cbd5e1;
            line-height: 1.5;
        }
        .objective-item {
            padding: 0.65rem 0.8rem;
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.74);
            border: 1px solid rgba(148, 163, 184, 0.10);
            margin-bottom: 0.55rem;
            color: #e2e8f0;
        }
        [data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.86);
            border: 1px solid rgba(148, 163, 184, 0.12);
            padding: 1rem;
            border-radius: 18px;
        }
        [data-testid="stMetricLabel"] {
            color: #94a3b8;
        }
        [data-testid="stMetricValue"] {
            color: #f8fafc;
        }
        [data-testid="stDataFrame"], .stTable {
            background: rgba(15, 23, 42, 0.74);
            border-radius: 18px;
        }
        .stButton button, .stDownloadButton button {
            border-radius: 14px;
            border: 0;
        }
        .stButton button {
            background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
            color: #ffffff;
            font-weight: 700;
            width: 100%;
            min-height: 3rem;
        }
        .stDownloadButton button {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: #ffffff;
            min-height: 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    st.session_state.setdefault("uploader_reset_key", 0)
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("current_user", None)
    st.session_state.setdefault("auth_notice", "")


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


def extract_rate_series(
    dataframe: pd.DataFrame,
    rate_column: str,
    selected_pair: str | None = None,
) -> pd.Series:
    return extract_history_frame(dataframe, rate_column, selected_pair)["Rate"]


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
                "Strategy": "A - Buy & Hold",
                "When to Buy": "Buy immediately at the starting exchange rate.",
                "When to Sell": f"Sell after {simulated_days} simulated days.",
                "Simple Meaning": "Best when you expect a steady rise over the whole period.",
            },
            {
                "Strategy": "B - Threshold",
                "When to Buy": "Buy only after the rate drops by 1% or more.",
                "When to Sell": "Sell after the bought rate rises by 1.5% or more.",
                "Simple Meaning": "Best when prices move up and down and you want disciplined entry and exit points.",
            },
            {
                "Strategy": "C - Trend-following",
                "When to Buy": "Buy after 3 straight days of upward movement.",
                "When to Sell": "Sell after 3 straight days of downward movement.",
                "Simple Meaning": "Best when the market shows short-term momentum trends.",
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


def reset_analysis() -> None:
    st.session_state.pop("simulation_results", None)
    st.session_state["uploader_reset_key"] = st.session_state.get("uploader_reset_key", 0) + 1


def logout_demo_user() -> None:
    reset_analysis()
    st.session_state["logged_in"] = False
    st.session_state["current_user"] = None
    st.session_state["auth_notice"] = ""


def get_best_strategy(summary_df: pd.DataFrame) -> pd.Series:
    ranked = summary_df.sort_values(
        by=["Risk-Adjusted Score", "Average Return (RWF)"],
        ascending=[False, False],
    )
    return ranked.iloc[0]


def explain_best_strategy(best_strategy: pd.Series) -> str:
    if best_strategy["Strategy"] == "Strategy A":
        return "This means the model expects a mostly steady increase, so buying once and holding gives the best result."
    if best_strategy["Strategy"] == "Strategy B":
        return "This means the model favors waiting for a cheaper buying point and taking profit after a clear rebound."
    return "This means the model favors following short upward and downward trends instead of trading immediately."


def render_auth_screen() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1>Stochastic Simulation of Currency Exchange Rates</h1>
            <p>
                Sign in to explore the project, upload currency history, simulate future exchange-rate paths,
                and present the best trading strategy in a clear classroom-friendly way.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.15, 0.85])

    with left_col:
        st.markdown(
            """
            <div class="auth-shell">
                <div class="auth-note">
                    This login and sign-up section is a presentation demo for class use only.
                    No real accounts are stored and no external authentication service is used.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        auth_tabs = st.tabs(["Login", "Sign Up", "About This App"])

        with auth_tabs[0]:
            with st.form("demo_login_form"):
                username = st.text_input("Username", placeholder="student")
                password = st.text_input("Password", type="password", placeholder="demo123")
                login_clicked = st.form_submit_button("Login to Demo")

            st.caption("Sample credentials: `student` / `demo123`")

            if login_clicked:
                if username == DEMO_CREDENTIALS["username"] and password == DEMO_CREDENTIALS["password"]:
                    st.session_state["logged_in"] = True
                    st.session_state["current_user"] = DEMO_CREDENTIALS["display_name"]
                    st.session_state["auth_notice"] = "Demo login successful."
                    st.rerun()
                st.error("Use the sample credentials shown above.")

        with auth_tabs[1]:
            with st.form("demo_signup_form"):
                full_name = st.text_input("Full name", placeholder="Your name")
                email = st.text_input("Email", placeholder="student@example.com")
                new_password = st.text_input("Create password", type="password", placeholder="any password")
                signup_clicked = st.form_submit_button("Create Demo Account")

            st.caption("This sign-up is only a demo screen for presentation. Nothing is stored online.")

            if signup_clicked:
                if full_name.strip() and email.strip() and new_password.strip():
                    st.session_state["logged_in"] = True
                    st.session_state["current_user"] = full_name.strip()
                    st.session_state["auth_notice"] = "Demo account created successfully."
                    st.rerun()
                st.error("Please fill in all fields to continue.")

        with auth_tabs[2]:
            st.write(
                """
                This app uses historical exchange-rate data to estimate the random-walk model,
                simulate many possible future paths, compare three trading strategies, and explain
                which strategy gives the best balance of profit and risk.
                """
            )
            st.write(
                """
                It is designed for a System Modeling and Simulation class presentation, so the layout
                focuses on clear charts, plain-language explanations, and easy reruns with new datasets.
                """
            )

    with right_col:
        st.markdown(
            """
            <div class="guide-card">
                <h4>What Happens After Login</h4>
                <p>
                    1. Upload exchange-rate history or enter values manually.<br>
                    2. The app estimates the model parameters.<br>
                    3. It simulates future exchange-rate paths.<br>
                    4. It compares buy and sell strategies and recommends the best one.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="guide-card">
                <h4>Why This App Is Easy To Follow</h4>
                <p>
                    Every section explains what it does, the charts include plain-language captions,
                    and the results area shows how each strategy buys and sells so viewers do not get confused.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.stop()


initialize_state()

if not st.session_state["logged_in"]:
    render_auth_screen()


with st.sidebar:
    st.markdown("## Demo User")
    st.caption(f"Signed in as: {st.session_state['current_user']}")
    if st.button("Logout", use_container_width=True):
        logout_demo_user()
        st.rerun()

    st.markdown("### Quick Guide")
    st.caption("1. Choose upload or manual mode.")
    st.caption("2. Run the simulation.")
    st.caption("3. Read the recommendation and charts.")
    st.caption("4. Click New Analysis to start again.")

    st.markdown("## Simulation Inputs")
    st.caption("Choose either a historical CSV file or manual model parameters.")

    with st.form("simulation_form"):
        input_mode = st.radio(
            "Input method",
            options=["Upload CSV", "Manual entry"],
            help="Use a CSV to estimate mu and sigma from historical data, or enter them manually.",
        )

        uploaded_df = None
        currency_column = "Close"
        selected_pair = None
        starting_rate = DEFAULTS["starting_rate"]
        mu = DEFAULTS["mu"]
        sigma = DEFAULTS["sigma"]

        if input_mode == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload exchange-rate CSV",
                type=["csv"],
                key=f"uploaded_csv_{st.session_state['uploader_reset_key']}",
                help="Supported formats: Date + numeric rate columns, or Kaggle-style forex files with currency_pair and Close/High/Low/Open.",
            )
            if uploaded_file is not None:
                uploaded_df = pd.read_csv(uploaded_file)
                is_valid, message = validate_csv(uploaded_df)
                if not is_valid:
                    st.error(message)
                else:
                    pair_options = get_currency_pair_options(uploaded_df)
                    rate_options = get_rate_columns(uploaded_df)
                    auto_detect = st.toggle(
                        "Use automatic dataset selection",
                        value=True,
                        help="Keep this on to let the app choose a default currency pair and rate column automatically. Turn it off if you want to choose another option yourself.",
                    )

                    if auto_detect:
                        selected_pair = choose_default_pair(uploaded_df)
                        currency_column = choose_default_rate_column(uploaded_df)
                    else:
                        if pair_options:
                            selected_pair = st.selectbox(
                                "Choose currency pair",
                                options=pair_options,
                                index=0,
                            )
                        currency_column = st.selectbox(
                            "Choose rate column",
                            options=rate_options,
                            index=0,
                        )

                    try:
                        starting_rate, mu, sigma = derive_parameters_from_history(
                            extract_rate_series(uploaded_df, currency_column, selected_pair)
                        )
                        st.success("Historical parameters calculated successfully.")
                        if selected_pair:
                            label = "Auto-detected pair" if auto_detect else "Selected pair"
                            st.write(f"{label}: {selected_pair}")
                        rate_label = "Auto-detected rate column" if auto_detect else "Selected rate column"
                        st.write(f"{rate_label}: {currency_column}")
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

    if "simulation_results" in st.session_state:
        st.markdown("### Start Again")
        st.caption("Clear the current file and results, then begin a new prediction.")
        if st.button("New Analysis", use_container_width=True):
            reset_analysis()
            st.rerun()


st.markdown(
    """
    <div class="hero-card">
        <h1>Rwanda Forex Strategy Lab</h1>
        <p>
            A market-style simulation dashboard for studying USD/RWF and EUR/RWF uncertainty with
            stochastic random walk modeling, Monte Carlo forecasting, and strategy advice for local forex traders.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.get("auth_notice"):
    st.success(st.session_state["auth_notice"])
    st.session_state["auth_notice"] = ""

col_intro, col_formula = st.columns([1.4, 1])

with col_intro:
    st.write(
        """
        Upload historical exchange-rate data and the app will estimate the model, recreate how the market
        has behaved, simulate many possible future paths, and turn those results into easy buy-and-sell advice.
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

st.subheader("Project Objectives")
objective_col1, objective_col2 = st.columns(2)

with objective_col1:
    st.markdown(
        """
        <div class="objective-item"><strong>1.</strong> Develop and implement a stochastic random walk model to replicate daily USD/RWF and EUR/RWF fluctuations.</div>
        <div class="objective-item"><strong>2.</strong> Compare three practical strategies under simulated market conditions: Buy & Hold, Threshold trading, and Trend-following.</div>
        <div class="objective-item"><strong>3.</strong> Identify the strategy that gives the highest expected profit with the lowest level of risk.</div>
        """,
        unsafe_allow_html=True,
    )

with objective_col2:
    st.markdown(
        """
        <div class="objective-item"><strong>4.</strong> Estimate useful buy and sell levels that can improve returns while reducing possible losses.</div>
        <div class="objective-item"><strong>5.</strong> Provide data-driven recommendations that are easy for local forex traders in Rwanda to understand and apply.</div>
        """,
        unsafe_allow_html=True,
    )

st.subheader("How This App Works")
guide_col1, guide_col2, guide_col3 = st.columns(3)

with guide_col1:
    st.markdown(
        """
        <div class="guide-card">
            <h4>Input Area</h4>
            <p>
                The sidebar is where the user uploads exchange-rate history or enters values manually.
                This section prepares the data that will be used for prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with guide_col2:
    st.markdown(
        """
        <div class="guide-card">
            <h4>Simulation Engine</h4>
            <p>
                The app calculates mu and sigma from the uploaded history, then applies the random walk
                formula to create many future exchange-rate paths with Monte Carlo simulation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with guide_col3:
    st.markdown(
        """
        <div class="guide-card">
            <h4>Results Area</h4>
            <p>
                The main panel explains which strategy is best, how each strategy buys and sells,
                and how profit and risk compare in an easy-to-understand format.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("Read This Before Running the App"):
    st.write(
        """
        `mu` is the average daily change seen in the historical data.
        `sigma` measures how much the exchange rate usually moves up and down.
        The profit chart answers: which strategy earns more?
        The risk chart answers: which strategy is more stable?
        The summary table answers: what are the exact numbers behind the recommendation?
        """
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
                history_df = extract_history_frame(uploaded_df, currency_column, selected_pair)
                store_results(
                    {
                        "summary_df": summary_df,
                        "paths": paths,
                        "history_df": history_df,
                        "currency_column": currency_column,
                        "selected_pair": selected_pair,
                        "input_mode": input_mode,
                        "starting_rate": starting_rate,
                        "mu": mu,
                        "sigma": sigma,
                        "days": days,
                        "buy_threshold": buy_threshold_percent / 100,
                        "sell_threshold": sell_threshold_percent / 100,
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
                    "history_df": None,
                    "currency_column": currency_column,
                    "selected_pair": selected_pair,
                    "input_mode": input_mode,
                    "starting_rate": starting_rate,
                    "mu": mu,
                    "sigma": sigma,
                    "days": days,
                    "buy_threshold": buy_threshold_percent / 100,
                    "sell_threshold": sell_threshold_percent / 100,
                }
            )


if "simulation_results" in st.session_state:
    results_payload = st.session_state["simulation_results"]
    summary_df = results_payload["summary_df"].copy()
    best_strategy = get_best_strategy(summary_df)
    chosen_dataset = (
        results_payload.get("selected_pair")
        or results_payload.get("currency_column")
    )
    history_df = results_payload.get("history_df")
    weekday_summary = build_weekday_market_summary(history_df) if isinstance(history_df, pd.DataFrame) else pd.DataFrame()
    suggested_buy, suggested_sell, level_explanation = build_reference_levels(
        best_strategy["Strategy"],
        results_payload["paths"],
        float(results_payload["starting_rate"]),
        float(results_payload.get("buy_threshold", DEFAULTS["buy_threshold"])),
        float(results_payload.get("sell_threshold", DEFAULTS["sell_threshold"])),
    )

    st.markdown(
        f"""
        <div class="recommendation-card">
            <h3 style="margin: 0 0 0.4rem 0; color: #f8fafc;">Best Strategy Recommendation</h3>
            <div style="font-size: 1.2rem; font-weight: 700; color: #60a5fa;">
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

    st.write(
        f"""
        **Simple explanation:** Based on the uploaded history for **{chosen_dataset}**, the model suggests
        **{best_strategy["Name"]}** as the clearest balance between expected profit and risk.
        {explain_best_strategy(best_strategy)}
        """
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Input Mode", results_payload["input_mode"])
    metric_col2.metric("Starting Rate", format_rwf(results_payload["starting_rate"]))
    metric_col3.metric("Dataset Selection", chosen_dataset)
    metric_col4.metric("Suggested Buy Level", format_rwf(suggested_buy))

    st.markdown(
        f"""
        <div class="accent-panel">
            <strong>Strategy Advice</strong><br>
            Suggested sell level: <strong>{format_rwf(suggested_sell)}</strong><br>
            {level_explanation}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if isinstance(history_df, pd.DataFrame) and not history_df.empty:
        st.subheader("Historical Market Pulse")
        st.caption("This section shows how the uploaded market history has behaved before the simulation starts.")

        recent_history = history_df.tail(14).copy()
        recent_history = recent_history.set_index("Date")
        pulse_col1, pulse_col2 = st.columns([1.2, 1])

        with pulse_col1:
            st.caption("Recent price movement from the uploaded history.")
            st.line_chart(recent_history[["Rate"]])

        with pulse_col2:
            st.caption("How the market usually behaved on each weekday.")
            weekday_cols = st.columns(min(5, max(1, len(weekday_summary))))
            for index, (_, row) in enumerate(weekday_summary.head(5).iterrows()):
                with weekday_cols[index]:
                    mood_color = "#22c55e" if row["Average_Change"] >= 0 else "#f97316"
                    st.markdown(
                        f"""
                        <div class="market-chip">
                            <h5>{row["Weekday"]}</h5>
                            <p>Avg rate: {row["Average_Rate"]:.3f}</p>
                            <p style="color:{mood_color};">Avg change: {row["Average_Change"]:.2f}%</p>
                            <p>{row["Market Mood"]}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.subheader("Weekday Market Summary")
        st.caption("This table helps explain whether the market tended to be stronger or weaker on Monday, Tuesday, and the other trading days.")
        display_weekday = weekday_summary.copy()
        for column in ["Average_Rate", "Average_Change"]:
            display_weekday[column] = display_weekday[column].map(lambda value: round(float(value), 3))
        st.dataframe(display_weekday, use_container_width=True, hide_index=True)

    st.subheader("How the Buy and Sell Rules Work")
    st.caption("This table explains exactly when each strategy enters and exits the market.")
    st.dataframe(
        build_strategy_rules_table(int(results_payload["days"])),
        use_container_width=True,
        hide_index=True,
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Profit Comparison")
        st.caption("Higher bars are better because they mean higher expected profit.")
        profit_chart = (
            summary_df.set_index("Name")[["Average Return (RWF)"]]
            .rename(columns={"Average Return (RWF)": "Average Profit"})
        )
        st.bar_chart(profit_chart)

    with chart_col2:
        st.subheader("Risk Comparison")
        st.caption("Lower bars are better because they mean less uncertainty and lower risk.")
        risk_chart = (
            summary_df.set_index("Name")[["Risk - Std Dev (RWF)"]]
            .rename(columns={"Risk - Std Dev (RWF)": "Risk"})
        )
        st.bar_chart(risk_chart)

    st.subheader("Summary Table")
    st.caption(
        "Average Return shows expected profit per simulation, Risk shows variability, and Total Profit sums all simulated profits."
    )
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
    st.caption(
        "Each line is one possible future path from the Monte Carlo simulation. An upward line means the exchange rate may rise, while a downward line means it may fall."
    )
    st.line_chart(sample_path_df)

    st.download_button(
        "Download Summary CSV",
        data=build_downloadable_results(summary_df),
        file_name="simulation_results_summary.csv",
        mime="text/csv",
    )
else:
    st.info("Run the simulation from the sidebar to generate strategy comparisons and charts.")
