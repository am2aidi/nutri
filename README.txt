Stochastic Simulation of Currency Exchange Rates for Profit Optimization

Files
- app.py: Streamlit web application
- simulation.py: Random walk model, trading strategies, and metrics
- requirements.txt: Python dependencies
- sample_exchange_rates.csv: Example dataset for testing

How to run locally
1. Install dependencies:
   pip install -r requirements.txt
2. Start the application:
   streamlit run app.py

CSV format
The upload file must contain these columns:
- Date
- USD/RWF
- EUR/RWF

What the app does
- Simulates future exchange rates with the required random walk equation:
  S(t+1) = S(t) + mu + sigma x Z(t)
- Runs Monte Carlo simulation with 1000 or more paths
- Compares three strategies:
  Strategy A: Buy & Hold
  Strategy B: Threshold
  Strategy C: Trend-following
- Reports average return, risk (standard deviation), and total profit

Recommended sample test
- Start with the provided sample_exchange_rates.csv file
- Choose either USD/RWF or EUR/RWF after upload
- Click "Run Simulation"
