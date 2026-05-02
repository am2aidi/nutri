Project overview: This repository contains the Flask web app and simulation logic used for the Vercel deployment of Simulation.

Stochastic Simulation of Currency Exchange Rates for Profit Optimization

Files
- app.py: Flask web application entrypoint
- app_logic.py: Shared data preparation and strategy summary helpers
- simulation.py: Random walk model, trading strategies, and metrics
- requirements.txt: Python dependencies
- templates/index.html: Browser interface for the deployed app
- sample_exchange_rates.csv: Example dataset for testing

How to run locally
1. Install dependencies:
   pip install -r requirements.txt
2. Start the application:
   flask --app app run

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
- Upload the CSV or enter parameters manually
- Click "Run Simulation"
