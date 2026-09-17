# CryptoInsight AI

A cryptocurrency analytics dashboard built with Streamlit for analyzing historical market data and predicting next-day price direction using a Random Forest classifier.

## Project purpose

This project uses the dataset in the workspace, `crypto_historical_365days.csv`, to provide:

- a market overview dashboard
- exploratory analysis of coin trends and volatility
- coin-by-coin comparison views
- next-day price movement prediction
- model evaluation metrics
- data quality checks

## Dataset source

The crypto dataset used in this project was downloaded from Kaggle:

https://www.kaggle.com/datasets/mihikaajayjadhav/top-100-cryptocurrencies-daily-price-data-2025

This project uses the dataset as-is in the workspace and does not modify the source data beyond the feature engineering used in the app.

## Key methodology

The model is trained on engineered features such as:

- price level
- market cap and volume
- daily return
- moving averages (7-day and 30-day)
- volatility
- cumulative return
- date-derived features such as day-of-week and month

To reduce leakage risk, the pipeline uses a chronological split based on date rather than random sampling. The target is defined as whether the next-day return is positive, and the model is trained only on historical data before the test period.

## Project structure

- `app.py` — the full Streamlit application and analytics pipeline
- `crypto_historical_365days.csv` — market data used for analysis and modeling
- `requirements.txt` — Python dependencies
- `README.md` — setup and usage guide

## Setup

1. Open a terminal in the project folder.
2. Create and activate a virtual environment if desired.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Then open the localhost URL shown in the terminal.

## Dashboard pages

- Overview
- EDA
- Comparison
- AI Prediction
- Model Evaluation
- Data Quality

## Frontend Pages

| Page | Description |
| --- | --- |
| **Overview** | Market summary with coin counts, date range, average daily return, and top cumulative performers |
| **EDA** | Explore price history, daily returns, return distributions, and numerical correlations for a selected coin |
| **Comparison** | Compare coins by average return, volatility, market capitalization, or total return |
| **AI Prediction** | Predict whether a coin's next-day price movement is likely to be up or down, with the model probability |
| **Model Evaluation** | Review accuracy, precision, recall, F1, ROC AUC, confusion matrix, and feature importance |
| **Data Quality** | Inspect missing values, duplicate rows, invalid dates, constant columns, and null patterns |

---

## Model Performance (runtime)

The app calculates these metrics on the chronological test split each time the model is prepared. Values are displayed on the **Model Evaluation** page and are not hardcoded.

| Metric | Description |
| --- | --- |
| Accuracy | Overall proportion of correct up/down predictions |
| Precision | Proportion of predicted upward movements that were correct |
| Recall | Proportion of actual upward movements identified by the model |
| F1 | Harmonic mean of precision and recall |
| ROC AUC | Ranking quality for the probability of an upward movement |

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| ML | scikit-learn Random Forest, numpy |
| Frontend | Streamlit, Plotly |
| Data analysis | pandas, matplotlib, seaborn |
| Data source | Historical cryptocurrency CSV dataset |

## Model notes

- Model type: `RandomForestClassifier`
- Target: next-day price movement (`up` vs `down`)
- Split method: chronological date-based train/test split
- Leakage prevention: no target leakage from future data, no random shuffle across dates

## Notes

- The app reads the actual CSV file present in the workspace.
- No fake metrics or hardcoded results are included.
- The project intentionally uses one Python file for the full frontend and backend logic.

## Recommended workflow

1. Launch the dashboard.
2. Review the overview page for market context.
3. Use EDA to inspect individual coin performance.
4. Compare assets on the comparison page.
5. Use the AI Prediction page to estimate tomorrow's direction for a selected coin using the latest available data.
6. Review model evaluation and data quality before drawing conclusions.
# Data-analytics-with-AI
