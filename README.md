# USD/LKR Exchange-Rate Forecasting

Forecasting the next published USD/LKR indicative exchange-rate observation under changing economic regimes.

## Project overview

This project develops and evaluates traditional time-series, machine-learning, and volatility models using 3,499 USD/LKR observations from **7 May 2010 to 25 September 2026**. The analysis focuses on the difficulty of forecasting an exchange rate that includes persistence, irregular observation gaps, changing volatility, and the major regime change around March 2022.

The complete workflow covers:

- data validation and cleaning;
- exchange-rate level and log-return analysis;
- distributional and stationarity tests;
- ACF, PACF, Ljung–Box, and ARCH diagnostics;
- economic-period and structural-break analysis;
- leakage-aware lag, rolling, and calendar-gap features;
- chronological validation and testing;
- one-step and multi-horizon forecasting;
- separate GARCH volatility evaluation;
- a 2022 historical stress test; and
- final model refitting and forecasting.

## Research question

> How effectively can the next published USD/LKR observation be forecast using traditional time-series and machine-learning methods, and how does performance change across horizons and economic regimes?

The target is the **next available observation**, not a rate assigned to an invented calendar date. Lags and forecast horizons are measured in observations, because the source series contains weekends, holidays, and longer data gaps.

## Main findings

The primary one-step comparison uses observations from 2014 onward where the next observation is no more than five calendar days away. The 3,067 eligible observations are split chronologically into training, validation, and test sets:

| Partition | Observations |
|---|---:|
| Training | 2,146 |
| Validation | 460 |
| Test | 461 |

ARIMA(1,1,1) is selected using validation MAE and achieves the lowest test MAE:

| Model | MAE | RMSE |
|---|---:|---:|
| **ARIMA(1,1,1)** | **0.406721** | **0.974414** |
| Naive | 0.424284 | 0.947432 |
| XGBoost-return | 0.448500 | 1.100929 |
| LSTM-return | 0.511477 | 0.982958 |
| MLP-return | 16.035584 | 19.975001 |

GARCH is evaluated separately as a volatility model rather than being ranked with level forecasts:

- Test volatility MSE: **0.411381**
- Test QLIKE: **-2.46633**

The validation-selected ARIMA(1,1,1), refit to the full available series, produces:

- **Forecast:** 330.3869
- **Approximate 95% prediction interval:** 327.8478 to 332.9260
- **Forecast horizon:** next observed exchange-rate observation

The March 2022 floating/shock period is substantially more volatile and harder to forecast than ordinary periods. The report therefore treats point forecasting, volatility forecasting, and crisis stress testing as separate tasks.

## Project pipeline

```text
Raw CBSL data
      ↓
Filtering and data-quality checks
      ↓
Level, return, distribution, and regime analysis
      ↓
Stationarity, dependence, and ARCH diagnostics
      ↓
Leakage-aware feature engineering
      ↓
Chronological train/validation/test split
      ↓
Model validation and selection
      ↓
Common test-set evaluation
      ↓
Volatility, multi-horizon, and 2022 stress analyses
      ↓
Full-data refit and final one-step forecast
```

## Models

- **Naive baseline:** carries the current USD/LKR rate forward.
- **ARIMA:** rolling one-step level forecasts; candidate orders are selected using validation MAE.
- **XGBoost-return:** predicts the next log return and converts it back to a rate level.
- **MLP-return:** provides an additional return-based machine-learning comparison.
- **LSTM-return:** uses the previous 20 observed returns to predict the next return.
- **GARCH(1,1):** forecasts conditional variance and is evaluated using volatility MSE and QLIKE.

## Data

The raw working file is [`Data/raw/cbsl.csv`](Data/raw/cbsl.csv). The notebooks filter it to:

```text
base == "USD"
quote == "LKR"
type == "indicative"
```

The cleaned dataset is [`Data/processed/master_dataset.csv`](Data/processed/master_dataset.csv). The source file contains 190,935 rows and the filtered series contains 3,499 observations. The notebooks identify the file as CBSL data; the exact source URL and retrieval date are not encoded in the CSV and should be added separately if required.

Important data-quality facts:

- no duplicate dates remain after cleaning;
- no missing USD/LKR values remain;
- 41 gaps exceed five calendar days;
- the longest calendar gap is 65 days; and
- returns and lags are observation-based rather than guaranteed daily measures.

## Repository structure

```text
USD-LKR-Forecasting/
├── Data/
│   ├── raw/
│   │   └── cbsl.csv
│   └── processed/
│       ├── master_dataset.csv
│       ├── feature_engineered_dataset.csv
│       ├── corrected_model_comparison.csv
│       ├── final_forecast.csv
│       └── other analysis outputs
├── figures_new/
│   └── updated report figures
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda_and_statistics.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_develpoment_and_forecasting.ipynb
│   └── 05_final_prediction.ipynb
├── Project_Report.md
├── Formatted_Project_Report_Updated.docx
└── README.md
```

## Notebook guide

| Notebook | Purpose |
|---|---|
| [`01_data_collection.ipynb`](notebooks/01_data_collection.ipynb) | Filters the raw file, checks quality and observation gaps, and saves the cleaned series. |
| [`02_eda_and_statistics.ipynb`](notebooks/02_eda_and_statistics.ipynb) | Creates returns, evaluates distributions and stationarity, tests dependence and ARCH effects, and compares economic periods. |
| [`03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb) | Creates lag, rolling, calendar-gap, and one-, five-, and twenty-observation target features. |
| [`04_model_develpoment_and_forecasting.ipynb`](notebooks/04_model_develpoment_and_forecasting.ipynb) | Performs chronological validation, evaluates models, generates diagnostics, evaluates volatility, and runs supplementary analyses. |
| [`05_final_prediction.ipynb`](notebooks/05_final_prediction.ipynb) | Refits the validation-selected model using all available observations and writes the final forecast record. |

## Updated figures

The current report uses the figures in `figures_new/`:

1. CBSL USD/LKR indicative exchange rate
2. USD/LKR exchange rate and log returns
3. Normal Q-Q plot and 20-observation rolling return volatility
4. Distribution of USD/LKR log returns
5. Log-return distribution by economic period
6. ACF and PACF of log returns
7. USD/LKR across defined economic periods
8. Log-return distribution and return standard deviation by economic period
9. Feature timing alignment versus future targets and trailing rolling metrics
10. Chronological modelling split across feature dates
11. Common-test point forecast MAE and error distributions
12. Naive baseline interval backtest and empirical coverage
13. One-step-ahead forecasts on the common test period
14. XGBoost feature importance
15. Mean absolute error (MAE) by direct forecast horizon
16. 2022 historical stress test forecasts and MAE comparison
17. Recent log returns and final ARIMA(1,1,1) forecast summary
18. Final forecast for the next observed observation

## Reproduction

1. Open the repository in Jupyter Notebook, JupyterLab, or VS Code.
2. Run the notebooks in numerical order.
3. Confirm that the working directory and relative paths point to the repository root.
4. Review the generated files in [`Data/processed/`](Data/processed/) and figures in [`figures_new/`](figures_new/).
5. Read the full formatted report in [`Formatted_Project_Report_Updated.docx`](Formatted_Project_Report_Updated.docx) or the source report in [`Project_Report.md`](Project_Report.md).

The modelling evaluation uses chronological splits only; random train/test splitting should not be substituted for the documented workflow.

## Limitations

- The models use one exchange-rate series and do not include macroeconomic or market-external predictors.
- The data is an indicative published rate and should not automatically be interpreted as an executable market quote.
- Irregular observation gaps affect the calendar interpretation of returns and lags.
- The five-day eligibility rule removes long-gap targets from the primary comparison.
- Results depend on the selected split, features, hyperparameters, and software environment.
- Prediction intervals are approximate and may under-cover during structural breaks.
- A single historical test period cannot establish universal model superiority.

## Full report

See [`Project_Report.md`](Project_Report.md) for the complete methodology, statistical results, model comparison, figures, discussion, limitations, and conclusion. The submission-ready formatted version is [`Formatted_Project_Report_Updated.docx`](Formatted_Project_Report_Updated.docx).
