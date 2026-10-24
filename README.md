# USD/LKR Exchange-Rate Forecasting

Forecasting the next published USD/LKR indicative exchange-rate observation with
time-series, machine-learning, and volatility models, while measuring how
accuracy changes across forecast horizons and economic regimes.

This project was completed for **Financial Time Series Analysis (CM3023)** at
General Sir John Kotelawala Defence University by **Team 07**.

## Overview

The project investigates the following question:

> How effectively can the next published USD/LKR observation be forecast using
> traditional time-series and machine-learning methods, and how does performance
> change across horizons and economic regimes?

The analysis uses **3,499 Central Bank of Sri Lanka (CBSL) USD/LKR indicative
exchange-rate observations** from **7 May 2010 to 25 September 2026**. It
compares a Naive benchmark, ARIMA, XGBoost, MLP, and LSTM models for point
forecasting. GARCH(1,1) is evaluated separately as a conditional-volatility
model.

The target is the **next available published observation**, not a rate assigned
to an invented calendar date. Lags and forecast horizons are consequently
measured in observations. Because the source series contains weekends, holidays,
and longer gaps, an observation-based return can cover more than one calendar
day.

## Key results

| Item | Result |
| --- | --- |
| Best point-forecast model by test MAE | ARIMA(1,1,1): **0.406721** |
| Naive test MAE / RMSE | 0.424284 / **0.947432** |
| ARIMA(1,1,1) test MAE / RMSE | **0.406721** / 0.974414 |
| GARCH(1,1) volatility MSE / QLIKE | 0.411381 / -2.46633 |
| Final one-step forecast | **330.3869** |
| Latest observed value | 330.3522 on 2026-09-25 |
| Approximate 95% prediction interval | 327.8478 to 332.9260 |
| Main structural break | March 2022 floating/shock regime |

### Main findings

- ARIMA(1,1,1) has the lowest test MAE and improves on the Naive MAE by about
  4.1%. A Diebold-Mariano test reports DM = -2.60 and p = 0.0094.
- The Naive benchmark has the lowest RMSE, showing that no point-forecast model
  wins on every metric when large errors dominate squared loss.
- XGBoost-return, LSTM-return, and MLP-return do not outperform ARIMA or Naive
  on the primary MAE comparison.
- Naive beats XGBoost-return at 1, 5, and 20 observations ahead and during the
  2022 stress test.
- Point forecasting and volatility forecasting are different tasks and are
  evaluated with different metrics.

## Data

### Source and cleaning

The raw file is a multi-country central-bank exchange-rate dataset sourced from
[Central Bank Exchange Rates on Kaggle](https://www.kaggle.com/datasets/allratestoday/central-bank-exchange-rates).
The Sri Lankan series is selected from the supplied file using:

- `base == "USD"`
- `quote == "LKR"`
- `type == "indicative"`

The raw file, [`Data/raw/cbsl.csv`](Data/raw/cbsl.csv), contains 190,935 rows
and the columns `date`, `base`, `quote`, `type`, and `value`. The cleaned series
is saved in [`Data/processed/master_dataset.csv`](Data/processed/master_dataset.csv)
with the columns `date` and `USD_LKR`.

### Data-quality summary

- 3,499 observations and 3,498 log returns remain after filtering and cleaning.
- No duplicate dates or missing exchange-rate values remain.
- There are 41 gaps longer than five calendar days; the longest is 65 days.
- There are 387 zero-change observations.
- 354 extreme returns were flagged and inspected, but not deleted.
- The primary one-step comparison excludes targets more than five calendar days
  after the forecast origin, leaving 3,067 eligible observations.

### Return and regime statistics

- Log-return mean: 0.000305
- Log-return standard deviation: 0.004890
- Skewness: 8.74
- Excess kurtosis: 201.42
- Minimum / maximum log return: -0.040162 / 0.127185

The project uses three economic periods:

| Period | Dates | Observations | Return standard deviation |
| --- | --- | ---: | ---: |
| Pre-float / managed | 2010-05-07 to 2022-03-08 | 2,399 | 0.003094 |
| Floating / 2022 shock | 2022-03-09 to 2022-12-30 | 198 | 0.014723 |
| Post-2022 stabilization | 2023-01-02 to 2026-09-25 | 902 | 0.004269 |

The level series is non-stationary (ADF p = 0.756), while log returns are
stationary (ADF p approximately 1.75 x 10^-13). Ljung-Box tests indicate
serial dependence, the ARCH-LM test indicates strong conditional
heteroskedasticity (p approximately 1.19 x 10^-123), and a Chow-style test
around 9 March 2022 is highly significant (F = 11,416).

## Methodology

```text
Raw exchange-rate data
        |
Filtering and data-quality checks
        |
Level, return, distribution, and regime analysis
        |
Stationarity, dependence, and ARCH diagnostics
        |
Leakage-aware feature engineering
        |
Chronological train/validation/test split
        |
Model validation and selection
        |
Common test-set evaluation
        |
Volatility, multi-horizon, and 2022 stress analyses
        |
Full-data refit and final one-step forecast
```

The primary comparison uses a chronological split of 3,067 eligible
observations:

| Partition | Observations |
| --- | ---: |
| Training | 2,146 |
| Validation | 460 |
| Test | 461 |

Random train/test splitting is deliberately not used.

### Features

The feature-engineered dataset,
[`Data/processed/feature_engineered_dataset.csv`](Data/processed/feature_engineered_dataset.csv),
contains:

- Current state: `USD_LKR`, `log_return`
- Calendar-gap features: `days_since_previous_obs`, `long_previous_gap`
- Level and return lags: 1, 5, and 20 observations
- Rolling features: five-observation mean, twenty-observation mean, and
  twenty-observation standard deviation/volatility
- Direct targets and target dates for 1-, 5-, and 20-observation horizons

All predictors are available at the forecast origin. Target columns are excluded
from model inputs, and the feature-availability audit is saved as
[`Data/processed/feature_availability_audit.csv`](Data/processed/feature_availability_audit.csv).

### Models

| Model | Forecast target | Details |
| --- | --- | --- |
| Naive | Next USD/LKR level | Carries the current rate forward |
| ARIMA | Next USD/LKR level | Orders (0,1,0), (1,1,0), (0,1,1), and (1,1,1) compared by validation MAE |
| XGBoost-return | Next log return | 300 trees, depth 3, learning rate 0.03, subsample 0.8, column subsample 0.8 |
| MLP-return | Next log return | Scaled return-based neural-network comparison |
| LSTM-return | Next log return | Previous 20 observed returns; requires TensorFlow |
| GARCH(1,1) | Conditional variance | Evaluated with volatility MSE and QLIKE, not level MAE/RMSE |

### Test-set comparison

| Model | MAE | RMSE | Directional accuracy |
| --- | ---: | ---: | ---: |
| **ARIMA(1,1,1)** | **0.406721** | 0.974414 | 0.620390 |
| Naive | 0.424284 | **0.947432** | N/A (no-change forecast) |
| XGBoost-return | 0.448500 | 1.100929 | 0.611714 |
| LSTM-return | 0.511477 | 0.982958 | 0.581345 |
| MLP-return | 18.030750 | 22.014662 | 0.568330 |

### Additional analyses

- GARCH(1,1) has alpha + beta = 0.995978, indicating highly persistent
  volatility.
- Multi-horizon MAE for Naive versus XGBoost-return is 1.116 versus 1.445 at
  five observations ahead, and 2.749 versus 4.616 at twenty observations ahead.
- In the 2022 stress test (239 observations), Naive MAE is 1.112 versus 4.680
  for XGBoost-return.
- The final ARIMA interval is approximate and may under-cover during structural
  breaks. The Naive interval backtest achieved 98.48% empirical coverage for a
  nominal 95% interval on 461 test observations.

## Final forecast

[`Data/processed/final_forecast.csv`](Data/processed/final_forecast.csv) records
the final forecast produced by refitting the validation-selected ARIMA(1,1,1)
model to the full available series:

| Field | Value |
| --- | --- |
| Latest observation date | 2026-09-25 |
| Latest observed USD/LKR | 330.3522 |
| Forecast horizon | Next observed date (not yet known) |
| Forecast | **330.38692185427385** |
| Approximate lower interval | 327.8478488293395 |
| Approximate upper interval | 332.9259948792082 |
| Interval method | ARIMA model interval |

The reproducibility record, including the generated timestamp and package
versions, is in
[`Data/processed/final_forecast_audit.json`](Data/processed/final_forecast_audit.json).

## Repository structure

```text
USD-LKR-Forecasting/
├── All_Graphs/                         # 21 generated PNG figures
├── Data/
│   ├── raw/
│   │   └── cbsl.csv                    # Supplied multi-country source data
│   └── processed/                      # Cleaned data and analysis outputs
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda_and_statistics.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_develpoment_and_forecasting.ipynb
│   └── 05_final_prediction.ipynb
├── reports/
│   ├── USD_LKR_Forecasting_Report - FTSA(Final).docx
│   └── USD_LKR_Forecasting_Report - FTSA(Final).pdf
├── src/
│   └── models/                         # Reserved source-model directory
├── Group Assignment.pdf
├── README.md
└── .gitignore
```

The spelling of `04_model_develpoment_and_forecasting.ipynb` is retained to
match the existing filename.

## Notebooks

Run the notebooks in this order:

| Notebook | Purpose | Main outputs |
| --- | --- | --- |
| [`01_data_collection.ipynb`](notebooks/01_data_collection.ipynb) | Loads the raw CSV, filters the USD/LKR indicative series, checks dates and gaps, inspects unusual movements, and saves the cleaned series. | `master_dataset.csv`, collection figures |
| [`02_eda_and_statistics.ipynb`](notebooks/02_eda_and_statistics.ipynb) | Creates returns, examines distributions and volatility, runs stationarity/dependence/ARCH tests, compares regimes, and checks the 2022 structural break. | `usd_lkr_eda.csv`, `stationarity_interpretation.csv`, EDA figures |
| [`03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb) | Creates date-gap, lag, rolling, target, and target-date features and performs leakage/alignment audits. | `feature_engineered_dataset.csv`, `feature_availability_audit.csv` |
| [`04_model_develpoment_and_forecasting.ipynb`](notebooks/04_model_develpoment_and_forecasting.ipynb) | Performs chronological validation, evaluates point and volatility models, and runs interval, horizon, and 2022 stress analyses. | Model comparison, validation, volatility, horizon, stress-test, and metadata files |
| [`05_final_prediction.ipynb`](notebooks/05_final_prediction.ipynb) | Refits the selected model on all observations, creates the next-observation forecast and interval, writes the audit record, and plots the result. | `final_forecast.csv`, `final_forecast_audit.json`, final figures |

## Processed outputs

The `Data/processed/` directory contains:

| File | Description |
| --- | --- |
| `master_dataset.csv` | Cleaned date and USD/LKR level series |
| `usd_lkr_eda.csv` | Level, log return, and rolling-volatility EDA data |
| `stationarity_interpretation.csv` | Stationarity-test interpretation |
| `feature_engineered_dataset.csv` | Predictors, targets, and target dates |
| `feature_availability_audit.csv` | Missingness audit for model features |
| `expanding_window_validation.csv` | Historical expanding-window validation |
| `model_results.csv`, `model_results_initial.csv` | Intermediate model results |
| `final_model_results.csv`, `corrected_model_comparison.csv` | Final common-test comparisons |
| `diebold_mariano_vs_naive.csv` | Diebold-Mariano comparisons with Naive |
| `multi_horizon_results.csv` | 1-, 5-, and 20-observation horizon results |
| `regime_2022_results.csv` | 2022 stress-test results |
| `garch_volatility_metrics.csv` | GARCH volatility MSE and QLIKE |
| `interval_backtest.csv` | Prediction-interval coverage diagnostics |
| `mlp_validation_scores.csv` | MLP validation configuration and scores |
| `forecast_metadata.json` | Selected model, configurations, and reproducibility metadata |
| `final_forecast.csv` | Final next-observation forecast |
| `final_forecast_audit.json` | Final forecast audit and environment details |

The `All_Graphs/` directory contains 21 generated figures covering data
collection, EDA, feature timing, modelling splits, point-forecast comparisons,
interval diagnostics, feature importance, horizon analysis, the 2022 stress
test, and the final forecast.

## Reproduction

### Requirements

The notebooks use Python and the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `statsmodels`
- `scikit-learn`
- `xgboost`
- `jupyter` or another compatible notebook interface
- `tensorflow` for the optional LSTM section

This repository does not currently include a `requirements.txt` or environment
lock file. Install compatible versions in a Python environment before running
the notebooks. The checked-in virtual environment is intentionally ignored by
Git.

### Steps

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies listed above.
4. Open the repository in JupyterLab, Jupyter Notebook, or VS Code.
5. Run notebooks 01 through 05 in numerical order.
6. Run them from the repository's `notebooks/` context so their relative data
   paths resolve as written.
7. Review generated tables in `Data/processed/` and figures in `All_Graphs/`.

The LSTM section is optional and is skipped when TensorFlow is unavailable;
other model comparisons and the final ARIMA forecast can still be reproduced.
The final forecast notebook uses the metadata generated by notebook 04, so
notebook 04 should be completed before notebook 05.

## Limitations

- The analysis is univariate and does not include interest rates, inflation,
  reserves, trade flows, or other external predictors.
- The target is an indicative published rate, not an executable market quote.
- Irregular observation gaps mean an observation-based return may span several
  calendar days; the five-day eligibility rule removes long-gap targets from
  the primary comparison.
- Results depend on the selected split, features, hyperparameters, random seed,
  and software versions.
- A single test period cannot establish universal model superiority.
- Prediction intervals are approximate and may under-cover during structural
  breaks.
- Level MAE/RMSE, directional accuracy, volatility MSE, and QLIKE measure
  different properties and should not be combined into one ranking.

## Team and contributions

This project was completed by **Team 07** for Financial Time Series Analysis
(CM3023), General Sir John Kotelawala Defence University:

| Student ID | Team member |
| --- | --- |
| D/DBA/24/0004 | YT Horawalavithana |
| D/DBA/24/0005 | RMSS Gamagedara |
| D/DBA/24/0009 | HM Mallikaarachchi |
| D/DBA/24/0024 | VJK Rodrigo |
| D/DBA/24/0034 | KKGS Hansana |

The project was completed collaboratively. Contributions covered the following
work areas:

- **Data preparation:** sourcing the exchange-rate data, filtering the
  USD/LKR indicative series, validating dates and gaps, and creating the
  cleaned master dataset.
- **Exploratory and statistical analysis:** calculating log returns, examining
  distributions and volatility, testing stationarity and dependence, testing
  ARCH effects, and analysing the March 2022 regime change.
- **Feature engineering:** creating observation-based lags, rolling statistics,
  calendar-gap variables, forecast targets, target dates, and leakage audits.
- **Model development:** implementing and comparing the Naive, ARIMA, XGBoost,
  MLP, LSTM, and GARCH approaches using chronological validation.
- **Evaluation and interpretation:** calculating point-forecast, directional,
  volatility, interval, multi-horizon, and stress-test metrics, including the
  Diebold-Mariano comparison.
- **Final forecast and reproducibility:** selecting and refitting the final
  ARIMA model, generating the forecast interval and audit record, checking
  reproducibility, and preparing the final figures and report.
- **Documentation and presentation:** preparing the notebooks, processed
  outputs, visualisations, final report, and repository documentation.

The final report is available in
[`reports/USD_LKR_Forecasting_Report - FTSA(Final).pdf`](reports/USD_LKR_Forecasting_Report%20-%20FTSA(Final).pdf)
and [`reports/USD_LKR_Forecasting_Report - FTSA(Final).docx`](reports/USD_LKR_Forecasting_Report%20-%20FTSA(Final).docx).
