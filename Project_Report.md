# Forecasting the USD/LKR Exchange Rate

## Abstract

This study investigates whether the next published USD/LKR exchange rate can be forecast more accurately using traditional time-series models or machine-learning models. The analysis uses 3,499 indicative USD/LKR observations extracted from a Central Bank of Sri Lanka exchange-rate dataset. The data covers the period from 7 May 2010 to 25 September 2026 and includes the major economic disruption experienced during 2022.

The project follows a complete forecasting workflow: data extraction and cleaning, exploratory analysis, return and volatility analysis, stationarity testing, feature engineering, chronological model evaluation, and final model selection. Five approaches are compared: a naive persistence benchmark, ARIMA(1,1,1), GARCH(1,1), XGBoost, and LSTM. The final modelling dataset contains 3,477 complete observations. The first 80% is used for training and the final 20% is used as an unseen chronological test set.

The naive forecast performs best, with an MAE of 0.463135 and RMSE of 0.886722. GARCH performs almost identically, while XGBoost, LSTM, and ARIMA produce larger errors. The results suggest that, for this one-step-ahead level forecast, the most recent exchange-rate observation is a stronger predictor than the additional univariate features used by the more complex models. The analysis also shows that 2022 was a distinct high-volatility regime: average rolling volatility was approximately twice the level observed during the 2023–2024 stabilization period.

**Keywords:** USD/LKR, exchange-rate forecasting, time series, ARIMA, GARCH, XGBoost, LSTM, volatility, Sri Lankan economic crisis

---

## 1. Introduction

Exchange-rate movements affect import costs, export earnings, foreign debt payments, tourism receipts, remittances, and business planning. For Sri Lanka, the USD/LKR exchange rate is particularly important because many domestic transactions and external obligations are influenced by the value of the US dollar relative to the Sri Lankan rupee.

Exchange rates are difficult to forecast. They can contain trends, random-walk-like behaviour, volatility clustering, structural breaks, policy effects, and sudden responses to economic news. A model that performs well during a stable period may perform poorly during a crisis. Therefore, comparing only one sophisticated model is not sufficient. A meaningful study should compare simple and complex approaches using a time-ordered evaluation design.

This project was developed in response to the requirements in [`Group Assignment.pdf`](Group%20Assignment.pdf), which require a real financial forecasting problem, actual financial data, exploratory and statistical analysis, traditional time-series models, machine-learning models, a simple baseline, time-based evaluation, additional financial analysis, interpretation, and limitations.

### 1.1 Research question

> **How effectively can the next published USD/LKR exchange-rate observation be forecast using traditional time-series methods compared with machine-learning methods, and what can be learned from the differences in their performance?**

### 1.2 Objectives

The study has six objectives:

1. obtain and clean a real USD/LKR exchange-rate dataset;
2. describe the long-term level, returns, and volatility of the exchange rate;
3. investigate stationarity, autocorrelation, and the 2022 crisis regime;
4. create historical and rolling features without using future information;
5. compare a naive benchmark with ARIMA, GARCH, XGBoost, and LSTM; and
6. interpret the results from a financial and risk-management perspective.

### 1.3 Forecasting target

The target is the USD/LKR value at the **next available observation**. The raw data contains published business-day observations, so the target is not an artificially interpolated calendar-day rate. For an observation at time \(t\), the target is:

$$
Target_t = USD\_LKR_{t+1}.
$$

The study focuses on one-step-ahead forecasting. It does not claim to forecast a specific future date several weeks or months ahead.

---

## 2. Data and dataset preparation

### 2.1 Raw dataset

The raw input is [`Data/raw/cbsl.csv`](Data/raw/cbsl.csv). It contains exchange-rate observations for multiple currency pairs. The file has 190,935 rows and five columns:

| Column | Description |
|---|---|
| `date` | Date on which the exchange rate was published |
| `base` | Base currency |
| `quote` | Quoted currency |
| `type` | Type of exchange rate |
| `value` | Numeric exchange-rate value |

The collection notebook filters the raw file using:

```text
base == "USD"
quote == "LKR"
type == "indicative"
```

This produces 3,499 USD/LKR observations. Only `date` and `value` are retained, and `value` is renamed to `USD_LKR`.

### 2.2 Cleaning procedure

The data-collection stage performs the following operations:

1. convert the date column to a datetime type;
2. sort observations chronologically;
3. remove duplicate dates;
4. set the date as the time-series index;
5. check for missing exchange-rate values; and
6. save the result as `master_dataset.csv`.

The resulting series contains no missing USD/LKR values. The first observation is 113.6975 on 2010-05-07 and the final stored observation is 330.3522 on 2026-09-25. The saved modelling forecast uses 329.0158 as its latest observed modelling value because the one-step target construction ends one observation earlier.

![Historical USD-LKR Exchange Rate](figures/Historical%20USD-LKR%20Exchange%20Rate.png)

**Figure 1. Historical USD/LKR exchange rate.** The complete series shows a long-term upward movement in the number of rupees required to buy one US dollar. The most visible change occurs around 2022, when the rate moves rapidly to a substantially higher level. This long-run movement is why the level series must not automatically be treated as stationary.

### 2.3 Processed datasets

The project saves intermediate datasets so that each stage can be reproduced independently.

| File | Contents | Role in the analysis |
|---|---|---|
| [`Data/processed/master_dataset.csv`](Data/processed/master_dataset.csv) | `date`, `USD_LKR`; 3,499 rows | Clean exchange-rate level used for EDA |
| [`Data/processed/usd_lkr_eda.csv`](Data/processed/usd_lkr_eda.csv) | Exchange rate, log return, 20-observation rolling volatility | EDA output and feature-engineering input |
| [`Data/processed/feature_engineered_dataset.csv`](Data/processed/feature_engineered_dataset.csv) | 14 columns and 3,477 complete rows | Final modelling dataset |
| [`Data/processed/model_results_initial.csv`](Data/processed/model_results_initial.csv) | Naive and ARIMA results | Intermediate comparison |
| [`Data/processed/model_results.csv`](Data/processed/model_results.csv) | Results for all five models | Consolidated model output |
| [`Data/processed/final_model_results.csv`](Data/processed/final_model_results.csv) | Five-model results | Final unsorted result table |
| [`Data/processed/final_model_comparison.csv`](Data/processed/final_model_comparison.csv) | Five-model results sorted by RMSE | Final ranking |
| [`Data/processed/final_forecast.csv`](Data/processed/final_forecast.csv) | Selected model and one-step forecast | Final forecast output |

The result files are workflow outputs, not separate experiments. `model_results_initial.csv` documents the early Naive-versus-ARIMA comparison; the later files include GARCH, XGBoost, and LSTM.

### 2.4 Data-quality and provenance considerations

The repository identifies the raw file as a CBSL dataset. For formal publication, the source URL, download date, release/version, and any upstream data transformations should also be recorded. The stored data extends to September 2026, so the group should verify the source snapshot before submitting the report. This report describes the files currently present in the repository.

---

## 3. Exploratory analysis

### 3.1 Descriptive statistics of the exchange-rate level

The cleaned level series contains 3,499 observations:

| Statistic | Value |
|---|---:|
| Mean | 206.526733 |
| Standard deviation | 80.919184 |
| Minimum | 109.459700 |
| 25th percentile | 141.541700 |
| Median | 179.672200 |
| 75th percentile | 299.827500 |
| Maximum | 364.760000 |

The large range reflects both gradual depreciation and the abrupt 2022 change. These are properties of the level, not direct measures of daily risk.

### 3.2 The 2022 crisis

The analysis defines January–December 2022 as the crisis period. The year contains 240 observations.

![USD-LKR Exchange Rate During 2022](figures/USD-LKR%20Exchange%20Rate%20During%202022.png)

**Figure 2. USD/LKR exchange rate during 2022.** Isolating 2022 makes the rapid level change easier to see than the full-history plot. The rate does not simply fluctuate around its earlier range; it experiences a sharp change in the exchange-rate regime.

The project compares this crisis period with January 2023–December 2024, labelled the stabilization period.

![USD-LKR Exchange Rate Crisis and Stabilization](figures/USD-LKR%20Exchange%20Rate-%20Crisis%20vs%20Stabilization.png)

**Figure 3. Crisis and stabilization exchange-rate levels.** The crisis and stabilization periods occupy different level ranges and show different movement patterns. The stabilization period is not a return to the pre-crisis level; rather, it is a period of lower short-run variability around a new level.

### 3.3 Log returns

To study changes rather than levels, the project calculates the logarithmic return:

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right),
$$

where \(P_t\) is the USD/LKR rate at observation \(t\). The first observation has no previous value and therefore has one missing return.

![Daily USD-LKR Log Returns](figures/Daily%20USD-LKR%20Log%20Returns.png)

**Figure 4. Daily USD/LKR log returns.** Most daily changes are close to zero, but large positive and negative movements appear around unstable periods. This indicates that the rate can look relatively smooth in levels while still having substantial short-run shocks.

### 3.4 Distribution of returns

![Distribution of USD-LKR Daily Log Returns](figures/Distribution%20of%20USD-LKR%20Daily%20Log%20Returns.png)

**Figure 5. Distribution of daily log returns.** Returns are concentrated near zero but have long tails. This means that extreme movements occur more often than a narrow, constant-variance distribution would suggest. Such observations influence RMSE strongly because RMSE squares each error.

![Boxplot of USD-LKR Daily Log Returns](figures/Boxplot%20of%20USD-LKR%20Daily%20Log%20Returns.png)

**Figure 6. Boxplot of daily log returns.** The boxplot shows a compact central range with several extreme outliers. These outliers are evidence of unusual market movements and reinforce the need to distinguish typical forecasting accuracy from performance during shocks.

### 3.5 Rolling volatility

The project uses the standard deviation of the previous 20 returns as a short-run volatility measure:

$$
\sigma_{20,t} = Std(r_{t-19}, \ldots, r_t).
$$

![20-Day Rolling Volatility of USD-LKR Log Returns](figures/20-Day%20Rolling%20Volatility%20of%20USD-LKR%20Log%20Returns.png)

**Figure 7. Twenty-observation rolling volatility.** Volatility changes over time rather than remaining constant. It rises during periods with large daily movements and falls during calmer periods. The first 19 observations cannot have a complete 20-observation rolling statistic.

![20-Day Rolling Volatility Crisis and Stabilization](figures/20-Day%20Rolling%20Volatility-%20Crisis%20vs%20Stabilization.png)

**Figure 8. Rolling volatility during crisis and stabilization.** The crisis period reaches higher and more variable volatility levels than the stabilization period. This is important for financial users because a model may have similar point forecasts while the risk of an unexpectedly large movement is very different.

![Distribution of USD-LKR Log Returns by Regime](figures/Distribution%20of%20USD-LKR%20Log%20Returns.png)

**Figure 9. Return distributions across regimes.** The crisis distribution is wider and has more extreme movements than the stabilization distribution. The chart provides a distributional explanation for the different volatility statistics.

### 3.6 Numerical crisis–stabilization comparison

| Statistic | Crisis 2022 | Stabilization 2023–2024 |
|---|---:|---:|
| Number of observations | 240 | 484 |
| Mean log return | 0.002476 | -0.000446 |
| Return standard deviation | 0.013481 | 0.005111 |
| Minimum return | -0.026258 | -0.040162 |
| Median return | 0.000000 | -0.000044 |
| Maximum return | 0.127185 | 0.030787 |
| Mean rolling volatility | 0.006927 | 0.003467 |
| Rolling-volatility standard deviation | 0.010817 | 0.003560 |
| Maximum rolling volatility | 0.035897 | 0.017321 |

The crisis mean return is positive, consistent with a rapid increase in the USD/LKR level. Its return standard deviation is more than twice the stabilization value, and its mean rolling volatility is approximately twice the stabilization value. The negative stabilization mean is small relative to the crisis movement and should not be interpreted as a permanent appreciation trend.

### 3.7 Stationarity

The Augmented Dickey–Fuller test results are:

| Series | ADF statistic | p-value | Interpretation |
|---|---:|---:|---|
| USD/LKR level | approximately -0.995 | approximately 0.756 | Evidence consistent with non-stationarity |
| Log returns | -8.442357 | 0.000000 | Evidence of stationarity |

The level test does not reject the unit-root null at the 5% level. The return test rejects it strongly. Therefore, the rate level is unsuitable for methods that require a stationary level without differencing, while returns are more appropriate for volatility analysis.

### 3.8 Autocorrelation and partial autocorrelation

![ACF of USD-LKR Daily Log Returns](figures/ACF%20of%20USD-LKR%20Daily%20Log%20Returns.png)

**Figure 10. Autocorrelation function of returns.** The ACF examines linear correlation between returns and their lagged values. Most lags are small, suggesting limited persistent linear dependence.

![PACF of USD-LKR Daily Log Returns](figures/PACF%20of%20USD-LKR%20Daily%20Log%20Returns.png)

**Figure 11. Partial autocorrelation function of returns.** The PACF measures the additional relationship at each lag after accounting for shorter lags. It does not reveal a strong stable lag structure. This does not prove that the market is perfectly efficient; nonlinear effects, volatility clustering, policy events, and external variables may still matter.

---

## 4. Feature engineering

The feature-engineering notebook uses `usd_lkr_eda.csv` and creates a modelling table with historical information only. The final dataset has 3,477 rows and no missing values.

| Feature group | Variables | Purpose |
|---|---|---|
| Current state | `USD_LKR`, `log_return`, `rolling_volatility_20` | Describe the current level, movement, and recent risk |
| Level lags | `Lag_1`, `Lag_5`, `Lag_20` | Capture short-, weekly-scale, and longer short-run persistence |
| Return lags | `Return_Lag_1`, `Return_Lag_5`, `Return_Lag_20` | Capture momentum or reversal information |
| Rolling level statistics | `Rolling_Mean_5`, `Rolling_Mean_20` | Capture local and medium short-run trends |
| Rolling dispersion | `Rolling_Std_20` | Capture recent level variability |
| Target | `Target` | Next-observation USD/LKR value |

Lag variables use `shift()`, rolling variables use trailing windows, and the target uses a forward shift. Rows without sufficient history are removed. This design prevents the model from seeing the future target while creating the predictors.

---

## 5. Forecasting methodology

### 5.1 Time-based train/test design

The modelling table is ordered by date and divided chronologically:

- training observations: 2,781;
- test observations: 696;
- training proportion: 80%;
- testing proportion: 20%.

![Training and Testing Periods](figures/Training%20and%20Testing%20Periods%20of%20USD-LKR%20Exchange%20Rate.png)

**Figure 12. Chronological training and testing periods.** The test observations occur after the training observations. This avoids the invalid practice of randomly mixing future observations into the training data.

The project uses a single hold-out test period. This is appropriate for a clear baseline comparison, but a stronger study would add rolling-origin validation and a separately reserved final test period.

### 5.2 Evaluation measures

For actual values \(y_i\) and predictions \(\hat{y}_i\):

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}.
$$

MAE measures the typical absolute error in exchange-rate units. RMSE gives more influence to large misses, which is useful when crisis shocks are financially important. Lower values indicate better point-forecast accuracy.

### 5.3 Models

#### Naive benchmark

The naive forecast is:

$$
\hat{P}_{t+1}=P_t.
$$

It is a necessary benchmark because exchange-rate levels often display strong short-term persistence. A complex model should outperform this forecast before its added complexity is justified.

#### ARIMA(1,1,1)

ARIMA combines autoregression, first-order differencing, and a moving-average error term. It is fitted to the training level series. The model is designed to represent temporal dependence in a non-stationary level after differencing.

#### GARCH(1,1)

GARCH is fitted to log returns and models time-varying conditional variance. Its predicted return is converted back into a level forecast for comparison with the other models. GARCH is therefore evaluated both as a point-forecast approach and as a volatility model.

#### XGBoost

XGBoost is a gradient-boosted tree regression model using the engineered level, return, and rolling features. It can learn nonlinear relationships and interactions that a linear time-series model may miss.

#### LSTM

The LSTM uses 20 previous USD/LKR observations as a sequence. The level is scaled using a Min-Max scaler fitted only on training data. The network contains LSTM layers with 64 and 32 units, dropout layers, and a one-unit dense output. The training input has shape `(2761, 20, 1)` and the test input has shape `(696, 20, 1)`.

---

## 6. Forecasting results

### 6.1 Naive forecast

![Actual versus Naive Forecast](figures/Actual%20vs%20Naive%20USD-LKR%20Forecast.png)

**Figure 13. Actual and naive forecasts.** The naive line is the previous observed rate. It tracks the actual level closely because consecutive published observations are generally close, even though individual daily changes remain unpredictable.

The naive model obtains MAE 0.463135 and RMSE 0.886722. This establishes a difficult benchmark for the other models.

### 6.2 ARIMA forecast

![Actual versus ARIMA Forecast](figures/Actual%20vs%20ARIMA%20USD-LKR%20Forecast.png)

**Figure 14. Actual and ARIMA forecasts.** The ARIMA forecast does not follow the test-period level as closely as the persistence forecast. The model summary also reports severe non-normality and heteroskedasticity diagnostics, which are consistent with a crisis-affected series that is difficult to represent with a simple ARIMA specification.

ARIMA obtains MAE 23.063531 and RMSE 25.768170. The result indicates weak out-of-sample level tracking for this configuration and test period.

### 6.3 GARCH forecast and volatility

![Actual versus GARCH Forecast](figures/Actual%20vs%20GARCH%20USD-LKR%20Forecast.png)

**Figure 15. Actual and GARCH level forecasts.** GARCH produces a level forecast very close to the naive forecast. Its MAE is 0.465039 and its RMSE is 0.887716, only slightly worse than the naive benchmark.

![GARCH Conditional Volatility](figures/GARCH%20Conditional%20Volatility%20of%20USD-LKR%20Returns.png)

**Figure 16. GARCH conditional volatility.** This chart shows the model’s estimate of changing return risk rather than only the expected level. Peaks indicate periods where the model estimates greater conditional uncertainty. Consequently, GARCH provides risk information that the naive point forecast cannot provide.

### 6.4 XGBoost forecast

![Actual versus XGBoost Forecast](figures/Actual%20vs%20XGBoost%20USD-LKR%20Forecast.png)

**Figure 17. Actual and XGBoost forecasts.** XGBoost uses all engineered predictors and performs better than ARIMA and LSTM, but its forecast is still less accurate than persistence. The result suggests that the available univariate features did not contain enough stable information to overcome the latest observed level.

XGBoost obtains MAE 1.857661 and RMSE 2.426569.

### 6.5 LSTM forecast

![Actual versus LSTM Forecast](figures/Actual%20vs%20LSTM%20USD-LKR%20Forecast.png)

**Figure 18. Actual and LSTM forecasts.** The LSTM uses a 20-observation sequence of the exchange-rate level. Its large test error indicates that the chosen architecture and training setup did not generalize well to the chronological test period.

![LSTM Training and Validation Loss](figures/LSTM%20Training%20and%20Validation%20Loss.png)

**Figure 19. LSTM training and validation loss.** The loss curves describe the training process. They are useful for diagnosing convergence and possible overfitting, but they are not a replacement for the final untouched test metrics. The LSTM obtains MAE 21.718804 and RMSE 22.096873.

### 6.6 Final model comparison

The initial notebook comparison contains only Naive and ARIMA:

![Initial Forecasting Model Comparison](figures/Initial%20Forecasting%20Model%20Comparison.png)

**Figure 20. Initial model comparison.** This is an intermediate project checkpoint and documents the first baseline-versus-ARIMA experiment. It should not be interpreted as the final five-model result.

The final comparison is:

| Rank by RMSE | Model | MAE | RMSE |
|---:|---|---:|---:|
| 1 | **Naive** | **0.463135** | **0.886722** |
| 2 | GARCH | 0.465039 | 0.887716 |
| 3 | XGBoost | 1.857661 | 2.426569 |
| 4 | LSTM | 21.718804 | 22.096873 |
| 5 | ARIMA | 23.063531 | 25.768170 |

![MAE Comparison](figures/MAE%20Comparison%20of%20USD-LKR%20Forecasting%20Models.png)

**Figure 21. Final MAE comparison.** Naive and GARCH are almost tied and substantially better than the remaining models. XGBoost is the strongest of the two feature-based nonlinear approaches, but it does not beat persistence.

![RMSE Comparison](figures/RMSE%20Comparison%20of%20USD-LKR%20Forecasting%20Models.png)

**Figure 22. Final RMSE comparison.** The RMSE ranking is the same as the MAE ranking. The larger gap between RMSE and MAE reflects the impact of occasional large forecasting errors.

---

## 7. Discussion

### 7.1 Why did the naive model win?

The naive model is strongest for four related reasons:

1. **Short-run persistence:** the next published rate is often close to the current rate.
2. **Weak return dependence:** the ACF and PACF provide limited evidence of a stable linear return pattern.
3. **Structural change:** the 2022 crisis creates a relationship that may not generalize to the later test period.
4. **Limited information set:** all predictive information is derived from the exchange-rate history; external economic drivers are absent.

The result is not evidence that complex models are universally ineffective. It is evidence that complexity did not improve this one-step-ahead univariate level forecast under the selected split and model settings.

### 7.2 Why was GARCH almost tied?

GARCH is primarily designed to model conditional variance. When the conditional mean return is close to zero, the resulting point level forecast can be very similar to the previous observed level. Therefore, its near-tie with Naive is not surprising. Its main contribution is the conditional-volatility estimate, which is useful for risk monitoring, hedging, and identifying unstable periods.

### 7.3 Why did XGBoost, LSTM, and ARIMA perform worse?

XGBoost can learn nonlinear patterns, but its features are still derived only from past USD/LKR observations. If the next movement is mainly driven by new information, the model cannot observe that information in advance. The LSTM has greater representational capacity, but capacity does not create predictive signal; it can also make training and generalization more sensitive to scaling, architecture, look-back length, and regime changes. ARIMA is interpretable and useful when its assumptions and specification match the data, but this series contains structural breaks, extreme returns, and changing variance that a simple ARIMA(1,1,1) may not capture.

### 7.4 Financial interpretation

For a basic operational one-step forecast, the naive model is attractive because it is accurate, transparent, inexpensive, and easy to update. However, an exchange-rate decision should not use the point forecast alone. A treasury team also needs an estimate of uncertainty. In that context, a practical system could combine the naive level forecast with a GARCH-based volatility estimate and external macroeconomic indicators.

The crisis analysis also shows why historical averages are not enough. During 2022, the average return and volatility were much higher than during stabilization. A model trained across all periods may be acceptable as a general benchmark but may need regime-specific treatment when the economic environment changes.

---

## 8. Limitations

1. **Univariate data:** the models do not include interest rates, inflation, foreign reserves, imports, exports, IMF events, oil, gold, DXY, or other explanatory variables.
2. **Single hold-out period:** one 80/20 split cannot establish performance stability across different historical regimes.
3. **No independent validation period:** the LSTM has a validation-loss curve, but the overall model selection does not use a separate chronological validation set.
4. **One-step horizon:** results may differ for multi-day or multi-month forecasts.
5. **Limited tuning:** model orders, hyperparameters, look-back windows, and training settings were not evaluated through a systematic search.
6. **Level-based metrics:** MAE and RMSE of the exchange-rate level favour persistence when adjacent levels are close; return and direction metrics could provide additional insight.
7. **No prediction intervals:** the final forecast is a point estimate, not a probabilistic forecast.
8. **Data provenance:** the exact raw-data download metadata should be added before formal submission.
9. **Date verification:** the late end date should be checked against the official source snapshot.

---

## 9. Recommendations for future work

Future research should:

- use rolling-origin or expanding-window validation;
- reserve a final test period that is not used for model selection;
- compare crisis-specific, stabilization-specific, and global models;
- add macroeconomic and market variables;
- test ARIMA and GARCH variants such as EGARCH and GJR-GARCH;
- tune XGBoost and LSTM hyperparameters systematically;
- compare level, return, and direction targets;
- report prediction intervals and volatility-specific metrics; and
- record data-source URLs, package versions, random seeds, and execution dates.

---

## 10. Conclusion

This project finds that the simple naive forecast is the most accurate model for one-step-ahead USD/LKR level prediction on the selected chronological test period. Its MAE is 0.463135 and RMSE is 0.886722, narrowly outperforming GARCH and clearly outperforming XGBoost, LSTM, and ARIMA.

The central lesson is that more complicated models do not automatically provide better forecasts. The latest observed exchange rate contains strong short-run information, while the univariate historical features provide limited additional predictive power. At the same time, GARCH remains valuable for modelling changing risk, and the crisis analysis demonstrates that volatility and return behaviour can differ substantially between economic regimes.

The project therefore provides a useful baseline study rather than a complete production forecasting system. A stronger practical system would combine the transparent persistence forecast with volatility estimates, external economic information, rolling validation, regime-aware modelling, and uncertainty intervals.

## 11. Reproduction guide

Run the notebooks in this order:

1. [`notebooks/01_data_collection.ipynb`](notebooks/01_data_collection.ipynb)  
   Reads `cbsl.csv` and creates `master_dataset.csv`.
2. [`notebooks/02_eda_and_statistics.ipynb`](notebooks/02_eda_and_statistics.ipynb)  
   Creates returns, rolling volatility, EDA outputs, statistical tests, and EDA figures.
3. [`notebooks/03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb)  
   Creates the lagged and rolling predictors and `feature_engineered_dataset.csv`.
4. [`notebooks/04_model_development_and_forecasting.ipynb`](notebooks/04_model_development_and_forecasting.ipynb)  
   Trains the five forecasting approaches, creates forecast figures, saves result tables, and writes `final_forecast.csv`.

The final outputs should reproduce the five-model metrics and the selected Naive forecast documented in this report.
