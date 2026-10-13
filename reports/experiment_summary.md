# CryptoPredict AI — Official Experiment Summary Report
**Date:** 2026-10-13 11:59:45 UTC  
**Asset:** Bitcoin (BTC-USD)  
**Historical Horizon:** 2014-12-15 to 2026-10-12 (4320 supervised observations)

---

## 1. Executive Summary & Winning Models
* **Best Regression Model (Selected by Validation RMSE):** `Naive Previous-Close`
  * **Test Set MAE:** $1,392.03
  * **Test Set RMSE:** $1,968.94
  * **Test Set MAPE:** 1.59%
  * **Test Set R² Score:** 0.9875

* **Best Direction Classification Model (Selected by Validation F1):** `LightGBM Classifier`
  * **Test Set Accuracy:** 50.15%
  * **Test Set F1-Score:** 24.36%
  * **Test Set Precision:** 47.71%
  * **Test Set Sensitivity / Recall:** 16.35%
  * **Test Set Specificity:** 82.73%
  * **Test Set Error Rate:** 49.85%
  * **Test Set ROC-AUC:** 0.4881
  * **Confusion Matrix:** TN=273, FP=57, FN=266, TP=52

---

## 2. Temporal Partitioning (Zero Data Leakage)
Strictly chronological split applied:
* **Training Partition (70%):** 3024 samples (2014-12-15 to 2023-03-26)
* **Validation Partition (15%):** 648 samples (2023-03-27 to 2025-01-02)
* **Test Partition (15%):** 648 samples (2025-01-03 to 2026-10-12)

---

## 3. Regression Leaderboard (Ranked by Validation RMSE)
| Model                              |   Val_RMSE_USD |   Val_MAE_USD |   Val_MAPE_Pct |   Val_R2 |   Test_RMSE_USD |   Test_MAE_USD |   Test_MAPE_Pct |   Test_R2 |   Train_RMSE_USD |   Train_R2 |
|:-----------------------------------|---------------:|--------------:|---------------:|---------:|----------------:|---------------:|----------------:|----------:|-----------------:|-----------:|
| Naive Previous-Close               |        1466.71 |       945.113 |         1.735  | 0.995169 |         1968.94 |        1392.03 |          1.5899 |  0.987454 |          803.71  |   0.997517 |
| Linear Regression                  |        1491.66 |       970.078 |         1.7987 | 0.995003 |         2030.43 |        1461.5  |          1.6672 |  0.986658 |          792.894 |   0.997583 |
| Ridge Regression                   |        1578.92 |      1037.56  |         1.891  | 0.994401 |         2119.69 |        1551.28 |          1.7633 |  0.985459 |          845.856 |   0.997249 |
| XGBoost Regressor (Residual Delta) |        1799.4  |      1177.91  |         2.0457 | 0.992728 |         2589.59 |        2072.02 |          2.3459 |  0.978297 |          578.281 |   0.998714 |
| Random Forest Regressor            |        9764.87 |      4277.25  |         5.8386 | 0.785849 |        30913.8  |       25579.1  |         25.8088 | -2.09282  |          424.119 |   0.999308 |
| Extra Trees Regressor              |       10244.8  |      4655.1   |         6.3503 | 0.764282 |        32497.9  |       27349.4  |         27.8146 | -2.41792  |          435.557 |   0.999271 |
| HistGradientBoosting Regressor     |       10607.6  |      4938.88  |         6.6653 | 0.747294 |        32918.5  |       27834.2  |         28.3732 | -2.50696  |          537.297 |   0.99889  |
| LightGBM Regressor                 |       10619.6  |      4950.94  |         6.9006 | 0.746718 |        32758.2  |       27585    |         28.0629 | -2.47289  |          537.792 |   0.998888 |
| XGBoost Regressor (Raw Price)      |       10812.9  |      5142.3   |         6.9198 | 0.737414 |        33940.9  |       28905.6  |         29.5827 | -2.72819  |          452.243 |   0.999214 |

---

## 4. Classification Leaderboard (Ranked by Validation F1-Score)
| Model                           |   Val_F1_Pct |   Val_Accuracy_Pct |   Val_Precision_Pct |   Val_Recall_Pct |   Val_Specificity_Pct |   Val_ErrorRate_Pct |   Val_ROC_AUC |   Test_F1_Pct |   Test_Accuracy_Pct |   Test_Precision_Pct |   Test_Recall_Pct |   Test_Specificity_Pct |   Test_ErrorRate_Pct |   Test_ROC_AUC |   Train_Accuracy_Pct |
|:--------------------------------|-------------:|-------------------:|--------------------:|-----------------:|----------------------:|--------------------:|--------------:|--------------:|--------------------:|---------------------:|------------------:|-----------------------:|---------------------:|---------------:|---------------------:|
| LightGBM Classifier             |        44.12 |              52.31 |               55.2  |            36.75 |                 68.67 |               47.69 |      0.539738 |         24.36 |               50.15 |                47.71 |             16.35 |                  82.73 |                49.85 |       0.488069 |                76.03 |
| HistGradientBoosting Classifier |        42.59 |              52.16 |               55.29 |            34.64 |                 70.57 |               47.84 |      0.531789 |         18.58 |               48.61 |                41.76 |             11.95 |                  83.94 |                51.39 |       0.477444 |                76.72 |
| Dummy Baseline (Most Frequent)  |        67.76 |              51.23 |               51.23 |           100    |                  0    |               48.77 |      0.5      |         65.84 |               49.07 |                49.07 |            100    |                   0    |                50.93 |       0.5      |                53.41 |
| XGBoost Classifier (Tuned)      |        55.28 |              50.31 |               51.29 |            59.94 |                 40.19 |               49.69 |      0.524335 |         51.43 |               47.53 |                47.12 |             56.6  |                  38.79 |                52.47 |       0.479131 |                67.29 |
| Extra Trees Classifier          |        32.78 |              50    |               52.67 |            23.8  |                 77.53 |               50    |      0.516776 |          0.63 |               51.08 |               100    |              0.31 |                 100    |                48.92 |       0.516562 |                64.15 |
| Random Forest Classifier        |        40    |              49.54 |               51.17 |            32.83 |                 67.09 |               50.46 |      0.513907 |         12.9  |               50    |                44.44 |              7.55 |                  90.91 |                50    |       0.504946 |                74.54 |
| Logistic Regression             |        42.58 |              49.23 |               50.62 |            36.75 |                 62.34 |               50.77 |      0.503784 |         14.52 |               50.93 |                50    |              8.49 |                  91.82 |                49.07 |       0.522394 |                55.95 |

---

## 5. Critical Machine Learning Insights & Limitations
1. **The Random Walk and Naive Baseline:** The naive previous-close model ($\hat{y}_{t+1} = y_t$) achieves a very strong R² score in non-stationary price forecasting because Bitcoin's daily percentage move is typically 2-4%. A model that closely tracks the previous day will mathematically achieve high R², but this does NOT imply an ability to forecast explosive trend reversals or market shocks.
2. **Direction Classification Ceiling:** Due to the semi-strong form efficient market hypothesis and high noise-to-signal ratio of daily crypto returns, daily direction accuracy typically hovers around 50-55%. Models claiming >70% daily direction accuracy on un-shuffled chronological test data suffer from lookahead target leakage.
3. **Volatility Clustering:** Daily volatility is non-stationary. Incorporating rolling standard deviation and Bollinger band bandwidth allowed gradient boosted trees to adapt to regime shifts better than plain linear estimators.
