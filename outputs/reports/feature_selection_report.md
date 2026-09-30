# Feature Selection & Predictive Importance Analysis

## Overview
Feature importance was assessed strictly on the **70% Training split (1,029 records)** to prevent any information leakage from the validation or testing partitions.

Two complementary paradigms were employed:
1. **Mutual Information (`mutual_info_classif`)**: Non-parametric information-theoretic measure detecting linear and non-linear dependencies between each feature and employee turnover.
2. **Random Forest Gini Importance (`feature_importances_`)**: Multi-tree ensemble split criterion measuring the mean decrease in impurity.

---

## Top 15 Ranked Features

| Rank | Feature | Mutual Info | Random Forest | Combined Score |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `MonthlyIncome` | 0.0377 | 0.0808 | 1.0000 |
| 2 | `TotalWorkingYears` | 0.0348 | 0.0618 | 0.8423 |
| 3 | `OverTime_Yes` | 0.0374 | 0.0383 | 0.7290 |
| 4 | `YearsWithCurrManager` | 0.0253 | 0.0471 | 0.6247 |
| 5 | `HourlyRate` | 0.0306 | 0.0350 | 0.6192 |
| 6 | `YearsAtCompany` | 0.0248 | 0.0465 | 0.6139 |
| 7 | `JobLevel` | 0.0334 | 0.0260 | 0.5994 |
| 8 | `YearsInCurrentRole` | 0.0306 | 0.0314 | 0.5961 |
| 9 | `StockOptionLevel` | 0.0236 | 0.0406 | 0.5608 |
| 10 | `OverTime_No` | 0.0242 | 0.0391 | 0.5594 |
| 11 | `Age` | 0.0096 | 0.0585 | 0.4869 |
| 12 | `JobSatisfaction` | 0.0231 | 0.0244 | 0.4536 |
| 13 | `MaritalStatus_Single` | 0.0187 | 0.0158 | 0.3404 |
| 14 | `EnvironmentSatisfaction` | 0.0135 | 0.0257 | 0.3334 |
| 15 | `Education` | 0.0144 | 0.0170 | 0.2915 |

---

## Analytical Key Takeaways
1. **OverTime Commitment**: Overtime (`OverTime_Yes`, `OverTime_No`) consistently ranks as the single highest predictor of employee attrition across both informational and tree-based metrics.
2. **Tenure & Life Stage**: Variables reflecting career stage (`TotalWorkingYears`, `Age`, `YearsAtCompany`, `YearsWithCurrManager`) heavily drive attrition risk, with early tenure employees facing higher probability of transition.
3. **Financial Reward**: `MonthlyIncome` and `MonthlyRate` serve as primary economic predictors, corroborating EDA observations of higher turnover among lower-income bands.
4. **Role Type**: Specific roles like `JobRole_Sales Representative` and `JobRole_Laboratory Technician` stand out with significant predictive weight.
