# Employee Attrition Prediction System

An end-to-end Machine Learning project developed to predict employee turnover risk using organizational and demographic indicators from the **IBM HR Analytics Employee Attrition & Performance** dataset.

The system features a leak-free modular pipeline architecture, cost-sensitive classification modeling to handle class imbalance, comprehensive exploratory data analysis, leak-free feature selection, unseen test evaluation, and an interactive Streamlit web dashboard for live human resource decision support.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Summary](#dataset-summary)
3. [System Architecture](#system-architecture)
4. [Modular Pipeline Structure](#modular-pipeline-structure)
5. [Machine Learning Methodology](#machine-learning-methodology)
6. [Empirical Results & Evaluation](#empirical-results--evaluation)
7. [Installation & Setup](#installation--setup)
8. [How to Run the Project](#how-to-run-the-project)
9. [Streamlit Web Application](#streamlit-web-application)
10. [Limitations & Future Enhancements](#limitations--future-enhancements)
11. [Project Structure](#project-structure)

---

## Project Overview
Unplanned employee attrition creates acute operational friction, institutional memory drain, and replacement costs typically ranging between 1.5x to 2x an employee's annual compensation. 

This project aims to:
- Predict whether an individual employee is likely to leave (`Attrition = Yes`) or stay (`Attrition = No`).
- Quantify individualized attrition probability risk (Low, Moderate, High, Critical).
- Pinpoint the primary institutional and psychological drivers behind departure risk (overtime, compensation disparities, supervisory friction, early-career mobility).
- Equip talent leaders with proactive, data-driven retention playbooks.

---

## Dataset Summary
- **Source**: IBM HR Analytics Employee Attrition & Performance Dataset (Kaggle).
- **Records**: 1,470 employee entries.
- **Attributes**: 35 original features (30 predictive features after preprocessing).
- **Target Variable**: `Attrition`
  - **No (Retained)**: 1,233 records (83.88%)
  - **Yes (Departed)**: 237 records (16.12%)
- **Target Class Imbalance**: ~5.2 : 1 majority ratio.
- **Missing / Duplicate Records**: 0 nulls, 0 duplicate rows.

---

## System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION & PREPROCESSING                 │
│  - Load IBM HR CSV Dataset (1,470 rows)                                │
│  - Drop Zero-Variance/ID Columns (EmployeeNumber, EmployeeCount, etc.) │
│  - Binary Encode Target (Yes -> 1, No -> 0)                            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        STRATIFIED DATA SPLITTING                       │
│  - 70% Training Partition   (1,029 records)                            │
│  - 15% Validation Partition (220 records)                              │
│  - 15% Unseen Test Partition(221 records)                              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     LEAK-FREE PREPROCESSING PIPELINE                   │
│  - Numeric Features (23): StandardScaler()                             │
│  - Categorical Features (7): OneHotEncoder(handle_unknown='ignore')    │
│  * FITTED STRICTLY ON 70% TRAINING DATA INSIDE SKLEARN PIPELINES *     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING & VALIDATION SELECTION               │
│  - Logistic Regression (Balanced L2)                                   │
│  - Decision Tree Classifier (Balanced, depth-limited)                  │
│  - Random Forest Classifier (150 estimators, balanced)                 │
│  - 5-Fold Stratified Cross-Validation on Train Data                    │
│  - Selection Metric: Attrition Recall & F1-Score (prioritizing leavers)│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FINAL UNSEEN TEST EVALUATION                    │
│  - Evaluated on completely held-out 15% Test Set                       │
│  - Accuracy: 74.66% | Recall: 72.22% | ROC-AUC: 0.8365                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 INTERACTIVE STREAMLIT DASHBOARD (app.py)               │
│  - Tab 1: Project Overview & Data Schema                               │
│  - Tab 2: Exploratory Data Analysis & Visual Gallery                   │
│  - Tab 3: Model Comparison & Validation Metrics                        │
│  - Tab 4: Final Test Performance & Confusion Matrix                    │
│  - Tab 5: Feature Importance Ranking                                   │
│  - Tab 6: Interactive Live Prediction Portal                           │
│  - Tab 7: Strategic HR Retention Playbook                              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Modular Pipeline Structure

| Module | File | Purpose |
| :--- | :--- | :--- |
| **Module 1: Data Collection** | `src/data_collection.py` | Automatically locates and loads the CSV dataset, inspects shape, data types, and target distribution. |
| **Module 2: Data Cleaning** | `src/data_cleaning.py` | Eliminates constant/ID columns, handles encoding, and builds Scikit-learn `ColumnTransformer`. |
| **Module 3: Data Visualization** | `src/data_visualization.py` | Generates 10 high-resolution publication plots and an empirical EDA report. |
| **Module 4: Feature Selection** | `src/feature_selection.py` | Computes Mutual Information and Random Forest feature importance strictly on the training set. |
| **Module 5: Data Splitting** | `src/data_splitting.py` | Creates a stratified 70% Train, 15% Val, and 15% Test split with fixed random seed (`42`). |
| **Module 6: Model Training** | `src/model_training.py` | Encapsulates models in full Scikit-learn pipelines with `class_weight='balanced'`. |
| **Module 7: Model Validation** | `src/model_validation.py` | Evaluates candidates on validation set and 5-fold CV, selecting champion model. |
| **Module 8: Model Testing** | `src/model_testing.py` | Performs final unbiased evaluation on the unseen test set, outputting ROC curve and confusion matrix. |
| **Module 9: Prediction Service**| `src/prediction.py` | Real-time inference engine calculating attrition probability, risk tiers, and key risk factors. |
| **Module 10: Dashboard** | `app.py` | Streamlit interactive web application with multi-tab interface. |

---

## Machine Learning Methodology

### Class Imbalance Strategy
In the IBM HR dataset, only **16.12% of records are positive (Attrited)**.
- If a naive model predicts `Stay` for every employee, it achieves an artificially high accuracy of **83.88%**, but catches **0% of employees leaving**.
- In enterprise HR, the **cost of a False Negative (failing to identify an employee who quits) is vastly higher than a False Positive (engaging an employee who intended to stay)**.
- Models are trained using `class_weight='balanced'` and evaluated primarily on **Recall (Sensitivity)**, **F1-Score**, and **ROC-AUC**.

### Leak-Free Engineering
- Data splitting occurs **before** any feature transformations are fitted.
- All imputation, standard scaling, and one-hot encoding occur inside `sklearn.pipeline.Pipeline`, preventing information leakage from validation or test folds.

---

## Empirical Results & Evaluation

All performance figures below are generated directly from model execution on the **15% unseen test partition (221 records: 185 Stayers, 36 Leavers)**:

| Metric | Score | Analytical Interpretation |
| :--- | :---: | :--- |
| **Test Accuracy** | **74.66%** | Overall correct classification rate across stayers and leavers. |
| **Test Recall (Sensitivity)** | **72.22%** | **Successfully flags 26 out of 36 actual leavers** on unseen data. |
| **Test Precision** | **36.11%** | 36.1% of employees flagged as at-risk actually leave (acceptable for HR intervention). |
| **Test F1-Score** | **0.4815** | Harmonic mean balancing sensitivity and precision on the positive class. |
| **Test ROC-AUC** | **0.8365** | Strong discriminative ability to distinguish stayers from leavers across thresholds. |

### Champion Model Selection
- **Champion Selected**: **Logistic Regression** (`class_weight='balanced', C=0.1, solver='liblinear'`).
- **5-Fold Cross-Validation Recall**: **76.49% (±3.8%)** on the training set.
- Demonstrates optimal generalization without the severe leaf-level overfitting observed in unconstrained decision trees.

### Top Predictive Feature Importance (Training Set)
1. `MonthlyIncome` (Relative Score: 1.000)
2. `TotalWorkingYears` (Relative Score: 0.842)
3. `OverTime_Yes` (Relative Score: 0.729)
4. `YearsWithCurrManager` (Relative Score: 0.625)
5. `HourlyRate` (Relative Score: 0.619)
6. `YearsAtCompany` (Relative Score: 0.614)
7. `JobLevel` (Relative Score: 0.599)
8. `YearsInCurrentRole` (Relative Score: 0.596)

---

## Installation & Setup

### Prerequisites
- Windows, macOS, or Linux
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### Step-by-Step Installation

1. **Navigate to the project directory**:
   ```bash
   cd C:\Users\Varsha\.gemini\antigravity\scratch\employee_attrition_prediction
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - On Windows (PowerShell):
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - On Windows (CMD):
     ```cmd
     .\.venv\Scripts\activate.bat
     ```
   - On Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```

4. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run the Project

### 1. Dataset Verification
Verify that the IBM HR Analytics CSV dataset is present in:
```text
data/raw/employee_attrition.csv
```
*(The file is already verified and placed in the project directory).*

### 2. Execute the Entire Pipeline End-to-End
Run the master pipeline script to execute all data loading, cleaning, EDA chart generation, splitting, training, validation, testing, and sample inference:
```bash
python main.py
```
*Execution takes approximately 45-55 seconds, automatically populating the `models/` and `outputs/` directories.*

### 3. Explore the Jupyter Analysis Notebook
To examine the step-by-step narrative analysis with inline code and results:
```bash
jupyter notebook notebooks/employee_attrition_analysis.ipynb
```

---

## Streamlit Web Application

Launch the interactive web portal locally:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Dashboard Features:
1. **🏠 Project Overview**: Executive summary, dataset telemetry, target class distributions.
2. **📊 Exploratory Data Analysis**: Gallery of 10 high-resolution charts paired with business insights.
3. **⚖️ Model Validation & Selection**: Cross-validation results, model comparison bar chart, validation confusion matrices.
4. **🎯 Final Test Performance**: Unseen test metrics, ROC curve, and classification report.
5. **🔍 Feature Importance**: Visual ranking of the top turnover drivers.
6. **🔮 Live Prediction Portal**:
   - Quick-load demo profiles (High-Risk Sales Rep, Low-Risk Director, Moderate-Risk Analyst).
   - Custom attribute input form with sliders and dropdowns.
   - Real-time probability gauge, risk classification (Low, Moderate, High, Critical), and automated risk driver checklist.
7. **💡 HR Retention Playbook**: Strategic organizational retention policies.

---

## Limitations & Future Enhancements

### Limitations
1. **Correlation vs. Causation**: Observational survey data indicates associations. Interventions must be tested empirically before broad policy adoption.
2. **Synthetic Dataset Nuances**: The IBM HR dataset is semi-synthetic; while capturing real-world organizational dynamics, live enterprise telemetry requires periodic retraining.
3. **Static Demographics**: The dataset captures single snapshots in time without time-series longitudinal tracking of employee sentiment shifts.

### Future Enhancements
1. **SHAP & LIME Explainability**: Integrating SHAP TreeExplainer for individual local feature contribution plots.
2. **Survival Analysis**: Modeling time-to-attrition (Cox Proportional Hazards) to predict *when* an employee is likely to depart.
3. **Automated Feedback Loop**: Integrating HR survey APIs to dynamically adjust risk scores as new monthly pulse survey data is recorded.

---

## Project Structure

```text
employee_attrition_prediction/
│
├── data/
│   ├── raw/
│   │   └── employee_attrition.csv           # Original dataset
│   └── processed/
│       └── cleaned_employee_data.csv        # Preprocessed dataset
│
├── notebooks/
│   └── employee_attrition_analysis.ipynb    # Full narrative analysis notebook
│
├── src/
│   ├── __init__.py
│   ├── utils.py                             # Paths, logging, styling helpers
│   ├── data_collection.py                   # Module 1: Automated loading & inspection
│   ├── data_cleaning.py                     # Module 2: Cleaning & ColumnTransformer
│   ├── data_visualization.py                # Module 3: EDA & high-resolution chart generation
│   ├── feature_selection.py                 # Module 4: Mutual info & tree-based importance
│   ├── data_splitting.py                    # Module 5: Stratified 70/15/15 train/val/test split
│   ├── model_training.py                    # Module 6: Model training with balanced weights
│   ├── model_validation.py                  # Module 7: Validation metrics, tuning, selection
│   ├── model_testing.py                     # Module 8: Unseen test evaluation & ROC curve
│   └── prediction.py                        # Module 9: Inference service with risk diagnostics
│
├── models/
│   └── trained_models/                      # Persisted pipelines (.joblib)
│       ├── best_model_pipeline.joblib
│       ├── logistic_regression.joblib
│       ├── decision_tree.joblib
│       └── random_forest.joblib
│
├── outputs/
│   ├── visualizations/                      # Saved high-resolution PNG charts
│   │   ├── 1_attrition_distribution.png
│   │   ├── 2_attrition_by_overtime.png
│   │   ├── 3_attrition_by_jobsatisfaction.png
│   │   ├── 4_attrition_by_jobrole.png
│   │   ├── 5_attrition_by_monthly_income.png
│   │   ├── 6_attrition_by_age_group.png
│   │   ├── 7_attrition_by_years_at_company.png
│   │   ├── 8_attrition_by_work_life_balance.png
│   │   ├── 9_correlation_heatmap.png
│   │   ├── 10_department_and_distance.png
│   │   ├── feature_importance.png
│   │   ├── validation_confusion_matrices.png
│   │   ├── roc_curve_test.png
│   │   └── test_confusion_matrix.png
│   ├── reports/                             # Exhaustive Markdown documentation
│   │   ├── eda_report.md
│   │   ├── feature_selection_report.md
│   │   ├── model_validation_report.md
│   │   └── final_test_evaluation_report.md
│   └── metrics/                             # JSON machine-readable evaluation metrics
│       ├── validation_metrics.json
│       └── test_metrics.json
│
├── app.py                                   # Module 10: Streamlit interactive web application
├── main.py                                  # CLI orchestrator running the entire pipeline
├── requirements.txt                         # Minimum dependency specifications
├── README.md                                # Comprehensive project documentation
└── .gitignore                               # Standard Python, ML, and IDE exclusions
```
