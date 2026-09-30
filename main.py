"""
Main Pipeline Orchestrator for Employee Attrition Prediction
Executes all modules end-to-end:
1. Data Collection & Inspection
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis & Visualizations
4. Data Splitting (Stratified 70/15/15)
5. Feature Selection (Leak-free on Training set)
6. Model Training (Logistic Regression, Decision Tree, Random Forest)
7. Model Validation, Cross-Validation & Champion Model Selection
8. Final Unseen Test Evaluation
9. Sample Prediction Verification
"""

import sys
import time
from pathlib import Path
import pandas as pd

from src.utils import get_logger, set_plotting_theme, MODELS_DIR, VIZ_DIR, REPORTS_DIR, METRICS_DIR
from src.data_collection import load_and_inspect_data
from src.data_cleaning import clean_raw_data, get_feature_types, build_preprocessor_pipeline, save_cleaned_data
from src.data_visualization import run_eda
from src.data_splitting import split_data
from src.feature_selection import run_feature_selection
from src.model_training import train_models
from src.model_validation import run_validation
from src.model_testing import run_testing
from src.prediction import AttritionPredictor

logger = get_logger("MainPipeline")


def run_full_pipeline():
    start_time = time.time()
    logger.info("================================================================================")
    logger.info("      STARTING EMPLOYEE ATTRITION PREDICTION MACHINE LEARNING PIPELINE          ")
    logger.info("================================================================================")

    # -------------------------------------------------------------------------
    # STAGE 1: Data Collection & Inspection
    # -------------------------------------------------------------------------
    logger.info("\n>>> STAGE 1: DATA COLLECTION & INSPECTION")
    df_raw, data_summary = load_and_inspect_data()
    logger.info(f"Loaded {df_raw.shape[0]} records across {df_raw.shape[1]} raw attributes.")

    # -------------------------------------------------------------------------
    # STAGE 2: Data Cleaning & Preprocessing
    # -------------------------------------------------------------------------
    logger.info("\n>>> STAGE 2: DATA CLEANING & PREPROCESSING")
    df_clean = clean_raw_data(df_raw)
    num_cols, cat_cols = get_feature_types(df_clean)
    preprocessor = build_preprocessor_pipeline(num_cols, cat_cols)
    save_cleaned_data(df_clean)
    logger.info(f"Cleaned feature matrix: {df_clean.shape[1]-1} features, 1 binary target.")

    # -------------------------------------------------------------------------
    # STAGE 3: Exploratory Data Analysis & Visualizations
    # -------------------------------------------------------------------------
    logger.info("\n>>> STAGE 3: EXPLORATORY DATA ANALYSIS (EDA)")
    run_eda(df_clean)
    logger.info(f"Generated and saved all publication-quality charts to: {VIZ_DIR}")

    # -------------------------------------------------------------------------
    # STAGE 4: Stratified Data Splitting
    # -------------------------------------------------------------------------
    logger.info("\n>>> STAGE 4: STRATIFIED DATA SPLITTING (70% Train, 15% Val, 15% Test)")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df_clean, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42
    )

    # -------------------------------------------------------------------------
    # STAGE 5: Feature Selection (Leak-free on Training set)
    # -------------------------------------------------------------------------
    logger.info("\n>>> STAGE 5: FEATURE SELECTION & IMPORTANCE ANALYSIS")
    importance_df, top_features = run_feature_selection(X_train, y_train, preprocessor)
    logger.info(f"Top 5 predictive drivers: {top_features[:5]}")

    # -------------------------------------------------------------------------
    # STAGE 6: Model Training
    # -------------------------------------------------------------------------
    logger.info("\n>>> STAGE 6: MODEL TRAINING WITH BALANCED WEIGHTS")
    pipelines, train_perf = train_models(X_train, y_train, random_state=42)

    # -------------------------------------------------------------------------
    # STAGE 7: Model Validation & Selection
    # -------------------------------------------------------------------------
    logger.info("\n>>> STAGE 7: MODEL VALIDATION & SELECTION")
    best_name, best_pipe, val_metrics = run_validation(pipelines, X_train, y_train, X_val, y_val)
    logger.info(f"Champion model determined: {best_name}")

    # -------------------------------------------------------------------------
    # STAGE 8: Model Testing on Completely Unseen 15% Test Partition
    # -------------------------------------------------------------------------
    logger.info("\n>>> STAGE 8: FINAL UNSEEN TEST EVALUATION")
    test_metrics = run_testing(best_pipe, X_test, y_test, model_name=best_name)

    # -------------------------------------------------------------------------
    # STAGE 9: Inference Engine Verification
    # -------------------------------------------------------------------------
    logger.info("\n>>> STAGE 9: INFERENCE VERIFICATION")
    predictor = AttritionPredictor()

    test_profiles = [
        {
            "ProfileName": "High-Risk Early Career Overtime Specialist",
            "Data": {
                "Age": 24,
                "JobRole": "Sales Representative",
                "MonthlyIncome": 2100,
                "OverTime": "Yes",
                "JobSatisfaction": 1,
                "WorkLifeBalance": 1,
                "YearsAtCompany": 1,
                "DistanceFromHome": 24,
            },
        },
        {
            "ProfileName": "Low-Risk Seasoned Research Director",
            "Data": {
                "Age": 48,
                "JobRole": "Research Director",
                "MonthlyIncome": 16500,
                "OverTime": "No",
                "JobSatisfaction": 4,
                "WorkLifeBalance": 3,
                "YearsAtCompany": 15,
                "StockOptionLevel": 2,
            },
        },
    ]

    for item in test_profiles:
        res = predictor.predict(item["Data"])
        logger.info(f"\n--- Verification: {item['ProfileName']} ---")
        logger.info(f"  Prediction    : {res['prediction_label']} ({res['risk_level']})")
        logger.info(f"  Attrition Risk: {res['attrition_probability']*100:.1f}%")
        logger.info(f"  Key Drivers   : {res['key_risk_factors'][0] if res['key_risk_factors'] else 'None'}")

    elapsed = time.time() - start_time
    logger.info("\n================================================================================")
    logger.info(f"      PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS        ")
    logger.info("================================================================================")
    logger.info(f"Outputs Directory: {REPORTS_DIR.parent}")
    logger.info(f"Visualizations   : {VIZ_DIR}")
    logger.info(f"Reports          : {REPORTS_DIR}")
    logger.info(f"Metrics JSON     : {METRICS_DIR}")
    logger.info(f"Saved Models     : {MODELS_DIR}")
    logger.info("================================================================================\n")


if __name__ == "__main__":
    run_full_pipeline()
