"""
Module 9: Employee Attrition Prediction Inference Service
Objective: Provide a robust, production-grade inference engine that validates employee attributes,
computes departure probabilities using the saved model pipeline, categorizes turnover risk,
and diagnoses key risk factors.
"""

from pathlib import Path
from typing import Dict, Any, Union, List
import pandas as pd
import numpy as np
import joblib

from src.utils import get_logger, MODELS_DIR, PROCESSED_DATA_DIR

logger = get_logger("PredictionService")

# Benchmark median and mode values extracted from the historical dataset for fallback validation
DEFAULT_EMPLOYEE_PROFILE = {
    "Age": 36,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 800,
    "Department": "Research & Development",
    "DistanceFromHome": 7,
    "Education": 3,
    "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": 3,
    "Gender": "Male",
    "HourlyRate": 66,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Research Scientist",
    "JobSatisfaction": 3,
    "MaritalStatus": "Married",
    "MonthlyIncome": 4900,
    "MonthlyRate": 14000,
    "NumCompaniesWorked": 2,
    "OverTime": "No",
    "PercentSalaryHike": 15,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 3,
}


class AttritionPredictor:
    """
    Loads the trained model pipeline and executes single or batch predictions.
    """

    def __init__(self, model_path: Path = None):
        if model_path is None:
            model_path = MODELS_DIR / "best_model_pipeline.joblib"
        if not model_path.exists():
            # Fallback to any trained pipeline in the directory
            available = list(MODELS_DIR.glob("*.joblib"))
            if available:
                model_path = available[0]
            else:
                raise FileNotFoundError(
                    f"No trained model pipeline found in {MODELS_DIR}. Run training and validation first."
                )

        logger.info(f"Loading prediction model from: {model_path}")
        self.pipeline = joblib.load(model_path)
        self.model_path = model_path

    def validate_and_prepare_input(self, raw_input: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """
        Validates input dictionary or DataFrame against the required 30 features.
        Populates any omitted non-essential columns with sensible population medians/modes.
        """
        if isinstance(raw_input, dict):
            input_df = pd.DataFrame([raw_input])
        else:
            input_df = raw_input.copy()

        # Remove identifier columns if erroneously supplied
        for col in ["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18", "Attrition"]:
            if col in input_df.columns:
                input_df = input_df.drop(columns=[col])

        # Fill any missing expected columns from DEFAULT_EMPLOYEE_PROFILE
        for col, def_val in DEFAULT_EMPLOYEE_PROFILE.items():
            if col not in input_df.columns:
                input_df[col] = def_val

        # Ensure correct data types
        for col, def_val in DEFAULT_EMPLOYEE_PROFILE.items():
            if isinstance(def_val, (int, float)):
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(def_val)
            else:
                input_df[col] = input_df[col].astype(str)

        # Enforce column order to match training feature set
        expected_cols = list(DEFAULT_EMPLOYEE_PROFILE.keys())
        return input_df[expected_cols]

    def identify_risk_factors(self, emp: pd.Series) -> List[str]:
        """
        Evaluates employee attributes against empirically identified attrition drivers.
        Returns bulleted human-readable risk indicators.
        """
        factors = []

        if str(emp.get("OverTime", "No")).strip().lower() == "yes":
            factors.append("Frequent Overtime: Overtime staff experience ~30.5% attrition compared to ~10.4% baseline.")

        if float(emp.get("JobSatisfaction", 3)) <= 1:
            factors.append("Low Job Satisfaction: Rating of 1/4 indicates acute turnover risk.")

        if float(emp.get("WorkLifeBalance", 3)) <= 1:
            factors.append("Poor Work-Life Balance: Rating of 1/4 correlates with a 31% departure rate.")

        if float(emp.get("MonthlyIncome", 5000)) < 3000:
            factors.append("Below-Average Compensation: Monthly income < $3,000 is strongly linked to early departure.")

        if float(emp.get("DistanceFromHome", 5)) >= 15:
            factors.append(f"Long Commute Distance: Commute of {int(emp.get('DistanceFromHome'))} miles increases burnout risk.")

        if float(emp.get("YearsWithCurrManager", 3)) < 1 and float(emp.get("YearsAtCompany", 3)) >= 2:
            factors.append("Recent Managerial Transition: Recent change in direct supervisor often triggers retention friction.")

        if float(emp.get("YearsSinceLastPromotion", 0)) >= 5:
            factors.append(f"Promotion Stagnation: {int(emp.get('YearsSinceLastPromotion'))} years since last promotion.")

        if float(emp.get("StockOptionLevel", 1)) == 0:
            factors.append("No Equity Incentive: Stock Option Level 0 offers minimal long-term organizational lock-in.")

        if not factors:
            factors.append("No critical individual risk flags detected across primary baseline indicators.")

        return factors

    def predict(self, employee_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Generates predictions and probability estimates for an employee.

        Returns:
            Dict[str, Any] containing prediction label, probabilities, risk classification,
            diagnosed risk factors, and institutional disclaimer.
        """
        clean_df = self.validate_and_prepare_input(employee_data)
        emp_series = clean_df.iloc[0]

        # Pipeline prediction
        pred = int(self.pipeline.predict(clean_df)[0])

        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(clean_df)[0]
            stay_prob = round(float(proba[0]), 4)
            leave_prob = round(float(proba[1]), 4)
        else:
            leave_prob = 1.0 if pred == 1 else 0.0
            stay_prob = 1.0 - leave_prob

        # Determine Risk Level Category
        if leave_prob >= 0.70:
            risk_level = "Critical Risk"
            risk_color = "#d9534f"
        elif leave_prob >= 0.50:
            risk_level = "High Risk"
            risk_color = "#f0ad4e"
        elif leave_prob >= 0.30:
            risk_level = "Moderate Risk"
            risk_color = "#5bc0de"
        else:
            risk_level = "Low Risk"
            risk_color = "#5cb85c"

        label = "Likely to Leave" if pred == 1 else "Likely to Stay"
        key_factors = self.identify_risk_factors(emp_series)

        return {
            "prediction_label": label,
            "prediction_code": pred,
            "attrition_probability": leave_prob,
            "retention_probability": stay_prob,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "key_risk_factors": key_factors,
            "model_used": self.model_path.stem.replace("_", " ").title(),
            "disclaimer": (
                "IMPORTANT: This prediction is an algorithmic statistical estimate for decision support "
                "and does not constitute a guaranteed outcome. HR interventions should be handled "
                "empathetically and holistically."
            ),
        }


if __name__ == "__main__":
    predictor = AttritionPredictor()
    sample_emp = {
        "Age": 29,
        "JobRole": "Sales Representative",
        "MonthlyIncome": 2400,
        "OverTime": "Yes",
        "JobSatisfaction": 1,
        "WorkLifeBalance": 1,
        "YearsAtCompany": 1,
    }
    result = predictor.predict(sample_emp)
    print("\n--- Sample Prediction Output ---")
    print(f"Prediction: {result['prediction_label']} ({result['risk_level']})")
    print(f"Attrition Risk Probability: {result['attrition_probability']*100:.1f}%")
    print("Identified Risk Factors:")
    for f in result["key_risk_factors"]:
        print(f"  - {f}")
