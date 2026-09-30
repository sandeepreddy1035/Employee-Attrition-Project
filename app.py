"""
Module 10: Interactive Streamlit Web Application
Employee Attrition Prediction System - College Demonstration & HR Decision Support Dashboard
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils import VIZ_DIR, REPORTS_DIR, METRICS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.prediction import AttritionPredictor, DEFAULT_EMPLOYEE_PROFILE

# Streamlit Page Config
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor():
    """Caches the model pipeline instance."""
    return AttritionPredictor()


@st.cache_data
def load_datasets():
    """Loads raw and cleaned datasets for summary rendering."""
    raw_path = RAW_DATA_DIR / "employee_attrition.csv"
    proc_path = PROCESSED_DATA_DIR / "cleaned_employee_data.csv"
    raw_df = pd.read_csv(raw_path) if raw_path.exists() else None
    proc_df = pd.read_csv(proc_path) if proc_path.exists() else None
    return raw_df, proc_df


@st.cache_data
def load_metrics_data():
    """Loads saved validation and test metrics."""
    val_path = METRICS_DIR / "validation_metrics.json"
    test_path = METRICS_DIR / "test_metrics.json"
    val_data = json.load(open(val_path, "r")) if val_path.exists() else {}
    test_data = json.load(open(test_path, "r")) if test_path.exists() else {}
    return val_data, test_data


raw_df, proc_df = load_datasets()
val_metrics, test_metrics = load_metrics_data()

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & METRICS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/conference-call.png", width=70)
    st.title("HR Analytics AI")
    st.markdown("**Employee Attrition Prediction System**")
    st.markdown("---")

    navigation = st.radio(
        "Navigation",
        [
            "🏠 1. Project Overview & Data",
            "📊 2. Exploratory Data Analysis",
            "⚖️ 3. Model Validation & Selection",
            "🎯 4. Final Test Performance",
            "🔍 5. Feature Importance",
            "🔮 6. Live Prediction Portal",
            "💡 7. HR Retention Playbook",
        ],
    )

    st.markdown("---")
    st.caption("### 📌 Quick Model Telemetry")
    if test_metrics:
        st.write(f"**Champion Model**: {test_metrics.get('Model_Name', 'Logistic Regression').replace('_', ' ')}")
        st.write(f"**Test ROC-AUC**: `{test_metrics.get('ROC_AUC', 0):.4f}`")
        st.write(f"**Test Recall**: `{test_metrics.get('Recall', 0)*100:.1f}%`")
        st.write(f"**Test Accuracy**: `{test_metrics.get('Accuracy', 0)*100:.1f}%`")
    st.caption("Built for local deployment with Scikit-learn & Streamlit.")


# -----------------------------------------------------------------------------
# TAB 1: PROJECT OVERVIEW & DATA SUMMARY
# -----------------------------------------------------------------------------
if navigation == "🏠 1. Project Overview & Data":
    st.markdown('<div class="main-title">Employee Attrition Prediction System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Predicting employee turnover risk using machine learning and empirical workforce telemetry.</div>',
        unsafe_allow_html=True,
    )

    # Key Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">1,470</div><div class="metric-label">Total Employee Records</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">16.12%</div><div class="metric-label">Historical Attrition Rate</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">30</div><div class="metric-label">Predictive Features</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">70 / 15 / 15</div><div class="metric-label">Train / Val / Test Split</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 📋 Project Abstract & Methodology")
    st.info(
        """
        **Problem Statement**: Employee turnover presents substantial operational friction, high replacement costs (1.5x - 2x annual salary), 
        and loss of institutional memory. The objective is to build a high-performance machine learning classifier capable of 
        identifying early warning signs of attrition to allow proactive retention interventions.
        
        **Data Source**: IBM HR Analytics Employee Attrition & Performance dataset (1,470 records, 35 raw attributes).
        
        **Rigorous ML Architecture**:
        - **Data Preprocessing**: Outlier checks, removal of constant/ID columns (`EmployeeNumber`, `EmployeeCount`, `StandardHours`, `Over18`).
        - **Leak-Free Transformers**: Numerical standard scaling and categorical one-hot encoding are strictly fitted on the 70% training split.
        - **Class Imbalance Optimization**: Prioritizes Attrition Recall and F1-Score over naive accuracy via balanced weights.
        """
    )

    if raw_df is not None:
        st.markdown("### 🗃️ Dataset Preview (First 5 Records)")
        st.dataframe(raw_df.head(5), use_container_width=True)

        with st.expander("🔍 View Raw Column Schema & Types"):
            dtypes_df = pd.DataFrame({"Column": raw_df.columns, "Type": raw_df.dtypes.astype(str)})
            st.dataframe(dtypes_df, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS
# -----------------------------------------------------------------------------
elif navigation == "📊 2. Exploratory Data Analysis":
    st.markdown('<div class="main-title">Exploratory Data Analysis & Empirical Insights</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Visualizing key workplace factors associated with employee attrition.</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "⚠️ **Statistical Note (Association vs. Causation)**: Observed patterns show statistical correlations rather than direct causality. Overtime or satisfaction scores serve as proxy indicators for broader workplace dynamics."
    )

    chart_option = st.selectbox(
        "Select Visual Analysis Category",
        [
            "1. Overall Attrition Distribution",
            "2. OverTime Status Impact",
            "3. Job Satisfaction Levels",
            "4. Job Role Comparison",
            "5. Monthly Income Disparities",
            "6. Age Group Trends",
            "7. Company Tenure (Years at Company)",
            "8. Work-Life Balance Ratings",
            "9. Numerical Feature Correlation Heatmap",
            "10. Department & Commute Distance",
        ],
    )

    chart_map = {
        "1. Overall Attrition Distribution": (
            "1_attrition_distribution.png",
            "The baseline workforce exhibits an 83.88% retention rate and 16.12% turnover rate. This 5.2:1 class imbalance necessitates cost-sensitive classification models rather than accuracy-only heuristics.",
        ),
        "2. OverTime Status Impact": (
            "2_attrition_by_overtime.png",
            "Overtime is the single highest individual risk indicator. Overtime employees experience a 30.5% attrition rate compared to only 10.4% for non-overtime peers (a ~3x multiplier).",
        ),
        "3. Job Satisfaction Levels": (
            "3_attrition_by_jobsatisfaction.png",
            "A clear inverse gradient is evident: Level 1 (Low) satisfaction workers exhibit 22.8% turnover, while Level 4 (Very High) workers have only 11.3% attrition.",
        ),
        "4. Job Role Comparison": (
            "4_attrition_by_jobrole.png",
            "Turnover varies drastically by function: Sales Representatives lead attrition at 39.8%, followed by Laboratory Technicians (23.9%). Conversely, Directors and Senior Managers experience sub-5% attrition.",
        ),
        "5. Monthly Income Disparities": (
            "5_attrition_by_monthly_income.png",
            "Median monthly earnings of departed employees ($3,200) are markedly lower than retained staff ($4,900). A sharp density peak occurs below the $3,500 threshold.",
        ),
        "6. Age Group Trends": (
            "6_attrition_by_age_group.png",
            "Workers under 25 have a 35%+ departure rate due to career exploration and mobility. Turnover stabilizes significantly between ages 35 and 54 (10-13%).",
        ),
        "7. Company Tenure (Years at Company)": (
            "7_attrition_by_years_at_company.png",
            "The majority of departures occur within the first 1 to 3 years. After year 5, retention rises exponentially due to cultural embedding and vestment.",
        ),
        "8. Work-Life Balance Ratings": (
            "8_attrition_by_work_life_balance.png",
            "Level 1 ('Bad') work-life balance leads to a 31.3% attrition rate. Moderate work-life balance (levels 2-4) stabilizes attrition around 15-16%.",
        ),
        "9. Numerical Feature Correlation Heatmap": (
            "9_correlation_heatmap.png",
            "Strong collinearities exist between JobLevel, MonthlyIncome, and TotalWorkingYears (r > 0.75), which are appropriately normalized in our modeling pipelines.",
        ),
        "10. Department & Commute Distance": (
            "10_department_and_distance.png",
            "Sales department has the highest relative turnover (20.6%). Additionally, employees with long commutes (15+ miles) exhibit elevated flight risk.",
        ),
    }

    img_name, explanation = chart_map[chart_option]
    img_path = VIZ_DIR / img_name

    col_img, col_exp = st.columns([1.5, 1])
    with col_img:
        if img_path.exists():
            image = Image.open(img_path)
            st.image(image, use_container_width=True)
        else:
            st.error(f"Chart {img_name} not found in {VIZ_DIR}. Please run main.py first.")
    with col_exp:
        st.markdown("### 💡 Key Empirical Insight")
        st.info(explanation)
        st.markdown("#### HR Implication")
        st.write(
            "Organizations should target early-tenure support, review sales commission structures, monitor overtime burnout, and foster flexible work arrangements."
        )


# -----------------------------------------------------------------------------
# TAB 3: MODEL VALIDATION & COMPARISON
# -----------------------------------------------------------------------------
elif navigation == "⚖️ 3. Model Validation & Selection":
    st.markdown('<div class="main-title">Model Validation & Comparative Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Evaluation on 15% Validation Split (220 records) with 5-Fold Stratified Cross-Validation.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 🏆 Algorithm Comparison Table")
    if val_metrics and "validation_results" in val_metrics:
        v_res = val_metrics["validation_results"]
        cv_res = val_metrics.get("cv_results", {})

        table_data = []
        for name, m in v_res.items():
            c = cv_res.get(name, {})
            table_data.append(
                {
                    "Algorithm": name.replace("_", " "),
                    "Val Accuracy": f"{m['Accuracy']*100:.2f}%",
                    "Val Precision": f"{m['Precision']*100:.2f}%",
                    "Val Recall (Sensitivity)": f"{m['Recall']*100:.2f}%",
                    "Val F1-Score": f"{m['F1_Score']:.4f}",
                    "Val ROC-AUC": f"{m['ROC_AUC']:.4f}",
                    "5-Fold CV Mean Recall": f"{c.get('CV_Mean_Recall', 0)*100:.2f}%",
                    "5-Fold CV Mean F1": f"{c.get('CV_Mean_F1', 0):.4f}",
                }
            )

        comp_df = pd.DataFrame(table_data)
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("### 📊 Metric Visual Comparison")
        chart_df = pd.DataFrame(
            {
                "Algorithm": [n.replace("_", " ") for n in v_res.keys()],
                "Recall (Catching Leavers)": [m["Recall"] * 100 for m in v_res.values()],
                "F1-Score": [m["F1_Score"] * 100 for m in v_res.values()],
                "ROC-AUC": [m["ROC_AUC"] * 100 for m in v_res.values()],
                "Accuracy": [m["Accuracy"] * 100 for m in v_res.values()],
            }
        ).set_index("Algorithm")

        st.bar_chart(chart_df)

    st.markdown("### 🧩 Validation Confusion Matrices")
    cm_img = VIZ_DIR / "validation_confusion_matrices.png"
    if cm_img.exists():
        st.image(Image.open(cm_img), use_container_width=True)

    st.markdown("### 🎯 Decision Rationale for Model Selection")
    st.success(
        f"""
        **Champion Selected**: **`{val_metrics.get('champion_model', 'Logistic_Regression').replace('_', ' ')}`**
        - **Prioritizing Recall on Attrition**: In corporate HR, failing to detect an employee who will leave (False Negative) has an immediate institutional cost.
        - **Superior Cross-Validation Stability**: Logistic Regression demonstrated the highest 5-Fold Cross-Validation Recall (~76.5%) and best test generalization without overfitting.
        """
    )


# -----------------------------------------------------------------------------
# TAB 4: FINAL TEST PERFORMANCE
# -----------------------------------------------------------------------------
elif navigation == "🎯 4. Final Test Performance":
    st.markdown('<div class="main-title">Final Unseen Test Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Performance on the completely held-out 15% Test partition (221 records).</div>',
        unsafe_allow_html=True,
    )

    if test_metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Test Accuracy", f"{test_metrics['Accuracy']*100:.1f}%")
        with c2:
            st.metric("Test Recall (Sensitivity)", f"{test_metrics['Recall']*100:.1f}%", help="Correctly identified 26/36 actual leavers")
        with c3:
            st.metric("Test Precision", f"{test_metrics['Precision']*100:.1f}%")
        with c4:
            st.metric("Test F1-Score", f"{test_metrics['F1_Score']:.4f}")
        with c5:
            st.metric("Test ROC-AUC", f"{test_metrics['ROC_AUC']:.4f}")

        col_roc, col_cm = st.columns(2)
        with col_roc:
            st.markdown("#### Receiver Operating Characteristic (ROC) Curve")
            roc_path = VIZ_DIR / "roc_curve_test.png"
            if roc_path.exists():
                st.image(Image.open(roc_path), use_container_width=True)

        with col_cm:
            st.markdown("#### Test Confusion Matrix")
            test_cm_path = VIZ_DIR / "test_confusion_matrix.png"
            if test_cm_path.exists():
                st.image(Image.open(test_cm_path), use_container_width=True)

        st.markdown("### 📑 Unseen Test Classification Report")
        if "Classification_Report" in test_metrics:
            rep = test_metrics["Classification_Report"]
            rep_df = pd.DataFrame(rep).transpose().round(3)
            st.dataframe(rep_df, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 5: FEATURE IMPORTANCE
# -----------------------------------------------------------------------------
elif navigation == "🔍 5. Feature Importance":
    st.markdown('<div class="main-title">Feature Importance Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Understanding the primary drivers of employee attrition identified by Mutual Information and Random Forests.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.4, 1])
    with col1:
        fi_path = VIZ_DIR / "feature_importance.png"
        if fi_path.exists():
            st.image(Image.open(fi_path), use_container_width=True)

    with col2:
        st.markdown("### 🏅 Top Predictors Ranking")
        st.markdown(
            """
            1. **MonthlyIncome**: Primary compensation driver.
            2. **TotalWorkingYears**: Overall career maturity indicator.
            3. **OverTime (Yes/No)**: Acute workload and burnout risk.
            4. **YearsWithCurrManager**: Quality of direct supervisory relationship.
            5. **YearsAtCompany**: Organizational tenure and loyalty.
            6. **StockOptionLevel**: Long-term equity vesting incentive.
            7. **Age**: Early career vs established career dynamics.
            8. **JobRole (Sales Rep / Lab Tech)**: High burnout operational roles.
            """
        )
        st.info("Feature importance was computed strictly on the training partition to prevent test data leakage.")


# -----------------------------------------------------------------------------
# TAB 6: LIVE PREDICTION PORTAL
# -----------------------------------------------------------------------------
elif navigation == "🔮 6. Live Prediction Portal":
    st.markdown('<div class="main-title">Interactive Employee Attrition Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Enter employee attributes to evaluate retention risk and receive automated risk factor diagnostics.</div>',
        unsafe_allow_html=True,
    )

    predictor = get_predictor()

    # Presets for Quick Demo
    st.markdown("##### ⚡ Quick Load Demo Profiles")
    cp1, cp2, cp3 = st.columns(3)
    preset_choice = None
    if cp1.button("🚨 Load High-Risk Profile (Junior Overtime Sales Rep)"):
        preset_choice = "high_risk"
    if cp2.button("🛡️ Load Low-Risk Profile (Senior Director)"):
        preset_choice = "low_risk"
    if cp3.button("⚖️ Load Moderate-Risk Profile (Mid-Level Developer)"):
        preset_choice = "moderate_risk"

    defaults = DEFAULT_EMPLOYEE_PROFILE.copy()
    if preset_choice == "high_risk":
        defaults.update({
            "Age": 24,
            "JobRole": "Sales Representative",
            "Department": "Sales",
            "MonthlyIncome": 2200,
            "OverTime": "Yes",
            "JobSatisfaction": 1,
            "WorkLifeBalance": 1,
            "YearsAtCompany": 1,
            "DistanceFromHome": 24,
            "StockOptionLevel": 0,
            "BusinessTravel": "Travel_Frequently",
        })
    elif preset_choice == "low_risk":
        defaults.update({
            "Age": 50,
            "JobRole": "Research Director",
            "Department": "Research & Development",
            "MonthlyIncome": 18500,
            "OverTime": "No",
            "JobSatisfaction": 4,
            "WorkLifeBalance": 3,
            "YearsAtCompany": 18,
            "YearsWithCurrManager": 10,
            "StockOptionLevel": 2,
            "BusinessTravel": "Non-Travel",
        })
    elif preset_choice == "moderate_risk":
        defaults.update({
            "Age": 32,
            "JobRole": "Research Scientist",
            "Department": "Research & Development",
            "MonthlyIncome": 5200,
            "OverTime": "Yes",
            "JobSatisfaction": 2,
            "WorkLifeBalance": 2,
            "YearsAtCompany": 4,
            "DistanceFromHome": 10,
            "StockOptionLevel": 1,
        })

    with st.form("employee_prediction_form"):
        st.markdown("#### 1. Demographics & Workload")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            age = st.slider("Age", 18, 65, int(defaults["Age"]))
            gender = st.selectbox("Gender", ["Female", "Male"], index=1 if defaults["Gender"] == "Male" else 0)
        with col2:
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=["Single", "Married", "Divorced"].index(defaults["MaritalStatus"]))
            distance = st.slider("Distance From Home (miles)", 1, 30, int(defaults["DistanceFromHome"]))
        with col3:
            overtime = st.selectbox("Works OverTime?", ["No", "Yes"], index=1 if defaults["OverTime"] == "Yes" else 0)
            travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], index=["Non-Travel", "Travel_Rarely", "Travel_Frequently"].index(defaults["BusinessTravel"]))
        with col4:
            education = st.select_slider("Education Level", options=[1, 2, 3, 4, 5], value=int(defaults["Education"]), help="1: Below College, 2: College, 3: Bachelor, 4: Master, 5: Doctor")
            edu_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"], index=0)

        st.markdown("#### 2. Role, Department & Compensation")
        curr_choice = st.radio("Select Salary Currency:", ["🇮🇳 INR (₹ Indian Rupees)", "💵 USD ($ US Dollars)"], horizontal=True)

        col5, col6, col7 = st.columns(3)
        with col5:
            dept = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"], index=["Research & Development", "Sales", "Human Resources"].index(defaults["Department"]))
            role = st.selectbox("Job Role", [
                "Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
                "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"
            ], index=1)
            job_level = st.select_slider("Job Level", options=[1, 2, 3, 4, 5], value=int(defaults["JobLevel"]))

        with col6:
            if "INR" in curr_choice:
                # Default INR: ~₹83 per USD (e.g. $4,900/mo ~ ₹4,00,000/mo or ₹40,000 - ₹15,00,000)
                inr_default = int(defaults["MonthlyIncome"] * 83)
                salary_inr = st.number_input("Monthly Salary (₹ INR)", min_value=15000, max_value=2500000, value=inr_default, step=5000, help="Employee monthly take-home/gross pay in Indian Rupees (₹)")
                income = salary_inr / 83.0  # Normalized to underlying IBM dataset scale
                st.caption(f"Equivalent Model Scale: ~${income:,.0f} USD/mo (Annual CTC: ₹{(salary_inr*12)/100000:.1f} Lakhs/year)")
            else:
                income = st.number_input("Monthly Salary ($ USD)", min_value=1000, max_value=25000, value=int(defaults["MonthlyIncome"]), step=250, help="Employee monthly salary in US Dollars ($)")
                st.caption(f"Equivalent: ~₹{int(income * 83):,} INR/month")

            hike = st.slider("Percent Salary Hike (%)", 10, 30, int(defaults["PercentSalaryHike"]))

        with col7:
            stock = st.select_slider("Stock Option Level", options=[0, 1, 2, 3], value=int(defaults["StockOptionLevel"]))
            st.caption("0: None, 1: Basic, 2: Moderate, 3: High Equity")

        with st.expander("⚙️ Advanced Billing Rates (Optional: DailyRate & HourlyRate from IBM Dataset)"):
            st.info("The original IBM dataset tracks consultant billing rates alongside salary. By default, these are automatically proportional to monthly salary.")
            adv_c1, adv_c2 = st.columns(2)
            with adv_c1:
                auto_daily = int(min(2000, max(100, (income / 22) * 3.6)))
                daily_rate = st.number_input("Daily Billing Rate (Dataset Units)", min_value=100, max_value=2000, value=auto_daily, step=25)
            with adv_c2:
                auto_hourly = int(min(100, max(30, auto_daily / 12)))
                hourly_rate = st.slider("Hourly Billing Rate (Dataset Units)", min_value=30, max_value=100, value=auto_hourly)

        st.markdown("#### 3. Job Satisfaction & Organizational Tenure")
        col9, col10, col11, col12 = st.columns(4)
        with col9:
            satisfaction = st.select_slider("Job Satisfaction", options=[1, 2, 3, 4], value=int(defaults["JobSatisfaction"]), help="1: Low -> 4: Very High")
            env_sat = st.select_slider("Environment Satisfaction", options=[1, 2, 3, 4], value=int(defaults["EnvironmentSatisfaction"]))
        with col10:
            wlb = st.select_slider("Work-Life Balance", options=[1, 2, 3, 4], value=int(defaults["WorkLifeBalance"]), help="1: Bad -> 4: Best")
            involve = st.select_slider("Job Involvement", options=[1, 2, 3, 4], value=int(defaults["JobInvolvement"]))
        with col11:
            yrs_company = st.slider("Years at Company", 0, 40, int(defaults["YearsAtCompany"]))
            total_working_yrs = st.slider("Total Working Years", 0, 45, int(defaults["TotalWorkingYears"]))
        with col12:
            yrs_manager = st.slider("Years With Current Manager", 0, 20, int(defaults["YearsWithCurrManager"]))
            yrs_promo = st.slider("Years Since Last Promotion", 0, 15, int(defaults["YearsSinceLastPromotion"]))

        submitted = st.form_submit_button("🔮 Predict Employee Attrition Risk", use_container_width=True)

    if submitted:
        emp_input = {
            "Age": age,
            "Gender": gender,
            "MaritalStatus": marital,
            "DistanceFromHome": distance,
            "OverTime": overtime,
            "BusinessTravel": travel,
            "Education": education,
            "EducationField": edu_field,
            "Department": dept,
            "JobRole": role,
            "JobLevel": job_level,
            "MonthlyIncome": income,
            "DailyRate": daily_rate,
            "HourlyRate": hourly_rate,
            "MonthlyRate": 15000,
            "NumCompaniesWorked": 2,
            "PercentSalaryHike": hike,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 3,
            "StockOptionLevel": stock,
            "TotalWorkingYears": total_working_yrs,
            "TrainingTimesLastYear": 3,
            "WorkLifeBalance": wlb,
            "JobSatisfaction": satisfaction,
            "EnvironmentSatisfaction": env_sat,
            "JobInvolvement": involve,
            "YearsAtCompany": yrs_company,
            "YearsInCurrentRole": max(0, yrs_company - 1),
            "YearsSinceLastPromotion": yrs_promo,
            "YearsWithCurrManager": yrs_manager,
        }

        result = predictor.predict(emp_input)

        st.markdown("---")
        st.markdown("### 📊 Prediction Result & Risk Assessment")

        res_col1, res_col2 = st.columns([1, 1.5])
        with res_col1:
            if result["prediction_code"] == 1:
                st.error(f"### ⚠️ Status: {result['prediction_label']}")
            else:
                st.success(f"### ✅ Status: {result['prediction_label']}")

            st.metric("Attrition Probability", f"{result['attrition_probability']*100:.1f}%")
            st.metric("Risk Category", result["risk_level"])
            st.progress(result["attrition_probability"])

        with res_col2:
            st.markdown("#### 🔍 Diagnosed Risk Drivers")
            for factor in result["key_risk_factors"]:
                st.write(f"- {factor}")

            st.markdown("#### 💼 Recommended Retention Intervention")
            if result["attrition_probability"] >= 0.5:
                st.warning(
                    "**Action Plan**: Schedule an immediate confidential 1-on-1 retention conversation. Investigate workload balance, explore internal career trajectory options, and review compensation against market benchmarks."
                )
            else:
                st.info(
                    "**Action Plan**: Employee exhibits sound organizational stability. Continue standard quarterly check-ins and acknowledge recent contributions."
                )

        st.caption(f"🛡️ *{result['disclaimer']}*")


# -----------------------------------------------------------------------------
# TAB 7: HR RETENTION PLAYBOOK
# -----------------------------------------------------------------------------
elif navigation == "💡 7. HR Retention Playbook":
    st.markdown('<div class="main-title">Organizational Retention Strategies</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Data-driven guidelines for minimizing involuntary and preventable employee attrition.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            ### 1. Workload & Overtime Governance
            - **Finding**: Overtime increases attrition risk by almost 300%.
            - **Recommendation**: Institute mandatory workload reviews when overtime exceeds 10 hours/week. Rotate high-stress project duties and offer compensatory rest periods.
            
            ### 2. Early Tenure Engagement (0-2 Years)
            - **Finding**: Nearly 40% of all departures occur within the first 24 months.
            - **Recommendation**: Deploy formal mentorship programs during the first 90 days. Establish 30-60-90 day goal alignments and transparent promotion roadmaps.
            
            ### 3. Sales Force Support
            - **Finding**: Sales Representatives exhibit the highest industry turnover (39.8%).
            - **Recommendation**: Restructure quota targets to be achievable, bolster sales enablement toolsets, and provide clearer commission tiering.
            """
        )

    with c2:
        st.markdown(
            """
            ### 4. Manager-Employee Relationships
            - **Finding**: Rapid departures spike during supervisory changes (under 1 year with current manager).
            - **Recommendation**: Conduct manager transition briefings to maintain open feedback loops when leadership personnel change.
            
            ### 5. Commute & Flexibility
            - **Finding**: Commute distances above 15 miles significantly elevate resignation probability.
            - **Recommendation**: Introduce hybrid work or flexible remote days for employees with lengthy commutes.
            
            ### 6. Equity & Long-Term Incentives
            - **Finding**: Employees with Stock Option Level 0 have markedly lower retention longevity.
            - **Recommendation**: Expand equity grant eligibility and introduce tenure-based milestone rewards.
            """
        )
