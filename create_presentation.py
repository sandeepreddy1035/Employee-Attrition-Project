"""
Script to generate the complete PowerPoint presentation (.pptx)
for Employee Attrition Prediction with custom styling, high-res visual embeds,
and all required academic and technical sections.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

BASE_DIR = Path(__file__).resolve().parent
VIZ_DIR = BASE_DIR / "outputs" / "visualizations"
PPTX_PATH = BASE_DIR / "Employee_Attrition_Prediction_Presentation.pptx"

# Initialize presentation with 16:9 widescreen dimensions
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

# Color Palette
NAVY = RGBColor(30, 58, 138)       # #1E3A8A - Header & Accents
DARK_GREY = RGBColor(55, 65, 81)   # #374151 - Body Text
LIGHT_BG = RGBColor(243, 244, 246) # #F3F4F6 - Card Background
WHITE = RGBColor(255, 255, 255)
ACCENT_BLUE = RGBColor(37, 99, 235)# #2563EB - Highlights
BORDER_COLOR = RGBColor(209, 213, 219) # #D1D5DB


def add_slide_header(slide, title_text, category_text=None):
    """Adds a standardized top banner and title to a slide."""
    # Top banner bar
    top_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = ACCENT_BLUE
    top_bar.line.color.rgb = ACCENT_BLUE

    # Title box
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.9))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    if category_text:
        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_BLUE
        p_cat.font.name = "Arial"
        p_title = tf.add_paragraph()
    else:
        p_title = tf.paragraphs[0]

    p_title.text = title_text
    p_title.font.size = Pt(24)
    p_title.font.bold = True
    p_title.font.color.rgb = NAVY
    p_title.font.name = "Arial"


def add_content_card(slide, left, top, width, height, title=None, items=None, bg_color=LIGHT_BG):
    """Draws a rounded card box with text bullets or paragraphs."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = BORDER_COLOR
    card.line.width = Pt(1)

    tb = slide.shapes.add_textbox(left + Inches(0.3), top + Inches(0.25), width - Inches(0.6), height - Inches(0.5))
    tf = tb.text_frame
    tf.word_wrap = True

    if title:
        p_title = tf.paragraphs[0]
        p_title.text = title
        p_title.font.size = Pt(16)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY
        p_title.font.name = "Arial"
        p_title.space_after = Pt(10)

    if items:
        for idx, item in enumerate(items):
            p = tf.add_paragraph() if (title or idx > 0) else tf.paragraphs[0]
            p.text = f"•  {item}" if not item.startswith("•") else item
            p.font.size = Pt(13)
            p.font.color.rgb = DARK_GREY
            p.font.name = "Calibri"
            p.space_after = Pt(8)

    return card


# =============================================================================
# SLIDE 1: TITLE SLIDE
# =============================================================================
slide_1 = prs.slides.add_slide(blank_layout)

# Decorative background accents
bg = slide_1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(248, 250, 252)
bg.line.fill.background()

accent_bar = slide_1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.4), Inches(7.5))
accent_bar.fill.solid()
accent_bar.fill.fore_color.rgb = NAVY
accent_bar.line.fill.background()

tb = slide_1.shapes.add_textbox(Inches(1.5), Inches(1.8), Inches(10.5), Inches(3.5))
tf = tb.text_frame
tf.word_wrap = True

p_sub = tf.paragraphs[0]
p_sub.text = "MACHINE LEARNING PROJECT PRESENTATION"
p_sub.font.size = Pt(14)
p_sub.font.bold = True
p_sub.font.color.rgb = ACCENT_BLUE
p_sub.space_after = Pt(14)

p_title = tf.add_paragraph()
p_title.text = "Employee Attrition Prediction"
p_title.font.size = Pt(40)
p_title.font.bold = True
p_title.font.color.rgb = NAVY
p_title.space_after = Pt(12)

p_desc = tf.add_paragraph()
p_desc.text = "A Machine Learning Classification Framework to Predict Employee Turnover & Drive Proactive Workforce Retention"
p_desc.font.size = Pt(17)
p_desc.font.color.rgb = DARK_GREY
p_desc.space_after = Pt(28)

p_meta = tf.add_paragraph()
p_meta.text = "Dataset: IBM HR Analytics  |  Technologies: Python, Scikit-learn, Pandas, Streamlit"
p_meta.font.size = Pt(13)
p_meta.font.color.rgb = RGBColor(107, 114, 128)


# =============================================================================
# SLIDE 2: PRESENTATION OUTLINE (Matching user's uploaded image)
# =============================================================================
slide_2 = prs.slides.add_slide(blank_layout)
add_slide_header(slide_2, "Presentation Outline", "Agenda")

add_content_card(
    slide_2,
    Inches(0.8),
    Inches(1.5),
    Inches(11.7),
    Inches(5.3),
    title="Core Topics Covered in this Presentation",
    items=[
        "Abstract — Executive overview and high-level outcomes",
        "Problem Statement — The operational and financial crisis of unplanned turnover",
        "Introduction — Strategic role of AI and predictive analytics in Human Resources",
        "Literature Review — Synthesis of academic benchmarks and ML algorithms in HR",
        "Data Source — IBM HR Analytics dataset, attribute breakdown, and class imbalance",
        "Methodology — Domain Knowledge: Empirical drivers behind employee turnover",
        "Methodology — Architecture Diagram: End-to-end data pipeline from raw ingestion to inference",
        "Methodology — Working Process & Selected Methods: Preprocessing, model training, and tuning",
        "Experimental Results — Evaluation metrics on 15% unseen test partition (Recall, ROC-AUC)",
        "Demonstration & Retention Playbook — Local Streamlit portal and strategic recommendations",
    ],
)


# =============================================================================
# SLIDE 3: ABSTRACT
# =============================================================================
slide_3 = prs.slides.add_slide(blank_layout)
add_slide_header(slide_3, "Abstract", "Executive Overview")

add_content_card(
    slide_3,
    Inches(0.8),
    Inches(1.5),
    Inches(6.6),
    Inches(5.3),
    title="Research Synopsis",
    items=[
        "Employee attrition is a critical threat to enterprise efficiency, with replacement expenses frequently reaching 1.5x to 2x an employee's annual compensation.",
        "This project presents an end-to-end supervised machine learning system engineered to predict employee turnover risk from 30 multi-dimensional workplace indicators.",
        "The model is developed on the 1,470-record IBM HR Analytics dataset, addressing an inherent 5.2 : 1 class imbalance through balanced cost-sensitive learning.",
        "A zero-data-leakage pipeline enforces that all scaling and categorical encodings are fitted strictly on the 70% training split.",
        "Rigorous 5-fold cross-validation and unbiased evaluation on a 15% held-out test set confirm that the champion model successfully captures 72.22% of actual departures with an ROC-AUC of 0.8365.",
        "The trained system is deployed as an interactive Streamlit decision-support dashboard providing real-time risk scores and automated retention diagnostics.",
    ],
)

add_content_card(
    slide_3,
    Inches(7.7),
    Inches(1.5),
    Inches(4.8),
    Inches(5.3),
    title="Key Project Highlights",
    items=[
        "Dataset Volume: 1,470 employees across 30 predictive features",
        "Class Imbalance: 83.9% Stayed vs. 16.1% Departed",
        "Data Splitting: 70% Train, 15% Validation, 15% Unseen Test",
        "Evaluated Models: Logistic Regression, Decision Tree, Random Forest",
        "Champion Model: Balanced Logistic Regression Pipeline",
        "Test Recall (Sensitivity): 72.22% (26 of 36 actual leavers caught)",
        "Test ROC-AUC: 0.8365 (High discriminative capacity)",
        "Deployment: Interactive local Streamlit web application",
    ],
    bg_color=RGBColor(239, 246, 255),
)


# =============================================================================
# SLIDE 4: PROBLEM STATEMENT
# =============================================================================
slide_4 = prs.slides.add_slide(blank_layout)
add_slide_header(slide_4, "Problem Statement", "The Business Challenge")

add_content_card(
    slide_4,
    Inches(0.8),
    Inches(1.5),
    Inches(5.6),
    Inches(5.3),
    title="The Industry Problem: Unplanned Turnover",
    items=[
        "Severe Financial Loss: Replacing a specialized knowledge worker costs between 50% to 200% of their annual salary in search fees, signing bonuses, and lost productivity.",
        "Brain Drain: Departures cause irreversible erosion of institutional knowledge, customer relationships, and proprietary technical skills.",
        "Team Burnout Contagion: When key personnel leave, remaining team members absorb excess workload, often triggering cascade resignations.",
        "Reactive Retention Failure: Exit interviews occur after decisions are finalized; counter-offers fail >70% of the time within 12 months.",
    ],
)

add_content_card(
    slide_4,
    Inches(6.8),
    Inches(1.5),
    Inches(5.7),
    Inches(5.3),
    title="The Technical & Machine Learning Challenge",
    items=[
        "Severe Class Imbalance: Attrition is an infrequent event (~16% positive). Standard algorithms achieve deceptive 84% accuracy by simply predicting no one leaves.",
        "High Cost of False Negatives: Missing an employee who leaves is catastrophic; reaching out to an employee who stays carries negligible cost. Recall must take precedence over accuracy.",
        "Multi-Modal Feature Complexity: Tabular HR data contains mixtures of numerical rates, discrete satisfaction ratings, and high-cardinality nominal roles.",
        "Data Leakage Pitfalls: Preprocessing transformations must be fitted solely on training splits to prevent over-optimistic generalization claims.",
    ],
    bg_color=RGBColor(254, 242, 242),
)


# =============================================================================
# SLIDE 5: INTRODUCTION
# =============================================================================
slide_5 = prs.slides.add_slide(blank_layout)
add_slide_header(slide_5, "Introduction", "Context & Objectives")

add_content_card(
    slide_5,
    Inches(0.8),
    Inches(1.5),
    Inches(5.6),
    Inches(5.3),
    title="Human Capital & Predictive AI",
    items=[
        "Modern Human Resources has evolved from traditional record-keeping into a strategic, data-driven discipline known as People Analytics.",
        "Employee resignation is rarely an impulsive choice; it is typically preceded by measurable behavioral and structural signals over months.",
        "By analyzing historical employee indicators (compensation, tenure, overtime, promotion cycles, managerial relationship), predictive algorithms can identify vulnerable employees before resignation letters are submitted.",
        "This empowers talent managers to shift from reactive damage control to proactive, empathetic retention conversations.",
    ],
)

add_content_card(
    slide_5,
    Inches(6.8),
    Inches(1.5),
    Inches(5.7),
    Inches(5.3),
    title="Primary Objectives of this Project",
    items=[
        "1. Automated Ingestion & Cleaning: Clean, validate, and standardize tabular HR datasets without hardcoded heuristics.",
        "2. Empirical Pattern Discovery: Uncover statistical correlations across tenure, compensation, work-life balance, and departments.",
        "3. Leak-Free Pipeline Architecture: Encapsulate transformations inside Scikit-learn pipelines fitted strictly on training data.",
        "4. Cost-Sensitive Model Optimization: Train and tune multiple classifiers (Logistic Regression, Decision Trees, Random Forests) prioritizing Attrition Recall.",
        "5. Local Interactive Decision Support: Deploy a Streamlit web portal with live risk gauges and tailored retention playbooks.",
    ],
    bg_color=RGBColor(240, 253, 244),
)


# =============================================================================
# SLIDE 6: LITERATURE REVIEW
# =============================================================================
slide_6 = prs.slides.add_slide(blank_layout)
add_slide_header(slide_6, "Literature Review", "Related Works & Theoretical Foundations")

add_content_card(
    slide_6,
    Inches(0.8),
    Inches(1.5),
    Inches(5.6),
    Inches(5.3),
    title="Organizational & Theoretical Foundations",
    items=[
        "Herzberg's Two-Factor Theory (1959): Highlights hygiene factors (salary, supervision) as baseline retention essentials and motivators (recognition, growth) as long-term loyalty drivers.",
        "Mobley's Intermediate Linkages Model (1977): Details the cognitive progression from initial job dissatisfaction to active job seeking and ultimate resignation.",
        "March & Simon's Participation Theory: Demonstrates that employee turnover is governed by the perceived desirability of leaving paired with the perceived ease of movement in the market.",
        "Modern HR Analytics Consensus: Research consistently demonstrates that overtime, supervisor tenure, and salary stagnation are the primary predictive signals in corporate knowledge firms.",
    ],
)

add_content_card(
    slide_6,
    Inches(6.8),
    Inches(1.5),
    Inches(5.7),
    Inches(5.3),
    title="Machine Learning in HR Attrition Literature",
    items=[
        "Logistic Regression Benchmarks: Extensively cited in human resource studies due to high interpretability, calibrated probability outputs, and linear coefficients.",
        "Tree-Based Ensembles (Breiman et al.): Random Forests and Gradient Boosting provide non-linear feature split boundaries and capture complex multi-variable interactions.",
        "Imbalance Mitigation Strategies: Published literature emphasizes cost-sensitive learning (loss weighting) and stratified sampling as superior to naive oversampling (SMOTE) for tabular HR data.",
        "Research Gap Addressed in this Project: Many existing studies report inflated accuracies by failing to isolate test data during preprocessing. Our project enforces rigorous leak-free pipeline encapsulation.",
    ],
    bg_color=RGBColor(254, 249, 195),
)


# =============================================================================
# SLIDE 7: DATA SOURCE
# =============================================================================
slide_7 = prs.slides.add_slide(blank_layout)
add_slide_header(slide_7, "Data Source & Characteristics", "IBM HR Analytics Dataset")

add_content_card(
    slide_7,
    Inches(0.8),
    Inches(1.5),
    Inches(5.6),
    Inches(5.3),
    title="Dataset Attributes & Telemetry",
    items=[
        "Source: IBM HR Analytics Employee Attrition & Performance dataset published on Kaggle.",
        "Record Count: 1,470 total observations representing diverse professional profiles.",
        "Raw Attributes: 35 features covering personal demographics, role characteristics, compensation metrics, and subjective survey evaluations.",
        "Target Variable: Attrition ('Yes' = 237, 'No' = 1,233).",
        "Constant Columns Removed: EmployeeCount (always 1), Over18 (always 'Y'), StandardHours (always 80) — dropped due to zero variance.",
        "Identifier Removed: EmployeeNumber (dropped to eliminate synthetic artifact memorization).",
        "Cleaned Matrix: 30 predictive features (23 numeric, 7 categorical).",
    ],
)

# Embed image of Attrition Distribution if available
img_dist = VIZ_DIR / "1_attrition_distribution.png"
if img_dist.exists():
    slide_7.shapes.add_picture(str(img_dist), Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
else:
    add_content_card(slide_7, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3), title="Class Imbalance", items=["16.1% Departed (237) vs 83.9% Stayed (1,233)"])


# =============================================================================
# SLIDE 8: METHODOLOGY --- DOMAIN KNOWLEDGE
# =============================================================================
slide_8 = prs.slides.add_slide(blank_layout)
add_slide_header(slide_8, "Methodology: Domain Knowledge", "Understanding Workplace Dynamics")

add_content_card(
    slide_8,
    Inches(0.8),
    Inches(1.5),
    Inches(5.6),
    Inches(5.3),
    title="Key Empirical Attrition Drivers",
    items=[
        "1. Workload Exhaustion (Overtime): Employees logging overtime exhibit a 30.5% attrition rate compared to only 10.4% for non-overtime peers (~3x increase).",
        "2. Compensation Inequity: Median monthly income for departed staff is ~$3,200 vs ~$4,900 for retained employees. Below $3,500/month, attrition peaks sharply.",
        "3. Early-Career Vulnerability: Turnover is highest during years 0 to 2 at the organization, and workers under 25 experience >35% attrition.",
        "4. Supervisory Relationship: Short tenure with current manager (<1 year) paired with multi-year company tenure correlates with elevated departure risk.",
        "5. Career Stagnation: Stagnating 5+ years without promotion triggers voluntary transitions to competitors.",
    ],
)

# Embed Feature Importance chart
img_fi = VIZ_DIR / "feature_importance.png"
if img_fi.exists():
    slide_8.shapes.add_picture(str(img_fi), Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))


# =============================================================================
# SLIDE 9: METHODOLOGY --- ARCHITECTURE DIAGRAM
# =============================================================================
slide_9 = prs.slides.add_slide(blank_layout)
add_slide_header(slide_9, "Methodology: System Architecture Diagram", "Complete Machine Learning Pipeline Flow")

# Flowchart boxes on Slide 9
boxes = [
    ("1. DATA COLLECTION", "Auto-load CSV, inspect schema, check nulls & target balance", Inches(0.8), Inches(1.8)),
    ("2. PREPROCESSING", "Drop 4 ID/constant cols, map binary target (Yes=1, No=0)", Inches(3.8), Inches(1.8)),
    ("3. STRATIFIED SPLIT", "70% Train (1,029), 15% Val (220), 15% Unseen Test (221)", Inches(6.8), Inches(1.8)),
    ("4. LEAK-FREE PIPELINE", "StandardScaler (numeric) + OneHotEncoder (categorical)", Inches(9.8), Inches(1.8)),
    ("5. MODEL TRAINING", "Logistic Regression, Decision Tree, Random Forest (Balanced)", Inches(0.8), Inches(4.3)),
    ("6. 5-FOLD CV & VAL", "Evaluate Recall, F1, ROC-AUC; select Champion model", Inches(3.8), Inches(4.3)),
    ("7. UNSEEN TESTING", "Final unbiased evaluation on held-out 15% test set", Inches(6.8), Inches(4.3)),
    ("8. STREAMLIT APP", "Interactive web UI with live risk gauge & retention advice", Inches(9.8), Inches(4.3)),
]

for title, desc, left, top in boxes:
    shape = slide_9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(2.7), Inches(2.0))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(239, 246, 255)
    shape.line.color.rgb = ACCENT_BLUE
    shape.line.width = Pt(1.5)

    tf = shape.text_frame
    tf.word_wrap = True
    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.bold = True
    p0.font.size = Pt(13)
    p0.font.color.rgb = NAVY
    p0.space_after = Pt(8)

    p1 = tf.add_paragraph()
    p1.text = desc
    p1.font.size = Pt(11)
    p1.font.color.rgb = DARK_GREY


# =============================================================================
# SLIDE 10: METHODOLOGY --- WORKING PROCESS / SELECTED METHODS
# (Matching user's uploaded image 'Working Process / Selected Methods')
# =============================================================================
slide_10 = prs.slides.add_slide(blank_layout)
add_slide_header(slide_10, "Methodology: Working Process / Selected Methods", "Algorithmic Implementation")

add_content_card(
    slide_10,
    Inches(0.8),
    Inches(1.5),
    Inches(5.6),
    Inches(5.3),
    title="Data Processing & Validation Protocol",
    items=[
        "Stratified Data Partitioning: Enforced a 70% Train, 15% Validation, and 15% Test split preserving the exact 16.1% attrition ratio in every split.",
        "Zero-Data-Leakage Encapsulation: Built unified Scikit-learn Pipelines pairing ColumnTransformer directly with each classifier so transformers never touch test data.",
        "Imputation Resilience: SimpleImputer with median strategy for continuous fields and most_frequent for nominal categories.",
        "5-Fold Stratified Cross-Validation: Evaluated candidate stability across 5 independent training folds to prevent selection of overfitted models.",
        "Composite Selection Criterion: Champion selected using 0.40 * Recall + 0.35 * F1 + 0.25 * ROC-AUC.",
    ],
)

add_content_card(
    slide_10,
    Inches(6.8),
    Inches(1.5),
    Inches(5.7),
    Inches(5.3),
    title="Selected Classification Algorithms",
    items=[
        "1. Logistic Regression (Selected Champion):",
        "    • Config: L2 regularization, C=0.1, liblinear solver, class_weight='balanced'.",
        "    • Strength: High interpretability, calibrated probabilistic outputs, strong test generalization without overfitting.",
        "2. Decision Tree Classifier:",
        "    • Config: max_depth=5, min_samples_split=10, min_samples_leaf=5, balanced.",
        "    • Strength: Captures non-linear decision thresholds; prone to high variance if unconstrained.",
        "3. Random Forest Classifier:",
        "    • Config: 150 estimators, max_depth=8, min_samples_split=8, balanced.",
        "    • Strength: Bagged ensemble reducing variance; exhibits high training fit.",
    ],
    bg_color=RGBColor(240, 253, 244),
)


# =============================================================================
# SLIDE 11: EXPERIMENTAL RESULTS & MODEL PERFORMANCE
# =============================================================================
slide_11 = prs.slides.add_slide(blank_layout)
add_slide_header(slide_11, "Experimental Results & Final Evaluation", "Held-Out Test Set Performance (221 Records)")

add_content_card(
    slide_11,
    Inches(0.8),
    Inches(1.5),
    Inches(5.6),
    Inches(5.3),
    title="Unseen Test Set Performance",
    items=[
        "Test Accuracy: 74.66% — Robust overall classification rate.",
        "Test Recall (Sensitivity): 72.22% — Successfully flags 26 out of 36 actual leavers on unseen data.",
        "Test Precision: 36.11% — 36% of flagged staff depart; ideal for non-punitive retention outreach.",
        "Test F1-Score: 0.4815 — Harmonic balance on the positive minority class.",
        "Test ROC-AUC: 0.8365 — Exceptional ranking discrimination across varying thresholds.",
        "5-Fold CV Mean Recall: 76.49% (±3.8%) on training data.",
        "Confusion Matrix: True Negatives = 139, True Positives = 26, False Positives = 46, False Negatives = 10.",
    ],
)

# Embed ROC Curve image
img_roc = VIZ_DIR / "roc_curve_test.png"
if img_roc.exists():
    slide_11.shapes.add_picture(str(img_roc), Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))


# =============================================================================
# SLIDE 12: STREAMLIT DASHBOARD & RETENTION PLAYBOOK
# =============================================================================
slide_12 = prs.slides.add_slide(blank_layout)
add_slide_header(slide_12, "Streamlit Decision Support Dashboard", "Deployment & Retention Playbook")

add_content_card(
    slide_12,
    Inches(0.8),
    Inches(1.5),
    Inches(5.6),
    Inches(5.3),
    title="Interactive Streamlit Application Features",
    items=[
        "Multi-Tab Navigation: Overview, EDA Gallery, Validation Benchmarks, Final Test Evaluation, Feature Importance, Live Prediction, and Retention Playbook.",
        "Dual Currency Flexibility: Supports Indian Rupees (₹ INR with LPA annual equivalents) and US Dollars ($ USD) with seamless real-time normalization.",
        "Automated Rate Estimation: Users enter Monthly Salary; internal Daily and Hourly rates are auto-calculated in the background.",
        "Live Risk Categorization: Instant probability gauge assigning Low (<30%), Moderate (30-50%), High (50-70%), and Critical (>70%) risk tiers.",
        "Automated Risk Diagnostics: Scans individual profiles and highlights active departure drivers (e.g. overtime, salary, commute).",
    ],
)

add_content_card(
    slide_12,
    Inches(6.8),
    Inches(1.5),
    Inches(5.7),
    Inches(5.3),
    title="Actionable HR Retention Playbook",
    items=[
        "1. Workload Governance: Institute mandatory reviews when overtime exceeds 10 hours/week; rotate high-pressure assignments.",
        "2. First-Year Onboarding Mentorship: Implement 30-60-90 day check-ins to curb the high 0-2 year tenure turnover spike.",
        "3. Sales Enablement: Rebalance quota realism and commission tiering to address the 39.8% Sales Rep attrition rate.",
        "4. Manager Transition Briefings: Facilitate structured transition discussions when leadership personnel change.",
        "5. Commute Flexibility: Provide hybrid or remote options for employees commuting >15 miles.",
        "6. Equity Participation: Expand Stock Option participation to lock in long-term employee commitment.",
    ],
    bg_color=RGBColor(239, 246, 255),
)


# =============================================================================
# SLIDE 13: THANK YOU SLIDE
# =============================================================================
slide_13 = prs.slides.add_slide(blank_layout)

bg_last = slide_13.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
bg_last.fill.solid()
bg_last.fill.fore_color.rgb = RGBColor(248, 250, 252)
bg_last.line.fill.background()

accent_bottom = slide_13.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.35), Inches(13.333), Inches(0.15))
accent_bottom.fill.solid()
accent_bottom.fill.fore_color.rgb = ACCENT_BLUE
accent_bottom.line.fill.background()

tb_ty = slide_13.shapes.add_textbox(Inches(1.5), Inches(1.8), Inches(10.5), Inches(4.5))
tf_ty = tb_ty.text_frame
tf_ty.word_wrap = True

p_ty = tf_ty.paragraphs[0]
p_ty.text = "Thank You!"
p_ty.font.size = Pt(48)
p_ty.font.bold = True
p_ty.font.color.rgb = NAVY
p_ty.space_after = Pt(14)

p_qa = tf_ty.add_paragraph()
p_qa.text = "Questions & Discussion"
p_qa.font.size = Pt(22)
p_qa.font.bold = True
p_qa.font.color.rgb = ACCENT_BLUE
p_qa.space_after = Pt(24)

p_details = tf_ty.add_paragraph()
p_details.text = (
    "Project Title: Employee Attrition Prediction\n"
    "GitHub Repository: https://github.com/sandeepreddy1035/Employee-Attrition-Project\n"
    "Technologies: Python 3.13 | Scikit-learn | Pandas | Streamlit | Joblib\n"
    "Live Dashboard: Localhost:8501"
)
p_details.font.size = Pt(15)
p_details.font.color.rgb = DARK_GREY

# Save presentation
prs.save(str(PPTX_PATH))
print(f"Presentation successfully generated at: {PPTX_PATH}")
