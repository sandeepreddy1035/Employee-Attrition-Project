"""
Module 8: Model Testing and Final Evaluation
Objective: Evaluate the chosen champion model pipeline on the completely unseen 15% test set,
plotting the ROC curve and confusion matrix, and outputting an authoritative final evaluation report.
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from src.utils import (
    get_logger,
    set_plotting_theme,
    save_plot,
    save_json,
    MODELS_DIR,
    REPORTS_DIR,
    VIZ_DIR,
)

logger = get_logger("ModelTesting")


def load_champion_model(model_path: Path = None) -> Pipeline:
    """Loads the persisted best model pipeline."""
    if model_path is None:
        model_path = MODELS_DIR / "best_model_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Champion model not found at: {model_path}. Train and validate models first.")
    logger.info(f"Loading champion model pipeline from: {model_path}")
    return joblib.load(model_path)


def test_model(
    model_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Champion Model",
) -> Dict[str, Any]:
    """
    Executes final evaluation on the 100% unseen test partition (15%).
    Generates test confusion matrix and ROC curve plots.
    Saves metrics and a final test evaluation report.
    """
    logger.info(f"Conducting final unseen test evaluation on {len(X_test)} samples...")

    # Predict
    y_pred = model_pipeline.predict(X_test)
    y_proba = (
        model_pipeline.predict_proba(X_test)[:, 1]
        if hasattr(model_pipeline.named_steps["classifier"], "predict_proba")
        else None
    )

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
    cm = confusion_matrix(y_test, y_pred)
    clf_rep_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    clf_rep_text = classification_report(y_test, y_pred, target_names=["Stayed (0)", "Left (1)"], zero_division=0)

    logger.info("--- Unseen Test Set Results ---")
    logger.info(f"  Test Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    logger.info(f"  Test Precision : {prec:.4f} ({prec*100:.2f}%)")
    logger.info(f"  Test Recall    : {rec:.4f} ({rec*100:.2f}%)")
    logger.info(f"  Test F1-Score  : {f1:.4f}")
    logger.info(f"  Test ROC-AUC   : {auc:.4f}")

    # Plot Test Confusion Matrix
    set_plotting_theme()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Predicted Stay", "Predicted Leave"],
        yticklabels=["Actual Stay", "Actual Leave"],
    )
    ax.set_title(f"Final Test Confusion Matrix\n({model_name.replace('_', ' ')})", pad=12)
    save_plot(fig, "test_confusion_matrix.png")

    # Plot ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"ROC Curve (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.50)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve - Test Set", pad=12)
        ax.legend(loc="lower right")
        save_plot(fig, "roc_curve_test.png")

    test_metrics = {
        "Model_Name": model_name,
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1_Score": round(float(f1), 4),
        "ROC_AUC": round(float(auc), 4),
        "Confusion_Matrix": cm.tolist(),
        "Classification_Report": clf_rep_dict,
        "Test_Set_Size": len(X_test),
        "Test_Positive_Samples": int((y_test == 1).sum()),
        "Test_Negative_Samples": int((y_test == 0).sum()),
    }

    # Save metrics JSON
    save_json(test_metrics, "test_metrics.json")

    # Save Markdown report
    write_final_test_report(test_metrics, clf_rep_text)

    return test_metrics


def write_final_test_report(metrics: Dict[str, Any], clf_rep_text: str) -> Path:
    """Writes the comprehensive final test evaluation report."""
    cm = metrics["Confusion_Matrix"]
    report = f"""# Final Model Evaluation Report (Unseen Test Partition)

## Executive Performance Summary
The champion machine learning model (**`{metrics['Model_Name'].replace('_', ' ')}`**) was evaluated on the **15% unseen test partition ({metrics['Test_Set_Size']} employee records)**. This partition was held out throughout the entire preprocessing, exploratory analysis, and training cycles.

| Evaluation Metric | Test Score | Strategic HR Meaning |
| :--- | :---: | :--- |
| **Accuracy** | **{metrics['Accuracy']*100:.2f}%** | Overall fraction of correct predictions across stayers and leavers. |
| **Recall (Sensitivity)** | **{metrics['Recall']*100:.2f}%** | Ability to detect employees who actually depart the organization. |
| **Precision** | **{metrics['Precision']*100:.2f}%** | Proportion of predicted departures that were genuine leavers. |
| **F1-Score** | **{metrics['F1_Score']:.4f}** | Harmonic mean of precision and recall on the positive minority class. |
| **ROC-AUC** | **{metrics['ROC_AUC']:.4f}** | Area under the ROC Curve; measures ranking discrimination quality. |

---

## Test Confusion Matrix
```text
                  Predicted Stay (0)   Predicted Leave (1)
Actual Stay (0)          {cm[0][0]:<10}          {cm[0][1]:<10} (False Positives)
Actual Leave (1)         {cm[1][0]:<10}          {cm[1][1]:<10} (True Positives)
```

- **True Negatives ({cm[0][0]})**: Retained employees correctly identified.
- **True Positives ({cm[1][1]})**: Departing employees successfully flagged for retention interventions.
- **False Positives ({cm[0][1]})**: Retained employees predicted as leaving (moderate cost: routine check-in).
- **False Negatives ({cm[1][0]})**: Departing employees missed by the model (high cost: unexpected departure).

---

## Detailed Classification Report
```text
{clf_rep_text}
```

---

## Saved Artifacts
- Confusion Matrix Plot: `outputs/visualizations/test_confusion_matrix.png`
- ROC Curve Plot: `outputs/visualizations/roc_curve_test.png`
- JSON Metrics: `outputs/metrics/test_metrics.json`
"""
    out_path = REPORTS_DIR / "final_test_evaluation_report.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Final test evaluation report written to: {out_path}")
    return out_path


def run_testing(
    model_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Champion Model",
):
    """Executes the complete test phase."""
    return test_model(model_pipeline, X_test, y_test, model_name=model_name)


if __name__ == "__main__":
    from src.data_collection import load_and_inspect_data
    from src.data_cleaning import clean_raw_data
    from src.data_splitting import split_data

    df_raw, _ = load_and_inspect_data()
    df_clean = clean_raw_data(df_raw)
    _, _, X_test, _, _, y_test = split_data(df_clean)
    model = load_champion_model()
    run_testing(model, X_test, y_test)
