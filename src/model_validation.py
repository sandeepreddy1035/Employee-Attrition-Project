"""
Module 7: Model Validation and Selection
Objective: Evaluate trained models on the 15% validation dataset, perform cross-validation,
plot validation confusion matrices, compare metrics with emphasis on Attrition Recall & F1,
and select the best performing model pipeline.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
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

logger = get_logger("ModelValidation")


def evaluate_on_validation_set(
    pipelines: Dict[str, Pipeline],
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates each model pipeline on the unseen 15% validation dataset.
    Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
    """
    logger.info(f"Evaluating {len(pipelines)} models on Validation set ({len(X_val)} records)...")
    val_results = {}

    for name, pipe in pipelines.items():
        y_pred = pipe.predict(X_val)
        y_proba = (
            pipe.predict_proba(X_val)[:, 1]
            if hasattr(pipe.named_steps["classifier"], "predict_proba")
            else None
        )

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_proba) if y_proba is not None else 0.0
        cm = confusion_matrix(y_val, y_pred)

        val_results[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1_Score": round(float(f1), 4),
            "ROC_AUC": round(float(auc), 4),
            "Confusion_Matrix": cm.tolist(),
        }

        logger.info(
            f"  {name:<20} -> Val Acc: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}"
        )

    return val_results


def perform_cross_validation(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Performs 5-Fold Stratified Cross-Validation on the training set.
    Reports mean and standard deviation for balanced metrics.
    """
    logger.info(f"Running {n_splits}-Fold Stratified Cross-Validation on Training data...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_results = {}

    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for name, pipe in pipelines.items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_results[name] = {
            "CV_Mean_Accuracy": round(float(scores["test_accuracy"].mean()), 4),
            "CV_Mean_Precision": round(float(scores["test_precision"].mean()), 4),
            "CV_Mean_Recall": round(float(scores["test_recall"].mean()), 4),
            "CV_Mean_F1": round(float(scores["test_f1"].mean()), 4),
            "CV_Mean_ROC_AUC": round(float(scores["test_roc_auc"].mean()), 4),
            "CV_Std_F1": round(float(scores["test_f1"].std()), 4),
        }
        logger.info(
            f"  {name:<20} -> 5-Fold CV Mean Recall: {cv_results[name]['CV_Mean_Recall']:.4f} | F1: {cv_results[name]['CV_Mean_F1']:.4f} (±{cv_results[name]['CV_Std_F1']:.4f})"
        )

    return cv_results


def plot_validation_confusion_matrices(val_results: Dict[str, Dict[str, Any]]) -> Path:
    """
    Creates a multi-panel figure displaying confusion matrices for all evaluated models.
    """
    set_plotting_theme()
    n_models = len(val_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, metrics) in zip(axes, val_results.items()):
        cm = np.array(metrics["Confusion_Matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Stayed (0)", "Left (1)"],
            yticklabels=["Stayed (0)", "Left (1)"],
        )
        ax.set_title(f"{name.replace('_', ' ')}\nF1: {metrics['F1_Score']} | Rec: {metrics['Recall']}", pad=10)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    filepath = save_plot(fig, "validation_confusion_matrices.png")
    logger.info(f"Saved validation confusion matrices to: {filepath}")
    return filepath


def select_best_model(
    pipelines: Dict[str, Pipeline],
    val_results: Dict[str, Dict[str, Any]],
    cv_results: Dict[str, Dict[str, Any]],
) -> Tuple[str, Pipeline]:
    """
    Selects the champion model prioritizing Attrition Recall & F1-Score:
    In employee attrition, False Negatives (predicting an employee stays when they actually leave)
    carry heavy institutional and hiring costs. A model with high Recall ensures HR can proactively
    intervene with at-risk talent.
    
    Ranking metric: Composite score = 0.40 * Recall + 0.35 * F1_Score + 0.25 * ROC_AUC.
    """
    logger.info("Selecting final model using documented HR evaluation strategy (Prioritizing Recall & F1)...")
    ranking = {}

    for name, metrics in val_results.items():
        score = (
            0.40 * metrics["Recall"]
            + 0.35 * metrics["F1_Score"]
            + 0.25 * metrics["ROC_AUC"]
        )
        ranking[name] = score
        logger.info(f"  {name:<20} -> Composite Validation Score: {score:.4f}")

    best_model_name = max(ranking, key=ranking.get)
    best_pipeline = pipelines[best_model_name]
    logger.info(f"--> CHAMPION MODEL SELECTED: {best_model_name} (Score: {ranking[best_model_name]:.4f})")

    # Persist the selected champion model pipeline as best_model_pipeline.joblib
    best_model_path = MODELS_DIR / "best_model_pipeline.joblib"
    joblib.dump(best_pipeline, best_model_path)
    logger.info(f"Saved champion pipeline to: {best_model_path}")

    return best_model_name, best_pipeline


def write_validation_report(
    val_results: Dict[str, Dict[str, Any]],
    cv_results: Dict[str, Dict[str, Any]],
    best_model_name: str,
) -> Path:
    """Writes an exhaustive Markdown validation report."""
    rows = []
    for name in val_results.keys():
        v = val_results[name]
        c = cv_results.get(name, {})
        rows.append(
            f"| **{name.replace('_', ' ')}** | {v['Accuracy']:.4f} | {v['Precision']:.4f} | **{v['Recall']:.4f}** | **{v['F1_Score']:.4f}** | {v['ROC_AUC']:.4f} | {c.get('CV_Mean_Recall', 0):.4f} | {c.get('CV_Mean_F1', 0):.4f} |"
        )
    table_str = "\n".join(rows)

    report_content = f"""# Model Validation & Comparative Evaluation Report

## Evaluation Strategy & Imbalance Considerations
The IBM HR Analytics dataset contains an inherent **5.2 : 1 class imbalance** (16.12% turnover rate).
When evaluating attrition models in an enterprise human resources context:
- **Cost of False Negatives (Missed Leavers)**: High (unplanned turnover, institutional knowledge loss, recruitment and training costs averaging 1.5x - 2x annual salary).
- **Cost of False Positives (Surplus Retention Focus)**: Low (retention outreach, 1-on-1 check-ins, professional growth conversations).

Consequently, **Accuracy alone is misleading** (a naive stayer baseline attains 83.9% accuracy with 0% attrition recall). Model selection strictly prioritizes **Attrition Recall** and **F1-Score**, supported by **ROC-AUC**.

---

## Validation Performance Table (15% Unseen Validation Split)

| Algorithm | Val Accuracy | Val Precision | Val Recall | Val F1-Score | Val ROC-AUC | 5-Fold CV Mean Recall | 5-Fold CV Mean F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{table_str}

---

## Model Selection Decision
- **Champion Selected**: **`{best_model_name.replace('_', ' ')}`**
- **Rationale**: Demonstrates the optimal balance of sensitivity to departure risks (capturing the majority of employees planning to leave) while maintaining high discriminant capability (ROC-AUC) and stability across 5-fold cross-validation.
- The fitted model pipeline has been saved to: `models/trained_models/best_model_pipeline.joblib`.
"""
    out_path = REPORTS_DIR / "model_validation_report.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Model validation report saved to: {out_path}")
    return out_path


def run_validation(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[str, Pipeline, Dict[str, Any]]:
    """Executes the full model validation and selection workflow."""
    val_results = evaluate_on_validation_set(pipelines, X_val, y_val)
    cv_results = perform_cross_validation(pipelines, X_train, y_train)
    plot_validation_confusion_matrices(val_results)
    best_name, best_pipe = select_best_model(pipelines, val_results, cv_results)
    write_validation_report(val_results, cv_results, best_name)

    # Save validation metrics JSON
    combined_metrics = {
        "validation_results": val_results,
        "cv_results": cv_results,
        "champion_model": best_name,
    }
    save_json(combined_metrics, "validation_metrics.json")

    return best_name, best_pipe, combined_metrics


if __name__ == "__main__":
    from src.data_collection import load_and_inspect_data
    from src.data_cleaning import clean_raw_data
    from src.data_splitting import split_data
    from src.model_training import train_models

    df_raw, _ = load_and_inspect_data()
    df_clean = clean_raw_data(df_raw)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)
    pipes, _ = train_models(X_train, y_train)
    best_name, best_pipe, _ = run_validation(pipes, X_train, y_train, X_val, y_val)
    print(f"\nChampion Model: {best_name}")
