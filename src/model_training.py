"""
Module 6: Machine Learning Model Training
Objective: Train classification pipelines (Logistic Regression, Decision Tree, Random Forest)
using the 70% training dataset with class imbalance handling, saving all models with Joblib.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.utils import get_logger, MODELS_DIR
from src.data_cleaning import build_preprocessor_pipeline, get_feature_types

logger = get_logger("ModelTraining")


def initialize_model_pipelines(preprocessor, random_state: int = 42) -> Dict[str, Pipeline]:
    """
    Constructs unified Scikit-learn Pipelines pairing the preprocessor with each classification model.
    Applies class_weight='balanced' across all models to effectively handle the ~5.2:1 class imbalance.
    """
    pipelines = {
        "Logistic_Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        C=0.1,
                        solver="liblinear",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Decision_Tree": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        class_weight="balanced",
                        max_depth=5,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random_Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=8,
                        min_samples_split=8,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    return pipelines


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
    """
    Fits all model pipelines strictly on the training partition.
    Evaluates training metrics to diagnose underfitting or overfitting.
    Saves trained pipelines to models/trained_models/.
    """
    logger.info(f"Training classification models on {len(X_train)} samples...")
    num_cols, cat_cols = get_feature_types(X_train)
    preprocessor = build_preprocessor_pipeline(num_cols, cat_cols)
    pipelines = initialize_model_pipelines(preprocessor, random_state=random_state)

    train_performance = {}

    for name, pipe in pipelines.items():
        logger.info(f"Fitting {name} pipeline...")
        pipe.fit(X_train, y_train)

        # Predictions on training data
        y_train_pred = pipe.predict(X_train)
        y_train_proba = (
            pipe.predict_proba(X_train)[:, 1] if hasattr(pipe.named_steps["classifier"], "predict_proba") else None
        )

        acc = accuracy_score(y_train, y_train_pred)
        prec = precision_score(y_train, y_train_pred, zero_division=0)
        rec = recall_score(y_train, y_train_pred, zero_division=0)
        f1 = f1_score(y_train, y_train_pred, zero_division=0)
        auc = roc_auc_score(y_train, y_train_proba) if y_train_proba is not None else 0.0

        train_performance[name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(auc, 4),
        }

        logger.info(
            f"  {name:<20} -> Train Acc: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}"
        )

        # Persist individual model pipeline using joblib
        model_save_path = MODELS_DIR / f"{name.lower()}.joblib"
        joblib.dump(pipe, model_save_path)
        logger.info(f"  Saved {name} to {model_save_path}")

    return pipelines, train_performance


if __name__ == "__main__":
    from src.data_collection import load_and_inspect_data
    from src.data_cleaning import clean_raw_data
    from src.data_splitting import split_data

    df_raw, _ = load_and_inspect_data()
    df_clean = clean_raw_data(df_raw)
    X_train, _, _, y_train, _, _ = split_data(df_clean)
    pipes, perf = train_models(X_train, y_train)
    print("\nTraining Metrics Summary:")
    print(pd.DataFrame(perf).T)
