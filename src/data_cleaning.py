"""
Module 2: Data Cleaning and Preprocessing
Objective: Clean the raw employee dataset, separate features and target, remove constant
and identifier columns, identify feature types, and build a reusable Scikit-learn ColumnTransformer.
"""

from typing import Tuple, List, Dict, Any
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.utils import get_logger, PROCESSED_DATA_DIR

logger = get_logger("DataCleaning")

# Columns identified as non-informative, identifiers, or constants with zero variance
COLUMNS_TO_DROP = ["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"]
TARGET_COL = "Attrition"


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataset:
    - Removes whitespace from column names and string cells
    - Drops zero-variance/constant columns and ID column
    - Maps target Attrition ('Yes' -> 1, 'No' -> 0)
    - Checks and removes duplicates if any
    """
    df_clean = df.copy()

    # Strip whitespace on column headers and object columns
    df_clean.columns = [col.strip() for col in df_clean.columns]
    for col in df_clean.select_dtypes(include=["object", "string"]):
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # Drop duplicate records if present
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    dropped_dupes = initial_len - len(df_clean)
    if dropped_dupes > 0:
        logger.info(f"Removed {dropped_dupes} duplicate rows.")

    # Remove non-predictive/constant columns
    cols_present_to_drop = [c for c in COLUMNS_TO_DROP if c in df_clean.columns]
    if cols_present_to_drop:
        df_clean = df_clean.drop(columns=cols_present_to_drop)
        logger.info(f"Dropped constant/identifier columns: {cols_present_to_drop}")

    # Binary encode target
    if TARGET_COL in df_clean.columns:
        if df_clean[TARGET_COL].dtype in [object, "string"] or isinstance(df_clean[TARGET_COL].iloc[0], str):
            df_clean[TARGET_COL] = df_clean[TARGET_COL].map({"Yes": 1, "No": 0})
            logger.info(f"Mapped '{TARGET_COL}': 'Yes' -> 1, 'No' -> 0")

    return df_clean


def get_feature_types(df: pd.DataFrame, target_col: str = TARGET_COL) -> Tuple[List[str], List[str]]:
    """
    Identifies numerical and categorical feature names, excluding target column.
    """
    features_df = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()

    categorical_cols = list(features_df.select_dtypes(include=["object", "string", "category"]).columns)
    numerical_cols = list(features_df.select_dtypes(include=["int64", "float64"]).columns)

    logger.info(f"Identified {len(numerical_cols)} numerical features and {len(categorical_cols)} categorical features.")
    return numerical_cols, categorical_cols


def build_preprocessor_pipeline(numerical_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Builds a Scikit-learn ColumnTransformer:
    - Numerical: SimpleImputer (median) + StandardScaler
    - Categorical: SimpleImputer (most_frequent) + OneHotEncoder (handle_unknown='ignore', sparse_output=False)
    
    This ensures no data leakage when fitted solely within cross-validation or training splits.
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def save_cleaned_data(df_clean: pd.DataFrame, filename: str = "cleaned_employee_data.csv") -> Path:
    """Saves the cleaned dataset to data/processed/."""
    out_path = PROCESSED_DATA_DIR / filename
    df_clean.to_csv(out_path, index=False)
    logger.info(f"Cleaned dataset saved to: {out_path}")
    return out_path


def run_data_cleaning(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], ColumnTransformer]:
    """
    End-to-end execution of Module 2.
    """
    logger.info("Executing Data Cleaning and Preprocessing Pipeline...")
    df_clean = clean_raw_data(df_raw)
    numerical_cols, categorical_cols = get_feature_types(df_clean)
    preprocessor = build_preprocessor_pipeline(numerical_cols, categorical_cols)
    save_cleaned_data(df_clean)

    return df_clean, numerical_cols, categorical_cols, preprocessor


if __name__ == "__main__":
    from src.data_collection import load_and_inspect_data
    df_raw, _ = load_and_inspect_data()
    df_clean, num_cols, cat_cols, preprocessor = run_data_cleaning(df_raw)
    print("Cleaned shape:", df_clean.shape)
    print("Numerical features:", num_cols)
    print("Categorical features:", cat_cols)
