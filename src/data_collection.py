"""
Module 1: Data Collection
Objective: Load, inspect, and summarize the IBM HR Analytics Employee Attrition dataset.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
from src.utils import get_logger, RAW_DATA_DIR

logger = get_logger("DataCollection")


def find_dataset_path(custom_path: str = None) -> Path:
    """
    Automatically detects the employee attrition CSV dataset.
    Looks in custom_path, RAW_DATA_DIR, and checks for standard filenames.
    """
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)

    # Search candidates in RAW_DATA_DIR
    candidates = [
        RAW_DATA_DIR / "employee_attrition.csv",
        RAW_DATA_DIR / "employee_attrition.csv.csv",
        RAW_DATA_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback: find any CSV in RAW_DATA_DIR
    all_csvs = list(RAW_DATA_DIR.glob("*.csv"))
    if all_csvs:
        return all_csvs[0]

    raise FileNotFoundError(
        f"Could not find employee attrition dataset in {RAW_DATA_DIR}. "
        f"Please place your IBM HR Analytics CSV file in: {RAW_DATA_DIR / 'employee_attrition.csv'}"
    )


def load_and_inspect_data(file_path: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads the CSV dataset, inspects its schema, columns, and target distribution.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Raw DataFrame and inspection summary dictionary.
    """
    dataset_path = find_dataset_path(file_path)
    logger.info(f"Loading dataset from: {dataset_path}")

    # Use utf-8-sig to automatically strip any potential BOM (\ufeff)
    df = pd.read_csv(dataset_path, encoding="utf-8-sig")

    # Clean any whitespace around column names
    df.columns = [col.strip() for col in df.columns]

    rows, cols = df.shape
    logger.info(f"Dataset successfully loaded: {rows} rows, {cols} columns.")

    # Target column check
    target_col = "Attrition"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    target_counts = df[target_col].value_counts().to_dict()
    target_pct = (df[target_col].value_counts(normalize=True) * 100).round(2).to_dict()

    logger.info("--- Data Inspection Summary ---")
    logger.info(f"Dimensions: {rows} rows x {cols} columns")
    logger.info(f"Target distribution ({target_col}):")
    for val, count in target_counts.items():
        logger.info(f"  - {val}: {count} records ({target_pct.get(val, 0)}%)")

    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    logger.info(f"Total missing values: {missing_count}")
    logger.info(f"Duplicate rows detected: {duplicate_count}")

    data_summary = {
        "dataset_path": str(dataset_path),
        "total_rows": int(rows),
        "total_columns": int(cols),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "target_distribution_counts": target_counts,
        "target_distribution_pct": target_pct,
        "total_missing_values": int(missing_count),
        "total_duplicate_rows": int(duplicate_count),
        "first_5_records": df.head(5).to_dict(orient="records"),
    }

    return df, data_summary


if __name__ == "__main__":
    df, summary = load_and_inspect_data()
    print("\n--- First 5 Records ---")
    print(df.head(5))
    print("\nColumn List:")
    print(summary["column_names"])
