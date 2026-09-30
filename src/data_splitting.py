"""
Module 5: Data Splitting
Objective: Divide the cleaned employee dataset into Stratified Training (70%),
Validation (15%), and Testing (15%) sets using a fixed random seed.
"""

from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import get_logger

logger = get_logger("DataSplitting")
TARGET_COL = "Attrition"


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Performs a two-step stratified train/validation/test split.
    First splits into Train (70%) and Temp (30%).
    Then splits Temp equally into Validation (15%) and Test (15%).

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    logger.info(f"Initiating Stratified Split: {int(train_ratio*100)}% Train, {int(val_ratio*100)}% Val, {int(test_ratio*100)}% Test")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Step 1: Split train (70%) and temp (30%)
    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_ratio,
        stratify=y,
        random_state=random_state,
    )

    # Step 2: Split temp equally into validation (15%) and test (15%)
    val_rel_ratio = val_ratio / temp_ratio  # 0.15 / 0.30 = 0.5
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_rel_ratio),
        stratify=y_temp,
        random_state=random_state,
    )

    logger.info("--- Data Split Summary ---")
    for name, split_y in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
        pos = (split_y == 1).sum()
        neg = (split_y == 0).sum()
        pct = (pos / len(split_y)) * 100
        logger.info(f"  {name:<11}: {len(split_y):>4} records | Stayed(0): {neg:>4} | Left(1): {pos:>3} ({pct:.2f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    from src.data_collection import load_and_inspect_data
    from src.data_cleaning import clean_raw_data

    df_raw, _ = load_and_inspect_data()
    df_clean = clean_raw_data(df_raw)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(df_clean)
