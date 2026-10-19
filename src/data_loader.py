"""
src/data_loader.py
==================
Production Data Ingestion, Validation, and Partitioning Pipeline.
"""

import os
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_CSV_PATH,
    RANDOM_SEED,
    RAW_NUMERICAL_FEATURES,
    TARGET_COL,
    TEST_SIZE,
)


def load_or_generate_dataset(csv_path: str = str(DATA_CSV_PATH), n_samples: int = 5000) -> pd.DataFrame:
    """
    Loads dataset from csv_path. If not found, invokes data generator to synthesize it.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        print(f"Dataset not found at {csv_path}. Generating {n_samples} records...")
        from data.generator import generate_esports_sponsorship_data
        df = generate_esports_sponsorship_data(n_samples=n_samples, seed=RANDOM_SEED)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Generated and saved dataset to {csv_path}")

    validate_schema(df)
    return df

def validate_schema(df: pd.DataFrame) -> bool:
    """
    Validates presence of essential target, categorical, and numerical features.
    Raises ValueError if schema requirements are breached.
    """
    missing_cols = []
    
    if TARGET_COL not in df.columns:
        missing_cols.append(TARGET_COL)
        
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            missing_cols.append(col)
            
    for col in RAW_NUMERICAL_FEATURES:
        if col not in df.columns:
            missing_cols.append(col)
            
    if missing_cols:
        raise ValueError(f"Schema Validation Failed! Missing required columns: {missing_cols}")
        
    # Check for empty dataframe
    if len(df) == 0:
        raise ValueError("Dataset is empty!")
        
    # Check for null target values
    null_targets = df[TARGET_COL].isnull().sum()
    if null_targets > 0:
        raise ValueError(f"Dataset contains {null_targets} null values in target column '{TARGET_COL}'.")
        
    return True

def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the validated dataframe into train and test subsets.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def compute_dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Computes key summary metrics for logging and reporting.
    """
    summary = {
        "total_campaigns": len(df),
        "unique_games": df["game_title"].nunique() if "game_title" in df else 0,
        "unique_teams": df["team_name"].nunique() if "team_name" in df else 0,
        "unique_sponsor_categories": df["sponsor_category"].nunique() if "sponsor_category" in df else 0,
        "media_value_mean": float(df[TARGET_COL].mean()),
        "media_value_median": float(df[TARGET_COL].median()),
        "media_value_std": float(df[TARGET_COL].std()),
        "media_value_min": float(df[TARGET_COL].min()),
        "media_value_max": float(df[TARGET_COL].max()),
    }
    return summary

if __name__ == "__main__":
    df = load_or_generate_dataset()
    print("Dataset Schema Validated Successfully.")
    print("Summary:", compute_dataset_summary(df))
