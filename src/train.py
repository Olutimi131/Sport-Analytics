"""
src/train.py
============
Model Training Pipeline for Esports Sponsorship Media Value Forecasting.
Fits the feature engineering pipeline and XGBoost Regressor with early stopping.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_CSV_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PIPELINE_PATH,
    RANDOM_SEED,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
    XGBOOST_PARAMS,
)
from src.data_loader import load_or_generate_dataset, validate_schema
from src.features import build_preprocessing_pipeline, get_feature_names


def train_model(
    csv_path: str = str(DATA_CSV_PATH),
    hyperparams: dict[str, Any] | None = None,
    save_artifacts: bool = True
) -> tuple[xgb.XGBRegressor, Any, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end model training routine:
    1. Ingestion & Validation
    2. Train/Val/Test Splitting
    3. Feature Engineering & Preprocessing Fitting
    4. XGBoost Regression Fitting with Validation Tracking
    5. Artifact Persistence
    """
    start_time = time.time()
    print("=" * 65)
    print("ESPORTS SPONSORSHIP MEDIA VALUE FORECASTING ENGINE - TRAINING")
    print("=" * 65)
    
    # 1. Load Data
    print(f"[1/5] Ingesting dataset from {csv_path}...")
    df = load_or_generate_dataset(csv_path)
    validate_schema(df)
    print(f"      Loaded {len(df):,} campaign records across {len(df.columns)} columns.")

    # 2. Partition Data: Train (70%), Val (10%), Test (20%)
    print(f"[2/5] Partitioning data (Test: {TEST_SIZE*100:.0f}%, Val: {VAL_SIZE*100:.0f}%)...")
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=True
    )
    val_ratio_adjusted = VAL_SIZE / (1.0 - TEST_SIZE)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio_adjusted, random_state=RANDOM_SEED, shuffle=True
    )

    print(f"      Training set:   {len(train_df):,} samples")
    print(f"      Validation set: {len(val_df):,} samples")
    print(f"      Hold-out Test:  {len(test_df):,} samples")

    # 3. Fit Preprocessing Pipeline
    print("[3/5] Fitting Feature Engineering Pipeline...")
    pipeline = build_preprocessing_pipeline()
    
    X_train_raw = train_df.drop(columns=[TARGET_COL, "campaign_id", "roi_percentage"], errors="ignore")
    y_train = train_df[TARGET_COL].values
    
    X_val_raw = val_df.drop(columns=[TARGET_COL, "campaign_id", "roi_percentage"], errors="ignore")
    y_val = val_df[TARGET_COL].values
    
    X_train_trans = pipeline.fit_transform(X_train_raw)
    X_val_trans = pipeline.transform(X_val_raw)
    
    feature_names = get_feature_names(pipeline)
    print(f"      Engineered {len(feature_names)} features (dense transformed matrix shape: {X_train_trans.shape}).")

    # 4. Train XGBoost Model
    print("[4/5] Training XGBoost Regressor on Log-Scaled Media Value with Early Stopping...")
    params = XGBOOST_PARAMS.copy()
    if hyperparams:
        params.update(hyperparams)

    # Use log1p transformation for monetary distribution stabilization
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    model = xgb.XGBRegressor(
        **params,
        early_stopping_rounds=40,
        eval_metric="rmse"
    )

    model.fit(
        X_train_trans,
        y_train_log,
        eval_set=[(X_train_trans, y_train_log), (X_val_trans, y_val_log)],
        verbose=100
    )

    best_iteration = getattr(model, "best_iteration", params.get("n_estimators", 400))
    print(f"      Optimal iteration achieved: {best_iteration}")

    # 5. Persist Model Artifacts
    if save_artifacts:
        print("[5/5] Persisting model artifacts to disk...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, MODEL_PATH)
        joblib.dump(pipeline, PIPELINE_PATH)
        
        feature_names_path = MODELS_DIR / "feature_names.json"
        with open(feature_names_path, "w") as f:
            json.dump(feature_names, f, indent=2)
            
        metadata = {
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "training_samples": len(train_df),
            "validation_samples": len(val_df),
            "test_samples": len(test_df),
            "num_features": len(feature_names),
            "use_log_target": True,
            "best_iteration": int(best_iteration) if best_iteration is not None else -1,
            "hyperparameters": {k: str(v) for k, v in params.items()}
        }
        with open(MODELS_DIR / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"      Model artifact:      {MODEL_PATH}")
        print(f"      Pipeline artifact:   {PIPELINE_PATH}")
        print(f"      Feature metadata:    {feature_names_path}")

    elapsed = time.time() - start_time
    print(f"Training completed successfully in {elapsed:.2f} seconds.")
    print("=" * 65)

    return model, pipeline, train_df, test_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost Media Value Forecasting Engine")
    parser.add_argument("--data", type=str, default=str(DATA_CSV_PATH), help="Path to input CSV dataset")
    parser.add_argument("--estimators", type=int, default=None, help="Number of estimators")
    parser.add_argument("--depth", type=int, default=None, help="Max depth")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    args = parser.parse_args()

    custom_params = {}
    if args.estimators is not None:
        custom_params["n_estimators"] = args.estimators
    if args.depth is not None:
        custom_params["max_depth"] = args.depth
    if args.lr is not None:
        custom_params["learning_rate"] = args.lr

    train_model(csv_path=args.data, hyperparams=custom_params)
