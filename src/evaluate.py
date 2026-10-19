"""
src/evaluate.py
===============
Comprehensive Model Evaluation & Feature Importance Analysis Pipeline.
Computes RMSE, MAE, R², MAPE, MedAE and exports diagnostic metrics and figures.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend for headless/Docker support
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_CSV_PATH,
    FEATURE_IMPORTANCE_JSON_PATH,
    METRICS_JSON_PATH,
    MODEL_PATH,
    PIPELINE_PATH,
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET_COL,
    TEST_SIZE,
)
from src.data_loader import load_or_generate_dataset
from src.features import get_feature_names


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Computes industry-standard regression metrics.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)
    medae = float(median_absolute_error(y_true, y_pred))
    
    # Residual statistics
    residuals = y_true - y_pred
    mean_residual = float(np.mean(residuals))
    std_residual = float(np.std(residuals))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2,
        "mape_percent": mape,
        "median_absolute_error": medae,
        "mean_residual": mean_residual,
        "std_residual": std_residual,
        "sample_count": len(y_true)
    }
    return metrics

def extract_feature_importance(model, feature_names: list) -> dict[str, Any]:
    """
    Extracts and normalizes feature importance scores (gain and weight),
    and aggregates them into multi-channel categories.
    """
    booster = model.get_booster()
    score_gain = booster.get_score(importance_type="gain")
    score_weight = booster.get_score(importance_type="weight")

    # Map internal f0, f1... to feature names
    importance_gain_dict = {}
    importance_weight_dict = {}
    
    for i, name in enumerate(feature_names):
        f_key = f"f{i}"
        gain = score_gain.get(f_key, 0.0)
        weight = score_weight.get(f_key, 0.0)
        importance_gain_dict[name] = gain
        importance_weight_dict[name] = weight

    total_gain = sum(importance_gain_dict.values()) or 1.0
    normalized_gain = {k: float(v / total_gain) for k, v in importance_gain_dict.items()}
    sorted_gain = sorted(normalized_gain.items(), key=lambda x: x[1], reverse=True)

    # Multi-channel aggregate attribution
    channel_groups = {
        "Streaming": 0.0,
        "Television": 0.0,
        "Social Media": 0.0,
        "Online Print & Editorial": 0.0,
        "Context & Tournament": 0.0,
        "Contract Cost": 0.0
    }

    for feat, val in normalized_gain.items():
        feat_lower = feat.lower()
        if "stream" in feat_lower:
            channel_groups["Streaming"] += val
        elif "tv_" in feat_lower or "television" in feat_lower:
            channel_groups["Television"] += val
        elif "social" in feat_lower or "influencer" in feat_lower:
            channel_groups["Social Media"] += val
        elif "print" in feat_lower or "online" in feat_lower or "headline" in feat_lower or "article" in feat_lower:
            channel_groups["Online Print & Editorial"] += val
        elif "cost" in feat_lower:
            channel_groups["Contract Cost"] += val
        else:
            channel_groups["Context & Tournament"] += val

    return {
        "sorted_gain_importance": [{"feature": k, "importance": round(v, 5)} for k, v in sorted_gain],
        "channel_group_attribution": {k: round(v, 4) for k, v in channel_groups.items()}
    }

def generate_evaluation_visualizations(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    importance_data: dict[str, Any],
    reports_dir: str = str(REPORTS_DIR)
):
    """
    Generates and saves diagnostic figures for model inspection.
    """
    os.makedirs(reports_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Actual vs. Predicted Plot
    _, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.scatter(y_test, y_pred, alpha=0.4, color="#008080", edgecolors="none", s=25)
    max_val = max(np.max(y_test), np.max(y_pred))
    min_val = min(np.min(y_test), np.min(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Forecast ($y = \\hat{y}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual Generated Media Value ($ USD)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted Media Value ($ USD)", fontsize=11, fontweight="bold")
    ax.set_title("Actual vs. Predicted Media Value (Log-Log Scale)", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "actual_vs_predicted.png"))
    plt.close()

    # 2. Residual Distribution Plot
    residuals = y_test - y_pred
    _, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.histplot(residuals, kde=True, ax=ax, color="#4A90E2", bins=50)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
    ax.set_xlabel("Residual Error (Actual - Predicted, USD)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax.set_title("Residual Error Distribution", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "residuals_distribution.png"))
    plt.close()

    # 3. Top Feature Importance Bar Chart
    top_features = importance_data["sorted_gain_importance"][:12]
    feat_names = [item["feature"] for item in top_features][::-1]
    feat_scores = [item["importance"] for item in top_features][::-1]

    _, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.barh(feat_names, feat_scores, color="#2ECC71")
    ax.set_xlabel("Normalized Feature Gain", fontsize=11, fontweight="bold")
    ax.set_title("Top 12 Predictive Features (XGBoost Gain)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "feature_importance.png"))
    plt.close()

    # 4. Multi-Channel Contribution Pie Chart
    channels = importance_data["channel_group_attribution"]
    _, ax = plt.subplots(figsize=(7, 7), dpi=300)
    labels = list(channels.keys())
    sizes = list(channels.values())
    colors = ["#3498DB", "#E74C3C", "#9B59B6", "#F1C40F", "#1ABC9C", "#95A5A6"]
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors,
           wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title("Channel Share of Model Predictive Power", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "channel_attribution.png"))
    plt.close()

def evaluate_model(
    test_csv_or_df=None,
    model_path: str = str(MODEL_PATH),
    pipeline_path: str = str(PIPELINE_PATH),
    output_dir: str = str(REPORTS_DIR)
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Evaluates the trained model on test data and writes reports.
    """
    print("=" * 65)
    print("ESPORTS SPONSORSHIP MEDIA VALUE FORECASTING ENGINE - EVALUATION")
    print("=" * 65)

    if not os.path.exists(model_path) or not os.path.exists(pipeline_path):
        raise FileNotFoundError(
            f"Trained model artifacts missing! Please run src/train.py first. (Checked: {model_path})"
        )

    print(f"Loading model artifacts from {model_path}...")
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)

    # Load test data
    if test_csv_or_df is None:
        print(f"Splitting full dataset from {DATA_CSV_PATH} to isolate hold-out test set...")
        df = load_or_generate_dataset()
        _, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=True)
    elif isinstance(test_csv_or_df, str):
        test_df = pd.read_csv(test_csv_or_df)
    else:
        test_df = test_csv_or_df

    X_test_raw = test_df.drop(columns=[TARGET_COL, "campaign_id", "roi_percentage"], errors="ignore")
    y_test = test_df[TARGET_COL].values

    # Preprocess test set
    X_test_trans = pipeline.transform(X_test_raw)
    feature_names = get_feature_names(pipeline)

    # Check if target was log-transformed during training
    metadata_path = os.path.join(os.path.dirname(model_path), "training_metadata.json")
    use_log_target = False
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                meta = json.load(f)
                use_log_target = meta.get("use_log_target", False)
        except (json.JSONDecodeError, OSError):
            use_log_target = False

    # Predict
    raw_preds = model.predict(X_test_trans)
    if use_log_target:
        y_pred = np.expm1(raw_preds)
    else:
        y_pred = raw_preds
    y_pred = np.maximum(y_pred, 0.0)  # non-negative constraint

    # Compute metrics
    metrics = compute_regression_metrics(y_test, y_pred)
    importance_data = extract_feature_importance(model, feature_names)

    # Print summary
    print("\nModel Evaluation Performance Metrics:")
    print("-" * 45)
    print(f"  R² Score (Variance Explained):   {metrics['r2_score']:.4f}")
    print(f"  RMSE (Root Mean Squared Error):  ${metrics['rmse']:,.2f}")
    print(f"  MAE (Mean Absolute Error):       ${metrics['mae']:,.2f}")
    print(f"  MAPE (Mean Absolute % Error):    {metrics['mape_percent']:.2f}%")
    print(f"  Median Absolute Error (MedAE):   ${metrics['median_absolute_error']:,.2f}")
    print(f"  Test Sample Size:                {metrics['sample_count']:,}")
    print("-" * 45)

    print("\nChannel Share of Predictive Value:")
    for ch, val in importance_data["channel_group_attribution"].items():
        print(f"  {ch:<26}: {val * 100:.1f}%")

    # Save to disk honoring the caller-provided output directory.
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    feature_importance_path = os.path.join(output_dir, "feature_importance.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(feature_importance_path, "w") as f:
        json.dump(importance_data, f, indent=2)

    # Generate figures
    generate_evaluation_visualizations(y_test, y_pred, importance_data, output_dir)
    print(f"\nEvaluation reports and figures saved to: {output_dir}")
    print("=" * 65)

    return metrics, importance_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Esports Media Value Model")
    parser.add_argument("--test-data", type=str, default=None, help="Optional separate test dataset CSV")
    args = parser.parse_args()
    evaluate_model(test_csv_or_df=args.test_data)

