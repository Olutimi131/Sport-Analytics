"""
src/config.py
=============
Centralized configuration, paths, feature schemas, and hyperparameters
for the Esports Sponsorship Media Value Forecasting Engine.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
DATA_CSV_PATH = DATA_DIR / "esports_sponsorship_data.csv"
MODEL_PATH = MODELS_DIR / "xgboost_media_value_model.joblib"
PIPELINE_PATH = MODELS_DIR / "feature_pipeline.joblib"
METRICS_JSON_PATH = REPORTS_DIR / "evaluation_metrics.json"
FEATURE_IMPORTANCE_JSON_PATH = REPORTS_DIR / "feature_importance.json"

# Target Variable
TARGET_COL = "generated_media_value_usd"

# Raw Feature Groups
CATEGORICAL_FEATURES = [
    "game_title",
    "tournament_tier",
    "stage_of_competition",
    "team_name",
    "sponsor_category",
    "placement_asset_type"
]

STREAMING_NUMERICAL_FEATURES = [
    "stream_broadcast_hours",
    "stream_avg_ccv",
    "stream_peak_ccv",
    "stream_hours_watched",
    "stream_logo_screen_time_mins",
    "stream_logo_quality_index",
    "stream_chat_mentions"
]

TV_NUMERICAL_FEATURES = [
    "tv_broadcast_hours",
    "tv_avg_household_reach",
    "tv_cpm_rate",
    "tv_brand_integrations"
]

SOCIAL_NUMERICAL_FEATURES = [
    "social_post_volume",
    "social_impressions",
    "social_engagement_rate",
    "social_video_views",
    "influencer_player_amplification_count"
]

PRINT_NUMERICAL_FEATURES = [
    "online_articles_count",
    "online_total_readership",
    "headline_mention_rate",
    "online_editorial_sentiment"
]

COST_FEATURE = ["sponsor_contract_cost_usd"]

RAW_NUMERICAL_FEATURES = (
    STREAMING_NUMERICAL_FEATURES
    + TV_NUMERICAL_FEATURES
    + SOCIAL_NUMERICAL_FEATURES
    + PRINT_NUMERICAL_FEATURES
    + COST_FEATURE
)

# Features created during feature engineering
ENGINEERED_NUMERICAL_FEATURES = [
    "total_cross_channel_impressions",
    "stream_effective_exposure_mins",
    "stream_ccv_peak_ratio",
    "social_engagement_volume",
    "social_video_view_ratio",
    "print_composite_impact_score",
    "channel_share_streaming",
    "channel_share_tv",
    "channel_share_social",
    "channel_share_print"
]

ALL_NUMERICAL_FEATURES = RAW_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES

# Model Hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.04,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.05,
    "reg_lambda": 1.2,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1
}

# Cross-Validation & Splitting
TEST_SIZE = 0.20
VAL_SIZE = 0.15
RANDOM_SEED = 42

