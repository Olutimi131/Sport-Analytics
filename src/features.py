"""
src/features.py
===============
Production Feature Engineering Pipeline and Scikit-Learn Transformers.
Transforms raw multi-channel exposure metrics into high-signal representations
for XGBoost regression.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.config import (
    ALL_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


class EsportsFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-specific feature engineering transformer for esports sponsorship exposure.
    Calculates cross-channel synergies, exposure ratios, and quality-adjusted variables.
    """
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        # 1. Total Cross-Channel Exposure Volume
        # Summing impressions / views across streaming, TV, social, and print
        stream_view_est = df["stream_hours_watched"].fillna(0)
        tv_reach = df["tv_avg_household_reach"].fillna(0)
        social_imp = df["social_impressions"].fillna(0)
        print_readers = df["online_total_readership"].fillna(0)
        
        total_impressions = stream_view_est + tv_reach + social_imp + print_readers
        df["total_cross_channel_impressions"] = np.maximum(total_impressions, 1.0)
        
        # 2. Quality-Adjusted Streaming Screen Time (Nielsen QI analogue)
        df["stream_effective_exposure_mins"] = (
            df["stream_logo_screen_time_mins"].fillna(0) 
            * df["stream_logo_quality_index"].fillna(0.5)
        )
        
        # 3. Viewership Volatility / Peak-to-Average Ratio
        avg_ccv = np.maximum(df["stream_avg_ccv"].fillna(1000), 1.0)
        df["stream_ccv_peak_ratio"] = df["stream_peak_ccv"].fillna(avg_ccv) / avg_ccv
        
        # 4. Social Engagement Absolute Volume & Video View Ratio
        df["social_engagement_volume"] = (
            df["social_impressions"].fillna(0) 
            * df["social_engagement_rate"].fillna(0.02)
        )
        safe_social_imp = np.maximum(df["social_impressions"].fillna(1000), 1.0)
        df["social_video_view_ratio"] = df["social_video_views"].fillna(0) / safe_social_imp
        
        # 5. Editorial Print Composite Impact
        sentiment_weight = np.maximum(0.2, 1.0 + df["online_editorial_sentiment"].fillna(0.0))
        headline_boost = 1.0 + df["headline_mention_rate"].fillna(0.0)
        df["print_composite_impact_score"] = (
            df["online_articles_count"].fillna(0) * headline_boost * sentiment_weight
        )
        
        # 6. Channel Share of Voice (SoV) Ratios
        denom = df["total_cross_channel_impressions"]
        df["channel_share_streaming"] = stream_view_est / denom
        df["channel_share_tv"] = tv_reach / denom
        df["channel_share_social"] = social_imp / denom
        df["channel_share_print"] = print_readers / denom

        return df

def build_preprocessing_pipeline() -> Pipeline:
    """
    Constructs a robust, production-grade scikit-learn ColumnTransformer pipeline.
    Combines domain feature engineering, median imputation, robust scaling,
    and unknown-resilient one-hot encoding.
    """
    feature_engineer = EsportsFeatureEngineer()

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(unit_variance=True))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, ALL_NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    full_pipeline = Pipeline(
        steps=[
            ("feature_engineer", feature_engineer),
            ("preprocessor", preprocessor)
        ]
    )

    return full_pipeline

def get_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """
    Extracts human-interpretable feature names from a fitted preprocessing pipeline.
    """
    ct: ColumnTransformer = fitted_pipeline.named_steps["preprocessor"]
    
    # Numerical feature names
    num_names = ALL_NUMERICAL_FEATURES
    
    # Categorical feature names from OneHotEncoder
    ohe_step = ct.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(ohe_step.get_feature_names_out(CATEGORICAL_FEATURES))
    
    return list(num_names) + cat_names
