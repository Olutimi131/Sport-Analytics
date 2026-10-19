"""
src/predict.py
==============
Production Inference Engine and Campaign Optimization Simulator.
Provides real-time scoring, confidence bounds, channel attribution,
and executive 'What-If' sensitivity analysis.
"""

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
import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    METRICS_JSON_PATH,
    MODEL_PATH,
    PIPELINE_PATH,
    RAW_NUMERICAL_FEATURES,
)


class MediaValuePredictor:
    """
    Production-grade predictor for Esports Sponsorship Media Value.
    Loads trained XGBoost pipeline, performs batch or single-record inference,
    calculates confidence intervals, and runs multi-channel sensitivity simulation.
    """
    def __init__(
        self,
        model_path: str = str(MODEL_PATH),
        pipeline_path: str = str(PIPELINE_PATH),
        metrics_path: str = str(METRICS_JSON_PATH)
    ):
        self.model_path = model_path
        self.pipeline_path = pipeline_path
        self.metrics_path = metrics_path
        self.model = None
        self.pipeline = None
        self.residual_std = 18500.0  # sensible default
        self.use_log_target = False
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads fitted model and pipeline artifacts from disk."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.pipeline_path):
            raise FileNotFoundError(
                f"Model artifacts not found at {self.model_path}. Train the model first via src/train.py."
            )
        self.model = joblib.load(self.model_path)
        self.pipeline = joblib.load(self.pipeline_path)

        metadata_path = os.path.join(os.path.dirname(self.model_path), "training_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    meta = json.load(f)
                    self.use_log_target = meta.get("use_log_target", False)
            except (json.JSONDecodeError, OSError):
                self.use_log_target = False

        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, "r") as f:
                    metrics = json.load(f)
                    self.residual_std = metrics.get("std_residual", metrics.get("rmse", 18500.0))
            except (json.JSONDecodeError, OSError):
                self.residual_std = 18500.0

    def _prepare_dataframe(self, data: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
        """Standardizes input dictionary or DataFrame into validated input format."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        # Ensure all expected columns exist, default if missing
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                df[col] = "Unknown"

        for col in RAW_NUMERICAL_FEATURES:
            if col not in df.columns:
                df[col] = 0.0

        return df

    def predict(self, data: dict[str, Any] | pd.DataFrame) -> np.ndarray:
        """
        Executes prediction pipeline on input campaign data.
        Returns array of predicted Media Value in USD.
        """
        df = self._prepare_dataframe(data)
        features_trans = self.pipeline.transform(df)
        raw_preds = self.model.predict(features_trans)
        if self.use_log_target:
            preds = np.expm1(raw_preds)
        else:
            preds = raw_preds
        return np.maximum(preds, 0.0)

    def predict_single(self, data: dict[str, Any], confidence_level: float = 0.95) -> dict[str, Any]:
        """
        Performs rich single-campaign forecast with confidence intervals,
        ROI metrics, and channel breakdown.
        """
        df = self._prepare_dataframe(data)
        pred_emv = float(self.predict(df)[0])

        # Confidence bounds (Z=1.96 for 95%)
        z_multiplier = 1.96 if confidence_level >= 0.95 else 1.645
        # Proportional uncertainty scaling
        uncert = max(self.residual_std * 0.5, pred_emv * 0.075)
        lower_bound = max(0.0, pred_emv - (z_multiplier * uncert))
        upper_bound = pred_emv + (z_multiplier * uncert)

        # Cost & ROI
        contract_cost = float(data.get("sponsor_contract_cost_usd", 0.0))
        net_value = pred_emv - contract_cost
        roi_pct = round((net_value / contract_cost * 100.0), 2) if contract_cost > 0 else 0.0
        value_multiplier = round(pred_emv / contract_cost, 2) if contract_cost > 0 else 0.0

        # Channel contribution breakdown estimation
        channel_contrib = self.decompose_channel_value(data)

        return {
            "predicted_media_value_usd": round(pred_emv, 2),
            "lower_bound_usd": round(lower_bound, 2),
            "upper_bound_usd": round(upper_bound, 2),
            "confidence_level": confidence_level,
            "contract_cost_usd": round(contract_cost, 2),
            "net_media_value_usd": round(net_value, 2),
            "roi_percentage": roi_pct,
            "value_multiplier": value_multiplier,
            "channel_contributions": channel_contrib
        }

    def decompose_channel_value(self, data: dict[str, Any]) -> dict[str, float]:
        """
        Decomposes the predicted media value across the 4 key exposure channels:
        Streaming, Television, Social Media, and Online Print.
        """
        # Base weights derived from exposure volumes
        stream_vol = (
            data.get("stream_hours_watched", 0.0) * 0.038
            + (data.get("stream_logo_screen_time_mins", 0.0) / 60.0)
            * (data.get("stream_avg_ccv", 0.0) / 1000.0) * 22.0 * data.get("stream_logo_quality_index", 0.5) * 6.5
            + data.get("stream_chat_mentions", 0.0) * 0.35
        )
        
        tv_vol = (
            (data.get("tv_avg_household_reach", 0.0) / 1000.0)
            * data.get("tv_cpm_rate", 20.0)
            * (data.get("tv_broadcast_hours", 0.0) / 1.5)
            * (1.0 + 0.12 * data.get("tv_brand_integrations", 0))
        )
        
        social_vol = (
            (data.get("social_impressions", 0.0) / 1000.0) * 7.50
            + (data.get("social_impressions", 0.0) * data.get("social_engagement_rate", 0.03)) * 0.18
            + data.get("social_video_views", 0.0) * 0.022
            + data.get("influencer_player_amplification_count", 0) * 950.0
        )
        
        sentiment = max(0.4, 1.0 + data.get("online_editorial_sentiment", 0.0) * 0.35)
        print_vol = (
            (data.get("online_articles_count", 0) * 380.0)
            + ((data.get("online_total_readership", 0) / 1000.0) * 4.80 * (1.0 + data.get("headline_mention_rate", 0.1) * 0.8))
        ) * sentiment

        total_est = stream_vol + tv_vol + social_vol + print_vol or 1.0
        # Normalize to predicted total
        pred_total = self.predict(data)[0]

        return {
            "Streaming": round(float(pred_total * (stream_vol / total_est)), 2),
            "Television": round(float(pred_total * (tv_vol / total_est)), 2),
            "Social Media": round(float(pred_total * (social_vol / total_est)), 2),
            "Online Print & Editorial": round(float(pred_total * (print_vol / total_est)), 2)
        }

    def recommend_optimizations(self, base_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Evaluates marginal sensitivities to recommend executive optimizations
        for increasing sponsorship media ROI.
        """
        baseline_pred = self.predict_single(base_data)["predicted_media_value_usd"]
        levers = []

        # Lever 1: Increase Logo Quality Index (e.g. from 0.70 to 0.90 via better UI placement)
        if base_data.get("stream_logo_quality_index", 0.7) < 0.95:
            mod_data = base_data.copy()
            mod_data["stream_logo_quality_index"] = min(0.98, base_data.get("stream_logo_quality_index", 0.7) * 1.25)
            new_pred = self.predict_single(mod_data)["predicted_media_value_usd"]
            delta = new_pred - baseline_pred
            levers.append({
                "channel": "Streaming",
                "action": "Elevate Broadcast Logo Quality Index (+25%)",
                "detail": "Negotiate cleaner HUD placement and exclusive corner screen rotation to boost Nielsen QI.",
                "projected_gain_usd": round(delta, 2),
                "gain_pct": round((delta / baseline_pred) * 100, 2)
            })

        # Lever 2: Increase Social Co-Branded Posts & Influencer Co-Streams
        mod_data = base_data.copy()
        mod_data["social_post_volume"] = int(base_data.get("social_post_volume", 10) * 1.5)
        mod_data["influencer_player_amplification_count"] = base_data.get("influencer_player_amplification_count", 2) + 4
        mod_data["social_impressions"] = int(base_data.get("social_impressions", 100000) * 1.35)
        new_pred = self.predict_single(mod_data)["predicted_media_value_usd"]
        delta = new_pred - baseline_pred
        levers.append({
            "channel": "Social Media",
            "action": "Activate 4 Pro Player Co-Streams & +50% Social Posts",
            "detail": "Require sponsored player highlight reels and roster retweets on matchday.",
            "projected_gain_usd": round(delta, 2),
            "gain_pct": round((delta / baseline_pred) * 100, 2)
        })

        # Lever 3: Add TV Airtime or TV Brand Integrations
        if base_data.get("tv_broadcast_hours", 0) == 0:
            mod_data = base_data.copy()
            mod_data["tv_broadcast_hours"] = 2.5
            mod_data["tv_avg_household_reach"] = 250000
            mod_data["tv_cpm_rate"] = 28.0
            mod_data["tv_brand_integrations"] = 3
            new_pred = self.predict_single(mod_data)["predicted_media_value_usd"]
            delta = new_pred - baseline_pred
            levers.append({
                "channel": "Television",
                "action": "Secure Linear TV Simulcast Package (2.5 hrs)",
                "detail": "Add regional television coverage and branded segment sponsorship.",
                "projected_gain_usd": round(delta, 2),
                "gain_pct": round((delta / baseline_pred) * 100, 2)
            })
        else:
            mod_data = base_data.copy()
            mod_data["tv_brand_integrations"] = base_data.get("tv_brand_integrations", 1) + 3
            new_pred = self.predict_single(mod_data)["predicted_media_value_usd"]
            delta = new_pred - baseline_pred
            levers.append({
                "channel": "Television",
                "action": "Increase TV On-Air Integrations (+3 Mentions)",
                "detail": "Include sponsor desk verbal mention and lower-third match recap banner.",
                "projected_gain_usd": round(delta, 2),
                "gain_pct": round((delta / baseline_pred) * 100, 2)
            })

        # Sort by projected gain
        levers.sort(key=lambda x: x["projected_gain_usd"], reverse=True)
        return levers

if __name__ == "__main__":
    predictor = MediaValuePredictor()
    sample_campaign = {
        "game_title": "Valorant",
        "tournament_tier": "Tier 1 (Major / Worlds)",
        "stage_of_competition": "Grand Final",
        "team_name": "Titan Esports",
        "sponsor_category": "Gaming Peripherals (Logitech, Razer)",
        "placement_asset_type": "Jersey Chest / Primary Sleeve",
        "stream_broadcast_hours": 6.5,
        "stream_avg_ccv": 185000,
        "stream_peak_ccv": 395000,
        "stream_hours_watched": 1202500,
        "stream_logo_screen_time_mins": 95.0,
        "stream_logo_quality_index": 0.88,
        "stream_chat_mentions": 12400,
        "tv_broadcast_hours": 3.0,
        "tv_avg_household_reach": 420000,
        "tv_cpm_rate": 32.0,
        "tv_brand_integrations": 5,
        "social_post_volume": 24,
        "social_impressions": 3800000,
        "social_engagement_rate": 0.054,
        "social_video_views": 1850000,
        "influencer_player_amplification_count": 8,
        "online_articles_count": 22,
        "online_total_readership": 650000,
        "headline_mention_rate": 0.45,
        "online_editorial_sentiment": 0.72,
        "sponsor_contract_cost_usd": 150000.0
    }

    result = predictor.predict_single(sample_campaign)
    print("Inference Test Result:")
    print(json.dumps(result, indent=2))

