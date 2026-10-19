"""
app.py
======
Esports Sponsorship Media Value Forecasting Engine.
Interactive Executive Dashboard, Real-Time Scenario Forecaster,
and Multi-Channel Attribution Analytics.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    DATA_CSV_PATH,
    FEATURE_IMPORTANCE_JSON_PATH,
    METRICS_JSON_PATH,
    MODEL_PATH,
    REPORTS_DIR,
)
from src.data_loader import load_or_generate_dataset
from src.evaluate import evaluate_model
from src.predict import MediaValuePredictor
from src.train import train_model

# -----------------------------------------------------------------------------
# STREAMLIT CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Esports Sponsorship Media Value Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Dark Esports Aesthetic */
    .main {
        background-color: #0c1017;
        color: #f0f6fc;
    }
    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #1f2937 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        margin-bottom: 12px;
    }
    .metric-label {
        color: #8b949e;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #58a6ff;
        font-size: 1.85rem;
        font-weight: 700;
        margin-top: 4px;
        margin-bottom: 4px;
    }
    .metric-delta {
        font-size: 0.82rem;
        font-weight: 600;
    }
    .metric-delta.positive {
        color: #3fb950;
    }
    .metric-delta.negative {
        color: #f85149;
    }
    .channel-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 6px;
    }
    .badge-stream { background-color: #9146ff; color: #fff; }
    .badge-tv { background-color: #da3633; color: #fff; }
    .badge-social { background-color: #1f6feb; color: #fff; }
    .badge-print { background-color: #d29922; color: #fff; }
    .recommendation-box {
        background-color: #161b22;
        border-left: 4px solid #3fb950;
        padding: 12px 16px;
        border-radius: 4px 8px 8px 4px;
        margin-bottom: 10px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# STATE & MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_predictor() -> MediaValuePredictor:
    """Cached loader for the trained ML predictor."""
    if not os.path.exists(MODEL_PATH):
        # Auto-train if not present
        train_model(str(DATA_CSV_PATH))
        evaluate_model()
    return MediaValuePredictor()

@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    """Cached loader for the dataset."""
    return load_or_generate_dataset(str(DATA_CSV_PATH))

@st.cache_data(show_spinner=False)
def get_metrics() -> dict[str, Any]:
    """Cached loader for evaluation metrics."""
    if os.path.exists(METRICS_JSON_PATH):
        with open(METRICS_JSON_PATH, "r") as f:
            return json.load(f)
    return {}

@st.cache_data(show_spinner=False)
def get_importance() -> dict[str, Any]:
    """Cached loader for feature importance metrics."""
    if os.path.exists(FEATURE_IMPORTANCE_JSON_PATH):
        with open(FEATURE_IMPORTANCE_JSON_PATH, "r") as f:
            return json.load(f)
    return {}

try:
    predictor = get_predictor()
    raw_df = get_data()
    metrics = get_metrics()
    importance = get_importance()
except Exception as e:  # noqa: BLE001
    st.error(f"Error initializing forecasting engine: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & ENGINE STATUS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ Esports EMV Engine")
    st.caption("Production Multi-Channel Sponsorship Media Valuation")
    
    st.markdown("---")
    navigation = st.radio(
        "Navigation",
        [
            "📊 Executive Dashboard",
            "🎯 Real-Time Forecaster",
            "⚖️ Scenario & Batch Studio",
            "🧠 Model Evaluation & XAI",
            "📁 Data & Maintenance"
        ]
    )
    
    st.markdown("---")
    st.subheader("Model Status")
    st.write("**Engine**: XGBoost Regressor")
    st.write("**Version**: 1.0.0 (Local / Docker)")
    if metrics:
        st.write(f"**Test $R^2$**: `{metrics.get('r2_score', 0):.4f}`")
        st.write(f"**Test MAPE**: `{metrics.get('mape_percent', 0):.2f}%`")
        st.write(f"**Test MAE**: `${metrics.get('mae', 0):,.0f}`")
    
    st.markdown("---")
    st.caption("Engineered for Nielsen Sports QI, Blinkfire Analytics, and Relo Metrics valuation frameworks.")

# -----------------------------------------------------------------------------
# TAB 1: EXECUTIVE DASHBOARD
# -----------------------------------------------------------------------------
if navigation == "📊 Executive Dashboard":
    st.title("📊 Executive Esports Sponsorship Portfolio Dashboard")
    st.markdown("Macro-level portfolio valuation across multi-channel broadcasts, tournaments, and brand assets.")

    total_campaigns = len(raw_df)
    total_val = raw_df["generated_media_value_usd"].sum()
    avg_val = raw_df["generated_media_value_usd"].mean()
    total_cost = raw_df["sponsor_contract_cost_usd"].sum()
    portfolio_roi = ((total_val - total_cost) / total_cost) * 100.0

    # High-level KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Portfolio EMV</div>
            <div class="metric-value">${total_val / 1e6:,.2f}M</div>
            <div class="metric-delta positive">Across {total_campaigns:,} campaigns</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Campaign EMV</div>
            <div class="metric-value">${avg_val:,.0f}</div>
            <div class="metric-delta">Median: ${raw_df['generated_media_value_usd'].median():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Portfolio Rights Cost</div>
            <div class="metric-value">${total_cost / 1e6:,.2f}M</div>
            <div class="metric-delta">Contract investment</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Net Sponsorship ROI</div>
            <div class="metric-value">{portfolio_roi:,.1f}%</div>
            <div class="metric-delta positive">{total_val / total_cost:.2f}x Value Multiplier</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Multi-Channel & Tournament Portfolio Analysis")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # Media Value by Game Title & Tournament Tier
        game_tier_df = raw_df.groupby(["game_title", "tournament_tier"])["generated_media_value_usd"].mean().reset_index()
        fig_game = px.bar(
            game_tier_df,
            x="game_title",
            y="generated_media_value_usd",
            color="tournament_tier",
            barmode="group",
            title="Average Media Value (USD) by Game Title & Tournament Tier",
            labels={"generated_media_value_usd": "Average Media Value ($)", "game_title": "Game Title"},
            color_discrete_sequence=["#38bdf8", "#818cf8", "#c084fc"],
            template="plotly_dark"
        )
        fig_game.update_layout(plot_bgcolor="#0c1017", paper_bgcolor="#0c1017")
        st.plotly_chart(fig_game, use_container_width=True)

    with col_chart2:
        # Placement Asset Type Value Distribution
        placement_df = raw_df.groupby("placement_asset_type")["generated_media_value_usd"].agg(["mean", "count"]).reset_index()
        fig_place = px.bar(
            placement_df,
            y="placement_asset_type",
            x="mean",
            orientation="h",
            title="Generated Media Value by Placement Asset Type",
            labels={"mean": "Mean Generated Media Value ($)", "placement_asset_type": "Asset Placement"},
            color="mean",
            color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        fig_place.update_layout(plot_bgcolor="#0c1017", paper_bgcolor="#0c1017")
        st.plotly_chart(fig_place, use_container_width=True)

    col_chart3, col_chart4 = st.columns(2)
    with col_chart3:
        # Top 8 Teams by Media Value Generated
        team_df = raw_df.groupby("team_name")["generated_media_value_usd"].mean().sort_values(ascending=False).head(8).reset_index()
        fig_teams = px.bar(
            team_df,
            x="generated_media_value_usd",
            y="team_name",
            orientation="h",
            title="Top Esports Franchises by Average Sponsorship EMV",
            labels={"generated_media_value_usd": "Average Generated EMV ($)", "team_name": "Team"},
            color="generated_media_value_usd",
            color_continuous_scale="Purples",
            template="plotly_dark"
        )
        fig_teams.update_layout(yaxis={"autorange": "reversed"}, plot_bgcolor="#0c1017", paper_bgcolor="#0c1017")
        st.plotly_chart(fig_teams, use_container_width=True)

    with col_chart4:
        # Sponsor Category ROI & Value Multiplier
        cat_df = raw_df.groupby("sponsor_category")[["generated_media_value_usd", "sponsor_contract_cost_usd"]].sum().reset_index()
        cat_df["ROI_pct"] = ((cat_df["generated_media_value_usd"] - cat_df["sponsor_contract_cost_usd"]) / cat_df["sponsor_contract_cost_usd"]) * 100.0
        cat_df["ROI_pct_magnitude"] = cat_df["ROI_pct"].abs()
        fig_roi = px.scatter(
            cat_df,
            x="sponsor_contract_cost_usd",
            y="generated_media_value_usd",
            size="ROI_pct_magnitude",
            color="sponsor_category",
            hover_name="sponsor_category",
            hover_data={"ROI_pct": ":.1f", "ROI_pct_magnitude": False},
            title="Sponsor Industry: Investment vs. Generated Value (Bubble = |ROI %|)",
            labels={
                "sponsor_contract_cost_usd": "Total Rights Cost ($)",
                "generated_media_value_usd": "Total EMV ($)",
                "ROI_pct": "ROI (%)",
            },
            template="plotly_dark"
        )
        fig_roi.update_layout(plot_bgcolor="#0c1017", paper_bgcolor="#0c1017")
        st.plotly_chart(fig_roi, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: REAL-TIME CAMPAIGN FORECASTER ("WHAT-IF" STUDIO)
# -----------------------------------------------------------------------------
elif navigation == "🎯 Real-Time Forecaster":
    st.title("🎯 Real-Time Sponsorship Media Value Forecaster")
    st.markdown("Simulate multi-channel sponsorship exposure scenarios to forecast Net Media Value (EMV), sponsor ROI, and channel attribution.")

    # Preset Quick Loaders
    st.write("**Quick-Load Strategy Presets:**")
    preset_cols = st.columns(4)
    preset_selected = None
    with preset_cols[0]:
        if st.button("🏆 Tier 1 Worlds Title Rights"):
            preset_selected = "t1_title"
    with preset_cols[1]:
        if st.button("👕 Tier 1 Jersey Chest"):
            preset_selected = "t1_jersey"
    with preset_cols[2]:
        if st.button("📱 Digital & Stream Only"):
            preset_selected = "stream_digital"
    with preset_cols[3]:
        if st.button("⚔️ Tier 3 Weekly Cup"):
            preset_selected = "t3_weekly"

    # Default values dictionary
    default_vals = {
        "game_title": "Valorant",
        "tournament_tier": "Tier 1 (Major / Worlds)",
        "stage_of_competition": "Grand Final",
        "team_name": "Titan Esports",
        "sponsor_category": "Gaming Peripherals (Logitech, Razer)",
        "placement_asset_type": "Jersey Chest / Primary Sleeve",
        # Streaming
        "stream_broadcast_hours": 6.5,
        "stream_avg_ccv": 165000,
        "stream_peak_ccv": 340000,
        "stream_logo_screen_time_mins": 90.0,
        "stream_logo_quality_index": 0.85,
        "stream_chat_mentions": 12500,
        # TV
        "tv_broadcast_hours": 2.5,
        "tv_avg_household_reach": 380000,
        "tv_cpm_rate": 32.0,
        "tv_brand_integrations": 4,
        # Social
        "social_post_volume": 20,
        "social_impressions": 3200000,
        "social_engagement_rate": 0.048,
        "social_video_views": 1400000,
        "influencer_player_amplification_count": 6,
        # Print
        "online_articles_count": 18,
        "online_total_readership": 520000,
        "headline_mention_rate": 0.40,
        "online_editorial_sentiment": 0.65,
        "sponsor_contract_cost_usd": 120000.0
    }

    if preset_selected == "t1_title":
        default_vals.update({
            "game_title": "League of Legends",
            "tournament_tier": "Tier 1 (Major / Worlds)",
            "stage_of_competition": "Grand Final",
            "placement_asset_type": "Tournament Title / Naming Rights",
            "stream_broadcast_hours": 8.0,
            "stream_avg_ccv": 450000,
            "stream_peak_ccv": 1100000,
            "stream_logo_screen_time_mins": 210.0,
            "stream_logo_quality_index": 0.95,
            "stream_chat_mentions": 45000,
            "tv_broadcast_hours": 4.5,
            "tv_avg_household_reach": 1200000,
            "tv_cpm_rate": 40.0,
            "tv_brand_integrations": 8,
            "social_post_volume": 45,
            "social_impressions": 12000000,
            "social_engagement_rate": 0.062,
            "social_video_views": 5800000,
            "influencer_player_amplification_count": 14,
            "online_articles_count": 55,
            "online_total_readership": 2400000,
            "headline_mention_rate": 0.70,
            "online_editorial_sentiment": 0.85,
            "sponsor_contract_cost_usd": 450000.0
        })
    elif preset_selected == "stream_digital":
        default_vals.update({
            "game_title": "CS2",
            "tournament_tier": "Tier 2 (Regional Masters)",
            "stage_of_competition": "Playoff",
            "placement_asset_type": "Broadcast HUD & Overlay",
            "stream_broadcast_hours": 5.0,
            "stream_avg_ccv": 65000,
            "stream_peak_ccv": 140000,
            "stream_logo_screen_time_mins": 75.0,
            "stream_logo_quality_index": 0.82,
            "stream_chat_mentions": 6200,
            "tv_broadcast_hours": 0.0,
            "tv_avg_household_reach": 0,
            "tv_cpm_rate": 0.0,
            "tv_brand_integrations": 0,
            "social_post_volume": 15,
            "social_impressions": 1400000,
            "social_engagement_rate": 0.045,
            "social_video_views": 650000,
            "influencer_player_amplification_count": 3,
            "online_articles_count": 8,
            "online_total_readership": 180000,
            "headline_mention_rate": 0.25,
            "online_editorial_sentiment": 0.45,
            "sponsor_contract_cost_usd": 40000.0
        })
    elif preset_selected == "t3_weekly":
        default_vals.update({
            "game_title": "Rocket League",
            "tournament_tier": "Tier 3 (Challengers)",
            "stage_of_competition": "Group Stage",
            "placement_asset_type": "In-Game Virtual Map Banner",
            "stream_broadcast_hours": 3.0,
            "stream_avg_ccv": 8500,
            "stream_peak_ccv": 19000,
            "stream_logo_screen_time_mins": 35.0,
            "stream_logo_quality_index": 0.62,
            "stream_chat_mentions": 850,
            "tv_broadcast_hours": 0.0,
            "tv_avg_household_reach": 0,
            "tv_cpm_rate": 0.0,
            "tv_brand_integrations": 0,
            "social_post_volume": 6,
            "social_impressions": 85000,
            "social_engagement_rate": 0.035,
            "social_video_views": 32000,
            "influencer_player_amplification_count": 0,
            "online_articles_count": 3,
            "online_total_readership": 25000,
            "headline_mention_rate": 0.10,
            "online_editorial_sentiment": 0.35,
            "sponsor_contract_cost_usd": 15000.0
        })

    # Multi-Channel Input Forms
    with st.expander("🎮 Tournament Context & Sponsor Agreement", expanded=True):
        ctx_c1, ctx_c2, ctx_c3 = st.columns(3)
        with ctx_c1:
            game_title = st.selectbox(
                "Esports Game Title",
                ["CS2", "League of Legends", "Valorant", "Dota 2", "Rocket League"],
                index=["CS2", "League of Legends", "Valorant", "Dota 2", "Rocket League"].index(default_vals["game_title"])
            )
            tournament_tier = st.selectbox(
                "Tournament Tier",
                ["Tier 1 (Major / Worlds)", "Tier 2 (Regional Masters)", "Tier 3 (Challengers)"],
                index=["Tier 1 (Major / Worlds)", "Tier 2 (Regional Masters)", "Tier 3 (Challengers)"].index(default_vals["tournament_tier"])
            )
        with ctx_c2:
            stage_comp = st.selectbox(
                "Competition Stage",
                ["Group Stage", "Playoff", "Semi-Final", "Grand Final"],
                index=["Group Stage", "Playoff", "Semi-Final", "Grand Final"].index(default_vals["stage_of_competition"])
            )
            teams_list = sorted(raw_df["team_name"].unique().tolist())
            team_default_idx = teams_list.index(default_vals["team_name"]) if default_vals["team_name"] in teams_list else 0
            team_name = st.selectbox("Partner Franchise / Team", teams_list, index=team_default_idx)
        with ctx_c3:
            sponsors_list = [
                "Gaming Peripherals (Logitech, Razer)",
                "Energy Drinks (Red Bull, Monster)",
                "Automotive (BMW, Mercedes)",
                "FinTech & Crypto (Coinbase, Revolut)",
                "Apparel & Footwear (Nike, Puma)",
                "Telecom & Hardware (Intel, AT&T)"
            ]
            sponsor_category = st.selectbox(
                "Sponsor Industry Category",
                sponsors_list,
                index=sponsors_list.index(default_vals["sponsor_category"])
            )
            placements_list = [
                "Jersey Chest / Primary Sleeve",
                "Broadcast HUD & Overlay",
                "In-Game Virtual Map Banner",
                "Tournament Title / Naming Rights",
                "Replay Transition Sting & Lower-Third"
            ]
            placement_asset = st.selectbox(
                "Placement Asset Type",
                placements_list,
                index=placements_list.index(default_vals["placement_asset_type"])
            )

    st.markdown("#### Multi-Channel Campaign Exposure Parameters")
    tab_st, tab_tv, tab_soc, tab_prn = st.tabs([
        "📺 Live Streaming (Twitch/YouTube)",
        "📡 Television (Linear Simulcast)",
        "📱 Social Media & Influencers",
        "📰 Online Editorial & Press"
    ])

    with tab_st:
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st_hours = st.number_input("Broadcast Airtime (Hours)", min_value=0.5, max_value=24.0, value=float(default_vals["stream_broadcast_hours"]), step=0.5)
            st_avg_ccv = st.number_input("Average Concurrent Viewers (CCV)", min_value=1000, max_value=2000000, value=int(default_vals["stream_avg_ccv"]), step=5000)
        with sc2:
            st_peak_ccv = st.number_input("Peak Concurrent Viewers", min_value=st_avg_ccv, max_value=3500000, value=max(int(default_vals["stream_peak_ccv"]), int(st_avg_ccv)), step=10000)
            st_screen_mins = st.number_input("Logo Visible Screen Time (Minutes)", min_value=1.0, max_value=720.0, value=float(default_vals["stream_logo_screen_time_mins"]), step=5.0)
        with sc3:
            st_qi = st.slider("Screen Exposure Quality Index (QI)", min_value=0.10, max_value=1.00, value=float(default_vals["stream_logo_quality_index"]), step=0.01, help="Nielsen QI: Screen percentage, placement isolation, and visual clarity.")
            st_chat = st.number_input("Sponsor Chat Brand Mentions", min_value=0, max_value=200000, value=int(default_vals["stream_chat_mentions"]), step=500)
        st_hours_watched = round(st_hours * st_avg_ccv, 1)
        st.info(f"Calculated Streaming Hours Watched: **{st_hours_watched:,.0f} Hours**")

    with tab_tv:
        tc1, tc2 = st.columns(2)
        with tc1:
            tv_hours = st.number_input("TV Broadcast Airtime (Hours)", min_value=0.0, max_value=12.0, value=float(default_vals["tv_broadcast_hours"]), step=0.5)
            tv_reach = st.number_input("Average TV Household Reach", min_value=0, max_value=8000000, value=int(default_vals["tv_avg_household_reach"]), step=25000)
        with tc2:
            tv_cpm = st.number_input("TV Benchmark Spot CPM ($)", min_value=0.0, max_value=75.0, value=float(default_vals["tv_cpm_rate"]), step=2.0)
            tv_integrations = st.number_input("On-Air Verbal / Segment Integrations", min_value=0, max_value=25, value=int(default_vals["tv_brand_integrations"]), step=1)

    with tab_soc:
        soc1, soc2, soc3 = st.columns(3)
        with soc1:
            soc_posts = st.number_input("Co-Branded Social Post Volume", min_value=1, max_value=200, value=int(default_vals["social_post_volume"]), step=2)
            soc_impressions = st.number_input("Total Social Impressions", min_value=10000, max_value=80000000, value=int(default_vals["social_impressions"]), step=50000)
        with soc2:
            soc_eng_rate = st.slider("Engagement Rate (%)", min_value=0.5, max_value=15.0, value=float(default_vals["social_engagement_rate"] * 100), step=0.1) / 100.0
            soc_views = st.number_input("Sponsored Video / Clip Views", min_value=5000, max_value=30000000, value=int(default_vals["social_video_views"]), step=50000)
        with soc3:
            influencers = st.number_input("Pro Player / Streamer Amplifications", min_value=0, max_value=30, value=int(default_vals["influencer_player_amplification_count"]), step=1)

    with tab_prn:
        prn1, prn2 = st.columns(2)
        with prn1:
            articles = st.number_input("Editorial Press Articles Count", min_value=0, max_value=150, value=int(default_vals["online_articles_count"]), step=2)
            readership = st.number_input("Estimated Editorial Readership", min_value=1000, max_value=10000000, value=int(default_vals["online_total_readership"]), step=25000)
        with prn2:
            headline_rate = st.slider("Headline Mention Rate (%)", min_value=0.0, max_value=100.0, value=float(default_vals["headline_mention_rate"] * 100), step=5.0) / 100.0
            sentiment = st.slider("Editorial Sentiment Index (-1.0 to +1.0)", min_value=-1.0, max_value=1.0, value=float(default_vals["online_editorial_sentiment"]), step=0.05)

    st.markdown("#### Contract Financial Investment")
    cost_c1, cost_c2 = st.columns([2, 1])
    with cost_c1:
        contract_cost = st.number_input("Sponsorship Acquisition / Rights Cost ($ USD)", min_value=1000.0, max_value=3000000.0, value=float(default_vals["sponsor_contract_cost_usd"]), step=5000.0)
    with cost_c2:
        forecast_button = st.button("🚀 Calculate Real-Time Forecast", type="primary", use_container_width=True)

    # Execute Forecast
    campaign_input = {
        "game_title": game_title,
        "tournament_tier": tournament_tier,
        "stage_of_competition": stage_comp,
        "team_name": team_name,
        "sponsor_category": sponsor_category,
        "placement_asset_type": placement_asset,
        "stream_broadcast_hours": st_hours,
        "stream_avg_ccv": st_avg_ccv,
        "stream_peak_ccv": st_peak_ccv,
        "stream_hours_watched": st_hours_watched,
        "stream_logo_screen_time_mins": st_screen_mins,
        "stream_logo_quality_index": st_qi,
        "stream_chat_mentions": st_chat,
        "tv_broadcast_hours": tv_hours,
        "tv_avg_household_reach": tv_reach,
        "tv_cpm_rate": tv_cpm,
        "tv_brand_integrations": tv_integrations,
        "social_post_volume": soc_posts,
        "social_impressions": soc_impressions,
        "social_engagement_rate": soc_eng_rate,
        "social_video_views": soc_views,
        "influencer_player_amplification_count": influencers,
        "online_articles_count": articles,
        "online_total_readership": readership,
        "headline_mention_rate": headline_rate,
        "online_editorial_sentiment": sentiment,
        "sponsor_contract_cost_usd": contract_cost
    }

    forecast = predictor.predict_single(campaign_input)

    st.markdown("---")
    st.subheader("📈 Real-Time Valuation & Executive Intelligence")

    r_c1, r_c2, r_c3, r_c4 = st.columns(4)
    pred_val = forecast["predicted_media_value_usd"]
    low_bound = forecast["lower_bound_usd"]
    high_bound = forecast["upper_bound_usd"]
    roi_val = forecast["roi_percentage"]
    mult_val = forecast["value_multiplier"]

    with r_c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Forecasted Media Value (EMV)</div>
            <div class="metric-value">${pred_val:,.0f}</div>
            <div class="metric-delta positive">Point Estimate</div>
        </div>
        """, unsafe_allow_html=True)
    with r_c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">95% Confidence Range</div>
            <div class="metric-value" style="font-size: 1.35rem; margin-top: 10px;">${low_bound:,.0f} - ${high_bound:,.0f}</div>
            <div class="metric-delta">±${(high_bound - low_bound) / 2:,.0f} Uncertainty</div>
        </div>
        """, unsafe_allow_html=True)
    with r_c3:
        roi_class = "positive" if roi_val >= 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Net ROI</div>
            <div class="metric-value">{roi_val:,.1f}%</div>
            <div class="metric-delta {roi_class}">{mult_val:.2f}x Value Multiplier</div>
        </div>
        """, unsafe_allow_html=True)
    with r_c4:
        net_emv = forecast["net_media_value_usd"]
        net_class = "positive" if net_emv >= 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Net Sponsorship Surplus</div>
            <div class="metric-value">${net_emv:,.0f}</div>
            <div class="metric-delta {net_class}">Over Contract Rights Cost</div>
        </div>
        """, unsafe_allow_html=True)

    # Multi-Channel Breakdown Visualizations
    st.markdown("### Multi-Channel Attribution Decomposition")
    ch_col1, ch_col2 = st.columns([1, 1])
    
    decomp = forecast["channel_contributions"]
    decomp_df = pd.DataFrame(list(decomp.items()), columns=["Channel", "Media_Value_USD"])
    decomp_df["Share_Percent"] = (decomp_df["Media_Value_USD"] / pred_val) * 100.0

    with ch_col1:
        fig_donut = px.pie(
            decomp_df,
            names="Channel",
            values="Media_Value_USD",
            hole=0.45,
            title="Channel Share of Forecasted EMV",
            color="Channel",
            color_discrete_map={
                "Streaming": "#9146ff",
                "Television": "#e74c3c",
                "Social Media": "#1f6feb",
                "Online Print & Editorial": "#d29922"
            },
            template="plotly_dark"
        )
        fig_donut.update_layout(plot_bgcolor="#0c1017", paper_bgcolor="#0c1017")
        st.plotly_chart(fig_donut, use_container_width=True)

    with ch_col2:
        fig_bar = px.bar(
            decomp_df,
            x="Channel",
            y="Media_Value_USD",
            text="Media_Value_USD",
            title="Generated Dollar Value by Exposure Channel",
            labels={"Media_Value_USD": "Generated Value ($)"},
            color="Channel",
            color_discrete_map={
                "Streaming": "#9146ff",
                "Television": "#e74c3c",
                "Social Media": "#1f6feb",
                "Online Print & Editorial": "#d29922"
            },
            template="plotly_dark"
        )
        fig_bar.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_bar.update_layout(plot_bgcolor="#0c1017", paper_bgcolor="#0c1017")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Executive AI Recommendations
    st.markdown("### 💡 Strategic Optimization Recommendations")
    st.write("Marginal sensitivity simulator evaluating highest ROI levers to maximize return on rights fees:")

    optimizations = predictor.recommend_optimizations(campaign_input)
    for opt in optimizations:
        st.markdown(f"""
        <div class="recommendation-box">
            <strong>{opt['channel']}: {opt['action']}</strong>
            <span style="float: right; color: #3fb950; font-weight: bold;">+${opt['projected_gain_usd']:,.0f} (+{opt['gain_pct']}%)</span>
            <div style="color: #8b949e; font-size: 0.88rem; margin-top: 4px;">{opt['detail']}</div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 3: SCENARIO & BATCH STUDIO
# -----------------------------------------------------------------------------
elif navigation == "⚖️ Scenario & Batch Studio":
    st.title("⚖️ Sponsorship Scenario & Batch Evaluation Studio")
    st.markdown("Compare package proposals head-to-head or upload campaign CSV files for batch scoring.")

    mode = st.radio("Studio Mode", ["Side-by-Side Package Comparison", "Batch CSV Scoring"], horizontal=True)

    if mode == "Side-by-Side Package Comparison":
        st.subheader("Head-to-Head Proposal Comparison")
        st.write("Compare two prospective sponsorship rights packages to decide investment allocation.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 🔵 Proposal A: Broadcast & TV Heavy")
            pkg_a_cost = st.number_input("Package A Cost ($)", value=180000.0, step=10000.0, key="pa_cost")
            pkg_a_game = st.selectbox("Game A", ["Valorant", "League of Legends", "CS2"], index=0, key="pa_game")
            pkg_a_tier = st.selectbox("Tier A", ["Tier 1 (Major / Worlds)", "Tier 2 (Regional Masters)"], index=0, key="pa_tier")
            pkg_a_st_ccv = st.number_input("Avg CCV A", value=195000, key="pa_ccv")
            pkg_a_tv_hours = st.number_input("TV Hours A", value=4.0, key="pa_tv_hrs")
            pkg_a_tv_reach = st.number_input("TV Reach A", value=550000, key="pa_tv_reach")
            pkg_a_soc_imp = st.number_input("Social Impressions A", value=2500000, key="pa_soc_imp")

        with col_b:
            st.markdown("#### 🟣 Proposal B: Creator & Social Blitz")
            pkg_b_cost = st.number_input("Package B Cost ($)", value=130000.0, step=10000.0, key="pb_cost")
            pkg_b_game = st.selectbox("Game B", ["Valorant", "League of Legends", "CS2"], index=1, key="pb_game")
            pkg_b_tier = st.selectbox("Tier B", ["Tier 1 (Major / Worlds)", "Tier 2 (Regional Masters)"], index=0, key="pb_tier")
            pkg_b_st_ccv = st.number_input("Avg CCV B", value=120000, key="pb_ccv")
            pkg_b_tv_hours = st.number_input("TV Hours B", value=0.0, key="pb_tv_hrs")
            pkg_b_tv_reach = st.number_input("TV Reach B", value=0, key="pb_tv_reach")
            pkg_b_soc_imp = st.number_input("Social Impressions B", value=9500000, key="pb_soc_imp")

        # Prepare records
        rec_a = {
            "game_title": pkg_a_game, "tournament_tier": pkg_a_tier, "stage_of_competition": "Grand Final",
            "team_name": "Titan Esports", "sponsor_category": "Gaming Peripherals (Logitech, Razer)",
            "placement_asset_type": "Jersey Chest / Primary Sleeve",
            "stream_broadcast_hours": 6.0, "stream_avg_ccv": pkg_a_st_ccv, "stream_peak_ccv": pkg_a_st_ccv * 2,
            "stream_hours_watched": 6.0 * pkg_a_st_ccv, "stream_logo_screen_time_mins": 90.0,
            "stream_logo_quality_index": 0.88, "stream_chat_mentions": 12000,
            "tv_broadcast_hours": pkg_a_tv_hours, "tv_avg_household_reach": pkg_a_tv_reach, "tv_cpm_rate": 32.0, "tv_brand_integrations": 5,
            "social_post_volume": 18, "social_impressions": pkg_a_soc_imp, "social_engagement_rate": 0.045, "social_video_views": pkg_a_soc_imp * 0.4,
            "influencer_player_amplification_count": 4, "online_articles_count": 18, "online_total_readership": 450000,
            "headline_mention_rate": 0.35, "online_editorial_sentiment": 0.60,
            "sponsor_contract_cost_usd": pkg_a_cost
        }

        rec_b = {
            "game_title": pkg_b_game, "tournament_tier": pkg_b_tier, "stage_of_competition": "Playoff",
            "team_name": "Team Nova", "sponsor_category": "Energy Drinks (Red Bull, Monster)",
            "placement_asset_type": "Broadcast HUD & Overlay",
            "stream_broadcast_hours": 8.0, "stream_avg_ccv": pkg_b_st_ccv, "stream_peak_ccv": pkg_b_st_ccv * 1.8,
            "stream_hours_watched": 8.0 * pkg_b_st_ccv, "stream_logo_screen_time_mins": 120.0,
            "stream_logo_quality_index": 0.82, "stream_chat_mentions": 18000,
            "tv_broadcast_hours": pkg_b_tv_hours, "tv_avg_household_reach": pkg_b_tv_reach, "tv_cpm_rate": 0.0, "tv_brand_integrations": 0,
            "social_post_volume": 42, "social_impressions": pkg_b_soc_imp, "social_engagement_rate": 0.065, "social_video_views": pkg_b_soc_imp * 0.55,
            "influencer_player_amplification_count": 12, "online_articles_count": 22, "online_total_readership": 620000,
            "headline_mention_rate": 0.40, "online_editorial_sentiment": 0.70,
            "sponsor_contract_cost_usd": pkg_b_cost
        }

        res_a = predictor.predict_single(rec_a)
        res_b = predictor.predict_single(rec_b)

        st.markdown("---")
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Proposal A Net Media Value</div>
                <div class="metric-value">${res_a['predicted_media_value_usd']:,.0f}</div>
                <div class="metric-delta positive">ROI: {res_a['roi_percentage']:.1f}% ({res_a['value_multiplier']:.2f}x)</div>
            </div>
            """, unsafe_allow_html=True)
        with comp_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Proposal B Net Media Value</div>
                <div class="metric-value">${res_b['predicted_media_value_usd']:,.0f}</div>
                <div class="metric-delta positive">ROI: {res_b['roi_percentage']:.1f}% ({res_b['value_multiplier']:.2f}x)</div>
            </div>
            """, unsafe_allow_html=True)

        # Comparison Bar Chart
        comp_df = pd.DataFrame([
            {"Proposal": "Proposal A (TV & Broadcast)", "Metric": "Contract Rights Cost ($)", "Value": pkg_a_cost},
            {"Proposal": "Proposal A (TV & Broadcast)", "Metric": "Forecasted EMV ($)", "Value": res_a['predicted_media_value_usd']},
            {"Proposal": "Proposal B (Creator & Social)", "Metric": "Contract Rights Cost ($)", "Value": pkg_b_cost},
            {"Proposal": "Proposal B (Creator & Social)", "Metric": "Forecasted EMV ($)", "Value": res_b['predicted_media_value_usd']}
        ])
        fig_comp = px.bar(
            comp_df,
            x="Proposal",
            y="Value",
            color="Metric",
            barmode="group",
            title="Financial Investment vs. Forecasted Return",
            labels={"Value": "Amount (USD)"},
            template="plotly_dark",
            color_discrete_sequence=["#ef4444", "#10b981"]
        )
        fig_comp.update_layout(plot_bgcolor="#0c1017", paper_bgcolor="#0c1017")
        st.plotly_chart(fig_comp, use_container_width=True)

    else:
        st.subheader("Batch CSV Campaign Scoring")
        st.write("Upload a CSV file containing campaign exposure parameters to generate predictions, uncertainty bounds, and ROI metrics in bulk.")

        uploaded_file = st.file_uploader("Upload Campaign CSV", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} records for scoring.")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("⚡ Score Batch Campaigns"):
                with st.spinner("Scoring campaigns via XGBoost pipeline..."):
                    preds = predictor.predict(batch_df)
                    batch_df["predicted_media_value_usd"] = np.round(preds, 2)
                    if "sponsor_contract_cost_usd" in batch_df.columns:
                        batch_df["net_media_value_usd"] = np.round(batch_df["predicted_media_value_usd"] - batch_df["sponsor_contract_cost_usd"], 2)
                        batch_df["predicted_roi_pct"] = np.round((batch_df["net_media_value_usd"] / batch_df["sponsor_contract_cost_usd"]) * 100.0, 2)
                    
                    st.success("Batch scoring complete!")
                    st.dataframe(batch_df.head(15), use_container_width=True)

                    # Export button
                    csv_export = batch_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Scored Campaigns CSV",
                        data=csv_export,
                        file_name="esports_campaigns_scored.csv",
                        mime="text/csv"
                    )

# -----------------------------------------------------------------------------
# TAB 4: MODEL EVALUATION & EXPLAINABILITY (XAI)
# -----------------------------------------------------------------------------
elif navigation == "🧠 Model Evaluation & XAI":
    st.title("🧠 XGBoost Model Evaluation & Explainability (XAI)")
    st.markdown("Transparent production ML validation on hold-out test set (20% sample, 1,000 campaigns).")

    # Metrics Display
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">R² Variance Explained</div>
            <div class="metric-value">{metrics.get('r2_score', 0):.4f}</div>
            <div class="metric-delta positive">Hold-Out Test Set</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Mean Abs % Error (MAPE)</div>
            <div class="metric-value">{metrics.get('mape_percent', 0):.2f}%</div>
            <div class="metric-delta positive">High Valuation Precision</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Root Mean Squared Error</div>
            <div class="metric-value">${metrics.get('rmse', 0):,.0f}</div>
            <div class="metric-delta">Test RMSE</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Median Absolute Error</div>
            <div class="metric-value">${metrics.get('median_absolute_error', 0):,.0f}</div>
            <div class="metric-delta positive">Robust to Outliers</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Feature Importance (XGBoost Normalized Gain)")
    feat_col1, feat_col2 = st.columns([3, 2])

    with feat_col1:
        if importance and "sorted_gain_importance" in importance:
            top_feats = pd.DataFrame(importance["sorted_gain_importance"][:15])
            fig_imp = px.bar(
                top_feats,
                x="importance",
                y="feature",
                orientation="h",
                title="Top 15 Predictive Features by Gain",
                labels={"importance": "Normalized Gain Score", "feature": "Feature"},
                color="importance",
                color_continuous_scale="Tealgrn",
                template="plotly_dark"
            )
            fig_imp.update_layout(yaxis={"autorange": "reversed"}, plot_bgcolor="#0c1017", paper_bgcolor="#0c1017")
            st.plotly_chart(fig_imp, use_container_width=True)

    with feat_col2:
        if importance and "channel_group_attribution" in importance:
            ch_attr = pd.DataFrame(list(importance["channel_group_attribution"].items()), columns=["Channel_Group", "Importance"])
            fig_pie = px.pie(
                ch_attr,
                names="Channel_Group",
                values="Importance",
                title="Channel Group Contribution to Model Power",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_layout(plot_bgcolor="#0c1017", paper_bgcolor="#0c1017")
            st.plotly_chart(fig_pie, use_container_width=True)

    # Diagnostic Figures from Disk
    st.markdown("### Diagnostic Plots")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        p1 = REPORTS_DIR / "actual_vs_predicted.png"
        if p1.exists():
            st.image(str(p1), caption="Actual vs. Predicted Media Value (Log-Log Scale)", use_container_width=True)
    with p_col2:
        p2 = REPORTS_DIR / "residuals_distribution.png"
        if p2.exists():
            st.image(str(p2), caption="Residual Error Distribution", use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: DATA & MAINTENANCE
# -----------------------------------------------------------------------------
elif navigation == "📁 Data & Maintenance":
    st.title("📁 Dataset Explorer & Engine Maintenance")
    st.markdown("Inspect training data, summary distributions, and trigger retraining.")

    # Data Filter
    f_c1, f_c2 = st.columns(2)
    with f_c1:
        filter_game = st.multiselect("Filter Game Title", raw_df["game_title"].unique().tolist(), default=raw_df["game_title"].unique().tolist())
    with f_c2:
        filter_tier = st.multiselect("Filter Tournament Tier", raw_df["tournament_tier"].unique().tolist(), default=raw_df["tournament_tier"].unique().tolist())

    filtered_df = raw_df[raw_df["game_title"].isin(filter_game) & raw_df["tournament_tier"].isin(filter_tier)]
    st.write(f"Showing **{len(filtered_df):,}** of **{len(raw_df):,}** records:")
    st.dataframe(filtered_df.head(25), use_container_width=True)

    st.markdown("---")
    st.subheader("🔁 Model Retraining Pipeline")
    st.write("Retrain the XGBoost regressor on the latest dataset with early stopping and refreshed evaluation.")

    if st.button("🚀 Trigger End-to-End Retraining"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Ingesting data & validating schema...")
        progress_bar.progress(20)
        time.sleep(0.5)

        status_text.text("Fitting feature engineering pipeline...")
        progress_bar.progress(40)
        time.sleep(0.5)

        status_text.text("Training XGBoost Regressor with early stopping...")
        train_model(str(DATA_CSV_PATH))
        progress_bar.progress(80)
        time.sleep(0.5)

        status_text.text("Running hold-out evaluation & generating diagnostic plots...")
        evaluate_model()
        progress_bar.progress(100)
        status_text.text("Model Retraining Completed Successfully!")
        st.success("New model artifacts and evaluation metrics saved to models/ and reports/!")
        st.cache_resource.clear()
        st.cache_data.clear()
