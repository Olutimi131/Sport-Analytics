"""
data/generator.py
=================
Esports Sponsorship Media Value Dataset Generator.
Synthesizes high-fidelity, multi-channel esports sponsorship campaign records
grounded in industry-standard Nielsen Sports Quality Index (QI) and Equivalent
Media Value (EMV) methodologies.
"""

import argparse
import os

import numpy as np
import pandas as pd


def generate_esports_sponsorship_data(
    n_samples: int = 5000,
    seed: int = 42,
    existing_player_csv: str = "esports_player_performance_tournament_analytics.csv"
) -> pd.DataFrame:
    """
    Generates realistic esports sponsorship multi-channel exposure and media value records.
    Optionally bridges team and match performance from existing tournament data.
    """
    np.random.seed(seed)
    
    # 1. Domain Entities & Distributions
    games = ["CS2", "League of Legends", "Valorant", "Dota 2", "Rocket League"]
    game_probs = [0.30, 0.30, 0.22, 0.12, 0.06]
    
    tournament_tiers = ["Tier 1 (Major / Worlds)", "Tier 2 (Regional Masters)", "Tier 3 (Challengers)"]
    tier_probs = [0.25, 0.50, 0.25]
    
    stages = ["Group Stage", "Playoff", "Semi-Final", "Grand Final"]
    stage_probs = [0.45, 0.30, 0.15, 0.10]
    
    sponsor_categories = [
        "Gaming Peripherals (Logitech, Razer)",
        "Energy Drinks (Red Bull, Monster)",
        "Automotive (BMW, Mercedes)",
        "FinTech & Crypto (Coinbase, Revolut)",
        "Apparel & Footwear (Nike, Puma)",
        "Telecom & Hardware (Intel, AT&T)"
    ]
    
    placements = [
        "Jersey Chest / Primary Sleeve",
        "Broadcast HUD & Overlay",
        "In-Game Virtual Map Banner",
        "Tournament Title / Naming Rights",
        "Replay Transition Sting & Lower-Third"
    ]
    
    # Check if existing player performance data exists to extract realistic team names
    known_teams = ["Titan Esports", "Team Alpha", "Team Nova", "Phantom Squad", "Red Vortex", 
                   "Cloud9", "FaZe Clan", "T1", "Fnatic", "G2 Esports"]
    if os.path.exists(existing_player_csv):
        try:
            player_df = pd.read_csv(existing_player_csv)
            if "team_name" in player_df.columns:
                extracted_teams = player_df["team_name"].dropna().unique().tolist()
                if len(extracted_teams) > 0:
                    known_teams = list(set(known_teams + extracted_teams))
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
            print(f"Notice: Could not extract teams from {existing_player_csv}: {e}")

    # Generate synthetic attributes
    sample_games = np.random.choice(games, size=n_samples, p=game_probs)
    sample_tiers = np.random.choice(tournament_tiers, size=n_samples, p=tier_probs)
    sample_stages = np.random.choice(stages, size=n_samples, p=stage_probs)
    sample_teams = np.random.choice(known_teams, size=n_samples)
    sample_sponsors = np.random.choice(sponsor_categories, size=n_samples)
    sample_placements = np.random.choice(placements, size=n_samples)
    
    # Tier multipliers for exposure scale
    tier_scale_map = {
        "Tier 1 (Major / Worlds)": 3.2,
        "Tier 2 (Regional Masters)": 1.4,
        "Tier 3 (Challengers)": 0.6
    }
    
    stage_scale_map = {
        "Group Stage": 0.8,
        "Playoff": 1.15,
        "Semi-Final": 1.6,
        "Grand Final": 2.4
    }
    
    placement_weight_map = {
        "Tournament Title / Naming Rights": 1.85,
        "Jersey Chest / Primary Sleeve": 1.45,
        "Broadcast HUD & Overlay": 1.25,
        "In-Game Virtual Map Banner": 1.0,
        "Replay Transition Sting & Lower-Third": 0.75
    }
    
    records = []
    
    for i in range(n_samples):
        campaign_id = f"CMP-{100000 + i}"
        game = sample_games[i]
        tier = sample_tiers[i]
        stage = sample_stages[i]
        team = sample_teams[i]
        category = sample_sponsors[i]
        placement = sample_placements[i]
        
        tier_scale = tier_scale_map[tier]
        stage_scale = stage_scale_map[stage]
        scale = tier_scale * stage_scale
        
        # 1. Live Streaming Channel Metrics (Twitch / YouTube / Kick)
        # Broadcast hours: 1.5 to 14 hours per event day/match series
        stream_broadcast_hours = round(float(np.random.gamma(shape=3.5, scale=1.2) * np.sqrt(scale)), 2)
        stream_broadcast_hours = max(1.0, min(stream_broadcast_hours, 24.0))
        
        # Concurrent Viewers (CCV)
        base_ccv = np.random.lognormal(mean=9.5, sigma=0.65) * scale
        stream_avg_ccv = int(np.clip(base_ccv, 3000, 1500000))
        stream_peak_ccv = int(stream_avg_ccv * np.random.uniform(1.35, 2.8))
        stream_hours_watched = round((stream_broadcast_hours * stream_avg_ccv), 1)
        
        # Brand visual exposure on stream
        stream_logo_screen_time_mins = round(float(np.random.uniform(0.08, 0.35) * stream_broadcast_hours * 60), 2)
        # Quality index (0.35 to 0.98 based on placement)
        placement_base_qi = {
            "Tournament Title / Naming Rights": 0.92,
            "Broadcast HUD & Overlay": 0.84,
            "Jersey Chest / Primary Sleeve": 0.76,
            "In-Game Virtual Map Banner": 0.65,
            "Replay Transition Sting & Lower-Third": 0.58
        }[placement]
        stream_logo_quality_index = round(float(np.clip(np.random.normal(placement_base_qi, 0.06), 0.30, 0.99)), 3)
        stream_chat_mentions = int(np.random.negative_binomial(n=5, p=0.001) * (scale / 2.0))
        stream_chat_mentions = max(50, min(stream_chat_mentions, 180000))
        
        # 2. Linear Television Exposure
        # TV airtime hours: lower for Tier 3, higher for Tier 1 majors / Asian/EU TV networks
        has_tv = np.random.rand() < (0.85 if "Tier 1" in tier else 0.45 if "Tier 2" in tier else 0.15)
        if has_tv:
            tv_broadcast_hours = round(float(np.random.uniform(0.5, 6.0) * (scale / 1.5)), 2)
            tv_avg_household_reach = int(np.random.lognormal(mean=11.2, sigma=0.7) * (scale / 1.5))
            tv_avg_household_reach = max(20000, min(tv_avg_household_reach, 4500000))
            tv_cpm_rate = round(float(np.random.uniform(18.0, 48.0)), 2)
            tv_brand_integrations = int(np.random.poisson(lam=4 * scale))
        else:
            tv_broadcast_hours = 0.0
            tv_avg_household_reach = 0
            tv_cpm_rate = 0.0
            tv_brand_integrations = 0
            
        # 3. Social Media Channel Exposure (X, TikTok, Instagram, YouTube VODs)
        social_post_volume = int(np.random.gamma(shape=4, scale=3) * np.sqrt(scale))
        social_post_volume = max(3, min(social_post_volume, 120))
        
        base_social_imp = np.random.lognormal(mean=12.2, sigma=0.8) * scale
        social_impressions = int(np.clip(base_social_imp, 25000, 45000000))
        social_engagement_rate = round(float(np.clip(np.random.beta(a=3, b=45), 0.012, 0.145)), 4)
        social_video_views = int(social_impressions * np.random.uniform(0.25, 0.70))
        influencer_player_amplification_count = int(np.random.poisson(lam=3.5 * tier_scale))
        
        # 4. Online Print & Digital Editorial
        online_articles_count = int(np.random.gamma(shape=3.5, scale=2.5) * np.sqrt(scale))
        online_articles_count = max(1, min(online_articles_count, 85))
        online_total_readership = int(online_articles_count * np.random.lognormal(mean=9.8, sigma=0.6))
        online_total_readership = max(5000, min(online_total_readership, 6000000))
        headline_mention_rate = round(float(np.clip(np.random.beta(a=2, b=5), 0.05, 0.75)), 3)
        online_editorial_sentiment = round(float(np.clip(np.random.normal(loc=0.55, scale=0.25), -0.5, 0.98)), 3)
        
        # 5. Sponsorship Contract Investment / Fee
        base_cost = (tier_scale * 65000) * placement_weight_map[placement] * np.random.uniform(0.7, 1.35)
        sponsor_contract_cost_usd = round(float(np.clip(base_cost, 12000, 1250000)), 2)
        
        # 6. Target Calculation: Generated Net Media Value (EMV / Quality-Adjusted EMV in USD)
        # Streaming EMV:
        # - Hours watched value (~$0.025 - $0.05 per hour watched)
        # - Logo Screen Time value adjusted by QI and CCV CPM
        # - Chat mention brand buzz
        val_stream_hw = stream_hours_watched * 0.038
        stream_cpm = 22.0  # streaming video ad rate benchmark
        val_stream_logo = (stream_logo_screen_time_mins / 60.0) * (stream_avg_ccv / 1000.0) * stream_cpm * stream_logo_quality_index * 6.5
        val_stream_chat = stream_chat_mentions * 0.35
        val_stream = val_stream_hw + val_stream_logo + val_stream_chat
        
        # TV EMV:
        if has_tv and tv_broadcast_hours > 0:
            val_tv = (tv_avg_household_reach / 1000.0) * tv_cpm_rate * (tv_broadcast_hours / 1.5) * (1.0 + 0.12 * tv_brand_integrations)
        else:
            val_tv = 0.0
            
        # Social Media EMV:
        # CPM on impressions ($6.50 CPM), plus CPE ($0.15 per engagement), video view rate ($0.02/view), creator amplification ($850/influencer)
        val_social_imp = (social_impressions / 1000.0) * 7.50
        val_social_eng = (social_impressions * social_engagement_rate) * 0.18
        val_social_vid = social_video_views * 0.022
        val_social_inf = influencer_player_amplification_count * 950.0
        val_social = val_social_imp + val_social_eng + val_social_vid + val_social_inf
        
        # Online Print & Editorial EMV:
        # Article mention baseline ($350/article), readership CPM ($4.20), headline boost, sentiment factor
        sentiment_factor = max(0.4, 1.0 + (online_editorial_sentiment * 0.35))
        val_print = ((online_articles_count * 380.0) + ((online_total_readership / 1000.0) * 4.80 * (1.0 + headline_mention_rate * 0.8))) * sentiment_factor
        
        # Synergy, placement and game popularity multiplier
        game_multiplier = {"CS2": 1.12, "League of Legends": 1.18, "Valorant": 1.10, "Dota 2": 1.05, "Rocket League": 0.95}[game]
        placement_multiplier = placement_weight_map[placement]
        
        # Category affinity synergy (e.g. Endemic gaming hardware or energy drinks get higher brand conversion)
        category_synergy = 1.15 if "Gaming" in category or "Energy" in category else 1.0
        
        total_channel_val = (val_stream + val_tv + val_social + val_print)
        synergy_factor = (game_multiplier * placement_multiplier * category_synergy) ** 0.45
        
        # Realistic market stochastic noise (log-normal, standard deviation ~10%)
        market_noise = np.random.lognormal(mean=0.0, sigma=0.08)
        
        generated_media_value = round(float(total_channel_val * synergy_factor * market_noise), 2)
        generated_media_value = max(4500.0, generated_media_value)
        
        # ROI %
        roi_percent = round(((generated_media_value - sponsor_contract_cost_usd) / sponsor_contract_cost_usd) * 100.0, 2)
        
        records.append({
            "campaign_id": campaign_id,
            "game_title": game,
            "tournament_tier": tier,
            "stage_of_competition": stage,
            "team_name": team,
            "sponsor_category": category,
            "placement_asset_type": placement,
            # Streaming
            "stream_broadcast_hours": stream_broadcast_hours,
            "stream_avg_ccv": stream_avg_ccv,
            "stream_peak_ccv": stream_peak_ccv,
            "stream_hours_watched": stream_hours_watched,
            "stream_logo_screen_time_mins": stream_logo_screen_time_mins,
            "stream_logo_quality_index": stream_logo_quality_index,
            "stream_chat_mentions": stream_chat_mentions,
            # TV
            "tv_broadcast_hours": tv_broadcast_hours,
            "tv_avg_household_reach": tv_avg_household_reach,
            "tv_cpm_rate": tv_cpm_rate,
            "tv_brand_integrations": tv_brand_integrations,
            # Social Media
            "social_post_volume": social_post_volume,
            "social_impressions": social_impressions,
            "social_engagement_rate": social_engagement_rate,
            "social_video_views": social_video_views,
            "influencer_player_amplification_count": influencer_player_amplification_count,
            # Online Print / Editorial
            "online_articles_count": online_articles_count,
            "online_total_readership": online_total_readership,
            "headline_mention_rate": headline_mention_rate,
            "online_editorial_sentiment": online_editorial_sentiment,
            # Cost & Target
            "sponsor_contract_cost_usd": sponsor_contract_cost_usd,
            "generated_media_value_usd": generated_media_value,
            "roi_percentage": roi_percent
        })
        
    df = pd.DataFrame(records)
    return df

def save_dataset(output_path: str = "data/esports_sponsorship_data.csv", n_samples: int = 5000):
    """Generates and saves the esports sponsorship dataset to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Generating {n_samples} esports sponsorship records...")
    df = generate_esports_sponsorship_data(n_samples=n_samples)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created and saved to: {output_path}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"Dataset Shape: {df.shape}")
    print("Media Value Summary (USD):")
    print(df["generated_media_value_usd"].describe())
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Esports Sponsorship Media Value Dataset")
    parser.add_argument("--samples", type=int, default=5000, help="Number of records to generate")
    parser.add_argument("--output", type=str, default="data/esports_sponsorship_data.csv", help="Output CSV path")
    args = parser.parse_args()
    save_dataset(args.output, args.samples)

