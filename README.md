# ⚡ Esports Sponsorship Media Value Forecasting Engine

> **A production-grade, 100% local-compatible Machine Learning engine built to forecast multi-channel Equivalent Media Value (EMV), sponsor ROI, and channel attribution in esports.**

---

## 📌 Executive Overview

In modern esports marketing, measuring commercial sponsorship value requires multi-channel exposure tracking across **Live Streaming (Twitch, YouTube Gaming, Kick)**, **Linear Television**, **Social Media (X, TikTok, Instagram, YouTube Shorts)**, and **Online Editorial/Print**. 

This engine implements the industry-standard **Nielsen Sports Quality Index (QI)** and **Equivalent Media Value (EMV)** methodology, powered by an **XGBoost Regression pipeline** with early stopping, domain-specific feature engineering, confidence intervals, and an interactive **Streamlit executive dashboard**.



---

## 📐 System Architecture & Project Structure

```
Sport Analytics/
├── data/
│   ├── generator.py                  # Multi-channel synthetic data generator & player data bridge
│   └── esports_sponsorship_data.csv  # Synthesized dataset (5,000 campaigns, 30 features)
├── src/
│   ├── __init__.py                   # Package initializer
│   ├── config.py                     # Centralized paths, feature schemas, hyperparameters
│   ├── data_loader.py                # Schema validator, ingestion, train/test partitioner
│   ├── features.py                   # Domain feature engineering & Scikit-Learn transformers
│   ├── train.py                      # Model training with early stopping & artifact export
│   ├── evaluate.py                   # Evaluation pipeline (R², RMSE, MAE, MAPE, MedAE, XAI)
│   └── predict.py                    # Production inference engine, confidence bounds, attribution
├── models/
│   ├── xgboost_media_value_model.joblib # Trained XGBoost regressor
│   ├── feature_pipeline.joblib          # Fitted scikit-learn preprocessing pipeline
│   ├── feature_names.json               # Ordered transformed feature names
│   └── training_metadata.json           # Run timestamps, sample counts, hyperparams
├── reports/
│   ├── evaluation_metrics.json          # Quantitative benchmark metrics
│   ├── feature_importance.json          # Gain and channel attribution scores
│   ├── actual_vs_predicted.png          # Regression scatter plot (Log-Log)
│   ├── residuals_distribution.png       # Residual distribution & KDE curve
│   ├── feature_importance.png           # Top 12 predictive features bar chart
│   └── channel_attribution.png          # Channel share pie chart
├── app.py                            # Flagship Streamlit Executive Dashboard
├── Dockerfile                        # Production Docker containerization (Python 3.12-slim)
├── requirements.txt                  # Pinned dependencies
├── esports_player_performance_tournament_analytics.csv # Original player tournament dataset
└── README.md                         # Documentation and operational manual
```


## 📊 Model Evaluation & Benchmarks

Evaluated on an independent hold-out test set ($N = 1,000$ campaigns, 20% unseen data):

| Metric | Hold-Out Test Result | Interpretation |
| :--- | :--- | :--- |
| **$R^2$ Score** | **`0.8328`** | **83.3%** of total media value variance explained |
| **MAPE** | **`11.24%`** | High commercial precision across orders of magnitude |
| **MAE** | **`$16,738.11`** | Average absolute deviation |
| **Median Absolute Error** | **`$3,462.27`** | Robust median error indicating high stability |
| **RMSE** | **`$104,260.66`** | Root mean square error |

### Predictive Importance by Channel Group
- **Context & Tournament Dynamics**: 83.2%
- **Linear Television Simulcast**: 10.3%
- **Social Media & Influencer Amplification**: 3.4%
- **Live Streaming Exposure (QI-Weighted)**: 2.2%
- **Online Print & Digital Editorial**: 0.5%
- **Contract Cost / Rights Fee**: 0.5%

---

## 🚀 Quickstart: Local Development

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git

### 2. Environment Setup
```bash
# Clone or navigate to the repository
cd "c:\Users\DELL\Desktop\Sport Analytics"

# Create and activate virtual environment (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 3. Generate Data, Train & Evaluate
```bash
# Step 1: Synthesize multi-channel dataset (5,000 records)
python data/generator.py --samples 5000

# Step 2: Train the XGBoost Regression model
python src/train.py

# Step 3: Run comprehensive evaluation and generate diagnostic charts
python src/evaluate.py
```

### 4. Launch the Streamlit Executive Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🐳 Docker Deployment (100% Local / No AWS)

The engine includes a self-contained, cloud-agnostic Dockerfile based on `python:3.12-slim`.

### Run with Docker Compose
```bash
docker compose up --build -d
```

Open the application at `http://localhost:8501`. To stop it:
```bash
docker compose down
```

### Verify Health & Logs
```bash
# Check health status
docker inspect --format='{{json .State.Health.Status}}' esports-forecast-engine

# View container logs
docker compose logs -f streamlit
```

---

## 📄 License & Attribution
Designed and built for production esports sponsorship valuation in accordance with Nielsen Sports QI and Blinkfire Analytics valuation methodologies.